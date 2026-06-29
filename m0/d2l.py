"""Lane D2L (M1) — fold d'un buffer de conversation en LoRA via mlx_lm.

Approche PRAGMATIQUE (pas le hypernetwork CUDA de Sakana, indisponible sur Mac) :
entrainement LoRA par descente de gradient avec mlx_lm, sur le modele MLX local.

Lecon empirique du dé-risque : on entraine au format CHAT (messages), car l'inference
passe par apply_chat_template ; entrainer en texte brut puis inferer en chat donne une
restitution brouillonne (mismatch de distribution).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

_LOSS_RE = re.compile(r"(Train|Val) loss ([0-9.]+)")


_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯؀-ۿЀ-ӿ]")


def clean_and_balance(pairs, max_per_answer: int = 3):
    """Nettoie les paires Q/R avant entrainement (corrige les pathologies vues via
    /info) : retire les reformulations en alphabet non-latin (bruit multilingue),
    deduplique, et **borne le nombre d'exemples par reponse** (anti-collapse : sinon
    le modele apprend a tout repondre par la reponse la plus frequente)."""
    out: list[tuple[str, str]] = []
    seen: set = set()
    per_answer: dict[str, int] = {}
    for q, a in pairs:
        q, a = (q or "").strip(), (a or "").strip()
        if not q or not a:
            continue
        if _CJK_RE.search(q + a):  # reformulation hors-langue -> bruit
            continue
        key = (q.lower(), a.lower())
        if key in seen:
            continue
        ak = a.lower()
        if per_answer.get(ak, 0) >= max_per_answer:
            continue
        seen.add(key)
        per_answer[ak] = per_answer.get(ak, 0) + 1
        out.append((q, a))
    return out


def looks_degenerate(text: str) -> bool:
    """Detecte une sortie degeneree (boucle de repetition type 'LLMLMLM...', mot
    repete, vide). Sert au controle d'integrite de la gate."""
    t = (text or "").strip()
    if not t:
        return True
    compact = re.sub(r"\s+", "", t)
    if len(compact) >= 20 and len(set(compact)) / len(compact) < 0.15:
        return True
    words = t.split()
    if len(words) >= 8 and len(set(w.lower() for w in words)) / len(words) < 0.3:
        return True
    return False


def answer_recalled(output: str, expected: str) -> bool:
    """Vrai si la sortie contient la reponse attendue. STRICT (discrimine les faits
    d'une meme entite) : TOUS les mots-cles significatifs (len>2) doivent etre presents
    en MOT ENTIER ('zephyrd' != 'zephyr.conf'). Pour une reponse tres courte (nombre,
    code), match exact de la forme normalisee."""
    on = re.sub(r"[^\w]", " ", (output or "").lower())
    en = re.sub(r"[^\w]", " ", (expected or "").lower()).strip()
    if not en or looks_degenerate(output):
        return False
    keys = [w for w in en.split() if len(w) > 2]
    if not keys:  # reponse courte (nombre, sigle) -> match exact
        return en in on
    o_words = set(on.split())
    return all(k in o_words for k in keys)


def ground_pairs(pairs, conversation_text: str):
    """Anti-hallucination : ne garde que les Q/R dont la REPONSE est ancree dans la
    conversation (vu via /info : le LLM extracteur invente des faits absents, ex.
    'path/to/your/model'). On exige qu'au moins la moitie des mots-cles de la reponse
    apparaissent dans la conversation."""
    conv = (conversation_text or "").lower()
    vague = {"beaucoup", "plusieurs", "environ", "certains", "divers", "nombreux"}
    out = []
    for q, a in pairs:
        al = (a or "").strip().lower()
        if not al or "path/to" in al or "your/" in al or "exemple" in al:
            continue
        words = al.split()
        if len(words) > 6:            # reponse-phrase = vague, on rejette
            continue
        if any(w in vague for w in words):
            continue
        keys = [w for w in re.sub(r"[^\w]", " ", al).split() if len(w) > 2] or words
        if keys and sum(1 for k in keys if k in conv) / len(keys) >= 0.5:
            out.append((q, a))
    return out


def extract_qa(conversation_text: str, generate_fn, n: int = 10) -> list[tuple[str, str]]:
    """Extrait des paires (question, reponse) FACTUELLES d'une conversation (recette
    SEAL "implications"). Crucial : folder la conversation brute apprend des accuses
    de reception bavards ; il faut des Q/R qui capturent le FAIT pour que le rappel
    fonctionne. `generate_fn(prompt, system)->str` = un LLM (idealement capable).
    """
    prompt = (
        "A partir du texte ci-dessous, extrais les FAITS importants sous forme de paires "
        "question/reponse AUTONOMES (comprehensibles sans le texte).\n"
        "REGLES STRICTES sur la reponse :\n"
        "- la reponse est un SPAN PRECIS copie du texte : un nom propre, un nombre, une "
        "date, ou un terme technique (1 a 5 mots MAX) ;\n"
        "- JAMAIS de phrase, ni de reponse vague ('beaucoup', 'plusieurs', 'environ') ;\n"
        "- une question ne porte que sur UN fait, et a une reponse non ambigue.\n"
        f"Donne jusqu'a {n} paires, une par ligne, au format EXACT :\n"
        "Q: <question precise> | R: <span court>\n\n"
        "Texte :\n" + conversation_text
    )
    try:
        raw = generate_fn(prompt, None) or ""
    except Exception:  # noqa: BLE001
        return []

    pairs: list[tuple[str, str]] = []
    # format inline "Q: ... | R: ..."
    inline = re.compile(r"(?i)^\s*Q\s*[:.\-]\s*(.+?)\s*\|\s*R\s*[:.\-]\s*(.+?)\s*$")
    for line in raw.splitlines():
        m = inline.match(line)
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
    # fallback : lignes Q:/R: alternees
    if len(pairs) < 2:
        pairs = []
        pending_q = None
        q_re = re.compile(r"(?i)^\s*Q\s*[:.\-]\s*(.+)$")
        a_re = re.compile(r"(?i)^\s*[RA]\s*[:.\-]\s*(.+)$")
        for line in raw.splitlines():
            qm = q_re.match(line)
            am = a_re.match(line)
            if qm:
                pending_q = qm.group(1).strip()
            elif am and pending_q:
                pairs.append((pending_q, am.group(1).strip()))
                pending_q = None
    pairs = [(q, a) for q, a in pairs if q and a]
    pairs = ground_pairs(pairs, conversation_text)  # anti-hallucination
    return pairs[:n]


def augment_pairs(pairs, generate_fn, n_paraphrases: int = 6, max_total: int = 90):
    """Augmentation type SEAL : pour chaque (question, reponse), garde l'original ET
    ajoute des reformulations de la QUESTION (meme reponse), pour que le LoRA apprenne
    le FAIT et pas le prompt exact. `generate_fn(prompt, system)->str` = un LLM.

    Robustesse : si la generation echoue/produit du bruit, on retombe sur l'original.
    """
    out: list[tuple[str, str]] = []
    for user_text, asst_text in pairs:
        out.append((user_text, asst_text))
        try:
            raw = generate_fn(
                "Reformule la question suivante de "
                f"{n_paraphrases} facons differentes (une par ligne, sans numero, "
                "sans y repondre). Question : " + user_text,
                None,
            )
        except Exception:  # noqa: BLE001
            raw = ""
        for line in (raw or "").splitlines():
            q = line.strip().lstrip("-*•0123456789.) ").strip()
            # garde des reformulations plausibles (assez courtes, non vides, != original)
            if 6 <= len(q) <= 200 and q.lower() != user_text.lower():
                out.append((q, asst_text))
        if len(out) >= max_total:
            break
    return out[:max_total]


# Jeu d'ANCRAGE (rehearsal) : comportements generiques que le modele de base maitrise
# deja. Les melanger a l'entrainement preserve les capacites generales pendant qu'on
# grave les faits -> casse la tension acquisition<->integrite (anti-oubli prouve).
ANCHOR_PAIRS: list[tuple[str, str]] = [
    ("Bonjour !", "Bonjour ! Comment puis-je vous aider ?"),
    ("Merci beaucoup.", "Avec plaisir !"),
    ("Quelle est la capitale de la France ?", "Paris."),
    ("Quelle est la capitale de l'Italie ?", "Rome."),
    ("Quelle est la capitale de l'Espagne ?", "Madrid."),
    ("Combien font 7 + 5 ?", "12."),
    ("Combien font 10 - 4 ?", "6."),
    ("Combien font 3 x 3 ?", "9."),
    ("Cite un fruit.", "Une pomme."),
    ("Quelle est la couleur du ciel par beau temps ?", "Bleu."),
    ("Quel jour vient apres lundi ?", "Mardi."),
    ("Traduis 'chat' en anglais.", "Cat."),
    ("Combien de jours dans une semaine ?", "Sept."),
    ("Quel est le contraire de 'grand' ?", "Petit."),
    ("Donne un synonyme de 'rapide'.", "Vite."),
    ("Sur quelle planete vivons-nous ?", "La Terre."),
    ("Dis bonjour en anglais.", "Hello."),
    ("Quelle est la capitale du Portugal ?", "Lisbonne."),
]


def _to_row(q: str, a: str) -> dict:
    return {"messages": [
        {"role": "user", "content": q.strip()},
        {"role": "assistant", "content": a.strip()},
    ]}


def build_chat_dataset(
    pairs: list[tuple[str, str]],
    out_dir: str,
    repeat: int = 4,
    anchors: list | None = None,
    anchor_repeat: int = 2,
) -> int:
    """Ecrit train.jsonl / valid.jsonl au format {"messages":[user, assistant]}.

    `pairs` = faits (q, r) ; `repeat` les duplique pour favoriser la memorisation.
    `anchors` = rehearsal (exemples generiques) melanges au train pour preserver les
    capacites generales. La valid ne contient que des faits (mesure l'overfit cote faits).
    """
    os.makedirs(out_dir, exist_ok=True)
    fact_rows = [_to_row(q, a) for q, a in pairs if (q or "").strip() and (a or "").strip()]
    if not fact_rows:  # garde-fou : jamais d'entrainement sur 0 fait
        return 0

    train = fact_rows * max(1, repeat)
    if anchors:
        anchor_rows = [_to_row(q, a) for q, a in anchors if (q or "").strip() and (a or "").strip()]
        train += anchor_rows * max(1, anchor_repeat)
    valid = fact_rows[: max(1, len(fact_rows) // 5)]

    with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "valid.jsonl"), "w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(train)


def split_train_eval(pairs, heldout_per_answer: int = 1):
    """Separe les paires en (train, held-out) en gardant, pour CHAQUE reponse, une
    reformulation de cote pour l'eval. La gate teste ainsi le rappel sur des
    questions NON vues a l'entrainement (vrai test de generalisation, pas de fuite)."""
    groups: dict[str, list] = {}
    order: list[str] = []
    for q, a in pairs:
        k = (a or "").strip().lower()
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append((q, a))
    train, evl = [], []
    for k in order:
        items = groups[k]
        if len(items) <= 1:
            train.extend(items)  # pas assez pour tenir un held-out
            continue
        h = min(heldout_per_answer, len(items) - 1)  # garder >=1 en train
        evl.extend(items[:h])
        train.extend(items[h:])
    return train, evl


def train_lora(
    base_model: str,
    data_dir: str,
    adapter_out: str,
    *,
    iters: int = 120,
    num_layers: int = 8,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    max_seq_length: int = 512,
    rank: int | None = None,
    mask_prompt: bool = False,
    python_exe: str | None = None,
    log_file: str | None = None,
) -> dict:
    """Entraine un LoRA via `mlx_lm lora --train` (sous-processus).

    Retourne un dict : {ok, adapter_path, adapter_file, train_loss, val_loss, iters,
    returncode, log_tail}.
    """
    py = python_exe or sys.executable
    os.makedirs(adapter_out, exist_ok=True)
    cmd = [
        py, "-m", "mlx_lm", "lora",
        "--model", base_model,
        "--train",
        "--data", data_dir,
        "--fine-tune-type", "lora",
        "--num-layers", str(num_layers),
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--val-batches", "1",
        "--steps-per-eval", str(iters),
        "--max-seq-length", str(max_seq_length),
        "--learning-rate", str(learning_rate),
        "--adapter-path", adapter_out,
    ]
    if mask_prompt:  # loss uniquement sur la reponse (masque le prompt/contexte)
        cmd.append("--mask-prompt")
    if rank:  # rang LoRA via fichier de config (pas de flag CLI dedie)
        cfg_path = os.path.join(adapter_out, "lora_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write(f"lora_parameters:\n  rank: {int(rank)}\n  scale: 20.0\n  dropout: 0.0\n")
        cmd += ["-c", cfg_path]
    if log_file:  # stream live ligne par ligne (suivi en direct via tail -f)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n--- mlx_lm lora: iters={iters} layers={num_layers} "
                     f"lr={learning_rate} rank={rank} ---\n")
            lf.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            lines = []
            for line in proc.stdout:
                lines.append(line)
                lf.write(line)
                lf.flush()
            proc.wait()
        out = "".join(lines)
        rc = proc.returncode
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        rc = proc.returncode

    train_loss = val_loss = None
    for m in _LOSS_RE.finditer(out):
        if m.group(1) == "Train":
            train_loss = float(m.group(2))
        else:
            val_loss = float(m.group(2))

    adapter_file = os.path.join(adapter_out, "adapters.safetensors")
    ok = rc == 0 and os.path.exists(adapter_file)
    tail = "\n".join(out.strip().splitlines()[-8:])
    return {
        "ok": ok,
        "adapter_path": adapter_out,
        "adapter_file": adapter_file,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "iters": iters,
        "rank": rank,
        "returncode": rc,
        "log_tail": tail,
    }
