"""Expérience M3 : empilage de connaissances (concat LoRA) + rebase SVD.

1. Entraine un LoRA par page Wikipedia (rang 16 chacun).
2. CONCAT -> rang 32 : empile les DEUX pages dans les poids (le rang s'additionne).
3. REBASE SVD -> rang 16 : compresse, borne le rang.
4. Mesure le recall (held-out de CHAQUE page) pour : base, A-seul, B-seul, stacked(32), rebased(16).

Suivi EN DIRECT : tout est ecrit dans logs/svd_run.log -> `tail -f logs/svd_run.log`.

Usage : M0_BACKEND=mlx M0_MLX_MODEL_PATH=models/qwen2.5-7b-it-mlx-4bit \
        ./.venv/bin/python scripts/svd_experiment.py
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import httpx  # noqa: E402

from m0 import d2l, lora_merge  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

PAGES = [("Tardigrade", "fr"), ("Aurore polaire", "fr")]
LOG_PATH = os.path.join(_PROJ, "logs", "svd_run.log")
_T0 = time.time()


def log(msg: str = "") -> None:
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def fetch_wiki(title, lang="fr", max_chars=6000):
    r = httpx.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={"format": "json", "action": "query", "prop": "extracts",
                "explaintext": 1, "redirects": 1, "titles": title},
        timeout=30.0, headers={"User-Agent": "m0-svd-exp/0.1"},
    )
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("title", title), (page.get("extract", "") or "")[:max_chars]


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()  # reset le log
    log("=== Expérience M3 : stacking LoRA + rebase SVD ===")

    cfg = Config.from_env()
    cfg.backend = "mlx"
    log("Chargement du modèle MLX…")
    llm = make_client(cfg)

    def prep(title, lang):
        log(f"[prep] fetch + extraction Q/R : {title}…")
        t, extract = fetch_wiki(title, lang)
        qa = d2l.clean_and_balance(d2l.extract_qa(extract, llm.generate, n=16), max_per_answer=2)
        log(f"[prep] {t}: {len(qa)} faits extraits, augmentation…")
        aug = d2l.clean_and_balance(
            d2l.augment_pairs(qa, llm.generate, n_paraphrases=2), max_per_answer=6) or qa
        tr, ev = d2l.split_train_eval(aug, 1)
        ev = ev or qa
        log(f"[prep] {t}: train={len(tr)} held-out={len(ev)}")
        return t, tr, ev

    log("[1/4] Préparation des 2 pages")
    tA, trA, evA = prep(*PAGES[0])
    tB, trB, evB = prep(*PAGES[1])

    def train(tag, train_pairs):
        data = os.path.join(_PROJ, "logs", f"svd_{tag}_data")
        adapt = os.path.join(_PROJ, "models", "lora", f"svd_{tag}")
        d2l.build_chat_dataset(train_pairs, data, repeat=cfg.d2l_repeat,
                               anchors=d2l.ANCHOR_PAIRS, anchor_repeat=cfg.d2l_anchor_repeat)
        log(f"[train] LoRA {tag} : entraînement (iters={cfg.d2l_iters})… (détail ci-dessous)")
        res = d2l.train_lora(cfg.mlx_model_path, data, adapt, iters=cfg.d2l_iters,
                             num_layers=cfg.d2l_num_layers, learning_rate=cfg.d2l_learning_rate,
                             rank=16, python_exe=sys.executable, log_file=LOG_PATH)
        log(f"[train] LoRA {tag} : ok={res['ok']} val_loss={res['val_loss']}")
        return adapt

    log("[2/4] Entraînement LoRA page A puis page B (rang 16)")
    adA = train("A", trA)
    adB = train("B", trB)

    log("[3/4] Chirurgie : concat (rang 32) -> rebase SVD (rang 16)")
    A = lora_merge.load_adapter(adA)
    B = lora_merge.load_adapter(adB)
    stacked = lora_merge.concat_adapters(A, B)
    energy = lora_merge.truncation_energy(stacked, 16)
    adStack = os.path.join(_PROJ, "models", "lora", "svd_stacked")
    adReb = os.path.join(_PROJ, "models", "lora", "svd_rebased")
    lora_merge.save_adapter(stacked, adStack, 32, os.path.join(adA, "adapter_config.json"))
    lora_merge.save_adapter(lora_merge.svd_rebase(stacked, 16), adReb, 16,
                            os.path.join(adA, "adapter_config.json"))
    log(f"[surgery] énergie spectrale conservée par rebase 32->16 : {energy:.1%}")

    def score(adapter, pairs):
        llm.set_adapter(adapter)
        ok = sum(d2l.answer_recalled(llm.generate(q, None), a) for q, a in pairs)
        return ok / max(1, len(pairs))

    log("[4/4] Évaluation recall (held-out)")
    variants = [("base", None), ("A-seul(r16)", adA), ("B-seul(r16)", adB),
                ("stacked(r32)", adStack), ("rebased(r16)", adReb)]
    rows = []
    for name, ad in variants:
        sa, sb = score(ad, evA), score(ad, evB)
        rows.append((name, sa, sb))
        log(f"[eval] {name:14s} | {tA[:16]}={sa:4.0%} | {tB[:16]}={sb:4.0%}")

    log("")
    log("=== TABLE M3 — recall held-out ===")
    log(f"{'variante':14s} | {tA[:16]:>16s} | {tB[:16]:>16s} | moyenne")
    for name, sa, sb in rows:
        log(f"{name:14s} | {sa:15.0%} | {sb:15.0%} | {(sa + sb) / 2:6.0%}")
    log(f"énergie SVD conservée (32->16): {energy:.1%}")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
