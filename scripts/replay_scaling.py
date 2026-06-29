"""Piste 1 — replay sur doc mémoire croissant (scaling).

À chaque étape k, le "doc mémoire" accumule les faits des pages 1..k, et on REENTRAINE
un LoRA FRAIS depuis la base sur tout le doc (pas de fusion d'adapters). On mesure le
recall held-out moyen sur les k pages -> courbe directement comparable à TIES scaling.

Itérations proportionnelles à la taille du corpus (sinon sous-entraînement).
Live : tail -f logs/replay_scaling_<TAG>.log
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import httpx  # noqa: E402

from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

PAGES = [("Tardigrade", "fr"), ("Aurore polaire", "fr"), ("Geyser", "fr"),
         ("Horloge atomique", "fr")]
TAG = os.environ.get("M0_RUN_TAG", "q8")
LOG_PATH = os.path.join(_PROJ, "logs", f"replay_scaling_{TAG}.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def fetch_wiki(title, lang="fr", max_chars=6000):
    r = httpx.get(f"https://{lang}.wikipedia.org/w/api.php",
                  params={"format": "json", "action": "query", "prop": "extracts",
                          "explaintext": 1, "redirects": 1, "titles": title},
                  timeout=30.0, headers={"User-Agent": "m0-replay-scaling/0.1"})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("title", title), (page.get("extract", "") or "")[:max_chars]


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== REPLAY scaling (doc croissant), modèle={os.path.basename(cfg.mlx_model_path)} ===")
    llm = make_client(cfg)

    def prep(title, lang):
        t, extract = fetch_wiki(title, lang)
        qa = d2l.clean_and_balance(d2l.extract_qa(extract, llm.generate, n=16), max_per_answer=2)
        aug = d2l.clean_and_balance(
            d2l.augment_pairs(qa, llm.generate, n_paraphrases=2), max_per_answer=6) or qa
        tr, ev = d2l.split_train_eval(aug, 1)
        ev = ev or qa
        log(f"[prep] {t}: train={len(tr)} held-out={len(ev)}")
        return t, tr, ev

    log("[1/2] Préparation des pages")
    prepped = [prep(*p) for p in PAGES]
    titles = [t for t, _, _ in prepped]
    evals = [ev for _, _, ev in prepped]

    def score(adapter, pairs):
        llm.set_adapter(adapter)
        return sum(d2l.answer_recalled(llm.generate(q, None), a) for q, a in pairs) / max(1, len(pairs))

    log("[2/2] Replay incrémental : réentraînement sur le doc croissant")
    corpus = []
    scaling = []
    for k in range(1, len(prepped) + 1):
        corpus += prepped[k - 1][1]  # accumule les train pairs de la page k
        iters = 100 * k               # iters proportionnelles au corpus
        data = f"{_PROJ}/logs/replay_scal_{TAG}_k{k}_data"
        adapt = f"{L}/replay_scal_{TAG}_k{k}"
        d2l.build_chat_dataset(corpus, data, repeat=cfg.d2l_repeat,
                               anchors=d2l.ANCHOR_PAIRS, anchor_repeat=cfg.d2l_anchor_repeat)
        log(f"[train] k={k} : corpus={len(corpus)} paires, {iters} iters (rang 32)…")
        res = d2l.train_lora(cfg.mlx_model_path, data, adapt, iters=iters,
                             num_layers=cfg.d2l_num_layers, learning_rate=cfg.d2l_learning_rate,
                             rank=16, python_exe=sys.executable, log_file=LOG_PATH)
        pers = [score(adapt, evals[i]) for i in range(k)]
        avg = sum(pers) / k
        scaling.append((k, avg, pers))
        log(f"[scal] k={k} replay : moy={avg:.0%}  val_loss={res['val_loss']}  détail=" +
            " ".join(f"{titles[i][:8]}={pers[i]:.0%}" for i in range(k)))

    log("")
    log("=== TABLE REPLAY scaling (recall moyen sur k pages, doc croissant) ===")
    for k, avg, pers in scaling:
        log(f"k={k} | moyenne={avg:4.0%} | " + " ".join(
            f"{titles[i][:10]}={pers[i]:.0%}" for i in range(len(pers))))
    log("=== FIN ===")


if __name__ == "__main__":
    main()
