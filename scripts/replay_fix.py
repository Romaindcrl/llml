"""Replay équitable : 1 LoRA entraîné sur A+B avec assez d'iters (le run M3 l'avait
sous-entraîné : val_loss 4.48 @ 80 iters sur ~2x les données). On régénère les jeux
(déterministe), on entraîne, on évalue base vs replay sur les held-out des 2 pages.
Live : tail -f logs/replay_fix.log
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

PAGES = [("Tardigrade", "fr"), ("Aurore polaire", "fr")]
LOG_PATH = os.path.join(_PROJ, "logs", "replay_fix.log")
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
                  timeout=30.0, headers={"User-Agent": "m0-replay/0.1"})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("title", title), (page.get("extract", "") or "")[:max_chars]


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    iters = int(os.environ.get("REPLAY_ITERS", "240"))
    log(f"=== Replay équitable (iters={iters}, rang 32) ===")
    cfg = Config.from_env(); cfg.backend = "mlx"
    llm = make_client(cfg)

    def prep(title, lang):
        t, extract = fetch_wiki(title, lang)
        qa = d2l.clean_and_balance(d2l.extract_qa(extract, llm.generate, n=16), max_per_answer=2)
        aug = d2l.clean_and_balance(
            d2l.augment_pairs(qa, llm.generate, n_paraphrases=2), max_per_answer=6) or qa
        tr, ev = d2l.split_train_eval(aug, 1)
        return t, tr, (ev or qa)

    log("[1/3] Préparation…")
    tA, trA, evA = prep(*PAGES[0]); tB, trB, evB = prep(*PAGES[1])
    log(f"[1/3] train union={len(trA) + len(trB)} | held-out {tA}={len(evA)} {tB}={len(evB)}")

    log(f"[2/3] Entraînement replay (union A+B, {iters} iters)…")
    data = os.path.join(_PROJ, "logs", "replay_fix_data")
    d2l.build_chat_dataset(trA + trB, data, repeat=cfg.d2l_repeat,
                           anchors=d2l.ANCHOR_PAIRS, anchor_repeat=cfg.d2l_anchor_repeat)
    res = d2l.train_lora(cfg.mlx_model_path, data, f"{L}/replay_fix", iters=iters,
                         num_layers=cfg.d2l_num_layers, learning_rate=cfg.d2l_learning_rate,
                         rank=32, python_exe=sys.executable, log_file=LOG_PATH)
    log(f"[2/3] replay ok={res['ok']} val_loss={res['val_loss']}")

    def score(adapter, pairs):
        llm.set_adapter(adapter)
        return sum(d2l.answer_recalled(llm.generate(q, None), a) for q, a in pairs) / max(1, len(pairs))

    log("[3/3] Évaluation…")
    for name, ad in [("base", None), (f"replay(r32,{iters}it)", f"{L}/replay_fix")]:
        sa, sb = score(ad, evA), score(ad, evB)
        log(f"[eval] {name:18s} | {tA[:16]}={sa:4.0%} | {tB[:16]}={sb:4.0%} | moy={ (sa+sb)/2:4.0%}")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
