"""M3 — comparaison des stratégies de mémoire cumulative sur 2 pages wiki.

Variantes évaluées (recall held-out de chaque page) :
  base · A-seul(r16) · B-seul(r16) · stacked(r32, concat naïf) · rebased(r16, SVD)
  · TIES(r16, trim+sign) · replay(r32, 1 LoRA entraîné sur A+B)

Réutilise svd_A/svd_B/svd_stacked/svd_rebased (disque) ; construit TIES ; entraîne replay.
Suivi live : tail -f logs/m3_compare.log
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
LOG_PATH = os.path.join(_PROJ, "logs", "m3_compare.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def fetch_wiki(title, lang="fr", max_chars=6000):
    r = httpx.get(f"https://{lang}.wikipedia.org/w/api.php",
                  params={"format": "json", "action": "query", "prop": "extracts",
                          "explaintext": 1, "redirects": 1, "titles": title},
                  timeout=30.0, headers={"User-Agent": "m0-m3/0.1"})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("title", title), (page.get("extract", "") or "")[:max_chars]


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    log("=== M3 comparaison : concat / SVD / TIES / replay ===")
    cfg = Config.from_env()
    cfg.backend = "mlx"
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

    log("[1/4] Préparation des 2 pages")
    tA, trA, evA = prep(*PAGES[0])
    tB, trB, evB = prep(*PAGES[1])

    log("[2/4] (a) TIES merge de A + B (trim 0.2 + élection de signe) -> rang 16")
    ties = lora_merge.ties_merge(
        lora_merge.load_adapter(f"{L}/svd_A"), lora_merge.load_adapter(f"{L}/svd_B"),
        density=0.2, target_rank=16)
    lora_merge.save_adapter(ties, f"{L}/svd_ties", 16, f"{L}/svd_A/adapter_config.json")
    log("[2/4] TIES sauvé.")

    log("[3/4] (b) replay : 1 LoRA entraîné sur A+B (rang 32)…")
    data = os.path.join(_PROJ, "logs", "svd_replay_data")
    d2l.build_chat_dataset(trA + trB, data, repeat=cfg.d2l_repeat,
                           anchors=d2l.ANCHOR_PAIRS, anchor_repeat=cfg.d2l_anchor_repeat)
    res = d2l.train_lora(cfg.mlx_model_path, data, f"{L}/svd_replay", iters=cfg.d2l_iters,
                         num_layers=cfg.d2l_num_layers, learning_rate=cfg.d2l_learning_rate,
                         rank=32, python_exe=sys.executable, log_file=LOG_PATH)
    log(f"[3/4] replay : ok={res['ok']} val_loss={res['val_loss']}")

    def score(adapter, pairs):
        llm.set_adapter(adapter)
        ok = sum(d2l.answer_recalled(llm.generate(q, None), a) for q, a in pairs)
        return ok / max(1, len(pairs))

    log("[4/4] Évaluation des 7 variantes…")
    variants = [
        ("base", None), ("A-seul(r16)", f"{L}/svd_A"), ("B-seul(r16)", f"{L}/svd_B"),
        ("stacked(r32)", f"{L}/svd_stacked"), ("rebased(r16)", f"{L}/svd_rebased"),
        ("TIES(r16)", f"{L}/svd_ties"), ("replay(r32)", f"{L}/svd_replay"),
    ]
    rows = []
    for name, ad in variants:
        sa, sb = score(ad, evA), score(ad, evB)
        rows.append((name, sa, sb))
        log(f"[eval] {name:14s} | {tA[:16]}={sa:4.0%} | {tB[:16]}={sb:4.0%}")

    log("")
    log("=== TABLE M3 — recall held-out (cumulatif) ===")
    log(f"{'variante':14s} | {tA[:16]:>16s} | {tB[:16]:>16s} | moyenne")
    for name, sa, sb in rows:
        log(f"{name:14s} | {sa:15.0%} | {sb:15.0%} | {(sa + sb) / 2:6.0%}")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
