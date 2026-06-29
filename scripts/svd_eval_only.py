"""Éval seule de l'expérience M3 : réutilise les adapters svd_* déjà entraînés et
mesure le recall held-out de base / A-seul / B-seul / stacked(r32) / rebased(r16).

Les jeux held-out sont régénérés (extraction déterministe à temp=0). Log live :
  tail -f logs/svd_eval.log
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
LOG_PATH = os.path.join(_PROJ, "logs", "svd_eval.log")
_T0 = time.time()


def log(msg=""):
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
        timeout=30.0, headers={"User-Agent": "m0-svd-eval/0.1"})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("title", title), (page.get("extract", "") or "")[:max_chars]


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    log("=== Éval M3 (adapters réutilisés) ===")
    cfg = Config.from_env()
    cfg.backend = "mlx"
    llm = make_client(cfg)

    def prep_eval(title, lang):
        t, extract = fetch_wiki(title, lang)
        qa = d2l.clean_and_balance(d2l.extract_qa(extract, llm.generate, n=16), max_per_answer=2)
        aug = d2l.clean_and_balance(
            d2l.augment_pairs(qa, llm.generate, n_paraphrases=2), max_per_answer=6) or qa
        _, ev = d2l.split_train_eval(aug, 1)
        ev = ev or qa
        log(f"[prep] {t}: held-out={len(ev)}")
        return t, ev

    log("[1/2] Régénération des held-out…")
    tA, evA = prep_eval(*PAGES[0])
    tB, evB = prep_eval(*PAGES[1])

    def score(adapter, pairs):
        llm.set_adapter(adapter)
        ok = sum(d2l.answer_recalled(llm.generate(q, None), a) for q, a in pairs)
        return ok / max(1, len(pairs))

    log("[2/2] Évaluation des variantes…")
    L = os.path.join(_PROJ, "models", "lora")
    variants = [("base", None), ("A-seul(r16)", f"{L}/svd_A"), ("B-seul(r16)", f"{L}/svd_B"),
                ("stacked(r32)", f"{L}/svd_stacked"), ("rebased(r16)", f"{L}/svd_rebased")]
    rows = []
    for name, ad in variants:
        sa, sb = score(ad, evA), score(ad, evB)
        rows.append((name, sa, sb))
        log(f"[eval] {name:14s} | {tA[:16]}={sa:4.0%} | {tB[:16]}={sb:4.0%}")

    stacked = lora_merge.concat_adapters(
        lora_merge.load_adapter(f"{L}/svd_A"), lora_merge.load_adapter(f"{L}/svd_B"))
    energy = lora_merge.truncation_energy(stacked, 16)

    log("")
    log("=== TABLE M3 — recall held-out ===")
    log(f"{'variante':14s} | {tA[:16]:>16s} | {tB[:16]:>16s} | moyenne")
    for name, sa, sb in rows:
        log(f"{name:14s} | {sa:15.0%} | {sb:15.0%} | {(sa + sb) / 2:6.0%}")
    log(f"énergie SVD conservée (32->16): {energy:.1%}")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
