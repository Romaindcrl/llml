"""M3 scaling : TIES sur N pages (accumulation incrémentale k=1..N).

Pour chaque page : LoRA spécialiste (rang 16). Puis pour k=2..N on TIES-merge les k
premiers et on mesure le recall MOYEN sur les k pages accumulées. Montre si l'empilage
TIES TIENT quand la mémoire grandit, ou se dégrade.

Paramétrable par modèle (4-bit / 8-bit) :
  M0_MLX_MODEL_PATH=... M0_RUN_TAG=q4|q8 ./.venv/bin/python scripts/ties_scaling.py
Live : tail -f logs/ties_scaling_<TAG>.log
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

PAGES = [("Tardigrade", "fr"), ("Aurore polaire", "fr"), ("Geyser", "fr"),
         ("Horloge atomique", "fr")]
TAG = os.environ.get("M0_RUN_TAG", "q4")
LOG_PATH = os.path.join(_PROJ, "logs", f"ties_scaling_{TAG}.log")
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
                  timeout=30.0, headers={"User-Agent": "m0-ties-scaling/0.1"})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("title", title), (page.get("extract", "") or "")[:max_chars]


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== TIES scaling (TAG={TAG}, modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    def prep(title, lang):
        t, extract = fetch_wiki(title, lang)
        qa = d2l.clean_and_balance(d2l.extract_qa(extract, llm.generate, n=16), max_per_answer=2)
        aug = d2l.clean_and_balance(
            d2l.augment_pairs(qa, llm.generate, n_paraphrases=2), max_per_answer=6) or qa
        tr, ev = d2l.split_train_eval(aug, 1)
        ev = ev or qa
        log(f"[prep] {t}: {len(qa)} faits, train={len(tr)} held-out={len(ev)}")
        return t, tr, ev

    log("[1/3] Préparation des pages")
    prepped = [prep(*p) for p in PAGES]
    titles = [t for t, _, _ in prepped]
    evals = [ev for _, _, ev in prepped]

    log("[2/3] Entraînement d'un LoRA spécialiste par page (rang 16)")
    adapters = []
    for i, (t, tr, _) in enumerate(prepped):
        ad = f"{L}/scal_{TAG}_{i}"
        d2l.build_chat_dataset(tr, f"{_PROJ}/logs/scal_{TAG}_{i}_data", repeat=cfg.d2l_repeat,
                               anchors=d2l.ANCHOR_PAIRS, anchor_repeat=cfg.d2l_anchor_repeat)
        res = d2l.train_lora(cfg.mlx_model_path, f"{_PROJ}/logs/scal_{TAG}_{i}_data", ad,
                             iters=cfg.d2l_iters, num_layers=cfg.d2l_num_layers,
                             learning_rate=cfg.d2l_learning_rate, rank=16,
                             python_exe=sys.executable, log_file=LOG_PATH)
        log(f"[train] {t}: ok={res['ok']} val_loss={res['val_loss']}")
        adapters.append((ad, lora_merge.load_adapter(ad)))

    def score(adapter_path, pairs):
        llm.set_adapter(adapter_path)
        return sum(d2l.answer_recalled(llm.generate(q, None), a) for q, a in pairs) / max(1, len(pairs))

    log("[3/3] Accumulation TIES incrémentale")
    scaling = []  # (k, recall moyen sur les k pages)
    # k=1 : adaptateur seul
    r1 = score(adapters[0][0], evals[0])
    scaling.append((1, r1, [r1]))
    log(f"[scal] k=1 ({titles[0][:14]}) : {r1:.0%}")
    for k in range(2, len(adapters) + 1):
        merged = lora_merge.ties_merge(*[ad for _, ad in adapters[:k]],
                                       density=0.2, target_rank=32)
        mpath = f"{L}/scal_{TAG}_ties{k}"
        lora_merge.save_adapter(merged, mpath, 32, f"{adapters[0][0]}/adapter_config.json")
        pers = [score(mpath, evals[i]) for i in range(k)]
        avg = sum(pers) / k
        scaling.append((k, avg, pers))
        log(f"[scal] k={k} TIES : moy={avg:.0%}  détail=" +
            " ".join(f"{titles[i][:8]}={pers[i]:.0%}" for i in range(k)))

    log("")
    log("=== TABLE scaling TIES (recall moyen sur k pages accumulées) ===")
    for k, avg, pers in scaling:
        log(f"k={k} | moyenne={avg:4.0%} | " + " ".join(
            f"{titles[i][:10]}={pers[i]:.0%}" for i in range(len(pers))))
    log("=== FIN ===")


if __name__ == "__main__":
    main()
