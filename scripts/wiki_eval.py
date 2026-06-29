"""Éval D2L sur une page Wikipédia (style SEAL no-context QA).

Pipeline : fetch wiki -> extraction Q/R groundée -> augmentation -> split held-out
-> SCORE BASE (modèle seul) -> entraînement LoRA (+ ancrage) -> SCORE BASE+LoRA.
Le held-out = questions NON entraînées ; le delta base->LoRA = ce que le LoRA a internalisé.

Usage : ./.venv/bin/python scripts/wiki_eval.py [Titre] [lang]
  ex : M0_BACKEND=mlx M0_MLX_MODEL_PATH=models/qwen2.5-7b-it-mlx-4bit \
       ./.venv/bin/python scripts/wiki_eval.py Tardigrade fr
"""

from __future__ import annotations

import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import httpx  # noqa: E402

from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402


def fetch_wiki(title: str, lang: str = "fr", max_chars: int = 6000) -> tuple[str, str]:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "format": "json", "action": "query", "prop": "extracts",
        "explaintext": 1, "redirects": 1, "titles": title,
    }
    r = httpx.get(url, params=params, timeout=30.0,
                  headers={"User-Agent": "m0-wiki-eval/0.1"})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("title", title), (page.get("extract", "") or "")[:max_chars]


def score(llm, pairs):
    ok, samples = 0, []
    for q, a in pairs:
        out = llm.generate(q, None)
        good = d2l.answer_recalled(out, a)
        ok += int(good)
        samples.append((q, a, out.replace("\n", " ")[:70], good))
    return (ok / max(1, len(pairs))), samples


def main():
    title_arg = sys.argv[1] if len(sys.argv) > 1 else "Tardigrade"
    lang = sys.argv[2] if len(sys.argv) > 2 else "fr"

    title, extract = fetch_wiki(title_arg, lang)
    print(f"PAGE : {title} ({lang}.wikipedia) — {len(extract)} caractères")
    if len(extract) < 400:
        print("Extrait trop court, abandon.")
        return

    cfg = Config.from_env()
    cfg.backend = "mlx"
    llm = make_client(cfg)

    print("\n[1] Extraction Q/R groundée…")
    qa = d2l.extract_qa(extract, llm.generate, n=18)
    qa = d2l.clean_and_balance(qa, max_per_answer=2)
    print(f"    {len(qa)} faits Q/R groundés")
    aug = d2l.augment_pairs(qa, llm.generate, n_paraphrases=2)
    aug = d2l.clean_and_balance(aug, max_per_answer=6) or qa
    train_pairs, eval_pairs = d2l.split_train_eval(aug, heldout_per_answer=1)
    if not eval_pairs:
        eval_pairs = qa
    print(f"    train={len(train_pairs)}  held-out={len(eval_pairs)}")

    print("\n[2] SCORE BASE (modèle seul, sans LoRA) sur held-out…")
    base_score, base_samples = score(llm, eval_pairs)
    print(f"    base recall = {base_score:.0%}")

    print("\n[3] Entraînement LoRA (+ ancrage)…")
    data_dir = os.path.join(_PROJ, "logs", "wiki_data")
    adapter = os.path.join(_PROJ, "models", "lora", "wiki")
    n = d2l.build_chat_dataset(train_pairs, data_dir, repeat=cfg.d2l_repeat,
                               anchors=d2l.ANCHOR_PAIRS, anchor_repeat=cfg.d2l_anchor_repeat)
    res = d2l.train_lora(cfg.mlx_model_path, data_dir, adapter,
                         iters=cfg.d2l_iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16,
                         python_exe=sys.executable)
    print(f"    train ok={res['ok']}  val_loss={res['val_loss']}  ({n} lignes, {cfg.d2l_iters} iters)")
    if not res["ok"]:
        print(res["log_tail"])
        return

    print("\n[4] SCORE BASE+LoRA sur held-out…")
    llm.set_adapter(adapter)
    lora_score, lora_samples = score(llm, eval_pairs)
    print(f"    base+LoRA recall = {lora_score:.0%}")

    print("\n=== RÉSULTAT ===")
    print(f"held-out recall : base {base_score:.0%}  ->  base+LoRA {lora_score:.0%}  "
          f"(delta {lora_score - base_score:+.0%})")
    print("\n--- détail (base -> +LoRA) ---")
    for (q, a, bo, bg), (_, _, lo, lg) in zip(base_samples, lora_samples):
        print(f"Q: {q}")
        print(f"   attendu : {a}")
        print(f"   base    : [{'OK' if bg else 'X '}] {bo}")
        print(f"   +LoRA   : [{'OK' if lg else 'X '}] {lo}")


if __name__ == "__main__":
    main()
