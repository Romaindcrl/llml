"""Benchmark du SYSTÈME ROUTÉ (la conclusion du projet) vs méthodes seules.

Routeur : RAPPEL de faits -> POIDS (LoRA, entraîné UNIQUEMENT sur des faits) ;
          GÉNÉRATION (code) -> BASE + RAG (référence en contexte).
Charge de travail MIXTE : questions factuelles + tâches de génération de code.

SUT : RAG-seul · compaction-seule · poids-seuls · ROUTÉ(nous).
Live : tail -f logs/benchmark_routed.log
"""

from __future__ import annotations

import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

import httpx  # noqa: E402

from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.rag import RAG, is_generation  # noqa: E402

WIKI = [("Tardigrade", "fr"), ("Stromboli", "fr"), ("Aurore polaire", "fr")]
LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_routed.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()

GLYPH_DOC = (
    "Framework Glyph v2.4 — API Python de rendu graphique.\n"
    "glyph.load(path) charge un fichier et retourne un objet Canvas.\n"
    "Canvas.render(mode) rend le canvas ; le mode par defaut est 'vector', l'autre 'raster'.\n"
    "Canvas.export(path, dpi) exporte ; le dpi par defaut est 300.\n"
    "Tout chemin d'export doit finir par '.glx'.\n"
    "glyph.palette(name) retourne une palette ; la palette par defaut s'appelle 'sol'.\n"
    "La constante glyph.MAX_LAYERS vaut 64.\n"
    "Canvas.merge(other, opacity) fusionne deux canvas.\n"
    "Canvas.add_layer(layer) ajoute une couche.\n"
    "glyph.Color(r, g, b) construit une couleur RVB.\n"
    "Canvas.rotate(degrees) fait pivoter le canvas.\n"
    "Canvas.save_project(path) sauvegarde au format '.glxproj'.\n"
)
GLYPH_QA = [
    ("Quelle fonction Glyph charge un fichier ?", "glyph.load"),
    ("Quel est le mode de rendu par defaut ?", "vector"),
    ("Quel est le dpi d'export par defaut ?", "300"),
    ("Comment s'appelle la palette par defaut ?", "sol"),
    ("Combien vaut glyph.MAX_LAYERS ?", "64"),
    ("Quelle methode fusionne deux canvas ?", "merge"),
]
CODE_TASKS = [
    ("Ecris un script Glyph qui charge 'logo.png', le rend en raster, puis l'exporte a 600 dpi.",
     ["glyph.load", "render", "raster", "export", "600", ".glx"]),
    ("Ecris un script Glyph qui charge deux images, les fusionne a 0.4 d'opacite, puis exporte.",
     ["glyph.load", "merge", "0.4", "render", "export", ".glx"]),
    ("Ecris un script Glyph qui cree un canvas, ajoute une couche couleur rouge, et sauvegarde le projet.",
     ["add_layer", "color", "save_project", "glxproj"]),
]


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def fetch_wiki(title, lang="fr", max_chars=4000):
    r = httpx.get(f"https://{lang}.wikipedia.org/w/api.php",
                  params={"format": "json", "action": "query", "prop": "extracts",
                          "explaintext": 1, "redirects": 1, "titles": title},
                  timeout=30.0, headers={"User-Agent": "m0-routed/0.1"})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("title", title), (page.get("extract", "") or "")[:max_chars]


def score_code(code, checklist):
    c = code.lower()
    return sum(1 for tok in checklist if tok.lower() in c) / len(checklist)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== BENCHMARK ROUTÉ (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    # 1) corpus + items
    log("[1/3] Préparation (faits + code)")
    rag = RAG(os.path.join(_PROJ, "logs", "rag_bench.txt")); rag.clear()
    train_qa, fact_items = [], []
    for title, lang in WIKI:
        log(f"  extraction {title}…")
        t, text = fetch_wiki(title, lang)
        rag.add_document(text)
        qa = d2l.clean_and_balance(d2l.extract_qa(text, llm.generate, n=12), max_per_answer=2)
        tr, ev = d2l.split_train_eval(d2l.clean_and_balance(
            d2l.augment_pairs(qa, llm.generate, n_paraphrases=2), max_per_answer=6) or qa, 1)
        train_qa += tr
        seen = set()
        for q, a in ev:
            if a.lower() not in seen:
                seen.add(a.lower()); fact_items.append((q, a))
                if len(seen) >= 3:
                    break
    # glyph : doc -> RAG ; QA -> faits (train + eval)
    rag.add_document(GLYPH_DOC)
    gtr, gev = d2l.split_train_eval(d2l.clean_and_balance(
        d2l.augment_pairs(GLYPH_QA, llm.generate, n_paraphrases=2), max_per_answer=6) or GLYPH_QA, 1)
    train_qa += gtr
    fact_items += gev[:3]
    code_items = list(CODE_TASKS)
    log(f"  faits eval={len(fact_items)} · code eval={len(code_items)} · RAG chunks={rag.count()}")

    summary = llm.generate("Resume ce corpus en faits concis (garde chiffres/noms/API exacts) :\n"
                           + "\n".join(rag.chunks), None)

    # 2) NOTRE LoRA : entraîné UNIQUEMENT sur des faits
    log("[2/3] Entraînement LoRA faits-seulement")
    data = f"{_PROJ}/logs/routed_data"
    adapter = f"{L}/routed"
    n = d2l.build_chat_dataset(train_qa, data, repeat=4, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=4)
    iters = min(500, max(200, 12 * len(train_qa)))
    res = d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16, python_exe=sys.executable,
                         log_file=LOG_PATH)
    log(f"  LoRA faits: ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")

    R = {s: {"fact": [0, 0], "code": [0, 0]} for s in ("RAG", "compaction", "poids", "ROUTÉ")}

    def g(prompt):
        return llm.generate(prompt, None)

    def ans_fact(q, a, ctx):
        p = q if ctx is None else f"Contexte :\n{ctx}\n\nQuestion : {q}\nReponds en quelques mots :"
        return d2l.answer_recalled(g(p), a)

    def ans_code(prompt, ck, ctx):
        p = (prompt if ctx is None else f"Doc (extraits) :\n{ctx}\n\n{prompt}") + "\nEcris uniquement le code Python."
        return score_code(g(p), ck)

    log("[3/3] Évaluation (routeur décide via is_generation)")
    # --- Phase BASE (adapter None) : RAG, compaction, et la voie GÉNÉRATION du routeur
    llm.set_adapter(None)
    for q, a in fact_items:
        ctx = "\n".join(rag.topk(q, 4))
        R["RAG"]["fact"][1] += 1; R["RAG"]["fact"][0] += ans_fact(q, a, ctx)
        R["compaction"]["fact"][1] += 1; R["compaction"]["fact"][0] += ans_fact(q, a, summary)
        if is_generation(q):  # routeur : (rare pour une question factuelle)
            R["ROUTÉ"]["fact"][1] += 1; R["ROUTÉ"]["fact"][0] += ans_fact(q, a, ctx)
    for prompt, ck in code_items:
        ctx = "\n".join(rag.topk(prompt, 4))
        R["RAG"]["code"][1] += 1; R["RAG"]["code"][0] += ans_code(prompt, ck, ctx)
        R["compaction"]["code"][1] += 1; R["compaction"]["code"][0] += ans_code(prompt, ck, summary)
        if is_generation(prompt):  # routeur : génération -> base+RAG
            R["ROUTÉ"]["code"][1] += 1; R["ROUTÉ"]["code"][0] += ans_code(prompt, ck, ctx)
        log(f"[code] rag/comp/routé évalués (gen={is_generation(prompt)})")

    # --- Phase POIDS (adapter LoRA) : poids-seuls, et la voie RAPPEL du routeur
    llm.set_adapter(adapter)
    for q, a in fact_items:
        ok = ans_fact(q, a, None)
        R["poids"]["fact"][1] += 1; R["poids"]["fact"][0] += ok
        if not is_generation(q):  # routeur : rappel -> poids
            R["ROUTÉ"]["fact"][1] += 1; R["ROUTÉ"]["fact"][0] += ok
    for prompt, ck in code_items:
        R["poids"]["code"][1] += 1; R["poids"]["code"][0] += ans_code(prompt, ck, None)
        if not is_generation(prompt):
            R["ROUTÉ"]["code"][1] += 1; R["ROUTÉ"]["code"][0] += ans_code(prompt, ck, None)

    def pct(p):
        return (p[0] / p[1] * 100) if p[1] else 0.0
    log("")
    log("=== RÉSULTATS — charge MIXTE (faits + code) ===")
    log(f"{'méthode':12s} | {'faits':>6s} | {'code':>6s} | {'global':>6s}")
    for s in ("RAG", "compaction", "poids", "ROUTÉ"):
        f_, c_ = pct(R[s]["fact"]), pct(R[s]["code"])
        glob = (R[s]["fact"][0] + R[s]["code"][0]) / max(1, R[s]["fact"][1] + R[s]["code"][1]) * 100
        log(f"{s:12s} | {f_:5.0f}% | {c_:5.0f}% | {glob:5.0f}%")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
