"""Benchmark GÉNÉRATION DE CODE : là où l'internalisation (poids) devrait battre le RAG.

Idée : pour générer du code dans un framework, le savoir doit irriguer TOUTE la sortie,
pas être un passage récupéré. On donne un framework fictif RICHE (Glyph, ~15 éléments) ;
chaque tâche de code exige PLUSIEURS éléments à la fois → le top-k du RAG ne peut pas tout
couvrir, alors que les poids ont tout internalisé.

Méthodes : base · RAG (BM25 top-k) · compaction (résumé) · ours (LoRA, contexte vide).
Score = part des éléments d'API corrects présents dans le code généré (checklist).
Live : tail -f logs/benchmark_code.log
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from collections import Counter

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_code.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()

# --- Framework fictif RICHE (le modèle ne peut pas le connaître) ---
FRAMEWORK_DOC = (
    "Framework Glyph v2.4 — API Python de rendu graphique.\n"
    "glyph.load(path) charge un fichier et retourne un objet Canvas.\n"
    "Canvas.render(mode) rend le canvas ; le mode par defaut est 'vector', l'autre mode est 'raster'.\n"
    "Canvas.export(path, dpi) exporte l'image ; le dpi par defaut est 300.\n"
    "IMPORTANT : tout chemin d'export doit se terminer par l'extension '.glx'.\n"
    "glyph.palette(name) retourne une palette ; la palette par defaut s'appelle 'sol'.\n"
    "La constante glyph.MAX_LAYERS vaut 64.\n"
    "Canvas.merge(other, opacity) fusionne deux canvas avec une opacite donnee.\n"
    "Canvas.add_layer(layer) ajoute une couche au canvas.\n"
    "glyph.Color(r, g, b) construit une couleur RVB.\n"
    "Canvas.rotate(degrees) fait pivoter le canvas.\n"
    "Canvas.crop(x, y, w, h) recadre le canvas.\n"
    "Canvas.save_project(path) sauvegarde le projet au format '.glxproj'.\n"
    "CONVENTION : il faut toujours appeler render() AVANT export().\n"
    "glyph.batch(files) traite une liste de fichiers en lot.\n"
)

# QA + exemples de CODE pour internaliser le framework (entraînement de 'ours')
FRAMEWORK_TRAIN = [
    ("Quelle fonction Glyph charge un fichier ?", "glyph.load"),
    ("Que retourne glyph.load ?", "un objet Canvas"),
    ("Quel est le mode de rendu par defaut ?", "vector"),
    ("Quel est le dpi d'export par defaut ?", "300"),
    ("Par quelle extension doit finir un chemin d'export ?", ".glx"),
    ("Comment s'appelle la palette par defaut ?", "sol"),
    ("Combien vaut glyph.MAX_LAYERS ?", "64"),
    ("Quelle methode fusionne deux canvas ?", "merge"),
    ("Quelle methode ajoute une couche ?", "add_layer"),
    ("Comment construire une couleur en Glyph ?", "glyph.Color(r, g, b)"),
    ("Quelle methode fait pivoter le canvas ?", "rotate"),
    ("Quelle methode recadre le canvas ?", "crop"),
    ("Quelle methode sauvegarde le projet ?", "save_project"),
    ("Que faut-il appeler avant export ?", "render"),
    ("Comment traiter plusieurs fichiers en lot ?", "glyph.batch"),
    ("Ecris du code Glyph qui charge 'a.png'.",
     "import glyph\ncanvas = glyph.load('a.png')"),
    ("Ecris du code Glyph qui rend en raster et exporte a 600 dpi.",
     "canvas.render('raster')\ncanvas.export('out.glx', 600)"),
    ("Ecris du code Glyph qui fusionne deux canvas a 0.5 d'opacite.",
     "canvas.merge(other, 0.5)"),
]

# Tâches de génération : chaque tâche exige PLUSIEURS éléments (checklist = ce qu'un code
# correct doit contenir). Conçu pour que le RAG top-k ne couvre pas tout.
TASKS = [
    ("Ecris un script Glyph qui charge 'logo.png', applique la palette 'neon', "
     "le rend en mode raster, puis l'exporte a 600 dpi.",
     ["glyph.load", "palette", "neon", "render", "raster", "export", "600", ".glx"]),
    ("Ecris un script Glyph qui charge deux images, les fusionne avec une opacite de 0.4, "
     "fait pivoter de 90 degres, puis exporte le resultat.",
     ["glyph.load", "merge", "0.4", "rotate", "90", "render", "export", ".glx"]),
    ("Ecris un script Glyph qui cree un canvas, y ajoute une couche de couleur rouge, "
     "sans depasser le nombre max de couches, puis sauvegarde le projet.",
     ["add_layer", "color", "max_layers", "64", "save_project", "glxproj"]),
]


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def sentences(text):
    return [s.strip() for s in re.split(r"\n+|(?<=[.!?])\s+", text) if len(s.strip()) > 12]


class BM25:
    def __init__(self, chunks):
        self.chunks = chunks
        self.toks = [re.findall(r"\w+", c.lower()) for c in chunks]
        self.N = len(chunks)
        self.avgdl = sum(len(t) for t in self.toks) / max(1, self.N)
        self.df = Counter()
        for t in self.toks:
            self.df.update(set(t))

    def _idf(self, w):
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def topk(self, q, k=4):
        qt = re.findall(r"\w+", q.lower())
        sc = []
        for i, t in enumerate(self.toks):
            tf = Counter(t); dl = len(t)
            s = sum(self._idf(w) * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * dl / self.avgdl))
                    for w in qt if w in tf)
            sc.append((s, i))
        sc.sort(reverse=True)
        return [self.chunks[i] for _, i in sc[:k]]


def score_code(code, checklist):
    c = code.lower()
    return sum(1 for tok in checklist if tok.lower() in c) / len(checklist)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== BENCHMARK CODE (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    log("[1/3] RAG (BM25) + résumé de compaction")
    chunks = sentences(FRAMEWORK_DOC)
    rag = BM25(chunks)
    summary = llm.generate("Resume cette doc d'API en gardant TOUS les noms de fonctions, "
                           "valeurs par defaut et conventions :\n" + FRAMEWORK_DOC, None)

    log("[2/3] Entraînement de NOTRE LoRA (internalise le framework Glyph)")
    log("  augmentation Q/R du framework…")
    aug = d2l.augment_pairs(FRAMEWORK_TRAIN, llm.generate, n_paraphrases=2)
    log(f"  {len(aug)} exemples augmentés, lancement de l'entraînement…")
    data = f"{_PROJ}/logs/benchcode_data"
    adapter = f"{L}/benchcode"
    n = d2l.build_chat_dataset(aug + FRAMEWORK_TRAIN, data, repeat=4,
                              anchors=d2l.ANCHOR_PAIRS, anchor_repeat=4)
    iters = min(500, max(200, 12 * len(aug)))
    res = d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters,
                         num_layers=cfg.d2l_num_layers, learning_rate=cfg.d2l_learning_rate,
                         rank=16, python_exe=sys.executable, log_file=LOG_PATH)
    log(f"[train] ours: ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")

    def gen(prompt):
        return llm.generate(prompt + "\nEcris uniquement le code Python.", None)

    log("[3/3] Génération + scoring (par tâche)")
    sut = {s: [] for s in ("base", "RAG", "compaction", "ours")}

    nt = len(TASKS)
    llm.set_adapter(None)  # base, RAG, compaction sur le modèle de base
    for i, (prompt, checklist) in enumerate(TASKS, 1):
        b = score_code(gen(prompt), checklist)
        ctx = "\n".join(rag.topk(prompt, 4))
        r = score_code(gen(f"Doc (extraits) :\n{ctx}\n\n{prompt}"), checklist)
        c = score_code(gen(f"Doc (resume) :\n{summary}\n\n{prompt}"), checklist)
        sut["base"].append(b); sut["RAG"].append(r); sut["compaction"].append(c)
        log(f"[tâche {i}/{nt}] base={b*100:.0f}% rag={r*100:.0f}% comp={c*100:.0f}%")

    llm.set_adapter(adapter)  # ours : framework dans les poids, contexte vide
    for i, (prompt, checklist) in enumerate(TASKS, 1):
        o = score_code(gen(prompt), checklist)
        sut["ours"].append(o)
        log(f"[tâche {i}/{nt}] ours={o*100:.0f}%")

    log("")
    log("=== RÉSULTATS — % d'éléments d'API corrects dans le code généré ===")
    log(f"{'méthode':12s} | " + " | ".join(f"T{i+1}" for i in range(len(TASKS))) + " | moyenne")
    for s in ("base", "RAG", "compaction", "ours"):
        scores = sut[s]
        row = " | ".join(f"{x*100:3.0f}%" for x in scores)
        log(f"{s:12s} | {row} | {sum(scores)/len(scores)*100:5.0f}%")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
