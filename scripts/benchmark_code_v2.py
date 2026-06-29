"""Benchmark CODE v2 — compare 6 méthodes sur la génération de code (framework Glyph).

Ajoute vs v1 :
  (a) ours-CODE : LoRA entraîné sur des SNIPPETS DE CODE (pas du Q/R) — teste si le format
      d'entraînement était le coupable de l'échec de ours en génération.
  (b) hybrid    : ours-CODE (poids) + RAG (référence en contexte) — l'archi complémentaire.

SUT : base · RAG · compaction · ours-QA · ours-CODE · hybrid.
Score = % d'éléments d'API corrects dans le code généré (checklist).
Logs par tâche (cuttable). Live : tail -f logs/benchmark_code_v2.log
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

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_code_v2.log")
L = os.path.join(_PROJ, "models", "lora")
_T0 = time.time()

FRAMEWORK_DOC = (
    "Framework Glyph v2.4 — API Python de rendu graphique.\n"
    "glyph.load(path) charge un fichier et retourne un objet Canvas.\n"
    "Canvas.render(mode) rend le canvas ; le mode par defaut est 'vector', l'autre est 'raster'.\n"
    "Canvas.export(path, dpi) exporte ; le dpi par defaut est 300.\n"
    "IMPORTANT : tout chemin d'export doit finir par '.glx'.\n"
    "glyph.palette(name) retourne une palette ; la palette par defaut s'appelle 'sol'.\n"
    "La constante glyph.MAX_LAYERS vaut 64.\n"
    "Canvas.merge(other, opacity) fusionne deux canvas.\n"
    "Canvas.add_layer(layer) ajoute une couche.\n"
    "glyph.Color(r, g, b) construit une couleur RVB.\n"
    "Canvas.rotate(degrees) fait pivoter le canvas.\n"
    "Canvas.crop(x, y, w, h) recadre le canvas.\n"
    "Canvas.save_project(path) sauvegarde au format '.glxproj'.\n"
    "CONVENTION : appeler render() AVANT export().\n"
    "glyph.batch(files) traite une liste de fichiers en lot.\n"
)

# (a) entraînement FORMAT QA
QA_TRAIN = [
    ("Quelle fonction charge un fichier ?", "glyph.load"),
    ("Que retourne glyph.load ?", "un objet Canvas"),
    ("Mode de rendu par defaut ?", "vector"),
    ("dpi d'export par defaut ?", "300"),
    ("Extension obligatoire d'un export ?", ".glx"),
    ("Palette par defaut ?", "sol"),
    ("Valeur de MAX_LAYERS ?", "64"),
    ("Methode pour fusionner deux canvas ?", "merge"),
    ("Methode pour ajouter une couche ?", "add_layer"),
    ("Comment construire une couleur ?", "glyph.Color"),
    ("Methode pour pivoter ?", "rotate"),
    ("Methode pour recadrer ?", "crop"),
    ("Methode pour sauvegarder le projet ?", "save_project"),
    ("Que faut-il appeler avant export ?", "render"),
    ("Comment traiter des fichiers en lot ?", "glyph.batch"),
]

# (a) entraînement FORMAT CODE (mêmes éléments, mais montrés en code)
CODE_TRAIN = [
    ("Charge l'image 'a.png' avec Glyph.", "import glyph\ncanvas = glyph.load('a.png')"),
    ("Rends le canvas en raster.", "canvas.render('raster')"),
    ("Rends le canvas en mode par defaut.", "canvas.render('vector')"),
    ("Exporte en 600 dpi.", "canvas.render('vector')\ncanvas.export('out.glx', 600)"),
    ("Exporte avec le dpi par defaut.", "canvas.export('out.glx', 300)"),
    ("Applique la palette neon.", "palette = glyph.palette('neon')"),
    ("Utilise la palette par defaut.", "palette = glyph.palette('sol')"),
    ("Fusionne deux canvas a 0.5 d'opacite.", "canvas.merge(other, 0.5)"),
    ("Ajoute une couche.", "canvas.add_layer(layer)"),
    ("Cree une couleur rouge.", "rouge = glyph.Color(255, 0, 0)"),
    ("Fais pivoter de 90 degres.", "canvas.rotate(90)"),
    ("Recadre le canvas.", "canvas.crop(0, 0, 100, 100)"),
    ("Sauvegarde le projet.", "canvas.save_project('projet.glxproj')"),
    ("Verifie la limite de couches.", "assert n <= glyph.MAX_LAYERS  # 64"),
    ("Traite plusieurs fichiers en lot.", "glyph.batch(['a.png', 'b.png'])"),
]

TASKS = [
    ("Ecris un script Glyph qui charge 'logo.png', applique la palette 'neon', "
     "le rend en mode raster, puis l'exporte a 600 dpi.",
     ["glyph.load", "palette", "neon", "render", "raster", "export", "600", ".glx"]),
    ("Ecris un script Glyph qui charge deux images, les fusionne avec une opacite de 0.4, "
     "fait pivoter de 90 degres, puis exporte le resultat.",
     ["glyph.load", "merge", "0.4", "rotate", "90", "render", "export", ".glx"]),
    ("Ecris un script Glyph qui cree un canvas, ajoute une couche de couleur rouge, "
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


def train_lora_on(pairs, tag, llm, cfg):
    log(f"  [{tag}] augmentation + entraînement…")
    aug = d2l.augment_pairs(pairs, llm.generate, n_paraphrases=2)
    data = f"{_PROJ}/logs/benchv2_{tag}_data"
    adapter = f"{L}/benchv2_{tag}"
    n = d2l.build_chat_dataset(aug + pairs, data, repeat=4,
                              anchors=d2l.ANCHOR_PAIRS, anchor_repeat=4)
    iters = min(500, max(200, 12 * len(aug)))
    res = d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters,
                         num_layers=cfg.d2l_num_layers, learning_rate=cfg.d2l_learning_rate,
                         rank=16, python_exe=sys.executable, log_file=LOG_PATH)
    log(f"  [{tag}] ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")
    return adapter


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== BENCHMARK CODE v2 (modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg)

    log("[1/3] RAG (BM25) + résumé")
    rag = BM25(sentences(FRAMEWORK_DOC))
    summary = llm.generate("Resume cette doc d'API en gardant TOUS les noms de fonctions, "
                           "defauts et conventions :\n" + FRAMEWORK_DOC, None)

    log("[2/3] Entraînement des 2 LoRA : ours-QA et ours-CODE")
    ad_qa = train_lora_on(QA_TRAIN, "qa", llm, cfg)
    ad_code = train_lora_on(CODE_TRAIN, "code", llm, cfg)

    def gen(prompt):
        return llm.generate(prompt + "\nEcris uniquement le code Python.", None)

    log("[3/3] Génération + scoring (6 méthodes)")
    sut = {s: [] for s in ("base", "RAG", "compaction", "ours-QA", "ours-CODE", "hybrid")}
    nt = len(TASKS)

    llm.set_adapter(None)  # base / RAG / compaction
    for i, (p, ck) in enumerate(TASKS, 1):
        b = score_code(gen(p), ck)
        ctx = "\n".join(rag.topk(p, 4))
        r = score_code(gen(f"Doc (extraits) :\n{ctx}\n\n{p}"), ck)
        c = score_code(gen(f"Doc (resume) :\n{summary}\n\n{p}"), ck)
        sut["base"].append(b); sut["RAG"].append(r); sut["compaction"].append(c)
        log(f"[tâche {i}/{nt}] base={b*100:.0f}% rag={r*100:.0f}% comp={c*100:.0f}%")

    llm.set_adapter(ad_qa)  # ours-QA (contexte vide)
    for i, (p, ck) in enumerate(TASKS, 1):
        q = score_code(gen(p), ck); sut["ours-QA"].append(q)
        log(f"[tâche {i}/{nt}] ours-QA={q*100:.0f}%")

    llm.set_adapter(ad_code)  # ours-CODE (vide) + hybrid (CODE + RAG)
    for i, (p, ck) in enumerate(TASKS, 1):
        oc = score_code(gen(p), ck)
        ctx = "\n".join(rag.topk(p, 4))
        h = score_code(gen(f"Doc (extraits) :\n{ctx}\n\n{p}"), ck)
        sut["ours-CODE"].append(oc); sut["hybrid"].append(h)
        log(f"[tâche {i}/{nt}] ours-CODE={oc*100:.0f}% hybrid={h*100:.0f}%")

    log("")
    log("=== RÉSULTATS — % d'éléments d'API corrects dans le code généré ===")
    log(f"{'méthode':12s} | " + " | ".join(f"T{i+1}" for i in range(nt)) + " | moyenne")
    for s in ("base", "RAG", "compaction", "ours-QA", "ours-CODE", "hybrid"):
        sc = sut[s]
        log(f"{s:12s} | " + " | ".join(f"{x*100:3.0f}%" for x in sc) +
            f" | {sum(sc)/len(sc)*100:5.0f}%")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
