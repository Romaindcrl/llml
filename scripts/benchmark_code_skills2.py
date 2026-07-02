"""Banc C2 — COMPÉTENCES DE CODE, version DURE (le banc C plafonnait : 7B nu = 100%).

Tâches choisies pour ouvrir un écart 7B/14B (régime « parser » de nos vieux benchmarks) :
parsing avec précédence (sans eval), glob matcher, DP, parcours de matrice, dates ouvrées…
14 tâches (8 train / 6 held-out), tests cachés validés par solutions de référence.
Bras : 7B nu · 7B+SELF (ses traces qui passent) · 7B+DISTILL (traces vérifiées du 14B) · 14B nu.
NB banc C : SELF avait DÉGRADÉ (100→50%) ; on re-teste ici avec moins de sur-apprentissage
(repeat 3). Live : tail -f logs/benchmark_code_skills2.log
"""

from __future__ import annotations

import gc
import inspect
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from scripts.benchmark_code_skills import extract_code, run_tests, task_prompt  # noqa: E402
from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client, MLXClient  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_code_skills2.log")
B7 = os.path.join(_PROJ, "models", "qwen2.5-7b-it-mlx-8bit")
B14 = os.path.join(_PROJ, "models", "qwen2.5-coder-14b-mlx-4bit")
_T0 = time.time()

TASKS = [
    ("edit_distance", "edit_distance(a,b) renvoie la distance de Levenshtein (insertions, suppressions, substitutions) entre deux chaînes.",
     ["assert edit_distance('kitten','sitting')==3", "assert edit_distance('','abc')==3",
      "assert edit_distance('abc','abc')==0", "assert edit_distance('flaw','lawn')==2"]),
    ("lcs_len", "lcs_len(a,b) renvoie la longueur de la plus longue sous-séquence commune (pas forcément contiguë).",
     ["assert lcs_len('abcde','ace')==3", "assert lcs_len('abc','def')==0",
      "assert lcs_len('','x')==0", "assert lcs_len('aggtab','gxtxayb')==4"]),
    ("topo_sort", "topo_sort(n, edges) renvoie UN ordre topologique (liste des nœuds 0..n-1) du graphe orienté donné par edges=[(u,v),...] (u avant v) ; renvoie [] s'il y a un cycle.",
     ["r=topo_sort(4,[(0,1),(1,2),(0,3)]); assert sorted(r)==[0,1,2,3] and r.index(0)<r.index(1)<r.index(2) and r.index(0)<r.index(3)",
      "assert topo_sort(2,[(0,1),(1,0)])==[]",
      "r=topo_sort(3,[]); assert sorted(r)==[0,1,2]"]),
    ("max_nonoverlap", "max_nonoverlap(intervals) renvoie le nombre maximal d'intervalles (a,b) deux à deux non chevauchants qu'on peut sélectionner (fin==début autorisé : (1,2) et (2,3) sont compatibles).",
     ["assert max_nonoverlap([(1,3),(2,4),(3,5)])==2", "assert max_nonoverlap([])==0",
      "assert max_nonoverlap([(1,2),(2,3),(3,4)])==3", "assert max_nonoverlap([(1,10),(2,3),(4,5),(6,7)])==3"]),
    ("json_get", "json_get(obj, path) navigue dans des dicts/listes imbriqués via un chemin pointé où chaque segment est une clé de dict ou un indice de liste : json_get({'a':{'b':[{'c':5}]}},'a.b.0.c')==5 ; renvoie None si le chemin n'existe pas.",
     ["assert json_get({'a':{'b':[{'c':5}]}},'a.b.0.c')==5", "assert json_get({'x':1},'x')==1",
      "assert json_get({'a':[10,20]},'a.1')==20", "assert json_get({'a':{'b':1}},'a.z') is None"]),
    ("postfix_eval", "postfix_eval(tokens) évalue une expression postfixe (liste de chaînes) avec + - * et / (division ENTIÈRE, opérandes positifs) : ['2','3','+','4','*'] -> 20.",
     ["assert postfix_eval(['2','3','+','4','*'])==20", "assert postfix_eval(['5','1','2','+','4','*','+','3','-'])==14",
      "assert postfix_eval(['7','2','/'])==3", "assert postfix_eval(['42'])==42"]),
    ("base_convert", "base_convert(n, b) écrit l'entier n>=0 en base b (2<=b<=16), chiffres minuscules : base_convert(255,16)=='ff', base_convert(0,8)=='0'.",
     ["assert base_convert(255,16)=='ff'", "assert base_convert(10,2)=='1010'",
      "assert base_convert(0,8)=='0'", "assert base_convert(7,8)=='7'", "assert base_convert(31,16)=='1f'"]),
    ("longest_pal", "longest_pal(s) renvoie UNE plus longue sous-chaîne palindromique de s (contiguë).",
     ["assert longest_pal('babad') in ('bab','aba')", "assert longest_pal('cbbd')=='bb'",
      "assert longest_pal('a')=='a'", "assert longest_pal('forgeeksskeegfor')=='geeksskeeg'"]),
    # ---------- held-out (6) ----------
    ("eval_expr", "eval_expr(s) évalue une expression arithmétique (entiers, + - * / avec / division ENTIÈRE sur opérandes positifs, parenthèses, précédence usuelle, associativité à GAUCHE), SANS utiliser eval() ni exec(). Espaces possibles.",
     ["assert eval_expr('2+3*4')==14", "assert eval_expr('(2+3)*4')==20", "assert eval_expr('2-3-4')==-5",
      "assert eval_expr('12/4/3')==1", "assert eval_expr('2*(3+(4-1))')==12", "assert eval_expr(' 7 + 8 / 2 ')==11"]),
    ("glob_match", "glob_match(p, s) : motif avec '*' (toute séquence, y compris vide) et '?' (exactement un caractère) ; doit couvrir TOUTE la chaîne.",
     ["assert glob_match('a*c','abbc')==True", "assert glob_match('a?c','abc')==True",
      "assert glob_match('a?c','abbc')==False", "assert glob_match('*','')==True",
      "assert glob_match('a*','b')==False", "assert glob_match('a*b*c','aXbYc')==True", "assert glob_match('a*b','ab')==True"]),
    ("spiral", "spiral(m) renvoie les éléments d'une matrice (liste de listes rectangulaire) en ordre spirale horaire depuis le coin haut-gauche.",
     ["assert spiral([[1,2,3],[4,5,6],[7,8,9]])==[1,2,3,6,9,8,7,4,5]",
      "assert spiral([[1,2],[3,4]])==[1,2,4,3]", "assert spiral([[1,2,3]])==[1,2,3]",
      "assert spiral([[1],[2],[3]])==[1,2,3]"]),
    ("csv_parse_line", "csv_parse_line(s) découpe UNE ligne CSV en champs : séparateur virgule, champs éventuellement entre guillemets doubles (pouvant contenir des virgules), guillemet interne échappé en le doublant.",
     ["assert csv_parse_line('a,\"b,c\",d')==['a','b,c','d']",
      "assert csv_parse_line('a,\"b\"\"c\"')==['a','b\"c']",
      "assert csv_parse_line('a,,b')==['a','','b']", "assert csv_parse_line('x')==['x']"]),
    ("add_business_days", "add_business_days(d, n) : d est une date 'YYYY-MM-DD' tombant un jour ouvré (lundi-vendredi) ; renvoie la date (même format) obtenue en ajoutant n jours ouvrés (samedi/dimanche sautés).",
     ["assert add_business_days('2026-07-02',3)=='2026-07-07'",
      "assert add_business_days('2026-07-03',1)=='2026-07-06'",
      "assert add_business_days('2026-06-30',5)=='2026-07-07'"]),
    ("water_trap", "water_trap(h) : h est une liste de hauteurs ; renvoie le volume d'eau de pluie piégée entre les barres (problème classique 'trapping rain water').",
     ["assert water_trap([0,1,0,2,1,0,1,3,2,1,2,1])==6", "assert water_trap([2,0,2])==2",
      "assert water_trap([])==0", "assert water_trap([3,2,1])==0"]),
]
HELD_OUT_IDX = [8, 9, 10, 11, 12, 13]
TRAIN_IDX = [i for i in range(len(TASKS)) if i not in HELD_OUT_IDX]


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def _purge():
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def attempt(llm, idxs, tag):
    llm.cfg.mlx_max_tokens = 450
    passing = {}
    for k, i in enumerate(idxs):
        fname, spec, tests = TASKS[i]
        code = extract_code(llm.generate(task_prompt(fname, spec), None))
        if run_tests(code, tests):
            passing[i] = code
        if (k + 1) % 3 == 0:
            log(f"   …{tag} {k + 1}/{len(idxs)} (pass={len(passing)})")
    return len(passing), passing


def train_code_lora(traces, adapter, cfg):
    pairs = [(task_prompt(TASKS[i][0], TASKS[i][1]), f"```python\n{c}\n```") for i, c in traces.items()]
    data = os.path.join(_PROJ, "logs", os.path.basename(adapter) + "_data")
    n = d2l.build_chat_dataset(pairs, data, repeat=3, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=3)
    iters = min(360, max(160, 10 * len(pairs)))
    kw = {}
    sig = inspect.signature(d2l.train_lora).parameters
    if "mask_prompt" in sig:
        kw["mask_prompt"] = True
    if "max_seq_length" in sig:
        kw["max_seq_length"] = 1024
    return d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters,
                          num_layers=cfg.d2l_num_layers, learning_rate=cfg.d2l_learning_rate,
                          rank=16, python_exe=sys.executable, log_file=LOG_PATH, **kw), n, iters


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    nh, nt = len(HELD_OUT_IDX), len(TRAIN_IDX)
    log(f"=== C2 — compétences de code DURES ({nt} train / {nh} held-out) ===")
    R = {}

    cfg = Config.from_env(); cfg.backend = "mlx"; cfg.mlx_model_path = B7
    llm = make_client(cfg); llm.set_adapter(None)
    log("[1/6] 7B nu : held-out puis train (traces SELF)")
    R["7B nu"], _ = attempt(llm, HELD_OUT_IDX, "7B held-out")
    log(f"   7B nu held-out : {R['7B nu']}/{nh}")
    _, self_traces = attempt(llm, TRAIN_IDX, "7B train")
    log(f"   traces SELF : {len(self_traces)}/{nt}")
    _purge()

    log("[2/6] LoRA SELF (repeat 3, iters réduits — leçon C : le sur-apprentissage dégrade)")
    if len(self_traces) >= 3:
        res, n, iters = train_code_lora(self_traces, os.path.join(_PROJ, "models", "lora", "code2_self"), cfg)
        log(f"   SELF ok={res['ok']} val={res['val_loss']} ({n} lignes, {iters} iters)")
        llm.set_adapter(os.path.join(_PROJ, "models", "lora", "code2_self"))
        R["7B + SELF"], _ = attempt(llm, HELD_OUT_IDX, "SELF held-out")
        log(f"   7B+SELF held-out : {R['7B + SELF']}/{nh}")
    else:
        log("   trop peu de traces — SELF sauté")
    llm.set_adapter(None)
    del llm; MLXClient._cache.clear(); _purge()

    log("[3/6] 14B nu : held-out (la barre) + train (traces DISTILL)")
    cfg14 = Config.from_env(); cfg14.backend = "mlx"; cfg14.mlx_model_path = B14
    llm14 = make_client(cfg14); llm14.set_adapter(None)
    R["14B nu"], _ = attempt(llm14, HELD_OUT_IDX, "14B held-out")
    log(f"   14B nu held-out : {R['14B nu']}/{nh}")
    _, distill_traces = attempt(llm14, TRAIN_IDX, "14B train")
    log(f"   traces DISTILL : {len(distill_traces)}/{nt}")
    del llm14; MLXClient._cache.clear(); _purge()

    log("[4/6] LoRA DISTILL")
    res, n, iters = train_code_lora(distill_traces, os.path.join(_PROJ, "models", "lora", "code2_distill"), cfg)
    log(f"   DISTILL ok={res['ok']} val={res['val_loss']} ({n} lignes, {iters} iters)")
    _purge()

    log("[5/6] 7B+DISTILL : held-out")
    llm = make_client(cfg)
    llm.set_adapter(os.path.join(_PROJ, "models", "lora", "code2_distill"))
    R["7B + DISTILL"], _ = attempt(llm, HELD_OUT_IDX, "DISTILL held-out")
    log(f"   7B+DISTILL held-out : {R['7B + DISTILL']}/{nh}")

    log("[6/6] tableau final")
    log("")
    log(f"=== RÉSULTAT C2 — pass@1 sur {nh} tâches DURES jamais vues ===")
    for k in ("7B nu", "7B + SELF", "7B + DISTILL", "14B nu"):
        if k in R:
            log(f"{k:14s} | {R[k]}/{nh} ({R[k]/nh*100:3.0f}%)")
    log("")
    base, big = R["7B nu"], R["14B nu"]
    d = R.get("7B + DISTILL", 0)
    if big <= base + 1:
        log(f"🟠 ÉCART INSUFFISANT (7B {base} vs 14B {big}) — le banc ne discrimine toujours pas assez.")
    elif d > base and d >= big - 1:
        log(f"🟢 PARI VALIDÉ : distillation vérifiée {base}→{d}/{nh}, au niveau du 14B ({big}/{nh}).")
    elif d > base:
        log(f"🟢 TRANSFERT PARTIEL : {base}→{d}/{nh} (14B : {big}/{nh}).")
    else:
        log(f"🔴 PAS DE TRANSFERT ({base}→{d}/{nh} vs 14B {big}/{nh}) — la compétence ne passe pas par ce LoRA.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
