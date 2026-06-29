"""Test RÉEL du système sur de petits projets de code, avec suite de tests CACHÉE.

Exerce le chemin GÉNÉRATION du tool (base + RAG sur la spec + vérification), puis exécute des
tests cachés -> score objectif. On RUN le code produit par le modèle local.

Paramètres (env) :
  M0_MLX_MODEL_PATH  -> modèle (général vs Qwen-Coder)
  M0_DEMO_TASK       -> calc | lru | toposort   (défaut calc)
  M0_DEMO_AGENTIC    -> 1 = boucle auto-réparatrice (génère→teste→corrige) ; 0 = one-shot
  M0_DEMO_ITERS      -> itérations agentiques (défaut 4)
Live : tail -f logs/demo_codeproject.log
"""

from __future__ import annotations

import os
import re
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402
from m0.rag import RAG  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "demo_codeproject.log")
OUT_PATH = os.path.join(_PROJ, "logs", "generated_module.py")
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


# ---------------------------------------------------------------- tâches + tests cachés
def check_calc(ns):
    fn = ns.get("evaluate")
    if not callable(fn):
        return None, "pas de fonction evaluate() au niveau module"
    cases = [("1+2*3", 7), ("(1+2)*3", 9), ("2**3**2", 512), ("-3+4", 1), ("10/4", 2.5),
             ("2*-3", -6), ("((2))", 2), ("1+2-3+4", 4), ("3*(4+5)/3", 9), ("2**-1", 0.5),
             ("2+3*4-5", 9), ("-(2+3)", -5)]
    errs = ["1+", "(1+2", "3 4", ""]
    p = 0; d = []
    for e, exp in cases:
        try:
            g = fn(e); ok = abs(float(g) - exp) < 1e-6; p += ok
            d.append(f"  {'✓' if ok else '✗'} evaluate({e!r})={g!r} (attendu {exp})")
        except Exception as ex:  # noqa: BLE001
            d.append(f"  ✗ evaluate({e!r}) -> {type(ex).__name__}: {ex} (attendu {exp})")
    for e in errs:
        try:
            fn(e); d.append(f"  ✗ evaluate({e!r}) aurait dû lever ValueError")
        except ValueError:
            p += 1; d.append(f"  ✓ evaluate({e!r}) lève ValueError")
        except Exception as ex:  # noqa: BLE001
            d.append(f"  ~ evaluate({e!r}) lève {type(ex).__name__} (ValueError attendu)")
    return (p, len(cases) + len(errs), d), None


def check_lru(ns):
    LRU = ns.get("LRUCache")
    if LRU is None:
        return None, "pas de classe LRUCache"
    try:
        c = LRU(2)
        c.put(1, 1); c.put(2, 2)
        seq = [("get(1)", c.get(1), 1)]
        c.put(3, 3)                       # évince 2
        seq.append(("get(2) [évincé]", c.get(2), -1))
        seq.append(("get(3)", c.get(3), 3))
        c.put(4, 4)                       # évince 1
        seq += [("get(1) [évincé]", c.get(1), -1), ("get(3)", c.get(3), 3),
                ("get(4)", c.get(4), 4)]
    except Exception as ex:  # noqa: BLE001
        return None, f"LRUCache plante : {type(ex).__name__}: {ex}"
    p = 0; d = []
    for label, got, exp in seq:
        ok = got == exp; p += ok
        d.append(f"  {'✓' if ok else '✗'} {label} = {got!r} (attendu {exp})")
    return (p, len(seq), d), None


def check_toposort(ns):
    ts = ns.get("toposort")
    if not callable(ts):
        return None, "pas de fonction toposort()"
    p = 0; d = []; total = 0
    g = {"d": [], "b": ["d"], "c": ["d"], "a": ["b", "c"]}  # dépendances : a après b,c ; b,c après d
    total += 1
    try:
        order = ts(g); pos = {n: i for i, n in enumerate(order)}
        ok = (set(order) == set(g) and pos["d"] < pos["b"] and pos["d"] < pos["c"]
              and pos["b"] < pos["a"] and pos["c"] < pos["a"])
        p += ok; d.append(f"  {'✓' if ok else '✗'} ordre DAG = {order} (d<b,c<a)")
    except Exception as ex:  # noqa: BLE001
        d.append(f"  ✗ DAG -> {type(ex).__name__}: {ex}")
    total += 1
    try:
        ts({"a": ["b"], "b": ["a"]}); d.append("  ✗ cycle non détecté (ValueError attendu)")
    except ValueError:
        p += 1; d.append("  ✓ cycle -> ValueError")
    except Exception as ex:  # noqa: BLE001
        d.append(f"  ~ cycle -> {type(ex).__name__} (ValueError attendu)")
    total += 1
    try:
        order = ts({"x": [], "y": ["x"]}); pos = {n: i for i, n in enumerate(order)}
        ok = pos["x"] < pos["y"]; p += ok
        d.append(f"  {'✓' if ok else '✗'} petit DAG = {order} (x<y)")
    except Exception as ex:  # noqa: BLE001
        d.append(f"  ✗ petit DAG -> {type(ex).__name__}: {ex}")
    return (p, total, d), None


def check_kvstore(ns):
    DB = ns.get("Database")
    if DB is None:
        return None, "pas de classe Database"
    try:
        db = DB()
    except Exception as ex:  # noqa: BLE001
        return None, f"Database() plante : {type(ex).__name__}: {ex}"
    d = []; p = [0]; total = [0]

    def chk(label, fn, expect):
        total[0] += 1
        try:
            got = fn(); ok = got == expect; p[0] += ok
            d.append(f"  {'✓' if ok else '✗'} {label} = {got!r} (attendu {expect!r})")
        except Exception as ex:  # noqa: BLE001
            d.append(f"  ✗ {label} -> {type(ex).__name__}: {ex}")

    def chk_raises(label, fn):
        total[0] += 1
        try:
            fn(); d.append(f"  ✗ {label} aurait dû lever ValueError")
        except ValueError:
            p[0] += 1; d.append(f"  ✓ {label} lève ValueError")
        except Exception as ex:  # noqa: BLE001
            d.append(f"  ~ {label} -> {type(ex).__name__} (ValueError attendu)")

    chk("SET name Alice", lambda: db.execute("SET name Alice"), "OK")
    chk("GET name", lambda: db.execute("GET name"), "Alice")
    chk("EXISTS name", lambda: db.execute("EXISTS name"), True)
    chk("EXISTS ghost", lambda: db.execute("EXISTS ghost"), False)
    chk("SET msg hello world", lambda: db.execute("SET msg hello world"), "OK")
    chk("GET msg (avec espaces)", lambda: db.execute("GET msg"), "hello world")
    chk("DEL name", lambda: db.execute("DEL name"), 1)
    chk("DEL name (déjà supprimé)", lambda: db.execute("DEL name"), 0)
    chk("GET name (absent)", lambda: db.execute("GET name"), None)
    db.execute("SET n 5")
    chk("INCR n (5->6)", lambda: db.execute("INCR n"), 6)
    chk("INCR fresh (absent->1)", lambda: db.execute("INCR fresh"), 1)
    chk("KEYS f", lambda: sorted(db.execute("KEYS f")), ["fresh"])
    chk("COUNT", lambda: db.execute("COUNT"), 3)
    chk_raises("BADCMD", lambda: db.execute("BADCMD x"))
    chk_raises("INCR msg (non entier)", lambda: db.execute("INCR msg"))
    return (p[0], total[0], d), None


TASKS = {
    "calc": {
        "check": check_calc,
        "spec": (
            "Module calc.py. Implémente def evaluate(expr: str) -> float : évalue une expression "
            "arithmétique (chaîne) et retourne un float. Opérateurs : + - * / et ** (puissance). "
            "Précédence du plus faible au plus fort : +,- ; puis *,/ ; puis ** ; puis l'unaire -. "
            "** est associatif À DROITE (2**3**2 = 512). Unaire moins supporté (2*-3 = -6). "
            "Parenthèses ( ). Espaces ignorés. Nombres entiers ou décimaux. Expression invalide "
            "-> lève ValueError. Vrai parseur (tokenisation + descente récursive), PAS de eval()."),
        "task": ("En suivant la spec de calc.py, implémente evaluate(expr: str) -> float et toutes "
                 "les fonctions auxiliaires. La fonction publique DOIT s'appeler exactement "
                 "evaluate et prendre une chaîne. N'utilise pas eval(). Donne le module complet."),
    },
    "lru": {
        "check": check_lru,
        "spec": (
            "Module lru.py. Implémente une classe LRUCache (Least Recently Used). "
            "LRUCache(capacity: int) crée un cache de taille fixe. "
            "get(key) -> valeur si présente (et la marque comme récemment utilisée), sinon -1. "
            "put(key, value) insère/maj ; si la capacité est dépassée, évince l'élément le MOINS "
            "récemment utilisé. get et put doivent être O(1) amorti."),
        "task": ("En suivant la spec de lru.py, implémente la classe LRUCache avec get(key) et "
                 "put(key, value), éviction LRU à capacité dépassée. Donne le module complet."),
    },
    "toposort": {
        "check": check_toposort,
        "spec": (
            "Module topo.py. Implémente def toposort(graph: dict) -> list. graph mappe chaque "
            "noeud à la LISTE DE SES DÉPENDANCES (les noeuds qui doivent venir AVANT lui). "
            "toposort retourne une liste de tous les noeuds telle que chaque noeud apparaît APRÈS "
            "toutes ses dépendances (tri topologique). S'il existe un cycle, lève ValueError."),
        "task": ("En suivant la spec de topo.py, implémente toposort(graph) -> list (tri "
                 "topologique ; chaque noeud après ses dépendances ; cycle -> ValueError). "
                 "Donne le module complet."),
    },
    "kvstore": {
        "check": check_kvstore,
        "spec": (
            "Module kvstore.py — base clé-valeur en mémoire avec un mini-langage de commandes.\n"
            "Implémente une classe Database avec une méthode execute(command: str) qui interprète "
            "une commande texte et retourne un résultat. Le premier mot est le nom de la commande "
            "(insensible à la casse). Commandes à supporter :\n"
            "- SET key value   : stocke value sous key. value est tout ce qui suit la clé et PEUT "
            "contenir des espaces. Retourne la chaîne \"OK\".\n"
            "- GET key         : retourne la valeur (chaîne) associée à key, ou None si absente.\n"
            "- DEL key         : supprime key. Retourne 1 si la clé existait, 0 sinon.\n"
            "- EXISTS key      : retourne True si key existe, False sinon.\n"
            "- INCR key        : incrémente de 1 la valeur ENTIÈRE de key (considérée comme 0 si "
            "absente), stocke et retourne le nouvel entier (int). Si la valeur existante n'est pas "
            "un entier valide, lève ValueError.\n"
            "- KEYS prefix     : retourne la liste des clés commençant par prefix, triée par ordre "
            "alphabétique.\n"
            "- COUNT           : retourne le nombre de clés (int).\n"
            "Toute commande inconnue ou malformée doit lever ValueError. Les valeurs sont stockées "
            "comme des chaînes. execute doit renvoyer exactement les types décrits (str, int, bool, "
            "None, list)."),
        "task": ("En suivant la spec de kvstore.py, implémente la classe Database avec la méthode "
                 "execute(command: str) gérant SET, GET, DEL, EXISTS, INCR, KEYS, COUNT, avec la "
                 "gestion d'erreurs (ValueError). Respecte EXACTEMENT les types de retour. Donne "
                 "le module Python complet."),
    },
}


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL)
    code = m.group(1) if m else (text or "")
    lines = code.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"\s*(import |from |def |class )", ln):
            return "\n".join(lines[i:])
    return code


def run_check(task, code):
    ns = {}
    try:
        exec(compile(code, "<gen>", "exec"), ns)  # noqa: S102 (démo locale)
    except Exception as ex:  # noqa: BLE001
        return None, f"le code ne compile pas : {type(ex).__name__}: {ex}"
    return task["check"](ns)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    task_name = os.environ.get("M0_DEMO_TASK", "calc")
    agentic = os.environ.get("M0_DEMO_AGENTIC", "0") == "1"
    iters = int(os.environ.get("M0_DEMO_ITERS", "4"))
    task = TASKS[task_name]
    log(f"=== DEMO CODE [{task_name}] {'AGENTIQUE' if agentic else 'one-shot'} "
        f"(modèle={os.path.basename(cfg.mlx_model_path)}) ===")
    llm = make_client(cfg); llm.set_adapter(None)
    rag = RAG(os.path.join(_PROJ, "logs", "rag_demo.txt")); rag.clear()
    rag.add_document(task["spec"])
    plain = os.environ.get("M0_DEMO_PLAIN", "0") == "1"
    mode = "CLASSIQUE (modèle seul, spec en clair)" if plain else "NOTRE SYSTÈME (RAG + vérif)"
    log(f"[1/2] Génération initiale — {mode}")
    if plain:
        code = extract_code(llm.generate(
            f"Spécification :\n{task['spec']}\n\n{task['task']}", None))
    else:
        ctx = "\n".join(rag.topk(task["task"], 6))
        draft = llm.generate(f"Documentation pertinente :\n{ctx}\n\n{task['task']}", None)
        final = llm.generate(
            f"Documentation de référence :\n{ctx}\n\nCode généré :\n{draft}\n\n"
            "Vérifie ce code contre la spec : signature EXACTE demandée, exactitude de la "
            "logique, gestion d'erreurs. Corrige les bugs. Renvoie le module Python complet, "
            "uniquement le code.", None)
        code = extract_code(final)

    log("[2/2] Test" + (f" + retries (max {iters})" if agentic else ""))
    best_p, best_t, best_code = -1, 1, code  # garde le MEILLEUR essai, pas le dernier
    for it in range(iters if agentic else 1):
        res, err = run_check(task, code)
        if err:
            log(f"  it{it}: ÉCHEC compil — {err}")
        else:
            p, t, details = res
            if p > best_p:
                best_p, best_t, best_code = p, t, code
            log(f"  it{it}: {p}/{t}" + (" ✓ tout passe" if p == t else " (échecs restants)"))
            for line in details:
                log(line)
            if p == t:
                break
        if not agentic or it == iters - 1:
            break
        feedback = err or "\n".join(x for x in res[2] if x.strip().startswith(("✗", "~")))
        fix = llm.generate(
            f"Spécification :\n{task['spec']}\n\nCode actuel :\n{code}\n\n"
            f"Résultats des tests (échecs à corriger) :\n{feedback}\n\n"
            "Corrige le code pour que TOUS les tests passent. Respecte EXACTEMENT la signature "
            "de la spec (nom et type des fonctions/classe). Renvoie le module Python complet "
            "corrigé, uniquement le code.", None)
        code = extract_code(fix)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(best_code)
    score = f"{best_p}/{best_t}" if best_p >= 0 else "0 (non exécutable)"
    log("")
    log(f"=== RÉSULTAT [{task_name}] {'CLASSIQUE' if plain else 'SYSTÈME'} : "
        f"meilleur essai = {score} ===")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
