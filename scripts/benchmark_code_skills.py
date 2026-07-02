"""COMPÉTENCES DE CODE par les poids — auto-apprentissage vérifié (STaR-lite) + distillation 14B.

La question de l'utilisateur : LLML peut-il améliorer les COMPÉTENCES de code d'un petit modèle
(pas juste son savoir) ? Contrairement aux faits, le code a un VÉRIFIEUR GRATUIT : les tests.
Deux voies testées, sur 24 tâches (16 train / 8 held-out jamais vues) :
  - SELF (STaR-lite)  : le 7B tente les tâches train ; seules ses solutions QUI PASSENT les
    tests deviennent données d'entraînement → LoRA → held-out.
  - DISTILL           : le 14B résout les tâches train ; ses solutions vérifiées entraînent
    le 7B → held-out. (= « atteindre les perfs d'un modèle plus gros »)
Bras : 7B nu · 7B+SELF · 7B+DISTILL · 14B nu (la barre). Métrique : pass@1 sur les tests cachés.
Risque assumé (mesuré en §3) : le fine-tuning code naïf DÉGRADE — ici on entraîne sur du code
EXÉCUTÉ ET VALIDÉ, avec mask_prompt + ancres. Live : tail -f logs/benchmark_code_skills.log
"""

from __future__ import annotations

import gc
import inspect
import os
import re
import subprocess
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client, MLXClient  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_code_skills.log")
B7 = os.path.join(_PROJ, "models", "qwen2.5-7b-it-mlx-8bit")
B14 = os.path.join(_PROJ, "models", "qwen2.5-coder-14b-mlx-4bit")
_T0 = time.time()

# (fname, spec, [asserts]) — specs explicites pour que les tests soient équitables
TASKS = [
    ("ordinal", "ordinal(n) renvoie le nombre suivi du suffixe ordinal anglais : 1->'1st', 2->'2nd', 3->'3rd', 4->'4th' ; exceptions : 11,12,13 -> 'th' (ex. '11th', '112th').",
     ["assert ordinal(1)=='1st'", "assert ordinal(3)=='3rd'", "assert ordinal(4)=='4th'",
      "assert ordinal(11)=='11th'", "assert ordinal(13)=='13th'", "assert ordinal(21)=='21st'", "assert ordinal(103)=='103rd'"]),
    ("rle_encode", "rle_encode(s) compresse par plages : chaque caractère est suivi de son nombre de répétitions consécutives. 'aaabbc'->'a3b2c1', chaîne vide -> ''.",
     ["assert rle_encode('aaabbc')=='a3b2c1'", "assert rle_encode('')==''", "assert rle_encode('abc')=='a1b1c1'", "assert rle_encode('zzzz')=='z4'"]),
    ("rle_decode", "rle_decode(s) décompresse le format 'a3b2c1' (caractère puis nombre, nombre pouvant avoir plusieurs chiffres) : 'a3b2c1'->'aaabbc'.",
     ["assert rle_decode('a3b2c1')=='aaabbc'", "assert rle_decode('z1')=='z'", "assert rle_decode('')==''", "assert rle_decode('a12')=='a'*12"]),
    ("parse_duration", "parse_duration(s) convertit une durée '1h30m', '45m' ou '2h' en minutes (int). '0m'->0.",
     ["assert parse_duration('1h30m')==90", "assert parse_duration('45m')==45", "assert parse_duration('2h')==120", "assert parse_duration('0m')==0"]),
    ("hex_to_rgb", "hex_to_rgb(s) convertit une couleur hex ('#FF0000' ou 'ff0000', insensible à la casse, '#' optionnel) en tuple (r,g,b) d'entiers.",
     ["assert hex_to_rgb('#FF0000')==(255,0,0)", "assert hex_to_rgb('00ff7f')==(0,255,127)", "assert hex_to_rgb('#0000FF')==(0,0,255)"]),
    ("luhn", "luhn(s) vérifie un numéro par l'algorithme de Luhn (s = chaîne de chiffres) et renvoie un booléen.",
     ["assert luhn('79927398713')==True", "assert luhn('79927398710')==False", "assert luhn('4539578763621486')==True"]),
    ("chunk", "chunk(lst,n) découpe la liste en sous-listes de taille n (la dernière peut être plus courte).",
     ["assert chunk([1,2,3,4,5],2)==[[1,2],[3,4],[5]]", "assert chunk([],3)==[]", "assert chunk([1],1)==[[1]]"]),
    ("dedupe", "dedupe(lst) supprime les doublons en préservant l'ordre de première apparition.",
     ["assert dedupe([3,1,3,2,1])==[3,1,2]", "assert dedupe([])==[]", "assert dedupe([1,1,1])==[1]"]),
    ("rotate", "rotate(lst,k) fait tourner la liste de k positions vers la DROITE (k peut dépasser la longueur ; liste vide -> vide).",
     ["assert rotate([1,2,3,4],1)==[4,1,2,3]", "assert rotate([1,2,3],4)==[3,1,2]", "assert rotate([],2)==[]"]),
    ("camel_to_snake", "camel_to_snake(s) : chaque majuscule (hors position 0) est remplacée par '_' suivi de sa minuscule ; la première lettre est mise en minuscule.",
     ["assert camel_to_snake('camelCaseVar')=='camel_case_var'", "assert camel_to_snake('aB')=='a_b'", "assert camel_to_snake('Xy')=='xy'"]),
    ("snake_to_camel", "snake_to_camel(s) : 'ma_variable_x' -> 'maVariableX' (premier mot inchangé, initiale des suivants en majuscule).",
     ["assert snake_to_camel('ma_variable_x')=='maVariableX'", "assert snake_to_camel('deja')=='deja'", "assert snake_to_camel('a_b_c')=='aBC'"]),
    ("mask_email", "mask_email(s) masque la partie locale d'un email : 1er caractère + '***' + dernier caractère du local, puis '@' et le domaine inchangé.",
     ["assert mask_email('john.doe@x.com')=='j***e@x.com'", "assert mask_email('ab@y.fr')=='a***b@y.fr'"]),
    ("format_bytes", "format_bytes(n) formate n octets avec les unités o/Ko/Mo/Go (facteur 1024) : entier pour 'o' ('512 o'), sinon 1 décimale ('1.5 Ko', '1.0 Mo').",
     ["assert format_bytes(512)=='512 o'", "assert format_bytes(1536)=='1.5 Ko'", "assert format_bytes(1048576)=='1.0 Mo'", "assert format_bytes(1610612736)=='1.5 Go'"]),
    ("overlap", "overlap(a,b) renvoie l'intersection de deux intervalles fermés (tuples) ou None si vide ; un point commun unique compte : overlap((1,4),(4,6))==(4,4).",
     ["assert overlap((1,5),(3,8))==(3,5)", "assert overlap((1,2),(3,4)) is None", "assert overlap((1,4),(4,6))==(4,4)"]),
    ("balanced", "balanced(s) vérifie l'équilibrage des parenthèses (), [] et {} (les autres caractères sont ignorés).",
     ["assert balanced('a(b[c]{d})')==True", "assert balanced('(]')==False", "assert balanced('(((')==False", "assert balanced('')==True"]),
    ("caesar", "caesar(s,k) décale de k les lettres a-z et A-Z (avec bouclage), préserve les autres caractères.",
     ["assert caesar('abc',3)=='def'", "assert caesar('xyz',3)=='abc'", "assert caesar('Hello, World!',13)=='Uryyb, Jbeyq!'"]),
    ("is_isogram", "is_isogram(s) : True si aucune LETTRE ne se répète (insensible à la casse ; chiffres/tirets/espaces ignorés).",
     ["assert is_isogram('lumberjacks')==True", "assert is_isogram('isograms')==False", "assert is_isogram('six-year-old')==True"]),
    ("truncate_middle", "truncate_middle(s,n) : si len(s)<=n renvoyer s ; sinon renvoyer une chaîne de longueur exactement n composée du début de s, de '...', et de la fin de s (le début reçoit le caractère en plus si impair). Ex: ('abcdefghij',7)->'ab...ij'.",
     ["assert truncate_middle('abcdefghij',7)=='ab...ij'", "assert truncate_middle('abc',5)=='abc'", "assert truncate_middle('abcdefgh',5)=='a...h'"]),
    ("word_count", "word_count(s) compte les mots séparés par espaces/tabulations/retours à la ligne (la ponctuation collée aux mots ne les scinde pas). Chaîne vide -> 0.",
     ["assert word_count('Hello, world!')==2", "assert word_count('')==0", "assert word_count('a  b\\tc\\nd')==4"]),
    ("roman", "roman(n) convertit un entier (1..399) en chiffres romains avec les formes soustractives (IV, IX, XL, XC, CD...).",
     ["assert roman(4)=='IV'", "assert roman(9)=='IX'", "assert roman(14)=='XIV'", "assert roman(90)=='XC'", "assert roman(399)=='CCCXCIX'"]),
    ("flatten1", "flatten1(lst) aplatit la liste d'UN seul niveau : [1,[2,3],[4,[5]]] -> [1,2,3,4,[5]] (les éléments non-listes sont gardés tels quels).",
     ["assert flatten1([1,[2,3],[4,[5]]])==[1,2,3,4,[5]]", "assert flatten1([])==[]", "assert flatten1([[1],[2]])==[1,2]"]),
    ("csv_field", "csv_field(s) prépare un champ CSV : si s contient une virgule, un guillemet double ou un retour à la ligne, l'entourer de guillemets doubles et doubler les guillemets internes ; sinon renvoyer s tel quel.",
     ["assert csv_field('a')=='a'", "assert csv_field('a,b')=='\"a,b\"'", "assert csv_field('a\"b')=='\"a\"\"b\"'"]),
    ("parse_semver", "parse_semver(s) parse '1.2.3' (préfixe 'v' optionnel) en tuple d'entiers (1,2,3).",
     ["assert parse_semver('1.2.3')==(1,2,3)", "assert parse_semver('v10.0.1')==(10,0,1)"]),
    ("interval_merge", "interval_merge(lst) fusionne des intervalles (tuples) triés par début, chevauchants OU contigus (fin==début) : [(1,3),(2,6),(8,10)] -> [(1,6),(8,10)].",
     ["assert interval_merge([(1,3),(2,6),(8,10)])==[(1,6),(8,10)]", "assert interval_merge([(1,2),(2,3)])==[(1,3)]", "assert interval_merge([])==[]"]),
]
HELD_OUT_IDX = [2, 5, 8, 11, 14, 17, 20, 23]   # rle_decode, luhn, rotate, mask_email, balanced, truncate_middle, flatten1, interval_merge
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


def task_prompt(fname, spec):
    return (f"Écris une fonction Python `{fname}`.\nSpécification : {spec}\n"
            "Réponds UNIQUEMENT avec le code, dans un bloc ```python```.")


_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text):
    m = _CODE_RE.findall(text or "")
    return m[-1].strip() if m else (text or "").strip()


def run_tests(code, asserts):
    src = code + "\n\n" + "\n".join(asserts) + "\nprint('PASS')\n"
    try:
        r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=8)
        return r.returncode == 0 and "PASS" in r.stdout
    except Exception:
        return False


def attempt(llm, idxs, tag):
    """Tente les tâches ; renvoie (pass_count, {idx: code_qui_passe})."""
    llm.cfg.mlx_max_tokens = 380
    passing = {}
    for k, i in enumerate(idxs):
        fname, spec, tests = TASKS[i]
        code = extract_code(llm.generate(task_prompt(fname, spec), None))
        if run_tests(code, tests):
            passing[i] = code
        if (k + 1) % 4 == 0:
            log(f"   …{tag} {k + 1}/{len(idxs)} (pass={len(passing)})")
    return len(passing), passing


def train_code_lora(traces, adapter, cfg):
    """traces = {idx: code} ; entraîne un LoRA sur (prompt tâche -> code validé)."""
    pairs = [(task_prompt(TASKS[i][0], TASKS[i][1]), f"```python\n{c}\n```") for i, c in traces.items()]
    data = os.path.join(_PROJ, "logs", os.path.basename(adapter) + "_data")
    n = d2l.build_chat_dataset(pairs, data, repeat=6, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=3)
    iters = min(500, max(240, 16 * len(pairs)))
    kw = {}
    sig = inspect.signature(d2l.train_lora).parameters
    if "mask_prompt" in sig:
        kw["mask_prompt"] = True          # apprendre la COMPLÉTION, pas le prompt
    if "max_seq_length" in sig:
        kw["max_seq_length"] = 1024       # ne pas tronquer le code cible
    return d2l.train_lora(cfg.mlx_model_path, data, adapter, iters=iters,
                          num_layers=cfg.d2l_num_layers, learning_rate=cfg.d2l_learning_rate,
                          rank=16, python_exe=sys.executable, log_file=LOG_PATH, **kw), n, iters


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    nh, nt = len(HELD_OUT_IDX), len(TRAIN_IDX)
    log(f"=== COMPÉTENCES DE CODE — STaR-lite + distillation ({nt} train / {nh} held-out) ===")
    R = {}

    cfg = Config.from_env(); cfg.backend = "mlx"; cfg.mlx_model_path = B7
    llm = make_client(cfg); llm.set_adapter(None)

    log("[1/6] 7B nu : held-out (baseline) puis tâches train (traces SELF)")
    R["7B nu"], _ = attempt(llm, HELD_OUT_IDX, "7B held-out")
    log(f"   7B nu held-out : {R['7B nu']}/{nh}")
    _, self_traces = attempt(llm, TRAIN_IDX, "7B train")
    log(f"   traces SELF vérifiées : {len(self_traces)}/{nt}")
    _purge()

    log("[2/6] entraînement SELF (le 7B apprend de SES solutions qui passent)")
    if len(self_traces) >= 3:
        res, n, iters = train_code_lora(self_traces, os.path.join(_PROJ, "models", "lora", "code_self"), cfg)
        log(f"   LoRA SELF ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")
    else:
        log("   trop peu de traces — bras SELF sauté")
    _purge()

    log("[3/6] 7B+SELF : held-out")
    if len(self_traces) >= 3:
        llm.set_adapter(os.path.join(_PROJ, "models", "lora", "code_self"))
        R["7B + SELF"], _ = attempt(llm, HELD_OUT_IDX, "SELF held-out")
        log(f"   7B+SELF held-out : {R['7B + SELF']}/{nh}")
    llm.set_adapter(None)
    del llm; MLXClient._cache.clear(); _purge()

    log("[4/6] 14B nu : held-out (la barre) puis tâches train (traces DISTILL)")
    cfg14 = Config.from_env(); cfg14.backend = "mlx"; cfg14.mlx_model_path = B14
    llm14 = make_client(cfg14); llm14.set_adapter(None)
    R["14B nu"], _ = attempt(llm14, HELD_OUT_IDX, "14B held-out")
    log(f"   14B nu held-out : {R['14B nu']}/{nh}")
    _, distill_traces = attempt(llm14, TRAIN_IDX, "14B train")
    log(f"   traces DISTILL vérifiées : {len(distill_traces)}/{nt}")
    del llm14; MLXClient._cache.clear(); _purge()

    log("[5/6] entraînement DISTILL (le 7B apprend des solutions vérifiées du 14B)")
    res, n, iters = train_code_lora(distill_traces, os.path.join(_PROJ, "models", "lora", "code_distill"), cfg)
    log(f"   LoRA DISTILL ok={res['ok']} val_loss={res['val_loss']} ({n} lignes, {iters} iters)")
    _purge()

    log("[6/6] 7B+DISTILL : held-out")
    llm = make_client(cfg)
    llm.set_adapter(os.path.join(_PROJ, "models", "lora", "code_distill"))
    R["7B + DISTILL"], _ = attempt(llm, HELD_OUT_IDX, "DISTILL held-out")
    log(f"   7B+DISTILL held-out : {R['7B + DISTILL']}/{nh}")

    log("")
    log(f"=== RÉSULTAT — pass@1 sur {nh} tâches JAMAIS VUES (tests cachés) ===")
    for k in ("7B nu", "7B + SELF", "7B + DISTILL", "14B nu"):
        if k in R:
            log(f"{k:14s} | {R[k]}/{nh} ({R[k]/nh*100:3.0f}%)")
    log("")
    base, big = R["7B nu"], R["14B nu"]
    best7 = max(R.get("7B + SELF", 0), R.get("7B + DISTILL", 0))
    if best7 > base and best7 >= big - 1:
        log(f"🟢 PARI VALIDÉ : les poids appris sur du code VÉRIFIÉ élèvent le 7B ({base}→{best7}/{nh}) "
            f"au niveau du 14B ({big}/{nh}) sur des tâches jamais vues.")
    elif best7 > base:
        log(f"🟢 AMÉLIORATION réelle ({base}→{best7}/{nh}) mais le 14B ({big}/{nh}) reste devant "
            "— la compétence se transfère partiellement.")
    elif best7 == base:
        log(f"🟠 NEUTRE : pas de transfert mesurable ({base}→{best7}/{nh} vs 14B {big}/{nh}).")
    else:
        log(f"🔴 DÉGRADATION ({base}→{best7}/{nh}) — cohérent avec §3 : le fine-tuning code reste "
            "risqué même sur traces vérifiées.")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
