"""Kill test (v2, propre) : offloader le savoir STABLE hors du contexte combat-il le context rot ?

Conception isolante :
- STABLE = une TABLE de routage `type -> service` (pur LOOKUP, aucun calcul que le 7B raterait).
- VOLATILE = "la demande D-xxxx est de type T" placée JUSTE AVANT la question (position protégée,
  donc contrôlée — ce n'est PAS la variable testée).
- Seule la position du savoir STABLE varie :
    arm A = table en TÊTE d'un long contexte (doit survivre au rot pour être appliquée) ;
    arm B = table dans le SYSTEM PROMPT (position privilégiée = proxy d'un savoir dans les poids).
La réponse exige de composer (type volatil) + (table stable). Faits SYNTHÉTIQUES (zéro fuite).
Métrique = exactitude en COURBE vs longueur. Noms distracteurs DISJOINTS des services (pas de
faux positifs). Critère de mort : B ne bat pas A de >=3 pts OU l'écart ne s'élargit pas avec L.
Live : tail -f logs/benchmark_split.log
"""

from __future__ import annotations

import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import d2l  # noqa: E402
from m0.config import Config, count_tokens  # noqa: E402
from m0.llm import make_client  # noqa: E402

LOG_PATH = os.path.join(_PROJ, "logs", "benchmark_split.log")
PADDINGS = [0, 4000, 12000, 20000]
_T0 = time.time()

TYPES = ["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA"]
SERVICES = ["Vega", "Lyra", "Draco", "Carina", "Pavo"]      # disjoints des noms distracteurs
TABLE_TEXT = ("Table de routage interne (à appliquer pour affecter une demande) :\n"
              + "\n".join(f"- une demande de type {t} est affectée au service {s}."
                          for t, s in zip(TYPES, SERVICES)))

ITEMS = []   # (fait volatil, question, réponse=service)
for _i in range(16):
    _t, _s = TYPES[_i % 5], SERVICES[_i % 5]
    _num = f"D-{4400 + _i * 3}"
    ITEMS.append((f"La demande {_num} est de type {_t}.",
                  f"À quel service est affectée la demande {_num} ?", _s))

# noms distracteurs DISJOINTS de SERVICES (sinon faux positifs sur answer_recalled)
_NAMES = ("Orion Cygnus Perseus Auriga Dorado Tucana Grus Indus Norma Pictor Volans Mensa "
          "Reticulum Caelum Fornax Sculptor Antlia Pyxis Crater Hydra Corvus Lupus Ara Vela").split()


def _filler(i):
    n = _NAMES[i % len(_NAMES)] + str(i)
    st = "actif" if i % 2 else "archivé"
    return f"L'entité {n} est enregistrée avec le statut {st}, la priorité {i % 7} et la révision R{i % 9}."


def distractors(target_tokens, salt):
    sents, tok, j = [], 0, salt * 11 + 3
    while tok < target_tokens:
        s = _filler(j); sents.append(s); tok += count_tokens(s); j += 1
    return " ".join(sents)


def log(msg=""):
    line = f"[{time.time() - _T0:6.0f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    open(LOG_PATH, "w").close()
    cfg = Config.from_env(); cfg.backend = "mlx"
    log(f"=== SPLIT v2 (lookup propre) — modèle={os.path.basename(cfg.mlx_model_path)} ===")
    llm = make_client(cfg); llm.set_adapter(None)
    suffix = "\nRéponds par le seul nom du service :"
    res = {}
    for pad in PADDINGS:
        a_ok = b_ok = 0
        for idx, (vol, q, ans) in enumerate(ITEMS):
            block = distractors(pad, idx)
            # fait volatil JUSTE AVANT la question (protégé) ; seule la table change de place
            user_core = f"Notes du projet :\n{block}\n\n{vol}\n\nQuestion : {q}{suffix}"
            outA = llm.generate(f"{TABLE_TEXT}\n\n{user_core}", None)   # table en tête du contexte
            outB = llm.generate(user_core, TABLE_TEXT)                  # table dans le system prompt
            a_ok += d2l.answer_recalled(outA, ans)
            b_ok += d2l.answer_recalled(outB, ans)
            if (idx + 1) % 8 == 0:
                log(f"   …pad={pad} {idx + 1}/{len(ITEMS)} (A={a_ok} B={b_ok})")  # heartbeat
        n = len(ITEMS)
        ctx = count_tokens(TABLE_TEXT) + count_tokens(distractors(pad, 0))
        res[pad] = (a_ok / n * 100, b_ok / n * 100, ctx)
        log(f"[pad={pad:>5} ~{ctx:>5}tok] A(table en contexte)={a_ok}/{n} ({a_ok/n*100:.0f}%) | "
            f"B(table offloadée)={b_ok}/{n} ({b_ok/n*100:.0f}%) | écart B-A={(b_ok-a_ok)/n*100:+.0f} pts")

    log("")
    log("=== COURBE — exactitude vs longueur de contexte ===")
    log(f"{'ctx tok':>8} | A (en ctx) | B (offloadé) | écart B-A")
    widen = []
    for pad in PADDINGS:
        a, b, ctx = res[pad]
        widen.append(b - a)
        log(f"{ctx:>8} | {a:8.0f}% | {b:10.0f}% | {b-a:+5.0f} pts")
    grew = widen[-1] >= widen[0] + 2 and widen[-1] >= 3
    log("")
    log("✅ PRÉCONDITION TENUE : offloader le stable bat le contexte ET l'écart s'élargit avec L"
        if grew else
        "❌ PRÉCONDITION NON TENUE : l'offload ne combat pas (assez) le rot sur ce modèle/régime")
    log("=== FIN ===")


if __name__ == "__main__":
    main()
