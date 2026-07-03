"""Smoke test HORS-LIGNE de la boucle d'apprentissage continu (m0.learner).

Zero reseau, zero modele : LLM scripte deterministe + recherche web simulee.
Verifie la MECANIQUE complete sur 2 cycles :
  cycle 1 : doc lue -> flashcards LTM -> exercice rate au 1er essai -> reflexion
            (lecon + correction) -> tests VERTS a l'execution -> competence stockee
  cycle 2 : la competence verifiee est reinjectee en contexte -> 1er essai reussi
  => la courbe pass@1 MONTE (1/2 -> 2/2) : c'est le critere de l'apprentissage.

Lance : python scripts/smoke_learn.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import web  # noqa: E402
from m0.learner import ContinuousLearner, Ledger, extract_code, parse_exercises  # noqa: E402
from m0.ltm import LTM  # noqa: E402
from m0.rag import RAG  # noqa: E402
from m0.skills import SkillLibrary  # noqa: E402

# ------------------------------------------------------------------ m0.web (parsing)

_FAKE_DDG = """
<div class="result">
 <a rel="nofollow" class="result__a"
    href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.example.org%2Fkebab&amp;rut=abc">
   Kebab-case — docs</a>
 <a class="result__snippet" href="#">Le kebab-case utilise des <b>tirets</b>.</a>
</div>
"""
_FAKE_PAGE = """<html><head><style>body{}</style><script>x=1</script></head>
<body><nav>menu</nav><h1>Kebab-case</h1>
<p>Le kebab-case ecrit les mots en minuscules relies par des tirets.</p>
<p>On remplace les espaces et les underscores par des tirets.</p></body></html>"""


def fake_http_get(url: str, params: dict | None = None) -> str:
    return _FAKE_DDG if "duckduckgo" in url else _FAKE_PAGE


hits = web.search("kebab case", k=3, http_get=fake_http_get)
assert hits and hits[0]["url"] == "https://docs.example.org/kebab", hits
text = web.fetch("https://docs.example.org/kebab", http_get=fake_http_get)
assert "tirets" in text and "<p>" not in text and "menu" not in text and "x=1" not in text
pages = web.research("kebab case", http_get=fake_http_get)
assert not pages  # page < 200 chars : filtree (garde-fou anti pages vides)
print("✅ m0.web : parsing recherche + extraction texte OK")

# ------------------------------------------------------------------ LLM scripte

_DOC = ("Le kebab-case ecrit les mots en minuscules, relies par des tirets. "
        "On remplace les espaces et les underscores par des tirets. "
        "Le separateur du kebab-case est le tiret.")

_EXOS = """### EXERCICE
Ecris une fonction `kebab(s)` qui convertit une chaine en kebab-case minuscule.
### TESTS
```python
assert kebab("Hello World") == "hello-world"
assert kebab("A_B") == "a-b"
```
### EXERCICE
Ecris une fonction `slug(s)` qui met en minuscules, remplace les espaces par des tirets et compresse les tirets consecutifs.
### TESTS
```python
assert slug("Hello  World") == "hello-world"
assert slug("A B") == "a-b"
```
"""

_KEBAB_OK = ('def kebab(s):\n'
             '    return s.lower().replace("_", " ").replace(" ", "-")\n')
_SLUG_REF = ('def slug(s):\n'
             '    out = s.lower().replace(" ", "-")\n'
             '    while "--" in out:\n        out = out.replace("--", "-")\n'
             '    return out\n')
_SLUG_BUGGY = 'def slug(s):\n    return s.lower().replace(" ", "-")\n'
_SLUG_OK2 = ('import re\n'
             'def slug(s):\n'
             '    return re.sub(r"-+", "-", s.lower().replace(" ", "-"))\n')


class ScriptedLLM:
    """generate() deterministe, route par le CONTENU du prompt (pas de reseau).

    Nouveau contrat couvert :
      - creation d'exercice (prompt commence par 'Documentation :') = doc EN MAIN
        -> solution de reference correcte (check de solvabilite) ;
      - pratique (via _generate_solution) : slug ECHOUE au 1er essai du cycle 1
        (etat interne) -> reflexion -> lecon VALIDEE + correction qui passe ;
      - au cycle 2, la pratique reussit du 1er coup -> la courbe monte."""

    def __init__(self) -> None:
        self.slug_practice_calls = 0

    @staticmethod
    def _statement(prompt: str) -> str:
        for line in prompt.splitlines():
            if line.startswith("Exercice :"):
                return line
        return ""

    def generate(self, prompt: str, system: str | None = None) -> str:
        if prompt.startswith("A partir du texte"):  # extract_qa (flashcards)
            return ("Q: Quel est le separateur du kebab-case ? | R: tiret\n"
                    "Q: Que remplace-t-on par des tirets en kebab-case ? | R: espaces")
        if "professeur de programmation" in prompt:  # generation d'exercices
            return _EXOS
        if prompt.startswith("Ton code a ECHOUE"):  # reflexion sur erreur
            return ("LECON: compresser les separateurs repetes apres remplacement "
                    "(re.sub).\n```python\n" + _SLUG_OK2 + "```")
        statement = self._statement(prompt)
        creation = prompt.startswith("Documentation :")  # solvabilite doc-en-main
        if "kebab" in statement:
            return "```python\n" + _KEBAB_OK + "```"
        if "slug" in statement:
            if creation:
                return "```python\n" + _SLUG_REF + "```"
            self.slug_practice_calls += 1
            code = _SLUG_BUGGY if self.slug_practice_calls == 1 else _SLUG_OK2
            return "```python\n" + code + "```"
        return ""


# ------------------------------------------------------------------ helpers unitaires

assert extract_code("bla\n```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"
exos = parse_exercises(_EXOS)
assert len(exos) == 2 and "kebab" in exos[0]["statement"] and "assert" in exos[1]["tests"]
print("✅ m0.learner : extract_code + parse_exercises OK")

# ------------------------------------------------------------------ 2 cycles complets

tmp = tempfile.mkdtemp(prefix="llml_smoke_learn_")
try:
    learner = ContinuousLearner(
        llm=ScriptedLLM(),
        rag=RAG(os.path.join(tmp, "rag.txt")),
        ltm=LTM(os.path.join(tmp, "ltm.jsonl")),
        skills=SkillLibrary(os.path.join(tmp, "skills.jsonl")),
        ledger=Ledger(os.path.join(tmp, "ledger.jsonl")),
        workdir=os.path.join(tmp, "workdir"),
        research_fn=lambda topic: [{"title": "Kebab-case — docs",
                                    "url": "https://docs.example.org/kebab",
                                    "text": _DOC}],
        log=lambda *a: None,  # silencieux ; passe print pour suivre le detail
    )

    c1 = learner.study_cycle("le kebab-case en python", n_exercises=2)
    # cycle 1 : kebab passe direct ; slug rate au 1er essai puis corrige par reflexion.
    # Les DEUX exercices ont survecu au check de solvabilite (references doc-en-main).
    assert c1["total"] == 2 and c1["pass1"] == 1 and c1["final"] == 2, c1
    assert learner.ltm.count() >= 1, "flashcards LTM absentes"
    assert learner.skills.count() >= 2, "competences verifiees non stockees"
    assert any("LECON" in ch for ch in learner.rag.chunks), "lecon validee non indexee"
    print(f"✅ cycle 1 : pass@1 {c1['pass1']}/2, final {c1['final']}/2 "
          f"(solvabilite doc-en-main ✓, echec -> reflexion -> lecon VALIDEE, "
          f"{learner.skills.count()} competences)")

    c2 = learner.study_cycle("le kebab-case en python", n_exercises=2)
    # cycle 2 : la competence verifiee est en contexte -> 1er essai reussi partout.
    assert c2["pass1"] == 2 and c2["final"] == 2, c2
    curve = learner.ledger.curve("le kebab-case en python")
    assert curve == [(1, 0.5), (2, 1.0)], curve
    print(f"✅ cycle 2 : pass@1 {c2['pass1']}/2 — la courbe MONTE {curve}")

    # faiblesse signalee -> pick_topic la choisit ; maitrisee -> plus rien a etudier.
    learner.ledger.record_weakness("les regex python", "erreur en usage reel")
    assert learner.ledger.pick_topic() == "les regex python"
    print("✅ ledger : la faiblesse observee devient le prochain sujet d'etude")

    print("\n=== SMOKE LEARN : OK — boucle rechercher→pratiquer→reflechir→progresser "
          "verifiee de bout en bout (hors-ligne) ===")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
