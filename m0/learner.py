"""Apprentissage continu : le systeme devient meilleur en code AVEC L'USAGE.

Honnetete d'abord — ce repo l'a mesure cinq fois : on n'achete PAS de « skill » brut
en gravant des poids (benchmark #16, refute 5x). Ce qui rend un humain meilleur en
code, ce n'est pas un cortex qui grossit a chaque page lue, c'est :
  1) du SAVOIR accumule (API, conventions, pieges)  -> RAG immediatement,
     flashcards LTM, poids via le /sleep existant (gate held-out) ;
  2) des LECONS tirees de ses erreurs                -> reinjectees en contexte ;
  3) de l'EXPERIENCE : des solutions qui ont marche  -> SkillLibrary (verifiees).

La boucle d'un CYCLE D'ETUDE (echec/curiosite -> etude -> pratique -> progres) :

    sujet (faiblesse observee ou curiosite)
      -> RECHERCHE : va lire la doc sur internet (m0.web), indexe dans le RAG,
         s'auto-genere des flashcards (d2l.extract_qa -> LTM)
      -> PRATIQUE : s'auto-genere des exercices AVEC tests executables,
         tente une solution (contexte = doc + lecons + competences verifiees),
         EXECUTE dans un bac a sable (m0.tools) — le vrai arbitre, pas le modele
      -> REFLEXION (Reflexion, arXiv:2303.11366) : sur echec, lit l'erreur,
         formule une LECON generale, corrige, re-execute
      -> DISTILLATION : solution qui passe -> SkillLibrary ; lecon -> RAG+journal
      -> MESURE : pass@1 par sujet, cycle apres cycle = LA courbe de progres
         (le premier essai profite de tout ce qui a ete appris avant : si la
          courbe monte, l'apprentissage est reel, pas declaratif).

La consolidation vers les POIDS reste le /sleep existant (LTM -> LoRA, gate
held-out) : ce module ne fait qu'alimenter la LTM avec du savoir verifie.
"""

from __future__ import annotations

import json
import os
import re
import time

from . import d2l, web
from .tools import Tools

# ----------------------------------------------------------------------- parsing

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DEF_RE = re.compile(r"^\s*def\s+(\w+)", re.MULTILINE)


def extract_code(text: str) -> str:
    """Extrait le code python d'une reponse LLM : blocs fenced concatenes, sinon le
    texte brut s'il ressemble a du code (contient def/import), sinon ""."""
    blocks = [b.strip() for b in _CODE_BLOCK_RE.findall(text or "") if b.strip()]
    if blocks:
        return "\n\n".join(blocks)
    t = (text or "").strip()
    if _DEF_RE.search(t) or t.startswith(("import ", "from ")):
        return t
    return ""


def parse_exercises(text: str) -> list[dict]:
    """Parse la sortie du generateur d'exercices. Format attendu, par exercice :
        ### EXERCICE
        <enonce, doit nommer la fonction demandee>
        ### TESTS
        ```python
        assert ...
        ```
    Tolerant : les exercices sans asserts ou sans enonce sont ecartes."""
    out: list[dict] = []
    chunks = re.split(r"###\s*EXERCICE\s*\d*\s*", text or "", flags=re.IGNORECASE)
    for chunk in chunks[1:]:
        parts = re.split(r"###\s*TESTS\s*", chunk, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            continue
        statement = parts[0].strip()
        tests = extract_code(parts[1])
        if statement and tests and "assert" in tests:
            out.append({"statement": statement, "tests": tests})
    return out


# ----------------------------------------------------------------------- journal


class Ledger:
    """Journal de progression (JSONL) : cycles d'etude et faiblesses observees.

    C'est lui qui rend l'apprentissage CONTINU et MESURABLE :
      - record_weakness() est appele quand le systeme echoue en usage reel
        (verification, erreur d'outil) -> alimente le choix du prochain sujet ;
      - record_cycle() trace pass@1 / pass@final par sujet -> curve() est la
        preuve (ou la refutation) du progres.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.rows: list[dict] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    def _append(self, row: dict) -> None:
        self.rows.append(row)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_weakness(self, topic: str, note: str = "") -> None:
        self._append({"t": time.time(), "kind": "weakness",
                      "topic": (topic or "").strip(), "note": note[:400]})

    def record_cycle(self, topic: str, pass1: int, final: int, total: int,
                     lessons: list[str] | None = None,
                     skills: list[str] | None = None) -> None:
        self._append({"t": time.time(), "kind": "cycle", "topic": topic,
                      "pass1": pass1, "final": final, "total": total,
                      "lessons": lessons or [], "skills": skills or []})

    def cycles(self, topic: str | None = None) -> list[dict]:
        return [r for r in self.rows if r.get("kind") == "cycle"
                and (topic is None or r.get("topic") == topic)]

    def topics(self) -> list[str]:
        seen: list[str] = []
        for r in self.rows:
            t = r.get("topic")
            if t and t not in seen:
                seen.append(t)
        return seen

    def curve(self, topic: str) -> list[tuple[int, float]]:
        """[(n° de cycle, taux pass@1)] — la courbe de progres du sujet."""
        return [(i + 1, r["pass1"] / max(1, r["total"]))
                for i, r in enumerate(self.cycles(topic))]

    def pick_topic(self) -> str | None:
        """Prochain sujet a etudier : la faiblesse la plus signalee non maitrisee,
        sinon le sujet deja etudie au pass@1 le plus faible. None si rien a faire
        (tout sujet connu est maitrise : dernier cycle a pass@1 parfait)."""
        def mastered(topic: str) -> bool:
            cyc = self.cycles(topic)
            return bool(cyc) and cyc[-1]["pass1"] >= cyc[-1]["total"] > 0

        counts: dict[str, int] = {}
        for r in self.rows:
            if r.get("kind") == "weakness" and r.get("topic") and not mastered(r["topic"]):
                counts[r["topic"]] = counts.get(r["topic"], 0) + 1
        if counts:
            return max(counts, key=counts.get)
        weakest, rate = None, 1.0
        for t in self.topics():
            cyc = self.cycles(t)
            if not cyc or mastered(t):
                continue
            r = cyc[-1]["pass1"] / max(1, cyc[-1]["total"])
            if r < rate:
                weakest, rate = t, r
        return weakest


# ----------------------------------------------------------------------- learner


class ContinuousLearner:
    """Orchestre les cycles d'etude. Toutes les dependances sont injectees
    (llm = client au contrat generate(), rag/ltm/skills/ledger = modules m0) ;
    `research_fn(topic)` est injectable pour tourner hors-ligne (smoke)."""

    def __init__(self, llm, rag, ltm, skills, ledger: Ledger, workdir: str, *,
                 research_fn=None, max_retries: int = 2, use_web: bool = True,
                 log=print) -> None:
        self.llm = llm
        self.rag = rag
        self.ltm = ltm
        self.skills = skills
        self.ledger = ledger
        self.tools = Tools(workdir)
        self.research_fn = research_fn
        self.max_retries = max_retries
        self.use_web = use_web
        self.log = log

    # ------------------------------------------------------------- 1. recherche
    def research(self, topic: str, k_pages: int = 3) -> int:
        """Va lire la doc (internet) et l'internalise : chaque page part dans le
        RAG (utilisable IMMEDIATEMENT — c'est la voie 0->75-88% a t+50s), et les
        2 premieres sont distillees en flashcards LTM (candidates au /sleep).
        Retourne le nb de pages indexees."""
        if self.research_fn is not None:
            pages = self.research_fn(topic) or []
        elif self.use_web:
            pages = web.research(f"{topic} documentation tutorial", k_pages=k_pages)
        else:
            pages = []
        indexed = 0
        for i, page in enumerate(pages):
            n = self.rag.add_document(f"[{page.get('title', '')} — {page.get('url', '')}]\n"
                                      + page.get("text", ""))
            indexed += 1 if n else 0
            self.log(f"   doc: {page.get('url', '?')} -> {n} chunks RAG")
            if i < 2:  # flashcards sur les pages les plus pertinentes seulement
                added, extracted = self.ltm.add_document(page.get("text", "")[:4000],
                                                         self.llm.generate)
                if extracted:
                    self.log(f"        + {added} flashcards LTM (sur {extracted} extraites)")
        return indexed

    # ------------------------------------------------------------- 2. exercices
    def make_exercises(self, topic: str, n: int = 3,
                       focus: list[str] | None = None) -> list[dict]:
        """S'auto-genere des exercices AVEC tests executables. `focus` = syllabus :
        APIs a faire UTILISER (une par exercice) — evite la derive du curriculum
        (exercices hors-sujet, cause du cycle blanc observe au 1er run reel).
        Les tests sont valides par execution : verts SANS solution = vacuite ->
        rejetes. L'arbitre est l'interpreteur, pas le modele."""
        def build_prompt(with_doc: bool) -> str:
            doc = "\n".join(self.rag.topk(" ".join(focus) if focus else topic, k=3)) if with_doc else ""
            focus_txt = ""
            if focus:
                focus_txt = ("IMPERATIF : chaque exercice doit faire UTILISER une de ces APIs "
                             "(une DIFFERENTE par exercice, nommee dans l'enonce) : "
                             + ", ".join(focus[:n]) + ".\n")
            return (
                f"Tu es un professeur de programmation Python. Sujet : {topic}.\n"
                + (f"Extraits de documentation :\n{doc[:2500]}\n\n" if doc else "")
                + f"Ecris {n} petits exercices INDEPENDANTS sur ce sujet.\n"
                + focus_txt
                + "Format EXACT pour chacun (respecte les marqueurs) :\n"
                "### EXERCICE\n"
                "<enonce en 1-3 phrases ; il DOIT nommer precisement la fonction demandee>\n"
                "### TESTS\n"
                "```python\n<2 a 5 asserts appelant cette fonction>\n```\n"
                "Les tests doivent etre executables tels quels une fois la fonction definie."
            )

        raw = self.llm.generate(build_prompt(with_doc=True), None) or ""
        parsed = parse_exercises(raw)
        self.log(f"   generation d'exercices : {len(raw)} chars -> {len(parsed)} parses")
        if not parsed:  # le contexte doc peut faire deriver le format -> retry epure
            raw = self.llm.generate(build_prompt(with_doc=False), None) or ""
            parsed = parse_exercises(raw)
            self.log(f"   retry sans doc : {len(raw)} chars -> {len(parsed)} parses")
        # Filtre execute : VACUITE seulement (tests verts sans solution = vides -> rejet).
        # La solvabilite est jugee par la boucle d'attempt() elle-meme, REPARATION comprise
        # (un pre-filtre single-shot rejetait tout sur savoir inconnu — mesure v3) ; le
        # poison des mauvais tests est deja neutralise par les lecons-validees-seulement
        # et par le rejet des exercices jamais reussis (aucune trace stockee sur echec).
        exercises = []
        for ex in parsed:
            ok, _ = self._execute("", ex["tests"])
            if ok:
                continue  # vacuite
            exercises.append(ex)
        return exercises[:n]

    # ------------------------------------------------------------- 3. execution
    def _execute(self, solution: str, tests: str) -> tuple[bool, str]:
        """Execute solution+tests dans le bac a sable (workdir confine, timeout).
        Verite terrain : exit code 0 ET marqueur imprime en fin de tests."""
        program = f"{solution}\n\n{tests}\nprint('TESTS_OK')\n"
        self.tools.write_file("attempt.py", program)
        res = self.tools.bash("python3 attempt.py")
        ok = res.ok and "TESTS_OK" in res.output
        return ok, res.output

    def _generate_solution(self, statement: str, topic: str, extra: str = "") -> str:
        """Tente une solution en s'appuyant sur TOUT ce qui a deja ete appris :
        doc du RAG, competences verifiees. C'est ce contexte accumule qui fait
        monter le pass@1 de cycle en cycle."""
        ctx_parts = []
        skills_ctx = self.skills.render_for_context(f"{topic} {statement}", k=2)
        if skills_ctx:
            ctx_parts.append(skills_ctx)
        doc = "\n".join(self.rag.topk(f"{topic} {statement}", k=4))
        if doc:
            ctx_parts.append("### Documentation et lecons pertinentes :\n" + doc[:2000])
        prompt = (
            ("\n\n".join(ctx_parts) + "\n\n" if ctx_parts else "")
            + f"Exercice : {statement}\n"
            + (extra + "\n" if extra else "")
            + "Ecris UNIQUEMENT le code Python demande (fonction complete, imports "
              "inclus), dans un bloc ```python."
        )
        return extract_code(self.llm.generate(prompt, None))

    # ------------------------------------------------------------- 4. reflexion
    def reflect(self, statement: str, solution: str, error: str) -> tuple[str, str]:
        """Reflexion sur l'echec : lit l'erreur d'execution, en tire une LECON
        generale (reutilisable au-dela de cet exercice) et un code corrige."""
        prompt = (
            "Ton code a ECHOUE a l'execution.\n"
            f"Exercice : {statement}\n"
            f"Ton code :\n```python\n{solution}\n```\n"
            f"Erreur d'execution :\n{error[:1200]}\n\n"
            "Reponds au format EXACT :\n"
            "LECON: <une regle generale et courte a retenir pour ne plus refaire "
            "cette erreur>\n"
            "```python\n<code corrige complet>\n```"
        )
        raw = self.llm.generate(prompt, None) or ""
        lesson = ""
        m = re.search(r"LECON\s*:\s*(.+)", raw)
        if m:
            lesson = m.group(1).strip().splitlines()[0].strip()
        return extract_code(raw), lesson

    # ------------------------------------------------------------- 5. le cycle
    def attempt(self, exercise: dict, topic: str) -> dict:
        """Tente un exercice : generation -> execution -> boucle de reflexion.
        Retourne {passed_first, passed, tries, solution, lesson}."""
        statement, tests = exercise["statement"], exercise["tests"]
        solution = self._generate_solution(statement, topic)
        ok, output = self._execute(solution, tests) if solution else (False, "aucun code produit")
        passed_first = ok
        lesson = ""
        tries = 1
        while not ok and tries <= self.max_retries:
            fixed, lesson_i = self.reflect(statement, solution or "(vide)", output)
            lesson = lesson_i or lesson
            if not fixed:
                break
            solution = fixed
            ok, output = self._execute(solution, tests)
            tries += 1
        if ok:
            fn = _DEF_RE.search(solution)
            added = self.skills.add(fn.group(1) if fn else "solution",
                                    statement, solution, topic=topic, tests=tests)
            if added:
                self.log(f"   ✓ competence verifiee ajoutee : {added.name}")
        if lesson and ok:
            # Une lecon n'entre en memoire que VALIDEE (la correction qui l'accompagne a
            # reussi) : une lecon tiree d'un echec non resolu est souvent fausse et
            # EMPOISONNE le retrieval (mesure : eval fixe 5->4 apres injection de
            # lecons non validees).
            self.rag.add_document(f"LECON ({topic}) : {lesson}")
            self.log(f"   ✎ lecon validee : {lesson}")
        return {"passed_first": passed_first, "passed": ok, "tries": tries,
                "solution": solution, "lesson": lesson}

    def study_cycle(self, topic: str, n_exercises: int = 3, k_pages: int = 3,
                    focus: list[str] | None = None) -> dict:
        """Un cycle complet : recherche -> pratique -> reflexion -> distillation
        -> mesure. `focus` = syllabus d'APIs pour ce cycle (curriculum tournant).
        Retourne le resume, et le journalise dans le ledger."""
        self.log(f"— cycle d'etude : « {topic} »"
                 + (f" — focus : {', '.join(focus)}" if focus else ""))
        self.log("  [1/3] recherche documentaire")
        n_pages = self.research(topic, k_pages=k_pages)
        self.log(f"  [2/3] pratique ({n_exercises} exercices auto-generes)")
        exercises = self.make_exercises(topic, n=n_exercises, focus=focus)
        if not exercises:
            self.log("  aucun exercice valide genere — cycle blanc (journalise)")
            self.ledger.record_cycle(topic, 0, 0, 0)
            return {"topic": topic, "pass1": 0, "final": 0, "total": 0, "pages": n_pages}
        results, lessons, new_skills = [], [], []
        for i, ex in enumerate(exercises, 1):
            self.log(f"   exercice {i}/{len(exercises)} : {ex['statement'][:70]}…")
            r = self.attempt(ex, topic)
            results.append(r)
            if r["lesson"]:
                lessons.append(r["lesson"])
            if r["passed"]:
                fn = _DEF_RE.search(r["solution"] or "")
                new_skills.append(fn.group(1) if fn else "solution")
            self.log(f"     -> 1er essai {'✅' if r['passed_first'] else '❌'} | "
                     f"final {'✅' if r['passed'] else '❌'} ({r['tries']} essai(s))")
        pass1 = sum(1 for r in results if r["passed_first"])
        final = sum(1 for r in results if r["passed"])
        self.ledger.record_cycle(topic, pass1, final, len(results),
                                 lessons=lessons, skills=new_skills)
        self.log(f"  [3/3] bilan : pass@1 {pass1}/{len(results)} | "
                 f"final {final}/{len(results)} | LTM {self.ltm.count()} paires | "
                 f"competences {self.skills.count()}")
        return {"topic": topic, "pass1": pass1, "final": final,
                "total": len(results), "pages": n_pages}

    def run(self, topics: list[str] | None = None, cycles: int = 1,
            n_exercises: int = 3) -> list[dict]:
        """Enchaine des cycles. Sans sujets fournis, suit le ledger (faiblesses
        observees en usage reel, puis sujets les moins maitrises)."""
        out = []
        for c in range(cycles):
            topic = (topics[c % len(topics)] if topics else self.ledger.pick_topic())
            if not topic:
                self.log("rien a etudier (aucune faiblesse signalee) — stop")
                break
            out.append(self.study_cycle(topic, n_exercises=n_exercises))
        return out
