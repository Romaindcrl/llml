# LLML v2 — Candidats « mémoire procédurale » (CDC §2.2) — 8 dépôts vérifiés

> Vérification 11/07/2026 : clone réel de chaque dépôt (dates de commit et comptages
> exacts depuis HEAD), docs de style lues à leur version courante. Tous actifs
> < 1 semaine, tous ≥ 245 fichiers de code. Exclusions respectées (pas de
> Romaindcrl, pas d'astral-sh). **Checkpoint humain Lot 1 : Romain valide 5 des 8.**

## 1. curl/curl — C — licence curl (≈MIT) — ~1 000 .c/.h — commit 10/07/2026
Docs/lint : docs/internals/CODE_STYLE.md · CHECKSRC.md · **linter maison scripts/checksrc.pl** (+ .checksrc par répertoire, CI checksrc.yml)
- `if(x)` — PAS d'espace mot-clé/parenthèse (anti-K&R) → checksrc SPACEBEFOREPAREN
- C89, `// interdit`, seulement `/* */` → CPPCOMMENTS
- `== NULL` / `!= 0` interdits en condition (`!var`) → EQUALSNULL, NOTEQUALSZERO
- typedef de struct interdit → TYPEDEFSTRUCT
- ~90 fonctions bannies (malloc, sprintf, strcpy…) → wrappers curl → BANNEDFUNC
- 2 espaces jamais TAB, 79 col., accolade fn seule sur sa ligne, else sur sa ligne
**Pourquoi** : l'oracle existe déjà (linter Perl committé, ~40 règles). **Risque** : célèbre → contamination possible.

## 2. FreeRTOS/FreeRTOS-Kernel — C — MIT — ~655 .c/.h — 09/07/2026
Docs/lint : Coding Standard (freertos.org) · .github/uncrustify.cfg · MISRA.md
- **Notation hongroise stricte** : ul/us/uc/x/ux/e/p* selon le type (uxPriority, pxCurrentTCB) → AST type↔préfixe
- Fonctions statiques préfixées `prv`
- API = préfixe type retour + nom fichier : vTaskDelete (void, tasks.c), xQueueSend
- Espaces DANS les parenthèses : `if( xReturn == pdTRUE )` (sp_inside_paren=force)
- Allman + accolades obligatoires, 4 espaces
- Macros préfixées par fichier : configUSE_*, pdTRUE, port* ; MISRA C:2012
**Pourquoi** : le système de types encodé dans les noms — déviation maximale au prior, 100% régexable. **Risque** : moyen.

## 3. tigerbeetle/tigerbeetle — Zig — Apache-2.0 — 245 .zig — 10/07/2026
Docs/lint : docs/TIGER_STYLE.md (in-repo)
- **70 lignes max par fonction** (limite dure) → comptage AST
- **≥2 assertions par fonction** ; `assert(a); assert(b);` plutôt que `assert(a and b)`
- **Récursion interdite** ; boucles bornées → graphe d'appels
- Types explicites (u32…), `usize` à éviter
- **snake_case pour les FICHIERS aussi** (anti-standard Zig), acronymes VSRState
- Unités par significativité décroissante : `latency_ms_max` PAS `max_latency_ms`
**Pourquoi** : règles quantifiables + Zig = contamination Qwen minimale. **Risque** : TIGER_STYLE médiatisé (HN).

## 4. twisted/twisted — Python — MIT — 1 115 .py — 06/07/2026
Docs/lint : docs/development/coding-standard.rst
- **camelCase pour les méthodes** — anti-PEP8 frontal → AST
- Callbacks `_cbNom` / errbacks `_ebNom` → regex
- Docstrings PARTOUT, format **epytext** (@param/@type + L{...}/C{...}, pas Sphinx)
- 1re ligne de module : cookie `# -*- test-case-name: twisted.test.test_x -*-`
- Wildcard imports ET relative imports interdits → AST
- Triple quotes ouvrantes/fermantes sur leur propre ligne ; interfaces IFoo
**Pourquoi** : LE cas « prior clash » Python — Qwen produit du snake_case/Sphinx par défaut. **Risque** : vieux projet très présent en pretraining.

## 5. nginx/nginx — C — BSD-2 — ~400 .c/.h — 08/07/2026
Docs/lint : Development guide §Code style (nginx.org)
- Type de retour de fonction SUR SA PROPRE LIGNE, nom en colonne 0
- Déclarations de variables ALIGNÉES en colonnes
- Comparaisons explicites OBLIGATOIRES : `if (p == NULL)` — **inverse exact de curl**
- Préfixes ngx_/NGX_ systématiques, types ngx_*_t
- 4 espaces, 80 col., accolades toujours, /* */ seulement
- 2 lignes vides entre fonctions
**Pourquoi** : le style C le plus visuellement rigide de l'open source. **Risque** : moyen-élevé.

## 6. godotengine/godot — C++ — MIT — ~7 600 fichiers — 10/07/2026
Docs/lint : code_style_guidelines · .clang-format (v17 épinglé) · .clang-tidy · misc/scripts/header_guards.py
- Paramètres `p_*`, sorties modifiées `r_*` → AST
- Indentation par TABS
- STL interdit (containers maison), `auto` interdit, exceptions interdites
- Types PascalCase, méthodes/variables snake_case (rare en C++)
- Macros d'erreur maison en garde : ERR_FAIL_COND, ERR_FAIL_NULL, ERR_FAIL_INDEX
- Ordre d'includes imposé + #pragma once → script in-repo
**Pourquoi** : C++ anti-mainstream dense avec formatteur épinglé. **Risque** : très populaire ; repo énorme (échantillonner core/).

## 7. zulip/zulip — Python + TypeScript — Apache-2.0 — ~2 500 fichiers — 10/07/2026
Docs/lint : docs/contributing/code-style.md · **tools/linter_lib/custom_check.py (1 021 lignes de règles regex maison)** · tools/semgrep-py.yml · .gitlint
- `msgid` interdit comme nom de variable → message_id ; `subject` interdit (vocabulaire « topic »)
- `@login_required` interdit → `@zulip_login_required`
- Requêtes directes interdites : UserProfile.objects.get() → get_user_profile_by_* ; semgrep access_stream_by_*
- Datetimes naïfs bannis → timezone_now()
- AJAX via module channel ($.get/$.post interdits) ; onclick interdit ; var interdit
- Format string : `x % (y,)` jamais `x % y`
**Pourquoi** : l'oracle est déjà écrit (zulint + semgrep committés). **Risque** : faible-moyen ; densité de règles par fichier variable.

## 8. micropython/micropython — C + Python — MIT — ~4 200 fichiers — 09/07/2026
Docs/lint : CODECONVENTIONS.md · tools/codeformat.py + uncrustify.cfg épinglé · **tools/verifygitlog.py** (vérificateur maison des messages de commit)
- Commits : `py/objstr: Add splitlines() method.` (préfixe chemin, phrase capitalisée, POINT FINAL, ≤72c)
- Commentaires **// uniquement, PAS /* */** — **inverse exact de curl/nginx**
- Publics préfixés mp_/MP_, préfixe = nom du fichier (py/obj.c → mp_obj_*) ; statiques SANS préfixe
- Typedefs : tag `_my_struct_t` (underscore initial) + suffixe `_t`
- 1 espace mot-clé/parenthèse + accolades même pour 1 ligne + else collé au } — **tout l'inverse de curl**
- m_new/m_renew/m_del obligatoires (pas de malloc/free nus)
**Pourquoi** : paire parfaite avec curl — règles opposées point à point → teste si la LoRA apprend LE dépôt, pas « un style C ». **Risque** : partie Python proche de PEP8 → cibler py/ et extmod/ (C).

## Réservistes (si un candidat saute au contrôle contamination C0 ≥ 85%)
- **postgres/postgres** (C, ~2 560 fichiers, 10/07) : Error Message Style Guide unique (messages primaires : pas de majuscule initiale, PAS de point final ; errdetail/errhint : l'inverse) + pgindent + typedefs.list. Risque : ultra-mémorisé.
- **openssl/openssl** (C, Apache-2.0, ~2 230 fichiers, 10/07) : STYLE.md — suffixes get0_/get1_/set0_/set1_, typedef ALL_CAPS ↔ tag minuscule _st, suffixes _cb/_fn, macros SUBSYSTEM_R_REASON. Risque : célèbre, code hétérogène.

## Synthèse — classement (idiosyncrasie × vérifiabilité)

| # | Repo | Lang. | Licence | Idio./5 | Vérif./5 | Score | Contamination |
|---|---|---|---|---|---|---|---|
| 1 | FreeRTOS-Kernel | C | MIT | 5 | 5 | 25 | moyen |
| 2 | curl | C | curl (≈MIT) | 5 | 5 | 25 | **élevé** |
| 3 | TigerBeetle | Zig | Apache-2.0 | 5 | 4 | 20 | faible-moyen |
| 4 | Twisted | Python | MIT | 4 | 5 | 20 | moyen |
| 5 | nginx | C | BSD-2 | 4 | 4 | 16 | moyen-élevé |
| 6 | Godot | C++ | MIT | 4 | 4 | 16 | **élevé** |
| 7 | MicroPython | C+Py | MIT | 3,5 | 4,5 | 16 | moyen |
| 8 | Zulip | Py+TS | Apache-2.0 | 3 | 5 | 15 | faible-moyen |

**Langues** : 4 dépôts majoritairement C → toute sélection de 5 garantit ≥ 2 langages.
**Combo recommandé** : FreeRTOS + curl + TigerBeetle + Twisted + Zulip (C, Zig, Python, TS —
4 langages, un seul « célèbre »). Variante zéro-célèbre : curl → MicroPython.
