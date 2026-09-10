# Amendement #2 au plan de pré-enregistrement — checkpoint QA du Lot 4

**Date :** 2026-07-05 · **Décidé par :** Romain (principal du projet).

## Objet

Le CDC (§5, Lot 4) impose un **checkpoint humain** : « Romain valide les 60 QA
générées » avant les runs de mémoire (Claim A). Le principal a choisi de **lever
ce checkpoint humain** et de déléguer la validation.

## Ce qui change

La validation des paires QA générées (20 par document technique) est confiée à
un **sous-agent Claude indépendant, en aveugle**, distinct de l'agent qui
génère les QA et de celui qui exécute/score les runs. Son mandat :

1. **Answerability** — la réponse est-elle réellement contenue dans le document
   ingéré (pas une connaissance générale) ?
2. **Correction** — la réponse annotée est-elle factuellement correcte vis-à-vis
   du texte source (citation exacte exigée) ?
3. **Non-trivialité** — un modèle nu (sans le document) ne devrait pas pouvoir
   répondre (cohérent avec le contrôle « C0-doit-échouer » du §4.2).

Une paire est **retenue** seulement si le validateur la confirme sur les trois
critères ; sinon elle est écartée et l'incident loggé. Le rôle de génération
(modèle non-Qwen) et le rôle de validation (Claude) sont séparés.

## Impact sur la valeur probante (déclaré honnêtement)

Le checkpoint humain servait à garantir l'**indépendance** du jeu de test
(éviter que le système « corrige sa propre copie »). Le remplacer par une
validation automatique — même par un modèle tiers en aveugle — **affaiblit**
cette garantie : il n'y a plus d'œil humain externe sur les 60 QA. Les résultats
de Claim A dérivés de ces QA générées doivent donc être lus avec ce caveat.

**Mitigations conservées** pour limiter la casse :
- Les QA restent **dérivées d'un corpus public vérifiable** (doc technique réel),
  jamais inventées de toutes pièces.
- Le contrôle **C0-doit-échouer** (§4.2) est appliqué sans exception : toute
  question à laquelle le modèle nu répond est écartée (doc contaminé/trivial).
- Le sous-partie **QuALITY** (§4.2) utilise les **QA natives du dataset**
  (rédigées par des humains, à choix multiples) — non concernée par cet
  amendement, elle offre un signal Claim A à indépendance forte préservée.
- Tous les transcripts de génération et de validation sont versionnés
  (`eval/lot4/qa/`) pour audit humain a posteriori.

## Traçabilité

Cet amendement sera reporté en commentaire sur l'issue de pré-enregistrement
(#1) avant les runs de Claim A, conformément au CDC §3.1 (« tout écart au plan
est documenté dans l'issue, jamais silencieux »).
