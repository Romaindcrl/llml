"""Entraîne un 2e tenant 'HelpDeskPro' (spec DISTINCTE de VIS) sur la MÊME base qwen-7B-8bit.

Sert la démo multi-tenant : une base, deux adaptateurs (VIS + HelpDeskPro), switch à la volée.
Même config que vis_spec_v2 (num_layers 8, rank 16, lr 5e-5) pour être hot-swappable sur la même base.
"""

from __future__ import annotations

import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from m0 import d2l  # noqa: E402
from m0.config import Config  # noqa: E402
from m0.llm import make_client  # noqa: E402

ADAPTER = os.path.join(_PROJ, "models", "lora", "clientB")

# Spec HelpDeskPro (fictif, DISTINCT de VIS) — conventions + faits, couverture forcée.
SEED = [
    ("Comment est déterminé le délai de traitement d'un ticket ?",
     "Chaque ticket a un SLA fondé sur sa priorité."),
    ("Quelle échelle de priorité utilise HelpDeskPro pour les tickets ?",
     "Les priorités vont de P1 (critique) à P4 (basse)."),
    ("Que se passe-t-il si un ticket dépasse son SLA sans résolution ?",
     "Il est escaladé automatiquement au manager."),
    ("Comment garde-t-on une trace des actions sur un ticket ?",
     "Toute action est tracée dans l'historique du ticket."),
    ("Comment le client est-il tenu informé de l'avancement ?",
     "Il est notifié par email à chaque changement de statut."),
    ("Quel est le code couleur de la marque HelpDeskPro ?",
     "La couleur de marque est #FF6B35."),
    ("Quel est le SLA par défaut d'un ticket P1 ?",
     "Le SLA par défaut d'un P1 est de 1 heure."),
    ("Avec quel fournisseur HelpDeskPro envoie-t-il ses emails ?",
     "Les emails sont envoyés via Postmark."),
    ("Quelle technologie alimente la recherche dans la base de connaissances ?",
     "La recherche utilise Algolia."),
    ("Quel est le statut initial d'un ticket nouvellement créé ?",
     "Le statut par défaut est 'open'."),
]


def main():
    cfg = Config.from_env(); cfg.backend = "mlx"
    print(f"=== entraînement tenant HelpDeskPro sur {os.path.basename(cfg.mlx_model_path)} ===", flush=True)
    llm = make_client(cfg)
    train = list(SEED) + d2l.augment_pairs(SEED, llm.generate, n_paraphrases=6)
    train = d2l.clean_and_balance(train, max_per_answer=14)
    print(f"{len(train)} Q/R d'entraînement", flush=True)
    data = os.path.join(_PROJ, "logs", "clientB_data")
    n = d2l.build_chat_dataset(train, data, repeat=5, anchors=d2l.ANCHOR_PAIRS, anchor_repeat=4)
    iters = min(700, max(350, 9 * len(train)))
    res = d2l.train_lora(cfg.mlx_model_path, data, ADAPTER, iters=iters, num_layers=cfg.d2l_num_layers,
                         learning_rate=cfg.d2l_learning_rate, rank=16, python_exe=sys.executable)
    print(f"LoRA HelpDeskPro: ok={res['ok']} val_loss={res['val_loss']} -> {ADAPTER}", flush=True)


if __name__ == "__main__":
    main()
