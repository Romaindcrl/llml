"""Configuration et comptage de tokens.

Compteur de tokens : approximation pure stdlib (len(text)//4). PAS de tiktoken
ni de dependance lourde : on veut un comportement deterministe et portable pour
le jalon TEXTE. L'approximation 1 token ~= 4 caracteres est suffisante pour
piloter les seuils de compaction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Racine projet (ce fichier est dans <projet>/m0/config.py).
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count_tokens(text: str) -> int:
    """Approxime le nombre de tokens d'un texte.

    Regle : 0 pour texte vide, sinon max(1, len(text)//4). Le max(1, ...)
    garantit qu'un texte non vide compte toujours pour au moins 1 token.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _tmp_workdir() -> str:
    """Repertoire de travail par defaut (un sous-dossier temporaire par defaut).

    L'agent/bench surchargera ce champ par tache pour confiner l'execution des
    outils. On donne ici un defaut neutre dans le tmp systeme.
    """
    import tempfile

    return os.path.join(tempfile.gettempdir(), "m0_workdir")


@dataclass
class Config:
    """Parametres du systeme M0.

    Les seuils sont VOLONTAIREMENT reduits par rapport a opencode (40K/20K) car
    le modele local vise une fenetre ~8K tokens. Origine opencode :
      - garde ~40K tokens d'outputs avant de pruner,
      - prune si > 20K tokens sont recuperables,
      - protege toujours les 2 derniers tours + les outputs de skill.
    On transpose ces ratios a une fenetre ~8K : trigger 4000, recuperable min 800,
    contexte max 6000, on protege les 2 derniers tours.
    """

    backend: str = "mock"
    ollama_host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.0
    seed: int = 0

    # Backend MLX (Apple Silicon, in-process via mlx-lm). Defaut : Llama 3.2 3B
    # 4bit deja present dans models/ (meilleur que le 1B pour l'agentique).
    mlx_model_path: str = field(
        default_factory=lambda: os.path.join(_PROJECT_DIR, "models", "mlx-3b-4bit")
    )
    mlx_max_tokens: int = 512
    # Adapter LoRA charge par-dessus le modele de base (None = base seule).
    mlx_adapter_path: str | None = None

    # Lane D2L (M1) : parametres d'entrainement du LoRA lors du /sleep.
    d2l_iters: int = 120
    d2l_num_layers: int = 8
    d2l_repeat: int = 4
    d2l_anchor_repeat: int = 4  # poids du rehearsal (ancres) vs faits
    d2l_learning_rate: float = 5e-5  # 5e-5 stable sur 8-bit ; 1e-4 divergeait (val_loss -> garbage)

    # Seuil d'acquisition (held-out) de la gate /sleep. Calibre au regime reel :
    # le recall held-out par session est ~30-50%, donc 0.6 rejetait presque tout.
    gate_acq: float = 0.45

    # Mode conversationnel (serveur/UI) : si False, le chat repond directement
    # sans exposer les outils (les petits modeles emettent du JSON de tool-call
    # au lieu de repondre). Les outils restent actifs en mode tache (run_task).
    chat_use_tools: bool = False

    # Seuils de contexte / compaction (reduits vs opencode, voir docstring).
    max_context_tokens: int = 6000
    compaction_trigger_tokens: int = 4000
    min_recoverable_tokens: int = 800
    keep_recent_turns: int = 2
    memory_inject_cap_tokens: int = 1200

    max_turns: int = 20

    # Persistance.
    db_path: str = ":memory:"
    memory_path: str = field(default_factory=lambda: os.path.join(_PROJECT_DIR, "logs", "MEMORY.md"))
    repo_dir: str = field(default_factory=lambda: _PROJECT_DIR)
    workdir: str = field(default_factory=_tmp_workdir)

    @classmethod
    def from_env(cls) -> "Config":
        """Construit une Config depuis l'environnement.

        Lit M0_BACKEND, M0_MODEL, M0_OLLAMA_HOST si presents, sinon defaults.
        """
        kwargs: dict = {}
        backend = os.environ.get("M0_BACKEND")
        if backend:
            kwargs["backend"] = backend
        model = os.environ.get("M0_MODEL")
        if model:
            kwargs["model"] = model
        ollama_host = os.environ.get("M0_OLLAMA_HOST")
        if ollama_host:
            kwargs["ollama_host"] = ollama_host
        mlx_path = os.environ.get("M0_MLX_MODEL_PATH")
        if mlx_path:
            kwargs["mlx_model_path"] = mlx_path
        mlx_adapter = os.environ.get("M0_MLX_ADAPTER")
        if mlx_adapter:
            kwargs["mlx_adapter_path"] = mlx_adapter
        mlx_max = os.environ.get("M0_MLX_MAX_TOKENS")
        if mlx_max:
            try:
                kwargs["mlx_max_tokens"] = int(mlx_max)
            except ValueError:
                pass
        lr = os.environ.get("M0_D2L_LR")
        if lr:
            try:
                kwargs["d2l_learning_rate"] = float(lr)
            except ValueError:
                pass
        gacq = os.environ.get("M0_GATE_ACQ")
        if gacq:
            try:
                kwargs["gate_acq"] = float(gacq)
            except ValueError:
                pass
        # Seuils memoire / compaction pilotables sans toucher au code.
        for env_name, field_name in (
            ("M0_MEMORY_CAP", "memory_inject_cap_tokens"),
            ("M0_COMPACT_TRIGGER", "compaction_trigger_tokens"),
            ("M0_MIN_RECOVERABLE", "min_recoverable_tokens"),
            ("M0_MAX_CONTEXT", "max_context_tokens"),
            ("M0_KEEP_RECENT_TURNS", "keep_recent_turns"),
            ("M0_MAX_TURNS", "max_turns"),
            ("M0_D2L_ITERS", "d2l_iters"),
            ("M0_D2L_LAYERS", "d2l_num_layers"),
            ("M0_D2L_REPEAT", "d2l_repeat"),
            ("M0_D2L_ANCHOR_REPEAT", "d2l_anchor_repeat"),
        ):
            raw = os.environ.get(env_name)
            if raw:
                try:
                    kwargs[field_name] = int(raw)
                except ValueError:
                    pass
        return cls(**kwargs)
