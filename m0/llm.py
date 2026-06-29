"""Clients LLM : abstraction + backend Ollama (httpx) + backend Mock deterministe.

Contrat chat() : retourne TOUJOURS un dict {"content": str, "tool_calls": list[ToolCall]}.
Contrat generate() : retourne un str (utilise pour les resumes de compaction).

  messages : list de {"role": str, "content": str}
  tools    : list de {"name": str, "description": str, "parameters": <json-schema dict>}
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .config import Config
from .types import ToolCall

# Marqueur de fin de tache emis par le modele (ou injecte par le mock).
DONE_MARKER = "[[DONE]]"

# Regex de secours pour parser un tool-call exprime en texte par un modele qui
# ne fait pas de function-calling natif : un bloc ```action {json} ```.
_ACTION_BLOCK_RE = re.compile(
    r"```action\s*(?P<json>\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


class LLMClient(ABC):
    """Interface commune a tous les backends LLM."""

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Un tour de conversation.

        Retourne {"content": str, "tool_calls": list[ToolCall]}.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generation simple (sans outils) utilisee pour les resumes."""
        raise NotImplementedError


def _find_balanced_json(text: str) -> list[str]:
    """Extrait les sous-chaines {...} a accolades equilibrees (hors chaines).

    Permet de recuperer un objet JSON nu emis par un modele qui n'entoure pas
    son tool-call d'un bloc ```action``` (cas observe avec Llama 3.2 3B)."""
    objs: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                objs.append(text[start : i + 1])
                start = -1
    return objs


def _coerce_tool_call(data: object) -> ToolCall | None:
    """Construit un ToolCall depuis un dict {"name","args"} si valide."""
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = data.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name=name, args=args)


def _parse_action_block(content: str) -> list[ToolCall]:
    """Parse un tool-call exprime en texte.

    1) bloc ```action {json {name,args}} ``` (forme guidee) ;
    2) fallback : objet JSON nu {"name":..., "args":...} sans fence.
    """
    text = content or ""
    calls: list[ToolCall] = []
    for m in _ACTION_BLOCK_RE.finditer(text):
        try:
            data = json.loads(m.group("json"))
        except (json.JSONDecodeError, ValueError):
            continue
        call = _coerce_tool_call(data)
        if call is not None:
            calls.append(call)
    if not calls:  # fallback : JSON nu sans bloc ```action```
        for blob in _find_balanced_json(text):
            try:
                data = json.loads(blob)
            except (json.JSONDecodeError, ValueError):
                continue
            call = _coerce_tool_call(data)
            if call is not None:
                calls.append(call)
    return calls


def _tools_prompt(tools: list[dict]) -> str:
    """Construit l'instruction texte du protocole d'outils pour un modele sans
    function-calling natif (petits Llama). Coherent avec _parse_action_block."""
    lines = [
        "Tu es un agent qui resout la tache en utilisant des outils.",
        "Pour APPELER un outil, reponds UNIQUEMENT par un bloc (rien d'autre) :",
        "```action",
        '{"name": "<nom_outil>", "args": {"<param>": "<valeur>"}}',
        "```",
        f"Quand la tache est terminee, ecris la ligne {DONE_MARKER}",
        "",
        "Outils disponibles :",
    ]
    for t in tools:
        params = t.get("parameters", {})
        props = params.get("properties", params) if isinstance(params, dict) else {}
        lines.append(
            f"- {t.get('name', '')}: {t.get('description', '')} "
            f"| args={json.dumps(props, ensure_ascii=False)}"
        )
    return "\n".join(lines)


class OllamaClient(LLMClient):
    """Backend Ollama via l'API /api/chat (httpx, stream=False)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.host = cfg.ollama_host.rstrip("/")

    def _options(self) -> dict:
        return {"temperature": self.cfg.temperature, "seed": self.cfg.seed}

    @staticmethod
    def _to_ollama_tools(tools: list[dict] | None) -> list[dict] | None:
        """Convertit nos schemas d'outils au format attendu par Ollama."""
        if not tools:
            return None
        out: list[dict] = []
        for t in tools:
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    },
                }
            )
        return out

    def _post(self, payload: dict) -> dict:
        import httpx

        url = f"{self.host}/api/chat"
        try:
            resp = httpx.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:  # erreur cote serveur Ollama
            raise RuntimeError(
                f"Ollama a repondu une erreur HTTP {exc.response.status_code} sur "
                f"{self.host} — verifier le modele `ollama pull {self.cfg.model}`."
            ) from exc
        except httpx.HTTPError as exc:  # connexion impossible / timeout
            raise RuntimeError(
                f"Ollama injoignable sur {self.host} — installer/lancer ollama et "
                f"`ollama pull {self.cfg.model}`."
            ) from exc

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": self._options(),
        }
        ollama_tools = self._to_ollama_tools(tools)
        if ollama_tools:
            payload["tools"] = ollama_tools

        data = self._post(payload)
        message = data.get("message", {}) or {}
        content = message.get("content", "") or ""

        tool_calls: list[ToolCall] = []
        # 1) tool-calls natifs Ollama.
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            name = fn.get("name")
            if not name:
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(ToolCall(name=name, args=args))

        # 2) fallback : bloc ```action ...``` dans le texte si pas de tool-call natif.
        if not tool_calls:
            tool_calls = _parse_action_block(content)

        return {"content": content, "tool_calls": tool_calls}

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": self._options(),
        }
        data = self._post(payload)
        return (data.get("message", {}) or {}).get("content", "") or ""


class MockClient(LLMClient):
    """Backend deterministe SANS reseau, pilote par un script fourni par la tache.

    Chaque step du script est l'un de :
      {"say": str}
      {"tool": {"name": str, "args": dict}}
      {"done": True, "say"?: str}

    chat() depile le prochain step et le mappe vers {"content", "tool_calls"}.
    Si le script est vide -> {"content": DONE_MARKER, "tool_calls": []}.
    """

    def __init__(self, script: list[dict] | None = None) -> None:
        self._script: list[dict] = list(script) if script else []
        self._pos = 0

    def set_script(self, steps: list[dict]) -> None:
        """Charge un nouveau script et remet le curseur a zero."""
        self._script = list(steps) if steps else []
        self._pos = 0

    def _next_step(self) -> dict | None:
        if self._pos >= len(self._script):
            return None
        step = self._script[self._pos]
        self._pos += 1
        return step

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        step = self._next_step()
        if step is None:
            # Script epuise (ou vide) -> fin de tache.
            return {"content": DONE_MARKER, "tool_calls": []}

        if step.get("done"):
            say = step.get("say", "")
            content = f"{say}\n{DONE_MARKER}" if say else DONE_MARKER
            return {"content": content, "tool_calls": []}

        if "tool" in step:
            tool = step["tool"] or {}
            name = tool.get("name", "")
            args = tool.get("args", {})
            if not isinstance(args, dict):
                args = {}
            return {
                "content": step.get("say", ""),
                "tool_calls": [ToolCall(name=name, args=args)],
            }

        # step {"say": ...} ordinaire.
        return {"content": step.get("say", ""), "tool_calls": []}

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Resume structure canne (5 champs), deterministe."""
        return (
            "## Resume (gist)\n"
            "- Goal: objectif courant de la tache.\n"
            "- Instructions: contraintes et regles a respecter.\n"
            "- Decouvertes: faits etablis pendant la session.\n"
            "- Travail: actions realisees et outils invoques.\n"
            "- Fichiers: chemins crees ou modifies pertinents."
        )


class MLXClient(LLMClient):
    """Backend MLX in-process (mlx-lm) : modele local Apple Silicon, sans serveur
    ni reseau. Les petits Llama ne font pas de function-calling natif fiable, donc
    on injecte un protocole texte (bloc ```action {json}```) parse par
    _parse_action_block — coherent avec DONE_MARKER et le reste du systeme.
    """

    _cache: dict[str, tuple] = {}

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.path = cfg.mlx_model_path
        self.adapter_path = getattr(cfg, "mlx_adapter_path", None)
        self._model = None
        self._tok = None

    def _cache_key(self) -> str:
        return f"{self.path}::{self.adapter_path or ''}"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        key = self._cache_key()
        if key not in MLXClient._cache:
            try:
                from mlx_lm import load
            except ImportError as exc:
                raise RuntimeError(
                    "mlx-lm absent du venv — `uv pip install mlx-lm` (Apple Silicon)."
                ) from exc
            if self.adapter_path:
                MLXClient._cache[key] = load(self.path, adapter_path=self.adapter_path)
            else:
                MLXClient._cache[key] = load(self.path)
        self._model, self._tok = MLXClient._cache[key]

    def set_adapter(self, adapter_path: str | None) -> None:
        """Hot-swap du LoRA : la prochaine generation rechargera base (+ adapter).

        Vide TOUT le cache modele pour ne garder qu'UN modele charge a la fois :
        empiler plusieurs variantes (base, A, B, stacked...) sature la RAM/GPU -> OOM.
        """
        import gc

        self.adapter_path = adapter_path
        MLXClient._cache.clear()
        self._model = None
        self._tok = None
        gc.collect()
        try:  # libere les buffers GPU mlx (best-effort selon version)
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def _generate_text(self, prompt_text: str) -> str:
        from mlx_lm import generate as mlx_generate

        kwargs: dict = {"max_tokens": self.cfg.mlx_max_tokens, "verbose": False}
        try:  # sampler explicite (temperature) si l'API le permet.
            from mlx_lm.sample_utils import make_sampler

            kwargs["sampler"] = make_sampler(temp=float(self.cfg.temperature))
        except Exception:  # pragma: no cover — compat versions mlx-lm
            pass
        return mlx_generate(self._model, self._tok, prompt=prompt_text, **kwargs)

    def _render(self, messages: list[dict]) -> str:
        return self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self._ensure_loaded()
        msgs = list(messages)
        if tools:
            tp = _tools_prompt(tools)
            if msgs and msgs[0].get("role") == "system":  # fusion : 1 seul system
                msgs[0] = {"role": "system", "content": tp + "\n\n" + msgs[0]["content"]}
            else:
                msgs = [{"role": "system", "content": tp}] + msgs
        content = self._generate_text(self._render(msgs))
        return {"content": content, "tool_calls": _parse_action_block(content)}

    def generate(self, prompt: str, system: str | None = None) -> str:
        self._ensure_loaded()
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self._generate_text(self._render(msgs))


def make_client(cfg: Config) -> LLMClient:
    """Factory : instancie le backend selon cfg.backend."""
    backend = (cfg.backend or "mock").lower()
    if backend == "ollama":
        return OllamaClient(cfg)
    if backend == "mlx":
        return MLXClient(cfg)
    if backend == "mock":
        return MockClient()
    raise ValueError(
        f"Backend LLM inconnu : {cfg.backend!r} (attendu: mock|ollama|mlx)"
    )
