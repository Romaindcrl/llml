"""Outils confines a un workdir : bash, read_file, write_file, edit.

Securite : TOUS les chemins sont resolus DANS le workdir. Toute tentative de
sortir du workdir (chemin absolu hors workdir, .. remontant au-dessus) est
refusee (ToolResult.ok = False) plutot que d'agir hors perimetre.

bash s'execute via subprocess avec timeout 30s et cwd=workdir.

tool_schemas() renvoie les 4 schemas (name/description/parameters au format
json-schema) a passer a llm.chat(). dispatch() route un ToolCall vers la bonne
methode de Tools.
"""

from __future__ import annotations

import os
import subprocess

from .types import ToolCall, ToolResult

_BASH_TIMEOUT = 30  # secondes


class Tools:
    """Boite a outils confinee a un repertoire de travail (workdir)."""

    def __init__(self, workdir: str) -> None:
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)

    # ------------------------------------------------------------- securite
    def _resolve(self, path: str) -> str | None:
        """Resout `path` DANS le workdir. Renvoie None si cela sortirait du
        workdir (refus). Les chemins relatifs sont ancres au workdir."""
        if path is None:
            return None
        candidate = path if os.path.isabs(path) else os.path.join(self.workdir, path)
        resolved = os.path.abspath(candidate)
        # Doit etre le workdir lui-meme ou un descendant.
        if resolved == self.workdir:
            return resolved
        prefix = self.workdir + os.sep
        if not resolved.startswith(prefix):
            return None
        return resolved

    # ---------------------------------------------------------------- outils
    def bash(self, cmd: str) -> ToolResult:
        """Execute une commande shell dans le workdir (timeout 30s)."""
        if not cmd or not str(cmd).strip():
            return ToolResult(name="bash", ok=False, output="commande vide")
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=_BASH_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                name="bash",
                ok=False,
                output=f"timeout: commande > {_BASH_TIMEOUT}s",
            )
        except OSError as exc:
            return ToolResult(name="bash", ok=False, output=f"erreur d'execution: {exc}")

        out = proc.stdout or ""
        err = proc.stderr or ""
        combined = out
        if err:
            combined = (combined + "\n" if combined else "") + err
        ok = proc.returncode == 0
        if not ok:
            combined = f"[exit {proc.returncode}]\n{combined}".rstrip()
        return ToolResult(name="bash", ok=ok, output=combined)

    def read_file(self, path: str) -> ToolResult:
        """Lit un fichier (confine au workdir)."""
        resolved = self._resolve(path)
        if resolved is None:
            return ToolResult(
                name="read_file", ok=False, output=f"chemin hors workdir: {path!r}"
            )
        try:
            with open(resolved, "r", encoding="utf-8") as fh:
                return ToolResult(name="read_file", ok=True, output=fh.read())
        except FileNotFoundError:
            return ToolResult(name="read_file", ok=False, output=f"introuvable: {path}")
        except OSError as exc:
            return ToolResult(name="read_file", ok=False, output=f"erreur lecture: {exc}")

    def write_file(self, path: str, content: str) -> ToolResult:
        """Ecrit (cree/ecrase) un fichier (confine au workdir)."""
        resolved = self._resolve(path)
        if resolved is None:
            return ToolResult(
                name="write_file", ok=False, output=f"chemin hors workdir: {path!r}"
            )
        try:
            os.makedirs(os.path.dirname(resolved) or self.workdir, exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as fh:
                fh.write(content if content is not None else "")
            return ToolResult(
                name="write_file", ok=True, output=f"ecrit: {os.path.basename(resolved)}"
            )
        except OSError as exc:
            return ToolResult(name="write_file", ok=False, output=f"erreur ecriture: {exc}")

    def edit(self, path: str, old: str, new: str) -> ToolResult:
        """Remplace `old` par `new` dans un fichier (1re occurrence requise)."""
        resolved = self._resolve(path)
        if resolved is None:
            return ToolResult(
                name="edit", ok=False, output=f"chemin hors workdir: {path!r}"
            )
        try:
            with open(resolved, "r", encoding="utf-8") as fh:
                data = fh.read()
        except FileNotFoundError:
            return ToolResult(name="edit", ok=False, output=f"introuvable: {path}")
        except OSError as exc:
            return ToolResult(name="edit", ok=False, output=f"erreur lecture: {exc}")

        if old not in data:
            return ToolResult(
                name="edit", ok=False, output=f"motif introuvable dans {os.path.basename(resolved)}"
            )
        data = data.replace(old, new, 1)
        try:
            with open(resolved, "w", encoding="utf-8") as fh:
                fh.write(data)
        except OSError as exc:
            return ToolResult(name="edit", ok=False, output=f"erreur ecriture: {exc}")
        return ToolResult(
            name="edit", ok=True, output=f"edite: {os.path.basename(resolved)}"
        )


def tool_schemas() -> list[dict]:
    """Les 4 schemas d'outils (format llm.chat : name/description/parameters)."""
    return [
        {
            "name": "bash",
            "description": "Execute une commande shell dans le repertoire de travail (timeout 30s).",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "La commande shell a executer."}
                },
                "required": ["cmd"],
            },
        },
        {
            "name": "read_file",
            "description": "Lit le contenu d'un fichier (chemin relatif au workdir).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin du fichier a lire."}
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Cree ou ecrase un fichier avec un contenu donne.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin du fichier a ecrire."},
                    "content": {"type": "string", "description": "Contenu a ecrire."},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "edit",
            "description": "Remplace la 1re occurrence d'un texte par un autre dans un fichier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin du fichier a editer."},
                    "old": {"type": "string", "description": "Texte a remplacer."},
                    "new": {"type": "string", "description": "Texte de remplacement."},
                },
                "required": ["path", "old", "new"],
            },
        },
    ]


def dispatch(tools: Tools, call: ToolCall) -> ToolResult:
    """Route un ToolCall vers la methode correspondante de `tools`."""
    name = call.name
    args = call.args or {}
    if name == "bash":
        return tools.bash(args.get("cmd", ""))
    if name == "read_file":
        return tools.read_file(args.get("path", ""))
    if name == "write_file":
        return tools.write_file(args.get("path", ""), args.get("content", ""))
    if name == "edit":
        return tools.edit(args.get("path", ""), args.get("old", ""), args.get("new", ""))
    return ToolResult(name=name or "?", ok=False, output=f"outil inconnu: {name!r}")
