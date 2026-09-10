#!/usr/bin/env python3
"""v2 — runner de la matrice (CDC v2 §2.5). Configs :
  C0     : modèle nu, aucune information de conventions (hors fenêtre de fichier,
           inhérente à la tâche) — sert aussi de contrôle de contamination (Lot 1).
  C_ctx  : + fichier de conventions complet dans le contexte
  C_lora : + adapter procédural du repo (zéro conventions en contexte)
  C_both : adapter + conventions en contexte
  C_wrong: adapter d'un AUTRE repo (interférence croisée)

Prompt IDENTIQUE pour toutes les configs, à l'exception du bloc conventions.
Décodage greedy. Sortie JSONL : 1 ligne/tâche {task_id, config, output brut,
latence, chars de prompt}. Le scoring (checks) est fait séparément, hors pod.

Usage (pod) :
  /workspace/venv/bin/python eval/v2/run_matrix.py --config C0 \
      --tasks-dir eval/v2/tasks --outdir /workspace/results/v2/matrix \
      [--conventions-dir eval/v2/conventions] [--adapters-json path] [--repo X]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ)

from m0.config import Config   # noqa: E402
from m0.llm import make_client  # noqa: E402

PROMPT = """You are completing a function inside the file `{file}` of an existing codebase.
{conventions}Here is the relevant part of the file (the body of `{func}` has been removed):

{skeleton}

Implement the body of `{func}`, i.e. the code that replaces the line containing <<< IMPLEMENT BODY >>>.
Output ONLY the raw code of the function body — no function signature, no markdown fences, no explanations."""

CONV_BLOCK = """This codebase has strict coding conventions. Follow them exactly:

<conventions>
{conv}
</conventions>

"""


def load_tasks(tasks_dir: str, only_repo: str | None):
    tasks = []
    for p in sorted(glob.glob(os.path.join(tasks_dir, "*.jsonl"))):
        repo = os.path.splitext(os.path.basename(p))[0]
        if only_repo and repo != only_repo:
            continue
        with open(p, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    tasks.append(json.loads(line))
    if not tasks:
        raise ValueError(
            f"No frozen tasks found in {tasks_dir!r}"
            + (f" for repo {only_repo!r}" if only_repo else "")
            + ". Restore and verify the task artifacts before running the matrix."
        )
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task_id in the frozen task set")
    return tasks


def validate_inputs(config, tasks, conventions, adapters, wrong_map):
    """Reject mislabeled experimental arms before loading a model or writing output."""
    for repo in sorted({t["repo"] for t in tasks}):
        if config in ("C_ctx", "C_both") and not conventions.get(repo, "").strip():
            raise ValueError(f"Missing conventions for {repo} in {config}")
        if config in ("C_lora", "C_both", "C_wrong"):
            source = wrong_map.get(repo) if config == "C_wrong" else repo
            if not source or (config == "C_wrong" and source == repo):
                raise ValueError(f"Missing or non-distinct wrong adapter source for {repo}")
            path = adapters.get(source)
            if not path or not os.path.isfile(os.path.join(path, "adapter_config.json")):
                raise ValueError(f"Missing PEFT adapter config for {source} in {config}")
            if not any(os.path.isfile(os.path.join(path, name)) for name in (
                "adapter_model.safetensors", "adapter_model.bin"
            )):
                raise ValueError(f"Missing PEFT adapter weights for {source} in {config}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", default="8bit")
    ap.add_argument("--config", required=True,
                    choices=["C0", "C_ctx", "C_lora", "C_both", "C_wrong"])
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--conventions-dir", default=None,
                    help="dossier <repo>.md pour C_ctx/C_both")
    ap.add_argument("--adapters-json", default=None,
                    help='{"<repo>": "<adapter_dir>"} pour C_lora/C_both/C_wrong')
    ap.add_argument("--wrong-map-json", default=None,
                    help='{"<repo>": "<repo_source_adapter>"} pour C_wrong')
    ap.add_argument("--repo", default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--window", choices=["full", "nowin"], default="full",
                    help="nowin = signature+docstring seulement (pas de fenêtre de fichier)")
    a = ap.parse_args()

    conventions, adapters, wrong_map = {}, {}, {}
    if a.conventions_dir:
        for p in glob.glob(os.path.join(a.conventions_dir, "*.md")):
            conventions[os.path.splitext(os.path.basename(p))[0]] = open(p, encoding="utf-8").read()
    if a.adapters_json:
        adapters = json.load(open(a.adapters_json, encoding="utf-8"))
    if a.wrong_map_json:
        wrong_map = json.load(open(a.wrong_map_json, encoding="utf-8"))

    tasks = load_tasks(a.tasks_dir, a.repo)
    validate_inputs(a.config, tasks, conventions, adapters, wrong_map)

    os.makedirs(a.outdir, exist_ok=True)
    suffix = "" if a.window == "full" else "_nowin"
    out_path = os.path.join(a.outdir, f"gen_{a.config}{suffix}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["task_id"] for l in open(out_path, encoding="utf-8") if l.strip()}

    cfg = Config.from_env()
    cfg.backend = "hf"
    cfg.hf_model_path = a.model
    cfg.hf_quant = a.quant
    cfg.temperature = 0.0
    llm = make_client(cfg)
    llm.cfg.mlx_max_tokens = a.max_tokens
    fout = open(out_path, "a", encoding="utf-8")
    print(f"[{a.config}] {len(tasks)} tâches ({len(done)} déjà faites)", flush=True)
    cur_adapter = "___unset___"
    for i, t in enumerate(tasks, 1):
        if t["task_id"] in done:
            continue
        repo = t["repo"]
        conv = ""
        if a.config in ("C_ctx", "C_both"):
            conv = CONV_BLOCK.format(conv=conventions.get(repo, ""))
        target_adapter = None
        if a.config in ("C_lora", "C_both"):
            target_adapter = adapters.get(repo)
        elif a.config == "C_wrong":
            target_adapter = adapters.get(wrong_map.get(repo, ""))
        if target_adapter != cur_adapter:
            llm.set_adapter(target_adapter)
            cur_adapter = target_adapter
        skel = t["skeleton_nowin"] if a.window == "nowin" else t["skeleton"]
        prompt = PROMPT.format(file=t["file"], func=t["func"],
                               skeleton=skel, conventions=conv)
        t0 = time.time()
        out = llm.generate(prompt, None)
        dt = round((time.time() - t0) * 1000)
        fout.write(json.dumps({
            "task_id": t["task_id"], "repo": repo, "config": a.config,
            "output": out or "", "lat_ms": dt, "prompt_chars": len(prompt),
            "conv_chars": len(conv),
        }, ensure_ascii=False) + "\n")
        fout.flush()
        if i % 10 == 0 or i == len(tasks):
            print(f"  {i}/{len(tasks)} [{dt}ms]", flush=True)
    print(f"MATRIX_{a.config}{suffix.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
