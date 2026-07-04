"""Entraînement LoRA /sleep sur CUDA : transformers + peft + bitsandbytes.

Réplique la recette mlx_lm de m0/d2l.py:train_lora (voir eval/CDC_EVAL_LLML.md §2.4) :
  - données : mêmes train.jsonl/valid.jsonl chat produits par d2l.build_chat_dataset ;
  - loss sur la séquence complète par défaut (mask_prompt=False, comme le /sleep MLX) ;
  - LoRA sur les `num_layers` DERNIÈRES couches ; équivalence d'échelle MLX->PEFT :
    delta MLX = scale * (x@a)@b avec scale=20 ; PEFT applique alpha/r, donc
    lora_alpha = scale * r (r=16 -> alpha=320). NE PAS remettre les défauts PEFT.
  - target_modules pinnés explicitement (les adapters MLX du repo incluaient les
    projections MLP — formes 18944x3584 dans lora_merge.py) : les 7 projections
    standard, mêmes noms sur Qwen2.5 / Llama 3.1 / Gemma 2 (rien d'hardcodé Qwen).
  - Adam, lr constant, batch_size=1, `iters` = pas d'optimiseur (pas des epochs).

Deux entrées :
  - train_lora(...) : wrapper sous-processus, MÊME signature/contrat de retour que
    d2l.train_lora (utilisé par scripts/serve.py) — l'entraînement tourne dans son
    propre process CUDA et libère sa VRAM en sortant ;
  - `python -m m0.d2l_hf --model ... --data ... --adapter-path ...` : le vrai train.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

_LOSS_RE = re.compile(r"(Train|Val) loss ([0-9.]+)")

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
MLX_SCALE = 20.0  # lora_parameters.scale du pipeline MLX (d2l.py:314)


def train_lora(
    base_model: str,
    data_dir: str,
    adapter_out: str,
    *,
    iters: int = 120,
    num_layers: int = 8,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    max_seq_length: int = 512,
    rank: int | None = None,
    mask_prompt: bool = False,
    python_exe: str | None = None,
    log_file: str | None = None,
    quant: str = "8bit",
) -> dict:
    """Contrat identique à d2l.train_lora : {ok, adapter_path, adapter_file,
    train_loss, val_loss, iters, rank, returncode, log_tail}."""
    py = python_exe or sys.executable
    os.makedirs(adapter_out, exist_ok=True)
    cmd = [
        py, "-m", "m0.d2l_hf",
        "--model", base_model,
        "--data", data_dir,
        "--adapter-path", adapter_out,
        "--iters", str(iters),
        "--num-layers", str(num_layers),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--max-seq-length", str(max_seq_length),
        "--rank", str(rank or 16),
        "--quant", quant,
    ]
    if mask_prompt:
        cmd.append("--mask-prompt")

    if log_file:
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n--- m0.d2l_hf: iters={iters} layers={num_layers} "
                     f"lr={learning_rate} rank={rank} quant={quant} ---\n")
            lf.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            lines = []
            for line in proc.stdout:
                lines.append(line)
                lf.write(line)
                lf.flush()
            proc.wait()
        out = "".join(lines)
        rc = proc.returncode
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        rc = proc.returncode

    train_loss = val_loss = None
    for m in _LOSS_RE.finditer(out):
        if m.group(1) == "Train":
            train_loss = float(m.group(2))
        else:
            val_loss = float(m.group(2))

    adapter_file = os.path.join(adapter_out, "adapter_model.safetensors")
    ok = rc == 0 and os.path.exists(adapter_file)
    tail = "\n".join(out.strip().splitlines()[-8:])
    return {
        "ok": ok,
        "adapter_path": adapter_out,
        "adapter_file": adapter_file,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "iters": iters,
        "rank": rank,
        "returncode": rc,
        "log_tail": tail,
    }


# --------------------------------------------------------------------- le train


def _load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _encode(tok, messages: list[dict], max_len: int, mask_prompt: bool):
    """Tokenise un échange chat complet ; labels = séquence entière, ou réponse
    seule si mask_prompt (équivalent du --mask-prompt mlx_lm)."""
    import torch

    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tok(full, truncation=True, max_length=max_len, return_tensors="pt")["input_ids"][0]
    labels = ids.clone()
    if mask_prompt:
        prompt = tok.apply_chat_template(messages[:-1], tokenize=False,
                                         add_generation_prompt=True)
        n = tok(prompt, truncation=True, max_length=max_len,
                return_tensors="pt")["input_ids"].shape[1]
        labels[:min(n, len(labels))] = -100
    return ids, labels


def _main() -> int:
    import argparse
    import random

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--mask-prompt", action="store_true")
    p.add_argument("--quant", default="8bit", choices=["8bit", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model_cfg = AutoConfig.from_pretrained(args.model)
    kwargs: dict = {"device_map": {"": 0}, "low_cpu_mem_usage": True,
                    "torch_dtype": torch.bfloat16}
    if getattr(model_cfg, "model_type", "") == "gemma2":
        kwargs["attn_implementation"] = "eager"
    if args.quant == "8bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.quant == "8bit":
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    n_layers = int(model_cfg.num_hidden_layers)
    first = max(0, n_layers - args.num_layers)
    lcfg = LoraConfig(
        r=args.rank,
        lora_alpha=int(MLX_SCALE * args.rank),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        layers_to_transform=list(range(first, n_layers)),
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    train_rows = _load_rows(os.path.join(args.data, "train.jsonl"))
    valid_rows = _load_rows(os.path.join(args.data, "valid.jsonl"))
    if not train_rows:
        print("ERREUR: train.jsonl vide")
        return 2

    opt = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad], lr=args.learning_rate
    )
    device = next(model.parameters()).device
    order = list(range(len(train_rows)))
    random.shuffle(order)
    pos, losses = 0, []
    model.train()
    for step in range(1, args.iters + 1):
        if pos >= len(order):
            random.shuffle(order)
            pos = 0
        row = train_rows[order[pos]]
        pos += 1
        ids, labels = _encode(tok, row["messages"], args.max_seq_length, args.mask_prompt)
        out = model(input_ids=ids.unsqueeze(0).to(device),
                    labels=labels.unsqueeze(0).to(device))
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if step % 10 == 0 or step == args.iters:
            avg = sum(losses[-10:]) / len(losses[-10:])
            print(f"Iter {step}: Train loss {avg:.3f}", flush=True)

    model.eval()
    vlosses = []
    with torch.no_grad():
        for row in valid_rows[:32]:
            ids, labels = _encode(tok, row["messages"], args.max_seq_length,
                                  args.mask_prompt)
            out = model(input_ids=ids.unsqueeze(0).to(device),
                        labels=labels.unsqueeze(0).to(device))
            vlosses.append(out.loss.item())
    if vlosses:
        print(f"Val loss {sum(vlosses) / len(vlosses):.3f}", flush=True)

    model.save_pretrained(args.adapter_path)
    meta = {
        "base_model": args.model,
        "quant": args.quant,
        "iters": args.iters,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "rank": args.rank,
        "lora_alpha": int(MLX_SCALE * args.rank),
        "target_modules": TARGET_MODULES,
        "mask_prompt": bool(args.mask_prompt),
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
    }
    with open(os.path.join(args.adapter_path, "training_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
