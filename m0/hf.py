"""Backend HF/CUDA : transformers + bitsandbytes (8-bit) + peft (hot-swap LoRA).

Portage du contrat MLXClient pour machines NVIDIA (RunPod) — voir eval/CDC_EVAL_LLML.md.
Contrats respectés :
  - chat()/generate() identiques à MLXClient (template chat du tokenizer, greedy à
    temperature 0, budget de génération lu dans cfg.mlx_max_tokens AU MOMENT de l'appel
    car les scripts de bench mutent cet attribut après construction) ;
  - set_adapter(path|None) : hot-swap PEFT SANS rechargement de la base (contrairement
    au MLXClient qui recharge tout) ; adapter_path reflète l'état courant — serve.py
    (_ensure_adapter, rollback de /sleep) s'appuie dessus ;
  - un seul modèle de base résident par (chemin, quantization), partagé entre instances.

Différence assumée vs MLX : les adapters restent résidents en VRAM une fois chargés
(quelques dizaines de Mo chacun) ; le swap est un set_adapter peft (~ms), pas un reload.
"""

from __future__ import annotations

import gc
import os
import threading

from .config import Config
from .llm import LLMClient, _parse_action_block, _tools_prompt


def _norm(path: str | None) -> str | None:
    return os.path.abspath(path) if path else None


class HFClient(LLMClient):
    """Backend transformers/CUDA. cfg.hf_model_path = repo HF ou dossier local."""

    # Registre partagé : base_key -> {"model", "tok", "adapters": {path: name}, "active": path|None}
    _registry: dict[str, dict] = {}
    _lock = threading.Lock()

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.path = cfg.hf_model_path
        self.quant = (cfg.hf_quant or "8bit").lower()
        self.adapter_path = _norm(getattr(cfg, "hf_adapter_path", None))
        self._entry = None

    # ------------------------------------------------------------------ chargement
    def _base_key(self) -> str:
        return f"{self.path}::{self.quant}"

    def _load_base(self) -> dict:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_cfg = AutoConfig.from_pretrained(self.path)
        kwargs: dict = {"device_map": {"": 0}, "low_cpu_mem_usage": True}
        if getattr(model_cfg, "model_type", "") == "gemma2":
            # Gemma 2 : logit soft-capping incompatible sdpa -> eager requis.
            kwargs["attn_implementation"] = "eager"
        if self.quant == "8bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.quant == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            raise ValueError(f"hf_quant inconnu : {self.quant!r} (attendu: 8bit|bf16)")

        tok = AutoTokenizer.from_pretrained(self.path)
        model = AutoModelForCausalLM.from_pretrained(self.path, **kwargs)
        model.eval()
        return {"model": model, "tok": tok, "adapters": {}, "active": None}

    def _ensure_loaded(self) -> None:
        if self._entry is not None:
            return
        key = self._base_key()
        with HFClient._lock:
            if key not in HFClient._registry:
                HFClient._registry[key] = self._load_base()
            self._entry = HFClient._registry[key]
        if self.adapter_path:
            self.set_adapter(self.adapter_path)

    # ------------------------------------------------------------------ hot-swap
    def set_adapter(self, adapter_path: str | None) -> None:
        """Active l'adapter donné (None = base nue). Idempotent, sans reload de base."""
        target = _norm(adapter_path)
        self._ensure_loaded()
        entry = self._entry
        if target == entry["active"]:
            self.adapter_path = target
            return
        with HFClient._lock:
            from peft import PeftModel

            model = entry["model"]
            if target is None:
                if entry["adapters"]:
                    model.base_model.disable_adapter_layers()
            else:
                if target not in entry["adapters"]:
                    name = f"a{len(entry['adapters'])}"
                    if not entry["adapters"]:  # premier adapter : on wrappe la base
                        model = PeftModel.from_pretrained(
                            model, target, adapter_name=name, is_trainable=False
                        )
                        entry["model"] = model
                    else:
                        model.load_adapter(target, adapter_name=name)
                    entry["adapters"][target] = name
                model.base_model.enable_adapter_layers()
                model.set_adapter(entry["adapters"][target])
            entry["active"] = target
            self.adapter_path = target
        gc.collect()

    # ------------------------------------------------------------------ génération
    @staticmethod
    def _sanitize(messages: list[dict]) -> list[dict]:
        """Remappe le rôle 'tool' (émis par Agent._build_messages) en 'user' :
        certains templates chat HF rejettent role='tool' sans métadonnées."""
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "tool":
                role, content = "user", f"[resultat outil]\n{content}"
            out.append({"role": role, "content": content})
        return out

    def _generate_text(self, messages: list[dict]) -> str:
        import torch

        entry = self._entry
        tok, model = entry["tok"], entry["model"]
        prompt = tok.apply_chat_template(
            self._sanitize(messages), tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        temp = float(self.cfg.temperature)
        gen_kwargs: dict = {
            "max_new_tokens": int(getattr(self.cfg, "mlx_max_tokens", 512)),
            "pad_token_id": tok.pad_token_id or tok.eos_token_id,
        }
        if temp <= 0:
            gen_kwargs.update({"do_sample": False, "temperature": None, "top_p": None,
                               "top_k": None})
        else:
            torch.manual_seed(int(self.cfg.seed))
            gen_kwargs.update({"do_sample": True, "temperature": temp})
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self._ensure_loaded()
        msgs = list(messages)
        if tools:
            tp = _tools_prompt(tools)
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = {"role": "system", "content": tp + "\n\n" + msgs[0]["content"]}
            else:
                msgs = [{"role": "system", "content": tp}] + msgs
        content = self._generate_text(msgs)
        return {"content": content, "tool_calls": _parse_action_block(content)}

    def generate(self, prompt: str, system: str | None = None) -> str:
        self._ensure_loaded()
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self._generate_text(msgs)
