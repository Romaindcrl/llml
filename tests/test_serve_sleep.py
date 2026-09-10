"""Exercise sleep with stubs, without importing the server or loading ML libraries."""

import ast
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from m0 import d2l_hf
from m0.config import Config


class SleepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.memory = Path(self.tmp.name) / "models/lora/memory"
        self.memory.mkdir(parents=True)
        self.weight = self.memory / "adapter_model.safetensors"
        self.weight.write_text("old")
        self.calls = []
        self.client = SimpleNamespace(adapter_path=str(self.memory), resident=True)
        self.client.generate = lambda *args: "paraphrase"
        self.client.unload = lambda: setattr(self.client, "resident", False)
        self.client.set_adapter = self.set_adapter
        self.reject_promotion = False
        cfg = Config(backend="HF", hf_quant="BF16")
        d2l = SimpleNamespace(
            clean_and_balance=lambda pairs, **kwargs: pairs,
            augment_pairs=lambda pairs, generate, **kwargs: pairs,
            split_train_eval=lambda pairs, **kwargs: (pairs, pairs),
            build_chat_dataset=lambda *args, **kwargs: 2,
            ANCHOR_PAIRS=[],
        )
        self.ns = {
            "_PROJ": self.tmp.name, "_CFG": cfg, "_AGENT": SimpleNamespace(llm=self.client),
            "_LTM": SimpleNamespace(all_qa=lambda: [("q1", "a1"), ("q2", "a2")]),
            "d2l": d2l, "os": os, "shutil": shutil, "sys": sys,
            "tempfile": tempfile, "time": time, "_gate_adapter": self.gate,
        }
        source = Path(__file__).resolve().parents[1] / "scripts/serve.py"
        functions = {"_mem_adapter", "_promote_adapter", "_do_sleep"}
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in functions]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), self.ns)
        self.patch = patch.object(d2l_hf, "train_lora", self.train)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def set_adapter(self, path):
        if path:
            value = (Path(path) / "adapter_model.safetensors").read_text()
            if self.reject_promotion and path == str(self.memory) and value == "new":
                raise RuntimeError("load failed")
        self.client.adapter_path = path
        self.client.resident = True

    def train(self, base, data, output, **kwargs):
        self.calls.append((self.client.resident, kwargs["quant"], output))
        target = Path(output)
        target.mkdir(exist_ok=True)
        weight = target / "adapter_model.safetensors"
        weight.write_text("new")
        return {"ok": True, "returncode": 0, "val_loss": 1, "train_loss": 1,
                "adapter_path": output, "adapter_file": str(weight)}

    def gate(self, path, pairs):
        self.client.set_adapter(path)
        return 0, True

    def test_rejected_candidates_preserve_previous_weights_and_release_each_time(self):
        result = self.ns["_do_sleep"]()
        self.assertIn("REJETÉ", result)
        self.assertEqual(self.weight.read_text(), "old")
        self.assertEqual(self.client.adapter_path, str(self.memory))
        self.assertEqual(len(self.calls), 3)
        for resident, quant, output in self.calls:
            self.assertFalse(resident)
            self.assertEqual(quant, "bf16")
            self.assertNotEqual(output, str(self.memory))
            self.assertFalse(Path(output).exists())

    def test_accepted_candidate_is_published_to_canonical_path(self):
        self.ns["_gate_adapter"] = lambda path, pairs: (1, True)
        result = self.ns["_do_sleep"]()
        self.assertIn("Sleep terminé", result)
        self.assertEqual(self.weight.read_text(), "new")
        self.assertEqual(self.client.adapter_path, str(self.memory))
        self.assertEqual(self.ns["_LAST_SLEEP"]["adapter"], str(self.memory))

    def test_gate_exception_restores_previous_adapter(self):
        def broken_gate(path, pairs):
            self.client.set_adapter(path)
            raise RuntimeError("gate failed")
        self.ns["_gate_adapter"] = broken_gate
        with self.assertRaisesRegex(RuntimeError, "gate failed"):
            self.ns["_do_sleep"]()
        self.assertEqual(self.weight.read_text(), "old")
        self.assertEqual(self.client.adapter_path, str(self.memory))

    def test_failed_promotion_restores_previous_directory(self):
        self.ns["_gate_adapter"] = lambda path, pairs: (1, True)
        self.reject_promotion = True
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            self.ns["_do_sleep"]()
        self.assertEqual(self.weight.read_text(), "old")
        self.assertEqual(self.client.adapter_path, str(self.memory))

    def test_memory_adapter_format_matches_backend(self):
        self.assertEqual(self.ns["_mem_adapter"](), str(self.memory))
        self.ns["_CFG"].backend = "mlx"
        self.assertIsNone(self.ns["_mem_adapter"]())
        (self.memory / "adapters.safetensors").write_text("mlx")
        self.assertEqual(self.ns["_mem_adapter"](), str(self.memory))
        self.weight.unlink()
        self.ns["_CFG"].backend = "hf"
        self.assertIsNone(self.ns["_mem_adapter"]())


if __name__ == "__main__":
    unittest.main()
