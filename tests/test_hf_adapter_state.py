"""HF adapter lifecycle checks using stdlib stubs; no ML runtime is imported."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from m0.hf import HFClient


class StubModel:
    def __init__(self, weights):
        self.weights = weights
        self.adapters = {}
        self.active = None
        self.enabled = False
        self.base_model = self

    def load_adapter(self, path, adapter_name):
        if path not in self.weights:
            raise FileNotFoundError(path)
        self.adapters[adapter_name] = (path, self.weights[path])

    def delete_adapter(self, name):
        del self.adapters[name]

    def set_adapter(self, name):
        self.active = name

    def enable_adapter_layers(self):
        self.enabled = True

    def disable_adapter_layers(self):
        self.enabled = False

    def output(self):
        if not self.enabled:
            return "base"
        path, version = self.adapters[self.active]
        return f"{path}:{version}"


class StubPeftModel:
    @staticmethod
    def from_pretrained(model, path, adapter_name, is_trainable):
        model.load_adapter(path, adapter_name)
        return model


class HFAdapterStateTests(unittest.TestCase):
    def setUp(self):
        HFClient._registry.clear()
        self.x, self.y = map(os.path.abspath, ("adapter-x", "adapter-y"))
        self.weights = {self.x: 1, self.y: 1}
        self.modules = patch.dict(sys.modules, {
            "peft": SimpleNamespace(PeftModel=StubPeftModel),
            "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(HFClient._registry.clear)
        self.sig = patch("m0.hf._adapter_sig", side_effect=self.weights.get)
        self.sig.start()
        self.addCleanup(self.sig.stop)

    def client(self, adapter=None):
        client = HFClient(SimpleNamespace(
            hf_model_path="stub-base", hf_quant="bf16", hf_adapter_path=adapter,
        ))
        client._load_base = lambda: {
            "model": StubModel(self.weights), "tok": None, "adapters": {},
            "sigs": {}, "active": None, "peft": False, "aseq": 0,
        }
        client._generate_text = lambda messages: client._entry["model"].output()
        return client

    def test_each_client_restores_its_adapter_for_chat_and_generate(self):
        a, b = self.client(self.x), self.client()
        self.assertEqual(a.generate("probe"), f"{self.x}:1")
        self.assertEqual(b.chat([])["content"], "base")
        b.set_adapter(self.y)
        self.assertEqual(a.chat([])["content"], f"{self.x}:1")
        self.assertEqual(b.generate("probe"), f"{self.y}:1")
        self.assertIs(a._entry, b._entry)

    def test_other_client_reattaches_after_shared_unload(self):
        a, b = self.client(self.x), self.client(self.y)
        a.generate("probe")
        b.generate("probe")
        old_entry = b._entry
        a.unload()
        self.assertEqual(a.generate("probe"), f"{self.x}:1")
        self.assertEqual(b.generate("probe"), f"{self.y}:1")
        self.assertIsNot(b._entry, old_entry)
        self.assertIs(a._entry, b._entry)

    def test_disable_after_unload_does_not_reload_missing_adapter(self):
        client = self.client(self.x)
        client.generate("probe")
        client.unload()
        del self.weights[self.x]
        client.set_adapter(None)
        self.assertEqual(client.generate("probe"), "base")
        self.assertFalse(client._entry["peft"])

    def test_replace_after_unload_does_not_reload_missing_adapter(self):
        client = self.client(self.x)
        client.generate("probe")
        client.unload()
        del self.weights[self.x]
        client.set_adapter(self.y)
        self.assertEqual(client.generate("probe"), f"{self.y}:1")

    def test_generation_refreshes_weights_rewritten_at_same_path(self):
        client = self.client(self.x)
        self.assertEqual(client.generate("probe"), f"{self.x}:1")
        self.weights[self.x] = 2
        self.assertEqual(client.generate("probe"), f"{self.x}:2")


if __name__ == "__main__":
    unittest.main()
