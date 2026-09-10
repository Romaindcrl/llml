"""Ensure incomplete v2 archives cannot silently become experimental results."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_matrix", ROOT / "eval/v2/run_matrix.py")
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


class MatrixInputTests(unittest.TestCase):
    def test_missing_tasks_fail_before_creating_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            run = subprocess.run(
                [sys.executable, str(ROOT / "eval/v2/run_matrix.py"), "--config", "C0",
                 "--tasks-dir", str(Path(tmp) / "missing"), "--outdir", str(output)],
                capture_output=True, text=True, timeout=10,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("No frozen tasks", run.stderr)
            self.assertNotIn("MATRIX_C0_DONE", run.stdout)
            self.assertFalse(output.exists())

    def test_duplicate_incidents_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = json.dumps({"task_id": "repo:one", "repo": "repo"})
            (Path(tmp) / "repo.jsonl").write_text(row + "\n" + row + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate task_id"):
                MATRIX.load_tasks(tmp, None)

    def test_missing_arm_inputs_cannot_fall_back_to_base(self):
        tasks = [{"task_id": "repo:one", "repo": "repo"}]
        MATRIX.validate_inputs("C0", tasks, {}, {}, {})
        for config in ("C_ctx", "C_lora", "C_both", "C_wrong"):
            with self.subTest(config=config), self.assertRaises(ValueError):
                MATRIX.validate_inputs(config, tasks, {}, {}, {})
        with self.assertRaisesRegex(ValueError, "non-distinct"):
            MATRIX.validate_inputs("C_wrong", tasks, {}, {}, {"repo": "repo"})


if __name__ == "__main__":
    unittest.main()
