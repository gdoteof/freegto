from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from freegto.cfr import CFRSolver
from freegto.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class SolverTests(unittest.TestCase):
    def test_six_max_solver_runs_and_exports_strategy(self) -> None:
        cfg = load_config(ROOT / "examples/6max_nlhe.json")
        solver = CFRSolver(cfg, device="cpu", max_depth=2, seed=1)
        solver.train(iterations=1)
        self.assertTrue(solver.infosets)

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
            out_path = Path(handle.name)

        try:
            solver.export_average_strategy(out_path)
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(data)
        finally:
            out_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
