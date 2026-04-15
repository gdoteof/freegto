from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from freegto.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_example_configs_load_with_expected_player_counts(self) -> None:
        cases = [
            ("examples/hu_nlhe.json", 2),
            ("examples/6max_nlhe.json", 6),
            ("examples/9max_nlhe.json", 9),
        ]
        for rel_path, expected_players in cases:
            with self.subTest(path=rel_path):
                cfg = load_config(ROOT / rel_path)
                self.assertEqual(cfg.max_players, expected_players)
                self.assertEqual(len(cfg.positions), expected_players)

    def test_rejects_position_count_mismatch(self) -> None:
        bad_config = {
            "max_players": 6,
            "positions": ["BTN", "BB"],
            "starting_stack": 100,
            "blind_structure": {"small_blind": 0.5, "big_blind": 1.0},
            "ranges": {
                "BTN": "22+,A2s+,KTo+",
                "BB": "22+,A2s+,KTo+",
            },
            "actions": {
                "preflop": ["fold", "call", "bet:2.5", "all-in"],
                "flop": ["check", "fold", "call", "bet:0.25", "all-in"],
                "turn": ["check", "fold", "call", "bet:0.5", "all-in"],
                "river": ["check", "fold", "call", "bet:0.75", "all-in"],
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(bad_config, handle)
            temp_path = Path(handle.name)

        try:
            with self.assertRaisesRegex(ValueError, "must match max_players"):
                load_config(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
