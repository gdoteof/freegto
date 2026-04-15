from __future__ import annotations

import unittest

from freegto.game import NodeState, TERMINAL_NONE, apply_action, parse_action


class GameStateTests(unittest.TestCase):
    def test_six_max_preflop_round_advances_to_flop(self) -> None:
        state = NodeState(
            street_idx=0,
            player_to_act=4,
            pot=5.5,
            stacks=[99.0, 99.0, 99.0, 99.0, 99.5, 99.0],
            contributions=[1.0, 1.0, 1.0, 1.0, 0.5, 1.0],
            street_contributions=[1.0, 1.0, 1.0, 1.0, 0.5, 1.0],
            folded=[False] * 6,
            all_in=[False] * 6,
            acted=[True, True, True, True, False, False],
            button_idx=3,
        )

        state, term, winner = apply_action(state, parse_action("call"), 4)
        self.assertEqual(term, TERMINAL_NONE)
        self.assertIsNone(winner)
        self.assertEqual(state.player_to_act, 5)
        self.assertEqual(state.street_idx, 0)

        state, term, winner = apply_action(state, parse_action("check"), 5)
        self.assertEqual(term, TERMINAL_NONE)
        self.assertIsNone(winner)
        self.assertEqual(state.street_idx, 1)
        self.assertEqual(state.player_to_act, 4)
        self.assertEqual(state.street_contributions, [0.0] * 6)

    def test_fold_when_facing_bet_ends_hand_if_one_player_remains(self) -> None:
        state = NodeState(
            street_idx=2,
            player_to_act=0,
            pot=10.0,
            stacks=[95.0, 95.0],
            contributions=[5.0, 5.0],
            street_contributions=[0.0, 2.0],
            folded=[False, False],
            all_in=[False, False],
            acted=[False, False],
            button_idx=0,
        )

        next_state, term, winner = apply_action(state, parse_action("fold"), 0)
        self.assertEqual(term, "terminal_fold")
        self.assertEqual(winner, 1)
        self.assertTrue(next_state.folded[0])


if __name__ == "__main__":
    unittest.main()
