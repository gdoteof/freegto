from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

from .cards import DECK, compare_hands
from .config import GameConfig
from .game import (
    TERMINAL_FOLD,
    TERMINAL_NONE,
    TERMINAL_SHOWDOWN,
    NodeState,
    ParsedAction,
    apply_action,
    parse_action,
)
from .range_parser import Combo, parse_range


@dataclass
class InfoSetRow:
    regret_sum: List[float]
    strategy_sum: List[float]


class CFRSolver:
    def __init__(self, cfg: GameConfig, device: str = "cpu", seed: int = 7, max_depth: int = 10):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.requested_device = device
        self.use_torch_cuda = bool(torch is not None and device == "cuda" and torch.cuda.is_available())
        self.device = "cuda" if self.use_torch_cuda else "cpu"
        self.max_depth = max_depth
        self.actions_by_street: Dict[str, List[ParsedAction]] = {
            st: [parse_action(a) for a in cfg.actions[st]] for st in cfg.actions
        }
        self.ranges: Dict[str, List[Combo]] = {p: parse_range(cfg.ranges[p]) for p in cfg.positions}
        self.infosets: Dict[str, InfoSetRow] = {}

    def _sample_private_and_board(self) -> Tuple[List[Combo], List[str]]:
        p0 = self.rng.choice(self.ranges[self.cfg.positions[0]])
        p1 = self.rng.choice(self.ranges[self.cfg.positions[1]])
        used = set(p0) | set(p1)
        if len(used) < 4:
            return self._sample_private_and_board()
        remain = [c for c in DECK if c not in used]
        board = self.rng.sample(remain, 5)
        return [p0, p1], board

    def _legal_actions(self, state: NodeState) -> List[ParsedAction]:
        acts = self.actions_by_street[state.street]
        if state.to_call == 0:
            return [a for a in acts if a.kind not in {"call", "fold"}]
        return [a for a in acts if a.kind != "check"]

    def _infoset_key(self, player: int, state: NodeState, hand: Combo, board: List[str], depth: int) -> str:
        vis_board = board[: 3 + max(0, state.street_idx - 1)] if state.street_idx > 0 else []
        hole = "".join(sorted(hand))
        return f"p{player}|{state.street}|{''.join(vis_board)}|{hole}|d{depth}|tc{state.to_call:.2f}|pot{state.pot:.2f}"

    def _get_row(self, key: str, n_actions: int) -> InfoSetRow:
        row = self.infosets.get(key)
        if row is None:
            row = InfoSetRow(regret_sum=[0.0] * n_actions, strategy_sum=[0.0] * n_actions)
            self.infosets[key] = row
        return row

    @staticmethod
    def _strategy_from_regret(regret_sum: List[float]) -> List[float]:
        pos = [max(r, 0.0) for r in regret_sum]
        denom = sum(pos)
        if denom > 0:
            return [x / denom for x in pos]
        return [1.0 / len(pos)] * len(pos)

    def _terminal_utility(self, terminal_kind: str, winner: int | None, traverser: int, pot: float,
                          hands: List[Combo], board: List[str]) -> float:
        if terminal_kind == TERMINAL_FOLD:
            assert winner is not None
            return pot if winner == traverser else -pot
        if terminal_kind == TERMINAL_SHOWDOWN:
            cmp = compare_hands(hands[0], hands[1], board)
            if cmp == 0:
                return 0.0
            if (cmp > 0 and traverser == 0) or (cmp < 0 and traverser == 1):
                return pot
            return -pot
        raise RuntimeError("unreachable")

    def _cfr(self, state: NodeState, hands: List[Combo], board: List[str], traverser: int, reach0: float, reach1: float,
             depth: int = 0) -> float:
        if depth >= self.max_depth:
            cmp = compare_hands(hands[0], hands[1], board)
            if cmp == 0:
                return 0.0
            return state.pot if ((cmp > 0 and traverser == 0) or (cmp < 0 and traverser == 1)) else -state.pot

        player = state.player_to_act
        actions = self._legal_actions(state)
        key = self._infoset_key(player, state, hands[player], board, depth)
        row = self._get_row(key, len(actions))
        strategy = self._strategy_from_regret(row.regret_sum)

        action_utils = [0.0] * len(actions)

        for i, a in enumerate(actions):
            ns, term, winner = apply_action(state, a, player)
            if term == TERMINAL_NONE:
                util = self._cfr(
                    ns,
                    hands,
                    board,
                    traverser,
                    reach0 * (strategy[i] if player == 0 else 1.0),
                    reach1 * (strategy[i] if player == 1 else 1.0),
                    depth + 1,
                )
            else:
                util = self._terminal_utility(term, winner, traverser, ns.pot, hands, board)
            action_utils[i] = util

        node_util = sum(s * u for s, u in zip(strategy, action_utils))

        if player == traverser:
            opp_reach = reach1 if player == 0 else reach0
            for i in range(len(actions)):
                row.regret_sum[i] += (action_utils[i] - node_util) * opp_reach

        self_reach = reach0 if player == 0 else reach1
        for i in range(len(actions)):
            row.strategy_sum[i] += strategy[i] * self_reach

        return node_util

    def train(self, iterations: int = 1000) -> None:
        for _ in range(iterations):
            hands, board = self._sample_private_and_board()
            init = NodeState(
                street_idx=0,
                player_to_act=0,
                pot=self.cfg.small_blind + self.cfg.big_blind,
                to_call=self.cfg.big_blind - self.cfg.small_blind,
                stacks=[self.cfg.starting_stack - self.cfg.small_blind, self.cfg.starting_stack - self.cfg.big_blind],
                contributions=[self.cfg.small_blind, self.cfg.big_blind],
                last_aggressor=None,
                checks_in_row=0,
            )
            self._cfr(init, hands, board, traverser=0, reach0=1.0, reach1=1.0)
            self._cfr(init, hands, board, traverser=1, reach0=1.0, reach1=1.0)

    def export_average_strategy(self, out_file: str | Path) -> None:
        out: Dict[str, Dict[str, float]] = {}
        for key, row in self.infosets.items():
            denom = sum(row.strategy_sum)
            if denom <= 0:
                continue
            probs = [x / denom for x in row.strategy_sum]
            out[key] = {f"a{i}": float(p) for i, p in enumerate(probs)}
        Path(out_file).write_text(json.dumps(out, indent=2), encoding="utf-8")
