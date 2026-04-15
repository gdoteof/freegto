from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

from .cards import DECK, hand_rank
from .config import GameConfig
from .game import (
    TERMINAL_FOLD,
    TERMINAL_NONE,
    TERMINAL_SHOWDOWN,
    NodeState,
    ParsedAction,
    active_players,
    apply_action,
    next_active_player,
    parse_action,
    to_call_for_player,
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
        self.button_idx = cfg.positions.index("BTN")
        self.bb_idx = cfg.positions.index("BB")
        self.sb_idx = self.button_idx if cfg.max_players == 2 else cfg.positions.index("SB")

    def _sample_private_and_board(self) -> Tuple[List[Combo], List[str]]:
        hands: List[Combo] = []
        used: set[str] = set()
        for position in self.cfg.positions:
            candidates = [combo for combo in self.ranges[position] if combo[0] not in used and combo[1] not in used]
            if not candidates:
                return self._sample_private_and_board()
            combo = self.rng.choice(candidates)
            hands.append(combo)
            used.update(combo)
        remain = [c for c in DECK if c not in used]
        board = self.rng.sample(remain, 5)
        return hands, board

    def _legal_actions(self, state: NodeState) -> List[ParsedAction]:
        acts = self.actions_by_street[state.street]
        to_call = to_call_for_player(state, state.player_to_act)
        if to_call == 0:
            return [a for a in acts if a.kind not in {"call", "fold"}]
        return [a for a in acts if a.kind != "check"]

    def _infoset_key(self, player: int, state: NodeState, hand: Combo, board: List[str], depth: int) -> str:
        vis_board = board[: 3 + max(0, state.street_idx - 1)] if state.street_idx > 0 else []
        hole = "".join(sorted(hand))
        active = "".join("1" if not state.folded[i] else "0" for i in range(len(state.folded)))
        street_bets = ",".join(f"{x:.2f}" for x in state.street_contributions)
        return (
            f"p{player}|{self.cfg.positions[player]}|{state.street}|{''.join(vis_board)}|{hole}|"
            f"d{depth}|tc{to_call_for_player(state, player):.2f}|pot{state.pot:.2f}|"
            f"active{active}|street{street_bets}"
        )

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

    def _terminal_payouts(self, terminal_kind: str, winner: int | None, state: NodeState,
                          hands: List[Combo], board: List[str]) -> List[float]:
        payouts = [-contrib for contrib in state.contributions]
        if terminal_kind == TERMINAL_FOLD:
            assert winner is not None
            payouts[winner] += state.pot
            return payouts
        if terminal_kind != TERMINAL_SHOWDOWN:
            raise RuntimeError("unreachable")

        remaining = [i for i in active_players(state) if state.contributions[i] > 0]
        if not remaining:
            return payouts

        hand_ranks = {i: hand_rank(hands[i], board) for i in remaining}
        contribution_levels = sorted({state.contributions[i] for i in remaining if state.contributions[i] > 0})
        previous = 0.0
        for level in contribution_levels:
            participants = [i for i, contrib in enumerate(state.contributions) if contrib >= level]
            if not participants:
                previous = level
                continue
            eligible = [i for i in remaining if state.contributions[i] >= level]
            if not eligible:
                previous = level
                continue
            pot_slice = (level - previous) * len(participants)
            best_rank = max(hand_ranks[i] for i in eligible)
            winners = [i for i in eligible if hand_ranks[i] == best_rank]
            share = pot_slice / len(winners)
            for seat in winners:
                payouts[seat] += share
            previous = level
        return payouts

    def _counterfactual_reach(self, reaches: List[float], player: int) -> float:
        value = 1.0
        for idx, reach in enumerate(reaches):
            if idx != player:
                value *= reach
        return value

    def _showdown_utility(self, state: NodeState, hands: List[Combo], board: List[str], traverser: int) -> float:
        payouts = self._terminal_payouts(TERMINAL_SHOWDOWN, None, state, hands, board)
        return payouts[traverser]

    def _cfr(self, state: NodeState, hands: List[Combo], board: List[str], traverser: int,
             reaches: List[float], depth: int = 0) -> float:
        if depth >= self.max_depth:
            return self._showdown_utility(state, hands, board, traverser)

        player = state.player_to_act
        actions = self._legal_actions(state)
        key = self._infoset_key(player, state, hands[player], board, depth)
        row = self._get_row(key, len(actions))
        strategy = self._strategy_from_regret(row.regret_sum)

        action_utils = [0.0] * len(actions)
        for i, action in enumerate(actions):
            ns, term, winner = apply_action(state, action, player)
            next_reaches = reaches.copy()
            next_reaches[player] *= strategy[i]
            if term == TERMINAL_NONE:
                util = self._cfr(ns, hands, board, traverser, next_reaches, depth + 1)
            else:
                payouts = self._terminal_payouts(term, winner, ns, hands, board)
                util = payouts[traverser]
            action_utils[i] = util

        node_util = sum(s * u for s, u in zip(strategy, action_utils))

        if player == traverser:
            opp_reach = self._counterfactual_reach(reaches, player)
            for i in range(len(actions)):
                row.regret_sum[i] += (action_utils[i] - node_util) * opp_reach

        row_weight = reaches[player] if math.isfinite(reaches[player]) else 1.0
        for i in range(len(actions)):
            row.strategy_sum[i] += strategy[i] * row_weight

        return node_util

    def _initial_state(self) -> NodeState:
        n_players = len(self.cfg.positions)
        stacks = [self.cfg.starting_stack] * n_players
        contributions = [0.0] * n_players
        street_contributions = [0.0] * n_players

        sb_post = min(self.cfg.small_blind, stacks[self.sb_idx])
        stacks[self.sb_idx] -= sb_post
        contributions[self.sb_idx] += sb_post
        street_contributions[self.sb_idx] += sb_post

        bb_post = min(self.cfg.big_blind, stacks[self.bb_idx])
        stacks[self.bb_idx] -= bb_post
        contributions[self.bb_idx] += bb_post
        street_contributions[self.bb_idx] += bb_post

        initial_cursor = self.bb_idx
        player_to_act = (self.bb_idx + 1) % n_players
        folded = [False] * n_players
        all_in = [stack == 0 for stack in stacks]
        acted = [folded[i] or all_in[i] for i in range(n_players)]
        acted[self.sb_idx] = False
        acted[self.bb_idx] = False
        if all_in[player_to_act]:
            nxt = next_active_player(
                NodeState(
                    street_idx=0,
                    player_to_act=player_to_act,
                    pot=sb_post + bb_post,
                    stacks=stacks,
                    contributions=contributions,
                    street_contributions=street_contributions,
                    folded=folded,
                    all_in=all_in,
                    acted=acted,
                    button_idx=self.button_idx,
                ),
                initial_cursor,
            )
            if nxt is not None:
                player_to_act = nxt

        return NodeState(
            street_idx=0,
            player_to_act=player_to_act,
            pot=sb_post + bb_post,
            stacks=stacks,
            contributions=contributions,
            street_contributions=street_contributions,
            folded=folded,
            all_in=all_in,
            acted=acted,
            button_idx=self.button_idx,
        )

    def train(self, iterations: int = 1000) -> None:
        n_players = len(self.cfg.positions)
        for _ in range(iterations):
            hands, board = self._sample_private_and_board()
            init = self._initial_state()
            for traverser in range(n_players):
                self._cfr(init, hands, board, traverser=traverser, reaches=[1.0] * n_players)

    def export_average_strategy(self, out_file: str | Path) -> None:
        out: Dict[str, Dict[str, float]] = {}
        for key, row in self.infosets.items():
            denom = sum(row.strategy_sum)
            if denom <= 0:
                continue
            probs = [x / denom for x in row.strategy_sum]
            out[key] = {f"a{i}": float(p) for i, p in enumerate(probs)}
        Path(out_file).write_text(json.dumps(out, indent=2), encoding="utf-8")
