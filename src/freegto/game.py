from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import STREETS


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    size: float = 0.0


def parse_action(a: str) -> ParsedAction:
    t = a.strip().lower()
    if t.startswith("bet:"):
        return ParsedAction("bet", float(t.split(":", 1)[1]))
    if t in {"check", "call", "fold", "all-in", "allin"}:
        return ParsedAction("all-in" if t in {"all-in", "allin"} else t)
    raise ValueError(f"Unsupported action: {a}")


@dataclass
class NodeState:
    street_idx: int
    player_to_act: int
    pot: float
    to_call: float
    stacks: List[float]
    contributions: List[float]
    last_aggressor: int | None
    checks_in_row: int

    @property
    def street(self) -> str:
        return STREETS[self.street_idx]


TERMINAL_FOLD = "terminal_fold"
TERMINAL_SHOWDOWN = "terminal_showdown"
TERMINAL_NONE = "none"


def apply_action(s: NodeState, action: ParsedAction, acting: int) -> tuple[NodeState, str, int | None]:
    opp = 1 - acting
    ns = NodeState(
        street_idx=s.street_idx,
        player_to_act=opp,
        pot=s.pot,
        to_call=s.to_call,
        stacks=s.stacks.copy(),
        contributions=s.contributions.copy(),
        last_aggressor=s.last_aggressor,
        checks_in_row=s.checks_in_row,
    )

    if action.kind == "fold":
        return ns, TERMINAL_FOLD, opp

    if action.kind == "check":
        if ns.to_call > 0:
            return ns, TERMINAL_FOLD, opp
        ns.checks_in_row += 1
        if ns.checks_in_row >= 2:
            if ns.street_idx == 3:
                return ns, TERMINAL_SHOWDOWN, None
            ns.street_idx += 1
            ns.player_to_act = 0
            ns.to_call = 0.0
            ns.checks_in_row = 0
            ns.last_aggressor = None
        return ns, TERMINAL_NONE, None

    if action.kind == "call":
        call_amt = min(ns.to_call, ns.stacks[acting])
        ns.stacks[acting] -= call_amt
        ns.contributions[acting] += call_amt
        ns.pot += call_amt
        ns.to_call = 0.0
        ns.checks_in_row = 0

        if ns.street_idx == 3:
            return ns, TERMINAL_SHOWDOWN, None
        ns.street_idx += 1
        ns.player_to_act = 0
        ns.last_aggressor = None
        return ns, TERMINAL_NONE, None

    if action.kind == "all-in":
        amount = ns.stacks[acting]
        ns.stacks[acting] = 0.0
        ns.contributions[acting] += amount
        ns.pot += amount
        ns.to_call = max(0.0, ns.contributions[acting] - ns.contributions[opp])
        ns.last_aggressor = acting
        ns.checks_in_row = 0
        return ns, TERMINAL_NONE, None

    if action.kind == "bet":
        target_total = ns.pot * action.size
        put_in = min(ns.stacks[acting], max(0.0, target_total))
        ns.stacks[acting] -= put_in
        ns.contributions[acting] += put_in
        ns.pot += put_in
        ns.to_call = max(0.0, ns.contributions[acting] - ns.contributions[opp])
        ns.last_aggressor = acting
        ns.checks_in_row = 0
        return ns, TERMINAL_NONE, None

    raise RuntimeError("unreachable")
