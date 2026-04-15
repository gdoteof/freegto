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
    stacks: List[float]
    contributions: List[float]
    street_contributions: List[float]
    folded: List[bool]
    all_in: List[bool]
    acted: List[bool]
    button_idx: int

    @property
    def street(self) -> str:
        return STREETS[self.street_idx]


TERMINAL_FOLD = "terminal_fold"
TERMINAL_SHOWDOWN = "terminal_showdown"
TERMINAL_NONE = "none"


def active_players(state: NodeState) -> List[int]:
    return [i for i, folded in enumerate(state.folded) if not folded]


def players_contesting_pot(state: NodeState) -> List[int]:
    return [i for i in active_players(state) if state.contributions[i] > 0 or state.stacks[i] > 0 or state.all_in[i]]


def current_bet(state: NodeState) -> float:
    live = [state.street_contributions[i] for i in range(len(state.folded)) if not state.folded[i]]
    return max(live, default=0.0)


def to_call_for_player(state: NodeState, player: int) -> float:
    if state.folded[player] or state.all_in[player]:
        return 0.0
    return max(0.0, current_bet(state) - state.street_contributions[player])


def is_eligible_to_act(state: NodeState, player: int) -> bool:
    return not state.folded[player] and not state.all_in[player]


def next_active_player(state: NodeState, start_idx: int) -> int | None:
    n_players = len(state.folded)
    for step in range(1, n_players + 1):
        idx = (start_idx + step) % n_players
        if is_eligible_to_act(state, idx):
            return idx
    return None


def only_one_player_left(state: NodeState) -> bool:
    return len(active_players(state)) == 1


def everyone_all_in_or_matched(state: NodeState) -> bool:
    live = [i for i in range(len(state.folded)) if not state.folded[i] and not state.all_in[i]]
    if not live:
        return True
    target = current_bet(state)
    return all(state.acted[i] and state.street_contributions[i] == target for i in live)


def clone_state(state: NodeState) -> NodeState:
    return NodeState(
        street_idx=state.street_idx,
        player_to_act=state.player_to_act,
        pot=state.pot,
        stacks=state.stacks.copy(),
        contributions=state.contributions.copy(),
        street_contributions=state.street_contributions.copy(),
        folded=state.folded.copy(),
        all_in=state.all_in.copy(),
        acted=state.acted.copy(),
        button_idx=state.button_idx,
    )


def advance_street(state: NodeState) -> tuple[NodeState, str, int | None]:
    ns = clone_state(state)
    if ns.street_idx == len(STREETS) - 1:
        return ns, TERMINAL_SHOWDOWN, None
    ns.street_idx += 1
    ns.street_contributions = [0.0] * len(ns.street_contributions)
    ns.acted = [not is_eligible_to_act(ns, i) for i in range(len(ns.folded))]
    first_to_act = next_active_player(ns, ns.button_idx)
    if first_to_act is None:
        return ns, TERMINAL_SHOWDOWN, None
    ns.player_to_act = first_to_act
    return ns, TERMINAL_NONE, None


def finalize_action(state: NodeState, acting: int) -> tuple[NodeState, str, int | None]:
    if only_one_player_left(state):
        winner = active_players(state)[0]
        return state, TERMINAL_FOLD, winner

    if everyone_all_in_or_matched(state):
        return advance_street(state)

    nxt = next_active_player(state, acting)
    if nxt is None:
        return state, TERMINAL_SHOWDOWN, None
    state.player_to_act = nxt
    return state, TERMINAL_NONE, None


def apply_action(s: NodeState, action: ParsedAction, acting: int) -> tuple[NodeState, str, int | None]:
    ns = clone_state(s)
    if acting != ns.player_to_act:
        raise ValueError("Action applied for non-acting player.")

    to_call = to_call_for_player(ns, acting)

    if action.kind == "fold":
        if to_call == 0:
            raise ValueError("Cannot fold when checking is available.")
        ns.folded[acting] = True
        ns.acted[acting] = True
        return finalize_action(ns, acting)

    if action.kind == "check":
        if to_call > 0:
            raise ValueError("Cannot check when facing a bet.")
        ns.acted[acting] = True
        return finalize_action(ns, acting)

    if action.kind == "call":
        if to_call <= 0:
            raise ValueError("Cannot call when there is nothing to call.")
        call_amt = min(to_call, ns.stacks[acting])
        ns.stacks[acting] -= call_amt
        ns.contributions[acting] += call_amt
        ns.street_contributions[acting] += call_amt
        ns.pot += call_amt
        ns.acted[acting] = True
        if ns.stacks[acting] == 0:
            ns.all_in[acting] = True
        return finalize_action(ns, acting)

    if action.kind == "all-in":
        amount = ns.stacks[acting]
        if amount <= 0:
            raise ValueError("Cannot go all-in with no chips remaining.")
        previous_bet = current_bet(ns)
        ns.stacks[acting] = 0.0
        ns.contributions[acting] += amount
        ns.street_contributions[acting] += amount
        ns.pot += amount
        ns.all_in[acting] = True
        ns.acted[acting] = True
        if ns.street_contributions[acting] > previous_bet:
            ns.acted = [ns.folded[i] or ns.all_in[i] for i in range(len(ns.acted))]
            ns.acted[acting] = True
        return finalize_action(ns, acting)

    if action.kind == "bet":
        previous_bet = current_bet(ns)
        raise_to = max(previous_bet, ns.pot * action.size)
        target_contribution = max(raise_to, ns.street_contributions[acting] + to_call)
        put_in = min(ns.stacks[acting], max(0.0, target_contribution - ns.street_contributions[acting]))
        if put_in <= 0:
            raise ValueError("Bet action must put chips into the pot.")
        ns.stacks[acting] -= put_in
        ns.contributions[acting] += put_in
        ns.street_contributions[acting] += put_in
        ns.pot += put_in
        ns.acted[acting] = True
        if ns.stacks[acting] == 0:
            ns.all_in[acting] = True
        if ns.street_contributions[acting] > previous_bet:
            ns.acted = [ns.folded[i] or ns.all_in[i] for i in range(len(ns.acted))]
            ns.acted[acting] = True
        return finalize_action(ns, acting)

    raise RuntimeError("unreachable")
