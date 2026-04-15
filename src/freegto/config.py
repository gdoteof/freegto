from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

STREETS = ("preflop", "flop", "turn", "river")


@dataclass(frozen=True)
class GameConfig:
    positions: List[str]
    max_players: int
    starting_stack: float
    small_blind: float
    big_blind: float
    ranges: Dict[str, str]
    actions: Dict[str, List[str]]


def load_config(path: str | Path) -> GameConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    positions = raw["positions"]
    max_players = int(raw.get("max_players", len(positions)))
    if max_players not in {2, 6, 9}:
        raise ValueError("Supported max_players values are 2, 6, or 9.")
    if len(positions) != max_players:
        raise ValueError("The number of configured positions must match max_players.")
    if max_players < len(positions):
        raise ValueError("max_players cannot be smaller than the number of configured positions.")
    if len(set(positions)) != len(positions):
        raise ValueError("Position names must be unique.")
    if "BTN" not in positions or "BB" not in positions:
        raise ValueError("Positions must include BTN and BB.")
    if max_players > 2 and "SB" not in positions:
        raise ValueError("Multi-player tables must include SB.")

    for s in STREETS:
        if s not in raw["actions"]:
            raise ValueError(f"Missing actions for street: {s}")

    for p in positions:
        if p not in raw["ranges"]:
            raise ValueError(f"Missing range for position: {p}")

    return GameConfig(
        positions=positions,
        max_players=max_players,
        starting_stack=float(raw.get("starting_stack", 100.0)),
        small_blind=float(raw["blind_structure"]["small_blind"]),
        big_blind=float(raw["blind_structure"]["big_blind"]),
        ranges=raw["ranges"],
        actions=raw["actions"],
    )
