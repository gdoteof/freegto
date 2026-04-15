from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

STREETS = ("preflop", "flop", "turn", "river")


@dataclass(frozen=True)
class GameConfig:
    positions: List[str]
    starting_stack: float
    small_blind: float
    big_blind: float
    ranges: Dict[str, str]
    actions: Dict[str, List[str]]


def load_config(path: str | Path) -> GameConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    positions = raw["positions"]
    if len(positions) != 2:
        raise ValueError("Current prototype supports exactly 2 positions (heads-up).")

    for s in STREETS:
        if s not in raw["actions"]:
            raise ValueError(f"Missing actions for street: {s}")

    for p in positions:
        if p not in raw["ranges"]:
            raise ValueError(f"Missing range for position: {p}")

    return GameConfig(
        positions=positions,
        starting_stack=float(raw.get("starting_stack", 100.0)),
        small_blind=float(raw["blind_structure"]["small_blind"]),
        big_blind=float(raw["blind_structure"]["big_blind"]),
        ranges=raw["ranges"],
        actions=raw["actions"],
    )
