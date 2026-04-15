from __future__ import annotations

from itertools import combinations, product
from typing import List, Set, Tuple

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}

Card = str
Combo = Tuple[Card, Card]


def canonical_combo(c1: Card, c2: Card) -> Combo:
    return tuple(sorted((c1, c2)))  # type: ignore[return-value]


def all_pair_combos(rank: str) -> Set[Combo]:
    cards = [rank + s for s in SUITS]
    return {canonical_combo(a, b) for a, b in combinations(cards, 2)}


def suited_combos(r1: str, r2: str) -> Set[Combo]:
    return {canonical_combo(r1 + s, r2 + s) for s in SUITS}


def offsuit_combos(r1: str, r2: str) -> Set[Combo]:
    out: Set[Combo] = set()
    for s1, s2 in product(SUITS, repeat=2):
        if s1 == s2:
            continue
        out.add(canonical_combo(r1 + s1, r2 + s2))
    return out


def exact_combos(r1: str, r2: str) -> Set[Combo]:
    if r1 == r2:
        return all_pair_combos(r1)
    return suited_combos(r1, r2) | offsuit_combos(r1, r2)


def expand_token(token: str) -> Set[Combo]:
    token = token.strip()
    if not token:
        return set()

    if len(token) == 2 and token[0] == token[1]:
        return all_pair_combos(token[0])

    if len(token) == 3 and token[0] != token[1] and token[2] in ("s", "o"):
        return suited_combos(token[0], token[1]) if token[2] == "s" else offsuit_combos(token[0], token[1])

    if token.endswith("+"):
        base = token[:-1]
        if len(base) == 2 and base[0] == base[1]:
            start = RANK_TO_IDX[base[0]]
            out: Set[Combo] = set()
            for idx in range(start, len(RANKS)):
                out |= all_pair_combos(RANKS[idx])
            return out
        if len(base) == 3 and base[0] != base[1] and base[2] in ("s", "o"):
            hi, lo, typ = base[0], base[1], base[2]
            hi_idx, lo_idx = RANK_TO_IDX[hi], RANK_TO_IDX[lo]
            out: Set[Combo] = set()
            for idx in range(lo_idx, hi_idx):
                if typ == "s":
                    out |= suited_combos(hi, RANKS[idx])
                else:
                    out |= offsuit_combos(hi, RANKS[idx])
            return out

    if len(token) == 2 and token[0] != token[1]:
        return exact_combos(token[0], token[1])

    raise ValueError(f"Unsupported range token: {token}")


def parse_range(range_text: str) -> List[Combo]:
    combos: Set[Combo] = set()
    for token in range_text.split(","):
        combos |= expand_token(token)
    return sorted(combos)
