from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
DECK = [r + s for r in RANKS for s in SUITS]


def rank_values(cards: Sequence[str]) -> List[int]:
    return sorted([RANK_VALUE[c[0]] for c in cards], reverse=True)


def is_straight(ranks: Iterable[int]) -> int | None:
    uniq = sorted(set(ranks), reverse=True)
    if 14 in uniq:
        uniq.append(1)
    for i in range(len(uniq) - 4):
        window = uniq[i : i + 5]
        if window[0] - window[4] == 4 and len(window) == 5:
            return window[0]
    return None


def eval_5(cards: Sequence[str]) -> Tuple[int, List[int]]:
    ranks = [RANK_VALUE[c[0]] for c in cards]
    suits = [c[1] for c in cards]
    cnt = Counter(ranks)
    by_freq = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    flush = len(set(suits)) == 1
    straight_high = is_straight(ranks)

    if flush and straight_high:
        return (8, [straight_high])

    if by_freq[0][1] == 4:
        four = by_freq[0][0]
        kicker = max(r for r in ranks if r != four)
        return (7, [four, kicker])

    if by_freq[0][1] == 3 and by_freq[1][1] == 2:
        return (6, [by_freq[0][0], by_freq[1][0]])

    if flush:
        return (5, sorted(ranks, reverse=True))

    if straight_high:
        return (4, [straight_high])

    if by_freq[0][1] == 3:
        trips = by_freq[0][0]
        kickers = sorted([r for r in ranks if r != trips], reverse=True)
        return (3, [trips] + kickers)

    if by_freq[0][1] == 2 and by_freq[1][1] == 2:
        p1, p2 = sorted([by_freq[0][0], by_freq[1][0]], reverse=True)
        kicker = max(r for r in ranks if r != p1 and r != p2)
        return (2, [p1, p2, kicker])

    if by_freq[0][1] == 2:
        pair = by_freq[0][0]
        kickers = sorted([r for r in ranks if r != pair], reverse=True)
        return (1, [pair] + kickers)

    return (0, sorted(ranks, reverse=True))


def eval_7(cards: Sequence[str]) -> Tuple[int, List[int]]:
    best = None
    for combo in combinations(cards, 5):
        val = eval_5(combo)
        if best is None or val > best:
            best = val
    assert best is not None
    return best


def hand_rank(two_cards: Sequence[str], board: Sequence[str]) -> Tuple[int, List[int]]:
    return eval_7(list(two_cards) + list(board))


def compare_hands(hero_two: Sequence[str], vill_two: Sequence[str], board: Sequence[str]) -> int:
    hero = hand_rank(hero_two, board)
    vill = hand_rank(vill_two, board)
    return (hero > vill) - (hero < vill)
