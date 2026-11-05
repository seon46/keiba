"""Bet hit probabilities from the rank model, multi-point bet proposals, and a race confidence score.

The rank distribution comes from a staged Plackett-Luce order plus an independent finish probability
per horse (``ranking_v3.staged_rank_distribution``).  The same process is simulated here, so the hit
probability of any ticket -- including combinations such as exacta or trifecta that the per-horse
marginals cannot give -- is consistent with the distribution shown on the race card.

Odds are only used to rank proposals by expected value; they never enter the model.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np

# (key, Japanese name, horses per combination, ordered)
BET_TYPES = (
    ('win', '単勝', 1, False),
    ('place', '複勝', 1, False),
    ('bracket', '枠連', 2, False),
    ('quinella', '馬連', 2, False),
    ('wide', 'ワイド', 2, False),
    ('exacta', '馬単', 2, True),
    ('trio', '3連複', 3, False),
    ('trifecta', '3連単', 3, True),
)
BET_NAMES = {key: name for key, name, _, _ in BET_TYPES}
# Proposal sizes tried per bet type (number of tickets bought at 100 yen each).
MAX_POINTS = dict(win=2, place=2, bracket=6, quinella=6, wide=5, exacta=8, trio=10, trifecta=12)
CANDIDATES = 40  # combinations per type considered for proposals


def place_slots(starters):
    """How many places pay for 複勝 (JRA/NAR rule): 3 with 8+ starters, 2 with 5-7, none with 4 or fewer."""
    return 3 if starters >= 8 else 2 if starters >= 5 else 0


def available(bet, starters):
    if bet == 'place':
        return place_slots(starters) > 0
    if bet == 'bracket':
        return starters >= 9
    return starters >= (3 if bet in ('wide', 'trio', 'trifecta') else 2)


def simulate_top3(scores, completion, temperatures, *, samples=20000, seed=42):
    """(samples, 3) indices of the first three finishers (-1 where fewer horses finish)."""
    from .ranking_v3 import _staged_orders, stage_temperatures
    scores = np.asarray(scores, dtype=float)
    completion = np.clip(np.asarray(completion, dtype=float), 0, 1)
    count = len(scores)
    per_stage = stage_temperatures(temperatures, count)
    generator = np.random.default_rng(int(seed))
    parts, remaining = [], int(samples)
    while remaining:
        batch = min(remaining, 4096)
        orders = _staged_orders(scores - scores.max(), per_stage, generator, batch)
        finished = generator.random((batch, count)) < completion[orders]
        rank = np.cumsum(finished, axis=1)
        top = np.full((batch, 3), -1, dtype=np.int64)
        rows = np.arange(batch)
        for place in range(3):
            chosen = finished & (rank == place + 1)
            has = chosen.any(axis=1)
            top[has, place] = orders[rows[has], chosen[has].argmax(axis=1)]
        parts.append(top)
        remaining -= batch
    return np.concatenate(parts)


def combo_key(bet, numbers):
    """Canonical ticket key: '5', '3-7' (unordered, ascending), '7>3' (ordered)."""
    ordered = dict((key, flag) for key, _, _, flag in BET_TYPES)[bet]
    values = [int(value) for value in numbers]
    return ('>' if ordered else '-').join(str(value) for value in (values if ordered else sorted(values)))


class RaceBets:
    """Hit probabilities of every ticket of one race from simulated finishing orders."""

    def __init__(self, horse_numbers, frames, top3):
        self.numbers = np.asarray(horse_numbers, dtype=np.int64)
        self.frames = np.asarray([0 if frame is None else frame for frame in frames], dtype=np.int64)
        self.starters = len(self.numbers)
        self.top3 = np.asarray(top3)
        self.samples = len(self.top3)
        top = self.top3
        valid = top >= 0
        self.horse = np.where(valid, self.numbers[np.where(valid, top, 0)], 0)  # horse numbers, 0 = none
        self.frame = np.where(valid, self.frames[np.where(valid, top, 0)], 0)
        self.slots = place_slots(self.starters)
        self._outcome_cache = {}

    # -- outcome of each sample, as ticket keys --------------------------
    def _outcomes(self, bet):
        h, f = self.horse, self.frame
        if bet == 'win':
            return [str(a) if a else None for a in h[:, 0].tolist()]
        if bet == 'bracket':
            return [f'{min(a, b)}-{max(a, b)}' if a and b else None for a, b in zip(f[:, 0].tolist(), f[:, 1].tolist())]
        if bet == 'quinella':
            return [f'{min(a, b)}-{max(a, b)}' if a and b else None for a, b in zip(h[:, 0].tolist(), h[:, 1].tolist())]
        if bet == 'exacta':
            return [f'{a}>{b}' if a and b else None for a, b in zip(h[:, 0].tolist(), h[:, 1].tolist())]
        if bet == 'trio':
            return ['-'.join(map(str, sorted(row))) if all(row) else None for row in h.tolist()]
        if bet == 'trifecta':
            return ['>'.join(map(str, row)) if all(row) else None for row in h.tolist()]
        raise ValueError(bet)

    def _hits(self, bet, key):
        """Boolean per sample: does this ticket pay?"""
        h = self.horse
        if bet == 'place':
            return (h[:, :self.slots] == int(key)).any(axis=1)
        if bet == 'wide':
            a, b = (int(value) for value in key.split('-'))
            return (h == a).any(axis=1) & (h == b).any(axis=1)
        outcomes = self._outcome_cache.setdefault(bet, np.asarray(self._outcomes(bet), dtype=object))
        return outcomes == key


    def probabilities(self, bet):
        """{ticket key: hit probability} for every ticket that hits in at least one sample."""
        if not available(bet, self.starters):
            return {}
        if bet == 'place':
            counts = {}
            for number in self.numbers.tolist():
                counts[str(number)] = float((self.horse[:, :self.slots] == number).any(axis=1).mean())
            return {key: value for key, value in counts.items() if value > 0}
        if bet == 'wide':
            result = {}
            top = self.horse
            for a, b in combinations(sorted(self.numbers.tolist()), 2):
                value = float(((top == a).any(axis=1) & (top == b).any(axis=1)).mean())
                if value > 0:
                    result[f'{a}-{b}'] = value
            return result
        outcomes = self._outcome_cache.setdefault(bet, np.asarray(self._outcomes(bet), dtype=object))
        keys, counts = np.unique(outcomes[outcomes != None], return_counts=True)  # noqa: E711
        return {str(key): float(count) / self.samples for key, count in zip(keys.tolist(), counts.tolist())}

    def hit_rate(self, bet, keys):
        hit = np.zeros(self.samples, dtype=bool)
        for key in keys:
            hit |= self._hits(bet, key)
        return float(hit.mean())


def _odds_value(entry, conservative=True):
    """Odds as a number from a stored entry (a number, or [low, high] for 複勝・ワイド)."""
    if entry is None:
        return None
    if isinstance(entry, (list, tuple)):
        values = [value for value in entry if isinstance(value, (int, float)) and value > 0]
        if not values:
            return None
        return min(values) if conservative else sum(values) / len(values)
    return float(entry) if isinstance(entry, (int, float)) and entry > 0 else None


def allocate(odds, budget, unit=100):
    """Stakes (yen, multiples of ``unit``) so that every ticket pays about the same when it hits.

    The largest common payout target whose stakes fit in ``budget`` is used; returns
    (stakes, smallest payout, total) or None when the budget cannot buy one unit of each ticket.
    """
    odds = [float(value) for value in odds]
    if not odds or budget < unit * len(odds):
        return None

    def stakes_for(target):
        return [max(unit, unit * math.ceil(target / (unit * value) - 1e-9)) for value in odds]

    low, high = 0.0, float(budget) * max(odds)
    for _ in range(60):  # largest target whose total fits in the budget
        middle = (low + high) / 2
        if sum(stakes_for(middle)) <= budget:
            low = middle
        else:
            high = middle
    stakes = stakes_for(low)
    return stakes, min(stake * value for stake, value in zip(stakes, odds)), sum(stakes)


def propose(bets: RaceBets, odds=None, *, min_hit=.2, max_points=10, budget=1000, limit=8, per_type=1):
    """Multi-point proposals.

    For each bet type the tickets are ranked by hit probability and every leading set of 1..N
    tickets is a candidate.  Candidates below ``min_hit`` are dropped.  With odds, the ``budget`` is
    split so that every ticket pays about the same (more on the shorter odds), sets that would lose
    money even when they hit (トリガミ) are dropped, and the rest are sorted by expected return.
    Without odds every ticket gets 100 yen and the sets are sorted by hit rate.  The lower odds of
    複勝 / ワイド ranges are used.
    """
    odds = odds or {}
    proposals = []
    for bet, name, _, _ in BET_TYPES:
        probabilities = bets.probabilities(bet)
        if not probabilities:
            continue
        table = odds.get(bet) or {}
        # Only "the N most likely tickets" sets: buying long shots because their odds look generous mostly
        # buys the model's tail error, and a high hit rate was the goal.
        order = sorted(probabilities.items(), key=lambda item: -item[1])[:CANDIDATES]
        for size in range(1, min(MAX_POINTS[bet], max_points, len(order)) + 1):
            keys = tuple(key for key, _ in order[:size])
            hit = bets.hit_rate(bet, keys) if bet in ('place', 'wide') else sum(p for _, p in order[:size])
            if hit < min_hit:
                continue
            values = [_odds_value(table.get(key)) for key in keys]
            priced = all(value is not None for value in values)
            item = dict(bet=bet, name=name, points=size, hit_rate=hit, priced=priced)
            if priced:
                allocation = allocate(values, budget)
                if allocation is None:
                    continue
                stakes, payout, total = allocation
                if payout < total:  # a hit would still lose money
                    continue
                item.update(cost=total, stakes=stakes, min_payout=round(payout),
                            max_payout=round(max(stake * value for stake, value in zip(stakes, values))),
                            expected_return=sum(probabilities[key] * stake * value
                                                for key, stake, value in zip(keys, stakes, values)) / total,
                            synthetic_odds=1 / sum(1 / value for value in values))
            else:
                stakes = [100] * size
                item.update(cost=100 * size, stakes=stakes, min_payout=None, max_payout=None, expected_return=None,
                            synthetic_odds=None)
            item['tickets'] = [dict(key=key, probability=probabilities[key], odds=table.get(key), stake=stake)
                               for key, stake in zip(keys, stakes)]
            proposals.append(item)
    proposals.sort(key=lambda item: (item['expected_return'] is None, -(item['expected_return'] or 0),
                                     -item['hit_rate'], item['points']))
    chosen, used = [], {}
    for item in proposals:
        if used.get(item['bet'], 0) >= per_type:
            continue
        used[item['bet']] = used.get(item['bet'], 0) + 1
        chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen


def concentration(win_probabilities):
    """0 (all horses equal) .. 1 (one certain winner): 1 - entropy / log(starters)."""
    values = np.asarray([value for value in win_probabilities if value is not None and value > 0], dtype=float)
    count = len(list(win_probabilities))
    if count < 2 or not len(values):
        return None
    values = values / values.sum()
    entropy = float(-(values * np.log(values)).sum())
    return float(np.clip(1 - entropy / math.log(count), 0, 1))
