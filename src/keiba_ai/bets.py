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


def combine(bets: RaceBets, odds, *, min_hit=.2, max_points=10, budget=1000, limit=5, pool=14):
    """Proposals that mix two or three bet types (e.g. 単勝 + 馬連 + 3連単).

    Parts are the "N most likely tickets" sets of each priced bet type.  The budget is split over all
    tickets of the mix so that any single ticket that hits pays back at least the total (so a hit
    never loses money even when only the cheapest part hits); hits of several parts add up.  Hit
    rate, payouts and expected return come from the simulated finishing orders.
    """
    from itertools import combinations as choose
    if not odds:
        return []
    parts = []
    for bet, name, _, _ in BET_TYPES:
        table = odds.get(bet) or {}
        probabilities = bets.probabilities(bet)
        if not table or not probabilities:
            continue
        order = sorted(probabilities.items(), key=lambda item: -item[1])[:CANDIDATES]
        for size in range(1, min(MAX_POINTS[bet], max_points - 1, len(order)) + 1):
            keys = [key for key, _ in order[:size]]
            values = [_odds_value(table.get(key)) for key in keys]
            if any(value is None for value in values):
                break
            hits = np.stack([bets._hits(bet, key) for key in keys])
            parts.append(dict(bet=bet, name=name, keys=keys, values=values, hits=hits,
                              probabilities=[probabilities[key] for key in keys]))
    if len(parts) < 2:
        return []

    def evaluate(members):
        tickets = [(part, index) for part in members for index in range(len(part['keys']))]
        if len(tickets) > max_points:
            return None
        values = [part['values'][index] for part, index in tickets]
        allocation = allocate(values, budget)
        if allocation is None or allocation[1] < allocation[2]:
            return None
        stakes, smallest, total = allocation
        payout = np.zeros(bets.samples)
        for (part, index), stake, value in zip(tickets, stakes, values):
            payout += part['hits'][index] * stake * value
        hit = payout > 0
        if hit.mean() < min_hit:
            return None
        position, rows = 0, []
        for part in members:
            size = len(part['keys'])
            share = stakes[position:position + size]
            rows.append(dict(bet=part['bet'], name=part['name'], cost=sum(share), tickets=[
                dict(key=key, probability=probability, odds=(odds[part['bet']] or {}).get(key), stake=stake)
                for key, probability, stake in zip(part['keys'], part['probabilities'], share)]))
            position += size
        return dict(parts=rows, points=len(tickets), cost=total, hit_rate=float(hit.mean()),
                    expected_return=float(payout.mean() / total), min_payout=round(float(payout[hit].min())),
                    max_payout=round(float(payout.max())), profit_rate=float((payout > total).mean()))

    singles = []
    for part in parts:
        result = evaluate([part])
        if result is not None:
            singles.append((part, result))
    singles.sort(key=lambda item: -item[1]['expected_return'])
    top = [part for part, _ in singles[:pool]]
    mixes = []
    for count in (2, 3):
        for members in choose(top, count):
            if len({part['bet'] for part in members}) < count:
                continue
            result = evaluate(list(members))
            if result is not None:
                mixes.append(result)
    mixes.sort(key=lambda item: (-item['expected_return'], -item['hit_rate']))
    chosen, seen = [], set()
    for item in mixes:
        kinds = tuple(part['bet'] for part in item['parts'])
        if kinds in seen:
            continue
        seen.add(kinds)
        chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen


# Race confidence = the top pick's win probability.  On the v4 evaluation year (2025, experiment_confidence with
# data/v4_check2) it sorted races by "the pick finishes in the top 3" better than the concentration of all win
# probabilities (Spearman .278/.288 vs .258/.274 for central/local) and better than model disagreement, seed spread,
# unknown pedigree or field size.  The thresholds are the year's tertiles; A/B/C are equal thirds of that year and the
# rates are what the picks did there.  None of the tiers made money on 単勝 at the final odds: confidence is how
# likely the pick is to hit, not a betting edge.
CONFIDENCE_TIERS = {
    'central': dict(thresholds=(.2483, .3757), races=3243, rates={
        'A': dict(pick_win_rate=.417, pick_top3_rate=.777), 'B': dict(pick_win_rate=.285, pick_top3_rate=.618),
        'C': dict(pick_win_rate=.175, pick_top3_rate=.461)}),
    'local': dict(thresholds=(.3110, .4684), races=12958, rates={
        'A': dict(pick_win_rate=.589, pick_top3_rate=.882), 'B': dict(pick_win_rate=.367, pick_top3_rate=.751),
        'C': dict(pick_win_rate=.266, pick_top3_rate=.598)}),
}


# Whole-race confidence = how predictable the top-3 finishing order is to the model itself:
# 1 - entropy(1st-2nd-3rd order) / log(n(n-1)(n-2)).  When the model is calibrated this is the expected "order skill"
# of the race, and on the v4 evaluation year (2025, data/v4_check2) it was: Spearman with the realised order skill
# .375 central / .435 local, ahead of the win-probability concentration (.356/.415), the pick's win probability
# (.283/.342) and odds-free models combining all candidates (.318/.429).  The tier means matched the realised skill
# (central A .270 expected vs .237 realised, B .154/.147, C .079/.086).  Thresholds are that year's tertiles; the rates
# are what those races did: how many of the three most likely top-3 finishers ran in the top three, and the pick.
OVERALL_TIERS = {
    'central': dict(thresholds=(.1182, .1948), races=3243, rates={
        'A': dict(top3_hits=1.77, order_skill=.237, pick_top3_rate=.756, pick_win_rate=.402),
        'B': dict(top3_hits=1.49, order_skill=.147, pick_top3_rate=.614, pick_win_rate=.276),
        'C': dict(top3_hits=1.22, order_skill=.086, pick_top3_rate=.486, pick_win_rate=.200)}),
    'local': dict(thresholds=(.1500, .2449), races=12958, rates={
        'A': dict(top3_hits=2.05, order_skill=.303, pick_top3_rate=.869, pick_win_rate=.558),
        'B': dict(top3_hits=1.78, order_skill=.194, pick_top3_rate=.749, pick_win_rate=.384),
        'C': dict(top3_hits=1.51, order_skill=.106, pick_top3_rate=.614, pick_win_rate=.281)}),
}


def top3_orders(scores, temperatures):
    """P[i, j, k] that i, j, k finish 1st, 2nd, 3rd under the staged Plackett-Luce model the rank distribution
    samples (score / temperature of each position; non-finishing ignored).  Zero on repeated horses."""
    from .ranking_v3 import stage_temperatures
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    t = stage_temperatures(temperatures, n)
    scores = scores - scores.max()
    w1, w2, w3 = (np.exp(scores / t[position]) for position in range(3))
    eye = np.eye(n, dtype=bool)
    first = w1 / w1.sum()
    second = np.where(eye, 0.0, w2[None, :]) / (w2.sum() - w2)[:, None]
    rest3 = w3.sum() - w3[:, None] - w3[None, :]
    third = w3[None, None, :] / np.where(eye, np.inf, rest3)[:, :, None]
    distinct = ~(eye[:, :, None] | eye[:, None, :] | eye[None, :, :])
    return first[:, None, None] * second[:, :, None] * np.where(distinct, third, 0.0)


def order_certainty(scores, temperatures):
    """1 - entropy(top-3 order) / log(n(n-1)(n-2)); None for fewer than three runners."""
    if len(scores) < 3:
        return None
    n = len(scores)
    p = top3_orders(scores, temperatures).ravel()
    p = p[p > 0]
    return float(1 + (p * np.log(p)).sum() / math.log(n * (n - 1) * (n - 2)))


def overall_tier(region, value):
    low, high = OVERALL_TIERS['central' if region == 'central' else 'local']['thresholds']
    return 'A' if value >= high else 'B' if value >= low else 'C'


def top_win(win_probabilities):
    """The largest win probability (renormalised over the known runners); None for fewer than two runners."""
    values = [value for value in win_probabilities if value is not None]
    if len(values) < 2 or sum(values) <= 0:
        return None
    return float(max(values) / sum(values))


def confidence_tier(region, value):
    low, high = CONFIDENCE_TIERS['central' if region == 'central' else 'local']['thresholds']
    return 'A' if value >= high else 'B' if value >= low else 'C'


def concentration(win_probabilities):
    """0 (all horses equal) .. 1 (one certain winner): 1 - entropy / log(starters)."""
    values = np.asarray([value for value in win_probabilities if value is not None and value > 0], dtype=float)
    count = len(list(win_probabilities))
    if count < 2 or not len(values):
        return None
    values = values / values.sum()
    entropy = float(-(values * np.log(values)).sum())
    return float(np.clip(1 - entropy / math.log(count), 0, 1))
