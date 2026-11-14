"""Bet hit probabilities from the rank model, multi-point bet proposals, and a race confidence score.

The rank distribution comes from a staged Plackett-Luce order plus an independent finish probability
per horse (``ranking_v3.staged_rank_distribution``).  The same process is simulated here, so the hit
probability of any ticket -- including combinations such as exacta or trifecta that the per-horse
marginals cannot give -- is consistent with the distribution shown on the race card.

Odds are only used to size and rank proposals (Kelly growth of a bankroll); they never enter the model.
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


# Betting strength = a * model strength + b * log(market win probability), with the model's stage temperatures
# relative to the first stage (experiment_uncertainty, v4 evaluation year 2025, full-year fit).  The blend beat
# both the model alone and the market alone on the 1st-2nd-3rd order out of sample in both half-year splits
# (test log likelihood central -5.910/-6.107 vs model -6.312/-6.445 and market -5.920/-6.122; local -5.011/-5.105
# vs -5.244/-5.347 and -5.075/-5.157).  Adding noise for the probabilities' uncertainty (constant or NN seed
# spread) did not improve it, so it is left out.  In the 単勝 Kelly back-test at the final odds the model alone
# lost about 3.5% of the bankroll per race at full Kelly; the blend lost 0.03% (central) / 0.2% (local): it bets
# far less, but no setting had a positive edge.
MARKET_BLEND = {'central': dict(a=.2942, b=.8899), 'local': dict(a=.4991, b=.6561)}


MARKET_TERMS = ('s', 'm', 'm2', 's_field', 'm_field', 's_exp', 'm_exp')


def default_market_config():
    """{region: spec} of the built-in blend (one set of weights for every finishing stage)."""
    return {region: dict(variant='blend', params={'all': dict(s=weights['a'], m=weights['b'])}, standardization=None)
            for region, weights in MARKET_BLEND.items()}


def load_market_config(path):
    """{region: spec} from the output of ``keiba_ai.experiment_market`` (its recommended variant per group), or
    None when the file is missing or unusable.  Regions it does not cover keep the built-in blend."""
    import json
    from pathlib import Path
    path = Path(path)
    if not path.is_file():
        return None
    try:
        groups = json.loads(path.read_text(encoding='utf-8')).get('groups') or {}
    except (OSError, ValueError):
        return None
    config = default_market_config()
    for region, group in (('central', 'central_flat'), ('local', 'local_flat')):
        recommended = (groups.get(group) or {}).get('recommended') or {}
        params = recommended.get('params') or {}
        rows = [params[name] for name in ('all', '1st', '2nd_3rd') if name in params]
        if not rows or any(set(row) - set(MARKET_TERMS) for row in rows):
            continue
        uses_covariates = any(key.endswith(('_field', '_exp')) for row in rows for key in row)
        standardization = recommended.get('standardization')
        if uses_covariates and not standardization:
            continue  # an older output without the covariates' scale: keep the built-in blend
        config[region] = dict(variant=recommended.get('variant'), params=params, standardization=standardization)
    return config


def blend_with_market(region, scores, temperatures, win_odds, *, prior_starts=None, config=None):
    """(betting strengths, stage temperatures relative to the first) mixing the model with the win odds, or None
    when any runner's win odds are missing.  ``win_odds`` and ``prior_starts`` line up with ``scores``.

    The strengths are one row per finishing stage group: a single row ('all'), or two rows (1st place, then
    2nd place onwards) when the mix differs by stage, as fitted by keiba_ai.experiment_market."""
    import math
    values = [_odds_value(value) for value in win_odds]
    if not values or any(value is None or value <= 1 for value in values):
        return None
    spec = (config or default_market_config())['central' if region == 'central' else 'local']
    inverse = np.asarray([1 / value for value in values])
    market = np.log(inverse / inverse.sum())
    model = np.asarray(scores, dtype=float)
    scale = spec.get('standardization') or {}
    field_z = ((len(model) - scale['field_mean']) / scale['field_sd']) if scale else 0.0
    starts = list(prior_starts or [None] * len(model))
    exp_z = np.asarray([(math.log1p(max(float(value), 0.0)) - scale['exp_mean']) / scale['exp_sd']
                        if scale and value is not None and math.isfinite(float(value)) else 0.0 for value in starts])
    terms = dict(s=model, m=market, m2=market ** 2, s_field=model * field_z, m_field=market * field_z,
                 s_exp=model * exp_z, m_exp=market * exp_z)
    params = spec['params']
    rows = [params['all']] if 'all' in params else [params['1st'], params['2nd_3rd']]
    blended = np.asarray([sum(float(weight) * terms[name] for name, weight in row.items()) for row in rows])
    stages = np.asarray([float(value) for value in temperatures])
    return blended, list(stages / stages[0])


def place_slots(starters):
    """How many places pay for 複勝 (JRA/NAR rule): 3 with 8+ starters, 2 with 5-7, none with 4 or fewer."""
    return 3 if starters >= 8 else 2 if starters >= 5 else 0


def available(bet, starters):
    if bet == 'place':
        return place_slots(starters) > 0
    if bet == 'bracket':
        return starters >= 9
    return starters >= (3 if bet in ('wide', 'trio', 'trifecta') else 2)


def _staged_orders_by_row(rows, per_stage, generator, batch):
    """Like ranking_v3._staged_orders, with the strengths of finishing position k taken from row min(k, last)."""
    count = rows.shape[1]
    remaining = np.ones((batch, count), dtype=bool)
    orders = np.empty((batch, count), dtype=np.int64)
    index = np.arange(batch)
    for position in range(count):
        scores = rows[min(position, len(rows) - 1)]
        uniforms = np.maximum(generator.random((batch, count)), np.finfo(float).tiny)
        keys = (scores - scores.max())[None, :] / per_stage[position] - np.log(-np.log(uniforms))
        keys[~remaining] = -np.inf
        picks = np.argmax(keys, axis=1)
        orders[:, position] = picks
        remaining[index, picks] = False
    return orders


def simulate_top3(scores, completion, temperatures, *, samples=20000, seed=42):
    """(samples, 3) indices of the first three finishers (-1 where fewer horses finish).  ``scores`` is one
    strength per runner, or rows of strengths by finishing stage (see blend_with_market)."""
    from .ranking_v3 import _staged_orders, stage_temperatures
    scores = np.asarray(scores, dtype=float)
    completion = np.clip(np.asarray(completion, dtype=float), 0, 1)
    count = scores.shape[-1]
    per_stage = stage_temperatures(temperatures, count)
    generator = np.random.default_rng(int(seed))
    parts, remaining = [], int(samples)
    while remaining:
        batch = min(remaining, 4096)
        orders = (_staged_orders(scores - scores.max(), per_stage, generator, batch) if scores.ndim == 1
                  else _staged_orders_by_row(scores, per_stage, generator, batch))
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
        """Boolean per sample: does this ticket pay?  (cached per ticket)"""
        cache = self.__dict__.setdefault('_hit_cache', {})
        if (bet, key) not in cache:
            cache[(bet, key)] = self._compute_hits(bet, key)
        return cache[(bet, key)]

    def _compute_hits(self, bet, key):
        h = self.horse
        if bet == 'place':
            return (h[:, :self.slots] == int(key)).any(axis=1)
        if bet == 'wide':
            a, b = (int(value) for value in key.split('-'))
            return (h == a).any(axis=1) & (h == b).any(axis=1)
        lookup, codes = self._codes(bet)
        code = lookup.get(key)
        return np.zeros(len(codes), dtype=bool) if code is None or key == '' else codes == code


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
        lookup, codes = self._codes(bet)
        counts = np.bincount(codes, minlength=len(lookup))
        return {key: float(counts[code]) / self.samples for key, code in lookup.items() if key and counts[code]}

    def _codes(self, bet):
        """({ticket key: integer code}, code per sample) of one bet type; '' marks samples that pay nothing."""
        cache = self.__dict__.setdefault('_code_cache', {})
        if bet not in cache:  # comparing integers is much faster than comparing strings
            labels = np.asarray(self._outcomes(bet), dtype=object)
            names, codes = np.unique(np.where(labels == None, '', labels).astype(str), return_inverse=True)  # noqa: E711
            cache[bet] = (dict(zip(names.tolist(), range(len(names)))), codes)
        return cache[bet]

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


# How the stakes are chosen (both for single bet types and mixes):
# * within a set of tickets the stakes keep every hit paying about the same (more on the shorter odds), so any
#   single hit pays back at least the total -- possible only when the sum of 1/odds is at most 1 (no トリガミ);
# * the total is chosen (in 100-yen steps, up to ``budget``) to maximise the expected log growth of the
#   ``bankroll``, mean(log(1 + (payout - total) / bankroll)) over the simulated finishing orders -- the Kelly
#   criterion: what makes the bankroll grow fastest when the same kind of bet is repeated;
# * sets are ranked by that growth.  A negative best growth means that, by the model's probabilities, not
#   betting is better; such sets are still listed (smallest stake), flagged by the sign.
# The model's probabilities are trusted as they are.  Stakes scale with the bankroll, so a bankroll set below the
# real one (e.g. half: half Kelly) gives a more cautious stake.
SEARCH_SAMPLES = 3000   # simulated orders used to screen candidate sets; the shortlist is re-scored on all
GROWTH_STEPS = 14       # totals tried per set when screening
POOL_PER_TYPE = 4       # best sets of each bet type that may enter a mix
SHORTLIST = 40          # mixes re-scored exactly


def _totals(budget, minimum, unit=100):
    """Up to GROWTH_STEPS totals between ``minimum`` and ``budget`` (multiples of ``unit``), denser at the low end."""
    lowest, highest = max(unit, int(minimum)), int(budget) // unit * unit
    if highest < lowest:
        return np.asarray([], dtype=float)
    grid = np.unique(np.round(np.geomspace(lowest, highest, GROWTH_STEPS) / unit) * unit)
    return grid[(grid >= lowest) & (grid <= highest)].astype(float)


def _screen(hits, values, bankroll, budget, unit=100):
    """Approximate best (total, growth) of a ticket set under the equal-payout split, or None (トリガミ / budget).

    ``hits``: (tickets, samples) booleans.  With stakes proportional to 1/odds every hit pays total / sum(1/odds),
    so the payout of a sample is (number of hits) x total / sum(1/odds)."""
    inverse = float(sum(1 / value for value in values))
    if inverse > 1 + 1e-9:
        return None
    grid = _totals(min(budget, bankroll * .999), unit * len(values), unit)
    if not len(grid):
        return None
    ratio = hits.sum(axis=0) / inverse - 1
    growth = np.log1p(np.outer(grid, ratio) / bankroll).mean(axis=1)
    best = int(np.argmax(growth))
    return float(grid[best]), float(growth[best])


def _exact(hits, values, total, bankroll):
    """Stakes in 100-yen units for ``total`` and their outcome over all samples; None when a hit would lose."""
    allocation = allocate(values, total)
    if allocation is None:
        return None
    stakes, smallest, cost = allocation
    if smallest < cost or cost > total:
        return None
    payout = (hits * (np.asarray(stakes, dtype=float) * np.asarray(values))[:, None]).sum(axis=0)
    hit = payout > 0
    return dict(stakes=stakes, cost=cost, payout=payout, hit_rate=float(hit.mean()),
                growth=float(np.log1p((payout - cost) / bankroll).mean()),
                expected_return=float(payout.mean() / cost), profit_rate=float((payout > cost).mean()),
                min_payout=round(float(smallest)), max_payout=round(float(payout.max())))


def _best_exact(hits, values, bankroll, budget, approximate_total, unit=100):
    """Exact plan at the screened total, trying nearby totals when rounding to 100 yen breaks the no-loss rule."""
    tried = sorted(set(float(total) for total in _totals(min(budget, bankroll * .999), unit * len(values), unit)),
                   key=lambda total: abs(total - approximate_total))
    best = None
    for total in tried[:6]:
        plan = _exact(hits, values, total, bankroll)
        if plan is not None and (best is None or plan['growth'] > best['growth']):
            best = plan
    return best


def _ticket_sets(bets, odds, bet, max_points):
    """'The N most likely tickets' sets of one bet type with all odds known: (keys, values, probabilities, hits)."""
    table = (odds or {}).get(bet) or {}
    probabilities = bets.probabilities(bet)
    if not table or not probabilities:
        return []
    order = sorted(probabilities.items(), key=lambda item: -item[1])[:CANDIDATES]
    sets = []
    for size in range(1, min(MAX_POINTS[bet], max_points, len(order)) + 1):
        keys = [key for key, _ in order[:size]]
        values = [_odds_value(table.get(key)) for key in keys]
        if any(value is None for value in values):
            break
        sets.append((keys, values, [probabilities[key] for key in keys],
                     np.stack([bets._hits(bet, key) for key in keys])))
    return sets


def propose(bets: RaceBets, odds=None, *, min_hit=.2, max_points=10, budget=1000, bankroll=50000, limit=8,
            per_type=1):
    """Multi-point proposals of one bet type each: the N most likely tickets (N = 1..), at least ``min_hit``.

    With odds: stakes and total by the Kelly criterion on ``bankroll`` (see above), sorted by growth.  Without
    odds every ticket gets 100 yen and the sets are sorted by hit rate.  The lower odds of 複勝 / ワイド ranges
    are used."""
    odds = odds or {}
    proposals = []
    for bet, name, _, _ in BET_TYPES:
        probabilities = bets.probabilities(bet)
        if not probabilities:
            continue
        table = odds.get(bet) or {}
        priced_sets = {tuple(keys): (values, hits) for keys, values, _, hits in _ticket_sets(bets, odds, bet,
                                                                                               max_points)}
        order = sorted(probabilities.items(), key=lambda item: -item[1])[:CANDIDATES]
        for size in range(1, min(MAX_POINTS[bet], max_points, len(order)) + 1):
            keys = tuple(key for key, _ in order[:size])
            hit = bets.hit_rate(bet, keys) if bet in ('place', 'wide') else sum(p for _, p in order[:size])
            if hit < min_hit:
                continue
            item = dict(bet=bet, name=name, points=size, hit_rate=hit, priced=keys in priced_sets)
            if keys in priced_sets:
                values, hits = priced_sets[keys]
                screened = _screen(hits[:, :SEARCH_SAMPLES], values, bankroll, budget)
                plan = screened and _best_exact(hits, values, bankroll, budget, screened[0])
                if not plan:
                    continue
                stakes = plan['stakes']
                item.update(cost=plan['cost'], stakes=stakes, min_payout=plan['min_payout'],
                            max_payout=plan['max_payout'], expected_return=plan['expected_return'],
                            growth=plan['growth'], growth_rate=math.expm1(plan['growth']),
                            profit_rate=plan['profit_rate'], synthetic_odds=1 / sum(1 / value for value in values))
            else:
                stakes = [100] * size
                item.update(cost=100 * size, stakes=stakes, min_payout=None, max_payout=None, expected_return=None,
                            growth=None, growth_rate=None, profit_rate=None, synthetic_odds=None)
            item['tickets'] = [dict(key=key, probability=probabilities[key], odds=table.get(key), stake=stake)
                               for key, stake in zip(keys, stakes)]
            proposals.append(item)
    proposals.sort(key=lambda item: (item['growth'] is None, -(item['growth'] if item['growth'] is not None else 0),
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


def combine(bets: RaceBets, odds, *, min_hit=.2, max_points=10, budget=1000, bankroll=50000, limit=5):
    """Proposals that mix two or three bet types (e.g. 単勝 + 馬連 + 3連単), ranked by Kelly growth.

    Every priced 'N most likely tickets' set may be a part, also when it alone hits rarely (a 3連単 set is
    what a mix adds to a safer part); ``min_hit`` applies to the mix.  The best POOL_PER_TYPE sets of each
    type by their own growth enter the search over all 2- and 3-type mixes; the shortlist is re-scored on all
    simulated orders.  Hits of several parts add up."""
    from itertools import combinations as choose
    if not odds:
        return []
    pool = []
    for bet, name, _, _ in BET_TYPES:
        ranked = []
        for keys, values, probabilities, hits in _ticket_sets(bets, odds, bet, max_points - 1):
            if sum(1 / value for value in values) > 1:
                continue  # a part that loses money when it hits can never be in a no-loss mix
            screened = _screen(hits[:, :SEARCH_SAMPLES], values, bankroll, budget)
            score = screened[1] if screened else -math.inf
            ranked.append((score, dict(bet=bet, name=name, keys=keys, values=values, probabilities=probabilities,
                                       hits=hits)))
        ranked.sort(key=lambda item: -item[0])
        pool.extend(part for _, part in ranked[:POOL_PER_TYPE])
    screened = []
    for count in (2, 3):
        for members in choose(pool, count):
            if len({part['bet'] for part in members}) < count:
                continue
            values = [value for part in members for value in part['values']]
            if len(values) > max_points or sum(1 / value for value in values) > 1:
                continue
            hits = np.concatenate([part['hits'] for part in members])
            if hits[:, :SEARCH_SAMPLES].any(axis=0).mean() < min_hit:
                continue
            result = _screen(hits[:, :SEARCH_SAMPLES], values, bankroll, budget)
            if result is not None:
                screened.append((result[1], result[0], members, values, hits))
    screened.sort(key=lambda item: -item[0])
    mixes = []
    for _, total, members, values, hits in screened[:SHORTLIST]:
        plan = _best_exact(hits, values, bankroll, budget, total)
        if plan is None or plan['hit_rate'] < min_hit:
            continue
        position, rows = 0, []
        for part in members:
            size = len(part['keys'])
            share = plan['stakes'][position:position + size]
            rows.append(dict(bet=part['bet'], name=part['name'], cost=sum(share), tickets=[
                dict(key=key, probability=probability, odds=(odds[part['bet']] or {}).get(key), stake=stake)
                for key, probability, stake in zip(part['keys'], part['probabilities'], share)]))
            position += size
        mixes.append(dict(parts=rows, points=len(values), cost=plan['cost'], hit_rate=plan['hit_rate'],
                          expected_return=plan['expected_return'], growth=plan['growth'],
                          growth_rate=math.expm1(plan['growth']), min_payout=plan['min_payout'],
                          max_payout=plan['max_payout'], profit_rate=plan['profit_rate']))
    mixes.sort(key=lambda item: (-item['growth'], -item['hit_rate']))
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
