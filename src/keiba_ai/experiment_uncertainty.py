"""How far to trust the model when sizing bets: market blend and probability uncertainty, fitted on the v4
evaluation year.

For every runner the betting strength is

    u = a * model score + b * log(market win probability) + e,   e ~ N(0, s0^2 + (c * NN seed spread)^2)

and the first three places follow the same staged Plackett-Luce process as the rank distribution (stage
temperatures of the model, relative to the first stage).  a, b, s0, c maximise the likelihood of the observed
1st-2nd-3rd order, integrating e by Monte Carlo with fixed draws.  The market probability is the normalised
inverse of the final win odds; odds never enter the model itself.

With log utility the expected growth of a bet is linear in the outcome probabilities, so "Kelly with
uncertain probabilities" is Kelly on the predictive probabilities: the order probabilities averaged over e.
The uncertainty therefore makes the probabilities flatter (and pulls them toward the market through b), which
is what reduces the stakes.

The fit is checked out of sample: parameters fitted on January-June are scored on July-December (and the
reverse).  Besides the order likelihood, a 単勝 Kelly back-test bets each race's best win ticket with the
Kelly fraction of the current bankroll (capped at ``max_fraction``) using the final odds, and reports the
mean log growth per race for the model alone and for the fitted predictive probabilities, at full and half
Kelly.  Final odds are what the pool paid, not the odds when the bet would be placed, so it is indicative.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

DRAWS = 32
PARAMETERS = ('a', 'b', 'log_s0', 'log_c')


def load_races(model_dir, dataset, group):
    """Padded arrays of the evaluation races with a clean 1st-2nd-3rd and known win odds for every runner."""
    from .model import readonly
    model_dir = Path(model_dir)
    path = model_dir / f'evaluation_{group}.csv'
    temps_path = model_dir / f'evaluation_temperatures_{group}.json'
    if not path.is_file() or not temps_path.is_file():
        raise ValueError(f'{path.name} / {temps_path.name} がありません。--evaluate-until 付きで学習したv4のフォルダーを'
                         '指定してください。')
    temperatures = json.loads(temps_path.read_text(encoding='utf-8'))['model']
    with path.open(encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if 'network_spread' not in rows[0]:
        raise ValueError('評価CSVにnetwork_spread列がありません。最新版のpipeline_v4 fitで作り直してください。')
    first_day = min(row['race_date'] for row in rows)
    odds = {}
    connection = readonly(dataset)
    try:
        for race, horse, value in connection.execute(
                "SELECT f.race_id,f.horse_id,json_extract(p.payload_json,'$.odds') FROM features f JOIN input_rows p "
                "ON p.region=f.region AND p.race_id=f.race_id AND p.horse_id=f.horse_id "
                "WHERE f.group_name=? AND f.day>=?", (group, first_day)):
            try:
                odds[(race, horse)] = float(value)
            except (TypeError, ValueError):
                pass
    finally:
        connection.close()
    races = {}
    for row in rows:
        races.setdefault(row['race_id'], []).append(row)
    kept = []
    for race_id, members in races.items():
        positions = [int(member['finish_position']) if member['finish_position'] else 0 for member in members]
        if sorted(position for position in positions if 1 <= position <= 3) != [1, 2, 3]:
            continue
        values = [odds.get((race_id, member['horse_id'])) for member in members]
        if len(members) < 3 or any(value is None or value <= 1 for value in values):
            continue
        kept.append((race_id, members[0]['race_date'], members, positions, values))
    if not kept:
        raise ValueError(f'{group}: 1〜3着が確定し、全馬の単勝オッズがあるレースがありません。')
    width = max(len(members) for _, _, members, _, _ in kept)
    count = len(kept)
    shape = (count, width)
    data = dict(race_ids=[], horse_ids=[], months=np.zeros(count, dtype=int), present=np.zeros(shape, dtype=bool),
                model=np.zeros(shape), market=np.zeros(shape), spread=np.zeros(shape), odds=np.ones(shape),
                place=np.zeros((count, 3), dtype=int), temperatures=temperatures)
    for index, (race_id, day, members, positions, values) in enumerate(kept):
        n = len(members)
        data['race_ids'].append(race_id)
        data['horse_ids'].append([member['horse_id'] for member in members])
        data['months'][index] = int(day[5:7])
        data['present'][index, :n] = True
        data['model'][index, :n] = [float(member['strength_score']) for member in members]
        data['spread'][index, :n] = [float(member['network_spread']) for member in members]
        inverse = np.asarray([1 / value for value in values])
        data['market'][index, :n] = np.log(inverse / inverse.sum())
        data['odds'][index, :n] = values
        for place in (1, 2, 3):
            data['place'][index, place - 1] = positions.index(place)
    return data


def _subset(data, mask):
    out = {key: value[mask] if isinstance(value, np.ndarray) else value for key, value in data.items()}
    for key in ('race_ids', 'horse_ids'):
        if key in data:
            out[key] = [item for item, keep in zip(data[key], mask) if keep]
    return out


def _stage_scale(temperatures):
    """Stage temperatures 1-3 relative to the first (the scale of a and b absorbs the first)."""
    values = [float(value) for value in temperatures]
    values = (values + values[-1:] * 3)[:3]
    return np.asarray(values) / values[0]


def _utilities(data, params, draws):
    """(races, draws, runners) strengths with the noise draws; absent runners at -inf."""
    a, b, log_s0, log_c = params
    sd = np.sqrt(np.exp(2 * log_s0) + (np.exp(log_c) * data['spread']) ** 2)
    u = (a * data['model'] + b * data['market'])[:, None, :] + sd[:, None, :] * draws
    return np.where(data['present'][:, None, :], u, -np.inf)


def order_log_likelihood(data, params, draws):
    """Mean over races of log P(observed 1st-2nd-3rd), the noise integrated over ``draws``."""
    scale = _stage_scale(data['temperatures'])
    u = _utilities(data, params, draws)
    races = np.arange(len(u))
    remaining = data['present'][:, None, :].repeat(u.shape[1], axis=1)
    log_p = np.zeros(u.shape[:2])
    for stage in range(3):
        stage_u = np.where(remaining, u / scale[stage], -np.inf)
        top = stage_u.max(axis=2, keepdims=True)
        log_norm = np.log(np.exp(stage_u - top).sum(axis=2)) + top[..., 0]
        chosen = data['place'][:, stage]
        log_p += stage_u[races, :, chosen] - log_norm
        remaining = remaining.copy()
        remaining[races, :, chosen] = False
    top = log_p.max(axis=1)
    return float((np.log(np.exp(log_p - top[:, None]).mean(axis=1)) + top).mean())


def fit(data, *, fix=None, seed=42):
    """Maximum likelihood (a, b, log_s0, log_c); ``fix`` holds parameters at given values."""
    from scipy.optimize import minimize
    fix = dict(fix or {})
    draws = np.random.default_rng(seed).standard_normal((len(data['model']), DRAWS, data['model'].shape[1]))
    start = dict(a=1.0, b=0.0, log_s0=math.log(.3), log_c=math.log(1.0))
    free = [name for name in PARAMETERS if name not in fix]

    def full(values):
        merged = dict(start, **fix, **dict(zip(free, values)))
        return [merged[name] for name in PARAMETERS]
    result = minimize(lambda values: -order_log_likelihood(data, full(values), draws),
                      [start[name] for name in free], method='L-BFGS-B',
                      bounds=[(-5, 5) if name in ('a', 'b') else (-12, 2) for name in free])
    params = full(result.x)
    return dict(zip(PARAMETERS, params)), float(-result.fun)


NO_NOISE = dict(log_s0=-30.0, log_c=-30.0)
VARIANTS = {
    'model': dict(b=0.0, **NO_NOISE),
    'market': dict(a=0.0, **NO_NOISE),
    'blend': dict(NO_NOISE),
    'model_uncertain': dict(b=0.0),
    'full': {},
}


def win_probabilities(data, params, *, seed=7, draws=64):
    """(races, runners) predictive win probabilities: the first-stage softmax averaged over the noise."""
    noise = np.random.default_rng(seed).standard_normal((len(data['model']), draws, data['model'].shape[1]))
    u = _utilities(data, params, noise)
    top = u.max(axis=2, keepdims=True)
    p = np.exp(u - top)
    p /= p.sum(axis=2, keepdims=True)
    return p.mean(axis=1)


def kelly_backtest(data, probabilities, *, fraction=1.0, max_fraction=.25):
    """単勝 Kelly on each race's best win ticket: mean log growth per race, races bet and hit rate."""
    growth, bets, hits = [], 0, 0
    for index in range(len(probabilities)):
        p, o = probabilities[index], data['odds'][index]
        present = data['present'][index]
        edge = np.where(present, (p * o - 1) / np.maximum(o - 1, 1e-9), -1)
        best = int(np.argmax(edge))
        f = min(max(edge[best], 0.0) * fraction, max_fraction)
        if f <= 0:
            growth.append(0.0)
            continue
        bets += 1
        won = data['place'][index, 0] == best
        hits += won
        growth.append(math.log1p(f * (o[best] - 1)) if won else math.log1p(-f))
    return dict(mean_log_growth=float(np.mean(growth)), races=len(growth), bets=bets,
                hit_rate=hits / bets if bets else None)


def evaluate_group(model_dir, dataset, group, report=None):
    report = report or (lambda **message: None)
    data = load_races(model_dir, dataset, group)
    report(event='races', group=group, races=len(data['model']))
    first_half = data['months'] <= 6
    halves = {'fit_jan_jun_test_jul_dec': (first_half, ~first_half),
              'fit_jul_dec_test_jan_jun': (~first_half, first_half)}
    result = dict(races=len(data['model']), out_of_sample={}, full_year={})
    for name, (train, test) in halves.items():
        if train.sum() < 50 or test.sum() < 50:
            result['out_of_sample'][name] = None  # the evaluation period does not cover both halves
            continue
        train_data, test_data = _subset(data, train), _subset(data, test)
        test_draws = np.random.default_rng(1).standard_normal((len(test_data['model']), DRAWS,
                                                               test_data['model'].shape[1]))
        entry = {}
        for variant, fix in VARIANTS.items():
            report(event='fit', group=group, split=name, variant=variant)
            params, train_ll = fit(train_data, fix=fix)
            values = [params[key] for key in PARAMETERS]
            probabilities = win_probabilities(test_data, values)
            entry[variant] = dict(params=params, train_log_likelihood=train_ll,
                                  test_log_likelihood=order_log_likelihood(test_data, values, test_draws),
                                  kelly=kelly_backtest(test_data, probabilities),
                                  half_kelly=kelly_backtest(test_data, probabilities, fraction=.5))
        result['out_of_sample'][name] = entry
    for variant, fix in VARIANTS.items():
        report(event='fit', group=group, split='full_year', variant=variant)
        params, ll = fit(data, fix=fix)
        result['full_year'][variant] = dict(params=params, log_likelihood=ll)
    return result


def main(argv=None):
    from .priority import background
    background()
    from .cli import emit
    parser = argparse.ArgumentParser(description='買い目の金額のための、市場との混ぜ方と確率の不確かさの推定（v4の評価期間）')
    parser.add_argument('--model-dir', required=True, type=Path, help='--evaluate-until 付きで学習したv4のフォルダー')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='結果のJSON')
    parser.add_argument('--groups', nargs='+', default=['central_flat', 'local_flat'])
    args = parser.parse_args(argv)
    try:
        summary = dict(model_dir=str(args.model_dir.resolve()), draws=DRAWS, groups={})
        for group in args.groups:
            summary['groups'][group] = evaluate_group(args.model_dir, args.dataset, group, report=emit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        emit(event='done', output=str(args.output))
        return 0
    except (ValueError, OSError) as exc:
        emit(event='error', message=str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
