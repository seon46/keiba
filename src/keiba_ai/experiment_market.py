"""Which way of mixing the model with the win odds predicts the 1st-2nd-3rd order best (v4 evaluation year).

The betting strength of a runner at finishing stage k is a linear combination of terms

    s       model strength score
    m       log market win probability (normalised inverse of the final win odds)
    m2      m squared: a favourite-longshot bias that is not a straight line in log odds
    s_field, m_field   s and m times the standardised field size
    s_exp,   m_exp     s and m times the runner's standardised log(1 + prior starts): how much the model knows

and the order follows the staged Plackett-Luce process of the rank distribution (the model's stage
temperatures relative to the first stage).  Variants:

    blend       s, m                            (what the app uses now)
    curve       s, m, m2
    stages      s, m with separate weights for 1st place and for 2nd-3rd
    conditions  s, m and their field-size / experience interactions
    all         every term, separate for 1st and 2nd-3rd

Each is fitted by maximum likelihood on one half of the year and scored on the other half (both ways); the
recommendation is the variant with the best mean out-of-sample log likelihood when it beats ``blend`` by at
least ``MIN_GAIN`` per race in both halves.  A 単勝 Kelly back-test at the final odds is reported as in
experiment_uncertainty.  Final odds only: odds at the time a bet is placed are less informative (see
keiba_app.odds_timing for that).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TERMS = ('s', 'm', 'm2', 's_field', 'm_field', 's_exp', 'm_exp')
VARIANTS = {
    'blend': dict(terms=('s', 'm'), split=False),
    'curve': dict(terms=('s', 'm', 'm2'), split=False),
    'stages': dict(terms=('s', 'm'), split=True),
    'conditions': dict(terms=('s', 'm', 's_field', 'm_field', 's_exp', 'm_exp'), split=False),
    'all': dict(terms=TERMS, split=True),
}
MIN_GAIN = .002
EXPERIENCE_FEATURE = 'horse_global_prior_starts'


def load(model_dir, dataset, group):
    """experiment_uncertainty.load_races plus each runner's prior starts, with the term tensor."""
    from .experiment_uncertainty import load_races
    from .features import FEATURE_NAMES
    from .model import readonly
    data = load_races(model_dir, dataset, group)
    column = FEATURE_NAMES.index(EXPERIENCE_FEATURE)
    wanted = set(data['race_ids'])
    starts = {}
    connection = readonly(dataset)
    try:
        for race, horse, value in connection.execute(
                f"SELECT race_id,horse_id,json_extract(feature_json,'$[{column}]') FROM features WHERE group_name=?",
                (group,)):
            if race in wanted:
                starts[(race, horse)] = value
    finally:
        connection.close()
    experience = np.zeros(data['model'].shape)
    for index, race in enumerate(data['race_ids']):
        for slot, horse in enumerate(data['horse_ids'][index]):
            value = starts.get((race, horse))
            experience[index, slot] = math.log1p(float(value)) if value not in (None, '') else 0.0
    data['experience'] = experience
    return with_terms(data)


def with_terms(data):
    """Add ``terms`` (races, runners, len(TERMS)); covariates standardised over the present runners."""
    present = data['present']
    s, m = data['model'], data['market']
    field = present.sum(axis=1).astype(float)
    field_z = (field - field.mean()) / (field.std() or 1.0)
    experience = data.get('experience', np.zeros(s.shape))
    values = experience[present]
    exp_z = np.where(present, (experience - values.mean()) / (values.std() or 1.0), 0.0)
    columns = dict(s=s, m=m, m2=m ** 2, s_field=s * field_z[:, None], m_field=m * field_z[:, None],
                   s_exp=s * exp_z, m_exp=m * exp_z)
    data['terms'] = np.stack([np.where(present, columns[name], 0.0) for name in TERMS], axis=2)
    return data


def _stage_scale(temperatures):
    values = [float(value) for value in temperatures]
    values = (values + values[-1:] * 3)[:3]
    return np.asarray(values) / values[0]


def _weights(variant, theta):
    """(3 stages, len(TERMS)) weights from the free parameters of a variant."""
    spec = VARIANTS[variant]
    index = [TERMS.index(name) for name in spec['terms']]
    groups = np.asarray(theta, dtype=float).reshape(2 if spec['split'] else 1, len(index))
    weights = np.zeros((3, len(TERMS)))
    for stage in range(3):
        weights[stage, index] = groups[min(stage, len(groups) - 1)]
    return weights


def order_log_likelihood(data, weights, *, per_race=False):
    """Mean (or per-race) log P(observed 1st-2nd-3rd) for stage weights (3, len(TERMS))."""
    scale = _stage_scale(data['temperatures'])
    races = np.arange(len(data['terms']))
    remaining = data['present'].copy()
    log_p = np.zeros(len(races))
    for stage in range(3):
        u = np.where(remaining, data['terms'] @ weights[stage] / scale[stage], -np.inf)
        top = u.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(u - top).sum(axis=1)) + top[:, 0]
        chosen = data['place'][:, stage]
        log_p += u[races, chosen] - log_norm
        remaining[races, chosen] = False
    return log_p if per_race else float(log_p.mean())


def fit(data, variant):
    from scipy.optimize import minimize
    spec = VARIANTS[variant]
    groups = 2 if spec['split'] else 1
    start = np.zeros((groups, len(spec['terms'])))
    for name, value in (('s', .3), ('m', .9)):
        if name in spec['terms']:
            start[:, spec['terms'].index(name)] = value
    result = minimize(lambda theta: -order_log_likelihood(data, _weights(variant, theta)), start.ravel(),
                      method='L-BFGS-B', bounds=[(-5, 5)] * start.size)
    theta = result.x.reshape(groups, len(spec['terms']))
    names = ['1st', '2nd_3rd'] if spec['split'] else ['all']
    params = {name: dict(zip(spec['terms'], map(float, row))) for name, row in zip(names, theta)}
    return params, result.x, float(-result.fun)


def win_probabilities(data, weights):
    u = np.where(data['present'], data['terms'] @ weights[0], -np.inf)
    p = np.exp(u - u.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def _subset(data, mask):
    out = {key: value[mask] if isinstance(value, np.ndarray) else value for key, value in data.items()}
    for key in ('race_ids', 'horse_ids'):
        if key in data:
            out[key] = [item for item, keep in zip(data[key], mask) if keep]
    return out


def evaluate_group(model_dir, dataset, group, report=None, data=None):
    from .experiment_uncertainty import kelly_backtest
    report = report or (lambda **message: None)
    data = data if data is not None else load(model_dir, dataset, group)
    report(event='races', group=group, races=len(data['terms']))
    first_half = data['months'] <= 6
    halves = {'fit_jan_jun_test_jul_dec': (first_half, ~first_half),
              'fit_jul_dec_test_jan_jun': (~first_half, first_half)}
    result = dict(races=len(data['terms']), out_of_sample={}, full_year={})
    gains = {variant: [] for variant in VARIANTS}
    for name, (train, test) in halves.items():
        if train.sum() < 50 or test.sum() < 50:
            result['out_of_sample'][name] = None
            continue
        train_data, test_data = _subset(data, train), _subset(data, test)
        entry = {}
        for variant in VARIANTS:
            report(event='fit', group=group, split=name, variant=variant)
            params, theta, train_ll = fit(train_data, variant)
            weights = _weights(variant, theta)
            probabilities = win_probabilities(test_data, weights)
            entry[variant] = dict(params=params, train_log_likelihood=train_ll,
                                  test_log_likelihood=order_log_likelihood(test_data, weights),
                                  kelly=kelly_backtest(test_data, probabilities),
                                  half_kelly=kelly_backtest(test_data, probabilities, fraction=.5))
        for variant in VARIANTS:
            gains[variant].append(entry[variant]['test_log_likelihood'] - entry['blend']['test_log_likelihood'])
        result['out_of_sample'][name] = entry
    for variant in VARIANTS:
        report(event='fit', group=group, split='full_year', variant=variant)
        params, _, ll = fit(data, variant)
        result['full_year'][variant] = dict(params=params, log_likelihood=ll)
    result['gain_over_blend'] = {variant: dict(per_half=values, mean=float(np.mean(values)) if values else None)
                                 for variant, values in gains.items()}
    candidates = [variant for variant, values in gains.items()
                  if variant != 'blend' and values and min(values) >= MIN_GAIN]
    best = max(candidates, key=lambda variant: np.mean(gains[variant])) if candidates else 'blend'
    result['recommended'] = dict(variant=best, params=result['full_year'][best]['params'],
                                 rule=f'out-of-sample gain over blend >= {MIN_GAIN} per race in both halves')
    return result


def main(argv=None):
    from .priority import background
    background()
    from .cli import emit
    parser = argparse.ArgumentParser(description='モデルと単勝オッズの混ぜ方の比較（v4の評価期間、確定オッズ）')
    parser.add_argument('--model-dir', required=True, type=Path, help='--evaluate-until 付きで学習したv4のフォルダー')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='結果のJSON')
    parser.add_argument('--groups', nargs='+', default=['central_flat', 'local_flat'])
    args = parser.parse_args(argv)
    try:
        summary = dict(model_dir=str(args.model_dir.resolve()), min_gain=MIN_GAIN, groups={})
        for group in args.groups:
            summary['groups'][group] = evaluate_group(args.model_dir, args.dataset, group, report=emit)
            emit(event='recommended', group=group, **summary['groups'][group]['recommended'])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        emit(event='done', output=str(args.output))
        return 0
    except (ValueError, OSError) as exc:
        emit(event='error', message=str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
