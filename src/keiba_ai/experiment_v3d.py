"""V3 experiment D: jockey skill separated from horse strength (design 5.4 / M4a).

Base = best configuration so far (experiment B tuned LightGBM-PL per group,
4 stage temperatures; local flat additionally uses the announced bodyweight
features of experiment C).  For each validation year Y:

1. ``f0``: horse-strength model *without* any jockey feature.
2. Cross-fitting: training rows before Y are split into three blocks by
   calendar quarter; each block's strength is predicted by an f0 refit on the
   other blocks, so a race's strength never comes from a model fitted on it.
   Validation-year strengths come from f0 (fitted before Y).
3. Jockey effect theta: a Gaussian online Plackett-Luce update run day by day
   with the horse strength as offset.  Each row receives the state *before*
   its own day, so the race's own result never enters its own theta.
   Precision decays with a half-life (long and short versions).
4. Variants on the same folds:
   ``base``          current jockey rates (as before),
   ``nojockey``      no jockey information at all (= f0),
   ``theta_feature`` theta-based jockey features instead of the rates (plan B),
   ``theta_offset``  f0 features, theta as a fixed offset while fitting; the
                     prediction is f + gamma * theta, gamma fitted on the
                     early-stopping period (plan A).

Temperatures and gamma are fitted on the early-stopping period, never on the
validation year.  The final-test period is never read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import sqlite3
import time

from .experiment_v3 import (DEFAULT_VALIDATION_YEARS, PLStructure, _m1_finisher, _shift_months,
                            experiment_feature_names, pl_metric, pl_objective)
from .experiment_v3b import _finisher_rows, _structure
from .experiment_v3c import (TUNED_PARAMS, WEIGHT_FEATURES, _race_dropout, _race_z, announced_features,
                             load_announced)

GROUPS = ('central_flat', 'local_flat')
STAGE_COUNT = 4
JOCKEY_RATE_FEATURES = ('jockey_prior_starts', 'jockey_prior_win_rate', 'jockey_prior_top3_rate', 'jockey_id_missing')
THETA_FEATURES = ('jockey_theta', 'jockey_theta_sd', 'jockey_theta_recent', 'jockey_theta_vs_previous',
                  'jockey_theta_race_z', 'jockey_rides_with_horse', 'jockey_prior_starts')
VARIANTS = ('base', 'nojockey', 'theta_feature', 'theta_offset')
WEIGHT_GROUPS = ('local_flat',)


def _params(threads, seed):
    params = dict(max_bin=255, num_threads=threads, verbosity=-1, seed=seed, deterministic=True,
                  force_row_wise=True, metric='None')
    params.update(TUNED_PARAMS)
    return params


# --------------------------------------------------------------------------
# Online jockey effect
# --------------------------------------------------------------------------

def jockey_timeline(days, race_ids, jockeys, horses, finished, positions, strength, *,
                    half_life=365.0, short_half_life=90.0, prior_sd=.3):
    """Day-by-day Gaussian approximation of a PL jockey effect with strength offsets.

    Returns per-row arrays holding the state *before* that row's day:
    theta, theta_sd, theta_recent, theta of the horse's previous jockey,
    rides of this jockey on this horse, and rides of this jockey.
    Rows must be ordered by day, with each race contiguous.
    """
    import numpy as np
    count = len(days)
    out = {name: np.full(count, np.nan) for name in ('theta', 'theta_sd', 'theta_recent', 'theta_previous_jockey',
                                                       'rides_with_horse', 'rides')}
    prior_precision = 1.0 / prior_sd ** 2
    state = {}  # jockey -> [theta, precision, theta_recent, precision_recent, last_day_ordinal, rides]
    last_jockey = {}  # horse -> jockey of the previous ride
    pair_rides = {}
    ordinals = np.asarray([date.fromisoformat(day).toordinal() for day in days.tolist()]) if count else np.empty(0)
    boundaries = np.flatnonzero(np.r_[True, days[1:] != days[:-1], True]) if count else np.asarray([0])

    def current(jockey, today):
        entry = state.get(jockey)
        if entry is None:
            return 0.0, prior_precision, 0.0, prior_precision, 0
        theta, precision, recent, precision_recent, last, rides = entry
        gap = today - last
        return (theta, max(precision * 2 ** (-gap / half_life), prior_precision), recent,
                max(precision_recent * 2 ** (-gap / short_half_life), prior_precision), rides)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        today = int(ordinals[start])
        rows = np.arange(start, end)
        for row in rows:
            jockey = jockeys[row]
            if not jockey:
                continue
            theta, precision, recent, _, rides = current(jockey, today)
            out['theta'][row], out['theta_sd'][row], out['theta_recent'][row] = theta, precision ** -.5, recent
            out['rides'][row] = rides
            out['rides_with_horse'][row] = pair_rides.get((jockey, horses[row]), 0)
            previous = last_jockey.get(horses[row])
            if previous:
                out['theta_previous_jockey'][row] = current(previous, today)[0]
        # Update with this day's finishing orders (races with at least two finishers and a known strength).
        update_rows = rows[finished[rows] & np.isfinite(strength[rows])]
        if len(update_rows):
            keys = race_ids[update_rows]
            change = np.r_[True, keys[1:] != keys[:-1]]
            code = np.cumsum(change) - 1
            keep = np.bincount(code)[code] >= 2
            update_rows = update_rows[keep]
        if len(update_rows):
            structure = PLStructure(race_ids[update_rows], positions[update_rows])
            stages = (structure.sizes - 1)[structure.code].astype(float)  # undo per-race normalization
            for long_term in (True, False):
                offsets = np.asarray([current(jockeys[row], today)[0 if long_term else 2] if jockeys[row] else 0.0
                                      for row in update_rows])
                _, grad, hess = structure.terms(strength[update_rows] + offsets)
                grad, hess = grad * stages, hess * stages
                totals = {}
                for row, g, h in zip(update_rows, grad, hess):
                    jockey = jockeys[row]
                    if jockey:
                        accumulated = totals.setdefault(jockey, [0.0, 0.0])
                        accumulated[0] += g
                        accumulated[1] += h
                for jockey, (g, h) in totals.items():
                    theta, precision, recent, precision_recent, rides = current(jockey, today)
                    if long_term:
                        theta -= g / (precision + h)
                        precision += h
                    else:
                        recent -= g / (precision_recent + h)
                        precision_recent += h
                    state[jockey] = [theta, precision, recent, precision_recent, today, rides]
        for row in rows:
            jockey = jockeys[row]
            if jockey:
                entry = state.get(jockey)
                if entry is None:
                    theta, precision, recent, precision_recent, _ = current(jockey, today)
                    entry = state[jockey] = [theta, precision, recent, precision_recent, today, 0]
                entry[5] += 1
                pair_rides[(jockey, horses[row])] = pair_rides.get((jockey, horses[row]), 0) + 1
                last_jockey[horses[row]] = jockey
    return out


def theta_feature_block(timeline, race_ids):
    import numpy as np
    theta = timeline['theta']
    return np.column_stack([
        theta, timeline['theta_sd'], timeline['theta_recent'], theta - timeline['theta_previous_jockey'],
        _race_z(theta, race_ids), timeline['rides_with_horse'], timeline['rides'],
    ]).astype(np.float32)


# --------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------

def load_jockeys(database, group, identities):
    import numpy as np
    from .model import readonly
    wanted = {(race, horse): index for index, (race, horse, _) in enumerate(identities)}
    jockeys = np.full(len(identities), '', dtype=object)
    connection = readonly(database)
    try:
        for race, horse, jockey in connection.execute(
                "SELECT f.race_id,f.horse_id,json_extract(p.payload_json,'$.jockey_id') FROM features f JOIN input_rows p "
                "ON p.region=f.region AND p.race_id=f.race_id AND p.horse_id=f.horse_id WHERE f.group_name=?", (group,)):
            index = wanted.get((race, horse))
            if index is not None and jockey:
                jockeys[index] = str(jockey)
    finally:
        connection.close()
    return jockeys


class Blocks:
    """Feature matrix = selected base columns + optional extra blocks."""

    def __init__(self, data, base_columns, blocks):
        self.data, self.base_columns, self.blocks = data, base_columns, blocks  # blocks: list of (array, blank_mask)

    def __call__(self, rows):
        import numpy as np
        parts = [self.data['x'][np.ix_(rows, self.base_columns)]]
        for array, blank in self.blocks:
            block = array[rows].copy()
            if blank is not None:
                block[blank[rows]] = np.nan
            parts.append(block)
        return np.concatenate(parts, axis=1)


def fit_pl(data, matrix, categorical, fit_rows, stop_rows, params, *, rounds, patience, offset=None, fixed_rounds=None):
    import numpy as np
    import lightgbm as lgb
    fit_structure = _structure(data, fit_rows)
    train_set = lgb.Dataset(matrix(fit_rows), categorical_feature=categorical, free_raw_data=False,
                            init_score=None if offset is None else offset[fit_rows])
    train_set.pl_structure = fit_structure
    params = dict(params, objective=pl_objective(fit_structure))
    if fixed_rounds is not None:
        booster = lgb.train(params, train_set, num_boost_round=max(int(fixed_rounds), 1))
        return booster, booster.current_iteration()
    stop_set = lgb.Dataset(matrix(stop_rows), categorical_feature=categorical, reference=train_set, free_raw_data=False,
                           init_score=None if offset is None else offset[stop_rows])
    stop_set.pl_structure = _structure(data, stop_rows)
    booster = lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[stop_set], valid_names=['stop'],
                        feval=pl_metric, callbacks=[lgb.early_stopping(patience, verbose=False)])
    return booster, int(booster.best_iteration or booster.current_iteration())


def _fit_gamma(structure, base_scores, theta):
    import numpy as np
    from scipy.optimize import minimize_scalar
    theta = np.nan_to_num(theta)
    result = minimize_scalar(lambda gamma: structure.mean_loss(base_scores + gamma * theta), bounds=(0, 3), method='bounded')
    return float(result.x)


def _quarter_block(days, blocks=3):
    import numpy as np
    years = np.asarray([int(day[:4]) for day in days.tolist()])
    months = np.asarray([int(day[5:7]) for day in days.tolist()])
    return (years * 4 + (months - 1) // 3) % blocks


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def run_experiment(database, output, *, validation_years=DEFAULT_VALIDATION_YEARS, train_start='2022-04-01',
                   test_start='2026-01-01', stop_months=2, threads=2, rounds=8000, patience=150,
                   half_life=365.0, short_half_life=90.0, prior_sd=.3, crossfit_blocks=3,
                   announce_dropout=.05, evaluation_samples=1024, seed=42, report=None):
    import numpy as np
    from .features import FEATURE_NAMES, CATEGORICAL_FEATURES
    from .model import feature_revision, readonly
    from .rank_model import load_rank_group
    from .ranking_v3 import fit_temperatures, paired_bootstrap, score_races
    from threadpoolctl import threadpool_limits

    report = report or (lambda **message: None)
    database, output = Path(database), Path(output)
    if output.exists():
        raise ValueError(f'出力先は既にあります。別の出力先を指定してください: {output}')
    if any(f'{year:04d}-12-31' >= test_start for year in validation_years):
        raise ValueError('最終評価期間（test-start以降）を検証年に含めることはできません。')
    for value in (train_start, test_start):
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError('日付はYYYY-MM-DDで指定してください。')
    connection = readonly(database)
    try:
        metadata = dict(connection.execute('SELECT key,value_json FROM ai_metadata'))
    except sqlite3.Error as exc:
        raise ValueError(f'prepare済みのfeatures.sqlite3を指定してください: {exc}') from exc
    finally:
        connection.close()
    if json.loads(metadata['feature_revision']) != feature_revision():
        raise ValueError('特徴量コードの版がこのfeatures.sqlite3と一致しません。prepareから作成し直してください。')

    base_names = experiment_feature_names()
    no_jockey_names = [name for name in base_names if name not in JOCKEY_RATE_FEATURES]
    folds = []
    for year in validation_years:
        start = f'{year:04d}-01-01'
        stop_start = _shift_months(start, -stop_months)
        if stop_start <= train_start:
            raise ValueError(f'{year}年の検証foldの学習期間がありません。')
        folds.append(dict(validation_year=year, train_start=train_start, stop_start=stop_start,
                          validation_start=start, validation_end=f'{year + 1:04d}-01-01'))
    started = time.time()
    result = dict(experiment='v3d_jockey_effect', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  dataset=str(database.resolve()), feature_revision=feature_revision(), folds=folds,
                  test_start=test_start, test_period_used=False,
                  settings=dict(threads=threads, rounds=rounds, patience=patience, params=TUNED_PARAMS,
                                stage_temperatures=STAGE_COUNT, half_life_days=half_life,
                                short_half_life_days=short_half_life, prior_sd=prior_sd,
                                crossfit_blocks=crossfit_blocks, weight_feature_groups=list(WEIGHT_GROUPS),
                                announce_dropout=announce_dropout, evaluation_samples=evaluation_samples, seed=seed,
                                variants=list(VARIANTS), theta_features=list(THETA_FEATURES),
                                baseline='base (current jockey rates)'),
                  groups={})
    output.mkdir(parents=True)
    losses_file = (output / 'race_losses.csv').open('w', encoding='utf-8', newline='')
    losses = csv.writer(losses_file)
    losses.writerow(['group', 'variant', 'validation_year', 'race_id', 'outcome_log_loss', 'top3_log_loss', 'win_log_loss'])
    params = _params(threads, seed)
    try:
        for group in GROUPS:
            report(event='loading_group', group=group)
            data = load_rank_group(database, group)
            data['race_ids'] = np.asarray([identity[0] for identity in data['identities']])
            horses = np.asarray([identity[1] for identity in data['identities']], dtype=object)
            jockeys = load_jockeys(database, group, data['identities'])
            days = data['days']
            blocks = []
            if group in WEIGHT_GROUPS:
                weight = announced_features(data, load_announced(database, group, data['identities']))[:, :len(WEIGHT_FEATURES)]
                blocks.append((weight, _race_dropout(data['race_ids'], announce_dropout, seed)))
            weight_names = list(WEIGHT_FEATURES) if blocks else []

            def columns_for(names):
                return [FEATURE_NAMES.index(name) for name in names]

            def categorical_for(names):
                return [index for index, name in enumerate(names) if name in CATEGORICAL_FEATURES]

            per_race = {}
            group_result = dict(jockey_id_coverage=float(np.mean(jockeys != '')), excluded_races=data['excluded_races'],
                                folds=[])
            quarter_block = _quarter_block(days, crossfit_blocks)
            with threadpool_limits(limits=threads):
                for fold in folds:
                    year = fold['validation_year']
                    fit = (days >= fold['train_start']) & (days < fold['stop_start'])
                    stop = (days >= fold['stop_start']) & (days < fold['validation_start'])
                    validation = (days >= fold['validation_start']) & (days < fold['validation_end'])
                    before = days < fold['validation_start']
                    fit_rows, stop_rows = _finisher_rows(data, fit), _finisher_rows(data, stop)
                    validation_rows = np.flatnonzero(validation)
                    fold_result = dict(validation_year=year, variants={})

                    def blanking(on_fit):
                        # Announcement dropout only on the rows being fitted.
                        return [(array, (mask & fit) if on_fit and mask is not None else None) for array, mask in blocks]

                    # 1. f0 without jockey information
                    f0_names = no_jockey_names + weight_names
                    f0_columns, f0_categorical = columns_for(no_jockey_names), categorical_for(no_jockey_names)
                    report(event='f0', group=group, validation_year=year)
                    f0, f0_best = fit_pl(data, Blocks(data, f0_columns, blanking(True)), f0_categorical, fit_rows, stop_rows,
                                         params, rounds=rounds, patience=patience)
                    f0_matrix = Blocks(data, f0_columns, blanking(False))
                    # 2. cross-fitted strengths before Y, f0 strengths in Y
                    strength = np.full(len(days), np.nan)
                    trainable = (days >= fold['train_start']) & before
                    for block in range(crossfit_blocks):
                        report(event='crossfit', group=group, validation_year=year, block=block)
                        rows = _finisher_rows(data, trainable & (quarter_block != block))
                        model, _ = fit_pl(data, Blocks(data, f0_columns, blanking(True)), f0_categorical, rows, None, params,
                                          rounds=rounds, patience=patience, fixed_rounds=f0_best)
                        target = np.flatnonzero(before & (quarter_block == block))
                        if len(target):
                            strength[target] = model.predict(f0_matrix(target), raw_score=True)
                    if len(validation_rows):
                        strength[validation_rows] = f0.predict(f0_matrix(validation_rows), num_iteration=f0_best, raw_score=True)
                    # 3. jockey timeline through the end of the validation year
                    upto = days < fold['validation_end']
                    span = np.flatnonzero(upto)
                    report(event='jockey_timeline', group=group, validation_year=year, rows=int(len(span)))
                    timeline_part = jockey_timeline(days[span], data['race_ids'][span], jockeys[span], horses[span],
                                                    data['finished'][span], data['positions'][span], strength[span],
                                                    half_life=half_life, short_half_life=short_half_life, prior_sd=prior_sd)
                    timeline = {key: np.full(len(days), np.nan) for key in timeline_part}
                    for key, values in timeline_part.items():
                        timeline[key][span] = values
                    theta_block = theta_feature_block(timeline, data['race_ids'])
                    theta_offset = np.nan_to_num(timeline['theta'])
                    validation_theta = timeline['theta'][validation_rows]
                    fold_result['theta_summary'] = dict(
                        validation_jockeys=int(len(set(jockeys[validation_rows]) - {''})),
                        theta_sd_across_validation_rows=float(np.nanstd(validation_theta)),
                        theta_p10=float(np.nanpercentile(validation_theta, 10)),
                        theta_p90=float(np.nanpercentile(validation_theta, 90)))
                    # completion probabilities (shared by every variant)
                    finisher, constant, finisher_rounds = _m1_finisher(
                        data, fit, stop, columns_for(base_names), categorical_for(base_names), threads=threads,
                        rounds=rounds, patience=patience, seed=seed)
                    completion = (np.full(len(validation_rows), constant) if finisher is None else
                                  finisher.predict(data['x'][np.ix_(validation_rows, columns_for(base_names))],
                                                   num_iteration=finisher_rounds))
                    # 4. variants
                    specs = {
                        'base': (base_names, [], None),
                        'nojockey': (no_jockey_names, [], None),
                        'theta_feature': (no_jockey_names, [(theta_block, None)], None),
                        'theta_offset': (no_jockey_names, [], theta_offset),
                    }
                    for name, (names, extra_blocks, offset) in specs.items():
                        report(event='variant', group=group, validation_year=year, variant=name)
                        columns, categorical = columns_for(names), categorical_for(names)
                        if name == 'nojockey':
                            model, best, matrix = f0, f0_best, f0_matrix
                        else:
                            train_matrix = Blocks(data, columns, blanking(True) + extra_blocks)
                            model, best = fit_pl(data, train_matrix, categorical, fit_rows, stop_rows, params,
                                                 rounds=rounds, patience=patience, offset=offset)
                            matrix = Blocks(data, columns, blanking(False) + extra_blocks)
                        stop_scores = model.predict(matrix(stop_rows), num_iteration=best, raw_score=True)
                        scores = model.predict(matrix(validation_rows), num_iteration=best, raw_score=True)
                        gamma = None
                        if offset is not None:
                            gamma = _fit_gamma(_structure(data, stop_rows), stop_scores, offset[stop_rows])
                            stop_scores = stop_scores + gamma * offset[stop_rows]
                            scores = scores + gamma * offset[validation_rows]
                        temperatures, stop_loss = fit_temperatures(_structure(data, stop_rows), stop_scores, STAGE_COUNT)
                        scored = score_races(data['race_ids'][validation_rows], data['positions'][validation_rows],
                                             data['finished'][validation_rows], data['race_numbers'][validation_rows],
                                             scores, completion, temperatures, samples=evaluation_samples, seed=seed)
                        entry = dict(best_iteration=best, temperatures=temperatures, stop_pl_nll=stop_loss,
                                     metrics=scored['metrics'])
                        if gamma is not None:
                            entry['gamma'] = gamma
                        if name == 'theta_feature':
                            importance = model.feature_importance(importance_type='gain', iteration=best)
                            all_names = names + weight_names + list(THETA_FEATURES)
                            order = np.argsort(-importance)
                            rank_of = {all_names[index]: position + 1 for position, index in enumerate(order)}
                            entry['theta_feature_gain_rank'] = {feature: rank_of[feature] for feature in THETA_FEATURES}
                        fold_result['variants'][name] = entry
                        per_race.setdefault(name, []).append(scored)
                        for race, a, b, c in zip(scored['race_ids'], scored['outcome'], scored['top3'], scored['win']):
                            losses.writerow([group, name, year, race, f'{a:.6f}', f'{b:.6f}', '' if np.isnan(c) else f'{c:.6f}'])
                        report(event='scored', group=group, validation_year=year, variant=name, best_iteration=best,
                               gamma=gamma, race_outcome_log_loss=scored['metrics']['race_outcome_log_loss'])
                    group_result['folds'].append(fold_result)
            summary = {}
            base = per_race['base']
            for name, scored_folds in per_race.items():
                row = {metric: float(np.mean([item['metrics'][metric] for item in scored_folds]))
                       for metric in ('race_outcome_log_loss', 'top3_race_log_loss', 'win_log_loss', 'top_pick_top3_rate')}
                if name != 'base':
                    combine = lambda key, folds_: np.concatenate([item[key] for item in folds_])
                    row['vs_base'] = dict(
                        outcome=paired_bootstrap(combine('outcome', scored_folds) - combine('outcome', base), seed=seed),
                        top3=paired_bootstrap(combine('top3', scored_folds) - combine('top3', base), seed=seed),
                        win=paired_bootstrap(combine('win', scored_folds) - combine('win', base), seed=seed),
                        every_fold_better=all(item['metrics']['race_outcome_log_loss'] < reference['metrics']['race_outcome_log_loss']
                                              for item, reference in zip(scored_folds, base)))
                summary[name] = row
            group_result['summary'] = summary
            group_result['best_variant'] = min(summary, key=lambda name: summary[name]['race_outcome_log_loss'])
            result['groups'][group] = group_result
            report(event='group_done', group=group, best=group_result['best_variant'],
                   summary={name: row['race_outcome_log_loss'] for name, row in summary.items()})
            del data
    finally:
        losses_file.close()
    result['seconds'] = round(time.time() - started, 1)
    (output / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='v3実験D: 馬の強さから騎手の腕（θ）を分離して効果を測ります。')
    parser.add_argument('--dataset', required=True, type=Path, help='prepare済みのfeatures.sqlite3')
    parser.add_argument('--output-dir', required=True, type=Path, help='新しい出力先（既存は不可）')
    parser.add_argument('--validation-years', nargs='+', type=int, default=list(DEFAULT_VALIDATION_YEARS))
    parser.add_argument('--train-start', default='2022-04-01')
    parser.add_argument('--test-start', default='2026-01-01')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=8000)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--half-life', type=float, default=365.0)
    parser.add_argument('--short-half-life', type=float, default=90.0)
    parser.add_argument('--prior-sd', type=float, default=.3)
    parser.add_argument('--evaluation-samples', type=int, default=1024)
    args = parser.parse_args(argv)
    from .cli import emit
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        emit(event='error', message='lightgbmがありません。scripts\\setup-ml.ps1 を実行してください。')
        return 2
    try:
        result = run_experiment(args.dataset, args.output_dir, validation_years=args.validation_years,
                                train_start=args.train_start, test_start=args.test_start, threads=args.threads,
                                rounds=args.rounds, patience=args.patience, half_life=args.half_life,
                                short_half_life=args.short_half_life, prior_sd=args.prior_sd,
                                evaluation_samples=args.evaluation_samples, report=emit)
    except (ValueError, OSError, sqlite3.Error, KeyError) as exc:
        emit(event='error', message=str(exc))
        return 2
    except KeyboardInterrupt:
        emit(event='interrupted', message='停止しました。別の出力先で再実行してください。')
        return 130
    emit(event='experiment_complete', report=str(args.output_dir / 'experiment.json'),
         best={group: value['best_variant'] for group, value in result['groups'].items()}, seconds=result['seconds'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
