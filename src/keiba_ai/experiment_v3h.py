"""V3 experiment H: how to estimate the stage temperatures.

The model per group is the best configuration so far (central flat:
speed + ratings; local flat: bodyweight + speed + jockey theta + ratings) and
is trained once per fold.  Only the temperature estimate differs:

=================  =========================================================
``stop_single``    one temperature, early-stopping period (last 2 months)
``stop_4``         4 stage temperatures, early-stopping period (current method)
``stop_4_shrunk``  as ``stop_4`` with a 10x stronger pull towards a common value
``stop_power``     T_j = T_1 * j**beta (two parameters), early-stopping period
``oof_4``          4 stages fitted on out-of-fold scores of the whole training
                   window (calendar-quarter blocks, same rounds as the model)
``oof_6``          6 stages, out-of-fold scores
=================  =========================================================

All methods use only races before the validation year.  Every method scores
the same validation predictions with common random numbers, so differences
are due to the temperatures alone.  The final-test period is never read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import sqlite3
import time

from .experiment_v3 import DEFAULT_VALIDATION_YEARS, _m1_finisher, _shift_months, experiment_feature_names
from .experiment_v3b import _finisher_rows, _structure
from .experiment_v3c import WEIGHT_FEATURES, _race_dropout, announced_features, load_announced
from .experiment_v3d import (JOCKEY_RATE_FEATURES, THETA_FEATURES, Blocks, _params, _quarter_block, fit_pl,
                             jockey_timeline, load_jockeys, theta_feature_block)
from .experiment_v3f import _crossfit_strength
from .ratings import RATING_FEATURES, rating_block
from .speed import SPEED_FEATURES, load_speed_block

GROUPS = ('central_flat', 'local_flat')
WEIGHT_GROUPS = ('local_flat',)
THETA_GROUPS = ('local_flat',)
METHODS = ('stop_single', 'stop_4', 'stop_4_shrunk', 'stop_power', 'oof_4', 'oof_6')
BASELINE = 'stop_4'


def fit_power_temperatures(structure, scores, *, width=18, bounds=((-1.5, 1.8), (-1.0, 1.0))):
    """T_j = T_1 * j**beta for stages j = 1..width; returns the per-stage list and loss."""
    import numpy as np
    from scipy.optimize import minimize
    from .ranking_v3 import staged_pl_loss
    stages = np.arange(1, width + 1, dtype=float)

    def temperatures(parameters):
        log_t1, beta = parameters
        return np.exp(log_t1) * stages ** beta

    result = minimize(lambda parameters: staged_pl_loss(structure, scores, temperatures(parameters)),
                      np.zeros(2), method='L-BFGS-B', bounds=bounds)
    fitted = temperatures(result.x)
    return [float(value) for value in fitted], float(result.fun), dict(t1=float(np.exp(result.x[0])), beta=float(result.x[1]))


def run_experiment(database, output, *, validation_years=DEFAULT_VALIDATION_YEARS, train_start='2022-04-01',
                   test_start='2026-01-01', stop_months=2, threads=2, rounds=8000, patience=150,
                   announce_dropout=.05, crossfit_blocks=3, half_life=365.0, short_half_life=90.0, prior_sd=.3,
                   evaluation_samples=1024, seed=42, report=None):
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

    names = experiment_feature_names()
    no_jockey = [name for name in names if name not in JOCKEY_RATE_FEATURES]

    def columns_for(selected):
        return [FEATURE_NAMES.index(name) for name in selected]

    def categorical_for(selected):
        return [index for index, name in enumerate(selected) if name in CATEGORICAL_FEATURES]

    folds = []
    for year in validation_years:
        start = f'{year:04d}-01-01'
        stop_start = _shift_months(start, -stop_months)
        if stop_start <= train_start:
            raise ValueError(f'{year}年の検証foldの学習期間がありません。')
        folds.append(dict(validation_year=year, train_start=train_start, stop_start=stop_start,
                          validation_start=start, validation_end=f'{year + 1:04d}-01-01'))
    started = time.time()
    result = dict(experiment='v3h_temperature_estimation', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  dataset=str(database.resolve()), feature_revision=feature_revision(), folds=folds,
                  test_start=test_start, test_period_used=False, methods=list(METHODS), baseline=BASELINE,
                  settings=dict(threads=threads, rounds=rounds, patience=patience, crossfit_blocks=crossfit_blocks,
                                weight_feature_groups=list(WEIGHT_GROUPS), theta_groups=list(THETA_GROUPS),
                                announce_dropout=announce_dropout, evaluation_samples=evaluation_samples, seed=seed,
                                model='central: speed + ratings, local: bodyweight + speed + jockey theta + ratings'),
                  groups={})
    output.mkdir(parents=True)
    losses_file = (output / 'race_losses.csv').open('w', encoding='utf-8', newline='')
    losses = csv.writer(losses_file)
    losses.writerow(['group', 'method', 'validation_year', 'race_id', 'outcome_log_loss', 'top3_log_loss', 'win_log_loss'])
    params = _params(threads, seed)
    try:
        for group in GROUPS:
            report(event='loading_group', group=group)
            data = load_rank_group(database, group)
            race_ids = data['race_ids'] = np.asarray([identity[0] for identity in data['identities']])
            days = data['days']
            horses = np.asarray([identity[1] for identity in data['identities']], dtype=object)
            speed, _ = load_speed_block(database, group, data['identities'])
            common = []
            if group in WEIGHT_GROUPS:
                weight = announced_features(data, load_announced(database, group, data['identities']))[:, :len(WEIGHT_FEATURES)]
                common.append((weight, _race_dropout(race_ids, announce_dropout, seed)))
            common.append((speed, None))
            report(event='horse_ratings', group=group)
            rating = rating_block(days, race_ids, horses, data['finished'], data['positions'])
            jockeys = load_jockeys(database, group, data['identities']) if group in THETA_GROUPS else None
            quarter_block = _quarter_block(days, crossfit_blocks)
            columns, categorical = columns_for(names), categorical_for(names)
            per_race = {}
            group_result = dict(excluded_races=data['excluded_races'], folds=[])
            with threadpool_limits(limits=threads):
                for fold in folds:
                    year = fold['validation_year']
                    fit = (days >= fold['train_start']) & (days < fold['stop_start'])
                    stop = (days >= fold['stop_start']) & (days < fold['validation_start'])
                    validation = (days >= fold['validation_start']) & (days < fold['validation_end'])
                    training = (days >= fold['train_start']) & (days < fold['validation_start'])
                    fit_rows, stop_rows = _finisher_rows(data, fit), _finisher_rows(data, stop)
                    validation_rows = np.flatnonzero(validation)
                    finisher, constant, finisher_rounds = _m1_finisher(
                        data, fit, stop, columns, categorical, threads=threads, rounds=rounds, patience=patience, seed=seed)
                    completion = (np.full(len(validation_rows), constant) if finisher is None else
                                  finisher.predict(data['x'][np.ix_(validation_rows, columns)], num_iteration=finisher_rounds))
                    blocks = list(common)
                    if group in THETA_GROUPS:
                        report(event='strength_for_theta', group=group, validation_year=year)
                        strength = _crossfit_strength(
                            data, columns_for(no_jockey), categorical_for(no_jockey), common, fold, quarter_block, params,
                            rounds=rounds, patience=patience, crossfit_blocks=crossfit_blocks, fit_mask=fit,
                            stop_mask=stop, report=report, group=group)
                        span = np.flatnonzero(days < fold['validation_end'])
                        part = jockey_timeline(days[span], race_ids[span], jockeys[span], horses[span],
                                               data['finished'][span], data['positions'][span], strength[span],
                                               half_life=half_life, short_half_life=short_half_life, prior_sd=prior_sd)
                        timeline = {key: np.full(len(days), np.nan) for key in part}
                        for key, values in part.items():
                            timeline[key][span] = values
                        blocks.append((theta_feature_block(timeline, race_ids), None))
                    blocks.append((rating, None))
                    report(event='training', group=group, validation_year=year)
                    model, best = fit_pl(data, Blocks(data, columns, [(a, (m & fit) if m is not None else None) for a, m in blocks]),
                                         categorical, fit_rows, stop_rows, params, rounds=rounds, patience=patience)
                    matrix = Blocks(data, columns, [(a, None) for a, _ in blocks])
                    stop_scores = model.predict(matrix(stop_rows), num_iteration=best, raw_score=True)
                    validation_scores = model.predict(matrix(validation_rows), num_iteration=best, raw_score=True)
                    # Out-of-fold scores over the whole training window, same number of rounds.
                    oof = np.full(len(days), np.nan)
                    for block in range(crossfit_blocks):
                        report(event='oof_model', group=group, validation_year=year, block=block)
                        rows = _finisher_rows(data, training & (quarter_block != block))
                        oof_model, _ = fit_pl(data, Blocks(data, columns, [(a, (m & training) if m is not None else None)
                                                                          for a, m in blocks]),
                                              categorical, rows, None, params, rounds=rounds, patience=patience,
                                              fixed_rounds=best)
                        target = _finisher_rows(data, training & (quarter_block == block))
                        if len(target):
                            oof[target] = oof_model.predict(matrix(target), raw_score=True)
                    oof_rows = _finisher_rows(data, training)
                    oof_rows = oof_rows[np.isfinite(oof[oof_rows])]
                    stop_structure, oof_structure = _structure(data, stop_rows), _structure(data, oof_rows)
                    temperatures, details = {}, {}
                    temperatures['stop_single'], details['stop_single'] = fit_temperatures(stop_structure, stop_scores, 1)
                    temperatures['stop_4'], details['stop_4'] = fit_temperatures(stop_structure, stop_scores, 4)
                    temperatures['stop_4_shrunk'], details['stop_4_shrunk'] = fit_temperatures(stop_structure, stop_scores, 4,
                                                                                                shrink=.05)
                    power, loss, shape = fit_power_temperatures(stop_structure, stop_scores)
                    temperatures['stop_power'], details['stop_power'] = power, dict(loss=loss, **shape)
                    temperatures['oof_4'], details['oof_4'] = fit_temperatures(oof_structure, oof[oof_rows], 4)
                    temperatures['oof_6'], details['oof_6'] = fit_temperatures(oof_structure, oof[oof_rows], 6)
                    fold_result = dict(validation_year=year, best_iteration=best, stop_races=int(stop_structure.races),
                                       oof_races=int(oof_structure.races), methods={})
                    for method in METHODS:
                        scored = score_races(race_ids[validation_rows], data['positions'][validation_rows],
                                             data['finished'][validation_rows], data['race_numbers'][validation_rows],
                                             validation_scores, completion, temperatures[method],
                                             samples=evaluation_samples, seed=seed)
                        shown = temperatures[method][:6]
                        fold_result['methods'][method] = dict(temperatures=shown, fit=details[method],
                                                              metrics=scored['metrics'])
                        per_race.setdefault(method, []).append(scored)
                        for race, a, b, c in zip(scored['race_ids'], scored['outcome'], scored['top3'], scored['win']):
                            losses.writerow([group, method, year, race, f'{a:.6f}', f'{b:.6f}', '' if np.isnan(c) else f'{c:.6f}'])
                        report(event='scored', group=group, validation_year=year, method=method,
                               temperatures=[round(value, 3) for value in shown],
                               race_outcome_log_loss=scored['metrics']['race_outcome_log_loss'],
                               win_log_loss=scored['metrics']['win_log_loss'])
                    group_result['folds'].append(fold_result)
            summary = {}
            base = per_race[BASELINE]
            for method, scored_folds in per_race.items():
                row = {metric: float(np.mean([item['metrics'][metric] for item in scored_folds]))
                       for metric in ('race_outcome_log_loss', 'top3_race_log_loss', 'win_log_loss', 'top_pick_top3_rate')}
                if method != BASELINE:
                    combine = lambda key, folds_: np.concatenate([item[key] for item in folds_])
                    row['vs_stop_4'] = dict(
                        outcome=paired_bootstrap(combine('outcome', scored_folds) - combine('outcome', base), seed=seed),
                        top3=paired_bootstrap(combine('top3', scored_folds) - combine('top3', base), seed=seed),
                        win=paired_bootstrap(combine('win', scored_folds) - combine('win', base), seed=seed),
                        every_fold_better=all(item['metrics']['race_outcome_log_loss'] < reference['metrics']['race_outcome_log_loss']
                                              for item, reference in zip(scored_folds, base)))
                summary[method] = row
            group_result['summary'] = summary
            group_result['best_method'] = min(summary, key=lambda method: summary[method]['race_outcome_log_loss'])
            result['groups'][group] = group_result
            report(event='group_done', group=group, best=group_result['best_method'],
                   summary={method: row['race_outcome_log_loss'] for method, row in summary.items()})
            del data, speed, rating
    finally:
        losses_file.close()
    result['seconds'] = round(time.time() - started, 1)
    (output / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    from .priority import background
    background()
    parser = argparse.ArgumentParser(description='v3実験H: 段ごとの温度の推定方法を比べます。')
    parser.add_argument('--dataset', required=True, type=Path, help='prepare済みのfeatures.sqlite3')
    parser.add_argument('--output-dir', required=True, type=Path, help='新しい出力先（既存は不可）')
    parser.add_argument('--validation-years', nargs='+', type=int, default=list(DEFAULT_VALIDATION_YEARS))
    parser.add_argument('--train-start', default='2022-04-01')
    parser.add_argument('--test-start', default='2026-01-01')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=8000)
    parser.add_argument('--patience', type=int, default=150)
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
                                rounds=args.rounds, patience=args.patience,
                                evaluation_samples=args.evaluation_samples, report=emit)
    except (ValueError, OSError, sqlite3.Error, KeyError) as exc:
        emit(event='error', message=str(exc))
        return 2
    except KeyboardInterrupt:
        emit(event='interrupted', message='停止しました。別の出力先で再実行してください。')
        return 130
    emit(event='experiment_complete', report=str(args.output_dir / 'experiment.json'),
         best={group: value['best_method'] for group, value in result['groups'].items()}, seconds=result['seconds'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
