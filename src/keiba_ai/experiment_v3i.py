"""V3 experiment I: track bias v3 (strength-based, PL) and a hybrid temperature.

Base per group = best configuration (central: speed + ratings; local:
bodyweight + speed + jockey theta + ratings).  Per fold:

1. Train the base model; out-of-fold strengths over the training window
   (calendar-quarter blocks, same rounds) and base-model strengths in the
   validation year are the offsets for the race-level bias estimates.
2. ``bias_v3`` builds input-side features (earlier races of the same day,
   course and surface) and label-side deviations (all other races of the day).
3. Variants:
   ``base``         the base model,
   ``bias_feature`` base + ``BIAS_V3_FEATURES`` (plan B),
   ``bias_offset``  base features, trained with init_score = label-side
                    deviation x actual position; predicted as
                    f + g_f * dev_front * (0.5 - x_hat) + g_i * dev_inner * (0.5 - gate),
                    with g fitted on the early-stopping period (plan A).
4. Each variant is scored with ``stop_power`` and ``stop_hybrid`` (T_1 free,
   T_j = a * j**b for j >= 2) temperatures, both fitted on the early-stopping
   period.  Baseline: base / stop_power.  The final-test period is never read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import sqlite3
import time

from .bias_v3 import BIAS_V3_FEATURES, day_features, load_geometry, race_betas
from .experiment_v3 import DEFAULT_VALIDATION_YEARS, _m1_finisher, _shift_months, experiment_feature_names
from .experiment_v3b import _finisher_rows, _structure
from .experiment_v3c import WEIGHT_FEATURES, _race_dropout, announced_features, load_announced
from .experiment_v3d import (JOCKEY_RATE_FEATURES, Blocks, _params, _quarter_block, fit_pl, jockey_timeline,
                             load_jockeys, theta_feature_block)
from .experiment_v3f import _crossfit_strength
from .experiment_v3h import fit_power_temperatures
from .ratings import rating_block
from .speed import load_speed_block

GROUPS = ('central_flat', 'local_flat')
WEIGHT_GROUPS = ('local_flat',)
THETA_GROUPS = ('local_flat',)
VARIANTS = ('base', 'bias_feature', 'bias_offset')
TEMPERATURES = ('stop_power', 'stop_hybrid')
BASELINE = 'base/stop_power'


def fit_hybrid_temperatures(structure, scores, *, width=18):
    """T_1 free; T_j = a * j**b for j >= 2."""
    import numpy as np
    from scipy.optimize import minimize
    from .ranking_v3 import staged_pl_loss
    stages = np.arange(1, width + 1, dtype=float)

    def temperatures(parameters):
        log_t1, log_a, b = parameters
        values = np.exp(log_a) * stages ** b
        values[0] = np.exp(log_t1)
        return values

    result = minimize(lambda parameters: staged_pl_loss(structure, scores, temperatures(parameters)),
                      np.asarray([np.log(.7), np.log(.7), .3]), method='L-BFGS-B',
                      bounds=[(-1.5, 1.8), (-1.5, 1.8), (-1.0, 1.0)])
    fitted = temperatures(result.x)
    return [float(value) for value in fitted], float(result.fun), dict(
        t1=float(np.exp(result.x[0])), a=float(np.exp(result.x[1])), b=float(result.x[2]))


def _fit_gammas(structure, base_scores, terms):
    import numpy as np
    from scipy.optimize import minimize
    terms = np.nan_to_num(terms)
    result = minimize(lambda gamma: structure.mean_loss(base_scores + terms @ gamma), np.zeros(terms.shape[1]),
                      method='L-BFGS-B', bounds=[(-1, 3)] * terms.shape[1])
    return result.x


def run_experiment(database, output, *, validation_years=DEFAULT_VALIDATION_YEARS, train_start='2022-04-01',
                   test_start='2026-01-01', stop_months=2, threads=2, rounds=8000, patience=150,
                   announce_dropout=.05, crossfit_blocks=3, half_life=365.0, short_half_life=90.0, prior_sd=.3,
                   decay=.8, evaluation_samples=1024, seed=42, report=None):
    import numpy as np
    from .features import FEATURE_NAMES, CATEGORICAL_FEATURES
    from .model import feature_revision, readonly
    from .rank_model import load_rank_group
    from .ranking_v3 import paired_bootstrap, score_races
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
    result = dict(experiment='v3i_track_bias_v3', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  dataset=str(database.resolve()), feature_revision=feature_revision(), folds=folds,
                  test_start=test_start, test_period_used=False, bias_features=list(BIAS_V3_FEATURES),
                  baseline=BASELINE,
                  settings=dict(threads=threads, rounds=rounds, patience=patience, crossfit_blocks=crossfit_blocks,
                                decay=decay, announce_dropout=announce_dropout, evaluation_samples=evaluation_samples,
                                seed=seed, variants=list(VARIANTS), temperatures=list(TEMPERATURES)),
                  groups={})
    output.mkdir(parents=True)
    losses_file = (output / 'race_losses.csv').open('w', encoding='utf-8', newline='')
    losses = csv.writer(losses_file)
    losses.writerow(['group', 'config', 'validation_year', 'race_id', 'outcome_log_loss', 'top3_log_loss', 'win_log_loss'])
    params = _params(threads, seed)
    expected_corner_index = FEATURE_NAMES.index('horse_group_recent5_last_corner_fraction')
    try:
        for group in GROUPS:
            report(event='loading_group', group=group)
            data = load_rank_group(database, group)
            race_ids = data['race_ids'] = np.asarray([identity[0] for identity in data['identities']])
            days = data['days']
            horses = np.asarray([identity[1] for identity in data['identities']], dtype=object)
            speed, _ = load_speed_block(database, group, data['identities'])
            geometry = load_geometry(database, group, data['identities'])
            expected_corner = data['x'][:, expected_corner_index].astype(float)
            common = []
            if group in WEIGHT_GROUPS:
                weight = announced_features(data, load_announced(database, group, data['identities']))[:, :len(WEIGHT_FEATURES)]
                common.append((weight, _race_dropout(race_ids, announce_dropout, seed)))
            common.append((speed, None))
            rating = rating_block(days, race_ids, horses, data['finished'], data['positions'])
            jockeys = load_jockeys(database, group, data['identities']) if group in THETA_GROUPS else None
            quarter_block = _quarter_block(days, crossfit_blocks)
            columns, categorical = columns_for(names), categorical_for(names)
            per_race = {}
            group_result = dict(excluded_races=data['excluded_races'],
                                geometry_coverage=dict(corner=float(np.isfinite(geometry['corner']).mean()),
                                                       gate=float(np.isfinite(geometry['gate']).mean())),
                                folds=[])
            with threadpool_limits(limits=threads):
                for fold in folds:
                    year = fold['validation_year']
                    fit = (days >= fold['train_start']) & (days < fold['stop_start'])
                    stop = (days >= fold['stop_start']) & (days < fold['validation_start'])
                    validation = (days >= fold['validation_start']) & (days < fold['validation_end'])
                    training = (days >= fold['train_start']) & (days < fold['validation_start'])
                    upto = days < fold['validation_end']
                    fit_rows, stop_rows = _finisher_rows(data, fit), _finisher_rows(data, stop)
                    validation_rows = np.flatnonzero(validation)
                    finisher, constant, finisher_rounds = _m1_finisher(
                        data, fit, stop, columns, categorical, threads=threads, rounds=rounds, patience=patience, seed=seed)
                    completion = (np.full(len(validation_rows), constant) if finisher is None else
                                  finisher.predict(data['x'][np.ix_(validation_rows, columns)], num_iteration=finisher_rounds))
                    blocks = list(common)
                    if group in THETA_GROUPS:
                        report(event='strength_for_theta', group=group, validation_year=year)
                        strength0 = _crossfit_strength(
                            data, columns_for(no_jockey), categorical_for(no_jockey), common, fold, quarter_block, params,
                            rounds=rounds, patience=patience, crossfit_blocks=crossfit_blocks, fit_mask=fit,
                            stop_mask=stop, report=report, group=group)
                        span = np.flatnonzero(upto)
                        part = jockey_timeline(days[span], race_ids[span], jockeys[span], horses[span],
                                               data['finished'][span], data['positions'][span], strength0[span],
                                               half_life=half_life, short_half_life=short_half_life, prior_sd=prior_sd)
                        timeline = {key: np.full(len(days), np.nan) for key in part}
                        for key, values in part.items():
                            timeline[key][span] = values
                        blocks.append((theta_feature_block(timeline, race_ids), None))
                    blocks.append((rating, None))
                    train_blocks = [(a, (m & fit) if m is not None else None) for a, m in blocks]
                    plain_blocks = [(a, None) for a, _ in blocks]
                    # 1. base model, out-of-fold strengths
                    report(event='base_model', group=group, validation_year=year)
                    base_model, base_best = fit_pl(data, Blocks(data, columns, train_blocks), categorical, fit_rows,
                                                   stop_rows, params, rounds=rounds, patience=patience)
                    base_matrix = Blocks(data, columns, plain_blocks)
                    strength = np.full(len(days), np.nan)
                    for block in range(crossfit_blocks):
                        report(event='oof_model', group=group, validation_year=year, block=block)
                        rows = _finisher_rows(data, training & (quarter_block != block))
                        model, _ = fit_pl(data, Blocks(data, columns, [(a, (m & training) if m is not None else None)
                                                                      for a, m in blocks]),
                                          categorical, rows, None, params, rounds=rounds, patience=patience,
                                          fixed_rounds=base_best)
                        target = np.flatnonzero(training & (quarter_block == block))
                        if len(target):
                            strength[target] = model.predict(base_matrix(target), raw_score=True)
                    if len(validation_rows):
                        strength[validation_rows] = base_model.predict(base_matrix(validation_rows), num_iteration=base_best,
                                                                       raw_score=True)
                    # 2. race betas and day features up to the end of the validation year
                    span = np.flatnonzero(upto)
                    report(event='race_bias', group=group, validation_year=year)
                    betas = race_betas(race_ids[span], data['finished'][span], data['positions'][span],
                                       geometry['corner'][span], geometry['gate'][span], strength[span])
                    feature_part, label_part = day_features(
                        days[span], race_ids[span], data['race_numbers'][span], geometry['course'][span],
                        geometry['surface'][span], geometry['distance'][span], expected_corner[span],
                        geometry['gate'][span], betas, decay=decay)
                    bias_block = np.full((len(days), len(BIAS_V3_FEATURES)), np.nan, dtype=np.float32)
                    bias_block[span] = feature_part
                    label_dev = np.zeros((len(days), 2))
                    label_dev[span] = label_part
                    u_actual = np.column_stack([.5 - geometry['corner'], .5 - geometry['gate']])
                    offset = np.nan_to_num((label_dev * u_actual).sum(axis=1))
                    offset[~training] = 0.0
                    measured = np.asarray([beta for beta, _ in betas.values()]) if betas else np.zeros((0, 2))
                    fold_result = dict(validation_year=year, base_best_iteration=base_best,
                                       races_with_beta=len(betas),
                                       beta_mean=measured.mean(0).tolist() if len(measured) else None,
                                       beta_sd=measured.std(0).tolist() if len(measured) else None,
                                       offset_sd_training=float(offset[training].std()), configs={})
                    # 3. variants
                    for name in VARIANTS:
                        report(event='variant', group=group, validation_year=year, variant=name)
                        gammas = None
                        if name == 'base':
                            model, best, matrix = base_model, base_best, base_matrix
                        elif name == 'bias_feature':
                            model, best = fit_pl(data, Blocks(data, columns, train_blocks + [(bias_block, None)]), categorical,
                                                 fit_rows, stop_rows, params, rounds=rounds, patience=patience)
                            matrix = Blocks(data, columns, plain_blocks + [(bias_block, None)])
                        else:
                            model, best = fit_pl(data, Blocks(data, columns, train_blocks), categorical, fit_rows, stop_rows,
                                                 params, rounds=rounds, patience=patience, offset=offset)
                            matrix = base_matrix
                        stop_scores = model.predict(matrix(stop_rows), num_iteration=best, raw_score=True)
                        scores = model.predict(matrix(validation_rows), num_iteration=best, raw_score=True)
                        if name == 'bias_offset':
                            terms = bias_block[:, [5, 6]].astype(float)
                            gammas = _fit_gammas(_structure(data, stop_rows), stop_scores, terms[stop_rows])
                            stop_scores = stop_scores + np.nan_to_num(terms[stop_rows]) @ gammas
                            scores = scores + np.nan_to_num(terms[validation_rows]) @ gammas
                        stop_structure = _structure(data, stop_rows)
                        fitted = dict(stop_power=fit_power_temperatures(stop_structure, stop_scores)[0],
                                      stop_hybrid=fit_hybrid_temperatures(stop_structure, stop_scores)[0])
                        entry = dict(best_iteration=best, gammas=None if gammas is None else gammas.tolist(), temperatures={})
                        for method in TEMPERATURES:
                            scored = score_races(race_ids[validation_rows], data['positions'][validation_rows],
                                                 data['finished'][validation_rows], data['race_numbers'][validation_rows],
                                                 scores, completion, fitted[method], samples=evaluation_samples, seed=seed)
                            key = f'{name}/{method}'
                            entry['temperatures'][method] = dict(values=fitted[method][:6], metrics=scored['metrics'])
                            per_race.setdefault(key, []).append(scored)
                            for race, a, b, c in zip(scored['race_ids'], scored['outcome'], scored['top3'], scored['win']):
                                losses.writerow([group, key, year, race, f'{a:.6f}', f'{b:.6f}', '' if np.isnan(c) else f'{c:.6f}'])
                            report(event='scored', group=group, validation_year=year, config=key,
                                   race_outcome_log_loss=scored['metrics']['race_outcome_log_loss'],
                                   win_log_loss=scored['metrics']['win_log_loss'])
                        if name == 'bias_feature':
                            importance = model.feature_importance(importance_type='gain', iteration=best)
                            order = np.argsort(-importance)
                            total = len(importance)
                            ranks = {int(index): position + 1 for position, index in enumerate(order)}
                            entry['bias_feature_gain_rank'] = {feature: ranks[total - len(BIAS_V3_FEATURES) + i]
                                                               for i, feature in enumerate(BIAS_V3_FEATURES)}
                        fold_result['configs'][name] = entry
                    group_result['folds'].append(fold_result)
            summary = {}
            base = per_race[BASELINE]
            for key, scored_folds in per_race.items():
                row = {metric: float(np.mean([item['metrics'][metric] for item in scored_folds]))
                       for metric in ('race_outcome_log_loss', 'top3_race_log_loss', 'win_log_loss', 'top_pick_top3_rate')}
                row['by_race_number'] = {}
                for bucket in ('1_4', '5_8', '9_plus'):
                    values = [item['metrics']['by_race_number'][bucket]['race_outcome_log_loss'] for item in scored_folds]
                    values = [value for value in values if value is not None]
                    row['by_race_number'][bucket] = float(np.mean(values)) if values else None
                if key != BASELINE:
                    combine = lambda field, folds_: np.concatenate([item[field] for item in folds_])
                    row['vs_base'] = dict(
                        outcome=paired_bootstrap(combine('outcome', scored_folds) - combine('outcome', base), seed=seed),
                        top3=paired_bootstrap(combine('top3', scored_folds) - combine('top3', base), seed=seed),
                        win=paired_bootstrap(combine('win', scored_folds) - combine('win', base), seed=seed),
                        every_fold_better=all(item['metrics']['race_outcome_log_loss'] < reference['metrics']['race_outcome_log_loss']
                                              for item, reference in zip(scored_folds, base)))
                summary[key] = row
            group_result['summary'] = summary
            group_result['best_config'] = min(summary, key=lambda key: summary[key]['race_outcome_log_loss'])
            result['groups'][group] = group_result
            report(event='group_done', group=group, best=group_result['best_config'],
                   summary={key: row['race_outcome_log_loss'] for key, row in summary.items()})
            del data, speed, rating
    finally:
        losses_file.close()
    result['seconds'] = round(time.time() - started, 1)
    (output / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='v3実験I: 馬の強さを基準にしたトラックバイアスと、1着を自由にした温度を比べます。')
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
         best={group: value['best_config'] for group, value in result['groups'].items()}, seconds=result['seconds'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
