"""V3 experiment F: speed figures + jockey theta together, and race-relative features (design 6.2 G).

Baseline per group = experiment E ``speed`` (tuned LightGBM-PL, 4 stage
temperatures, 89 features + 11 speed figures; local flat also has the
announced bodyweight block).

* ``speed_rel``        + race-relative block (both groups)
* ``speed_theta``      + jockey theta features (local flat only; jockey rates kept)
* ``speed_theta_rel``  + both (local flat only)

The race-relative block holds, for a set of key ability columns, the
within-race z-score and the within-race rank fraction.  It only uses the
same race's already-causal columns, so it adds no new information source.
For theta, the horse strength (no jockey features, with speed figures) is
cross-fitted by calendar-quarter blocks before the validation year, exactly
as in experiment D.  The final-test period is never read.
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
from .speed import SPEED_FEATURES, load_speed_block

GROUPS = ('central_flat', 'local_flat')
STAGE_COUNT = 4
WEIGHT_GROUPS = ('local_flat',)
THETA_GROUPS = ('local_flat',)
RELATIVE_BASE_COLUMNS = (
    'horse_group_recent5_mean_time_gap_per_km', 'horse_group_previous_mean_time_gap_per_km',
    'horse_group_recent5_winner_gap_per_km', 'horse_group_recent5_last3f_vs_mean',
    'horse_group_previous_last3f_vs_mean', 'horse_group_recent5_last_corner_fraction',
    'horse_course_surface_distance_top3_rate', 'horse_surface_distance_top3_rate',
    'jockey_prior_top3_rate', 'trainer_prior_top3_rate', 'carried_weight_kg', 'age',
    'horse_days_since_last_start',
)
RELATIVE_SPEED_COLUMNS = ('speed_fig_previous', 'speed_fig_same_surface_band_best', 'speed_fig_trend')
RELATIVE_FEATURES = tuple(f'{name}_race_{kind}' for name in RELATIVE_BASE_COLUMNS + RELATIVE_SPEED_COLUMNS
                          for kind in ('z', 'rank'))


def race_relative(values, race_ids):
    """Per column: within-race z-score and rank fraction (0 = smallest); missing stays missing."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    out = np.full((len(values), values.shape[1] * 2), np.nan, dtype=np.float32)
    if not len(values):
        return out
    boundaries = np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block = values[start:end]
        for column in range(block.shape[1]):
            series = block[:, column]
            known = np.isfinite(series)
            count = int(known.sum())
            if count < 2:
                continue
            spread = series[known].std()
            z = np.full(len(series), np.nan)
            z[known] = (series[known] - series[known].mean()) / spread if spread > 0 else 0.0
            order = series[known].argsort(kind='stable')
            ranks = np.empty(count)
            ranks[order] = np.arange(count)
            # Average ranks for ties so equal values get equal ranks.
            for value in np.unique(series[known]):
                same = series[known] == value
                if same.sum() > 1:
                    ranks[same] = ranks[same].mean()
            rank = np.full(len(series), np.nan)
            rank[known] = ranks / (count - 1)
            out[start:end, 2 * column] = z
            out[start:end, 2 * column + 1] = rank
    return out


def _crossfit_strength(data, columns, categorical, blocks, fold, quarter_block, params, *, rounds, patience,
                       crossfit_blocks, fit_mask, stop_mask, report, group):
    """Strength without jockey features: cross-fitted before the validation year, f0 inside it."""
    import numpy as np
    days = data['days']
    fit_rows, stop_rows = _finisher_rows(data, fit_mask), _finisher_rows(data, stop_mask)
    train_blocks = [(array, (mask & fit_mask) if mask is not None else None) for array, mask in blocks]
    plain = Blocks(data, columns, [(array, None) for array, _ in blocks])
    f0, f0_best = fit_pl(data, Blocks(data, columns, train_blocks), categorical, fit_rows, stop_rows, params,
                         rounds=rounds, patience=patience)
    strength = np.full(len(days), np.nan)
    before = days < fold['validation_start']
    trainable = (days >= fold['train_start']) & before
    for block in range(crossfit_blocks):
        report(event='crossfit', group=group, validation_year=fold['validation_year'], block=block)
        rows = _finisher_rows(data, trainable & (quarter_block != block))
        model, _ = fit_pl(data, Blocks(data, columns, train_blocks), categorical, rows, None, params,
                          rounds=rounds, patience=patience, fixed_rounds=f0_best)
        target = np.flatnonzero(before & (quarter_block == block))
        if len(target):
            strength[target] = model.predict(plain(target), raw_score=True)
    inside = np.flatnonzero((days >= fold['validation_start']) & (days < fold['validation_end']))
    if len(inside):
        strength[inside] = f0.predict(plain(inside), num_iteration=f0_best, raw_score=True)
    return strength


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
    result = dict(experiment='v3f_speed_theta_relative', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  dataset=str(database.resolve()), feature_revision=feature_revision(), folds=folds,
                  test_start=test_start, test_period_used=False, relative_features=list(RELATIVE_FEATURES),
                  settings=dict(threads=threads, rounds=rounds, patience=patience, stage_temperatures=STAGE_COUNT,
                                weight_feature_groups=list(WEIGHT_GROUPS), theta_groups=list(THETA_GROUPS),
                                announce_dropout=announce_dropout, crossfit_blocks=crossfit_blocks,
                                half_life_days=half_life, short_half_life_days=short_half_life, prior_sd=prior_sd,
                                evaluation_samples=evaluation_samples, seed=seed,
                                baseline='speed (experiment E best configuration)'),
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
            race_ids = data['race_ids'] = np.asarray([identity[0] for identity in data['identities']])
            days = data['days']
            speed, speed_diagnostics = load_speed_block(database, group, data['identities'])
            common, common_names = [], []
            if group in WEIGHT_GROUPS:
                weight = announced_features(data, load_announced(database, group, data['identities']))[:, :len(WEIGHT_FEATURES)]
                common.append((weight, _race_dropout(race_ids, announce_dropout, seed)))
                common_names += list(WEIGHT_FEATURES)
            common.append((speed, None))
            common_names += list(SPEED_FEATURES)
            relative_inputs = np.concatenate(
                [data['x'][:, columns_for(RELATIVE_BASE_COLUMNS)].astype(float),
                 speed[:, [SPEED_FEATURES.index(name) for name in RELATIVE_SPEED_COLUMNS]].astype(float)], axis=1)
            relative = race_relative(relative_inputs, race_ids)
            del relative_inputs
            variants = ['speed', 'speed_rel'] + (['speed_theta', 'speed_theta_rel'] if group in THETA_GROUPS else [])
            jockeys = load_jockeys(database, group, data['identities']) if group in THETA_GROUPS else None
            horses = np.asarray([identity[1] for identity in data['identities']], dtype=object)
            quarter_block = _quarter_block(days, crossfit_blocks)
            columns, categorical = columns_for(names), categorical_for(names)
            per_race = {}
            group_result = dict(speed_diagnostics=speed_diagnostics, excluded_races=data['excluded_races'],
                                variants=variants, folds=[])
            with threadpool_limits(limits=threads):
                for fold in folds:
                    year = fold['validation_year']
                    fit = (days >= fold['train_start']) & (days < fold['stop_start'])
                    stop = (days >= fold['stop_start']) & (days < fold['validation_start'])
                    validation = (days >= fold['validation_start']) & (days < fold['validation_end'])
                    fit_rows, stop_rows = _finisher_rows(data, fit), _finisher_rows(data, stop)
                    validation_rows = np.flatnonzero(validation)
                    finisher, constant, finisher_rounds = _m1_finisher(
                        data, fit, stop, columns, categorical, threads=threads, rounds=rounds, patience=patience, seed=seed)
                    completion = (np.full(len(validation_rows), constant) if finisher is None else
                                  finisher.predict(data['x'][np.ix_(validation_rows, columns)], num_iteration=finisher_rounds))
                    theta = None
                    fold_result = dict(validation_year=year, variants={})
                    if group in THETA_GROUPS:
                        report(event='strength_for_theta', group=group, validation_year=year)
                        strength = _crossfit_strength(
                            data, columns_for(no_jockey), categorical_for(no_jockey), common, fold, quarter_block, params,
                            rounds=rounds, patience=patience, crossfit_blocks=crossfit_blocks, fit_mask=fit,
                            stop_mask=stop, report=report, group=group)
                        span = np.flatnonzero(days < fold['validation_end'])
                        timeline_part = jockey_timeline(days[span], race_ids[span], jockeys[span], horses[span],
                                                        data['finished'][span], data['positions'][span], strength[span],
                                                        half_life=half_life, short_half_life=short_half_life,
                                                        prior_sd=prior_sd)
                        timeline = {key: np.full(len(days), np.nan) for key in timeline_part}
                        for key, values in timeline_part.items():
                            timeline[key][span] = values
                        theta = theta_feature_block(timeline, race_ids)
                        fold_result['theta_sd_validation'] = float(np.nanstd(timeline['theta'][validation_rows]))
                    for name in variants:
                        blocks, added = list(common), list(common_names)
                        if 'theta' in name:
                            blocks.append((theta, None))
                            added += list(THETA_FEATURES)
                        if name.endswith('rel'):
                            blocks.append((relative, None))
                            added += list(RELATIVE_FEATURES)
                        report(event='training', group=group, validation_year=year, variant=name)
                        train_blocks = [(array, (mask & fit) if mask is not None else None) for array, mask in blocks]
                        model, best = fit_pl(data, Blocks(data, columns, train_blocks), categorical, fit_rows, stop_rows,
                                             params, rounds=rounds, patience=patience)
                        matrix = Blocks(data, columns, [(array, None) for array, _ in blocks])
                        stop_scores = model.predict(matrix(stop_rows), num_iteration=best, raw_score=True)
                        temperatures, stop_loss = fit_temperatures(_structure(data, stop_rows), stop_scores, STAGE_COUNT)
                        scores = model.predict(matrix(validation_rows), num_iteration=best, raw_score=True)
                        scored = score_races(race_ids[validation_rows], data['positions'][validation_rows],
                                             data['finished'][validation_rows], data['race_numbers'][validation_rows],
                                             scores, completion, temperatures, samples=evaluation_samples, seed=seed)
                        importance = model.feature_importance(importance_type='gain', iteration=best)
                        all_names = names + added
                        order = np.argsort(-importance)
                        rank_of = {}
                        for position, index in enumerate(order):
                            rank_of.setdefault(all_names[index], position + 1)
                        entry = dict(best_iteration=best, temperatures=temperatures, stop_pl_nll=stop_loss,
                                     metrics=scored['metrics'],
                                     top_gain_features=[dict(feature=all_names[i], gain=float(importance[i])) for i in order[:20]])
                        if name.endswith('rel'):
                            entry['relative_feature_gain_rank'] = {feature: rank_of[feature] for feature in RELATIVE_FEATURES}
                        fold_result['variants'][name] = entry
                        per_race.setdefault(name, []).append(scored)
                        for race, a, b, c in zip(scored['race_ids'], scored['outcome'], scored['top3'], scored['win']):
                            losses.writerow([group, name, year, race, f'{a:.6f}', f'{b:.6f}', '' if np.isnan(c) else f'{c:.6f}'])
                        report(event='scored', group=group, validation_year=year, variant=name, best_iteration=best,
                               race_outcome_log_loss=scored['metrics']['race_outcome_log_loss'])
                    group_result['folds'].append(fold_result)
            summary = {}
            base = per_race['speed']
            for name, scored_folds in per_race.items():
                row = {metric: float(np.mean([item['metrics'][metric] for item in scored_folds]))
                       for metric in ('race_outcome_log_loss', 'top3_race_log_loss', 'win_log_loss', 'top_pick_top3_rate')}
                if name != 'speed':
                    combine = lambda key, folds_: np.concatenate([item[key] for item in folds_])
                    row['vs_speed'] = dict(
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
            del data, speed, relative
    finally:
        losses_file.close()
    result['seconds'] = round(time.time() - started, 1)
    (output / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='v3実験F: 指数＋騎手θの組み合わせと、レース内の相対化特徴量の効果を測ります。')
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
         best={group: value['best_variant'] for group, value in result['groups'].items()}, seconds=result['seconds'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
