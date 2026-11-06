"""V3 experiment E: speed figures with track variant and class-aware standards.

Base per group = best non-jockey configuration so far (tuned LightGBM-PL,
4 stage temperatures; local flat also uses the announced bodyweight block).
The ``speed`` variant adds ``speed.SPEED_FEATURES``.  Figures of day d only
enter features from day d+1 on.  Temperatures are fitted on the early-stopping
period; the final-test period is never read.
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
from .experiment_v3d import Blocks, _params, fit_pl
from .speed import SPEED_FEATURES, load_speed_block

GROUPS = ('central_flat', 'local_flat')
STAGE_COUNT = 4
WEIGHT_GROUPS = ('local_flat',)


def run_experiment(database, output, *, validation_years=DEFAULT_VALIDATION_YEARS, train_start='2022-04-01',
                   test_start='2026-01-01', stop_months=2, threads=2, rounds=8000, patience=150,
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

    names = experiment_feature_names()
    columns = [FEATURE_NAMES.index(name) for name in names]
    categorical = [index for index, name in enumerate(names) if name in CATEGORICAL_FEATURES]
    folds = []
    for year in validation_years:
        start = f'{year:04d}-01-01'
        stop_start = _shift_months(start, -stop_months)
        if stop_start <= train_start:
            raise ValueError(f'{year}年の検証foldの学習期間がありません。')
        folds.append(dict(validation_year=year, train_start=train_start, stop_start=stop_start,
                          validation_start=start, validation_end=f'{year + 1:04d}-01-01'))
    started = time.time()
    result = dict(experiment='v3e_speed_figures', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  dataset=str(database.resolve()), feature_revision=feature_revision(), folds=folds,
                  test_start=test_start, test_period_used=False, speed_features=list(SPEED_FEATURES),
                  settings=dict(threads=threads, rounds=rounds, patience=patience, stage_temperatures=STAGE_COUNT,
                                weight_feature_groups=list(WEIGHT_GROUPS), announce_dropout=announce_dropout,
                                evaluation_samples=evaluation_samples, seed=seed,
                                baseline='base (best non-jockey configuration per group)'),
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
            report(event='speed_figures', group=group)
            speed, diagnostics = load_speed_block(database, group, data['identities'])
            base_blocks, extra_names = [], []
            if group in WEIGHT_GROUPS:
                weight = announced_features(data, load_announced(database, group, data['identities']))[:, :len(WEIGHT_FEATURES)]
                base_blocks.append((weight, _race_dropout(data['race_ids'], announce_dropout, seed)))
                extra_names += list(WEIGHT_FEATURES)
            specs = {'base': (base_blocks, extra_names),
                     'speed': (base_blocks + [(speed, None)], extra_names + list(SPEED_FEATURES))}
            days = data['days']
            usable = days >= train_start
            group_result = dict(speed_diagnostics=diagnostics,
                                speed_coverage={name: float(np.isfinite(speed[usable, index]).mean())
                                                for index, name in enumerate(SPEED_FEATURES)},
                                excluded_races=data['excluded_races'], folds=[])
            per_race = {}
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
                    fold_result = dict(validation_year=year, variants={})
                    for name, (blocks, added) in specs.items():
                        report(event='training', group=group, validation_year=year, variant=name)
                        train_blocks = [(array, (mask & fit) if mask is not None else None) for array, mask in blocks]
                        plain_blocks = [(array, None) for array, _ in blocks]
                        model, best = fit_pl(data, Blocks(data, columns, train_blocks), categorical, fit_rows, stop_rows,
                                             params, rounds=rounds, patience=patience)
                        matrix = Blocks(data, columns, plain_blocks)
                        stop_scores = model.predict(matrix(stop_rows), num_iteration=best, raw_score=True)
                        temperatures, stop_loss = fit_temperatures(_structure(data, stop_rows), stop_scores, STAGE_COUNT)
                        scores = model.predict(matrix(validation_rows), num_iteration=best, raw_score=True)
                        scored = score_races(data['race_ids'][validation_rows], data['positions'][validation_rows],
                                             data['finished'][validation_rows], data['race_numbers'][validation_rows],
                                             scores, completion, temperatures, samples=evaluation_samples, seed=seed)
                        importance = model.feature_importance(importance_type='gain', iteration=best)
                        all_names = names + added
                        order = np.argsort(-importance)
                        rank_of = {all_names[index]: position + 1 for position, index in enumerate(order)}
                        fold_result['variants'][name] = dict(
                            best_iteration=best, temperatures=temperatures, stop_pl_nll=stop_loss, metrics=scored['metrics'],
                            top_gain_features=[dict(feature=all_names[i], gain=float(importance[i])) for i in order[:15]],
                            speed_feature_gain_rank={feature: rank_of[feature] for feature in SPEED_FEATURES if feature in rank_of})
                        per_race.setdefault(name, []).append(scored)
                        for race, a, b, c in zip(scored['race_ids'], scored['outcome'], scored['top3'], scored['win']):
                            losses.writerow([group, name, year, race, f'{a:.6f}', f'{b:.6f}', '' if np.isnan(c) else f'{c:.6f}'])
                        report(event='scored', group=group, validation_year=year, variant=name, best_iteration=best,
                               race_outcome_log_loss=scored['metrics']['race_outcome_log_loss'])
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
            del data, speed
    finally:
        losses_file.close()
    result['seconds'] = round(time.time() - started, 1)
    (output / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    from .priority import background
    background()
    parser = argparse.ArgumentParser(description='v3実験E: 馬場差・クラスを補正したスピード指数の効果を測ります。')
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
