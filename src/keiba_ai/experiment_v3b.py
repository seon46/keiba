"""V3 experiment B: stage-wise temperatures, pooled flat model, LightGBM tuning.

Starting point is M1 of ``experiment_v3`` (LightGBM + Plackett-Luce on all
finishing stages, 89 features without odds).  This experiment compares, on the
same rolling 2023/2024/2025 validation folds:

* structure: one model per group (``separate``) or one model for central and
  local flat racing together with an ``is_central`` flag (``pooled``);
* LightGBM hyperparameters: defaults vs an Optuna search (the structure is a
  searched choice too).  The search objective is the mean validation PL
  negative log likelihood over folds and groups (no sampling, fast);
* temperatures: one temperature, or stage-wise temperatures (1st, 2nd, 3rd,
  4th+), both fitted by maximum likelihood on the early-stopping period at the
  end of each training window -- never on the validation year.

Every configuration is scored per race with common random numbers, and paired
bootstrap intervals are reported against ``separate_default`` with one
temperature (the previous M1).  The final-test period is never read.
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

GROUPS = ('central_flat', 'local_flat')
STAGE_COUNT = 4
DEFAULT_PARAMS = dict(learning_rate=.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=.8,
                      lambda_l2=5.0)


def _params(threads, seed, tuned):
    params = dict(max_bin=255, num_threads=threads, verbosity=-1, seed=seed, deterministic=True,
                  force_row_wise=True, metric='None')
    params.update(DEFAULT_PARAMS)
    params.update(tuned)
    return params


def load_flat(database, groups=GROUPS):
    """Concatenate the rank data of several groups; races stay contiguous."""
    import numpy as np
    from .rank_model import load_rank_group
    parts = []
    for group in groups:
        part = load_rank_group(database, group)
        part['group_name'] = np.full(len(part['identities']), group)
        parts.append(part)
    combined = {}
    for key in ('x', 'days', 'finished', 'positions', 'field_sizes', 'race_numbers', 'group_name'):
        combined[key] = np.concatenate([part[key] for part in parts])
        for part in parts:
            del part[key]
    combined['identities'] = [identity for part in parts for identity in part['identities']]
    combined['excluded_races'] = {group: part['excluded_races'] for group, part in zip(groups, parts)}
    combined['is_central'] = (combined['group_name'] == 'central_flat').astype(np.float32)
    combined['race_ids'] = np.asarray([identity[0] for identity in combined['identities']])
    return combined


class Matrix:
    def __init__(self, data, columns):
        self.data, self.columns = data, columns

    def __call__(self, rows):
        import numpy as np
        block = self.data['x'][np.ix_(rows, self.columns)]
        return np.concatenate([block, self.data['is_central'][rows, None]], axis=1)


def _finisher_rows(data, mask):
    import numpy as np
    rows = np.flatnonzero(mask & data['finished'])
    if not len(rows):  # e.g. a segment with no race in the period
        return rows
    keys = data['race_ids'][rows]
    change = np.r_[True, keys[1:] != keys[:-1]]
    code = np.cumsum(change) - 1
    keep = np.bincount(code)[code] >= 2
    return rows[keep]


def _structure(data, rows):
    return PLStructure(data['race_ids'][rows], data['positions'][rows])


def fit_strength(data, matrix, categorical, fit_mask, stop_mask, params, *, rounds, patience):
    import lightgbm as lgb
    fit_rows, stop_rows = _finisher_rows(data, fit_mask), _finisher_rows(data, stop_mask)
    if not len(fit_rows) or not len(stop_rows):
        raise ValueError('学習または早期終了用のレースが不足しています。')
    fit_structure, stop_structure = _structure(data, fit_rows), _structure(data, stop_rows)
    train_set = lgb.Dataset(matrix(fit_rows), categorical_feature=categorical, free_raw_data=False)
    stop_set = lgb.Dataset(matrix(stop_rows), categorical_feature=categorical, reference=train_set,
                           free_raw_data=False)
    train_set.pl_structure, stop_set.pl_structure = fit_structure, stop_structure
    booster = lgb.train(dict(params, objective=pl_objective(fit_structure)), train_set, num_boost_round=rounds,
                        valid_sets=[stop_set], valid_names=['stop'], feval=pl_metric,
                        callbacks=[lgb.early_stopping(patience, verbose=False)])
    return booster, int(booster.best_iteration or booster.current_iteration())


class FoldContext:
    """Row masks of one fold, per group."""

    def __init__(self, data, fold, groups):
        days = data['days']
        self.fold = fold
        self.fit = (days >= fold['train_start']) & (days < fold['stop_start'])
        self.stop = (days >= fold['stop_start']) & (days < fold['validation_start'])
        self.validation = (days >= fold['validation_start']) & (days < fold['validation_end'])
        self.groups = {group: data['group_name'] == group for group in groups}


def run_config(data, matrix, categorical, context, structure, params, *, rounds, patience):
    """Train one configuration on one fold; return per-group scores."""
    import numpy as np
    result = {}
    if structure == 'pooled':
        booster, best = fit_strength(data, matrix, categorical, context.fit, context.stop, params,
                                     rounds=rounds, patience=patience)
        models = {group: (booster, best) for group in context.groups}
    else:
        models = {group: fit_strength(data, matrix, categorical, context.fit & member, context.stop & member, params,
                                      rounds=rounds, patience=patience)
                  for group, member in context.groups.items()}
    for group, member in context.groups.items():
        booster, best = models[group]
        validation_rows = np.flatnonzero(context.validation & member)
        stop_rows = _finisher_rows(data, context.stop & member)
        validation_finishers = _finisher_rows(data, context.validation & member)
        predict = lambda rows: booster.predict(matrix(rows), num_iteration=best, raw_score=True)
        validation_scores = predict(validation_rows)
        lookup = dict(zip(validation_rows.tolist(), validation_scores))
        result[group] = dict(best_iteration=best, validation_rows=validation_rows,
                             validation_scores=validation_scores,
                             stop_rows=stop_rows, stop_scores=predict(stop_rows),
                             validation_pl_nll=_structure(data, validation_finishers).mean_loss(
                                 np.asarray([lookup[row] for row in validation_finishers.tolist()])))
    return result


def _tuning_space(trial):
    return dict(structure=trial.suggest_categorical('structure', ['separate', 'pooled']),
                learning_rate=trial.suggest_float('learning_rate', .02, .1, log=True),
                num_leaves=trial.suggest_int('num_leaves', 15, 255, log=True),
                min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 20, 2000, log=True),
                feature_fraction=trial.suggest_float('feature_fraction', .4, 1.0),
                bagging_fraction=trial.suggest_float('bagging_fraction', .6, 1.0),
                lambda_l2=trial.suggest_float('lambda_l2', .01, 50, log=True))


def run_experiment(database, output, *, validation_years=DEFAULT_VALIDATION_YEARS, train_start='2022-04-01',
                   test_start='2026-01-01', stop_months=2, threads=2, rounds=5000, patience=100,
                   tuning_trials=20, tuning_hours=4.0, evaluation_samples=1024, seed=42, report=None):
    import numpy as np
    from .features import FEATURE_NAMES, CATEGORICAL_FEATURES
    from .model import feature_revision, readonly
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
    report(event='loading', groups=list(GROUPS))
    data = load_flat(database)
    matrix = Matrix(data, columns)
    contexts = [FoldContext(data, fold, GROUPS) for fold in folds]
    for context in contexts:
        for group, member in context.groups.items():
            if (context.fit & member).sum() < 100 or (context.validation & member).sum() < 50:
                raise ValueError(f'{group} の {context.fold["validation_year"]}年foldのデータが不足しています。')
    started = time.time()
    result = dict(experiment='v3b_temperature_pooling_tuning', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  dataset=str(database.resolve()), feature_revision=feature_revision(),
                  feature_names=names + ['is_central'], folds=folds, test_start=test_start, test_period_used=False,
                  settings=dict(threads=threads, rounds=rounds, patience=patience, default_params=DEFAULT_PARAMS,
                                tuning_trials=tuning_trials, tuning_hours=tuning_hours, stage_temperatures=STAGE_COUNT,
                                temperature_fit='maximum likelihood on the early-stopping period (never validation)',
                                evaluation_samples=evaluation_samples, seed=seed,
                                baseline='separate_default with one temperature (previous M1)'),
                  excluded_races=data['excluded_races'], tuning=None, groups={})

    # ------------------------------------------------------------ tuning
    tuned = None
    with threadpool_limits(limits=threads):
        if tuning_trials > 0:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                space = _tuning_space(trial)
                structure = space.pop('structure')
                params = _params(threads, seed, dict(space, bagging_freq=1))
                values = []
                for step, context in enumerate(contexts):
                    scored = run_config(data, matrix, categorical, context, structure, params,
                                        rounds=rounds, patience=patience)
                    values.append(float(np.mean([item['validation_pl_nll'] for item in scored.values()])))
                    trial.set_user_attr(f'fold_{context.fold["validation_year"]}',
                                        {group: item['validation_pl_nll'] for group, item in scored.items()})
                    trial.set_user_attr(f'iterations_{context.fold["validation_year"]}',
                                        {group: item['best_iteration'] for group, item in scored.items()})
                    trial.report(float(np.mean(values)), step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                value = float(np.mean(values))
                report(event='tuning_trial', number=trial.number, value=value, structure=structure,
                       params={key: params[key] for key in space}, elapsed_minutes=round((time.time() - started) / 60, 1))
                return value

            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed),
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
            # The defaults are the first trial, so tuning can only match or beat them on the search objective.
            study.enqueue_trial(dict(structure='separate', bagging_fraction=1.0, **DEFAULT_PARAMS))
            study.enqueue_trial(dict(structure='pooled', bagging_fraction=1.0, **DEFAULT_PARAMS))
            study.optimize(objective, n_trials=tuning_trials, timeout=tuning_hours * 3600)
            trials = [dict(number=trial.number, state=trial.state.name, value=trial.value, params=trial.params,
                           **trial.user_attrs) for trial in study.trials]
            completed = [trial for trial in study.trials if trial.state.name == 'COMPLETE']
            if completed:
                best = study.best_trial
                tuned = dict(best.params)
                result['tuning'] = dict(best_number=best.number, best_value=best.value, best_params=tuned,
                                        trials=trials, objective='mean validation PL NLL over folds and groups')
            else:
                result['tuning'] = dict(trials=trials, note='完了したtrialがありません')

        # ------------------------------------------------------- final scoring
        configs = {'separate_default': ('separate', {}), 'pooled_default': ('pooled', {})}
        if tuned is not None:
            space = dict(tuned)
            structure = space.pop('structure')
            if not (space.get('bagging_fraction') == 1.0 and all(space.get(key) == value for key, value in DEFAULT_PARAMS.items())):
                configs['tuned'] = (structure, dict(space, bagging_freq=1))
        per_race = {group: {} for group in GROUPS}
        details = {group: [] for group in GROUPS}
        for context in contexts:
            year = context.fold['validation_year']
            completion = {}
            for group, member in context.groups.items():
                report(event='finisher', group=group, validation_year=year)
                finisher, constant, finisher_rounds = _m1_finisher(
                    data, context.fit & member, context.stop & member, columns, categorical[:],
                    threads=threads, rounds=rounds, patience=patience, seed=seed)
                rows = np.flatnonzero(context.validation & member)
                x = data['x'][np.ix_(rows, columns)]
                completion[group] = (np.full(len(rows), constant) if finisher is None
                                     else finisher.predict(x, num_iteration=finisher_rounds))
            fold_details = {group: dict(validation_year=year, configs={}) for group in GROUPS}
            for name, (structure, overrides) in configs.items():
                report(event='final_config', config=name, structure=structure, validation_year=year)
                params = _params(threads, seed, overrides)
                scored = run_config(data, matrix, categorical, context, structure, params,
                                    rounds=rounds, patience=patience)
                for group, item in scored.items():
                    stop_structure = _structure(data, item['stop_rows'])
                    single, single_loss = fit_temperatures(stop_structure, item['stop_scores'], 1)
                    staged, staged_loss = fit_temperatures(stop_structure, item['stop_scores'], STAGE_COUNT)
                    rows = item['validation_rows']
                    entry = dict(structure=structure, best_iteration=item['best_iteration'],
                                 validation_pl_nll=item['validation_pl_nll'],
                                 stop_pl_nll=dict(single=single_loss, staged=staged_loss),
                                 temperatures=dict(single=single, staged=staged), metrics={})
                    for label, temperatures in (('single_t', single), ('staged_t', staged)):
                        scored_races = score_races(data['race_ids'][rows], data['positions'][rows], data['finished'][rows],
                                                   data['race_numbers'][rows], item['validation_scores'],
                                                   completion[group], temperatures, samples=evaluation_samples, seed=seed)
                        key = f'{name}/{label}'
                        entry['metrics'][label] = scored_races['metrics']
                        per_race[group].setdefault(key, []).append(scored_races)
                    fold_details[group]['configs'][name] = entry
                    report(event='scored', config=name, group=group, validation_year=year,
                           single_t=entry['metrics']['single_t']['race_outcome_log_loss'],
                           staged_t=entry['metrics']['staged_t']['race_outcome_log_loss'],
                           temperatures=entry['temperatures'])
            for group in GROUPS:
                details[group].append(fold_details[group])

    # ------------------------------------------------------------- summary
    baseline_key = 'separate_default/single_t'
    output.mkdir(parents=True)
    with (output / 'race_losses.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['group', 'config', 'race_id', 'outcome_log_loss', 'top3_log_loss', 'win_log_loss'])
        for group in GROUPS:
            for key, folds_scored in per_race[group].items():
                for scored_races in folds_scored:
                    for race, a, b, c in zip(scored_races['race_ids'], scored_races['outcome'],
                                             scored_races['top3'], scored_races['win']):
                        writer.writerow([group, key, race, f'{a:.6f}', f'{b:.6f}', '' if np.isnan(c) else f'{c:.6f}'])
    for group in GROUPS:
        summary = {}
        base = per_race[group][baseline_key]
        for key, folds_scored in per_race[group].items():
            outcome = np.concatenate([item['outcome'] for item in folds_scored])
            top3 = np.concatenate([item['top3'] for item in folds_scored])
            win = np.concatenate([item['win'] for item in folds_scored])
            row = dict(race_outcome_log_loss_fold_mean=float(np.mean([item['metrics']['race_outcome_log_loss'] for item in folds_scored])),
                       top3_race_log_loss_fold_mean=float(np.mean([item['metrics']['top3_race_log_loss'] for item in folds_scored])),
                       win_log_loss_fold_mean=float(np.mean([item['metrics']['win_log_loss'] for item in folds_scored])),
                       top_pick_top3_rate_fold_mean=float(np.mean([item['metrics']['top_pick_top3_rate'] for item in folds_scored])))
            if key != baseline_key:
                base_outcome = np.concatenate([item['outcome'] for item in base])
                base_top3 = np.concatenate([item['top3'] for item in base])
                base_win = np.concatenate([item['win'] for item in base])
                row['vs_baseline'] = dict(
                    outcome=paired_bootstrap(outcome - base_outcome, seed=seed),
                    top3=paired_bootstrap(top3 - base_top3, seed=seed),
                    win=paired_bootstrap(win - base_win, seed=seed),
                    every_fold_better=all(item['metrics']['race_outcome_log_loss'] < reference['metrics']['race_outcome_log_loss']
                                          for item, reference in zip(folds_scored, base)))
            summary[key] = row
        best = min(summary, key=lambda key: summary[key]['race_outcome_log_loss_fold_mean'])
        result['groups'][group] = dict(summary=summary, best_config=best, baseline=baseline_key, folds=details[group])
    result['seconds'] = round(time.time() - started, 1)
    (output / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    from .priority import background
    background()
    parser = argparse.ArgumentParser(description='v3実験B: 段ごとの温度、中央・地方の統合モデル、LightGBMの調整を比較します。')
    parser.add_argument('--dataset', required=True, type=Path, help='prepare済みのfeatures.sqlite3')
    parser.add_argument('--output-dir', required=True, type=Path, help='新しい出力先（既存は不可）')
    parser.add_argument('--validation-years', nargs='+', type=int, default=list(DEFAULT_VALIDATION_YEARS))
    parser.add_argument('--train-start', default='2022-04-01')
    parser.add_argument('--test-start', default='2026-01-01')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--tuning-trials', type=int, default=20, help='Optunaの試行数（0で調整なし）')
    parser.add_argument('--tuning-hours', type=float, default=4.0, help='調整に使う最大時間（時間）')
    parser.add_argument('--evaluation-samples', type=int, default=1024)
    args = parser.parse_args(argv)
    from .cli import emit
    try:
        import lightgbm  # noqa: F401
        if args.tuning_trials > 0:
            import optuna  # noqa: F401
    except ImportError:
        emit(event='error', message='lightgbm/optunaがありません。scripts\\setup-ml.ps1 を実行してください。')
        return 2
    try:
        result = run_experiment(args.dataset, args.output_dir, validation_years=args.validation_years,
                                train_start=args.train_start, test_start=args.test_start, threads=args.threads,
                                rounds=args.rounds, patience=args.patience, tuning_trials=args.tuning_trials,
                                tuning_hours=args.tuning_hours, evaluation_samples=args.evaluation_samples,
                                report=emit)
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
