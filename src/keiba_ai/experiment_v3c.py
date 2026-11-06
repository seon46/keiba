"""V3 experiment C: announced current-race bodyweight and going (design 6.2 I).

Base model = best of experiment B: one LightGBM Plackett-Luce model per group
with the tuned parameters, 89 features without odds, four stage temperatures
fitted on the early-stopping period.  Variants add features built from values
announced before the start:

* ``weight`` (I-1): today's bodyweight and its change, compared with the
  horse's own recent weights, change size versus its usual change, carried
  weight per kg of bodyweight, change after a layoff, race-relative weight.
* ``going`` (I-2): today's going and weather, the horse's record on the same
  going class (good/slightly heavy vs heavy/bad), going change from last start.
* ``both``.

The current race's finishing order, times, passages, odds, popularity and
moisture are never read.  To keep predictions usable when an announcement is
missing, whole races in the training window have these columns blanked with
probability ``--announce-dropout``.  Validation keeps the observed values.
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
from .experiment_v3b import _finisher_rows, _structure, fit_strength

GROUPS = ('central_flat', 'local_flat')
STAGE_COUNT = 4
# Best trial of experiment B (structure: separate).
TUNED_PARAMS = dict(learning_rate=0.0252582539546178, num_leaves=28, min_data_in_leaf=182,
                    feature_fraction=0.6269212343357444, bagging_fraction=0.6812699712112874,
                    bagging_freq=1, lambda_l2=1.9674885250802805)
WEIGHT_FEATURES = (
    'current_bodyweight_kg', 'current_bodyweight_change_kg', 'current_weight_vs_recent5',
    'current_weight_change_abs_vs_usual', 'current_carried_per_bodyweight',
    'current_weight_change_after_layoff', 'current_weight_race_z', 'current_weight_change_race_z',
)
GOING_FEATURES = (
    'current_going_code', 'current_weather_code', 'current_wet_flag',
    'horse_going_match_starts', 'horse_going_match_top3_rate', 'current_going_change_from_previous',
)
CATEGORICAL_EXTRA = ('current_going_code', 'current_weather_code')
VARIANTS = {'base': (), 'weight': WEIGHT_FEATURES, 'going': GOING_FEATURES,
            'both': WEIGHT_FEATURES + GOING_FEATURES}


def _params(threads, seed):
    params = dict(max_bin=255, num_threads=threads, verbosity=-1, seed=seed, deterministic=True,
                  force_row_wise=True, metric='None')
    params.update(TUNED_PARAMS)
    return params


def load_announced(database, group, identities):
    """Announced values from the normalized input rows, aligned to ``identities``."""
    import numpy as np
    from .features import _GOING, _WEATHER
    from .model import readonly
    wanted = {(race, horse): index for index, (race, horse, _) in enumerate(identities)}
    count = len(identities)
    weight, change = np.full(count, np.nan), np.full(count, np.nan)
    going, weather = np.full(count, np.nan), np.full(count, np.nan)
    connection = readonly(database)
    try:
        rows = connection.execute(
            "SELECT f.race_id,f.horse_id,json_extract(p.payload_json,'$.weight_kg'),"
            "json_extract(p.payload_json,'$.weight_change_kg'),json_extract(p.payload_json,'$.track_condition'),"
            "json_extract(p.payload_json,'$.weather') FROM features f JOIN input_rows p "
            "ON p.region=f.region AND p.race_id=f.race_id AND p.horse_id=f.horse_id WHERE f.group_name=?", (group,))
        for race, horse, kilograms, delta, condition, sky in rows:
            index = wanted.get((race, horse))
            if index is None:
                continue
            weight[index] = kilograms if kilograms is not None else np.nan
            change[index] = delta if delta is not None else np.nan
            going[index] = _GOING.get(condition, np.nan) if condition is not None else np.nan
            weather[index] = _WEATHER.get(sky, np.nan) if sky is not None else np.nan
    finally:
        connection.close()
    return dict(weight=weight, change=change, going=going, weather=weather)


def _race_z(values, race_ids):
    import numpy as np
    result = np.full(len(values), np.nan)
    boundaries = np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block = values[start:end]
        known = np.isfinite(block)
        if known.sum() >= 2:
            spread = block[known].std()
            result[start:end] = (block - block[known].mean()) / spread if spread > 0 else 0.0
            result[start:end][~known] = np.nan
    return result


def announced_features(data, announced):
    """Columns in WEIGHT_FEATURES + GOING_FEATURES order, from announced values and causal history."""
    import numpy as np
    from .features import FEATURE_NAMES, _GOING
    column = lambda name: data['x'][:, FEATURE_NAMES.index(name)].astype(float)
    weight, change = announced['weight'], announced['change']
    recent_weight = column('horse_group_recent5_bodyweight_kg')
    usual_change = column('horse_group_recent5_absolute_weight_change_kg')
    days = column('horse_days_since_last_start')
    carried = column('carried_weight_kg')
    layoff = np.where(np.isfinite(days), (days >= 60).astype(float), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        weight_columns = [
            weight, change, weight - recent_weight, np.abs(change) - usual_change,
            carried / weight, change * layoff,
            _race_z(weight, data['race_ids']), _race_z(change, data['race_ids']),
        ]
    going = announced['going']
    wet = np.where(np.isfinite(going), (going >= _GOING['重']).astype(float), np.nan)
    wet_starts, wet_rate = column('horse_group_wet_starts'), column('horse_group_wet_top3_rate')
    dry_starts, dry_rate = column('horse_group_dryish_starts'), column('horse_group_dryish_top3_rate')
    match_starts = np.where(wet == 1, wet_starts, np.where(wet == 0, dry_starts, np.nan))
    match_rate = np.where(wet == 1, wet_rate, np.where(wet == 0, dry_rate, np.nan))
    going_columns = [going, announced['weather'], wet, match_starts, match_rate,
                     going - column('horse_group_previous_going_code')]
    return np.column_stack(weight_columns + going_columns).astype(np.float32)


class VariantMatrix:
    def __init__(self, data, base_columns, extra, extra_names, names):
        self.data, self.base_columns, self.extra = data, base_columns, extra
        all_extra = list(WEIGHT_FEATURES + GOING_FEATURES)
        self.extra_columns = [all_extra.index(name) for name in extra_names]
        self.names = names
        self.blank = None  # boolean row mask: blank the extra columns (announcement dropout)

    def __call__(self, rows):
        import numpy as np
        base = self.data['x'][np.ix_(rows, self.base_columns)]
        if not self.extra_columns:
            return base
        extra = self.extra[np.ix_(rows, self.extra_columns)].copy()
        if self.blank is not None:
            extra[self.blank[rows]] = np.nan
        return np.concatenate([base, extra], axis=1)


def _race_dropout(race_ids, probability, seed):
    import numpy as np
    from .rank_model import race_seed
    unique = {race: (race_seed(race, seed) % 10_000) / 10_000 < probability for race in dict.fromkeys(race_ids.tolist())}
    return np.asarray([unique[race] for race in race_ids.tolist()])


def run_experiment(database, output, *, variants=('base', 'weight', 'going', 'both'),
                   validation_years=DEFAULT_VALIDATION_YEARS, train_start='2022-04-01', test_start='2026-01-01',
                   stop_months=2, threads=2, rounds=8000, patience=150, announce_dropout=.05,
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
    if 'base' not in variants or any(name not in VARIANTS for name in variants):
        raise ValueError(f'variantsには base を含め、{sorted(VARIANTS)} から指定してください。')
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
    base_columns = [FEATURE_NAMES.index(name) for name in base_names]
    base_categorical = [index for index, name in enumerate(base_names) if name in CATEGORICAL_FEATURES]
    folds = []
    for year in validation_years:
        start = f'{year:04d}-01-01'
        stop_start = _shift_months(start, -stop_months)
        if stop_start <= train_start:
            raise ValueError(f'{year}年の検証foldの学習期間がありません。')
        folds.append(dict(validation_year=year, train_start=train_start, stop_start=stop_start,
                          validation_start=start, validation_end=f'{year + 1:04d}-01-01'))
    started = time.time()
    result = dict(experiment='v3c_announced_weight_going', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  dataset=str(database.resolve()), feature_revision=feature_revision(),
                  base_feature_count=len(base_names), variant_features={name: list(VARIANTS[name]) for name in variants},
                  folds=folds, test_start=test_start, test_period_used=False,
                  settings=dict(threads=threads, rounds=rounds, patience=patience, params=TUNED_PARAMS,
                                stage_temperatures=STAGE_COUNT, announce_dropout=announce_dropout,
                                evaluation_samples=evaluation_samples, seed=seed,
                                baseline='base/staged_t (best configuration of experiment B)'),
                  groups={})
    output.mkdir(parents=True)
    losses_file = (output / 'race_losses.csv').open('w', encoding='utf-8', newline='')
    losses = csv.writer(losses_file)
    losses.writerow(['group', 'variant', 'validation_year', 'race_id', 'outcome_log_loss', 'top3_log_loss', 'win_log_loss'])
    try:
        for group in GROUPS:
            report(event='loading_group', group=group)
            data = load_rank_group(database, group)
            data['race_ids'] = np.asarray([identity[0] for identity in data['identities']])
            announced = load_announced(database, group, data['identities'])
            extra = announced_features(data, announced)
            coverage = {name: float(np.isfinite(extra[:, index]).mean())
                        for index, name in enumerate(WEIGHT_FEATURES + GOING_FEATURES)}
            dropout = _race_dropout(data['race_ids'], announce_dropout, seed)
            group_result = dict(coverage=coverage, excluded_races=data['excluded_races'], folds=[])
            per_race = {}
            with threadpool_limits(limits=threads):
                for fold in folds:
                    days = data['days']
                    fit = (days >= fold['train_start']) & (days < fold['stop_start'])
                    stop = (days >= fold['stop_start']) & (days < fold['validation_start'])
                    validation = (days >= fold['validation_start']) & (days < fold['validation_end'])
                    year = fold['validation_year']
                    report(event='finisher', group=group, validation_year=year)
                    finisher, constant, finisher_rounds = _m1_finisher(
                        data, fit, stop, base_columns, base_categorical, threads=threads, rounds=rounds,
                        patience=patience, seed=seed)
                    validation_rows = np.flatnonzero(validation)
                    x = data['x'][np.ix_(validation_rows, base_columns)]
                    completion = (np.full(len(validation_rows), constant) if finisher is None
                                  else finisher.predict(x, num_iteration=finisher_rounds))
                    fold_result = dict(validation_year=year, variants={})
                    for name in variants:
                        extra_names = list(VARIANTS[name])
                        names = base_names + extra_names
                        categorical = base_categorical + [len(base_names) + index for index, extra_name in enumerate(extra_names)
                                                          if extra_name in CATEGORICAL_EXTRA]
                        matrix = VariantMatrix(data, base_columns, extra, extra_names, names)
                        # Blank announcements only while fitting; early stopping, temperatures and validation see real values.
                        matrix.blank = dropout & fit
                        report(event='training', group=group, validation_year=year, variant=name)
                        booster, best = fit_strength(data, matrix, categorical, fit, stop, _params(threads, seed),
                                                     rounds=rounds, patience=patience)
                        matrix.blank = None
                        stop_rows = _finisher_rows(data, stop)
                        stop_scores = booster.predict(matrix(stop_rows), num_iteration=best, raw_score=True)
                        temperatures, stop_loss = fit_temperatures(_structure(data, stop_rows), stop_scores, STAGE_COUNT)
                        scores = booster.predict(matrix(validation_rows), num_iteration=best, raw_score=True)
                        scored = score_races(data['race_ids'][validation_rows], data['positions'][validation_rows],
                                             data['finished'][validation_rows], data['race_numbers'][validation_rows],
                                             scores, completion, temperatures, samples=evaluation_samples, seed=seed)
                        importance = booster.feature_importance(importance_type='gain', iteration=best)
                        order = np.argsort(-importance)
                        rank_of = {names[index]: position + 1 for position, index in enumerate(order)}
                        fold_result['variants'][name] = dict(
                            best_iteration=best, temperatures=temperatures, stop_pl_nll=stop_loss,
                            metrics=scored['metrics'],
                            top_gain_features=[dict(feature=names[i], gain=float(importance[i])) for i in order[:20]],
                            added_feature_gain_rank={feature: rank_of[feature] for feature in extra_names})
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
            del data, extra
    finally:
        losses_file.close()
    result['seconds'] = round(time.time() - started, 1)
    (output / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    from .priority import background
    background()
    parser = argparse.ArgumentParser(description='v3実験C: 発走前に公表される今回の馬体重・馬場状態の効果を測ります。')
    parser.add_argument('--dataset', required=True, type=Path, help='prepare済みのfeatures.sqlite3')
    parser.add_argument('--output-dir', required=True, type=Path, help='新しい出力先（既存は不可）')
    parser.add_argument('--variants', nargs='+', default=['base', 'weight', 'going', 'both'], choices=sorted(VARIANTS))
    parser.add_argument('--validation-years', nargs='+', type=int, default=list(DEFAULT_VALIDATION_YEARS))
    parser.add_argument('--train-start', default='2022-04-01')
    parser.add_argument('--test-start', default='2026-01-01')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=8000)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--announce-dropout', type=float, default=.05)
    parser.add_argument('--evaluation-samples', type=int, default=1024)
    args = parser.parse_args(argv)
    from .cli import emit
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        emit(event='error', message='lightgbmがありません。scripts\\setup-ml.ps1 を実行してください。')
        return 2
    try:
        result = run_experiment(args.dataset, args.output_dir, variants=args.variants,
                                validation_years=args.validation_years, train_start=args.train_start,
                                test_start=args.test_start, threads=args.threads, rounds=args.rounds,
                                patience=args.patience, announce_dropout=args.announce_dropout,
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
