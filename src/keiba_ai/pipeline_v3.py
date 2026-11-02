"""V3 production pipeline: fit and predict rank distributions with the best configuration.

Per group (no odds, no track bias):

* central_flat: 89 base features + speed figures + horse ratings.
* local_flat:   89 base features + announced bodyweight (5% race dropout while
                training) + speed figures + jockey theta (from cross-fitted
                no-jockey strengths) + horse ratings.

Strength: LightGBM with the listwise Plackett-Luce objective (tuned parameters),
fitted on [train_start, train_end - stop_months) and early-stopped on the last
``stop_months`` before ``train_end``.  Finisher model: binary LightGBM.
Temperatures: ``stop_hybrid`` (T1 free, T_j = a * j**b) fitted on the stop period.

``fit`` saves one bundle per group; ``predict`` turns an entries CSV into rank
distributions for days on or after ``train_end``, rebuilding every causal block
from the database rows of earlier days plus the entries themselves.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

BUNDLE_VERSION = 'v3-pipeline-1'
GROUPS = ('central_flat', 'local_flat')
CONFIG = {'central_flat': dict(weight=False, theta=False),
          'local_flat': dict(weight=True, theta=True)}
DEFAULT_SETTINGS = dict(train_start='2022-04-01', stop_months=2, crossfit_blocks=3, announce_dropout=.05,
                        half_life=365.0, short_half_life=90.0, prior_sd=.3, rounds=8000, patience=150, seed=42)
TEST_START = '2026-01-01'


# --------------------------------------------------------------------------
# Shared feature assembly
# --------------------------------------------------------------------------

def _names():
    from .experiment_v3 import experiment_feature_names
    from .experiment_v3d import JOCKEY_RATE_FEATURES
    names = experiment_feature_names()
    return names, [name for name in names if name not in JOCKEY_RATE_FEATURES]


def _columns(selected):
    from .features import FEATURE_NAMES, CATEGORICAL_FEATURES
    return ([FEATURE_NAMES.index(name) for name in selected],
            [index for index, name in enumerate(selected) if name in CATEGORICAL_FEATURES])


def feature_layout(group):
    """Names of the strength-model columns, in matrix order."""
    from .experiment_v3c import WEIGHT_FEATURES
    from .experiment_v3d import THETA_FEATURES
    from .ratings import RATING_FEATURES
    from .speed import SPEED_FEATURES
    names, _ = _names()
    config = CONFIG[group]
    layout = list(names)
    if config['weight']:
        layout += list(WEIGHT_FEATURES)
    layout += list(SPEED_FEATURES)
    if config['theta']:
        layout += ['theta_' + name if name in names else name for name in THETA_FEATURES]
    layout += list(RATING_FEATURES)
    return layout


def _f0_blocks(group, weight, speed):
    blocks = []
    if CONFIG[group]['weight']:
        blocks.append(weight)
    blocks.append(speed)
    return blocks


def _main_blocks(group, weight, speed, theta, rating):
    blocks = _f0_blocks(group, weight, speed)
    if CONFIG[group]['theta']:
        blocks.append(theta)
    blocks.append(rating)
    return blocks


def _theta(data, horses, jockeys, strength, settings):
    import numpy as np
    from .experiment_v3d import jockey_timeline, theta_feature_block
    timeline = jockey_timeline(data['days'], data['race_ids'], jockeys, horses, data['finished'], data['positions'],
                               strength, half_life=settings['half_life'],
                               short_half_life=settings['short_half_life'], prior_sd=settings['prior_sd'])
    return theta_feature_block(timeline, data['race_ids']).astype(np.float32)


def _check_dataset(database):
    from .model import feature_revision, readonly
    connection = readonly(database)
    try:
        metadata = dict(connection.execute('SELECT key,value_json FROM ai_metadata'))
        last_day = connection.execute('SELECT MAX(day) FROM features').fetchone()[0]
    except sqlite3.Error as exc:
        raise ValueError(f'prepare済みのfeatures.sqlite3を指定してください: {exc}') from exc
    finally:
        connection.close()
    if json.loads(metadata['feature_revision']) != feature_revision():
        raise ValueError('特徴量コードの版がこのfeatures.sqlite3と一致しません。prepareから作成し直してください。')
    return last_day


def _iso(value, label):
    try:
        if date.fromisoformat(value).isoformat() == value:
            return value
    except (TypeError, ValueError):
        pass
    raise ValueError(f'{label}はYYYY-MM-DDで指定してください。')


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

def fit(database, output, *, train_end=None, evaluate_until=None, allow_test_period=False, groups=GROUPS,
        threads=2, rounds=None, patience=None, evaluation_samples=1024, report=None, **overrides):
    import numpy as np
    import joblib
    from threadpoolctl import threadpool_limits
    from .cli import fresh_directory
    from .experiment_v3 import _m1_finisher, _shift_months
    from .experiment_v3b import _finisher_rows, _structure
    from .experiment_v3c import WEIGHT_FEATURES, _race_dropout, announced_features, load_announced
    from .experiment_v3d import Blocks, _params, _quarter_block, fit_pl, load_jockeys
    from .experiment_v3i import fit_hybrid_temperatures
    from .model import feature_revision
    from .rank_model import load_rank_group
    from .ranking_v3 import score_races
    from .ratings import rating_block
    from .speed import load_speed_block

    report = report or (lambda **message: None)
    settings = dict(DEFAULT_SETTINGS, **overrides)
    if rounds is not None:
        settings['rounds'] = rounds
    if patience is not None:
        settings['patience'] = patience
    unknown = [group for group in groups if group not in CONFIG]
    if unknown:
        raise ValueError(f'v3パイプラインの対象外の区分です: {unknown}（対応: {list(GROUPS)}）')
    if evaluate_until is not None and _iso(evaluate_until, 'evaluate-until') > TEST_START and not allow_test_period:
        raise ValueError(f'{TEST_START}以降の最終評価期間を評価するには--allow-test-periodが必要です。')
    database = Path(database)
    last_day = _check_dataset(database)
    if train_end is None:
        train_end = (date.fromisoformat(last_day) + timedelta(days=1)).isoformat()
    train_end = _iso(train_end, 'train-end')
    train_start = _iso(settings['train_start'], 'train-start')
    stop_start = _shift_months(train_end, -settings['stop_months'])
    if stop_start <= train_start:
        raise ValueError('学習期間が短すぎます。train-endを後ろにしてください。')
    if evaluate_until is not None:
        evaluate_until = _iso(evaluate_until, 'evaluate-until')
        if evaluate_until <= train_end:
            raise ValueError('evaluate-untilはtrain-endより後にしてください。')
        if evaluate_until > TEST_START and not allow_test_period:
            raise ValueError(f'{TEST_START}以降の最終評価期間を評価するには--allow-test-periodが必要です。')
    names, no_jockey = _names()
    columns, categorical = _columns(names)
    f0_columns, f0_categorical = _columns(no_jockey)
    params = _params(threads, settings['seed'])
    started = time.time()
    summary = dict(pipeline=BUNDLE_VERSION, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   dataset=str(database.resolve()), feature_revision=feature_revision(), train_start=train_start,
                   stop_start=stop_start, train_end=train_end, evaluate_until=evaluate_until,
                   evaluated_test_period=bool(evaluate_until and evaluate_until > TEST_START),
                   trained_on_test_period=train_end > TEST_START,
                   settings=settings, groups={})
    with fresh_directory(output) as temporary:
        for group in groups:
            report(event='loading_group', group=group)
            data = load_rank_group(database, group)
            race_ids = data['race_ids'] = np.asarray([identity[0] for identity in data['identities']])
            horses = np.asarray([identity[1] for identity in data['identities']], dtype=object)
            days = data['days']
            fit_mask = (days >= train_start) & (days < stop_start)
            stop_mask = (days >= stop_start) & (days < train_end)
            training = (days >= train_start) & (days < train_end)
            before = days < train_end
            fit_rows, stop_rows = _finisher_rows(data, fit_mask), _finisher_rows(data, stop_mask)
            if not len(fit_rows) or not len(stop_rows):
                raise ValueError(f'{group}: 学習または早期終了用のレースが不足しています。')
            config = CONFIG[group]
            speed, _ = load_speed_block(database, group, data['identities'])
            rating = rating_block(days, race_ids, horses, data['finished'], data['positions'])
            weight = None
            dropout = None
            if config['weight']:
                weight = announced_features(data, load_announced(database, group, data['identities']))[:, :len(WEIGHT_FEATURES)]
                dropout = _race_dropout(race_ids, settings['announce_dropout'], settings['seed'])
            group_result = dict(rows=int(len(days)), training_races=int(len(set(race_ids[training].tolist()))),
                                excluded_races=data['excluded_races'])
            bundle = dict(version=BUNDLE_VERSION, group=group, config=config, feature_revision=feature_revision(),
                          feature_names=feature_layout(group), base_feature_names=names, settings=settings,
                          train_start=train_start, stop_start=stop_start, train_end=train_end,
                          trained_through=max(days[before].tolist()))
            with threadpool_limits(limits=threads):
                report(event='finisher', group=group)
                finisher, constant, finisher_best = _m1_finisher(
                    data, fit_mask, stop_mask, columns, categorical, threads=threads, rounds=settings['rounds'],
                    patience=settings['patience'], seed=settings['seed'])
                bundle.update(finisher=None if finisher is None else finisher.model_to_string(num_iteration=finisher_best),
                              finisher_constant=constant)
                theta = None
                if config['theta']:
                    # No-jockey strength: cross-fitted before train_end, f0 afterwards.
                    f0_plain = [(block, None) for block in _f0_blocks(group, weight, speed)]
                    f0_train = [(block, (dropout & fit_mask) if block is weight else None)
                                for block in _f0_blocks(group, weight, speed)]
                    report(event='f0', group=group)
                    f0, f0_best = fit_pl(data, Blocks(data, f0_columns, f0_train), f0_categorical, fit_rows, stop_rows,
                                         params, rounds=settings['rounds'], patience=settings['patience'])
                    plain = Blocks(data, f0_columns, f0_plain)
                    strength0 = np.full(len(days), np.nan)
                    quarter = _quarter_block(days, settings['crossfit_blocks'])
                    for block in range(settings['crossfit_blocks']):
                        report(event='crossfit', group=group, block=block)
                        rows = _finisher_rows(data, training & (quarter != block))
                        model, _ = fit_pl(data, Blocks(data, f0_columns, f0_train), f0_categorical, rows, None, params,
                                          rounds=settings['rounds'], patience=settings['patience'],
                                          fixed_rounds=f0_best)
                        target = np.flatnonzero(before & (quarter == block))
                        if len(target):
                            strength0[target] = model.predict(plain(target), raw_score=True)
                    after = np.flatnonzero(~before)
                    if len(after):
                        strength0[after] = f0.predict(plain(after), num_iteration=f0_best, raw_score=True)
                    theta = _theta(data, horses, load_jockeys(database, group, data['identities']), strength0, settings)
                    keep = np.flatnonzero(before)
                    bundle.update(f0=f0.model_to_string(num_iteration=f0_best), f0_best_iteration=f0_best,
                                  oof_race_ids=race_ids[keep], oof_horse_ids=horses[keep].astype(str),
                                  oof_strength=strength0[keep].astype(np.float64))
                    group_result['f0_best_iteration'] = f0_best
                blocks = _main_blocks(group, weight, speed, theta, rating)
                train_blocks = [(block, (dropout & fit_mask) if block is weight else None) for block in blocks]
                matrix = Blocks(data, columns, [(block, None) for block in blocks])
                report(event='strength', group=group)
                booster, best = fit_pl(data, Blocks(data, columns, train_blocks), categorical, fit_rows, stop_rows,
                                       params, rounds=settings['rounds'], patience=settings['patience'])
                stop_scores = booster.predict(matrix(stop_rows), num_iteration=best, raw_score=True)
                temperatures, stop_loss, shape = fit_hybrid_temperatures(_structure(data, stop_rows), stop_scores)
                bundle.update(strength=booster.model_to_string(num_iteration=best), best_iteration=best,
                              temperatures=temperatures, temperature_shape=shape)
                group_result.update(best_iteration=best, finisher_best_iteration=finisher_best,
                                    temperatures=temperatures[:6], temperature_shape=shape,
                                    stop_pl_loss=stop_loss)
                if evaluate_until is not None:
                    rows = np.flatnonzero((days >= train_end) & (days < evaluate_until))
                    if len(rows):
                        report(event='evaluate', group=group, rows=int(len(rows)))
                        scores = booster.predict(matrix(rows), num_iteration=best, raw_score=True)
                        completion = (np.full(len(rows), constant) if finisher is None else
                                      finisher.predict(data['x'][np.ix_(rows, columns)], num_iteration=finisher_best))
                        args = (race_ids[rows], data['positions'][rows], data['finished'][rows],
                                data['race_numbers'][rows])
                        model_scored = score_races(*args, scores, completion, temperatures,
                                                   samples=evaluation_samples, seed=settings['seed'])
                        uniform = score_races(*args, np.zeros(len(rows)), completion, [1.0],
                                              samples=evaluation_samples, seed=settings['seed'])
                        group_result['evaluation'] = dict(model=model_scored['metrics'], uniform=uniform['metrics'])
                        with (temporary / f'evaluation_{group}.csv').open('w', encoding='utf-8', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(['race_date', 'race_id', 'horse_id', 'finish_position', 'strength_score',
                                             'finish_probability'])
                            for row, score, probability in zip(rows, scores, completion):
                                writer.writerow([days[row], race_ids[row], horses[row],
                                                 int(data['positions'][row]) if data['finished'][row] else '',
                                                 f'{score:.6f}', f'{probability:.6f}'])
            joblib.dump(bundle, temporary / f'{group}.joblib', compress=3)
            summary['groups'][group] = group_result
            report(event='group_done', group=group, best_iteration=group_result['best_iteration'],
                   evaluation=(group_result.get('evaluation') or {}).get('model', {}).get('race_outcome_log_loss'))
            del data, speed, rating, weight, theta
        summary['seconds'] = round(time.time() - started, 1)
        (temporary / 'model.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + '\n',
                                              encoding='utf-8')
    return summary


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------

def _load_bundle(model_dir, group):
    import joblib
    import lightgbm as lgb
    from .model import feature_revision
    path = Path(model_dir) / f'{group}.joblib'
    if not path.is_file():
        raise ValueError(f'この区分の学習済みモデルがありません: {group}')
    bundle = joblib.load(path)
    if bundle.get('version') != BUNDLE_VERSION or bundle.get('group') != group:
        raise ValueError(f'v3パイプラインのモデルではありません: {path}')
    if bundle['feature_revision'] != feature_revision() or bundle['feature_names'] != feature_layout(group):
        raise ValueError('モデルと特徴量の版が一致しません。fitから作成し直してください。')
    bundle['strength_model'] = lgb.Booster(model_str=bundle['strength'])
    bundle['finisher_model'] = None if bundle['finisher'] is None else lgb.Booster(model_str=bundle['finisher'])
    if bundle['config']['theta']:
        bundle['f0_model'] = lgb.Booster(model_str=bundle['f0'])
        bundle['oof'] = {(race, horse): value for race, horse, value in
                         zip(bundle['oof_race_ids'].tolist(), bundle['oof_horse_ids'].tolist(),
                             bundle['oof_strength'].tolist())}
    return bundle


def entry_races(raw_rows):
    """Validate raw entries rows (CSV/export dicts) into [(day, region, race_id, rows)] sorted by race."""
    from .dataset import normalize_row
    grouped, seen = {}, set()
    for raw in raw_rows:
        row = normalize_row(raw, page='entries')
        key = (row['region'], row['race_id'], row['horse_id'])
        if key in seen:
            raise ValueError('出馬表に同じ馬・レースの重複があります。')
        seen.add(key)
        grouped.setdefault((row['day'], row['region'], row['race_id']), []).append(row)
    races = []
    for (day, region, race_id), race in sorted(grouped.items()):
        fields = ('day', 'group_name', 'course', 'surface', 'distance_m', 'declared_field_size', 'snapshot_id')
        if any(len({row.get(field) for row in race}) > 1 for field in fields):
            raise ValueError(f'出馬表の条件が一致しません: {race_id}')
        if len({row['horse_number'] for row in race}) != len(race):
            raise ValueError(f'出馬表の馬番が重複しています: {race_id}')
        declared = race[0].get('declared_field_size')
        if declared is not None and declared != len(race):
            raise ValueError(f'出馬表の頭数が不足しています: {race_id}')
        race.sort(key=lambda row: row['horse_number'])
        races.append((day, region, race_id, race))
    return races


def _read_entries(entries):
    with Path(entries).open(encoding='utf-8-sig', newline='') as file:
        races = entry_races(csv.DictReader(file))
    if not races:
        raise ValueError('出馬表がありません。')
    return races


def _group_day(database, group, data, horses, stored, bundle, day, races, history, priors, *, samples):
    """Rank distributions for one group's races of one day."""
    import numpy as np
    from .bias import DEFAULT_BIAS_CONFIG, IntradayBiasState
    from .experiment_v3c import WEIGHT_FEATURES, announced_features
    from .experiment_v3d import Blocks
    from .features import _GOING, _WEATHER, race_features
    from .rank_model import race_seed
    from .ranking_v3 import staged_rank_distribution
    from .ratings import rating_block
    from .speed import load_speed_block
    settings = bundle['settings']
    config = bundle['config']
    names, no_jockey = _names()
    columns, _ = _columns(names)
    f0_columns, _ = _columns(no_jockey)
    history_rows = np.flatnonzero(data['days'] < day)
    entry_rows, entry_x = [], []
    for race, active in races:
        vectors = race_features(active, history, intraday=IntradayBiasState(day), priors=priors, full_race=race,
                                bias_config=DEFAULT_BIAS_CONFIG)
        entry_x.extend([[np.nan if value is None else value for value in vector] for vector in vectors])
        entry_rows.extend(active)
    count_history, count_entry = len(history_rows), len(entry_rows)
    combined = dict(
        x=np.concatenate([data['x'][history_rows], np.asarray(entry_x, dtype=np.float32)]),
        days=np.concatenate([data['days'][history_rows], np.asarray([day] * count_entry, dtype=data['days'].dtype)]),
        race_ids=np.concatenate([data['race_ids'][history_rows], np.asarray([row['race_id'] for row in entry_rows])]),
        finished=np.concatenate([data['finished'][history_rows], np.zeros(count_entry, dtype=bool)]),
        positions=np.concatenate([data['positions'][history_rows], np.zeros(count_entry, dtype=data['positions'].dtype)]),
    )
    combined_horses = np.concatenate([horses[history_rows], np.asarray([row['horse_id'] for row in entry_rows], dtype=object)])
    identities = [data['identities'][index] for index in history_rows] + \
        [(row['race_id'], row['horse_id'], row['horse_number']) for row in entry_rows]
    entry = np.arange(count_history, count_history + count_entry)
    speed, _ = load_speed_block(database, group, identities, before_day=day, extra_races=[active for _, active in races])
    rating = rating_block(combined['days'], combined['race_ids'], combined_horses, combined['finished'],
                          combined['positions'])
    weight = theta = None
    if config['weight']:
        announced = {key: np.concatenate([values[history_rows], np.full(count_entry, np.nan)])
                     for key, values in stored['announced'].items()}
        for offset, row in enumerate(entry_rows):
            index = count_history + offset
            announced['weight'][index] = row['weight_kg'] if row.get('weight_kg') is not None else np.nan
            announced['change'][index] = row['weight_change_kg'] if row.get('weight_change_kg') is not None else np.nan
            announced['going'][index] = _GOING.get(row.get('track_condition'), np.nan)
            announced['weather'][index] = _WEATHER.get(row.get('weather'), np.nan)
        weight = announced_features(combined, announced)[:, :len(WEIGHT_FEATURES)]
    if config['theta']:
        jockeys = np.concatenate([stored['jockeys'][history_rows],
                                  np.asarray([row.get('jockey_id') or '' for row in entry_rows], dtype=object)])
        strength0 = np.full(len(combined_horses), np.nan)
        missing = []
        for index in range(count_history):
            value = bundle['oof'].get((combined['race_ids'][index], combined_horses[index]))
            if value is None:
                missing.append(index)
            else:
                strength0[index] = value
        missing = np.asarray(missing, dtype=int)
        missing = missing[combined['finished'][missing]] if len(missing) else missing
        if len(missing):
            plain = Blocks(combined, f0_columns, [(block, None) for block in _f0_blocks(group, weight, speed)])
            strength0[missing] = bundle['f0_model'].predict(plain(missing), raw_score=True)
        theta = _theta(combined, combined_horses, jockeys, strength0, settings)
    matrix = Blocks(combined, columns, [(block, None) for block in _main_blocks(group, weight, speed, theta, rating)])
    x_entry = matrix(entry)
    scores = bundle['strength_model'].predict(x_entry, raw_score=True)
    completion = (np.full(count_entry, bundle['finisher_constant']) if bundle['finisher_model'] is None else
                  bundle['finisher_model'].predict(combined['x'][np.ix_(entry, columns)]))
    output, start = [], 0
    for race, active in races:
        size = len(active)
        part = slice(start, start + size)
        distribution = staged_rank_distribution(scores[part], completion[part], bundle['temperatures'],
                                                samples=samples, seed=race_seed(active[0]['race_id'], settings['seed']))
        for local, row in enumerate(active):
            ranks = distribution[local, :size]
            finish = float(completion[part][local])
            result = dict(race_date=day, region=row['region'], race_id=row['race_id'], horse_id=row['horse_id'],
                          horse_number=row['horse_number'], group=group, starters=size,
                          strength_score=float(scores[part][local]), finish_probability=finish,
                          nonfinish_probability=float(distribution[local, -1]),
                          win_probability=float(ranks[0]),
                          top3_probability=float(np.clip(ranks[:3].sum(), 0, 1)),
                          expected_rank_given_finish=float((ranks * np.arange(1, size + 1)).sum() / finish)
                          if finish > 0 else None,
                          bodyweight_used=bool(config['weight'] and row.get('weight_kg') is not None))
            result.update({f'rank_{rank}_probability': float(value) for rank, value in enumerate(ranks, 1)})
            output.append(result)
        start += size
    return output


def evaluate(database, model_dir, output, *, start=None, end=None, groups=GROUPS, samples=1024, threads=2,
             report=None):
    """Score a saved model on results after its train_end (a newer prepared database).

    Every block is rebuilt causally over the whole database, exactly as predict
    does with an up-to-date history: a race only sees results of earlier days.
    """
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .cli import fresh_directory
    from .experiment_v3c import WEIGHT_FEATURES, announced_features, load_announced
    from .experiment_v3d import Blocks, load_jockeys
    from .rank_model import load_rank_group
    from .ranking_v3 import paired_bootstrap, score_races
    from .ratings import rating_block
    from .speed import load_speed_block
    report = report or (lambda **message: None)
    database = Path(database)
    last_day = _check_dataset(database)
    if end is None:
        end = (date.fromisoformat(last_day) + timedelta(days=1)).isoformat()
    end = _iso(end, 'end')
    names, no_jockey = _names()
    columns, _ = _columns(names)
    f0_columns, _ = _columns(no_jockey)
    started = time.time()
    summary = dict(pipeline=BUNDLE_VERSION, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   dataset=str(database.resolve()), model_dir=str(Path(model_dir).resolve()), end=end, groups={})
    with fresh_directory(output) as temporary, threadpool_limits(limits=threads):
        races_file = (temporary / 'race_losses.csv').open('w', encoding='utf-8', newline='')
        races_writer = csv.writer(races_file)
        races_writer.writerow(['group', 'race_date', 'race_id', 'outcome_log_loss', 'uniform_outcome_log_loss',
                               'top3_log_loss', 'win_log_loss'])
        try:
            for group in groups:
                bundle = _load_bundle(model_dir, group)
                first = _iso(start, 'start') if start is not None else bundle['train_end']
                if first < bundle['train_end']:
                    raise ValueError(f'{group}: 評価の開始日がモデルの学習期間（{bundle["train_end"]}より前）に入っています。')
                report(event='loading_group', group=group)
                data = load_rank_group(database, group)
                race_ids = data['race_ids'] = np.asarray([identity[0] for identity in data['identities']])
                horses = np.asarray([identity[1] for identity in data['identities']], dtype=object)
                days = data['days']
                rows = np.flatnonzero((days >= first) & (days < end))
                if not len(rows):
                    summary['groups'][group] = dict(start=first, races=0)
                    continue
                config = bundle['config']
                speed, _ = load_speed_block(database, group, data['identities'])
                rating = rating_block(days, race_ids, horses, data['finished'], data['positions'])
                weight = theta = None
                if config['weight']:
                    weight = announced_features(data, load_announced(database, group, data['identities']))[:, :len(WEIGHT_FEATURES)]
                if config['theta']:
                    strength0 = np.full(len(days), np.nan)
                    missing = []
                    for index, key in enumerate(zip(race_ids.tolist(), horses.tolist())):
                        value = bundle['oof'].get(key)
                        if value is None:
                            missing.append(index)
                        else:
                            strength0[index] = value
                    missing = np.asarray(missing, dtype=int)
                    missing = missing[data['finished'][missing]] if len(missing) else missing
                    if len(missing):
                        plain = Blocks(data, f0_columns, [(block, None) for block in _f0_blocks(group, weight, speed)])
                        strength0[missing] = bundle['f0_model'].predict(plain(missing), raw_score=True)
                    theta = _theta(data, horses, load_jockeys(database, group, data['identities']), strength0,
                                   bundle['settings'])
                matrix = Blocks(data, columns, [(block, None) for block in _main_blocks(group, weight, speed, theta, rating)])
                report(event='evaluate', group=group, rows=int(len(rows)))
                scores = bundle['strength_model'].predict(matrix(rows), raw_score=True)
                completion = (np.full(len(rows), bundle['finisher_constant']) if bundle['finisher_model'] is None else
                              bundle['finisher_model'].predict(data['x'][np.ix_(rows, columns)]))
                args = (race_ids[rows], data['positions'][rows], data['finished'][rows], data['race_numbers'][rows])
                seed = bundle['settings']['seed']
                model_scored = score_races(*args, scores, completion, bundle['temperatures'], samples=samples, seed=seed)
                uniform = score_races(*args, np.zeros(len(rows)), completion, [1.0], samples=samples, seed=seed)
                first_rows = {}
                for row in rows.tolist():
                    first_rows.setdefault(race_ids[row], row)
                for race, a, u, b, c in zip(model_scored['race_ids'], model_scored['outcome'], uniform['outcome'],
                                            model_scored['top3'], model_scored['win']):
                    races_writer.writerow([group, days[first_rows[race]], race, f'{a:.6f}', f'{u:.6f}', f'{b:.6f}',
                                           '' if np.isnan(c) else f'{c:.6f}'])
                by_week = {}
                race_days = np.asarray([days[first_rows[race]] for race in model_scored['race_ids']])
                for week_start in sorted({(date.fromisoformat(day) - timedelta(days=date.fromisoformat(day).weekday())).isoformat()
                                          for day in race_days.tolist()}):
                    week_end = (date.fromisoformat(week_start) + timedelta(days=7)).isoformat()
                    mask = (race_days >= week_start) & (race_days < week_end)
                    by_week[week_start] = dict(races=int(mask.sum()),
                                               model=float(np.mean(model_scored['outcome'][mask])),
                                               uniform=float(np.mean(uniform['outcome'][mask])))
                summary['groups'][group] = dict(
                    start=first, train_end=bundle['train_end'], last_day=max(days[rows].tolist()),
                    model=model_scored['metrics'], uniform=uniform['metrics'],
                    vs_uniform=paired_bootstrap(model_scored['outcome'] - uniform['outcome'], seed=seed),
                    by_week=by_week)
                report(event='group_done', group=group, races=model_scored['metrics']['races'],
                       race_outcome_log_loss=model_scored['metrics']['race_outcome_log_loss'],
                       uniform=uniform['metrics']['race_outcome_log_loss'])
                del data, speed, rating, weight, theta
        finally:
            races_file.close()
        summary['seconds'] = round(time.time() - started, 1)
        (temporary / 'evaluation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + '\n',
                                                   encoding='utf-8')
    return summary


class Predictor:
    """Cached v3 predictor: bundles, group data and per-day histories stay in memory.

    ``predict_races`` takes validated races (``entry_races`` output) and returns
    one result dict per active runner.  Thread-safe (one prediction at a time).
    """

    def __init__(self, database, model_dir, *, threads=2, report=None, cached_days=2):
        import threading
        self.database, self.model_dir = Path(database), Path(model_dir)
        self.threads, self.cached_days = threads, cached_days
        self.report = report or (lambda **message: None)
        self.last_day = _check_dataset(self.database)
        self.bundles, self.loaded, self.histories = {}, {}, {}
        self.lock = threading.Lock()

    def bundle(self, group):
        if group not in self.bundles:
            self.bundles[group] = _load_bundle(self.model_dir, group)
        return self.bundles[group]

    def train_end(self, group):
        return self.bundle(group)['train_end']

    def _group(self, group):
        import numpy as np
        from .experiment_v3c import load_announced
        from .experiment_v3d import load_jockeys
        from .rank_model import load_rank_group
        if group not in self.loaded:
            self.report(event='loading_group', group=group)
            data = load_rank_group(self.database, group)
            data['race_ids'] = np.asarray([identity[0] for identity in data['identities']])
            horses = np.asarray([identity[1] for identity in data['identities']], dtype=object)
            stored = {}
            if self.bundle(group)['config']['weight']:
                stored['announced'] = load_announced(self.database, group, data['identities'])
            if self.bundle(group)['config']['theta']:
                stored['jockeys'] = load_jockeys(self.database, group, data['identities'])
            self.loaded[group] = (data, horses, stored)
        return self.loaded[group]

    def _history(self, day):
        from .same_day import rebuild_prediction_states
        if day not in self.histories:
            self.report(event='history', day=day)
            while len(self.histories) >= self.cached_days:
                self.histories.pop(next(iter(self.histories)))
            self.histories[day] = rebuild_prediction_states(self.database, day)[:2]
        return self.histories[day]

    @staticmethod
    def supported(race):
        active = [row for row in race if row.get('finish_status') not in ('scratched', 'excluded')]
        return bool(active) and active[0]['group_name'] in CONFIG

    def predict_races(self, races, *, samples=4096):
        from threadpoolctl import threadpool_limits
        by_day = {}
        for day, region, race_id, race in races:
            active = [row for row in race if row.get('finish_status') not in ('scratched', 'excluded')]
            if not active:
                continue
            group = active[0]['group_name']
            if group not in CONFIG:
                self.report(event='skipped_race', race_id=race_id, group=group, reason='v3パイプラインの対象外の区分')
                continue
            by_day.setdefault(day, {}).setdefault(group, []).append((race, active))
        results = []
        with self.lock, threadpool_limits(limits=self.threads):
            for day in sorted(by_day):
                for group in by_day[day]:
                    if day < self.train_end(group):
                        raise ValueError(f'学習期間内の日付には予測できません: {day}（モデルは{self.train_end(group)}より前で学習）')
                history, priors = self._history(day)
                for group, group_races in sorted(by_day[day].items()):
                    data, horses, stored = self._group(group)
                    self.report(event='predicting', day=day, group=group, races=len(group_races))
                    results.extend(_group_day(self.database, group, data, horses, stored, self.bundle(group), day,
                                              group_races, history, priors, samples=samples))
        return results


def predict(database, model_dir, entries, output, *, samples=4096, threads=2, report=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'出力先は既にあります: {output}')
    predictor = Predictor(database, model_dir, threads=threads, report=report)
    results = predictor.predict_races(_read_entries(entries), samples=samples)
    if not results:
        raise ValueError('予測対象の馬がありません。')
    maximum = max(row['starters'] for row in results)
    for row in results:
        for rank in range(1, maximum + 1):
            row.setdefault(f'rank_{rank}_probability', 0.0)
    common = [key for key in results[0] if not key.startswith('rank_')]
    fields = common + [f'rank_{rank}_probability' for rank in range(1, maximum + 1)]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=output.parent,
                                         prefix='.keiba-prediction-', delete=False) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(results)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description='v3順位分布モデル（オッズ不使用）の学習と予測')
    commands = parser.add_subparsers(dest='command', required=True)
    fitting = commands.add_parser('fit', help='最良設定で区分ごとのモデルを学習・保存します')
    fitting.add_argument('--dataset', required=True, type=Path, help='prepare済みのfeatures.sqlite3')
    fitting.add_argument('--output-dir', required=True, type=Path, help='新しいモデル保存先（既存は不可）')
    fitting.add_argument('--train-end', help='この日より前で学習（既定: データ最終日の翌日）')
    fitting.add_argument('--evaluate-until', help='train-end以降、この日より前を評価')
    fitting.add_argument('--allow-test-period', action='store_true', help=f'{TEST_START}以降の評価を許可（最終評価用）')
    fitting.add_argument('--groups', nargs='+', default=list(GROUPS), choices=list(GROUPS))
    fitting.add_argument('--threads', type=int, default=2)
    fitting.add_argument('--rounds', type=int)
    fitting.add_argument('--patience', type=int)
    fitting.add_argument('--evaluation-samples', type=int, default=1024)
    predicting = commands.add_parser('predict', help='出馬表CSVから各馬の順位分布を出力します')
    predicting.add_argument('--dataset', required=True, type=Path)
    predicting.add_argument('--model-dir', required=True, type=Path)
    predicting.add_argument('--entries', required=True, type=Path)
    predicting.add_argument('--output', required=True, type=Path)
    predicting.add_argument('--samples', type=int, default=4096)
    predicting.add_argument('--threads', type=int, default=2)
    evaluating = commands.add_parser('evaluate', help='保存済みモデルを、学習期間より後の確定結果で採点します')
    evaluating.add_argument('--dataset', required=True, type=Path, help='新しい成績を含めてprepareしたfeatures.sqlite3')
    evaluating.add_argument('--model-dir', required=True, type=Path)
    evaluating.add_argument('--output-dir', required=True, type=Path, help='新しい出力先（既存は不可）')
    evaluating.add_argument('--start', help='評価の開始日（既定: モデルのtrain-end）')
    evaluating.add_argument('--end', help='この日より前まで評価（既定: データ最終日の翌日）')
    evaluating.add_argument('--groups', nargs='+', default=list(GROUPS), choices=list(GROUPS))
    evaluating.add_argument('--samples', type=int, default=1024)
    evaluating.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    from .cli import emit
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        emit(event='error', message='lightgbmがありません。scripts\\setup-ml.ps1 を実行してください。')
        return 2
    try:
        if args.command == 'fit':
            summary = fit(args.dataset, args.output_dir, train_end=args.train_end, evaluate_until=args.evaluate_until,
                          allow_test_period=args.allow_test_period, groups=args.groups, threads=args.threads,
                          rounds=args.rounds, patience=args.patience, evaluation_samples=args.evaluation_samples,
                          report=emit)
            emit(event='fit_complete', model=str(args.output_dir / 'model.json'), train_end=summary['train_end'],
                 seconds=summary['seconds'])
        elif args.command == 'evaluate':
            summary = evaluate(args.dataset, args.model_dir, args.output_dir, start=args.start, end=args.end,
                               groups=args.groups, samples=args.samples, threads=args.threads, report=emit)
            emit(event='evaluation_complete', report=str(args.output_dir / 'evaluation.json'),
                 result={group: (value.get('model') or {}).get('race_outcome_log_loss')
                         for group, value in summary['groups'].items()})
        else:
            rows = predict(args.dataset, args.model_dir, args.entries, args.output, samples=args.samples,
                           threads=args.threads, report=emit)
            emit(event='predicted', output=str(args.output), rows=rows)
        return 0
    except (ValueError, OSError, sqlite3.Error, KeyError, csv.Error) as exc:
        emit(event='error', message=str(exc))
        return 2
    except KeyboardInterrupt:
        emit(event='interrupted', message='停止しました。別の出力先で再実行してください。')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
