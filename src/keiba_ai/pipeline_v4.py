"""V4 production pipeline: LightGBM-PL with pedigree features blended with a TabM-style network.

Per group (central_flat / local_flat), validated in experiments J-M (``D_tabm_search_x3``):

* A   LightGBM-PL on the v3 best features (local adds bodyweight and jockey theta),
* B   A's features + causal pedigree effects (sire / damsire / dam thetas; strength offset = out-of-fold A),
* NN  TabM-style network (16 members, periodic numerical embeddings, no race attention) with learned
      pedigree / jockey / trainer embeddings, averaged over 3 seeds,
* D   = w_B * B + w_NN * NN (weights fitted on the stop period), then ``stop_hybrid`` temperatures.

Prediction rebuilds every causal block from the database rows of earlier days.  The expensive history
pass (speed figures, ratings, jockey theta, pedigree effects) runs once per (group, day) and is kept as
a state; each race card is then answered from that state in seconds.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from dataclasses import asdict
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from .neural import NeuralConfig
from .pipeline_v3 import (CONFIG, DEFAULT_SETTINGS, GROUPS, TEST_START, _check_dataset, _columns, _f0_blocks, _iso,
                          _main_blocks, _names, _read_entries, entry_races, feature_layout)

BUNDLE_VERSION = 'v4-pipeline-1'
# experiment M: the setting chosen for 2025 in both groups
NN_CONFIG = NeuralConfig(width=256, embedding=16, pedigree_dim=16, layers=1, heads=4, dropout=.1339214045043637,
                         ancestor_dropout=.2846840960616961, learning_rate=.002171388466007867,
                         weight_decay=.017197099468577025, batch_races=128, max_epochs=12, patience=12,
                         ancestor_min_count=5, entity_min_count=50, seed=42, ensemble=16, numeric_embedding=True,
                         n_frequencies=32, frequency_scale=.005364088271314505, numeric_dim=16, attention=False,
                         beta2=.95)
NN_SEEDS = (42, 7, 1234)


def _key_map(race_ids, horses, values):
    return {key: value for key, value in zip(zip(race_ids.tolist(), horses.tolist()), values.tolist())}


def _fill_strength(keys, stored, predict_missing):
    """Stored out-of-fold strengths where known; the model's prediction for the other rows (as in fit)."""
    import numpy as np
    strength = np.full(len(keys), np.nan)
    missing = []
    for index, key in enumerate(keys):
        value = stored.get(key)
        if value is None:
            missing.append(index)
        else:
            strength[index] = value
    missing = np.asarray(missing, dtype=int)
    if len(missing):
        strength[missing] = predict_missing(missing)
    return strength


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

def fit(database, race_db, output, *, train_end=None, evaluate_until=None, allow_test_period=False, groups=GROUPS,
        threads=2, device='auto', seeds=NN_SEEDS, nn_config=None, rounds=None, patience=None,
        evaluation_samples=1024, report=None, segments=False, **overrides):
    """``segments``: also fit the segment corrections of ``keiba_ai.segments`` (approach A) and, with
    ``evaluate_until``, compare them with the group model.  The saved bundles are the same either way."""
    import numpy as np
    import joblib
    from threadpoolctl import threadpool_limits
    from .cli import fresh_directory
    from .experiment_v3 import _m1_finisher, _shift_months
    from .experiment_v3b import _finisher_rows, _structure
    from .experiment_v3d import Blocks, _params, fit_pl, jockey_timeline, theta_feature_block
    from .experiment_v3i import fit_hybrid_temperatures
    from .experiment_v3j import _blend, group_context
    from .model import feature_revision
    from .neural import Inputs, device_name, predict as nn_predict, train as nn_train
    from .pedigree import PEDIGREE_FEATURES, load_pedigrees, pedigree_block
    from .ranking_v3 import score_races

    report = report or (lambda **message: None)
    settings = dict(DEFAULT_SETTINGS, **overrides)
    if rounds is not None:
        settings['rounds'] = rounds
    if patience is not None:
        settings['patience'] = patience
    nn_config = nn_config or NN_CONFIG
    unknown = [group for group in groups if group not in CONFIG]
    if unknown:
        raise ValueError(f'v4パイプラインの対象外の区分です: {unknown}（対応: {list(GROUPS)}）')
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
    device = device_name(device)
    report(event='loading_pedigrees')
    pedigrees = load_pedigrees(race_db)
    names, no_jockey = _names()
    columns, categorical = _columns(names)
    f0_columns, f0_categorical = _columns(no_jockey)
    params = _params(threads, settings['seed'])
    lgbm = dict(rounds=settings['rounds'], patience=settings['patience'])
    started = time.time()
    summary = dict(pipeline=BUNDLE_VERSION, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   dataset=str(database.resolve()), race_db=str(Path(race_db).resolve()),
                   feature_revision=feature_revision(), train_start=train_start, stop_start=stop_start,
                   train_end=train_end, evaluate_until=evaluate_until,
                   evaluated_test_period=bool(evaluate_until and evaluate_until > TEST_START),
                   trained_on_test_period=train_end > TEST_START, device=device, pedigrees=len(pedigrees),
                   settings=settings, nn_config=asdict(nn_config), nn_seeds=list(seeds), segments=bool(segments),
                   groups={})
    with fresh_directory(output) as temporary:
        for group in groups:
            ctx = group_context(database, group, pedigrees, announce_dropout=settings['announce_dropout'],
                                crossfit_blocks=settings['crossfit_blocks'], seed=settings['seed'], report=report)
            data, race_ids, days, horses = ctx['data'], ctx['race_ids'], ctx['days'], ctx['horses']
            config = CONFIG[group]
            quarter = ctx['quarter']
            fit_mask = (days >= train_start) & (days < stop_start)
            stop_mask = (days >= stop_start) & (days < train_end)
            training = (days >= train_start) & (days < train_end)
            before = days < train_end
            fit_rows, stop_rows = _finisher_rows(data, fit_mask), _finisher_rows(data, stop_mask)
            if not len(fit_rows) or not len(stop_rows):
                raise ValueError(f'{group}: 学習または早期終了用のレースが不足しています。')
            common = ctx['common']
            weight = common[0][0] if config['weight'] else None
            speed = common[-1][0]
            train_version = lambda blocks: [(array, (mask & fit_mask) if mask is not None else None)
                                            for array, mask in blocks]
            bundle = dict(version=BUNDLE_VERSION, group=group, config=config, feature_revision=feature_revision(),
                          feature_names=feature_layout(group), pedigree_features=list(PEDIGREE_FEATURES),
                          base_feature_names=names, settings=settings, train_start=train_start, stop_start=stop_start,
                          train_end=train_end, trained_through=max(days[before].tolist()))
            group_result = dict(rows=int(len(days)), training_races=int(len(set(race_ids[training].tolist()))),
                                excluded_races=data['excluded_races'])
            with threadpool_limits(limits=threads):
                report(event='finisher', group=group)
                finisher, constant, finisher_best = _m1_finisher(
                    data, fit_mask, stop_mask, columns, categorical, threads=threads, seed=settings['seed'], **lgbm)
                bundle.update(finisher=None if finisher is None else finisher.model_to_string(num_iteration=finisher_best),
                              finisher_constant=constant)
                theta = None
                if config['theta']:
                    f0_train = train_version(common)
                    f0_plain = Blocks(data, f0_columns, [(array, None) for array, _ in common])
                    report(event='f0', group=group)
                    f0, f0_best = fit_pl(data, Blocks(data, f0_columns, f0_train), f0_categorical, fit_rows, stop_rows,
                                         params, **lgbm)
                    strength0 = np.full(len(days), np.nan)
                    for block in range(settings['crossfit_blocks']):
                        report(event='crossfit_f0', group=group, block=block)
                        rows = _finisher_rows(data, training & (quarter != block))
                        model, _ = fit_pl(data, Blocks(data, f0_columns, f0_train), f0_categorical, rows, None, params,
                                          fixed_rounds=f0_best, **lgbm)
                        target = np.flatnonzero(before & (quarter == block))
                        if len(target):
                            strength0[target] = model.predict(f0_plain(target), raw_score=True)
                    after = np.flatnonzero(~before)
                    if len(after):
                        strength0[after] = f0.predict(f0_plain(after), num_iteration=f0_best, raw_score=True)
                    timeline = jockey_timeline(days, race_ids, ctx['jockeys'], horses, data['finished'],
                                               data['positions'], strength0, half_life=settings['half_life'],
                                               short_half_life=settings['short_half_life'], prior_sd=settings['prior_sd'])
                    theta = theta_feature_block(timeline, race_ids).astype(np.float32)
                    keep = np.flatnonzero(before)
                    bundle.update(f0=f0.model_to_string(num_iteration=f0_best), f0_best_iteration=f0_best,
                                  oof0_race_ids=race_ids[keep], oof0_horse_ids=horses[keep].astype(str),
                                  oof0_strength=strength0[keep].astype(np.float64))
                blocks = _main_blocks(group, weight, speed, theta, ctx['rating'])
                masks = {id(common[0][0]): common[0][1]} if config['weight'] else {}
                pairs = [(block, masks.get(id(block))) for block in blocks]
                train_blocks, plain = train_version(pairs), Blocks(data, columns, [(block, None) for block in blocks])
                # A: base LightGBM-PL, and its out-of-fold strengths for the pedigree effects
                report(event='model_a', group=group)
                model_a, best_a = fit_pl(data, Blocks(data, columns, train_blocks), categorical, fit_rows, stop_rows,
                                         params, **lgbm)
                strength = np.full(len(days), np.nan)
                for block in range(settings['crossfit_blocks']):
                    report(event='crossfit_a', group=group, block=block)
                    rows = _finisher_rows(data, training & (quarter != block))
                    model, _ = fit_pl(data, Blocks(data, columns, train_blocks), categorical, rows, None, params,
                                      fixed_rounds=best_a, **lgbm)
                    target = np.flatnonzero(before & (quarter == block))
                    if len(target):
                        strength[target] = model.predict(plain(target), raw_score=True)
                after = np.flatnonzero(~before)
                if len(after):
                    strength[after] = model_a.predict(plain(after), num_iteration=best_a, raw_score=True)
                keep = np.flatnonzero(before)
                bundle.update(model_a=model_a.model_to_string(num_iteration=best_a), best_iteration_a=best_a,
                              oofa_race_ids=race_ids[keep], oofa_horse_ids=horses[keep].astype(str),
                              oofa_strength=strength[keep].astype(np.float64))
                # B: + pedigree effects
                report(event='pedigree_features', group=group)
                ped = pedigree_block(days, race_ids, horses, data['finished'], data['positions'], strength,
                                     ctx['surfaces'], ctx['distances'], pedigrees)
                report(event='model_b', group=group)
                model_b, best_b = fit_pl(data, Blocks(data, columns, train_blocks + [(ped, None)]), categorical,
                                         fit_rows, stop_rows, params, **lgbm)
                ped_matrix = Blocks(data, columns, [(block, None) for block in blocks] + [(ped, None)])
                bundle.update(model_b=model_b.model_to_string(num_iteration=best_b), best_iteration_b=best_b)
                stop_b = model_b.predict(ped_matrix(stop_rows), num_iteration=best_b, raw_score=True)
                # NN: TabM-style network, averaged over seeds
                report(event='network', group=group, device=device)
                all_rows = np.arange(len(days))
                inputs = Inputs(Blocks(data, columns, train_blocks)(all_rows), categorical, race_ids, data['finished'],
                                data['positions'], ctx['jockeys'], ctx['trainers'], pedigrees, horses,
                                np.flatnonzero(fit_mask), nn_config)
                states, stop_nn, nn_info, networks = [], [], [], []
                for seed in seeds:
                    member = NeuralConfig(**dict(asdict(nn_config), seed=seed))
                    network, info = nn_train(inputs, np.flatnonzero(fit_mask), np.flatnonzero(stop_mask), member,
                                             device=device, report=lambda **message: report(group=group, seed=seed,
                                                                                            **message))
                    stop_nn.append(nn_predict(network, inputs, stop_rows))
                    states.append({key: value.detach().cpu().numpy() for key, value in network.state_dict().items()})
                    nn_info.append(dict(seed=seed, best_epoch=info['best_epoch'], stop_loss=info['stop_loss'],
                                        seconds=info['seconds']))
                    networks.append(network)
                stop_nn = np.mean(stop_nn, axis=0)
                stop_structure = _structure(data, stop_rows)
                weights = _blend(stop_structure, stop_b, stop_nn)
                stop_d = weights[0] * stop_b + weights[1] * stop_nn
                temperatures, stop_loss, shape = fit_hybrid_temperatures(stop_structure, stop_d)
                segment_models, labels = {}, None
                if segments:
                    segment_models, labels, group_result['segments'] = _fit_segments(
                        data, group, race_ids, days, fit_mask, stop_mask, training, evaluate_until, train_end,
                        Blocks(data, columns, train_blocks + [(ped, None)]), categorical, params, model_b, best_b,
                        networks, seeds, nn_config, inputs, device, report)
                bundle.update(nn=dict(config=asdict(nn_config), encoders=inputs.encoders(), states=states),
                              blend_weights=weights, temperatures=temperatures, temperature_shape=shape)
                group_result.update(best_iteration=best_b, best_iteration_a=best_a, finisher_best_iteration=finisher_best,
                                    networks=nn_info, blend_weights=weights, temperatures=temperatures[:6],
                                    temperature_shape=shape, stop_pl_loss=stop_loss)
                if evaluate_until is not None:
                    rows = np.flatnonzero((days >= train_end) & (days < evaluate_until))
                    if len(rows):
                        report(event='evaluate', group=group, rows=int(len(rows)))
                        score_b = model_b.predict(ped_matrix(rows), num_iteration=best_b, raw_score=True)
                        seed_scores = np.asarray([nn_predict(network, inputs, rows) for network in networks])
                        score_nn = seed_scores.mean(axis=0)
                        spread = seed_scores.std(axis=0)
                        score_d = weights[0] * score_b + weights[1] * score_nn
                        completion = (np.full(len(rows), constant) if finisher is None else
                                      finisher.predict(data['x'][np.ix_(rows, columns)], num_iteration=finisher_best))
                        args = (race_ids[rows], data['positions'][rows], data['finished'][rows], data['race_numbers'][rows])
                        evaluation = {}
                        score_segmented = None
                        if segments:
                            score_segmented = _segment_scores(rows, labels, segment_models, score_b, score_nn,
                                                              weights, Blocks(data, columns, train_blocks + [(ped, None)]),
                                                              inputs)
                        part_temperatures = dict(model=temperatures,
                                                 lightgbm_pedigree=fit_hybrid_temperatures(stop_structure, stop_b)[0],
                                                 network=fit_hybrid_temperatures(stop_structure, stop_nn)[0])
                        (temporary / f'evaluation_temperatures_{group}.json').write_text(
                            json.dumps({key: [float(v) for v in value] for key, value in part_temperatures.items()}),
                            encoding='utf-8')
                        for name, scores, temps in (
                                ('model', score_d, temperatures),
                                ('lightgbm_pedigree', score_b, part_temperatures['lightgbm_pedigree']),
                                ('network', score_nn, part_temperatures['network']),
                                ('uniform', np.zeros(len(rows)), [1.0])):
                            evaluation[name] = score_races(*args, scores, completion, temps, samples=evaluation_samples,
                                                           seed=settings['seed'])['metrics']
                        if score_segmented is not None:
                            evaluation['segmented'] = score_races(*args, score_segmented, completion, temperatures,
                                                                  samples=evaluation_samples,
                                                                  seed=settings['seed'])['metrics']
                            for item in group_result['segments']:
                                chosen = np.flatnonzero(labels[rows] == item['name'])
                                if not len(chosen):
                                    continue
                                part = tuple(array[chosen] for array in args)
                                item['evaluation'] = {
                                    name: score_races(*part, scores[chosen], completion[chosen], temperatures,
                                                      samples=evaluation_samples, seed=settings['seed'])['metrics']
                                    for name, scores in (('model', score_d), ('segmented', score_segmented))}
                        group_result['evaluation'] = evaluation
                        with (temporary / f'evaluation_{group}.csv').open('w', encoding='utf-8', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(['race_date', 'race_id', 'horse_id', 'finish_position', 'strength_score',
                                             'finish_probability', 'lightgbm_score', 'network_score', 'network_spread']
                                            + (['segment', 'segmented_score'] if score_segmented is not None else []))
                            for index, (row, score, probability, b, nn, sd) in enumerate(
                                    zip(rows, score_d, completion, score_b, score_nn, spread)):
                                writer.writerow([days[row], race_ids[row], horses[row],
                                                 int(data['positions'][row]) if data['finished'][row] else '',
                                                 f'{score:.6f}', f'{probability:.6f}', f'{b:.6f}', f'{nn:.6f}',
                                                 f'{sd:.6f}']
                                                + ([labels[row], f'{score_segmented[index]:.6f}']
                                                   if score_segmented is not None else []))
                networks.clear()
            joblib.dump(bundle, temporary / f'{group}.joblib', compress=3)
            summary['groups'][group] = group_result
            report(event='group_done', group=group,
                   evaluation=(group_result.get('evaluation') or {}).get('model', {}).get('race_outcome_log_loss'))
            del data, ctx, inputs
        summary['seconds'] = round(time.time() - started, 1)
        (temporary / 'model.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + '\n',
                                              encoding='utf-8')
    return summary


def _fit_segments(data, group, race_ids, days, fit_mask, stop_mask, training, evaluate_until, train_end, matrix_b,
                  categorical, params, model_b, best_b, networks, seeds, nn_config, inputs, device, report):
    """Approach A of ``keiba_ai.segments``: per segment, B continued from its scores and each NN seed fine-tuned,
    both early stopped on the segment's stop races.  Returns (models by segment, row labels, summary)."""
    import numpy as np
    from .experiment_v3b import _finisher_rows
    from .segments import MIN_FIT_RACES, MIN_STOP_RACES, fine_tune_network, residual_lightgbm, segment_labels
    labels = segment_labels(data['x'], group)
    needed = fit_mask | stop_mask | ((days >= train_end) & (days < evaluate_until) if evaluate_until else False)
    base = np.full(len(days), np.nan)
    rows = np.flatnonzero(needed)
    for start in range(0, len(rows), 100000):
        chunk = rows[start:start + 100000]
        base[chunk] = model_b.predict(matrix_b(chunk), num_iteration=best_b, raw_score=True)
    races_of = lambda selected: len(set(race_ids[selected].tolist()))
    names = sorted(set(labels[training].tolist()), key=lambda name: -int((labels[training] == name).sum()))
    models, summary = {}, []
    for name in names:
        mask = labels == name
        fit_rows, stop_rows = _finisher_rows(data, fit_mask & mask), _finisher_rows(data, stop_mask & mask)
        item = dict(name=name, fit_races=races_of(fit_rows), stop_races=races_of(stop_rows))
        summary.append(item)
        if item['fit_races'] < MIN_FIT_RACES or item['stop_races'] < MIN_STOP_RACES:
            item['used'] = False
            continue
        report(event='segment', group=group, segment=name, fit_races=item['fit_races'], stop_races=item['stop_races'])
        booster, rounds, losses = residual_lightgbm(data, matrix_b, categorical, fit_rows, stop_rows, base, params)
        tuned, epochs = [], []
        for seed, network in zip(seeds, networks):
            member = NeuralConfig(**dict(asdict(nn_config), seed=seed))
            model, info = fine_tune_network(network, inputs, np.flatnonzero(fit_mask & mask),
                                            np.flatnonzero(stop_mask & mask), member, device=device)
            tuned.append(model)
            epochs.append(info['best_epoch'])
        models[name] = (booster, rounds, tuned)
        item.update(used=True, lgbm_rounds=rounds, lgbm_stop_loss=losses, nn_best_epochs=epochs)
        report(event='segment_done', group=group, segment=name, lgbm_rounds=rounds, nn_best_epochs=epochs)
    return models, labels, summary


def _segment_scores(rows, labels, models, score_b, score_nn, weights, matrix_b, inputs):
    """Blended scores of ``rows`` with each used segment's corrections (others keep the group model)."""
    import numpy as np
    from .neural import predict as nn_predict
    b, nn = score_b.copy(), score_nn.copy()
    for name, (booster, rounds, tuned) in models.items():
        chosen = np.flatnonzero(labels[rows] == name)
        if not len(chosen):
            continue
        if booster is not None:
            b[chosen] += booster.predict(matrix_b(rows[chosen]), num_iteration=rounds, raw_score=True)
        nn[chosen] = np.mean([nn_predict(model, inputs, rows[chosen]) for model in tuned], axis=0)
    return weights[0] * b + weights[1] * nn


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
        raise ValueError(f'v4パイプラインのモデルではありません: {path}')
    if bundle['feature_revision'] != feature_revision() or bundle['feature_names'] != feature_layout(group):
        raise ValueError('モデルと特徴量の版が一致しません。fitから作成し直してください。')
    booster = lambda text: lgb.Booster(model_str=text)
    bundle['model_a_booster'], bundle['model_b_booster'] = booster(bundle['model_a']), booster(bundle['model_b'])
    bundle['finisher_booster'] = None if bundle['finisher'] is None else booster(bundle['finisher'])
    bundle['oofa'] = _key_map(bundle['oofa_race_ids'], bundle['oofa_horse_ids'], bundle['oofa_strength'])
    if bundle['config']['theta']:
        bundle['f0_booster'] = booster(bundle['f0'])
        bundle['oof0'] = _key_map(bundle['oof0_race_ids'], bundle['oof0_horse_ids'], bundle['oof0_strength'])
    return bundle


def _networks(bundle, inputs):
    import torch
    from .neural import build_model
    if '_networks' in bundle:
        return bundle['_networks']
    config = NeuralConfig(**bundle['nn']['config'])
    models = bundle['_networks'] = []
    for state in bundle['nn']['states']:
        model = build_model(inputs, config)
        model.load_state_dict({key: torch.from_numpy(value) for key, value in state.items()})
        model.eval()
        models.append(model)
    return models


class Predictor:
    """Cached v4 predictor with the same interface as ``pipeline_v3.Predictor``."""

    def __init__(self, database, model_dir, race_db, *, threads=2, report=None, cached_days=2):
        import threading
        self.database, self.model_dir, self.race_db = Path(database), Path(model_dir), Path(race_db)
        self.threads, self.cached_days = threads, cached_days
        self.report = report or (lambda **message: None)
        self.last_day = _check_dataset(self.database)
        self.bundles, self.loaded, self.histories, self.day_states = {}, {}, OrderedDict(), OrderedDict()
        self.pedigrees = None
        self.lock = threading.Lock()

    @staticmethod
    def supported(race):
        from .pipeline_v3 import Predictor as PredictorV3
        return PredictorV3.supported(race)

    def bundle(self, group):
        if group not in self.bundles:
            self.bundles[group] = _load_bundle(self.model_dir, group)
        return self.bundles[group]

    def train_end(self, group):
        return self.bundle(group)['train_end']

    def _pedigrees(self, horses=()):
        from .pedigree import load_pedigrees
        if self.pedigrees is None:
            self.report(event='loading_pedigrees')
            self.pedigrees = load_pedigrees(self.race_db)
        missing = [horse for horse in horses if horse not in self.pedigrees]
        if missing:
            self.pedigrees.update(load_pedigrees(self.race_db, horse_ids=missing))
        return self.pedigrees

    def _group(self, group):
        import numpy as np
        from .experiment_v3c import load_announced
        from .experiment_v3d import load_jockeys
        from .experiment_v3j import load_field
        from .rank_model import load_rank_group
        if group not in self.loaded:
            self.report(event='loading_group', group=group)
            data = load_rank_group(self.database, group)
            data['race_ids'] = np.asarray([identity[0] for identity in data['identities']])
            horses = np.asarray([identity[1] for identity in data['identities']], dtype=object)
            stored = dict(jockeys=load_jockeys(self.database, group, data['identities']),
                          trainers=load_field(self.database, group, data['identities'], 'trainer_id'))
            if self.bundle(group)['config']['weight']:
                stored['announced'] = load_announced(self.database, group, data['identities'])
            self.loaded[group] = (data, horses, stored)
        return self.loaded[group]

    def _history(self, day):
        from .same_day import rebuild_prediction_states
        if day not in self.histories:
            self.report(event='history', day=day)
            while len(self.histories) >= self.cached_days:
                self.histories.popitem(last=False)
            self.histories[day] = rebuild_prediction_states(self.database, day)[:2]
        return self.histories[day]

    def _day_state(self, group, day):
        """Causal states after every stored day before ``day`` (computed once per group and day)."""
        import numpy as np
        from .experiment_v3c import WEIGHT_FEATURES, announced_features
        from .experiment_v3d import jockey_timeline, theta_feature_block
        from .features import FEATURE_NAMES
        from .pedigree import pedigree_block
        from .ratings import rating_block
        from .speed import SPEED_FEATURES, load_speed_state
        key = (group, day)
        if key in self.day_states:
            return self.day_states[key]
        bundle, settings = self.bundle(group), self.bundle(group)['settings']
        config = bundle['config']
        data, horses, stored = self._group(group)
        names, no_jockey = _names()
        columns, _ = _columns(names)
        f0_columns, _ = _columns(no_jockey)
        self.report(event='history_states', group=group, day=day)
        rows = np.flatnonzero(data['days'] < day)
        x = data['x'][rows]
        days, race_ids, hist_horses = data['days'][rows], data['race_ids'][rows], horses[rows]
        finished, positions = data['finished'][rows], data['positions'][rows]
        keys = list(zip(race_ids.tolist(), hist_horses.tolist()))
        figures, table = load_speed_state(self.database, group, day)
        speed = np.full((len(rows), len(SPEED_FEATURES)), np.nan, dtype=np.float32)
        for index, identity in enumerate(keys):
            values = table.get(identity)
            if values is not None:
                speed[index] = [np.nan if value is None else value for value in values]
        rating, rating_state = rating_block(days, race_ids, hist_horses, finished, positions, return_state=True)
        weight = None
        if config['weight']:
            announced = {name: values[rows] for name, values in stored['announced'].items()}
            weight = announced_features(dict(x=x, race_ids=race_ids), announced)[:, :len(WEIGHT_FEATURES)]
        theta, theta_state = None, None
        if config['theta']:
            f0_matrix = lambda index: np.concatenate([x[np.ix_(index, f0_columns)]] +
                                                     [block[index] for block in _f0_blocks(group, weight, speed)], axis=1)
            strength0 = _fill_strength(keys, bundle['oof0'],
                                       lambda index: bundle['f0_booster'].predict(f0_matrix(index), raw_score=True))
            timeline, theta_state = jockey_timeline(
                days, race_ids, stored['jockeys'][rows], hist_horses, finished, positions, strength0,
                half_life=settings['half_life'], short_half_life=settings['short_half_life'],
                prior_sd=settings['prior_sd'], return_state=True)
            theta = theta_feature_block(timeline, race_ids).astype(np.float32)
        main = lambda index: np.concatenate([x[np.ix_(index, columns)]] +
                                            [block[index] for block in _main_blocks(group, weight, speed, theta, rating)],
                                            axis=1)
        strength = _fill_strength(keys, bundle['oofa'],
                                  lambda index: bundle['model_a_booster'].predict(main(index), raw_score=True))
        _, ped_state = pedigree_block(days, race_ids, hist_horses, finished, positions, strength,
                                      x[:, FEATURE_NAMES.index('surface_code')], x[:, FEATURE_NAMES.index('distance_m')],
                                      self._pedigrees(), return_state=True)
        while len([item for item in self.day_states if item[0] == group]) >= self.cached_days:
            self.day_states.pop(next(item for item in self.day_states if item[0] == group))
        state = dict(speed=figures, rating=rating_state, theta=theta_state, pedigree=ped_state)
        self.day_states[key] = state
        return state

    def _group_day(self, group, day, races, history, priors, *, samples):
        import numpy as np
        import torch
        from .bias import DEFAULT_BIAS_CONFIG, IntradayBiasState
        from .experiment_v3c import WEIGHT_FEATURES, announced_features
        from .experiment_v3d import theta_feature_block, timeline_query
        from .features import FEATURE_NAMES, _GOING, _WEATHER, race_features
        from .neural import Inputs, predict as nn_predict
        from .pedigree import pedigree_query
        from .rank_model import race_seed
        from .ranking_v3 import staged_rank_distribution
        from .ratings import rating_query
        from .speed import speed_row
        bundle = self.bundle(group)
        config = bundle['config']
        state = self._day_state(group, day)
        names, _ = _names()
        columns, _ = _columns(names)
        entry_rows, entry_x, speed = [], [], []
        for race, active in races:
            vectors = race_features(active, history, intraday=IntradayBiasState(day), priors=priors, full_race=race,
                                    bias_config=DEFAULT_BIAS_CONFIG)
            entry_x.extend([[np.nan if value is None else value for value in vector] for vector in vectors])
            speed.extend(state['speed'].vectors([speed_row(dict(row, day=day)) for row in active]))
            entry_rows.extend(active)
        count = len(entry_rows)
        x = np.asarray(entry_x, dtype=np.float32)
        race_ids = np.asarray([row['race_id'] for row in entry_rows])
        horses = np.asarray([row['horse_id'] for row in entry_rows], dtype=object)
        jockeys = np.asarray([row.get('jockey_id') or '' for row in entry_rows], dtype=object)
        trainers = np.asarray([row.get('trainer_id') or '' for row in entry_rows], dtype=object)
        speed = np.asarray([[np.nan if value is None else value for value in vector] for vector in speed],
                           dtype=np.float32)
        rating = rating_query(state['rating'], race_ids, horses, day)
        weight = theta = None
        if config['weight']:
            announced = dict(
                weight=np.asarray([row['weight_kg'] if row.get('weight_kg') is not None else np.nan for row in entry_rows]),
                change=np.asarray([row['weight_change_kg'] if row.get('weight_change_kg') is not None else np.nan
                                   for row in entry_rows]),
                going=np.asarray([_GOING.get(row.get('track_condition'), np.nan) for row in entry_rows]),
                weather=np.asarray([_WEATHER.get(row.get('weather'), np.nan) for row in entry_rows]))
            weight = announced_features(dict(x=x, race_ids=race_ids), announced)[:, :len(WEIGHT_FEATURES)]
        if config['theta']:
            theta = theta_feature_block(timeline_query(state['theta'], jockeys, horses, day), race_ids).astype(np.float32)
        pedigrees = self._pedigrees(horses.tolist())
        ped = pedigree_query(state['pedigree'], race_ids, horses, x[:, FEATURE_NAMES.index('surface_code')],
                             x[:, FEATURE_NAMES.index('distance_m')], pedigrees, day)
        main = np.concatenate([x[:, columns]] + _main_blocks(group, weight, speed, theta, rating), axis=1)
        score_b = bundle['model_b_booster'].predict(np.concatenate([main, ped], axis=1), raw_score=True)
        inputs = Inputs.from_encoders(bundle['nn']['encoders'], main, race_ids, np.zeros(count, dtype=bool),
                                      np.zeros(count, dtype=np.int64), jockeys, trainers, pedigrees, horses)
        rows = np.arange(count)
        with torch.no_grad():
            score_nn = np.mean([nn_predict(network, inputs, rows, device='cpu') for network in _networks(bundle, inputs)],
                               axis=0)
        weights = bundle['blend_weights']
        scores = weights[0] * score_b + weights[1] * score_nn
        completion = (np.full(count, bundle['finisher_constant']) if bundle['finisher_booster'] is None else
                      bundle['finisher_booster'].predict(x[:, columns]))
        output, start = [], 0
        for race, active in races:
            size = len(active)
            part = slice(start, start + size)
            distribution = staged_rank_distribution(scores[part], completion[part], bundle['temperatures'],
                                                    samples=samples,
                                                    seed=race_seed(active[0]['race_id'], bundle['settings']['seed']))
            for local, row in enumerate(active):
                ranks = distribution[local, :size]
                finish = float(completion[part][local])
                result = dict(race_date=day, region=row['region'], race_id=row['race_id'], horse_id=row['horse_id'],
                              horse_number=row['horse_number'], group=group, starters=size,
                              strength_score=float(scores[part][local]), lightgbm_score=float(score_b[part][local]),
                              network_score=float(score_nn[part][local]), finish_probability=finish,
                              nonfinish_probability=float(distribution[local, -1]), win_probability=float(ranks[0]),
                              top3_probability=float(np.clip(ranks[:3].sum(), 0, 1)),
                              expected_rank_given_finish=float((ranks * np.arange(1, size + 1)).sum() / finish)
                              if finish > 0 else None,
                              bodyweight_used=bool(config['weight'] and row.get('weight_kg') is not None))
                result.update({f'rank_{rank}_probability': float(value) for rank, value in enumerate(ranks, 1)})
                output.append(result)
            start += size
        return output

    def predict_races(self, races, *, samples=4096):
        from threadpoolctl import threadpool_limits
        by_day = {}
        for day, region, race_id, race in races:
            active = [row for row in race if row.get('finish_status') not in ('scratched', 'excluded')]
            if not active:
                continue
            group = active[0]['group_name']
            if group not in CONFIG:
                self.report(event='skipped_race', race_id=race_id, group=group, reason='v4パイプラインの対象外の区分')
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
                    self.report(event='predicting', day=day, group=group, races=len(group_races))
                    results.extend(self._group_day(group, day, group_races, history, priors, samples=samples))
        return results


def predict(database, race_db, model_dir, entries, output, *, samples=4096, threads=2, report=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'出力先は既にあります: {output}')
    results = Predictor(database, model_dir, race_db, threads=threads, report=report).predict_races(
        _read_entries(entries), samples=samples)
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


def load_predictor(database, model_dir, race_db, *, threads=2, report=None):
    """The predictor that matches the saved model (v4, or v3 for older model folders)."""
    try:
        summary = json.loads((Path(model_dir) / 'model.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        summary = {}
    if summary.get('pipeline') == BUNDLE_VERSION:
        return Predictor(database, model_dir, race_db, threads=threads, report=report)
    from .pipeline_v3 import Predictor as PredictorV3
    return PredictorV3(database, model_dir, threads=threads, report=report)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    from .priority import background
    background()
    parser = argparse.ArgumentParser(description='v4順位分布モデル（LightGBM＋血統 と TabM型ニューラルネットの混ぜ合わせ）')
    commands = parser.add_subparsers(dest='command', required=True)
    fitting = commands.add_parser('fit', help='区分ごとのモデルを学習・保存します')
    fitting.add_argument('--dataset', required=True, type=Path)
    fitting.add_argument('--race-db', required=True, type=Path, help='血統を収集した data\\keiba.sqlite3')
    fitting.add_argument('--output-dir', required=True, type=Path)
    fitting.add_argument('--train-end', help='この日より前で学習（既定: データ最終日の翌日）')
    fitting.add_argument('--train-start', help='この日以降で学習（既定: 2022-04-01。DBの最初の日から半年ほど後にします）')
    fitting.add_argument('--evaluate-until', help='train-end以降、この日より前を評価')
    fitting.add_argument('--segments', action='store_true',
                         help='区分（芝ダ×距離×年齢×クラス）ごとの補正も学習し、--evaluate-untilの期間で比べる（保存するモデルは同じ）')
    fitting.add_argument('--allow-test-period', action='store_true')
    fitting.add_argument('--groups', nargs='+', default=list(GROUPS), choices=list(GROUPS))
    fitting.add_argument('--threads', type=int, default=2)
    fitting.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    fitting.add_argument('--rounds', type=int)
    fitting.add_argument('--patience', type=int)
    fitting.add_argument('--evaluation-samples', type=int, default=1024)
    predicting = commands.add_parser('predict', help='出馬表CSVから各馬の順位分布を出力します')
    predicting.add_argument('--dataset', required=True, type=Path)
    predicting.add_argument('--race-db', required=True, type=Path)
    predicting.add_argument('--model-dir', required=True, type=Path)
    predicting.add_argument('--entries', required=True, type=Path)
    predicting.add_argument('--output', required=True, type=Path)
    predicting.add_argument('--samples', type=int, default=4096)
    predicting.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    from .cli import emit
    try:
        import lightgbm  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        emit(event='error', message=f'{exc.name}がありません。scripts\\setup-ml.ps1 と scripts\\setup-nn.ps1 を実行してください。')
        return 2
    try:
        if args.command == 'fit':
            summary = fit(args.dataset, args.race_db, args.output_dir, train_end=args.train_end,
                          evaluate_until=args.evaluate_until, allow_test_period=args.allow_test_period,
                          groups=args.groups, threads=args.threads, device=args.device, rounds=args.rounds,
                          patience=args.patience, evaluation_samples=args.evaluation_samples, report=emit, segments=args.segments,
                          **({'train_start': args.train_start} if args.train_start else {}))
            emit(event='fit_complete', model=str(args.output_dir / 'model.json'), train_end=summary['train_end'],
                 seconds=summary['seconds'])
        else:
            rows = predict(args.dataset, args.race_db, args.model_dir, args.entries, args.output, samples=args.samples,
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
