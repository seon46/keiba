"""V3 experiment J: pedigree features and a neural strength model.

Base per group = the pipeline's best configuration (central: speed + ratings;
local: bodyweight + speed + jockey theta + ratings).  Per validation year:

``A_lgbm``      base LightGBM-PL (the current model),
``B_lgbm_ped``  A + causal pedigree features (sire / damsire / dam thetas with
                an out-of-fold strength offset, by surface and distance band),
``C_nn``        neural PL model on A's features + learned pedigree (62 ancestors),
                jockey and trainer embeddings + attention across the runners,
``D_blend``     a * B + b * C, weights fitted on the early-stopping period,
``E_blend``     a * A + b * C (the neural gain without pedigree features).

Every variant gets its own ``stop_hybrid`` temperatures (fitted on the stop
period) and the same finisher model.  The final-test period is never read.
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
from .experiment_v3d import (JOCKEY_RATE_FEATURES, Blocks, _params, _quarter_block, fit_pl, jockey_timeline,
                             load_jockeys, theta_feature_block)
from .experiment_v3f import _crossfit_strength
from .experiment_v3i import fit_hybrid_temperatures
from .pedigree import PEDIGREE_FEATURES, load_pedigrees, pedigree_block
from .ratings import rating_block
from .speed import load_speed_block

GROUPS = ('central_flat', 'local_flat')
WEIGHT_GROUPS = ('local_flat',)
THETA_GROUPS = ('local_flat',)
VARIANTS = ('A_lgbm', 'B_lgbm_ped', 'C_nn', 'D_blend', 'E_blend')
BASELINE = 'A_lgbm'


def load_field(database, group, identities, field):
    import numpy as np
    from .model import readonly
    wanted = {(race, horse): index for index, (race, horse, _) in enumerate(identities)}
    values = np.full(len(identities), '', dtype=object)
    connection = readonly(database)
    try:
        for race, horse, value in connection.execute(
                f"SELECT f.race_id,f.horse_id,json_extract(p.payload_json,'$.{field}') FROM features f JOIN input_rows p "
                "ON p.region=f.region AND p.race_id=f.race_id AND p.horse_id=f.horse_id WHERE f.group_name=?", (group,)):
            index = wanted.get((race, horse))
            if index is not None and value:
                values[index] = str(value)
    finally:
        connection.close()
    return values


def _blend(structure, first, second):
    import numpy as np
    from scipy.optimize import minimize
    result = minimize(lambda weights: structure.mean_loss(weights[0] * first + weights[1] * second),
                      np.asarray([.5, .5]), method='L-BFGS-B', bounds=[(0, 3), (0, 3)])
    return [float(value) for value in result.x]


def group_context(database, group, pedigrees, *, announce_dropout=.05, crossfit_blocks=3, seed=42, report=None):
    """Everything per group that does not depend on the fold."""
    import numpy as np
    from .features import FEATURE_NAMES
    from .rank_model import load_rank_group
    report = report or (lambda **message: None)
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
    return dict(group=group, data=data, race_ids=race_ids, days=days, horses=horses, common=common,
                rating=rating_block(days, race_ids, horses, data['finished'], data['positions']),
                jockeys=load_jockeys(database, group, data['identities']),
                trainers=load_field(database, group, data['identities'], 'trainer_id'),
                surfaces=data['x'][:, FEATURE_NAMES.index('surface_code')],
                distances=data['x'][:, FEATURE_NAMES.index('distance_m')],
                quarter=_quarter_block(days, crossfit_blocks),
                coverage=float(np.mean([horse in pedigrees for horse in horses.tolist()])))


def lgbm_fold(ctx, fold, pedigrees, *, params, threads, rounds, patience, crossfit_blocks, seed, report=None):
    """Finisher model, A (base LightGBM-PL) and B (+ pedigree features) for one fold."""
    import numpy as np
    from .features import FEATURE_NAMES, CATEGORICAL_FEATURES
    report = report or (lambda **message: None)
    names = experiment_feature_names()
    no_jockey = [name for name in names if name not in JOCKEY_RATE_FEATURES]
    columns_for = lambda selected: [FEATURE_NAMES.index(name) for name in selected]
    categorical_for = lambda selected: [index for index, name in enumerate(selected) if name in CATEGORICAL_FEATURES]
    columns, categorical = columns_for(names), categorical_for(names)
    group, data, race_ids, days, horses = ctx['group'], ctx['data'], ctx['race_ids'], ctx['days'], ctx['horses']
    common, rating, jockeys, quarter = ctx['common'], ctx['rating'], ctx['jockeys'], ctx['quarter']
    surfaces, distances = ctx['surfaces'], ctx['distances']
    year = fold['validation_year']
    fit = (days >= fold['train_start']) & (days < fold['stop_start'])
    stop = (days >= fold['stop_start']) & (days < fold['validation_start'])
    validation = (days >= fold['validation_start']) & (days < fold['validation_end'])
    training = (days >= fold['train_start']) & (days < fold['validation_start'])
    before = days < fold['validation_start']
    upto = days < fold['validation_end']
    fit_rows, stop_rows = _finisher_rows(data, fit), _finisher_rows(data, stop)
    validation_rows = np.flatnonzero(validation)
    finisher, constant, finisher_best = _m1_finisher(
        data, fit, stop, columns, categorical, threads=threads, rounds=rounds, patience=patience, seed=seed)
    completion = (np.full(len(validation_rows), constant) if finisher is None else
                  finisher.predict(data['x'][np.ix_(validation_rows, columns)], num_iteration=finisher_best))
    blocks = list(common)
    if group in THETA_GROUPS:
        report(event='strength_for_theta', group=group, validation_year=year)
        strength0 = _crossfit_strength(
            data, columns_for(no_jockey), categorical_for(no_jockey), common, fold, quarter, params,
            rounds=rounds, patience=patience, crossfit_blocks=crossfit_blocks, fit_mask=fit,
            stop_mask=stop, report=report, group=group)
        span = np.flatnonzero(upto)
        part = jockey_timeline(days[span], race_ids[span], jockeys[span], horses[span],
                               data['finished'][span], data['positions'][span], strength0[span])
        timeline = {key: np.full(len(days), np.nan) for key in part}
        for key, values in part.items():
            timeline[key][span] = values
        blocks.append((theta_feature_block(timeline, race_ids), None))
    blocks.append((rating, None))
    train_blocks = [(a, (m & fit) if m is not None else None) for a, m in blocks]
    plain_blocks = [(a, None) for a, _ in blocks]
    base_matrix = Blocks(data, columns, plain_blocks)
    variants = {}
    scores, stop_scores = {}, {}
    # A ----------------------------------------------------------------
    report(event='variant', group=group, validation_year=year, variant='A_lgbm')
    model_a, best_a = fit_pl(data, Blocks(data, columns, train_blocks), categorical, fit_rows, stop_rows,
                             params, rounds=rounds, patience=patience)
    scores['A_lgbm'] = model_a.predict(base_matrix(validation_rows), num_iteration=best_a, raw_score=True)
    stop_scores['A_lgbm'] = model_a.predict(base_matrix(stop_rows), num_iteration=best_a, raw_score=True)
    variants['A_lgbm'] = dict(best_iteration=best_a)
    # strength offsets for the pedigree effects: out-of-fold before, A inside the year
    strength = np.full(len(days), np.nan)
    for block in range(crossfit_blocks):
        report(event='oof_model', group=group, validation_year=year, block=block)
        rows = _finisher_rows(data, training & (quarter != block))
        model, _ = fit_pl(data, Blocks(data, columns, train_blocks), categorical, rows, None, params,
                          rounds=rounds, patience=patience, fixed_rounds=best_a)
        target = np.flatnonzero(before & (quarter == block))
        if len(target):
            strength[target] = model.predict(base_matrix(target), raw_score=True)
    strength[validation_rows] = scores['A_lgbm']
    # B ----------------------------------------------------------------
    report(event='pedigree_features', group=group, validation_year=year)
    span = np.flatnonzero(upto)
    ped = np.full((len(days), len(PEDIGREE_FEATURES)), np.nan, dtype=np.float32)
    ped[span] = pedigree_block(days[span], race_ids[span], horses[span], data['finished'][span],
                               data['positions'][span], strength[span], surfaces[span],
                               distances[span], pedigrees)
    report(event='variant', group=group, validation_year=year, variant='B_lgbm_ped')
    model_b, best_b = fit_pl(data, Blocks(data, columns, train_blocks + [(ped, None)]), categorical,
                             fit_rows, stop_rows, params, rounds=rounds, patience=patience)
    ped_matrix = Blocks(data, columns, plain_blocks + [(ped, None)])
    scores['B_lgbm_ped'] = model_b.predict(ped_matrix(validation_rows), num_iteration=best_b, raw_score=True)
    stop_scores['B_lgbm_ped'] = model_b.predict(ped_matrix(stop_rows), num_iteration=best_b, raw_score=True)
    importance = model_b.feature_importance(importance_type='gain', iteration=best_b)
    order = np.argsort(-importance)
    ranks = {int(index): position + 1 for position, index in enumerate(order)}
    offset = len(importance) - len(PEDIGREE_FEATURES)
    variants['B_lgbm_ped'] = dict(
        best_iteration=best_b, features=len(importance),
        pedigree_gain_rank={name: ranks[offset + i] for i, name in enumerate(PEDIGREE_FEATURES)})
    all_rows = np.arange(len(days))
    return dict(fit_mask=fit, stop_mask=stop, validation_mask=validation, stop_rows=stop_rows,
                validation_rows=validation_rows, completion=completion, train_blocks=train_blocks,
                nn_matrix=Blocks(data, columns, train_blocks)(all_rows), categorical=categorical,
                scores=scores, stop_scores=stop_scores, variants=variants)


def run_experiment(database, race_db, output, *, validation_years=DEFAULT_VALIDATION_YEARS, groups=GROUPS,
                   train_start='2022-04-01', test_start='2026-01-01', stop_months=2, threads=2, rounds=8000,
                   patience=150, announce_dropout=.05, crossfit_blocks=3, evaluation_samples=1024, seed=42,
                   device='auto', neural=None, report=None):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .features import FEATURE_NAMES, CATEGORICAL_FEATURES
    from .model import feature_revision, readonly
    from .neural import Inputs, NeuralConfig, device_name, predict as nn_predict, train as nn_train
    from .rank_model import load_rank_group
    from .ranking_v3 import paired_bootstrap, score_races

    report = report or (lambda **message: None)
    database, output = Path(database), Path(output)
    neural = neural or NeuralConfig(seed=seed)
    device = device_name(device)
    if output.exists():
        raise ValueError(f'出力先は既にあります。別の出力先を指定してください: {output}')
    if any(f'{year:04d}-12-31' >= test_start for year in validation_years):
        raise ValueError('最終評価期間（test-start以降）を検証年に含めることはできません。')
    connection = readonly(database)
    try:
        metadata = dict(connection.execute('SELECT key,value_json FROM ai_metadata'))
    except sqlite3.Error as exc:
        raise ValueError(f'prepare済みのfeatures.sqlite3を指定してください: {exc}') from exc
    finally:
        connection.close()
    if json.loads(metadata['feature_revision']) != feature_revision():
        raise ValueError('特徴量コードの版がこのfeatures.sqlite3と一致しません。prepareから作成し直してください。')
    report(event='loading_pedigrees')
    pedigrees = load_pedigrees(race_db)
    names = experiment_feature_names()
    no_jockey = [name for name in names if name not in JOCKEY_RATE_FEATURES]
    columns_for = lambda selected: [FEATURE_NAMES.index(name) for name in selected]
    categorical_for = lambda selected: [index for index, name in enumerate(selected) if name in CATEGORICAL_FEATURES]
    folds = []
    for year in validation_years:
        start = f'{year:04d}-01-01'
        stop_start = _shift_months(start, -stop_months)
        folds.append(dict(validation_year=year, train_start=train_start, stop_start=stop_start,
                          validation_start=start, validation_end=f'{year + 1:04d}-01-01'))
    started = time.time()
    result = dict(experiment='v3j_pedigree_neural', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  dataset=str(database.resolve()), feature_revision=feature_revision(), folds=folds,
                  test_start=test_start, test_period_used=False, baseline=BASELINE, pedigrees=len(pedigrees),
                  pedigree_features=list(PEDIGREE_FEATURES), device=device,
                  settings=dict(threads=threads, rounds=rounds, patience=patience, crossfit_blocks=crossfit_blocks,
                                evaluation_samples=evaluation_samples, seed=seed, neural=neural.__dict__),
                  groups={})
    output.mkdir(parents=True)
    losses_file = (output / 'race_losses.csv').open('w', encoding='utf-8', newline='')
    losses = csv.writer(losses_file)
    losses.writerow(['group', 'config', 'validation_year', 'race_id', 'outcome_log_loss', 'top3_log_loss', 'win_log_loss'])
    params = _params(threads, seed)
    columns, categorical = columns_for(names), categorical_for(names)
    try:
        for group in groups:
            ctx = group_context(database, group, pedigrees, announce_dropout=announce_dropout,
                                crossfit_blocks=crossfit_blocks, seed=seed, report=report)
            data, race_ids, days = ctx['data'], ctx['race_ids'], ctx['days']
            group_result = dict(excluded_races=data['excluded_races'], pedigree_coverage=ctx['coverage'], folds=[])
            per_race = {}
            with threadpool_limits(limits=threads):
                for fold in folds:
                    year = fold['validation_year']
                    prepared = lgbm_fold(ctx, fold, pedigrees, params=params, threads=threads, rounds=rounds,
                                         patience=patience, crossfit_blocks=crossfit_blocks, seed=seed, report=report)
                    fit, stop = prepared['fit_mask'], prepared['stop_mask']
                    stop_rows, validation_rows = prepared['stop_rows'], prepared['validation_rows']
                    completion, train_blocks = prepared['completion'], prepared['train_blocks']
                    scores, stop_scores = dict(prepared['scores']), dict(prepared['stop_scores'])
                    fold_result = dict(validation_year=year, variants=prepared['variants'])
                    jockeys, trainers, horses = ctx['jockeys'], ctx['trainers'], ctx['horses']
                    # C ----------------------------------------------------------------
                    report(event='variant', group=group, validation_year=year, variant='C_nn', device=device)
                    nn_matrix = prepared['nn_matrix']
                    inputs = Inputs(nn_matrix, prepared['categorical'], race_ids, data['finished'], data['positions'], jockeys,
                                    trainers, pedigrees, horses, np.flatnonzero(fit), neural)
                    network, info = nn_train(inputs, np.flatnonzero(fit), np.flatnonzero(stop), neural, device=device,
                                             report=lambda **message: report(group=group, validation_year=year, **message))
                    scores['C_nn'] = nn_predict(network, inputs, validation_rows)
                    stop_scores['C_nn'] = nn_predict(network, inputs, stop_rows)
                    if not (np.isfinite(scores['C_nn']).all() and np.isfinite(stop_scores['C_nn']).all()):
                        raise ValueError('ニューラルネットの予測値に欠損があります。')
                    fold_result['variants']['C_nn'] = info
                    # D, E -------------------------------------------------------------
                    stop_structure = _structure(data, stop_rows)
                    for name, (first, second) in (('D_blend', ('B_lgbm_ped', 'C_nn')), ('E_blend', ('A_lgbm', 'C_nn'))):
                        weights = _blend(stop_structure, stop_scores[first], stop_scores[second])
                        scores[name] = weights[0] * scores[first] + weights[1] * scores[second]
                        stop_scores[name] = weights[0] * stop_scores[first] + weights[1] * stop_scores[second]
                        fold_result['variants'][name] = dict(weights=weights, parts=[first, second])
                    # score --------------------------------------------------------------
                    for name in VARIANTS:
                        temperatures, stop_loss, shape = fit_hybrid_temperatures(stop_structure, stop_scores[name])
                        scored = score_races(race_ids[validation_rows], data['positions'][validation_rows],
                                             data['finished'][validation_rows], data['race_numbers'][validation_rows],
                                             scores[name], completion, temperatures, samples=evaluation_samples, seed=seed)
                        fold_result['variants'][name].update(temperatures=temperatures[:6], stop_pl_loss=stop_loss,
                                                             metrics=scored['metrics'])
                        per_race.setdefault(name, []).append(scored)
                        for race, a, b, c in zip(scored['race_ids'], scored['outcome'], scored['top3'], scored['win']):
                            losses.writerow([group, name, year, race, f'{a:.6f}', f'{b:.6f}', '' if np.isnan(c) else f'{c:.6f}'])
                        report(event='scored', group=group, validation_year=year, config=name,
                               race_outcome_log_loss=scored['metrics']['race_outcome_log_loss'],
                               win_log_loss=scored['metrics']['win_log_loss'])
                    group_result['folds'].append(fold_result)
                    del inputs, network, nn_matrix, prepared
            summary = {}
            base = per_race[BASELINE]
            for key, scored_folds in per_race.items():
                row = {metric: float(np.mean([item['metrics'][metric] for item in scored_folds]))
                       for metric in ('race_outcome_log_loss', 'top3_race_log_loss', 'win_log_loss', 'top_pick_top3_rate')}
                if key != BASELINE:
                    combine = lambda field, folds_: np.concatenate([item[field] for item in folds_])
                    row['vs_A'] = dict(
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
            del data, ctx
    finally:
        losses_file.close()
    result['seconds'] = round(time.time() - started, 1)
    (output / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n',
                                            encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='v3実験J: 血統の特徴量とニューラルネットの強さモデルを比べます。')
    parser.add_argument('--dataset', required=True, type=Path, help='prepare済みのfeatures.sqlite3')
    parser.add_argument('--race-db', required=True, type=Path, help='血統を収集した data\\keiba.sqlite3')
    parser.add_argument('--output-dir', required=True, type=Path, help='新しい出力先（既存は不可）')
    parser.add_argument('--validation-years', nargs='+', type=int, default=list(DEFAULT_VALIDATION_YEARS))
    parser.add_argument('--groups', nargs='+', default=list(GROUPS), choices=list(GROUPS))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=8000)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--evaluation-samples', type=int, default=1024)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--nn-epochs', type=int, default=40)
    parser.add_argument('--nn-width', type=int, default=192)
    parser.add_argument('--nn-batch-races', type=int, default=256)
    args = parser.parse_args(argv)
    from .cli import emit
    try:
        import lightgbm  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        emit(event='error', message=f'{exc.name}がありません。scripts\\setup-ml.ps1 と scripts\\setup-nn.ps1 を実行してください。')
        return 2
    from .neural import NeuralConfig
    try:
        result = run_experiment(args.dataset, args.race_db, args.output_dir, validation_years=args.validation_years,
                                groups=args.groups, threads=args.threads, rounds=args.rounds, patience=args.patience,
                                evaluation_samples=args.evaluation_samples, device=args.device,
                                neural=NeuralConfig(max_epochs=args.nn_epochs, width=args.nn_width,
                                                    batch_races=args.nn_batch_races), report=emit)
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
