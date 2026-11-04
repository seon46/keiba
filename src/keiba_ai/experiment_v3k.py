"""V3 experiment K: improving the neural strength model and its blend with LightGBM.

The LightGBM parts of experiment J (finisher model, A, B, the neural input
matrix) are computed once per (group, fold) and cached in ``--cache-dir``;
later runs reuse them, so only the networks are trained again.

Neural variants (all PL-trained; each is scored alone and blended with B,
the blend weights and every temperature fitted on the early-stopping period):

``nn_j``        experiment J's setting (40 epochs, early stopping after 4).
``nn_sched``    the one-cycle schedule completes (12 epochs; best epoch kept).
``nn_reg``      nn_sched + stronger regularisation (dropout, weight decay,
                ancestor dropout, rarer entities pooled).
``nn_reg_x5``   nn_reg averaged over 5 seeds.
``nn_search_x5`` per fold: random search (``--search-trials``) on the stop-period
                PL loss, then the best setting averaged over 5 seeds.

``D_<variant>`` = a * B + b * <variant>.  Baselines: A_lgbm and D_nn_j (J's best).
The validation years only score; the final-test period is never read.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from .experiment_v3 import DEFAULT_VALIDATION_YEARS, _shift_months
from .experiment_v3j import GROUPS, _blend, group_context, lgbm_fold
from .neural import NeuralConfig

BASELINE = 'A_lgbm'
REFERENCE = 'D_nn_j'
SEEDS = (42, 7, 1234, 2024, 99)
VARIANT_NAMES = ('nn_j', 'nn_sched', 'nn_reg', 'nn_reg_x5', 'nn_search_x5', 'tabm', 'tabm_noattn', 'tabm_noemb',
                 'tabm_search', 'tabm_search_x3')
# D_multi blends B with every one of these that was trained in the run
MULTI_PARTS = ('tabm_search_x3', 'tabm_search', 'tabm', 'tabm_noattn', 'nn_search_x5', 'nn_reg_x5')


def variant_configs(seed=42):
    j = NeuralConfig(seed=seed)
    sched = replace(j, max_epochs=12, patience=12)
    reg = replace(sched, dropout=.3, weight_decay=.05, ancestor_dropout=.25, entity_min_count=50, width=160)
    # experiment L: TabM (32 members) + periodic numerical embeddings + race attention, and its ablations
    tabm = replace(sched, ensemble=32, numeric_embedding=True, attention=True, batch_races=128, dropout=.1,
                   weight_decay=1e-3, beta2=.95)
    return dict(nn_j=j, nn_sched=sched, nn_reg=reg, tabm=tabm,
                tabm_noattn=replace(tabm, attention=False),
                tabm_noemb=replace(tabm, numeric_embedding=False))


def sample_config(rng, seed=42):
    epochs = int(rng.choice([8, 12, 16, 24]))
    return NeuralConfig(
        width=int(rng.choice([96, 128, 192, 256])), layers=int(rng.choice([1, 2, 3])),
        heads=4, pedigree_dim=int(rng.choice([16, 32, 64])), embedding=int(rng.choice([8, 16, 32])),
        dropout=float(rng.uniform(.1, .4)), ancestor_dropout=float(rng.uniform(0, .3)),
        learning_rate=float(10 ** rng.uniform(-3.5, -2.5)), weight_decay=float(10 ** rng.uniform(-4, -1)),
        batch_races=int(rng.choice([128, 256, 512])), max_epochs=epochs, patience=epochs,
        entity_min_count=int(rng.choice([20, 50, 100])), ancestor_min_count=int(rng.choice([2, 5])), seed=seed)


def sample_tabm_config(rng, seed=42):
    """Random TabM setting (experiment M search space)."""
    epochs = int(rng.choice([8, 12, 16]))
    width = int(rng.choice([128, 192, 256]))
    return NeuralConfig(
        width=width, layers=int(rng.choice([1, 2])), heads=4,
        attention=bool(rng.random() < .5), ensemble=int(rng.choice([16, 32])) if width <= 192 else 16,
        numeric_embedding=bool(rng.random() < .8), numeric_dim=int(rng.choice([8, 16])),
        n_frequencies=int(rng.choice([16, 32, 48])), frequency_scale=float(10 ** rng.uniform(-2.5, -.5)),
        embedding=int(rng.choice([8, 16])), pedigree_dim=int(rng.choice([16, 32])),
        dropout=float(rng.uniform(0, .3)), ancestor_dropout=float(rng.uniform(0, .3)),
        learning_rate=float(10 ** rng.uniform(-3.3, -2.5)), weight_decay=float(10 ** rng.uniform(-4, -1.5)),
        batch_races=int(rng.choice([64, 128])), max_epochs=epochs, patience=epochs,
        entity_min_count=int(rng.choice([20, 50])), ancestor_min_count=int(rng.choice([2, 5])), beta2=.95, seed=seed)


def _blend_many(structure, parts):
    """Non-negative weights of several score vectors minimising the stop-period PL loss."""
    import numpy as np
    from scipy.optimize import minimize
    stacked = np.column_stack(parts)
    start = np.full(stacked.shape[1], 1.0 / stacked.shape[1])
    result = minimize(lambda weights: structure.mean_loss(stacked @ weights), start, method='L-BFGS-B',
                      bounds=[(0, 3)] * stacked.shape[1])
    return [float(value) for value in result.x]


# --------------------------------------------------------------------------
# Cache of the LightGBM parts
# --------------------------------------------------------------------------

_ARRAYS = ('nn_matrix', 'categorical', 'race_ids', 'finished', 'positions', 'race_numbers', 'jockeys', 'trainers',
           'horses', 'fit_rows', 'stop_all_rows', 'stop_rows', 'validation_rows', 'completion',
           'score_A_lgbm', 'score_B_lgbm_ped', 'stop_A_lgbm', 'stop_B_lgbm_ped')


def _fingerprint(database, fold, settings):
    from .model import feature_revision
    path = Path(database).resolve()
    stat = path.stat()
    text = json.dumps(dict(dataset=str(path), size=stat.st_size, mtime=stat.st_mtime_ns, fold=fold,
                           revision=feature_revision(), settings=settings), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def load_or_build(cache_dir, database, pedigrees, group, fold, ctx_holder, *, settings, report):
    """Cached fold arrays (dict); builds them (and the group context, once) when missing."""
    import numpy as np
    from .experiment_v3d import _params
    cache_dir = Path(cache_dir)
    key = _fingerprint(database, fold, settings)
    path = cache_dir / f'{group}_{fold["validation_year"]}_{key}.npz'
    if path.is_file():
        report(event='cache_hit', group=group, validation_year=fold['validation_year'])
        with np.load(path, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}, True
    if ctx_holder.get('group') != group:
        ctx_holder.clear()
        ctx_holder.update(group_context(database, group, pedigrees, announce_dropout=settings['announce_dropout'],
                                        crossfit_blocks=settings['crossfit_blocks'], seed=settings['seed'],
                                        report=report))
    ctx = ctx_holder
    params = _params(settings['threads'], settings['seed'])
    prepared = lgbm_fold(ctx, fold, pedigrees, params=params, threads=settings['threads'], rounds=settings['rounds'],
                         patience=settings['patience'], crossfit_blocks=settings['crossfit_blocks'],
                         seed=settings['seed'], report=report)
    data = ctx['data']
    arrays = dict(
        nn_matrix=prepared['nn_matrix'].astype(np.float32), categorical=np.asarray(prepared['categorical'], dtype=np.int64),
        race_ids=ctx['race_ids'].astype(str), finished=data['finished'], positions=data['positions'],
        race_numbers=data['race_numbers'], jockeys=ctx['jockeys'].astype(str), trainers=ctx['trainers'].astype(str),
        horses=ctx['horses'].astype(str), fit_rows=np.flatnonzero(prepared['fit_mask']),
        stop_all_rows=np.flatnonzero(prepared['stop_mask']), stop_rows=prepared['stop_rows'],
        validation_rows=prepared['validation_rows'], completion=np.asarray(prepared['completion'], dtype=np.float64))
    for name in ('A_lgbm', 'B_lgbm_ped'):
        arrays[f'score_{name}'] = np.asarray(prepared['scores'][name], dtype=np.float64)
        arrays[f'stop_{name}'] = np.asarray(prepared['stop_scores'][name], dtype=np.float64)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)
    return arrays, False


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def run_experiment(database, race_db, output, cache_dir, *, validation_years=DEFAULT_VALIDATION_YEARS, groups=GROUPS,
                   train_start='2022-04-01', test_start='2026-01-01', stop_months=2, threads=2, rounds=8000,
                   patience=150, announce_dropout=.05, crossfit_blocks=3, evaluation_samples=1024, seed=42,
                   device='auto', search_trials=16, seeds=SEEDS, variants=None, trial_minutes=15.0, resume=False,
                   report=None):
    import numpy as np
    from .experiment_v3 import PLStructure
    from .experiment_v3i import fit_hybrid_temperatures
    from .model import feature_revision, readonly
    from .neural import Inputs, TrialTooSlow, device_name, predict as nn_predict, train as nn_train
    from .pedigree import load_pedigrees
    from .ranking_v3 import paired_bootstrap, score_races

    report = report or (lambda **message: None)
    database, output = Path(database), Path(output)
    device = device_name(device)
    if output.exists() and not resume:
        raise ValueError(f'出力先は既にあります。続きから再開するには --resume を付けてください: {output}')
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
    settings = dict(threads=threads, rounds=rounds, patience=patience, announce_dropout=announce_dropout,
                    crossfit_blocks=crossfit_blocks, seed=seed)
    report(event='loading_pedigrees')
    pedigrees = load_pedigrees(race_db)
    folds = []
    for year in validation_years:
        start = f'{year:04d}-01-01'
        folds.append(dict(validation_year=year, train_start=train_start, stop_start=_shift_months(start, -stop_months),
                          validation_start=start, validation_end=f'{year + 1:04d}-01-01'))
    fixed = variant_configs(seed)
    wanted = list(variants) if variants else ['nn_j', 'nn_sched', 'nn_reg', 'nn_reg_x5'] + (
        ['nn_search_x5'] if search_trials > 0 else [])
    started = time.time()
    result = dict(experiment='v3k_neural_improvement', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  dataset=str(database.resolve()), feature_revision=feature_revision(), folds=folds,
                  test_start=test_start, test_period_used=False, baseline=BASELINE, reference=REFERENCE, device=device,
                  variants=wanted, settings=dict(settings, evaluation_samples=evaluation_samples, seeds=list(seeds),
                                                 search_trials=search_trials,
                                                 fixed={name: asdict(config) for name, config in fixed.items()}),
                  groups={})
    output.mkdir(parents=True, exist_ok=True)
    partial = output / 'partial'
    partial.mkdir(exist_ok=True)
    new_losses = not (output / 'race_losses.csv').exists()
    losses_file = (output / 'race_losses.csv').open('a', encoding='utf-8', newline='')
    losses = csv.writer(losses_file)
    if new_losses:
        losses.writerow(['group', 'config', 'validation_year', 'race_id', 'outcome_log_loss', 'top3_log_loss',
                         'win_log_loss'])
    variants_key = hashlib.sha256(json.dumps([wanted, search_trials, list(seeds)]).encode()).hexdigest()[:10]
    ctx_holder = {}
    try:
        for group in groups:
            group_result = dict(folds=[])
            per_race = {}
            for fold in folds:
                year = fold['validation_year']
                saved = partial / f'{group}_{year}_{variants_key}.json'
                if saved.is_file():
                    stored = json.loads(saved.read_text(encoding='utf-8'))
                    for name, scored in stored['scored'].items():
                        per_race.setdefault(name, []).append(dict(
                            scored, outcome=np.asarray(scored['outcome']), top3=np.asarray(scored['top3']),
                            win=np.asarray([np.nan if value is None else value for value in scored['win']])))
                    group_result['folds'].append(stored['fold_result'])
                    report(event='fold_resumed', group=group, validation_year=year)
                    continue
                fold_scored = {}
                arrays, cached = load_or_build(cache_dir, database, pedigrees, group, fold, ctx_holder, settings=settings,
                                               report=report)
                race_ids = arrays['race_ids'].astype(object)
                horses = arrays['horses'].astype(object)
                jockeys, trainers = arrays['jockeys'].astype(object), arrays['trainers'].astype(object)
                finished, positions = arrays['finished'], arrays['positions']
                fit_rows, stop_all, stop_rows = arrays['fit_rows'], arrays['stop_all_rows'], arrays['stop_rows']
                validation_rows = arrays['validation_rows']
                stop_structure = PLStructure(race_ids[stop_rows], positions[stop_rows])
                scores = {name: arrays[f'score_{name}'] for name in ('A_lgbm', 'B_lgbm_ped')}
                stop_scores = {name: arrays[f'stop_{name}'] for name in ('A_lgbm', 'B_lgbm_ped')}
                fold_result = dict(validation_year=year, cached=cached, variants={})
                inputs_cache = {}

                def inputs_for(config):
                    key = (config.entity_min_count, config.ancestor_min_count)
                    if key not in inputs_cache:
                        inputs_cache.clear()
                        inputs_cache[key] = Inputs(arrays['nn_matrix'], arrays['categorical'].tolist(), race_ids, finished,
                                                   positions, jockeys, trainers, pedigrees, horses, fit_rows, config)
                    return inputs_cache[key]

                def run(config, label, time_limit=None):
                    inputs = inputs_for(config)
                    network, info = nn_train(inputs, fit_rows, stop_all, config, device=device, time_limit=time_limit,
                                             report=lambda **message: report(group=group, validation_year=year,
                                                                             config=label, **message))
                    validation_scores = nn_predict(network, inputs, validation_rows)
                    stop_values = nn_predict(network, inputs, stop_rows)
                    if not (np.isfinite(validation_scores).all() and np.isfinite(stop_values).all()):
                        raise ValueError('ニューラルネットの予測値に欠損があります。')
                    del network
                    return validation_scores, stop_values, info

                def ensemble(config, label):
                    parts = [run(replace(config, seed=member), f'{label}/seed{member}') for member in seeds]
                    return (np.mean([part[0] for part in parts], axis=0), np.mean([part[1] for part in parts], axis=0),
                            dict(seeds=list(seeds), best_epochs=[part[2]['best_epoch'] for part in parts],
                                 stop_losses=[part[2]['stop_loss'] for part in parts], config=asdict(config)))

                searched = {}

                def search(sampler, label):
                    """Random search on the stop-period PL loss (a setting that runs out of GPU memory is skipped)."""
                    if label in searched:
                        return searched[label]
                    import torch
                    generator = np.random.default_rng(seed + year + (1000 if label == 'tabm' else 0))
                    trials = []
                    for trial in range(search_trials):
                        config = sampler(generator, seed)
                        try:
                            _, _, info = run(config, f'{label}_search{trial}', time_limit=trial_minutes * 60)
                            loss = info['stop_loss']
                        except (torch.cuda.OutOfMemoryError, TrialTooSlow) as exc:
                            torch.cuda.empty_cache()
                            report(event='search_trial_skipped', group=group, validation_year=year, trial=trial,
                                   reason='GPUのメモリ不足' if isinstance(exc, torch.cuda.OutOfMemoryError) else str(exc))
                            loss = float('inf')
                        trials.append((loss, config))
                        report(event='search_trial', group=group, validation_year=year, search=label, trial=trial,
                               stop_loss=None if loss == float('inf') else round(loss, 5))
                    finite = [item for item in trials if item[0] != float('inf')]
                    if not finite:
                        raise ValueError('設定探索がすべてGPUのメモリ不足で失敗しました。')
                    searched[label] = (min(finite, key=lambda item: item[0]), trials)
                    return searched[label]

                for name in wanted:
                    report(event='variant', group=group, validation_year=year, variant=name)
                    if name in fixed:
                        values, stop_values, info = run(fixed[name], name)
                    elif name == 'nn_reg_x5':
                        values, stop_values, info = ensemble(fixed['nn_reg'], name)
                    elif name == 'nn_search_x5':
                        generator = np.random.default_rng(seed + year)
                        trials = []
                        for trial in range(search_trials):
                            config = sample_config(generator, seed)
                            _, _, info = run(config, f'search{trial}')
                            trials.append((info['stop_loss'], config))
                            report(event='search_trial', group=group, validation_year=year, trial=trial,
                                   stop_loss=round(info['stop_loss'], 5))
                        best_loss, best_config = min(trials, key=lambda item: item[0])
                        values, stop_values, info = ensemble(best_config, name)
                        info.update(search_best_stop_loss=best_loss,
                                    search=[dict(stop_loss=loss, config=asdict(config)) for loss, config in trials])
                    elif name in ('tabm_search', 'tabm_search_x3'):
                        (best_loss, best_config), trials = search(sample_tabm_config, 'tabm')
                        if name == 'tabm_search':
                            values, stop_values, info = run(best_config, name)
                        else:
                            members = seeds[:3]
                            parts = [run(replace(best_config, seed=member), f'{name}/seed{member}') for member in members]
                            values = np.mean([part[0] for part in parts], axis=0)
                            stop_values = np.mean([part[1] for part in parts], axis=0)
                            info = dict(seeds=list(members), best_epochs=[part[2]['best_epoch'] for part in parts],
                                        stop_losses=[part[2]['stop_loss'] for part in parts], config=asdict(best_config))
                        info.update(search_best_stop_loss=best_loss,
                                    search=[dict(stop_loss=None if loss == float('inf') else loss, config=asdict(config))
                                            for loss, config in trials])
                    else:
                        raise ValueError(f'未対応の設定です: {name}')
                    scores[name], stop_scores[name] = values, stop_values
                    fold_result['variants'][name] = info
                    weights = _blend(stop_structure, stop_scores['B_lgbm_ped'], stop_values)
                    scores[f'D_{name}'] = weights[0] * scores['B_lgbm_ped'] + weights[1] * values
                    stop_scores[f'D_{name}'] = weights[0] * stop_scores['B_lgbm_ped'] + weights[1] * stop_values
                    fold_result['variants'][f'D_{name}'] = dict(weights=weights)
                members = [name for name in MULTI_PARTS if name in scores]
                if len(members) >= 2:
                    parts = ['B_lgbm_ped'] + members
                    weights = _blend_many(stop_structure, [stop_scores[part] for part in parts])
                    scores['D_multi'] = sum(weight * scores[part] for weight, part in zip(weights, parts))
                    stop_scores['D_multi'] = sum(weight * stop_scores[part] for weight, part in zip(weights, parts))
                    fold_result['variants']['D_multi'] = dict(weights=dict(zip(parts, weights)))
                for name in list(scores):
                    temperatures, stop_loss, _ = fit_hybrid_temperatures(stop_structure, stop_scores[name])
                    scored = score_races(race_ids[validation_rows], positions[validation_rows], finished[validation_rows],
                                         arrays['race_numbers'][validation_rows], scores[name], arrays['completion'],
                                         temperatures, samples=evaluation_samples, seed=seed)
                    entry = fold_result['variants'].setdefault(name, {})
                    entry.update(temperatures=temperatures[:6], stop_pl_loss=stop_loss, metrics=scored['metrics'])
                    per_race.setdefault(name, []).append(scored)
                    fold_scored[name] = dict(race_ids=list(scored['race_ids']), metrics=scored['metrics'],
                                             outcome=scored['outcome'].tolist(), top3=scored['top3'].tolist(),
                                             win=[None if np.isnan(value) else float(value) for value in scored['win']])
                    for race, a, b, c in zip(scored['race_ids'], scored['outcome'], scored['top3'], scored['win']):
                        losses.writerow([group, name, year, race, f'{a:.6f}', f'{b:.6f}', '' if np.isnan(c) else f'{c:.6f}'])
                    report(event='scored', group=group, validation_year=year, config=name,
                           race_outcome_log_loss=scored['metrics']['race_outcome_log_loss'])
                group_result['folds'].append(fold_result)
                temporary = saved.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(fold_result=fold_result, scored=fold_scored), ensure_ascii=False,
                                                default=str), encoding='utf-8')
                temporary.replace(saved)
                del arrays
            summary = {}
            combine = lambda field, folds_: np.concatenate([item[field] for item in folds_])
            for key, scored_folds in per_race.items():
                row = {metric: float(np.mean([item['metrics'][metric] for item in scored_folds]))
                       for metric in ('race_outcome_log_loss', 'top3_race_log_loss', 'win_log_loss', 'top_pick_top3_rate')}
                for reference in (BASELINE, REFERENCE):
                    if key != reference and reference in per_race:
                        base = per_race[reference]
                        row[f'vs_{reference}'] = dict(
                            outcome=paired_bootstrap(combine('outcome', scored_folds) - combine('outcome', base), seed=seed),
                            win=paired_bootstrap(combine('win', scored_folds) - combine('win', base), seed=seed),
                            every_fold_better=all(item['metrics']['race_outcome_log_loss'] < ref['metrics']['race_outcome_log_loss']
                                                  for item, ref in zip(scored_folds, base)))
                summary[key] = row
            group_result['summary'] = summary
            group_result['best_config'] = min(summary, key=lambda key: summary[key]['race_outcome_log_loss'])
            result['groups'][group] = group_result
            report(event='group_done', group=group, best=group_result['best_config'],
                   summary={key: round(row['race_outcome_log_loss'], 5) for key, row in summary.items()})
            ctx_holder.clear()
    finally:
        losses_file.close()
    result['seconds'] = round(time.time() - started, 1)
    (output / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n',
                                            encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='v3実験K: ニューラルネットの改善とLightGBMとの混ぜ合わせ。')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--race-db', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path, help='LightGBM部分の保存先（2回目以降は再利用）')
    parser.add_argument('--validation-years', nargs='+', type=int, default=list(DEFAULT_VALIDATION_YEARS))
    parser.add_argument('--groups', nargs='+', default=list(GROUPS), choices=list(GROUPS))
    parser.add_argument('--variants', nargs='+', choices=VARIANT_NAMES)
    parser.add_argument('--search-trials', type=int, default=16)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS), help='平均する乱数の種（既定5個）')
    parser.add_argument('--trial-minutes', type=float, default=15.0, help='設定探索の1回あたりの時間の上限（分）')
    parser.add_argument('--resume', action='store_true', help='同じ出力先で、終わった年を飛ばして続きから再開')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=8000)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--evaluation-samples', type=int, default=1024)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    args = parser.parse_args(argv)
    from .cli import emit
    try:
        import lightgbm  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        emit(event='error', message=f'{exc.name}がありません。scripts\\setup-ml.ps1 と scripts\\setup-nn.ps1 を実行してください。')
        return 2
    try:
        result = run_experiment(args.dataset, args.race_db, args.output_dir, args.cache_dir,
                                validation_years=args.validation_years, groups=args.groups, threads=args.threads,
                                rounds=args.rounds, patience=args.patience, evaluation_samples=args.evaluation_samples,
                                device=args.device, search_trials=args.search_trials, variants=args.variants, seeds=tuple(args.seeds),
                                trial_minutes=args.trial_minutes, resume=args.resume,
                                report=emit)
    except (ValueError, OSError, sqlite3.Error, KeyError) as exc:
        emit(event='error', message=str(exc))
        return 2
    except KeyboardInterrupt:
        emit(event='interrupted', message='停止しました。別の出力先で再実行してください（LightGBM部分の保存は残ります）。')
        return 130
    emit(event='experiment_complete', report=str(args.output_dir / 'experiment.json'),
         best={group: value['best_config'] for group, value in result['groups'].items()}, seconds=result['seconds'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
