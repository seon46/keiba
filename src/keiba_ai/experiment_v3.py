"""V3 experiment: M0 (current rank model without odds) vs M1 (LightGBM + Plackett-Luce).

Both models use the same rows, the same features (the ``context`` set without
any odds/popularity-derived column) and the same rolling yearly folds.  The
final-test period (2026 onward by default) is never used: a validation year at
or after ``--test-start`` is refused.

M0  sklearn HistGradientBoosting regression of the finishing-rank fraction
    (exactly the current ``train_rank`` recipe), temperature grid (.1,.2,.4).
M1  LightGBM with a custom listwise Plackett-Luce objective on finishers, early
    stopping on the last months of the training period (never the validation
    year), LightGBM completion classifier, temperature grid around 1.

The prepared ``features.sqlite3`` of the current code is reused unchanged.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sqlite3
import time


ODDS_FEATURES = (
    'horse_group_previous_log_odds', 'horse_group_recent5_log_odds',
    'horse_group_previous_popularity_fraction', 'horse_group_recent5_popularity_fraction',
    'horse_group_previous_market_rank_gain', 'horse_group_recent5_market_rank_gain',
    'horse_group_recent5_odds_starts',
)
DEFAULT_GROUPS = ('central_flat', 'local_flat')
DEFAULT_VALIDATION_YEARS = (2023, 2024, 2025)
STAGE_WEIGHTS = {'all': None, 'top5': 5, 'top3': 3}
M0_TEMPERATURES = (.1, .2, .4)
M1_TEMPERATURES = (.8, 1.0, 1.25, 1.6, 2.0)


def experiment_feature_names():
    """The context set minus every odds/popularity-derived column (89 features)."""
    from .features import CONTEXT_FEATURE_NAMES
    names = [name for name in CONTEXT_FEATURE_NAMES if name not in ODDS_FEATURES]
    if len(names) != len(CONTEXT_FEATURE_NAMES) - len(ODDS_FEATURES):
        raise ValueError('オッズ由来の特徴量名が現行コードと一致しません。')
    return names


def _shift_months(day: str, months: int) -> str:
    value = date.fromisoformat(day)
    index = value.year * 12 + value.month - 1 + months
    return date(index // 12, index % 12 + 1, 1).isoformat()


# --------------------------------------------------------------------------
# Plackett-Luce objective
# --------------------------------------------------------------------------

class PLStructure:
    """Padded race x finishing-slot layout for a fixed set of finisher rows.

    ``race_keys`` and ``positions`` are given in the row order LightGBM uses.
    Rows of one race must be contiguous and finishing positions within a race
    must be exactly 1..m (ties/gaps are excluded upstream).
    """

    def __init__(self, race_keys, positions, top_k=None):
        import numpy as np
        race_keys = np.asarray(race_keys)
        positions = np.asarray(positions, dtype=np.int64)
        if len(race_keys) != len(positions) or not len(race_keys):
            raise ValueError('PL構造の行数が不正です。')
        change = np.r_[True, race_keys[1:] != race_keys[:-1]]
        code = np.cumsum(change) - 1
        races = int(code[-1]) + 1
        sizes = np.bincount(code, minlength=races)
        if len(np.unique(race_keys[change])) != races:
            raise ValueError('PL構造: 同じレースの行が連続していません。')
        slot = positions - 1
        if (slot < 0).any() or (slot >= sizes[code]).any():
            raise ValueError('PL構造: 着順が完走頭数の範囲外です。')
        occupied = np.zeros((races, int(sizes.max())), dtype=bool)
        occupied[code, slot] = True
        if occupied.sum() != len(positions) or (occupied.sum(1) != sizes).any():
            raise ValueError('PL構造: 同着または欠番があります。')
        width = int(sizes.max())
        stages = np.arange(width)[None, :]
        # Stage j (0-based) is informative when at least two horses remain.
        informative = stages < (sizes[:, None] - 1)
        if top_k is not None:
            informative &= stages < top_k
        weights = informative.astype(float)
        totals = weights.sum(1, keepdims=True)
        weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
        self.code, self.slot, self.races, self.width = code, slot, races, width
        self.sizes, self.weights = sizes, weights
        self.counted = totals[:, 0] > 0

    def terms(self, scores, *, need_derivatives=True):
        import numpy as np
        scores = np.asarray(scores, dtype=float)
        S = np.full((self.races, self.width), -np.inf)
        S[self.code, self.slot] = scores
        S = S - S.max(axis=1, keepdims=True)
        S = np.maximum(S, -300.0)  # padding stays at -300 and is masked below
        present = np.zeros_like(S, dtype=bool)
        present[self.code, self.slot] = True
        S = np.where(present, S, -np.inf)
        L = np.logaddexp.accumulate(S[:, ::-1], axis=1)[:, ::-1]
        W = self.weights
        active = W > 0
        safe_L = np.where(active, L, 0.0)
        chosen = np.where(active, S - safe_L, 0.0)
        loss = -(W * chosen).sum(axis=1)
        if not need_derivatives:
            return loss
        A = np.where(active, W * np.exp(-safe_L), 0.0)
        B = np.where(active, W * np.exp(-2.0 * safe_L), 0.0)
        C, D = np.cumsum(A, axis=1), np.cumsum(B, axis=1)
        E = np.where(present, np.exp(np.where(present, S, 0.0)), 0.0)
        grad = E * C - W
        hess = np.maximum(E * C - E * E * D, 1e-6)
        return loss, grad[self.code, self.slot], hess[self.code, self.slot]

    def mean_loss(self, scores):
        loss = self.terms(scores, need_derivatives=False)
        return float(loss[self.counted].mean())


def pl_objective(structure):
    def objective(preds, dataset):
        _, grad, hess = structure.terms(preds)
        return grad, hess
    return objective


def pl_metric(preds, dataset):
    return 'pl_nll', dataset.pl_structure.mean_loss(preds), False


# --------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------

def _race_keys(data, rows):
    return [data['identities'][index][0] for index in rows]


def _m0_fit(data, train_mask, columns, categories, leaves, iterations, threads):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    parameters = dict(max_iter=iterations, learning_rate=.07, min_samples_leaf=30,
                      l2_regularization=2.0, early_stopping=False, random_state=42)
    fractions = (data['positions'] - 1) / np.maximum(data['field_sizes'] - 1, 1)
    completed = train_mask & data['finished']
    ranker = HistGradientBoostingRegressor(max_leaf_nodes=leaves, categorical_features=categories, **parameters)
    ranker.fit(data['x'][np.ix_(np.flatnonzero(completed), columns)], fractions[completed])
    constant = float(data['finished'][train_mask].mean())
    finisher = None
    if len(np.unique(data['finished'][train_mask])) == 2:
        finisher = HistGradientBoostingClassifier(max_leaf_nodes=leaves, categorical_features=categories, **parameters)
        finisher.fit(data['x'][np.ix_(np.flatnonzero(train_mask), columns)],
                     data['finished'][train_mask].astype(np.uint8))
    return ranker, finisher, constant


def _lgb_params(threads, seed, **overrides):
    params = dict(learning_rate=.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=.8,
                  lambda_l2=5.0, max_bin=255, num_threads=threads, verbosity=-1, seed=seed,
                  deterministic=True, force_row_wise=True)
    params.update(overrides)
    return params


def _m1_strength(data, fit_mask, stop_mask, columns, categorical, *, top_k, threads, rounds, patience, seed):
    import numpy as np
    import lightgbm as lgb

    def finisher_rows(mask):
        rows = np.flatnonzero(mask & data['finished'])
        keys = np.asarray(_race_keys(data, rows))
        # Drop races with fewer than two finishers: no ordering information.
        change = np.r_[True, keys[1:] != keys[:-1]]
        code = np.cumsum(change) - 1
        counts = np.bincount(code)
        keep = counts[code] >= 2
        return rows[keep], keys[keep]

    fit_rows, fit_keys = finisher_rows(fit_mask)
    stop_rows, stop_keys = finisher_rows(stop_mask)
    if not len(fit_rows) or not len(stop_rows):
        raise ValueError('M1の学習または早期終了用のレースが不足しています。')
    fit_structure = PLStructure(fit_keys, data['positions'][fit_rows], top_k=top_k)
    stop_structure = PLStructure(stop_keys, data['positions'][stop_rows], top_k=top_k)
    train_set = lgb.Dataset(data['x'][np.ix_(fit_rows, columns)], categorical_feature=categorical,
                            free_raw_data=False)
    stop_set = lgb.Dataset(data['x'][np.ix_(stop_rows, columns)], categorical_feature=categorical,
                           reference=train_set, free_raw_data=False)
    train_set.pl_structure, stop_set.pl_structure = fit_structure, stop_structure
    params = _lgb_params(threads, seed, objective=pl_objective(fit_structure), metric='None')
    booster = lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[stop_set], valid_names=['stop'],
                        feval=pl_metric, callbacks=[lgb.early_stopping(patience, verbose=False)])
    best = int(booster.best_iteration or booster.current_iteration())
    return booster, best, float(booster.best_score['stop']['pl_nll'])


def _m1_finisher(data, fit_mask, stop_mask, columns, categorical, *, threads, rounds, patience, seed):
    import numpy as np
    import lightgbm as lgb
    constant = float(data['finished'][fit_mask].mean())
    labels_fit = data['finished'][fit_mask].astype(np.uint8)
    labels_stop = data['finished'][stop_mask].astype(np.uint8)
    if len(np.unique(labels_fit)) < 2 or len(np.unique(labels_stop)) < 2:
        return None, constant, None
    train_set = lgb.Dataset(data['x'][np.ix_(np.flatnonzero(fit_mask), columns)], label=labels_fit,
                            categorical_feature=categorical, free_raw_data=False)
    stop_set = lgb.Dataset(data['x'][np.ix_(np.flatnonzero(stop_mask), columns)], label=labels_stop,
                           categorical_feature=categorical, reference=train_set, free_raw_data=False)
    params = _lgb_params(threads, seed, objective='binary', metric='binary_logloss',
                         num_leaves=31, min_data_in_leaf=200)
    booster = lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[stop_set], valid_names=['stop'],
                        callbacks=[lgb.early_stopping(patience, verbose=False)])
    return booster, constant, int(booster.best_iteration or booster.current_iteration())


def _summary(metrics):
    keep = ('races', 'rows', 'race_outcome_log_loss', 'outcome_log_loss', 'race_multiclass_brier',
            'conditional_expected_rank_mae', 'nonfinish_brier', 'by_race_number')
    result = {key: metrics.get(key) for key in keep if key in metrics}
    top3 = metrics.get('top3') or {}
    result['top3'] = {key: top3.get(key) for key in ('race_log_loss', 'race_brier', 'top_pick_top3_rate', 'precision_at_3')}
    return result


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def run_experiment(database, output, *, groups=DEFAULT_GROUPS, validation_years=DEFAULT_VALIDATION_YEARS,
                   train_start='2022-04-01', test_start='2026-01-01', stop_months=2,
                   stage_weights=('all', 'top5'), threads=2, m0_iterations=150, rounds=4000,
                   patience=100, validation_samples=128, evaluation_samples=1024, seed=42, report=None):
    import numpy as np
    from .features import FEATURE_NAMES, CATEGORICAL_FEATURES
    from .model import feature_revision, readonly
    from .rank_model import estimate_completion, load_rank_group, rank_metrics
    from threadpoolctl import threadpool_limits

    report = report or (lambda **message: None)
    database, output = Path(database), Path(output)
    if date.fromisoformat(train_start).isoformat() != train_start or date.fromisoformat(test_start).isoformat() != test_start:
        raise ValueError('日付はYYYY-MM-DDで指定してください。')
    if any(f'{year:04d}-12-31' >= test_start for year in validation_years):
        raise ValueError('最終評価期間（test-start以降）を検証年に含めることはできません。')
    unknown = [name for name in stage_weights if name not in STAGE_WEIGHTS]
    if unknown:
        raise ValueError(f'未対応の段の重みです: {unknown}（{sorted(STAGE_WEIGHTS)}から指定）')
    connection = readonly(database)
    try:
        metadata = dict(connection.execute('SELECT key,value_json FROM ai_metadata'))
        available = [row[0] for row in connection.execute('SELECT DISTINCT group_name FROM features')]
    except sqlite3.Error as exc:
        raise ValueError(f'prepare済みのfeatures.sqlite3を指定してください: {exc}') from exc
    finally:
        connection.close()
    if json.loads(metadata['feature_revision']) != feature_revision():
        raise ValueError('特徴量コードの版がこのfeatures.sqlite3と一致しません。prepareから作成し直してください。')

    names = experiment_feature_names()
    columns = [FEATURE_NAMES.index(name) for name in names]
    categories = [name in CATEGORICAL_FEATURES for name in names]
    categorical = [index for index, flag in enumerate(categories) if flag]
    folds = []
    for year in validation_years:
        start, end = f'{year:04d}-01-01', f'{year + 1:04d}-01-01'
        stop_start = _shift_months(start, -stop_months)
        if stop_start <= train_start:
            raise ValueError(f'{year}年の検証foldの学習期間がありません。')
        folds.append(dict(validation_year=year, train_start=train_start, stop_start=stop_start,
                          validation_start=start, validation_end=end))
    result = dict(experiment='v3_m0_vs_m1', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  dataset=str(database.resolve()), feature_revision=feature_revision(),
                  feature_names=names, feature_count=len(names),
                  excluded_features=dict(odds=list(ODDS_FEATURES), bias_v1='all 8 bias columns (popularity based)'),
                  folds=folds, test_start=test_start, test_period_used=False,
                  settings=dict(threads=threads, m0_iterations=m0_iterations, m0_temperatures=list(M0_TEMPERATURES),
                                m1_rounds=rounds, m1_patience=patience, m1_temperatures=list(M1_TEMPERATURES),
                                m1_stage_weights=list(stage_weights), m1_params=_lgb_params(threads, seed),
                                validation_samples=validation_samples, evaluation_samples=evaluation_samples,
                                seed=seed, primary_metric='validation race_outcome_log_loss (lower is better)'),
                  groups={})
    for group in groups:
        if group not in available:
            result['groups'][group] = dict(state='skipped', reason='データにこの区分がありません')
            continue
        started = time.time()
        report(event='loading_group', group=group)
        data = load_rank_group(database, group)
        group_result = dict(state='trained', excluded_races=data['excluded_races'], folds=[])
        for fold in folds:
            days = data['days']
            fit_mask = (days >= fold['train_start']) & (days < fold['stop_start'])
            stop_mask = (days >= fold['stop_start']) & (days < fold['validation_start'])
            m0_mask = (days >= fold['train_start']) & (days < fold['validation_start'])
            validation = (days >= fold['validation_start']) & (days < fold['validation_end'])
            sizes = dict(m0_train_rows=int(m0_mask.sum()), m1_fit_rows=int(fit_mask.sum()),
                         m1_stop_rows=int(stop_mask.sum()), validation_rows=int(validation.sum()))
            if min(sizes.values()) < 50 or data['finished'][m0_mask].sum() < 100:
                group_result['folds'].append(dict(fold, state='skipped', reason='期間別データが不足', **sizes))
                continue
            x_validation = data['x'][np.ix_(np.flatnonzero(validation), columns)]
            fold_result = dict(fold, state='trained', **sizes)
            with threadpool_limits(limits=threads):
                # ---------------- M0 ----------------
                m0_candidates = []
                for leaves in (15, 31):
                    report(event='m0_candidate', group=group, validation_year=fold['validation_year'], leaves=leaves)
                    ranker, finisher, constant = _m0_fit(data, m0_mask, columns, categories, leaves, m0_iterations, threads)
                    scores = -ranker.predict(x_validation)
                    completion = estimate_completion(finisher, constant, x_validation)
                    for temperature in M0_TEMPERATURES:
                        metrics = rank_metrics(data, validation, scores, completion, temperature=temperature,
                                               samples=validation_samples, seed=seed)
                        m0_candidates.append(dict(leaves=leaves, temperature=temperature, scores=scores,
                                                  completion=completion,
                                                  race_outcome_log_loss=metrics['race_outcome_log_loss']))
                best = min(m0_candidates, key=lambda item: (item['race_outcome_log_loss'], item['leaves'], item['temperature']))
                m0_metrics = rank_metrics(data, validation, best['scores'], best['completion'], temperature=best['temperature'],
                                          samples=evaluation_samples, seed=seed, stratify=True)
                fold_result['m0'] = dict(selected=dict(leaves=best['leaves'], temperature=best['temperature']),
                                         candidates=[{key: value for key, value in item.items()
                                                      if key not in ('scores', 'completion')} for item in m0_candidates],
                                         validation=_summary(m0_metrics))
                constant = float(data['finished'][m0_mask].mean())
                uniform = rank_metrics(data, validation, np.zeros(len(x_validation)), np.full(len(x_validation), constant),
                                       temperature=1, samples=evaluation_samples, seed=seed, uniform=True)
                fold_result['uniform_prior'] = dict(race_outcome_log_loss=uniform['race_outcome_log_loss'])
                del m0_candidates
                # ---------------- M1 ----------------
                report(event='m1_finisher', group=group, validation_year=fold['validation_year'])
                finisher, constant, finisher_rounds = _m1_finisher(
                    data, fit_mask, stop_mask, columns, categorical, threads=threads, rounds=rounds,
                    patience=patience, seed=seed)
                completion = (np.full(len(x_validation), constant) if finisher is None
                              else finisher.predict(x_validation, num_iteration=finisher_rounds))
                variants = {}
                for name in stage_weights:
                    report(event='m1_strength', group=group, validation_year=fold['validation_year'], stage_weights=name)
                    booster, best_iteration, stop_loss = _m1_strength(
                        data, fit_mask, stop_mask, columns, categorical, top_k=STAGE_WEIGHTS[name],
                        threads=threads, rounds=rounds, patience=patience, seed=seed)
                    scores = booster.predict(x_validation, num_iteration=best_iteration, raw_score=True)
                    candidates = []
                    for temperature in M1_TEMPERATURES:
                        metrics = rank_metrics(data, validation, scores, completion, temperature=temperature,
                                               samples=validation_samples, seed=seed)
                        candidates.append(dict(temperature=temperature,
                                               race_outcome_log_loss=metrics['race_outcome_log_loss']))
                    chosen = min(candidates, key=lambda item: (item['race_outcome_log_loss'], abs(item['temperature'] - 1)))
                    at_edge = chosen['temperature'] in (M1_TEMPERATURES[0], M1_TEMPERATURES[-1])
                    metrics = rank_metrics(data, validation, scores, completion, temperature=chosen['temperature'],
                                           samples=evaluation_samples, seed=seed, stratify=True)
                    importance = booster.feature_importance(importance_type='gain', iteration=best_iteration)
                    order = np.argsort(-importance)[:30]
                    variants[name] = dict(best_iteration=best_iteration, stop_pl_nll=stop_loss,
                                          selected_temperature=chosen['temperature'],
                                          temperature_at_grid_edge=at_edge, candidates=candidates,
                                          validation=_summary(metrics),
                                          top_gain_features=[dict(feature=names[i], gain=float(importance[i]))
                                                             for i in order if importance[i] > 0])
                    report(event='m1_evaluated', group=group, validation_year=fold['validation_year'],
                           stage_weights=name, best_iteration=best_iteration,
                           m1_log_loss=metrics['race_outcome_log_loss'],
                           m0_log_loss=fold_result['m0']['validation']['race_outcome_log_loss'])
                fold_result['m1'] = dict(finisher_best_iteration=finisher_rounds, variants=variants)
            group_result['folds'].append(fold_result)
        trained = [fold for fold in group_result['folds'] if fold.get('state') == 'trained']
        if trained:
            def mean(values):
                return float(np.mean(values))
            summary = dict(folds=len(trained),
                           uniform=mean([fold['uniform_prior']['race_outcome_log_loss'] for fold in trained]),
                           m0=mean([fold['m0']['validation']['race_outcome_log_loss'] for fold in trained]),
                           m1={name: mean([fold['m1']['variants'][name]['validation']['race_outcome_log_loss']
                                           for fold in trained]) for name in stage_weights})
            best_name = min(summary['m1'], key=summary['m1'].get)
            summary.update(best_m1_variant=best_name,
                           m1_minus_m0=summary['m1'][best_name] - summary['m0'],
                           m1_improves_on_every_fold=all(
                               fold['m1']['variants'][best_name]['validation']['race_outcome_log_loss']
                               < fold['m0']['validation']['race_outcome_log_loss'] for fold in trained))
            group_result['summary'] = summary
        else:
            group_result['state'] = 'skipped'
        group_result['seconds'] = round(time.time() - started, 1)
        result['groups'][group] = group_result
        report(event='group_done', group=group, summary=group_result.get('summary'), seconds=group_result['seconds'])
        del data
    output.mkdir(parents=True, exist_ok=False)
    (output / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='v3実験: 現行の順位モデル（オッズ由来除外, M0）とLightGBM+PL（M1）を時系列foldで比較します。')
    parser.add_argument('--dataset', required=True, type=Path, help='prepare済みのfeatures.sqlite3')
    parser.add_argument('--output-dir', required=True, type=Path, help='新しい出力先（既存は不可）')
    parser.add_argument('--groups', nargs='+', default=list(DEFAULT_GROUPS))
    parser.add_argument('--validation-years', nargs='+', type=int, default=list(DEFAULT_VALIDATION_YEARS))
    parser.add_argument('--train-start', default='2022-04-01')
    parser.add_argument('--test-start', default='2026-01-01')
    parser.add_argument('--stage-weights', nargs='+', default=['all', 'top5'], choices=sorted(STAGE_WEIGHTS))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--m0-iterations', type=int, default=150)
    parser.add_argument('--rounds', type=int, default=4000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--validation-samples', type=int, default=128)
    parser.add_argument('--evaluation-samples', type=int, default=1024)
    args = parser.parse_args(argv)
    from .cli import emit
    if args.output_dir.exists():
        emit(event='error', message=f'出力先は既にあります。別の出力先を指定してください: {args.output_dir}')
        return 2
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        emit(event='error', message='lightgbmがありません。scripts\\setup-ml.ps1 を実行してください。')
        return 2
    try:
        result = run_experiment(args.dataset, args.output_dir, groups=args.groups,
                                validation_years=args.validation_years, train_start=args.train_start,
                                test_start=args.test_start, stage_weights=args.stage_weights,
                                threads=args.threads, m0_iterations=args.m0_iterations, rounds=args.rounds,
                                patience=args.patience, validation_samples=args.validation_samples,
                                evaluation_samples=args.evaluation_samples, report=emit)
    except (ValueError, OSError, sqlite3.Error, KeyError) as exc:
        emit(event='error', message=str(exc))
        return 2
    except KeyboardInterrupt:
        emit(event='interrupted', message='停止しました。同じコマンドを別の出力先で再実行してください。')
        return 130
    emit(event='experiment_complete', report=str(args.output_dir / 'experiment.json'),
         summary={group: value.get('summary') for group, value in result['groups'].items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
