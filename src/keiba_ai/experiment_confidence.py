"""Which race-level "confidence" separates races that go well, and is 単勝 value betting worth it?

Uses the out-of-sample evaluation of a v4 model fitted with ``--evaluate-until`` (the 2025 check fit:
train_end 2025-01-01) and the final win odds stored with the results.  Odds never enter the model;
here they only price the bets.  Final odds are what the pool paid, not what was on the board when a
bet would have been placed, so ROI figures are indicative.

Candidates (computed per race, all available before the start when odds are fetched):
  concentration   1 - entropy(win) / log(starters)          (the app's current confidence)
  top_win         highest win probability
  margin          highest minus second-highest win probability
  disagree_rank   1 - Spearman(LightGBM+血統, NN) inside the race
  disagree_pick   LightGBM+血統 and NN pick different winners (0/1)
  disagree_kl     symmetric KL between the two parts' win distributions
  spread          NN seed spread relative to the score spread in the race
  unknown_share   share of starters without a previous start
  field_size      starters
  market_kl       KL(model win || market win) from the odds
  market_agree    the model's pick is the favourite (0/1)
  edge_max        highest win probability x odds in the race
  learned_*       cross-fitted (by month) models over all of the above
Outcomes: pick wins, pick top 3, race log loss, and 単勝 returns of: the model's pick, the
favourite, and every horse with probability x odds >= 1.0 / 1.2 / 1.5.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

CANDIDATES = ('concentration', 'top_win', 'margin', 'disagree_rank', 'disagree_pick', 'disagree_kl', 'spread',
              'unknown_share', 'field_size', 'market_kl', 'market_agree', 'edge_max')
# The direction in which the candidate is expected to mean "more confident" (the tier table is ordered by it).
CONFIDENT_WHEN_HIGH = dict(concentration=True, top_win=True, margin=True, disagree_rank=False, disagree_pick=False,
                           disagree_kl=False, spread=False, unknown_share=False, field_size=False, market_kl=True,
                           market_agree=True, edge_max=True)
THRESHOLDS = (1.0, 1.2, 1.5)


def _kl(p, q):
    p, q = np.clip(p, 1e-9, 1), np.clip(q, 1e-9, 1)
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 1.0
    return float(spearmanr(a, b).statistic)


def race_table(model_dir, dataset, group, *, samples=2048, seed=42, report=None):
    """One dict per race with the candidates and outcomes."""
    from .experiment_v3j import load_field
    from .features import FEATURE_NAMES
    from .rank_model import load_rank_group, race_seed
    from .ranking_v3 import staged_rank_distribution
    report = report or (lambda **message: None)
    model_dir = Path(model_dir)
    path = model_dir / f'evaluation_{group}.csv'
    temps_path = model_dir / f'evaluation_temperatures_{group}.json'
    if not path.is_file() or not temps_path.is_file():
        raise ValueError(f'{path.name} / {temps_path.name} がありません。最新版のpipeline_v4 fitを --evaluate-until 付きで'
                         '実行したフォルダーを指定してください。')
    temperatures = json.loads(temps_path.read_text(encoding='utf-8'))
    with path.open(encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if 'lightgbm_score' not in rows[0]:
        raise ValueError('評価CSVにlightgbm_score列がありません。最新版のpipeline_v4 fitで作り直してください。')
    data = load_rank_group(dataset, group)
    identities = data['identities']
    index = {(race, horse): position for position, (race, horse, *_rest) in enumerate(identities)}
    odds = load_field(dataset, group, identities, 'odds')
    numbers = load_field(dataset, group, identities, 'horse_number')
    prior = data['x'][:, FEATURE_NAMES.index('horse_prior_starts')]
    races = {}
    for row in rows:
        races.setdefault(row['race_id'], []).append(row)
    table, started = [], time.time()
    for count, (race_id, members) in enumerate(races.items(), 1):
        if count % 2000 == 0:
            report(event='races', group=group, done=count, total=len(races), seconds=round(time.time() - started))
        n = len(members)
        if n < 3:
            continue
        get = lambda key: np.asarray([float(member[key]) for member in members])
        d, b, nn, sd, completion = (get(key) for key in ('strength_score', 'lightgbm_score', 'network_score',
                                                         'network_spread', 'finish_probability'))
        positions = np.asarray([int(member['finish_position']) if member['finish_position'] else 0 for member in members])
        rows_db = [index.get((race_id, member['horse_id'])) for member in members]
        if any(position is None for position in rows_db):
            continue
        race_odds = np.asarray([float(odds[position]) if odds[position] else np.nan for position in rows_db])
        known = np.asarray([prior[position] for position in rows_db], dtype=float)
        race_seed_value = race_seed(race_id, seed)
        dist = staged_rank_distribution(d, completion, temperatures['model'], samples=samples, seed=race_seed_value)
        win = dist[:, 0]
        win_b = staged_rank_distribution(b, completion, temperatures['lightgbm_pedigree'], samples=samples,
                                         seed=race_seed_value)[:, 0]
        win_nn = staged_rank_distribution(nn, completion, temperatures['network'], samples=samples,
                                          seed=race_seed_value)[:, 0]
        outcome = np.where(positions > 0, positions - 1, n)
        log_loss = float(-np.log(np.clip(dist[np.arange(n), np.minimum(outcome, dist.shape[1] - 1)], 1e-12, 1)).mean())
        pick = int(np.argmax(win))
        ordered = np.sort(win)[::-1]
        normalized = win / win.sum()
        entry = dict(group=group, race_id=race_id, race_date=members[0]['race_date'], month=members[0]['race_date'][:7],
                     starters=n, log_loss=log_loss,
                     pick_win=float(positions[pick] == 1), pick_top3=float(0 < positions[pick] <= 3),
                     concentration=float(1 + (normalized * np.log(np.clip(normalized, 1e-12, 1))).sum() / math.log(n)),
                     top_win=float(ordered[0]), margin=float(ordered[0] - ordered[1]),
                     disagree_rank=1 - _spearman(b, nn), disagree_pick=float(np.argmax(b) != np.argmax(nn)),
                     disagree_kl=(_kl(win_b, win_nn) + _kl(win_nn, win_b)) / 2,
                     spread=float(sd.mean() / max(np.std(d), 1e-6)),
                     unknown_share=float(np.mean(~(known > 0))), field_size=float(n))
        priced = np.isfinite(race_odds) & (race_odds > 0)
        if priced.all():
            market = (1 / race_odds) / (1 / race_odds).sum()
            value = win * race_odds
            winner = positions == 1
            entry.update(market_kl=_kl(win, market), market_agree=float(pick == int(np.argmax(market))),
                         edge_max=float(value.max()),
                         pick_cost=1.0, pick_return=float(race_odds[pick] if winner[pick] else 0.0),
                         favourite_cost=1.0,
                         favourite_return=float(race_odds[int(np.argmin(race_odds))] if winner[int(np.argmin(race_odds))]
                                                else 0.0))
            for threshold in THRESHOLDS:
                chosen = value >= threshold
                entry[f'value{threshold}_cost'] = float(chosen.sum())
                entry[f'value{threshold}_return'] = float((race_odds * (chosen & winner)).sum())
        table.append(entry)
    return table


def _learned(table, rng_seed=42):
    """Cross-fitted (leave one month out) scores: probability that the pick finishes top 3, and
    the expected 単勝 value-bet profit of the race."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    features = np.asarray([[entry.get(name, np.nan) for name in CANDIDATES] for entry in table], dtype=float)
    months = np.asarray([entry['month'] for entry in table])
    top3 = np.asarray([entry['pick_top3'] for entry in table])
    profit = np.asarray([entry.get('value1.2_return', np.nan) - entry.get('value1.2_cost', np.nan) for entry in table])
    learned_top3 = np.full(len(table), np.nan)
    learned_value = np.full(len(table), np.nan)
    for month in sorted(set(months.tolist())):
        test, train = months == month, months != month
        classifier = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15,
                                                    l2_regularization=1.0, random_state=rng_seed)
        classifier.fit(features[train], top3[train])
        learned_top3[test] = classifier.predict_proba(features[test])[:, 1]
        usable = train & np.isfinite(profit)
        if usable.sum() > 200:
            regressor = HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_leaf_nodes=15,
                                                      l2_regularization=1.0, random_state=rng_seed)
            regressor.fit(features[usable], profit[usable])
            learned_value[test] = regressor.predict(features[test])
    for entry, a, b in zip(table, learned_top3, learned_value):
        entry['learned_top3'] = float(a)
        entry['learned_value'] = float(b)


def _roi(entries, name, rng, boot=400):
    cost = np.asarray([entry.get(f'{name}_cost', np.nan) for entry in entries])
    ret = np.asarray([entry.get(f'{name}_return', np.nan) for entry in entries])
    keep = np.isfinite(cost) & np.isfinite(ret)
    cost, ret = cost[keep], ret[keep]
    if cost.sum() == 0:
        return dict(bets=0, roi=None, ci=None)
    draws = []
    for _ in range(boot):
        sample = rng.integers(0, len(cost), len(cost))
        if cost[sample].sum() > 0:
            draws.append(ret[sample].sum() / cost[sample].sum())
    return dict(bets=int(cost.sum()), roi=round(float(ret.sum() / cost.sum()), 4),
                ci=[round(float(np.quantile(draws, .025)), 4), round(float(np.quantile(draws, .975)), 4)])


def summarize(table, candidates, *, tiers=3, seed=42):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    bets = ['pick', 'favourite'] + [f'value{threshold}' for threshold in THRESHOLDS]
    overall = dict(races=len(table), pick_win=float(np.mean([e['pick_win'] for e in table])),
                   pick_top3=float(np.mean([e['pick_top3'] for e in table])),
                   log_loss=float(np.mean([e['log_loss'] for e in table])),
                   roi={name: _roi(table, name, rng) for name in bets})
    result = dict(overall=overall, candidates={})
    for name in candidates:
        values = np.asarray([entry.get(name, np.nan) for entry in table], dtype=float)
        keep = np.isfinite(values)
        entries = [entry for entry, ok in zip(table, keep) if ok]
        values = values[keep]
        if len(values) < 3 * tiers:
            continue
        high = CONFIDENT_WHEN_HIGH.get(name, True)
        order = np.argsort(-values if high else values, kind='stable')  # most confident first
        chunks = np.array_split(order, tiers)
        rows = []
        for tier, chunk in zip('ABCDE', chunks):
            part = [entries[i] for i in chunk]
            rows.append(dict(tier=tier, races=len(part), value_range=[round(float(values[chunk].min()), 4),
                                                                    round(float(values[chunk].max()), 4)],
                             pick_win=round(float(np.mean([e['pick_win'] for e in part])), 4),
                             pick_top3=round(float(np.mean([e['pick_top3'] for e in part])), 4),
                             log_loss=round(float(np.mean([e['log_loss'] for e in part])), 4),
                             roi={bet: _roi(part, bet, rng) for bet in bets}))
        sign = 1 if high else -1
        outcome = [e['pick_top3'] for e in entries]
        correlation = (float(sign * spearmanr(values, outcome).statistic)
                       if np.std(values) > 0 and np.std(outcome) > 0 else None)
        result['candidates'][name] = dict(confident_when_high=high,
                                          spearman_pick_top3=None if correlation is None else round(correlation, 4),
                                          tiers=rows)
    return result


def run(model_dir, dataset, output, *, groups=('central_flat', 'local_flat'), samples=2048, report=None):
    report = report or (lambda **message: None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(model_dir=str(Path(model_dir).resolve()), dataset=str(Path(dataset).resolve()), samples=samples,
                   note='単勝の確定オッズでの回収率（roi=払戻/購入額）。発売中のオッズとは異なるので目安。', groups={})
    for group in groups:
        report(event='group', group=group)
        table = race_table(model_dir, dataset, group, samples=samples, report=report)
        _learned(table)
        summary['groups'][group] = summarize(table, CANDIDATES + ('learned_top3', 'learned_value'))
        with (output / f'races_{group}.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=sorted({key for entry in table for key in entry}))
            writer.writeheader()
            writer.writerows(table)
    (output / 'confidence.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return summary


def main(argv=None):
    from .priority import background
    background()
    from .cli import emit
    parser = argparse.ArgumentParser(description='自信度の候補と単勝の回収率の検証（v4の評価期間）')
    parser.add_argument('--model-dir', required=True, type=Path, help='--evaluate-until 付きで学習したv4のフォルダー')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--groups', nargs='+', default=['central_flat', 'local_flat'])
    parser.add_argument('--samples', type=int, default=2048)
    args = parser.parse_args(argv)
    try:
        run(args.model_dir, args.dataset, args.output_dir, groups=args.groups, samples=args.samples, report=emit)
        emit(event='done', output=str(args.output_dir / 'confidence.json'))
        return 0
    except (ValueError, OSError, FileExistsError) as exc:
        emit(event='error', message=str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
