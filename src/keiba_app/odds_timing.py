"""How much the win odds are worth at different times before the start, from the app's own records.

Every time odds are fetched in the app the snapshot is saved with its time.  For each finished race with a
logged prediction (the last one made before the start) and a clean 1st-2nd-3rd, every saved win-odds snapshot
is put in a bucket by how long before the start it was taken, and the final odds of the results page form a
last bucket.  In each bucket the order likelihood of keiba_ai.experiment_market is compared for: the model
alone, the market alone, the app's current blend weights (keiba_ai.bets.MARKET_BLEND, fitted on final odds),
and weights fitted on that bucket.  When early odds are worth less, the fitted market weight is smaller there
and the app should trust them less.  Few races per bucket make the fitted weights noisy: ``races`` says how
many there were.

    python -X utf8 -m keiba_app.odds_timing --data-dir data --output data\\odds_timing.json
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

BUCKETS = (('12時間以上前', 720, None), ('3〜12時間前', 180, 720), ('1〜3時間前', 60, 180), ('15〜60分前', 15, 60),
           ('15分以内', None, 15))
FINAL = '確定オッズ'
MIN_RACES = 30


def _bucket(minutes):
    for name, low, high in BUCKETS:
        if (low is None or minutes >= low) and (high is None or minutes < high):
            return name
    return None


def collect(app):
    """{bucket: [race sample]}; a sample is (scores, odds, place indices, temperatures, region)."""
    from keiba_ai.bets import _odds_value
    from .engine import race_start
    from .logbook import Logbook
    from .sources import SavedRaces
    saved = SavedRaces(app.scraper_db)
    logbook = Logbook(app.logbook_path)
    if not app.scraper_db.exists():
        raise ValueError(f'スクレイパーのデータベースがありません: {app.scraper_db}')
    connection = sqlite3.connect(f'file:{app.scraper_db}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        snapshots = {}
        for row in connection.execute('SELECT d.region,d.race_id,d.metadata_json,o.retrieved_at FROM odds d '
                                      'JOIN observations o ON o.id=d.snapshot_id ORDER BY o.retrieved_at'):
            win = (json.loads(row['metadata_json']).get('odds') or {}).get('win')
            if win:
                snapshots.setdefault((row['region'], row['race_id']), []).append((row['retrieved_at'], win))
    finally:
        connection.close()
    with logbook._connect() as log:
        predictions = [dict(row) for row in log.execute(
            "SELECT id,region,race_id,race_date,created_at,temperatures_json FROM predictions "
            "WHERE temperatures_json IS NOT NULL ORDER BY created_at")]
    by_race = {}
    for prediction in predictions:
        by_race.setdefault((prediction['region'], prediction['race_id']), []).append(prediction)
    samples = {name: [] for name, _, _ in BUCKETS}
    samples[FINAL] = []
    for key, candidates in by_race.items():
        results = saved.snapshot(*key, 'results')
        entries = saved.snapshot(*key, 'entries') or results
        if not results or not entries:
            continue
        start = race_start(candidates[0]['race_date'], entries[0].get('race_start_time'))
        if start is None:
            continue
        before = [p for p in candidates if datetime.fromisoformat(p['created_at']) < start]
        if not before:
            continue
        prediction = logbook.prediction(before[-1]['id'])
        runners = [runner for runner in prediction['runners'] if runner.get('strength_score') is not None]
        finish = {str(row.get('horse_number')): row for row in results}
        numbers = [str(runner['horse_number']) for runner in runners
                   if (finish.get(str(runner['horse_number'])) or {}).get('finish_status') not in ('scratched', 'excluded')]
        positions = {number: int(finish[number]['finish_position']) for number in numbers
                     if str((finish.get(number) or {}).get('finish_position') or '').isdigit()}
        order = sorted((position, number) for number, position in positions.items() if position <= 3)
        if [position for position, _ in order] != [1, 2, 3] or len(numbers) < 3:
            continue
        scores = {str(runner['horse_number']): float(runner['strength_score']) for runner in runners}
        temperatures = json.loads(prediction['temperatures_json'])

        def add(bucket, table):
            values = [_odds_value(table.get(number)) for number in numbers]
            if any(value is None or value <= 1 for value in values):
                return
            samples[bucket].append(dict(scores=[scores[number] for number in numbers], odds=values,
                                        place=[numbers.index(number) for _, number in order],
                                        temperatures=temperatures, region=key[0]))
        add(FINAL, {number: (finish[number] or {}).get('odds') for number in numbers})
        latest = {}
        for retrieved_at, table in snapshots.get(key, ()):
            minutes = (start - datetime.fromisoformat(retrieved_at)).total_seconds() / 60
            bucket = _bucket(minutes) if minutes >= 0 else None
            if bucket:
                latest[bucket] = table  # the latest snapshot within each bucket
        for bucket, table in latest.items():
            add(bucket, table)
    return samples


def _arrays(samples):
    """experiment_market data from samples sharing one region (temperatures per race are averaged)."""
    from keiba_ai.experiment_market import with_terms
    width = max(len(sample['scores']) for sample in samples)
    shape = (len(samples), width)
    data = dict(present=np.zeros(shape, dtype=bool), model=np.zeros(shape), market=np.zeros(shape),
                odds=np.ones(shape), place=np.zeros((len(samples), 3), dtype=int),
                temperatures=list(np.mean([(s['temperatures'] + s['temperatures'][-1:] * 3)[:3] for s in samples],
                                          axis=0)))
    for index, sample in enumerate(samples):
        n = len(sample['scores'])
        inverse = 1 / np.asarray(sample['odds'])
        data['present'][index, :n] = True
        data['model'][index, :n] = sample['scores']
        data['market'][index, :n] = np.log(inverse / inverse.sum())
        data['odds'][index, :n] = sample['odds']
        data['place'][index] = sample['place']
    return with_terms(data)


def analyse(samples):
    from keiba_ai.bets import MARKET_BLEND
    from keiba_ai.experiment_market import TERMS, fit, order_log_likelihood

    def weights(a, b):
        row = np.zeros(len(TERMS))
        row[TERMS.index('s')], row[TERMS.index('m')] = a, b
        return np.tile(row, (3, 1))
    report = {}
    for region in ('central', 'local'):
        rows = {}
        for bucket in [name for name, _, _ in BUCKETS] + [FINAL]:
            chosen = [sample for sample in samples[bucket] if sample['region'] == region]
            entry = dict(races=len(chosen))
            if len(chosen) >= MIN_RACES:
                data = _arrays(chosen)
                current = MARKET_BLEND[region]
                params, _, fitted = fit(data, 'blend')
                entry.update(model_only=order_log_likelihood(data, weights(1.0, 0.0)),
                             market_only=fit(dict(data, terms=data['terms'] * np.isin(
                                 np.arange(len(TERMS)), [TERMS.index('m')])), 'blend')[2],
                             current_blend=order_log_likelihood(data, weights(current['a'], current['b'])),
                             fitted_blend=fitted, fitted_weights=dict(a=params['all']['s'], b=params['all']['m']))
            rows[bucket] = entry
        report[region] = rows
    return report


def main(argv=None):
    from keiba_ai.cli import emit
    from .state import App
    parser = argparse.ArgumentParser(description='オッズの取得時刻ごとに、オッズの当てになり方を比べる（アプリの記録から）')
    parser.add_argument('--data-dir', default='data', type=Path)
    parser.add_argument('--home', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        app = App(args.data_dir, home=args.home)
        report = dict(min_races=MIN_RACES, regions=analyse(collect(app)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        for region, rows in report['regions'].items():
            emit(event='buckets', region=region, races={bucket: entry['races'] for bucket, entry in rows.items()})
        emit(event='done', output=str(args.output))
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        emit(event='error', message=str(exc))
        return 2


if __name__ == '__main__':
    sys.exit(main())
