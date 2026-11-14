"""Synthetic scraper store + prepared DB + v3 model for keiba_app tests (no network)."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sqlite3

NAMES = ['アオノカゼ', 'ミドリノオカ', 'ハヤテボシ', 'シロガネ', 'クロフネマル', 'ユキアカリ', 'サクラドリーム',
         'ホシノシズク', 'カゼマチ', 'ツキミソウ']


def save_snapshot(store_dir, region, race_id, page, rows, retrieved_at):
    """Insert one race page snapshot directly into a scraper store (schema created by Store)."""
    from keiba_scraper.storage import Store
    Store(store_dir).close()
    connection = sqlite3.connect(Path(store_dir) / 'keiba.sqlite3')
    snapshot = f'{race_id}-{page}-{retrieved_at}'
    first = rows[0]
    metadata = dict(race_id=race_id, region=region, page=page, race_date=first['race_date'],
                    course=first['race_course'], race_number=int(race_id[-2:]), race_name=f'テスト特別{race_id[-2:]}',
                    start_time='15:40', surface=first['race_surface'], distance_m=int(first['race_distance_m']),
                    declared_field_size=int(first['race_declared_field_size']),
                    track_condition=first['race_track_condition'], weather=first['race_weather'])
    connection.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?)',
                       (snapshot, f'https://example.invalid/{race_id}/{page}', retrieved_at, 200, 'utf-8', '0' * 64,
                        'raw/none'))
    connection.execute('INSERT INTO races VALUES (?,?,?,?,?)',
                       (snapshot, race_id, region, page, json.dumps(metadata, ensure_ascii=False)))
    for index, row in enumerate(rows):
        payload = dict(horse_id=row['horse_id'], horse_name=NAMES[(int(row['horse_number']) - 1) % len(NAMES)],
                       horse_number=int(row['horse_number']), frame=int(row['frame']), sex=row['sex'],
                       age=int(row['age']), jockey_id=row['jockey_id'], jockey_name=f'騎手{row["jockey_id"][-2:]}',
                       trainer_id=row['trainer_id'], trainer_name=f'厩舎{row["trainer_id"][-2:]}',
                       carried_weight_kg=float(row['carried_weight_kg']), weight_kg=float(row['weight_kg']),
                       weight_change_kg=float(row['weight_change_kg']))
        if page == 'results':
            if row.get('odds'):
                payload['odds'] = float(row['odds'])
            payload.update(finish_position=int(row['finish_position']), finish_status=row['finish_status'],
                           time_seconds=float(row['time_seconds']))
        connection.execute('INSERT INTO runners VALUES (?,?,?,?)',
                           (snapshot, index, row['horse_id'], json.dumps(payload, ensure_ascii=False)))
    connection.commit()
    connection.close()


class FakeScraper:
    def __init__(self, listing=None):
        self.listing = listing or {}
        self.fetched = []
        self.pedigree_requests = []

    def discover(self, day, region, *, refresh=True):
        if (day, region) not in self.listing and region == 'central' and self.listing:
            raise ValueError('中央の一覧を読めませんでした（開催がない日の可能性があります）。')
        return self.listing.get((day, region), [])

    def fetch(self, region, race_ids, page, *, refresh=True, priority=False):
        self.fetched.append((region, tuple(race_ids), page))
        return []

    def pedigrees(self, horse_ids, *, chunk=200, log=None, priority=False):
        self.pedigree_requests.append(sorted(horse_ids))
        return 0


def build(root, *, train_end='2025-07-01'):
    """Returns dict(data_dir, dataset, model_dir, source_rows, days_after)."""
    from keiba_ai.cli import prepare
    from keiba_ai.pipeline_v3 import fit
    from test_ai_experiment_v3d import jockey_csv
    root = Path(root)
    jockey_csv(root / 'source.csv', years=(2021, 2022, 2023, 2024, 2025))
    with redirect_stdout(io.StringIO()):
        dataset = prepare(root / 'source.csv', root / 'prepared')
    fit(dataset, root / 'model', train_end=train_end, threads=1, rounds=200, patience=20)
    with (root / 'source.csv').open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    days = sorted({row['race_date'] for row in rows if row['race_date'] >= train_end})
    data_dir = root / 'data'
    data_dir.mkdir()
    return dict(data_dir=data_dir, dataset=Path(dataset), model_dir=root / 'model', rows=rows, days_after=days)


def races_on(rows, day):
    races = {}
    for row in rows:
        if row['race_date'] == day:
            races.setdefault((row['region'], row['race_id']), []).append(row)
    return races
