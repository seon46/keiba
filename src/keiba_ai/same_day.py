"""Validated live-result input and causal reconstruction of prior-day states.

Race numbers represent scheduled order. The historical result archive has no
completion/publication timestamps and cannot reconstruct delayed publication.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import groupby
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from .bias import BiasPriorState, IntradayBiasState, race_order
from .dataset import ingest_csv, iter_races


def rebuild_prediction_states(database, before_date):
    """Rebuild history and priors using strictly earlier calendar days."""
    from .features import HistoryState
    history, priors, last_date = HistoryState(), BiasPriorState(), None
    for day, iterator in groupby(iter_races(database, before_date=before_date),
                                key=lambda race: race[0]['day']):
        daily = list(iterator)
        intraday = IntradayBiasState(day)
        for race in sorted(daily, key=lambda rows: (
                rows[0]['region'], rows[0]['course'], rows[0].get('surface') or '',
                race_order(rows) or 1000, rows[0]['race_id'])):
            intraday.add_result(race)
        priors.update_day(intraday)
        for race in daily:
            history.update_race(race)
            last_date = day
    return history, priors, last_date


class SameDayResults:
    """One validated temporary snapshot; create a fresh prefix per target."""

    def __init__(self, database, *, source, sha256, quality):
        self.database = Path(database)
        self.source = Path(source)
        self.sha256 = sha256
        self.quality = quality
        self.connection = sqlite3.connect(self.database.resolve().as_uri() + '?mode=ro', uri=True)
        self.connection.execute('PRAGMA query_only=ON')

    def close(self):
        self.connection.close()

    def for_race(self, race):
        """Select same-day/venue/surface results with a lower verified R.

        Other dates, venues, the target itself and later races never enter the
        state. A separate state prevents earlier predictions from seeing a
        later prediction's already-populated state.
        """
        first = race[0]
        state = IntradayBiasState(first['day'])
        target = race_order(race)
        if target is None:
            return state
        cursor = self.connection.execute(
            'SELECT race_id,payload_json FROM input_rows WHERE day=? AND region=? AND group_name=? '
            'ORDER BY race_id,horse_number',
            (first['day'], first['region'], first['group_name']))
        try:
            candidates = []
            for _, iterator in groupby(cursor, key=lambda row: row[0]):
                previous = [json.loads(row[1]) for row in iterator]
                row = previous[0]
                if (row.get('course'), row.get('surface')) != (first.get('course'), first.get('surface')):
                    continue
                number = race_order(previous)
                if number is not None and number < target:
                    candidates.append((number, previous))
            for _, previous in sorted(candidates, key=lambda item: item[0]):
                state.add_result(previous)
        finally:
            cursor.close()
        return state


@contextmanager
def load_same_day_results(source):
    """Stage in a temporary DB; never update the training DB or source CSV."""
    if source is None:
        yield None
        return
    source = Path(source).resolve()
    original = source.stat()
    with TemporaryDirectory(prefix='keiba-same-day-') as directory:
        database = Path(directory)/'results.sqlite3'
        quality = ingest_csv(source, database)
        if quality['excluded_rows'] or not quality['usable_races']:
            raise ValueError('当日結果CSVに不正・不完全なレースがあります。全頭を含む確定結果を指定してください。')
        digest = hashlib.sha256()
        with source.open('rb') as file:
            for block in iter(lambda: file.read(1024*1024), b''):
                digest.update(block)
        current = source.stat()
        if (original.st_size, original.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
            raise ValueError('当日結果CSVが読み込み中に変更されました。出力を終えてから実行してください。')
        results = SameDayResults(database, source=source, sha256=digest.hexdigest(), quality=quality)
        try:
            yield results
        finally:
            results.close()
