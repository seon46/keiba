"""Access to the scraper: saved race pages (read-only SQLite) and network commands (subprocess)."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

CENTRAL_COURSES = {f'{index:02d}': name for index, name in enumerate(
    ('札幌', '函館', '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉'), 1)}
LOCAL_COURSES = {
    '30': '門別', '35': '盛岡', '36': '水沢', '42': '浦和', '43': '船橋', '44': '大井', '45': '川崎',
    '46': '金沢', '47': '笠松', '48': '名古屋', '50': '園田', '51': '姫路', '54': '高知', '55': '佐賀',
    '65': '帯広(ば)',
}
# Outcome fields of a results page; removed when a finished race is replayed as an entries card.
_RESULT_ONLY = ('finish_position', 'time_seconds', 'last_3f_seconds', 'margin', 'corner_order', 'odds',
                'popularity', 'prize')


def course_of(region, race_id):
    code = race_id[4:6]
    table = CENTRAL_COURSES if region == 'central' else LOCAL_COURSES
    return table.get(code, code)


def race_number_of(race_id):
    return int(race_id[-2:])


class SavedRaces:
    """Latest saved snapshot of race pages from the scraper database (never writes)."""

    def __init__(self, database):
        self.database = Path(database)

    def _connect(self):
        connection = sqlite3.connect(self.database.resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def snapshot(self, region, race_id, page):
        """Export-style rows (as ``keiba_scraper export``) of the latest snapshot, or None."""
        if not self.database.exists():
            return None
        connection = self._connect()
        try:
            head = connection.execute(
                'SELECT r.snapshot_id,r.metadata_json,o.retrieved_at,o.url FROM races r JOIN observations o '
                'ON o.id=r.snapshot_id WHERE r.region=? AND r.race_id=? AND r.page=? '
                'ORDER BY o.retrieved_at DESC,o.rowid DESC LIMIT 1', (region, race_id, page)).fetchone()
            if head is None:
                return None
            children = connection.execute('SELECT payload_json FROM runners WHERE snapshot_id=? ORDER BY row_index',
                                          (head['snapshot_id'],)).fetchall()
        finally:
            connection.close()
        metadata = json.loads(head['metadata_json'])
        metadata = {key if key in ('race_id', 'region', 'page') or key.startswith('race_') else 'race_' + key: value
                    for key, value in metadata.items()}
        rows = [{**metadata, **json.loads(child['payload_json']), 'source': 'netkeiba', 'source_url': head['url'],
                 'snapshot_id': head['snapshot_id'], 'retrieved_at': head['retrieved_at']} for child in children]
        return rows or None

    def missing_pedigrees(self, horse_ids=None, *, since=None):
        """Horse IDs without a saved pedigree: of ``horse_ids``, or of every runner of races dated ``since`` or
        later (ばんえい excluded)."""
        if not self.database.exists():
            return sorted(set(horse_ids or ()))
        candidates = set()
        connection = self._connect()
        try:
            if horse_ids is None:
                found = connection.execute(
                    "SELECT DISTINCT n.horse_id FROM runners n JOIN races r ON r.snapshot_id=n.snapshot_id "
                    "WHERE n.horse_id IS NOT NULL AND json_extract(r.metadata_json,'$.race_date')>=? "
                    "AND COALESCE(json_extract(r.metadata_json,'$.course'),'') NOT LIKE '帯広%'", (since,))
                candidates = {row[0] for row in found}
            else:
                candidates = {str(horse) for horse in horse_ids if horse}
            known = set()
            ordered = sorted(candidates)
            for start in range(0, len(ordered), 500):
                chunk = ordered[start:start + 500]
                known.update(row[0] for row in connection.execute(
                    f'SELECT DISTINCT horse_id FROM pedigrees WHERE horse_id IN ({",".join("?" * len(chunk))})', chunk))
        except sqlite3.Error:
            return sorted(candidates) if horse_ids is not None else []
        finally:
            connection.close()
        return sorted(candidates - known)

    def odds(self, region, race_id):
        """The latest saved odds of the race ({odds: {bet: {key: value}}, retrieved_at, status, official_at}) or None."""
        if not self.database.exists():
            return None
        connection = self._connect()
        try:
            rows = connection.execute(
                'SELECT d.metadata_json,o.retrieved_at FROM odds d JOIN observations o ON o.id=d.snapshot_id '
                'WHERE d.region=? AND d.race_id=? ORDER BY o.retrieved_at DESC,o.rowid DESC LIMIT 20',
                (region, race_id)).fetchall()
        except sqlite3.Error:
            return None
        finally:
            connection.close()
        if not rows:
            return None
        latest = json.loads(rows[0]['metadata_json'])
        merged = dict(latest, odds={}, retrieved_at=rows[0]['retrieved_at'])
        for row in rows:  # newest first; each bet type from its latest page
            for bet, table in json.loads(row['metadata_json']).get('odds', {}).items():
                merged['odds'].setdefault(bet, table)
        return merged

    def races_on(self, day):
        """{(region, race_id): summary} of every saved race page with this race date."""
        if not self.database.exists():
            return {}
        connection = self._connect()
        try:
            found = connection.execute(
                "SELECT r.region,r.race_id,r.page,r.metadata_json,o.retrieved_at FROM races r JOIN observations o "
                "ON o.id=r.snapshot_id WHERE json_extract(r.metadata_json,'$.race_date')=? "
                "ORDER BY o.retrieved_at", (day,)).fetchall()
        finally:
            connection.close()
        races = {}
        for row in found:
            metadata = json.loads(row['metadata_json'])
            entry = races.setdefault((row['region'], row['race_id']), dict(region=row['region'], race_id=row['race_id'],
                                                                            pages=[]))
            if row['page'] not in entry['pages']:
                entry['pages'].append(row['page'])
            for key in ('race_name', 'start_time', 'course', 'race_number', 'surface', 'distance_m', 'race_type',
                        'declared_field_size', 'track_condition'):
                if metadata.get(key) is not None:
                    entry[key] = metadata[key]
            entry[f'{row["page"]}_retrieved_at'] = row['retrieved_at']
        return races


def entries_from_results(rows):
    """A finished race's results rows turned into an entries card (outcomes removed)."""
    card = []
    for row in rows:
        row = {key: value for key, value in row.items() if key not in _RESULT_ONLY and 'corner' not in key}
        status = row.get('finish_status')
        row['finish_status'] = status if status in ('scratched', 'excluded') else None
        row['page'] = 'entries'
        card.append(row)
    return card


class Scraper:
    """Network commands run as separate processes (so a stuck request never blocks the server)."""

    def __init__(self, data_dir, *, python=None, interval=None):
        self.data_dir = Path(data_dir)
        self.python = python or sys.executable
        self.interval = interval

    def command(self, *arguments, refresh=False, priority=False):
        base = [self.python, '-X', 'utf8', '-m', 'keiba_scraper', '--data-dir', str(self.data_dir)]
        if self.interval:
            base += ['--interval', str(self.interval)]
        if refresh:
            base.append('--refresh')
        if priority:
            base.append('--priority')
        return base + [str(argument) for argument in arguments]

    def run(self, *arguments, refresh=False, log=None, timeout=None, priority=False):
        """Run and return (exit code, parsed JSON lines); a process silent past ``timeout`` seconds is stopped."""
        import threading
        process = subprocess.Popen(self.command(*arguments, refresh=refresh, priority=priority), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        messages = []
        expired = threading.Event()

        def stop():
            expired.set()
            process.kill()
        watchdog = threading.Timer(timeout, stop) if timeout else None
        if watchdog:
            watchdog.daemon = True
            watchdog.start()
        try:
            for line in process.stdout:
                if log is not None:
                    log(line.rstrip('\n'))
                try:
                    messages.append(json.loads(line))
                except ValueError:
                    messages.append(dict(text=line.rstrip('\n')))
            code = process.wait()
        except BaseException:
            process.kill()
            raise
        finally:
            if watchdog:
                watchdog.cancel()
        if expired.is_set():
            raise ValueError(f'netkeibaへの取得が{timeout}秒以内に終わらなかったため中止しました（他の収集と取得間隔を共有しているため、収集中は遅くなります）。')
        return code, messages

    def discover(self, day, region, *, refresh=True):
        code, messages = self.run('list', '--date', day, '--region', region, refresh=refresh, timeout=300)
        listed = [message for message in messages if 'race_ids' in message]
        if code != 0 or not listed:
            detail = ' '.join(message.get('text', '') for message in messages).replace('ERROR: ', '')[-300:]
            raise ValueError(f'{"中央" if region == "central" else "地方"}の一覧を読めませんでした（開催がない日の可能性があります）。{detail}')
        return listed[-1]['race_ids']

    def pedigrees(self, horse_ids, *, chunk=200, log=None, priority=False):
        """Fetch the 5-generation pedigrees of ``horse_ids``; returns how many chunks failed."""
        horse_ids = sorted(set(horse_ids))
        failed = 0
        for start in range(0, len(horse_ids), chunk):
            part = horse_ids[start:start + chunk]
            arguments = ['pedigree']
            for horse in part:
                arguments += ['--horse-id', horse]
            try:
                code, _ = self.run(*arguments, log=log, timeout=300 + 30 * len(part), priority=priority)
            except ValueError:
                code = 1
            failed += code != 0
        return failed

    def odds(self, region, race_id):
        code, messages = self.run('odds', '--region', region, '--race-id', race_id, timeout=180, priority=True)
        if code != 0:
            detail = ' '.join(message.get('text', '') for message in messages).replace('ERROR: ', '')[-300:]
            raise ValueError(f'オッズを取得できません: {detail}')
        return [message for message in messages if 'bets' in message]

    def fetch(self, region, race_ids, page, *, refresh=True, priority=False):
        race_ids = list(race_ids)
        if not race_ids:
            return []
        arguments = ['fetch', '--region', region, '--page', page]
        for race_id in race_ids:
            arguments += ['--race-id', race_id]
        code, messages = self.run(*arguments, refresh=refresh, timeout=120 + 60 * len(race_ids), priority=priority)
        if code != 0:
            detail = ' '.join(message.get('text', '') for message in messages)[-300:]
            raise ValueError(f'出馬表を取得できません: {detail}')
        return messages
