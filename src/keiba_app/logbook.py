"""Prediction log (app.sqlite3): every prediction, its runners, and the scored outcome."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL, race_id TEXT NOT NULL, region TEXT NOT NULL, race_date TEXT NOT NULL,
    course TEXT, race_number INTEGER, race_name TEXT, start_time TEXT, surface TEXT, distance_m INTEGER,
    track_condition TEXT, starters INTEGER NOT NULL, timing TEXT NOT NULL, source TEXT NOT NULL,
    origin TEXT NOT NULL, model_dir TEXT, model_train_end TEXT, dataset TEXT, history_through TEXT,
    bodyweight_known INTEGER, entries_retrieved_at TEXT
);
CREATE INDEX IF NOT EXISTS predictions_race ON predictions(race_id, created_at);
CREATE INDEX IF NOT EXISTS predictions_day ON predictions(race_date);
CREATE TABLE IF NOT EXISTS prediction_runners (
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    horse_id TEXT NOT NULL, horse_number INTEGER NOT NULL, frame INTEGER, horse_name TEXT, sex TEXT, age INTEGER,
    carried_weight_kg REAL, jockey_name TEXT, trainer_name TEXT, weight_kg REAL, weight_change_kg REAL,
    status TEXT, win_probability REAL, top3_probability REAL, expected_rank REAL, strength_score REAL,
    finish_probability REAL, distribution_json TEXT,
    PRIMARY KEY (prediction_id, horse_id)
);
CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    scored_at TEXT NOT NULL, status TEXT NOT NULL, actual_starters INTEGER,
    outcome_log_loss REAL, uniform_log_loss REAL, top3_log_loss REAL, win_log_loss REAL,
    pick_horse_number INTEGER, pick_position INTEGER, pick_top3 INTEGER,
    favourite_horse_number INTEGER, favourite_won INTEGER, results_json TEXT
);
CREATE TABLE IF NOT EXISTS day_races (
    race_date TEXT NOT NULL, region TEXT NOT NULL, race_id TEXT NOT NULL, course TEXT, race_number INTEGER,
    discovered_at TEXT NOT NULL, PRIMARY KEY (race_date, region, race_id)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT, kind TEXT NOT NULL,
    status TEXT NOT NULL, steps_json TEXT NOT NULL DEFAULT '[]', log_path TEXT
);
"""

RUNNER_FIELDS = ('horse_id', 'horse_number', 'frame', 'horse_name', 'sex', 'age', 'carried_weight_kg', 'jockey_name',
                 'trainer_name', 'weight_kg', 'weight_change_kg', 'status', 'win_probability', 'top3_probability',
                 'expected_rank', 'strength_score', 'finish_probability', 'prior_starts')
PREDICTION_FIELDS = ('created_at', 'race_id', 'region', 'race_date', 'course', 'race_number', 'race_name',
                     'start_time', 'surface', 'distance_m', 'track_condition', 'starters', 'timing', 'source',
                     'origin', 'model_dir', 'model_train_end', 'dataset', 'history_through', 'bodyweight_known',
                     'entries_retrieved_at', 'temperatures_json')


class Logbook:
    def __init__(self, path):
        self.path = Path(path)
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            columns = {row['name'] for row in connection.execute('PRAGMA table_info(predictions)')}
            if 'temperatures_json' not in columns:  # added with the bet proposals
                connection.execute('ALTER TABLE predictions ADD COLUMN temperatures_json TEXT')
            runner_columns = {row['name'] for row in connection.execute('PRAGMA table_info(prediction_runners)')}
            if 'prior_starts' not in runner_columns:  # added with the stage-wise market blend
                connection.execute('ALTER TABLE prediction_runners ADD COLUMN prior_starts REAL')

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute('PRAGMA journal_mode=WAL')
            with connection:
                yield connection
        finally:
            connection.close()

    # -- predictions ---------------------------------------------------
    def save_prediction(self, meta, runners):
        with self._connect() as connection:
            cursor = connection.execute(
                f'INSERT INTO predictions ({",".join(PREDICTION_FIELDS)}) VALUES ({",".join("?" * len(PREDICTION_FIELDS))})',
                [meta.get(field) for field in PREDICTION_FIELDS])
            identifier = cursor.lastrowid
            connection.executemany(
                f'INSERT INTO prediction_runners (prediction_id,{",".join(RUNNER_FIELDS)},distribution_json) '
                f'VALUES (?,{",".join("?" * len(RUNNER_FIELDS))},?)',
                [[identifier] + [runner.get(field) for field in RUNNER_FIELDS] + [json.dumps(runner.get('distribution'))]
                 for runner in runners])
        return identifier

    def _expand(self, connection, prediction):
        item = dict(prediction)
        runners = connection.execute('SELECT * FROM prediction_runners WHERE prediction_id=? ORDER BY horse_number',
                                     (item['id'],)).fetchall()
        item['runners'] = []
        for runner in runners:
            runner = dict(runner)
            runner['distribution'] = json.loads(runner.pop('distribution_json') or 'null')
            item['runners'].append(runner)
        outcome = connection.execute('SELECT * FROM outcomes WHERE prediction_id=?', (item['id'],)).fetchone()
        item['outcome'] = None
        if outcome is not None:
            outcome = dict(outcome)
            outcome['results'] = json.loads(outcome.pop('results_json') or 'null')
            item['outcome'] = outcome
        return item

    def predictions_for_race(self, race_id):
        with self._connect() as connection:
            rows = connection.execute('SELECT * FROM predictions WHERE race_id=? ORDER BY created_at DESC, id DESC',
                                      (race_id,)).fetchall()
            return [self._expand(connection, row) for row in rows]

    def prediction(self, identifier):
        with self._connect() as connection:
            row = connection.execute('SELECT * FROM predictions WHERE id=?', (identifier,)).fetchone()
            return None if row is None else self._expand(connection, row)

    def predicted_races(self, day):
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT race_id, COUNT(*) AS count, MAX(created_at) AS latest FROM predictions WHERE race_date=? '
                'GROUP BY race_id', (day,)).fetchall()
        return {row['race_id']: dict(row) for row in rows}

    def unscored(self, before_day):
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT p.id,p.race_id,p.region,p.race_date FROM predictions p LEFT JOIN outcomes o '
                'ON o.prediction_id=p.id WHERE o.prediction_id IS NULL AND p.race_date<=? ORDER BY p.id',
                (before_day,)).fetchall()
        return [dict(row) for row in rows]

    def save_outcome(self, prediction_id, outcome):
        fields = ('scored_at', 'status', 'actual_starters', 'outcome_log_loss', 'uniform_log_loss', 'top3_log_loss',
                  'win_log_loss', 'pick_horse_number', 'pick_position', 'pick_top3', 'favourite_horse_number',
                  'favourite_won')
        with self._connect() as connection:
            connection.execute(
                f'INSERT OR REPLACE INTO outcomes (prediction_id,{",".join(fields)},results_json) '
                f'VALUES (?,{",".join("?" * len(fields))},?)',
                [prediction_id] + [outcome.get(field) for field in fields] + [json.dumps(outcome.get('results'))])

    def scored_without_odds(self, limit=500):
        """Scored predictions whose stored results predate keeping the final win odds."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT prediction_id FROM outcomes WHERE status='scored' AND results_json NOT LIKE '%\"odds\"%' "
                'ORDER BY prediction_id DESC LIMIT ?', (limit,)).fetchall()
        return [row[0] for row in rows]

    def scored_days(self, include_after=False):
        """Race days with at least one scored prediction, newest first."""
        timing = '' if include_after else "AND p.timing='before'"
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT DISTINCT p.race_date FROM predictions p JOIN outcomes o ON o.prediction_id=p.id '
                f"WHERE o.status='scored' {timing} ORDER BY p.race_date DESC").fetchall()
        return [row[0] for row in rows]

    # -- records -------------------------------------------------------
    def records(self, *, since=None, day=None, region=None, include_after=False, limit=300):
        """The last pre-race prediction of each race with a scored outcome, newest first, summary metrics, and
        the same metrics by racecourse, confidence, surface and distance."""
        conditions, arguments = ["o.status='scored'"], []
        if not include_after:
            conditions.append("p.timing='before'")
        if day:
            conditions.append('p.race_date=?')
            arguments.append(day)
        if since:
            conditions.append('p.race_date>=?')
            arguments.append(since)
        if region:
            conditions.append('p.region=?')
            arguments.append(region)
        where = ' AND '.join(conditions)
        inner = '' if include_after else "AND q.timing='before'"
        with self._connect() as connection:
            rows = connection.execute(
                f'SELECT p.*,o.outcome_log_loss,o.uniform_log_loss,o.top3_log_loss,o.win_log_loss,o.pick_horse_number,'
                f'o.pick_position,o.pick_top3,o.favourite_horse_number,o.favourite_won,o.results_json,'
                f'(SELECT MAX(win_probability)/SUM(win_probability) FROM prediction_runners x WHERE x.prediction_id=p.id '
                f'AND x.distribution_json IS NOT NULL) AS top_win,'
                f'(SELECT GROUP_CONCAT(strength_score) FROM prediction_runners x WHERE x.prediction_id=p.id '
                f'AND x.distribution_json IS NOT NULL AND x.strength_score IS NOT NULL) AS strengths FROM predictions p '
                f'JOIN outcomes o ON o.prediction_id=p.id WHERE {where} AND p.id=('
                f'SELECT q.id FROM predictions q JOIN outcomes r ON r.prediction_id=q.id WHERE q.race_id=p.race_id '
                f"AND r.status='scored' {inner} "
                f'ORDER BY q.created_at DESC, q.id DESC LIMIT 1) ORDER BY p.race_date DESC, p.start_time DESC, p.race_id',
                arguments).fetchall()
            rows = [dict(row) for row in rows]
            names = {}
            for row in rows[:limit]:
                pick = connection.execute(
                    'SELECT horse_name FROM prediction_runners WHERE prediction_id=? AND horse_number=?',
                    (row['id'], row['pick_horse_number'])).fetchone()
                names[row['id']] = pick['horse_name'] if pick else None
            pending = connection.execute(
                "SELECT COUNT(DISTINCT p.race_id) FROM predictions p LEFT JOIN outcomes o ON o.prediction_id=p.id "
                "WHERE o.prediction_id IS NULL").fetchone()[0]
        for row in rows[:limit]:
            row['pick_horse_name'] = names.get(row['id'])
        for row in rows:
            _describe(row)

        def mean(values):
            values = [value for value in values if value is not None and math.isfinite(value)]
            return sum(values) / len(values) if values else None

        summary = dict(races=len(rows),
                       outcome_log_loss=mean([row['outcome_log_loss'] for row in rows]),
                       uniform_log_loss=mean([row['uniform_log_loss'] for row in rows]),
                       top3_log_loss=mean([row['top3_log_loss'] for row in rows]),
                       win_log_loss=mean([row['win_log_loss'] for row in rows]),
                       pick_top3_rate=mean([row['pick_top3'] for row in rows]),
                       favourite_win_rate=mean([row['favourite_won'] for row in rows]),
                       pending_races=pending, **_group_metrics(rows))
        by_day = {}
        for row in rows:
            entry = by_day.setdefault(row['race_date'], [[], []])
            entry[0].append(row['outcome_log_loss'])
            entry[1].append(row['uniform_log_loss'])
        daily = [dict(day=day, races=len(values[0]), model=mean(values[0]), uniform=mean(values[1]))
                 for day, values in sorted(by_day.items())]
        breakdowns = {name: _breakdown(rows, key, order) for name, (key, order) in BREAKDOWNS.items()}
        # the same tables with each condition split by confidence tier (A, B, C)
        by_confidence = {}
        for name, (key, order) in BREAKDOWNS.items():
            if name == 'confidence':
                continue
            split = []
            for item in _breakdown(rows, key, order):
                members = [row for row in rows if row.get(key) == item['name']]
                for tier in _breakdown(members, 'confidence', BREAKDOWNS['confidence'][1]):
                    split.append(dict(tier, name=item['name'], confidence=tier['name']))
            by_confidence[name] = split
        return dict(summary=summary, daily=daily, breakdowns=breakdowns, breakdowns_by_confidence=by_confidence,
                    races=rows[:limit])

    def confidence_history(self, region):
        """[(win probabilities, pick_won or None, pick_top3 or None)] of the last pre-race prediction of each race."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT p.id,o.status,o.pick_position,o.pick_top3,GROUP_CONCAT(r.win_probability) AS wins "
                "FROM predictions p JOIN prediction_runners r ON r.prediction_id=p.id "
                "LEFT JOIN outcomes o ON o.prediction_id=p.id "
                "WHERE p.region=? AND p.timing='before' AND r.win_probability IS NOT NULL AND p.id=("
                "SELECT q.id FROM predictions q WHERE q.race_id=p.race_id AND q.timing='before' "
                "ORDER BY q.created_at DESC, q.id DESC LIMIT 1) GROUP BY p.id", (region,)).fetchall()
        history = []
        for row in rows:
            scored = row['status'] == 'scored'
            history.append(([float(value) for value in row['wins'].split(',')],
                            (row['pick_position'] == 1) if scored else None,
                            bool(row['pick_top3']) if scored and row['pick_top3'] is not None else None))
        return history

    def day_win_probabilities(self, day):
        """{race_id: (region, win probabilities, strength scores, stage temperatures or None)} of the latest
        prediction of each race on ``day``."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT p.race_id,p.region,p.temperatures_json,r.win_probability,r.strength_score FROM predictions p "
                "JOIN prediction_runners r ON r.prediction_id=p.id WHERE p.race_date=? AND r.win_probability IS NOT NULL "
                "AND p.id=(SELECT q.id FROM predictions q WHERE q.race_id=p.race_id ORDER BY q.created_at DESC, q.id DESC "
                "LIMIT 1)", (day,)).fetchall()
        result = {}
        for row in rows:
            entry = result.setdefault(row['race_id'], (row['region'], [], [],
                                                       json.loads(row['temperatures_json'] or 'null')))
            entry[1].append(float(row['win_probability']))
            if row['strength_score'] is not None:
                entry[2].append(float(row['strength_score']))
        return result

    # -- day races and runs --------------------------------------------
    def set_day_races(self, day, region, races, discovered_at):
        with self._connect() as connection:
            connection.executemany(
                'INSERT OR REPLACE INTO day_races VALUES (?,?,?,?,?,?)',
                [(day, region, race['race_id'], race['course'], race['race_number'], discovered_at) for race in races])

    def day_races(self, day):
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                'SELECT * FROM day_races WHERE race_date=? ORDER BY region, course, race_number', (day,))]

    def start_run(self, kind, started_at, log_path):
        with self._connect() as connection:
            return connection.execute("INSERT INTO runs (started_at,kind,status,log_path) VALUES (?,?,'running',?)",
                                      (started_at, kind, str(log_path))).lastrowid

    def update_run(self, identifier, *, status=None, steps=None, finished_at=None):
        with self._connect() as connection:
            if steps is not None:
                connection.execute('UPDATE runs SET steps_json=? WHERE id=?',
                                   (json.dumps(steps, ensure_ascii=False), identifier))
            if status is not None:
                connection.execute('UPDATE runs SET status=?, finished_at=? WHERE id=?', (status, finished_at, identifier))

    def change_token(self):
        """Cheap fingerprint of everything the pages show from the log (changes when anything is added)."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT (SELECT COALESCE(MAX(id),0) FROM predictions),(SELECT COUNT(*) FROM outcomes),"
                "(SELECT COALESCE(MAX(scored_at),'') FROM outcomes),(SELECT COUNT(*) FROM day_races),"
                "(SELECT COALESCE(MAX(discovered_at),'') FROM day_races),"
                "(SELECT COALESCE(MAX(id),0)||':'||COALESCE((SELECT status||COALESCE(steps_json,'') FROM runs "
                "ORDER BY id DESC LIMIT 1),'') FROM runs)").fetchone()
        return '|'.join(str(value) for value in row)

    def runs(self, limit=20):
        with self._connect() as connection:
            rows = connection.execute('SELECT * FROM runs ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        result = []
        for row in rows:
            row = dict(row)
            row['steps'] = json.loads(row.pop('steps_json') or '[]')
            result.append(row)
        return result


# -- record breakdowns -----------------------------------------------------
DISTANCES = (('短距離（〜1399m）', 0, 1400), ('マイル（1400〜1899m）', 1400, 1900),
             ('中距離（1900〜2399m）', 1900, 2400), ('長距離（2400m〜）', 2400, 10 ** 6))
SURFACES = {'芝': '芝', 'ダ': 'ダート', '障': '障害'}
BREAKDOWNS = {
    'course': ('course', None),
    'confidence': ('confidence', ['A', 'B', 'C']),
    'overall': ('overall', ['A', 'B', 'C']),
    'surface': ('surface_name', ['芝', 'ダート', '障害']),
    'distance': ('distance_band', [name for name, *_ in DISTANCES]),
    'race_number': ('race_band', ['1〜4R', '5〜8R', '9R〜']),
}


def _describe(row):
    """Per-race fields for the breakdowns: the favourite (highest win probability) and how it ran, its 単勝 return
    at the final odds, the confidence tier, and the order skill (1 - log loss / log loss of equal chances)."""
    from keiba_ai.bets import confidence_tier
    results = json.loads(row.pop('results_json', None) or 'null') or {}
    favourite = results.get(str(row.get('favourite_horse_number'))) or {}
    position = favourite.get('position') if favourite.get('status') == 'finished' else None
    row['favourite_position'] = position
    row['favourite_top3'] = int(bool(position) and position <= 3)
    odds = favourite.get('odds')
    row['favourite_odds'] = odds
    row['favourite_return'] = (odds if position == 1 else 0.0) if isinstance(odds, (int, float)) and odds > 0 else None
    model, uniform = row.get('outcome_log_loss'), row.get('uniform_log_loss')
    row['skill'] = 1 - model / uniform if model is not None and uniform else None
    top_win = row.get('top_win')
    row['confidence'] = confidence_tier(row['region'], top_win) if top_win is not None else None
    strengths = [float(value) for value in (row.pop('strengths', None) or '').split(',') if value]
    temperatures = json.loads(row.get('temperatures_json') or 'null')
    row['overall'] = row['overall_value'] = None
    if temperatures and len(strengths) >= 3:
        from keiba_ai.bets import order_certainty, overall_tier
        row['overall_value'] = order_certainty(strengths, temperatures)
        row['overall'] = overall_tier(row['region'], row['overall_value'])
    row['surface_name'] = SURFACES.get(row.get('surface'), row.get('surface'))
    distance = row.get('distance_m')
    row['distance_band'] = next((name for name, low, high in DISTANCES if distance and low <= distance < high), None)
    number = row.get('race_number')
    row['race_band'] = None if not number else '1〜4R' if number <= 4 else '5〜8R' if number <= 8 else '9R〜'
    return row


def _group_metrics(rows):
    def mean(values):
        values = [value for value in values if value is not None and math.isfinite(value)]
        return sum(values) / len(values) if values else None
    priced = [row['favourite_return'] for row in rows if row.get('favourite_return') is not None]
    return dict(favourite_top3_rate=mean([row.get('favourite_top3') for row in rows]),
                skill=mean([row.get('skill') for row in rows]),
                favourite_roi=sum(priced) / len(priced) if priced else None, priced_races=len(priced))


def _breakdown(rows, key, order=None):
    groups = {}
    for row in rows:
        if row.get(key):
            groups.setdefault(row[key], []).append(row)
    result = [dict(name=name, races=len(members),
                   favourite_win_rate=sum(bool(row['favourite_won']) for row in members) / len(members),
                   **_group_metrics(members)) for name, members in groups.items()]
    if order:
        rank = {name: index for index, name in enumerate(order)}
        return sorted(result, key=lambda item: (rank.get(item['name'], len(rank)), item['name']))
    return sorted(result, key=lambda item: (-item['races'], item['name']))
