"""Predictions for races (with logging) and scoring of logged predictions against results."""
from __future__ import annotations

from datetime import datetime
import math
import threading

from .logbook import Logbook
from .sources import SavedRaces, Scraper, course_of, entries_from_results, race_number_of
from .state import JST, App, now, today

UNSUPPORTED = 'このレースは対象外です（中央・地方の平地のみ予想します）。'


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def race_start(day, start_time):
    if not start_time:
        return None
    try:
        hour, minute = (int(part) for part in str(start_time).split(':'))
        return datetime.fromisoformat(day).replace(hour=hour, minute=minute, tzinfo=JST)
    except ValueError:
        return None


class Engine:
    """Owns the cached predictor for the current database/model and the prediction log."""

    def __init__(self, app: App, *, scraper=None, samples=4096, report=None):
        self.app = app
        self.logbook = Logbook(app.logbook_path)
        self.saved = SavedRaces(app.scraper_db)
        self.scraper = scraper or Scraper(app.data_dir)
        self.samples = samples
        self.report = report or (lambda **message: None)
        self._predictor, self._key = None, None
        self._lock = threading.Lock()

    # -- predictor -----------------------------------------------------
    def predictor(self):
        from keiba_ai.pipeline_v3 import Predictor
        state = self.app.require_ready()
        key = (state['dataset'], state['model_dir'])
        with self._lock:
            if self._key != key:
                self._predictor = Predictor(state['dataset'], state['model_dir'], threads=state['threads'],
                                            report=self.report)
                self._key = key
            return self._predictor

    def reset(self):
        with self._lock:
            self._predictor, self._key = None, None

    # -- race cards ----------------------------------------------------
    def card(self, region, race_id, *, refresh=False):
        """(rows, source) for the race: the latest saved entries, else the results without outcomes."""
        if refresh:
            self.scraper.fetch(region, [race_id], 'entries', refresh=True)
        entries = self.saved.snapshot(region, race_id, 'entries')
        if entries is None and not refresh:
            results = self.saved.snapshot(region, race_id, 'results')
            if results is None:
                self.scraper.fetch(region, [race_id], 'entries', refresh=True)
                entries = self.saved.snapshot(region, race_id, 'entries')
            else:
                return entries_from_results(results), 'results'
        if entries is None:
            raise ValueError('出馬表を取得できませんでした。')
        return entries, 'entries'

    def _timing(self, day, start_time, region, race_id):
        start = race_start(day, start_time)
        if start is not None:
            return 'before' if now() < start else 'after'
        if day > today():
            return 'before'
        if day < today():
            return 'after'
        return 'after' if self.saved.snapshot(region, race_id, 'results') else 'before'

    def _record(self, race, rows, source, results, *, origin, predictor):
        day, region, race_id, normalized = race
        raw = {str(row.get('horse_id')): row for row in rows}
        predicted = {row['horse_id']: row for row in results}
        first = rows[0]
        runners = []
        for row in normalized:
            original = raw.get(row['horse_id'], {})
            result = predicted.get(row['horse_id'])
            runner = dict(horse_id=row['horse_id'], horse_number=row['horse_number'], frame=row.get('frame'),
                          horse_name=original.get('horse_name'), sex=row.get('sex'), age=row.get('age'),
                          carried_weight_kg=row.get('carried_weight_kg'), jockey_name=original.get('jockey_name'),
                          trainer_name=original.get('trainer_name'), weight_kg=row.get('weight_kg'),
                          weight_change_kg=row.get('weight_change_kg'),
                          status=row.get('finish_status') or 'starter')
            if result is not None:
                size = result['starters']
                runner.update(win_probability=result['win_probability'], top3_probability=result['top3_probability'],
                              expected_rank=result['expected_rank_given_finish'],
                              strength_score=result['strength_score'], finish_probability=result['finish_probability'],
                              distribution=[result[f'rank_{rank}_probability'] for rank in range(1, size + 1)]
                              + [result['nonfinish_probability']])
            runners.append(runner)
        starters = sum(runner.get('distribution') is not None for runner in runners)
        group = next(iter(predicted.values()))['group']
        start_time = first.get('race_start_time')
        meta = dict(created_at=now().isoformat(timespec='seconds'), race_id=race_id, region=region, race_date=day,
                    course=first.get('race_course') or course_of(region, race_id),
                    race_number=first.get('race_number') or race_number_of(race_id),
                    race_name=first.get('race_name'), start_time=start_time, surface=normalized[0].get('surface'),
                    distance_m=normalized[0].get('distance_m'), track_condition=normalized[0].get('track_condition'),
                    starters=starters, timing=self._timing(day, start_time, region, race_id), source=source,
                    origin=origin, model_dir=str(predictor.model_dir), model_train_end=predictor.train_end(group),
                    dataset=str(predictor.database), history_through=predictor.last_day,
                    bodyweight_known=int(any(runner.get('weight_kg') for runner in runners if runner.get('distribution'))),
                    entries_retrieved_at=first.get('retrieved_at'))
        identifier = self.logbook.save_prediction(meta, runners)
        self.score(identifier)
        return identifier

    def predict_race(self, region, race_id, *, refresh=False, origin='manual'):
        from keiba_ai.pipeline_v3 import Predictor, entry_races
        rows, source = self.card(region, race_id, refresh=refresh)
        races = entry_races(rows)
        if len(races) != 1:
            raise ValueError('出馬表の内容が1レース分ではありません。')
        if not Predictor.supported(races[0][3]):
            raise ValueError(UNSUPPORTED)
        predictor = self.predictor()
        results = predictor.predict_races(races, samples=self.samples)
        if not results:
            raise ValueError('予想できる出走馬がいません。')
        return self._record(races[0], rows, source, results, origin=origin, predictor=predictor)

    def discover(self, day):
        races = []
        for region in ('central', 'local'):
            identifiers = self.scraper.discover(day, region, refresh=day >= today())
            listed = [dict(race_id=race_id, course=course_of(region, race_id), race_number=race_number_of(race_id))
                      for race_id in identifiers]
            self.logbook.set_day_races(day, region, listed, now().isoformat(timespec='seconds'))
            races.extend(dict(listed_race, region=region) for listed_race in listed)
        return races

    def predict_day(self, day, *, refresh=True, origin='manual', progress=None):
        """Fetch every listed race card of the day and predict all supported races in one batch."""
        from keiba_ai.pipeline_v3 import Predictor, entry_races
        progress = progress or (lambda **message: None)
        listed = self.logbook.day_races(day) or self.discover(day)
        cards, skipped = [], []
        for index, race in enumerate(listed, 1):
            if race['course'] == '帯広(ば)':
                skipped.append(dict(race_id=race['race_id'], reason='ばんえい'))
                continue
            progress(message=f'出馬表を取得中 {index}/{len(listed)}: {race["course"]}{race["race_number"]}R')
            try:
                rows, source = self.card(race['region'], race['race_id'], refresh=refresh)
                parsed = entry_races(rows)
                if len(parsed) == 1 and Predictor.supported(parsed[0][3]):
                    cards.append((parsed[0], rows, source))
                else:
                    skipped.append(dict(race_id=race['race_id'], reason='対象外'))
            except (ValueError, KeyError) as exc:
                skipped.append(dict(race_id=race['race_id'], reason=str(exc)[:200]))
        if not cards:
            return dict(predicted=[], skipped=skipped)
        progress(message=f'{len(cards)}レースを予想中')
        predictor = self.predictor()
        results = predictor.predict_races([card[0] for card in cards], samples=self.samples)
        by_race = {}
        for row in results:
            by_race.setdefault(row['race_id'], []).append(row)
        predicted = []
        for race, rows, source in cards:
            if race[2] in by_race:
                predicted.append(self._record(race, rows, source, by_race[race[2]], origin=origin, predictor=predictor))
        return dict(predicted=predicted, skipped=skipped)

    # -- scoring -------------------------------------------------------
    def score(self, prediction_id):
        """Score one prediction if its race has results; returns the outcome status or None."""
        import numpy as np
        from keiba_ai.ranking_v3 import staged_rank_distribution
        prediction = self.logbook.prediction(prediction_id)
        if prediction is None:
            return None
        results = self.saved.snapshot(prediction['region'], prediction['race_id'], 'results')
        if results is None:
            return None
        actual = {}
        for row in results:
            status = row.get('finish_status')
            if status in ('scratched', 'excluded'):
                continue
            actual[str(row.get('horse_id'))] = dict(horse_number=row.get('horse_number'), status=status,
                                                    position=row.get('finish_position'))
        runners = [runner for runner in prediction['runners'] if runner.get('distribution')]
        outcome = dict(scored_at=now().isoformat(timespec='seconds'), actual_starters=len(actual),
                       results={str(item['horse_number']): item for item in actual.values()})
        if not actual:
            return None
        if {runner['horse_id'] for runner in runners} != set(actual):
            outcome['status'] = 'field_changed'
            self.logbook.save_outcome(prediction_id, outcome)
            return outcome['status']
        if any(item['status'] is None or (item['status'] == 'finished' and not item['position'])
               for item in actual.values()):
            return None
        size = len(runners)
        matrix = np.asarray([runner['distribution'] for runner in runners], dtype=float)
        finished = np.asarray([actual[runner['horse_id']]['status'] == 'finished' for runner in runners])
        positions = np.asarray([actual[runner['horse_id']]['position'] or 0 for runner in runners], dtype=int)
        index = np.where(finished, np.clip(positions - 1, 0, size - 1), size)
        completion = np.asarray([runner['finish_probability'] for runner in runners], dtype=float)
        uniform = staged_rank_distribution(np.zeros(size), completion, [1.0], samples=2048, seed=0)
        rows = np.arange(size)
        top3 = np.clip(matrix[:, :min(3, size)].sum(axis=1), 1e-7, 1 - 1e-7)
        in_top3 = finished & (positions <= 3)
        winners = np.flatnonzero(finished & (positions == 1))
        pick = int(np.argmax(top3))
        favourite = int(np.argmax(matrix[:, 0]))
        outcome.update(
            status='scored',
            outcome_log_loss=float(-np.log(np.clip(matrix[rows, index], 1e-12, 1)).mean()),
            uniform_log_loss=float(-np.log(np.clip(uniform[rows, index], 1e-12, 1)).mean()),
            top3_log_loss=float(-(in_top3 * np.log(top3) + (~in_top3) * np.log(1 - top3)).mean()),
            win_log_loss=float(-np.log(max(matrix[winners[0], 0], 1e-12))) if len(winners) == 1 else None,
            pick_horse_number=runners[pick]['horse_number'],
            pick_position=int(positions[pick]) if finished[pick] else None, pick_top3=int(in_top3[pick]),
            favourite_horse_number=runners[favourite]['horse_number'],
            favourite_won=int(finished[favourite] and positions[favourite] == 1))
        self.logbook.save_outcome(prediction_id, outcome)
        return outcome['status']

    def score_pending(self):
        counts = {}
        for item in self.logbook.unscored(today()):
            status = self.score(item['id'])
            counts[status or 'waiting'] = counts.get(status or 'waiting', 0) + 1
        return counts
