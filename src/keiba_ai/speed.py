"""Causal speed figures: track-variant and class-aware standard times (design 6.2 A).

Times are handled as pace p = seconds per 1000 m.  For each race the
reference pace is the mean pace of its first three finishers.

* Standard pace S[course, surface, distance]          (class-free; used for figures)
* Standard pace S[course, surface, distance, class]   (used only to measure the day)
* Day variant V[day, course, surface] = shrunken median over that day's races
  of (race pace - class standard).  A fast or slow track moves every race of
  the day; a day of weak classes does not, because classes have own standards.

The figure of a finisher is ``-(p_i - S - V) + weight_coef * (carried - ref)``
in seconds per km (positive = faster than standard).  Everything for day d is
computed only after all of day d's feature rows were emitted, so a row only
sees figures from earlier days.  Standards use exponential averaging that
starts as a running mean.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math

SPEED_FEATURES = (
    'speed_fig_previous', 'speed_fig_recent5_mean', 'speed_fig_recent5_best', 'speed_fig_same_surface_band_best',
    'speed_fig_trend', 'speed_fig_count', 'speed_fig_wet_best',
    'speed_fig_mean_race_z', 'speed_fig_best_race_z', 'speed_fig_best_minus_field_max', 'speed_fig_field_top3_mean',
)
_WET = {'重', '不良'}


class _Average:
    __slots__ = ('value', 'count')

    def __init__(self):
        self.value, self.count = None, 0

    def update(self, observation, alpha):
        self.count += 1
        if self.value is None:
            self.value = observation
        else:
            self.value += max(alpha, 1.0 / self.count) * (observation - self.value)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _z(values):
    known = [value for value in values if value is not None]
    if len(known) < 2:
        return [None] * len(values)
    mean = sum(known) / len(known)
    spread = math.sqrt(sum((value - mean) ** 2 for value in known) / len(known))
    return [None if value is None else (0.0 if spread == 0 else (value - mean) / spread) for value in values]


class SpeedFigures:
    """Causal speed-figure state.  ``feed(races)`` consumes race row lists in day order (rows of a day see
    only earlier days); ``finish()`` closes the last day; ``vectors(race)`` gives the features of a new race of
    a later day without changing the state."""

    def __init__(self, *, alpha=.03, class_min_races=5, weight_coef=.12, reference_weight=55.0, history_length=10):
        self.alpha, self.class_min_races = alpha, class_min_races
        self.weight_coef, self.reference_weight = weight_coef, reference_weight
        self.standard = defaultdict(_Average)        # (course, surface, distance)
        self.class_standard = defaultdict(_Average)  # (course, surface, distance, class)
        self.history = defaultdict(lambda: deque(maxlen=history_length))  # horse -> (figure, surface, band, wet)
        self.counts = defaultdict(int)
        self.variants = []
        self.pending_day, self.pending = None, []

    def _flush(self, day_races):
        standard, class_standard, history, counts = self.standard, self.class_standard, self.history, self.counts
        by_track = defaultdict(list)
        measured = []
        for race in day_races:
            first = race[0]
            distance = _number(first.get('distance_m'))
            if not distance:
                continue
            finishers = []
            for row in race:
                time, position = _number(row.get('time_seconds')), _number(row.get('finish_position'))
                if row.get('finish_status') == 'finished' and time and position:
                    finishers.append((position, time / distance * 1000, row))
            if len(finishers) < 3:
                continue
            finishers.sort(key=lambda item: item[0])
            pace = sum(item[1] for item in finishers[:3]) / 3
            track = (first.get('course'), first.get('surface'))
            key = (*track, distance)
            class_key = (*key, first.get('race_class') or '?')
            reference = (class_standard[class_key] if class_standard[class_key].count >= self.class_min_races
                         else standard[key])
            if reference.value is not None:
                by_track[track].append(pace - reference.value)
            measured.append((track, key, class_key, pace, finishers, first))
        variant = {}
        for track, residuals in by_track.items():
            residuals = sorted(residuals)
            middle = len(residuals) // 2
            median = residuals[middle] if len(residuals) % 2 else (residuals[middle - 1] + residuals[middle]) / 2
            variant[track] = median * len(residuals) / (len(residuals) + 1)
            self.variants.append(variant[track])
        for track, key, class_key, pace, finishers, first in measured:
            shift = variant.get(track, 0.0)
            baseline = standard[key].value
            wet = first.get('track_condition') in _WET
            band = int(_number(first.get('distance_m')) // 400)
            if baseline is not None:
                for _, horse_pace, row in finishers:
                    carried = _number(row.get('carried_weight_kg'))
                    figure = -(horse_pace - baseline - shift)
                    if carried is not None:
                        figure += self.weight_coef * (carried - self.reference_weight)
                    horse = row['horse_id']
                    history[horse].append((figure, first.get('surface'), band, wet))
                    counts[horse] += 1
            standard[key].update(pace - shift, self.alpha)
            class_standard[class_key].update(pace - shift, self.alpha)

    def vectors(self, race):
        """Feature tuples (SPEED_FEATURES order) of every row of ``race`` from the current state."""
        history, counts = self.history, self.counts
        first = race[0]
        distance = _number(first.get('distance_m'))
        band = int(distance // 400) if distance else None
        surface = first.get('surface')
        vectors = []
        for row in race:
            past = history.get(row['horse_id'])
            figures = [item[0] for item in past] if past else []
            recent = figures[-5:]
            same = [item[0] for item in past or () if item[1] == surface and item[2] == band]
            wet = [item[0] for item in past or () if item[3]]
            vectors.append([
                recent[-1] if recent else None,
                sum(recent) / len(recent) if recent else None,
                max(recent) if recent else None,
                max(same) if same else None,
                recent[-1] - sum(recent[:-1]) / len(recent[:-1]) if len(recent) >= 2 else None,
                float(counts.get(row['horse_id'], 0)),
                max(wet) if wet else None,
            ])
        means, bests = [vector[1] for vector in vectors], [vector[2] for vector in vectors]
        mean_z, best_z = _z(means), _z(bests)
        known_best = sorted((value for value in bests if value is not None), reverse=True)
        top3 = sum(known_best[:3]) / len(known_best[:3]) if known_best else None
        result = []
        for index, vector in enumerate(vectors):
            others = [value for other, value in enumerate(bests) if other != index and value is not None]
            result.append(tuple(vector + [
                mean_z[index], best_z[index],
                bests[index] - max(others) if bests[index] is not None and others else None, top3]))
        return result

    def feed(self, races, output=None):
        for race in races:
            day = race[0]['day']
            if self.pending_day is not None and day != self.pending_day:
                self._flush(self.pending)
                self.pending = []
            self.pending_day = day
            if output is not None:
                for row, vector in zip(race, self.vectors(race)):
                    output[(row['race_id'], row['horse_id'])] = vector
            self.pending.append(race)
        return output

    def finish(self):
        if self.pending:
            self._flush(self.pending)
            self.pending = []
        return self

    def diagnostics(self):
        return dict(standards=len(self.standard), class_standards=len(self.class_standard),
                    variant_days=len(self.variants),
                    variant_abs_mean=sum(abs(value) for value in self.variants) / len(self.variants)
                    if self.variants else None)


def speed_feature_rows(races, *, alpha=.03, class_min_races=5, weight_coef=.12, reference_weight=55.0,
                       history_length=10):
    """``races``: iterable of race row lists in day order (one list per race).

    Each row dict needs: day, race_id, horse_id, course, surface, distance_m,
    finish_status, finish_position, time_seconds, race_class, track_condition,
    carried_weight_kg.  Returns {(race_id, horse_id): tuple(SPEED_FEATURES order)}
    and diagnostics.
    """
    figures = SpeedFigures(alpha=alpha, class_min_races=class_min_races, weight_coef=weight_coef,
                           reference_weight=reference_weight, history_length=history_length)
    output = figures.feed(races, {})
    figures.finish()
    return output, figures.diagnostics()


_SPEED_FIELDS = ('course', 'surface', 'distance_m', 'finish_status', 'finish_position', 'time_seconds', 'race_class',
                 'track_condition', 'carried_weight_kg')


def speed_row(row):
    return dict({key: row.get(key) for key in _SPEED_FIELDS}, day=row['day'], race_id=row['race_id'],
                horse_id=row['horse_id'])


def load_speed_state(database, group, before_day):
    """(SpeedFigures after every stored day before ``before_day``, {(race_id, horse_id): features} of those rows)."""
    import json
    from itertools import groupby
    from .model import readonly
    connection = readonly(database)
    try:
        cursor = connection.execute(
            'SELECT f.day,f.race_id,f.horse_id,p.payload_json FROM features f JOIN input_rows p '
            'ON p.region=f.region AND p.race_id=f.race_id AND p.horse_id=f.horse_id WHERE f.group_name=? AND f.day<? '
            'ORDER BY f.day,f.region,f.race_id,f.horse_number', (group, before_day))

        def rows():
            for day, race, horse, payload in cursor:
                values = json.loads(payload)
                yield dict({key: values.get(key) for key in _SPEED_FIELDS}, day=day, race_id=race, horse_id=horse)

        figures = SpeedFigures()
        table = figures.feed((list(race) for _, race in groupby(rows(), key=lambda row: (row['day'], row['race_id']))), {})
        figures.finish()
    finally:
        connection.close()
    return figures, table


def load_speed_block(database, group, identities, *, before_day=None, extra_races=()):
    """Speed features aligned to ``identities`` (race_id, horse_id, horse_number).

    ``before_day`` keeps only stored rows of earlier days; ``extra_races`` (race
    row lists of later days, e.g. today's entries) are appended after them.
    """
    import json
    import numpy as np
    from itertools import chain, groupby
    from .model import readonly
    fields = ('course', 'surface', 'distance_m', 'finish_status', 'finish_position', 'time_seconds', 'race_class',
              'track_condition', 'carried_weight_kg')
    connection = readonly(database)
    try:
        cursor = connection.execute(
            'SELECT f.day,f.race_id,f.horse_id,p.payload_json FROM features f JOIN input_rows p '
            'ON p.region=f.region AND p.race_id=f.race_id AND p.horse_id=f.horse_id WHERE f.group_name=? '
            + ('AND f.day<? ' if before_day is not None else '')
            + 'ORDER BY f.day,f.region,f.race_id,f.horse_number', (group,) + ((before_day,) if before_day is not None else ()))

        def rows():
            for day, race, horse, payload in cursor:
                values = json.loads(payload)
                yield dict({key: values.get(key) for key in fields}, day=day, race_id=race, horse_id=horse)

        races = (list(race) for _, race in groupby(rows(), key=lambda row: (row['day'], row['race_id'])))
        extra = [[dict({key: row.get(key) for key in fields}, day=row['day'], race_id=row['race_id'],
                       horse_id=row['horse_id']) for row in race] for race in extra_races]
        table, diagnostics = speed_feature_rows(chain(races, extra))
    finally:
        connection.close()
    block = np.full((len(identities), len(SPEED_FEATURES)), np.nan, dtype=np.float32)
    for index, (race, horse, _) in enumerate(identities):
        values = table.get((race, horse))
        if values is not None:
            block[index] = [np.nan if value is None else value for value in values]
    return block, diagnostics
