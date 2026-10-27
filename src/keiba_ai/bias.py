"""Day-local track-tendency measurements, separated from horse histories.

Race numbers are an explicit availability proxy, not a claim that historical
publication timestamps were observed.  All measured outcomes belong to lower
numbered races at the same course, surface and day.  The target's observed
going, odds, corners and outcome are never read by ``bias_features``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import math
import re


BIAS_FEATURE_NAMES = (
    'day_front_bias', 'day_inner_bias', 'day_bias_races', 'day_bias_prior_front',
    'horse_front_bias_fit', 'horse_inner_bias_fit', 'field_front_runner_count',
    'meeting_day_index',
)
_CENTRAL_COURSES = {f'{index:02d}': name for index, name in enumerate(
    ('札幌', '函館', '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉'), 1)}
_LOCAL_COURSES = {
    '30': '門別', '35': '盛岡', '36': '水沢', '42': '浦和', '43': '船橋',
    '44': '大井', '45': '川崎', '46': '金沢', '47': '笠松', '48': '名古屋',
    '50': '園田', '51': '姫路', '54': '高知', '55': '佐賀', '65': '帯広(ば)',
}
_SUPPORTED = {'central_flat', 'local_flat'}
_WITHDRAWN = {'scratched', 'excluded'}
_GOING = {'良', '稍重', '重', '不良'}


@dataclass(frozen=True)
class BiasConfig:
    prior_races: float = 4.0
    decay: float = 0.8

    def __post_init__(self):
        if (isinstance(self.prior_races, bool) or isinstance(self.decay, bool)
                or not math.isfinite(self.prior_races) or self.prior_races <= 0
                or not math.isfinite(self.decay) or not 0 < self.decay <= 1):
            raise ValueError('バイアスの事前レース数は正数、減衰は0より大きく1以下で指定してください。')

    @property
    def identifier(self):
        return f'n{self.prior_races:g}_d{self.decay:g}'

    def as_dict(self):
        return dict(prior_races=float(self.prior_races), decay=float(self.decay))


DEFAULT_BIAS_CONFIG = BiasConfig()
BIAS_CONFIGS = (BiasConfig(3, .7), DEFAULT_BIAS_CONFIG, BiasConfig(5, .9))


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _integer(value):
    value = _number(value)
    return int(value) if value is not None and value.is_integer() else None


def _context(race):
    if not race:
        raise ValueError('空のレースから当日バイアスは作れません。')
    first = race[0]
    try:
        day = date.fromisoformat(first.get('day'))
    except (TypeError, ValueError) as exc:
        raise ValueError('当日バイアスの日付はYYYY-MM-DDで指定してください。') from exc
    if day.isoformat() != first.get('day'):
        raise ValueError('当日バイアスの日付はYYYY-MM-DDで指定してください。')
    fields = ('day', 'region', 'race_id', 'course', 'surface', 'group_name', 'distance_m')
    if any(any(row.get(key) != first.get(key) for key in fields) for row in race):
        raise ValueError('当日バイアスのレース条件が一致しません。')
    return first, day


def race_order(race) -> int | None:
    """Validate an already observed date/course against a known ID format.

    No date or course is filled from an ID.  Unrecognized IDs simply cannot
    contribute intraday measurements; the long-term prior remains available.
    """
    first, day = _context(race)
    identifier = first.get('race_id')
    if not isinstance(identifier, str) or not re.fullmatch(r'[0-9]{12}', identifier):
        return None
    if identifier[:4] != f'{day.year:04d}':
        return None
    number = int(identifier[-2:])
    if first.get('region') == 'central':
        if (_CENTRAL_COURSES.get(identifier[4:6]) != first.get('course')
                or not 1 <= number <= 12
                or int(identifier[6:8]) < 1 or int(identifier[8:10]) < 1):
            return None
    elif first.get('region') == 'local':
        course = '帯広(ば)' if first.get('course') == '帯広' else first.get('course')
        if (_LOCAL_COURSES.get(identifier[4:6]) != course
                or identifier[6:10] != day.strftime('%m%d') or not 1 <= number <= 99):
            return None
    else:
        return None
    return number


def _meeting(race):
    first, _ = _context(race)
    if first.get('region') != 'central' or race_order(race) is None:
        return None
    identifier = first['race_id']
    return (identifier[:8], int(identifier[8:10]))


def _track(first):
    surface = {'ダート': 'ダ'}.get(first.get('surface'), first.get('surface'))
    return first.get('region'), first.get('course'), surface


def _band(first):
    distance = _number(first.get('distance_m'))
    return int(distance // 200) if distance is not None and distance > 0 else None


def _prior_key(first):
    return (*_track(first), _band(first))


def gate_position(row, race) -> float | None:
    """Horse-number position, preserving gaps left by scratched runners.

    This is a gate proxy, not the numbered frame (枠番).  The full original
    field is useful when a declared count is absent and the outside horse was
    scratched; callers may supply that field separately to ``bias_features``.
    """
    number = _integer(row.get('horse_number'))
    if number is None or number < 1:
        return None
    declarations = {_integer(item.get('declared_field_size')) for item in race
                    if item.get('declared_field_size') is not None}
    if None in declarations or len(declarations) > 1:
        return None
    maximum = max((_integer(item.get('horse_number')) or 0 for item in race), default=0)
    count = next(iter(declarations)) if declarations else maximum
    if count < maximum or count < 2 or number > count:
        return None
    return (number - 1) / (count - 1)


def _going(race):
    choices = {{'稍': '稍重', '不': '不良'}.get(item.get('track_condition'), item.get('track_condition'))
               for item in race if item.get('track_condition') is not None}
    return next(iter(choices)) if len(choices) == 1 and choices <= _GOING else None


def _negative_slope(points):
    if len(points) < 6:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    # Near-constant positions do not support a stable slope estimate.
    if denominator / len(points) < .005:
        return None
    return -sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


@dataclass(frozen=True)
class RaceBias:
    day: date
    track: tuple
    number: int
    distance_m: float
    going: str | None
    meeting: tuple | None
    front: float | None
    inner: float | None
    heads: int


class IntradayBiasState:
    """One date's completed races; never inserted into horse HistoryState."""

    def __init__(self, day: str):
        parsed = date.fromisoformat(day)
        if parsed.isoformat() != day:
            raise ValueError('当日状態の日付はYYYY-MM-DDで指定してください。')
        self.day = day
        self._date = parsed
        self._records = []
        self._observed = defaultdict(dict)
        self._seen = set()
        self.skipped_reasons = Counter()

    @property
    def records(self):
        return tuple(self._records)

    @property
    def valid_races(self):
        return len(self._records)

    def add_result(self, race) -> bool:
        from .features import _corner_form
        first, day = _context(race)
        if day != self._date:
            raise ValueError('別の日の結果を当日バイアス状態に追加できません。')
        identity = first.get('region'), first.get('race_id')
        if identity in self._seen:
            raise ValueError('同じレースの当日結果を二重に追加できません。')
        self._seen.add(identity)
        if first.get('group_name') not in _SUPPORTED:
            self.skipped_reasons['unsupported_group'] += 1
            return False
        number = race_order(race)
        if number is None:
            self.skipped_reasons['unknown_race_order'] += 1
            return False
        track = _track(first)
        if number in self._observed[track]:
            raise ValueError('同じ競馬場・馬場のレース番号が重複しています。')
        self._observed[track][number] = _going(race)
        active = [row for row in race if row.get('finish_status') not in _WITHDRAWN]
        finished = [row for row in active if row.get('finish_status') == 'finished']
        if len(finished) < 6:
            self.skipped_reasons['small_finished_field'] += 1
            return False
        size = len(active)
        points = []
        for row in finished:
            position = _integer(row.get('finish_position'))
            popularity = _integer(row.get('popularity'))
            if position is None or popularity is None or not 1 <= position <= size or not 1 <= popularity <= size:
                continue
            corner = _corner_form(row, size, position)['last_corner_fraction']
            gate = gate_position(row, race)
            if corner is None or gate is None:
                continue
            # Identical to _Run.market_rank_gain's popularity-to-rank gain.
            gain = (popularity - position) / max(size - 1, 1)
            points.append((corner, gate, gain))
        if len(points) < max(6, math.ceil(.7 * len(finished))):
            self.skipped_reasons['insufficient_corner_market_coverage'] += 1
            return False
        front = _negative_slope([(corner, gain) for corner, _, gain in points])
        inner = _negative_slope([(gate, gain) for _, gate, gain in points])
        distance = _number(first.get('distance_m'))
        if (front is None and inner is None) or distance is None or distance <= 0:
            self.skipped_reasons['unidentified_slope'] += 1
            return False
        self._records.append(RaceBias(day, track, number, distance, _going(race),
                                      _meeting(race), front, inner, len(points)))
        return True

    def before(self, race):
        first, day = _context(race)
        if day != self._date:
            raise ValueError('対象レースと当日バイアス状態の日付が一致しません。')
        number = race_order(race)
        if number is None:
            return (), None
        track = _track(first)
        observed = self._observed.get(track, {})
        if any(previous >= number for previous in observed):
            raise ValueError('対象レース自身または後続レースの結果が当日状態に含まれています。')
        records = tuple(item for item in self._records if item.track == track)
        known_going = [(previous, going) for previous, going in observed.items() if going is not None]
        latest_going = max(known_going)[1] if known_going else None
        return records, latest_going


@dataclass
class _Moment:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value):
        if value is None:
            return
        self.count += 1
        change = value - self.mean
        self.mean += change / self.count
        self.m2 += change * (value - self.mean)

    @property
    def variance(self):
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0


class BiasPriorState:
    """Earlier dates' equal-race weighted means and variances by track/band."""

    def __init__(self):
        self.latest_day = None
        self._moments = {}
        self._previous = {}

    def update_day(self, intraday: IntradayBiasState):
        day = date.fromisoformat(intraday.day)
        if self.latest_day is not None and day <= self.latest_day:
            raise ValueError('バイアス事前値は日付順で、1日につき一度だけ更新してください。')
        per_day = defaultdict(list)
        for record in intraday.records:
            key = (*record.track, int(record.distance_m // 200))
            front, inner = self._moments.setdefault(key, (_Moment(), _Moment()))
            front.add(record.front)
            inner.add(record.inner)
            per_day[key].append(record)
        for key, records in per_day.items():
            front = [item.front for item in records if item.front is not None]
            inner = [item.inner for item in records if item.inner is not None]
            meetings = {item.meeting for item in records}
            self._previous[key] = (day, next(iter(meetings)) if len(meetings) == 1 else None,
                                   sum(front) / len(front) if front else None,
                                   sum(inner) / len(inner) if inner else None)
        self.latest_day = day

    def _check(self, race):
        _, day = _context(race)
        if self.latest_day is not None and self.latest_day >= day:
            raise ValueError('当日以降の結果を含むバイアス事前値は使用できません。')
        return day

    def statistics(self, race):
        self._check(race)
        first, _ = _context(race)
        front, inner = self._moments.get(_prior_key(first), (_Moment(), _Moment()))
        return dict(front_count=front.count, front_mean=front.mean, front_variance=front.variance,
                    inner_count=inner.count, inner_mean=inner.mean, inner_variance=inner.variance)

    def estimate(self, race):
        day = self._check(race)
        first, _ = _context(race)
        stats = self.statistics(race)
        front, inner = stats['front_mean'], stats['inner_mean']
        previous = self._previous.get(_prior_key(first))
        meeting = _meeting(race)
        if previous is not None and meeting is not None:
            previous_day, previous_meeting, previous_front, previous_inner = previous
            if (previous_day == day - timedelta(days=1) and previous_meeting is not None
                    and previous_meeting[0] == meeting[0] and previous_meeting[1] + 1 == meeting[1]):
                if previous_front is not None:
                    front = .5 * front + .5 * previous_front
                if previous_inner is not None:
                    inner = .5 * inner + .5 * previous_inner
        return front, inner


def bias_features(race, intraday, priors, *, form_vectors, config=DEFAULT_BIAS_CONFIG,
                  full_race=None):
    """Return eight pure features using observed earlier races and past form."""
    from .features import RANK_FEATURE_NAMES
    first, day = _context(race)
    if len(form_vectors) != len(race):
        raise ValueError('バイアス用の馬別履歴特徴量の頭数が一致しません。')
    if intraday is None:
        intraday = IntradayBiasState(day.isoformat())
    if priors is None:
        priors = BiasPriorState()
    # Keep the date guard even for groups that do not use track tendencies.
    priors._check(race)
    records, latest_going = intraday.before(race)
    if first.get('group_name') not in _SUPPORTED:
        return [[None, None, 0.0, None, None, None, None, None] for _ in race]
    number = race_order(race)
    prior_front, prior_inner = priors.estimate(race)
    front_sum = inner_sum = front_weight = inner_weight = 0.0
    target_distance = _number(first.get('distance_m'))
    used = 0
    for record in records:
        distance_weight = math.exp(-abs(target_distance - record.distance_m) / 400) if target_distance else .5
        going_weight = .5 if latest_going is not None and record.going is not None and record.going != latest_going else 1.0
        weight = config.decay ** (number - record.number) * distance_weight * going_weight
        if record.front is not None:
            front_sum += weight * record.front
            front_weight += weight
        if record.inner is not None:
            inner_sum += weight * record.inner
            inner_weight += weight
        used += 1
    front = (front_sum + config.prior_races * prior_front) / (front_weight + config.prior_races)
    inner = (inner_sum + config.prior_races * prior_inner) / (inner_weight + config.prior_races)
    corner_index = RANK_FEATURE_NAMES.index('horse_group_recent5_last_corner_fraction')
    front_index = RANK_FEATURE_NAMES.index('horse_group_recent5_corner_front_rate')
    corner_values = [_number(vector[corner_index]) for vector in form_vectors]
    historical_front = [_number(vector[front_index]) for vector in form_vectors]
    known = [value for value in historical_front if value is not None]
    front_count = float(sum(value >= .5 for value in known)) if known else None
    meeting = _meeting(race)
    gate_race = full_race if full_race is not None else race
    if full_race is not None:
        full_first, _ = _context(full_race)
        if any(full_first.get(key) != first.get(key) for key in ('day', 'region', 'race_id', 'course', 'surface', 'group_name')):
            raise ValueError('馬番補正用の全出走馬が対象レースと一致しません。')
    results = []
    for row, corner in zip(race, corner_values):
        gate = gate_position(row, gate_race)
        results.append([front, inner, float(used), prior_front,
                        front * (.5 - corner) if corner is not None else None,
                        inner * (.5 - gate) if gate is not None else None,
                        front_count, float(meeting[1]) if meeting else None])
    return results
