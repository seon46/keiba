"""Previous-day form and separate earlier-race track tendencies.

Horse, jockey and trainer histories use earlier calendar days only. Separate
day-local bias features use earlier numbered races at the same venue/surface.
Horse recent form spans groups; career rates remain specific to each model
group. Verified race-ID fields provide scheduled order and meeting day, while
horse, jockey and trainer IDs remain opaque lookup keys.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from itertools import groupby
import json
import math
from pathlib import Path
import sqlite3

from .bias import (BIAS_FEATURE_NAMES, BIAS_CONFIGS, DEFAULT_BIAS_CONFIG,
                   BiasPriorState, IntradayBiasState, bias_features, race_order)


BASE_FEATURE_NAMES = (
    "course_code", "surface_code", "sex_code", "frame", "horse_number", "age",
    "carried_weight_kg", "distance_m", "declared_field_size", "field_size", "month",
    "horse_prior_starts", "horse_prior_win_rate", "horse_prior_top3_rate",
    "horse_recent5_win_rate", "horse_recent5_top3_rate", "horse_recent5_mean_rank_fraction",
    "horse_previous_finish_position", "horse_previous_rank_fraction",
    "horse_days_since_last_start", "horse_distance_change_m", "horse_global_prior_starts",
    "horse_new_in_group", "horse_no_prior_history",
    "jockey_prior_starts", "jockey_prior_win_rate", "jockey_prior_top3_rate", "jockey_id_missing",
    "trainer_prior_starts", "trainer_prior_win_rate", "trainer_prior_top3_rate", "trainer_id_missing",
    "horse_top3_rate_vs_field", "horse_recent5_rank_vs_field",
)
FORM_FEATURE_NAMES = BASE_FEATURE_NAMES + (
    "horse_group_recent5_speed_mps", "horse_group_previous_speed_mps",
    "horse_group_recent5_winner_gap_per_km", "horse_group_previous_winner_gap_per_km",
    "horse_group_recent5_mean_time_gap_per_km", "horse_group_previous_mean_time_gap_per_km",
    "horse_group_recent5_last3f_seconds", "horse_group_previous_last3f_seconds",
    "horse_group_recent5_last3f_vs_mean", "horse_group_previous_last3f_vs_mean",
    "horse_group_recent5_timed_starts", "horse_group_recent5_section_timed_starts",
    "horse_course_surface_starts", "horse_course_surface_top3_rate",
    "horse_surface_distance_starts", "horse_surface_distance_top3_rate",
    "horse_course_surface_distance_starts", "horse_course_surface_distance_top3_rate",
)
RANK_FEATURE_NAMES = FORM_FEATURE_NAMES + (
    "horse_group_recent5_first_corner_fraction", "horse_group_previous_first_corner_fraction",
    "horse_group_recent5_last_corner_fraction", "horse_group_previous_last_corner_fraction",
    "horse_group_recent5_observed_corner_gain", "horse_group_previous_observed_corner_gain",
    "horse_group_recent5_third_to_fourth_gain", "horse_group_previous_third_to_fourth_gain",
    "horse_group_recent5_fourth_to_finish_gain", "horse_group_previous_fourth_to_finish_gain",
    "horse_group_recent5_corner_leading_rate", "horse_group_recent5_corner_front_rate",
    "horse_group_recent5_corner_starts", "opponents_recent5_corner_front_rate",
)
CONTEXT_FEATURE_NAMES = RANK_FEATURE_NAMES + (
    "race_class_code", "race_grade_code", "race_weight_rule_code",
    "race_age_min", "race_age_max", "race_first_prize_man",
    "horse_group_previous_class_code", "horse_group_previous_grade_code",
    "horse_group_previous_same_class", "horse_group_previous_same_grade",
    "horse_group_previous_bodyweight_kg", "horse_group_recent5_bodyweight_kg",
    "horse_group_previous_bodyweight_change_kg", "horse_group_recent5_absolute_weight_change_kg",
    "horse_group_previous_log_odds", "horse_group_recent5_log_odds",
    "horse_group_previous_popularity_fraction", "horse_group_recent5_popularity_fraction",
    "horse_group_previous_market_rank_gain", "horse_group_recent5_market_rank_gain",
    "horse_group_previous_weather_code", "horse_group_previous_going_code",
    "horse_group_previous_track_moisture", "horse_group_recent5_track_moisture",
    "horse_group_wet_starts", "horse_group_wet_top3_rate",
    "horse_group_dryish_starts", "horse_group_dryish_top3_rate",
    "horse_group_recent5_bodyweight_starts", "horse_group_recent5_odds_starts",
)
FEATURE_NAMES = CONTEXT_FEATURE_NAMES + BIAS_FEATURE_NAMES
CATEGORICAL_FEATURES = (
    "course_code", "surface_code", "sex_code", "race_class_code", "race_grade_code", "race_weight_rule_code",
    "horse_group_previous_class_code", "horse_group_previous_grade_code",
    "horse_group_previous_weather_code", "horse_group_previous_going_code",
)

# Fixed vocabulary, not fitted to the complete dataset or to future categories.
_COURSES = {
    name: number for number, name in enumerate((
        "札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉",
        "門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋",
        "園田", "姫路", "高知", "佐賀", "帯広(ば)",
    ))
}
_SEX = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}
_SURFACES = {"芝": 0, "ダ": 1, "ダート": 1, "障": 2, "障芝": 2, "障ダ": 2, "直": 3}
_CLASSES = {name: index for index, name in enumerate((
    "newcomer", "maiden", "1win", "2win", "3win", "open",
    "A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "A4", "B4", "C4",
))}
_GRADES = {name: index for index, name in enumerate((
    "G1", "G2", "G3", "Jpn1", "Jpn2", "Jpn3", "JG1", "JG2", "JG3", "L",
))}
_WEIGHT_RULES = {name: index for index, name in enumerate(("定量", "別定", "ハンデ", "馬齢"))}
_WEATHER = {name: index for index, name in enumerate(("晴", "曇", "雨", "小雨", "雪", "小雪", "霧"))}
_GOING = {name: index for index, name in enumerate(("良", "稍重", "重", "不良"))}
_GROUPS = {"central_flat", "central_jump", "local_flat", "local_banei"}
_GROUP_NAMES = {group: group for group in _GROUPS}
_WITHDRAWN = {"scratched", "excluded"}
_STARTED = {"finished", "did_not_finish", "disqualified"}


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _identifier(value) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _positive(value) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _nonnegative(value) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0 else None


def _same_known(current, previous) -> float | None:
    # These codes are categorical, with no shared ordinal class scale across
    # central, regional and banei racing. Equality is the only comparison.
    return float(current == previous) if current is not None and previous is not None else None


def _historical_context(row: dict, size: int, position: int | None) -> dict:
    """Read context of an observed earlier start, including its known market.

    A non-finisher may still have an observed pre-race weight, odds and going.
    Its popularity-to-finish gain is unknown rather than an invented last rank.
    """
    odds = _positive(row.get("odds"))
    popularity = _number(row.get("popularity"))
    if popularity is None or not popularity.is_integer() or not 1 <= popularity <= size:
        popularity = None
    denominator = max(size - 1, 1)
    moisture = _nonnegative(row.get("track_moisture"))
    if moisture is not None and moisture > 100:
        moisture = None
    going = {"稍": "稍重", "不": "不良"}.get(row.get("track_condition"), row.get("track_condition"))
    return dict(
        bodyweight_kg=_positive(row.get("weight_kg")),
        bodyweight_change_kg=_number(row.get("weight_change_kg")),
        log_odds=math.log(odds) if odds is not None else None,
        popularity_fraction=(popularity - 1) / denominator if popularity is not None else None,
        market_rank_gain=(popularity - position) / denominator if popularity is not None and position is not None else None,
        weather_code=_WEATHER.get(row.get("weather")), going_code=_GOING.get(going),
        track_moisture=moisture, class_code=_CLASSES.get(row.get("race_class")),
        grade_code=_GRADES.get(row.get("race_grade")),
    )


def _mean(values) -> float | None:
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def _course_code(row: dict) -> int | None:
    course = row.get("course")
    return _COURSES.get("帯広(ば)" if course == "帯広" else course)


def _aptitude_keys(row: dict, group: str) -> tuple[tuple | None, tuple | None, tuple | None]:
    """Group-specific course/surface/200m-band contexts; unknown flat surfaces stay unknown."""
    course = _course_code(row)
    if group == "local_banei":
        surface = 3  # Race domain, not an assertion of turf/dirt ground type.
    elif group == "central_jump":
        surface = 2
    else:
        surface = _SURFACES.get(row.get("surface"))
    distance = _positive(row.get("distance_m"))
    band = int(distance // 200) if distance is not None else None
    horse = _identifier(row.get("horse_id"))
    course_key = (group, horse, course, surface) if course is not None and surface is not None else None
    distance_key = (group, horse, surface, band) if surface is not None and band is not None else None
    combined_key = ((group, horse, course, surface, band)
                    if course_key is not None and band is not None else None)
    return course_key, distance_key, combined_key


def _corner_form(row: dict, size: int, finish_position: int | None) -> dict:
    """Summarize one completed historical run without inventing corner labels.

    NAR parallel groups use their supplied leading rank. It is a position in
    the field, not a claim that horses within the group have a known order.
    Unlabelled observations support first/last summaries only.
    """
    names = ("first_corner_fraction", "last_corner_fraction", "observed_corner_gain",
             "third_to_fourth_gain", "fourth_to_finish_gain", "corner_leading", "corner_front")
    missing = dict.fromkeys(names)
    if finish_position is None:
        return missing

    def valid_rank(value):
        rank = _number(value)
        return int(rank) if rank is not None and rank.is_integer() and 1 <= rank <= size else None

    positions = row.get("corner_positions")
    labelled = {}
    if isinstance(positions, dict) and positions:
        for label, value in positions.items():
            number = _number(label)
            rank = valid_rank(value)
            if number is None or not number.is_integer() or not 1 <= number <= 4 or rank is None:
                return missing
            if int(number) in labelled:
                return missing
            labelled[int(number)] = rank
        sequence = [labelled[label] for label in sorted(labelled)]
    else:
        observations = row.get("corner_sequence")
        if not isinstance(observations, list) or not observations:
            return missing
        sequence = [valid_rank(value) for value in observations]
        if any(value is None for value in sequence):
            return missing
        labels = row.get("corner_labels")
        if labels is not None:
            if not isinstance(labels, list) or len(labels) != len(sequence):
                return missing
            for label, value in zip(labels, sequence):
                number = _number(label)
                if number is None or not number.is_integer() or not 1 <= number <= 4 or int(number) in labelled:
                    return missing
                labelled[int(number)] = value
            # Explicit corner labels establish order; the input need not have
            # been serialized in corner-number order.
            sequence = [labelled[label] for label in sorted(labelled)]
    denominator = max(size - 1, 1)
    first = (sequence[0] - 1) / denominator
    return dict(
        first_corner_fraction=first,
        last_corner_fraction=(sequence[-1] - 1) / denominator,
        observed_corner_gain=(sequence[0] - sequence[-1]) / denominator if len(sequence) >= 2 else None,
        third_to_fourth_gain=(labelled[3] - labelled[4]) / denominator if 3 in labelled and 4 in labelled else None,
        fourth_to_finish_gain=(labelled[4] - finish_position) / denominator if 4 in labelled else None,
        corner_leading=float(sequence[0] == 1), corner_front=float(first <= .25),
    )


def _day(value) -> date:
    if not isinstance(value, str):
        raise ValueError("特徴量の開催日がありません。")
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("特徴量の開催日はYYYY-MM-DDで指定してください。")
    return result


def _context(race: list[dict]) -> tuple[date, str]:
    if not race:
        raise ValueError("空のレースから特徴量は作れません。")
    first = race[0]
    day, raw_group = _day(first.get("day")), first.get("group_name")
    group = _GROUP_NAMES.get(raw_group)
    if group is None:
        raise ValueError(f"未対応のモデル区分です: {raw_group}")
    keys = ("day", "region", "race_id", "group_name")
    if any(any(row.get(key) != first.get(key) for key in keys) for row in race):
        raise ValueError("異なる日付・地域・レース・区分の行が混在しています。")
    identifiers = [_identifier(row.get("horse_id")) for row in race]
    if any(identifier is None for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("馬IDが欠落または重複しています。馬名からの推定は行いません。")
    return day, group


def _outcome(row: dict) -> tuple[int | None, int, int]:
    status = row.get("finish_status")
    if status not in _STARTED:
        raise ValueError(f"学習に使えない着順状態です: {status}")
    if status != "finished":
        return None, 0, 0
    position = _number(row.get("finish_position"))
    if position is None or position <= 0 or not position.is_integer():
        raise ValueError("完走馬の確定順位が欠落または不正です。0補完は行いません。")
    rank = int(position)
    return rank, int(rank == 1), int(rank <= 3)


@dataclass(slots=True)
class _Aggregate:
    starts: int = 0
    wins: int = 0
    top3: int = 0

    def add(self, win: int, top3: int) -> None:
        self.starts += 1
        self.wins += win
        self.top3 += top3

    def features(self) -> tuple[float, float | None, float | None]:
        return (float(self.starts), self.wins / self.starts if self.starts else None,
                self.top3 / self.starts if self.starts else None)


@dataclass(frozen=True, slots=True)
class _Run:
    day: date
    key: tuple
    position: int | None
    win: int
    top3: int
    rank_fraction: float
    distance_m: float | None
    speed_mps: float | None = None
    winner_gap_per_km: float | None = None
    mean_time_gap_per_km: float | None = None
    last3f_seconds: float | None = None
    last3f_vs_mean: float | None = None
    first_corner_fraction: float | None = None
    last_corner_fraction: float | None = None
    observed_corner_gain: float | None = None
    third_to_fourth_gain: float | None = None
    fourth_to_finish_gain: float | None = None
    corner_leading: float | None = None
    corner_front: float | None = None
    bodyweight_kg: float | None = None
    bodyweight_change_kg: float | None = None
    log_odds: float | None = None
    popularity_fraction: float | None = None
    market_rank_gain: float | None = None
    weather_code: int | None = None
    going_code: int | None = None
    track_moisture: float | None = None
    class_code: int | None = None
    grade_code: int | None = None


def _append_recent(recent: deque, run: _Run) -> deque:
    # Normal input is chronological, so keep the common case O(1). The ID only
    # makes exceptional same-day ordering deterministic; it is not race time.
    if not recent or (recent[-1].day, recent[-1].key) <= (run.day, run.key):
        recent.append(run)
        return recent
    return deque(sorted((*recent, run), key=lambda item: (item.day, item.key))[-5:], maxlen=5)


@dataclass(slots=True)
class _Horse:
    starts: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=5))

    def add(self, run: _Run) -> None:
        self.starts += 1
        self.recent = _append_recent(self.recent, run)


class HistoryState:
    """Observed earlier-day outcomes; construct a fresh state for each cutoff."""

    def __init__(self):
        self.horses: dict[str, _Horse] = {}
        self.horse_groups: dict[tuple[str, str], _Aggregate] = {}
        self.group_recent: dict[tuple[str, str], deque] = {}
        self.course_surface: dict[tuple, _Aggregate] = {}
        self.surface_distance: dict[tuple, _Aggregate] = {}
        self.course_surface_distance: dict[tuple, _Aggregate] = {}
        self.going_results: dict[tuple[str, str, bool], _Aggregate] = {}
        self.jockeys: dict[tuple[str, str], _Aggregate] = {}
        self.trainers: dict[tuple[str, str], _Aggregate] = {}
        self._races: set[tuple] = set()
        self._latest_day: date | None = None

    @property
    def asof_day(self) -> str | None:
        return self._latest_day.isoformat() if self._latest_day else None

    def update_race(self, race: list[dict]) -> None:
        active = [row for row in race if row.get("finish_status") not in _WITHDRAWN]
        if not active:
            return
        day, group = _context(active)
        if self._latest_day is not None and day < self._latest_day:
            raise ValueError("履歴は開催日順に更新してください。")
        race_key = (active[0].get("region"), active[0].get("race_id"))
        if race_key in self._races:
            raise ValueError("同じレースの履歴を二重に更新しようとしました。")
        updates = []
        size = len(active)
        outcomes = [_outcome(row) for row in active]
        # These are outcomes of this historical race. They become usable only
        # after build_features has frozen every feature vector for this date.
        times = [_positive(row.get("time_seconds")) if result[0] is not None else None
                 for row, result in zip(active, outcomes)]
        sections = [_positive(row.get("last_3f_seconds")) if result[0] is not None else None
                    for row, result in zip(active, outcomes)]
        winner_times = [value for value, result in zip(times, outcomes) if result[0] == 1 and value is not None]
        winner_time = min(winner_times) if winner_times else None
        mean_time = _mean(times) if sum(value is not None for value in times) >= 2 else None
        mean_section = _mean(sections) if sum(value is not None for value in sections) >= 2 else None
        for row, result, time_value, section in zip(active, outcomes, times, sections):
            position, win, top3 = result
            if position is not None and position > size:
                raise ValueError("確定順位が実出走馬数を超えています。")
            fraction = (position - 1) / max(size - 1, 1) if position is not None else 1.0
            distance = _positive(row.get("distance_m"))
            run = _Run(day, (str(row.get("region")), str(row.get("race_id")),
                            _number(row.get("horse_number")) or 0),
                       position, win, top3, fraction, _number(row.get("distance_m")),
                       speed_mps=distance / time_value if distance is not None and time_value is not None else None,
                       winner_gap_per_km=((time_value - winner_time) * 1000 / distance
                                          if time_value is not None and winner_time is not None and distance is not None else None),
                       mean_time_gap_per_km=((time_value - mean_time) * 1000 / distance
                                             if time_value is not None and mean_time is not None and distance is not None else None),
                       last3f_seconds=section,
                       last3f_vs_mean=section - mean_section if section is not None and mean_section is not None else None,
                       **_corner_form(row, size, position), **_historical_context(row, size, position))
            updates.append((row, run))
        # Validate every outcome before mutating any aggregate.
        for row, run in updates:
            horse = _identifier(row["horse_id"])
            self.horses.setdefault(horse, _Horse()).add(run)
            self.horse_groups.setdefault((group, horse), _Aggregate()).add(run.win, run.top3)
            recent_key = (group, horse)
            recent = self.group_recent.get(recent_key)
            self.group_recent[recent_key] = _append_recent(recent if recent is not None else deque(maxlen=5), run)
            if run.going_code is not None:
                going_key = (group, horse, run.going_code in (_GOING["重"], _GOING["不良"]))
                self.going_results.setdefault(going_key, _Aggregate()).add(run.win, run.top3)
            for key, lookup in zip(_aptitude_keys(row, group),
                                   (self.course_surface, self.surface_distance, self.course_surface_distance)):
                if key is not None:
                    lookup.setdefault(key, _Aggregate()).add(run.win, run.top3)
            for field_name, lookup in (("jockey_id", self.jockeys), ("trainer_id", self.trainers)):
                identifier = _identifier(row.get(field_name))
                if identifier is not None:
                    lookup.setdefault((group, identifier), _Aggregate()).add(run.win, run.top3)
        self._races.add(race_key)
        self._latest_day = day


def _context_features(race: list[dict], histories: HistoryState) -> list[list[float | None]]:
    """Return one vector per supplied runner, preserving the supplied order.

    Callers filter known scratched/excluded runners before this function. Its
    current-race inputs never inspect outcomes, odds, weight or race-time data.
    """
    if not race:
        return []
    day, group = _context(race)
    if histories._latest_day is not None and histories._latest_day >= day:
        raise ValueError("予測日と同日以降の成績を含む履歴は使用できません。履歴を過去日だけで再構築してください。")
    vectors = []
    empty = _Aggregate()
    for row in race:
        identifier = _identifier(row["horse_id"])
        horse = histories.horses.get(identifier)
        horse_group = histories.horse_groups.get((group, identifier), empty)
        recent = list(horse.recent) if horse else []
        latest = recent[-1] if recent else None
        # Multiple starts on a day have no known order. Keep previous-day form
        # rates, but do not claim one specific finish/distance was the last one.
        previous = latest if latest and sum(run.day == latest.day for run in recent) == 1 else None
        distance = _number(row.get("distance_m"))
        sex, surface = row.get("sex"), row.get("surface")
        jockey_id, trainer_id = _identifier(row.get("jockey_id")), _identifier(row.get("trainer_id"))
        jockey = histories.jockeys.get((group, jockey_id), empty) if jockey_id else empty
        trainer = histories.trainers.get((group, trainer_id), empty) if trainer_id else empty
        vector = [
            _course_code(row), _SURFACES.get(surface), _SEX.get(sex),
            _number(row.get("frame")), _number(row.get("horse_number")), _number(row.get("age")),
            _number(row.get("carried_weight_kg")), distance, _number(row.get("declared_field_size")),
            float(len(race)), float(day.month), *horse_group.features(),
            sum(run.win for run in recent) / len(recent) if recent else None,
            sum(run.top3 for run in recent) / len(recent) if recent else None,
            sum(run.rank_fraction for run in recent) / len(recent) if recent else None,
            float(previous.position) if previous and previous.position is not None else None,
            previous.rank_fraction if previous else None,
            float((day - latest.day).days) if latest else None,
            distance - previous.distance_m if previous and distance is not None and previous.distance_m is not None else None,
            float(horse.starts) if horse else 0.0, float(horse_group.starts == 0), float(horse is None),
            *jockey.features(), float(jockey_id is None),
            *trainer.features(), float(trainer_id is None), None, None,
        ]
        timed_recent = list(histories.group_recent.get((group, identifier), ()))
        group_latest = timed_recent[-1] if timed_recent else None
        group_previous = (group_latest if group_latest and
                          sum(run.day == group_latest.day for run in timed_recent) == 1 else None)
        for name in ("speed_mps", "winner_gap_per_km", "mean_time_gap_per_km",
                     "last3f_seconds", "last3f_vs_mean"):
            vector.extend((_mean(getattr(run, name) for run in timed_recent),
                           getattr(group_previous, name) if group_previous else None))
        vector.extend((float(sum(run.speed_mps is not None for run in timed_recent)),
                       float(sum(run.last3f_seconds is not None for run in timed_recent))))
        for key, lookup in zip(_aptitude_keys(row, group),
                               (histories.course_surface, histories.surface_distance, histories.course_surface_distance)):
            aggregate = lookup.get(key, empty) if key is not None else empty
            vector.extend((float(aggregate.starts), aggregate.top3 / aggregate.starts if aggregate.starts else None))
        for name in ("first_corner_fraction", "last_corner_fraction", "observed_corner_gain",
                     "third_to_fourth_gain", "fourth_to_finish_gain"):
            vector.extend((_mean(getattr(run, name) for run in timed_recent),
                           getattr(group_previous, name) if group_previous else None))
        vector.extend((_mean(run.corner_leading for run in timed_recent),
                       _mean(run.corner_front for run in timed_recent),
                       float(sum(run.first_corner_fraction is not None for run in timed_recent)), None))
        class_code = _CLASSES.get(row.get("race_class"))
        grade_code = _GRADES.get(row.get("race_grade"))
        previous_class = group_previous.class_code if group_previous else None
        previous_grade = group_previous.grade_code if group_previous else None
        wet = histories.going_results.get((group, identifier, True), empty)
        dryish = histories.going_results.get((group, identifier, False), empty)
        vector.extend((
            class_code, grade_code, _WEIGHT_RULES.get(row.get("weight_rule")),
            _positive(row.get("age_min")), _positive(row.get("age_max")), _nonnegative(row.get("first_prize_man")),
            previous_class, previous_grade, _same_known(class_code, previous_class), _same_known(grade_code, previous_grade),
            group_previous.bodyweight_kg if group_previous else None, _mean(run.bodyweight_kg for run in timed_recent),
            group_previous.bodyweight_change_kg if group_previous else None,
            _mean(abs(run.bodyweight_change_kg) if run.bodyweight_change_kg is not None else None for run in timed_recent),
            group_previous.log_odds if group_previous else None, _mean(run.log_odds for run in timed_recent),
            group_previous.popularity_fraction if group_previous else None, _mean(run.popularity_fraction for run in timed_recent),
            group_previous.market_rank_gain if group_previous else None, _mean(run.market_rank_gain for run in timed_recent),
            group_previous.weather_code if group_previous else None, group_previous.going_code if group_previous else None,
            group_previous.track_moisture if group_previous else None, _mean(run.track_moisture for run in timed_recent),
            float(wet.starts), wet.top3 / wet.starts if wet.starts else None,
            float(dryish.starts), dryish.top3 / dryish.starts if dryish.starts else None,
            float(sum(run.bodyweight_kg is not None for run in timed_recent)),
            float(sum(run.log_odds is not None for run in timed_recent)),
        ))
        vectors.append(vector)
    # Relative form uses only opponents' already-causal histories.
    for source, destination in (("horse_prior_top3_rate", "horse_top3_rate_vs_field"),
                                ("horse_recent5_mean_rank_fraction", "horse_recent5_rank_vs_field")):
        source_index, destination_index = FEATURE_NAMES.index(source), FEATURE_NAMES.index(destination)
        values = [vector[source_index] for vector in vectors if vector[source_index] is not None]
        mean = sum(values) / len(values) if values else None
        for vector in vectors:
            if mean is not None and vector[source_index] is not None:
                vector[destination_index] = vector[source_index] - mean
    front_index = FEATURE_NAMES.index("horse_group_recent5_corner_front_rate")
    opponent_index = FEATURE_NAMES.index("opponents_recent5_corner_front_rate")
    for index, vector in enumerate(vectors):
        vector[opponent_index] = _mean(other[front_index] for number, other in enumerate(vectors) if number != index)
    return vectors


def race_features(race: list[dict], histories: HistoryState, *, intraday=None, priors=None,
                  full_race=None, bias_config=DEFAULT_BIAS_CONFIG) -> list[list[float | None]]:
    """Combine previous-day form with a separate, earlier-race bias state.

    The historical state retains its same-day guard.  The bias state never
    updates it, and a caller with no live results receives prior-only values.
    ``full_race`` retains withdrawn horse numbers for gate normalization.
    """
    if not race:
        return []
    vectors = _context_features(race, histories)
    day = race[0]["day"]
    intraday = IntradayBiasState(day) if intraday is None else intraday
    priors = BiasPriorState() if priors is None else priors
    additions = bias_features(race, intraday, priors, form_vectors=vectors,
                              config=bias_config, full_race=full_race)
    return [vector + addition for vector, addition in zip(vectors, additions)]


def build_features(database: Path, report=None) -> dict:
    """Stream races through daily frozen histories into the separate AI cache."""
    from .dataset import iter_races

    database = Path(database)
    if not database.is_file():
        raise ValueError(f"AI用データベースがありません: {database}")
    emit = report or (lambda **payload: None)
    connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""CREATE TABLE IF NOT EXISTS features (
            region TEXT NOT NULL, race_id TEXT NOT NULL, day TEXT NOT NULL,
            horse_id TEXT NOT NULL, horse_number INTEGER, group_name TEXT NOT NULL,
            target INTEGER NOT NULL, feature_json TEXT NOT NULL,
            PRIMARY KEY(region,race_id,horse_id)
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS bias_variants (
            region TEXT NOT NULL, race_id TEXT NOT NULL, horse_id TEXT NOT NULL,
            config_id TEXT NOT NULL, feature_json TEXT NOT NULL,
            PRIMARY KEY(region,race_id,horse_id,config_id)
        )""")
        connection.execute("CREATE INDEX IF NOT EXISTS bias_variants_config "
                           "ON bias_variants(config_id,region,race_id,horse_id)")
        with connection:
            connection.execute("DELETE FROM features")
            connection.execute("DELETE FROM bias_variants")
        histories = HistoryState()
        priors = BiasPriorState()
        rows_count = races_count = days_count = skipped = 0
        bias_races = bias_rows = 0
        source_bias_races = 0
        bias_skips = defaultdict(int)
        start = finish = None
        groups = defaultdict(lambda: {"rows": 0, "races": 0, "positives": 0})
        for day, iterator in groupby(iter_races(connection), key=lambda race: race[0]["day"]):
            daily_races = []
            inserts = []
            variants = []
            intraday = IntradayBiasState(day)
            # Results only become available to a later race at the same venue.
            # Race numbers encode scheduled order; historical completion and
            # publication timestamps are unavailable in the result archive.
            def order(race):
                first = race[0]
                number = race_order(race)
                return (first["region"], first.get("course") or "", first.get("surface") or "",
                        number if number is not None else 1000, first["race_id"])

            for race in sorted(iterator, key=order):
                active = [row for row in race if row.get("finish_status") not in _WITHDRAWN]
                if not active:
                    skipped += 1
                    continue
                vectors = race_features(active, histories, intraday=intraday, priors=priors,
                                        full_race=race)
                context = [vector[:len(CONTEXT_FEATURE_NAMES)] for vector in vectors]
                usable = vectors[0][FEATURE_NAMES.index("day_bias_races")]
                if usable is not None and usable > 0:
                    bias_races += 1
                    bias_rows += len(active)
                for config in BIAS_CONFIGS:
                    additions = ([vector[len(CONTEXT_FEATURE_NAMES):] for vector in vectors]
                                 if config == DEFAULT_BIAS_CONFIG else
                                 bias_features(active, intraday, priors, form_vectors=context,
                                               config=config, full_race=race))
                    variants.extend((row["region"], row["race_id"], row["horse_id"], config.identifier,
                                     json.dumps(addition, allow_nan=False, separators=(",", ":")))
                                    for row, addition in zip(active, additions))
                for row, vector in zip(active, vectors):
                    _, _, target = _outcome(row)
                    inserts.append((row["region"], row["race_id"], day, row["horse_id"],
                                    row.get("horse_number"), row["group_name"], target,
                                    json.dumps(vector, allow_nan=False, separators=(",", ":"))))
                daily_races.append(active)
                # This includes the original scratched/excluded slots only to
                # preserve gate positions. The bias estimator filters outcomes.
                source_bias_races += int(intraday.add_result(race))
            for reason, count in intraday.skipped_reasons.items():
                bias_skips[reason] += count
            with connection:
                connection.executemany("INSERT INTO features VALUES (?,?,?,?,?,?,?,?)", inserts)
                connection.executemany("INSERT INTO bias_variants VALUES (?,?,?,?,?)", variants)
                # No outcome from this date is made available until all its
                # feature vectors have been computed, including other groups.
                for race in daily_races:
                    histories.update_race(race)
                priors.update_day(intraday)
            for row in inserts:
                group = groups[row[5]]
                group["rows"] += 1
                group["positives"] += row[6]
            for race in daily_races:
                groups[race[0]["group_name"]]["races"] += 1
            rows_count += len(inserts)
            races_count += len(daily_races)
            days_count += 1
            start = day if start is None else start
            finish = day
            emit(event="features_day", day=day, rows=len(inserts), races=len(daily_races), total_rows=rows_count)
        if not rows_count:
            raise ValueError("学習可能な出走行がありません。入力品質の診断を確認してください。")
        return {"rows": rows_count, "races": races_count, "days": days_count,
                "start_date": start, "end_date": finish, "groups": dict(groups),
                "skipped_inactive_races": skipped, "feature_names": list(FEATURE_NAMES),
                "categorical_features": list(CATEGORICAL_FEATURES),
                "bias_configs": [dict(identifier=config.identifier, **config.as_dict()) for config in BIAS_CONFIGS],
                "default_bias_config": dict(identifier=DEFAULT_BIAS_CONFIG.identifier, **DEFAULT_BIAS_CONFIG.as_dict()),
                "races_with_intraday_bias": bias_races, "rows_with_intraday_bias": bias_rows,
                "intraday_source_races": source_bias_races, "intraday_skipped_reasons": dict(bias_skips),
                "intraday_availability_assumption": "Same venue/date/surface and a smaller verified race number; "
                    "historical completion/publication timestamps are unavailable.",
                "target_track_condition_used": False}
    finally:
        connection.close()
