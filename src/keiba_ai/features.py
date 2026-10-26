"""Causal, day-delayed features shared by offline training and prediction.

Only declared race/runner attributes and outcomes from earlier calendar days
are used. IDs are opaque lookup keys, never numeric model inputs. Horse recent
form spans groups; horse/jockey/trainer career rates are specific to the model
group so that JRA, regional flat racing and banei are not pooled indiscriminately.
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
FEATURE_NAMES = BASE_FEATURE_NAMES + (
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
CATEGORICAL_FEATURES = ("course_code", "surface_code", "sex_code")

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
                       last3f_vs_mean=section - mean_section if section is not None and mean_section is not None else None)
            updates.append((row, run))
        # Validate every outcome before mutating any aggregate.
        for row, run in updates:
            horse = _identifier(row["horse_id"])
            self.horses.setdefault(horse, _Horse()).add(run)
            self.horse_groups.setdefault((group, horse), _Aggregate()).add(run.win, run.top3)
            recent_key = (group, horse)
            recent = self.group_recent.get(recent_key)
            self.group_recent[recent_key] = _append_recent(recent if recent is not None else deque(maxlen=5), run)
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


def race_features(race: list[dict], histories: HistoryState) -> list[list[float | None]]:
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
    return vectors


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
        with connection:
            connection.execute("DELETE FROM features")
        histories = HistoryState()
        rows_count = races_count = days_count = skipped = 0
        start = finish = None
        groups = defaultdict(lambda: {"rows": 0, "races": 0, "positives": 0})
        for day, iterator in groupby(iter_races(connection), key=lambda race: race[0]["day"]):
            daily_races = []
            inserts = []
            for race in iterator:
                active = [row for row in race if row.get("finish_status") not in _WITHDRAWN]
                if not active:
                    skipped += 1
                    continue
                vectors = race_features(active, histories)
                for row, vector in zip(active, vectors):
                    _, _, target = _outcome(row)
                    inserts.append((row["region"], row["race_id"], day, row["horse_id"],
                                    row.get("horse_number"), row["group_name"], target,
                                    json.dumps(vector, allow_nan=False, separators=(",", ":"))))
                daily_races.append(active)
            with connection:
                connection.executemany("INSERT INTO features VALUES (?,?,?,?,?,?,?,?)", inserts)
                # No outcome from this date is made available until all its
                # feature vectors have been computed, including other groups.
                for race in daily_races:
                    histories.update_race(race)
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
                "categorical_features": list(CATEGORICAL_FEATURES)}
    finally:
        connection.close()
