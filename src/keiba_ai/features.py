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


FEATURE_NAMES = (
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
    day, group = _day(first.get("day")), first.get("group_name")
    if group not in _GROUPS:
        raise ValueError(f"未対応のモデル区分です: {group}")
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


@dataclass
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


@dataclass(frozen=True)
class _Run:
    day: date
    key: tuple
    position: int | None
    win: int
    top3: int
    rank_fraction: float
    distance_m: float | None


@dataclass
class _Horse:
    starts: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=5))

    def add(self, run: _Run) -> None:
        self.starts += 1
        # ID is only a deterministic tie breaker for rare same-day records,
        # never evidence of their chronological race order.
        self.recent = deque(sorted((*self.recent, run), key=lambda item: (item.day, item.key))[-5:], maxlen=5)


class HistoryState:
    """Observed earlier-day outcomes; construct a fresh state for each cutoff."""

    def __init__(self):
        self.horses: dict[str, _Horse] = {}
        self.horse_groups: dict[tuple[str, str], _Aggregate] = {}
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
        for row in active:
            position, win, top3 = _outcome(row)
            if position is not None and position > size:
                raise ValueError("確定順位が実出走馬数を超えています。")
            fraction = (position - 1) / max(size - 1, 1) if position is not None else 1.0
            run = _Run(day, (str(row.get("region")), str(row.get("race_id")),
                            _number(row.get("horse_number")) or 0),
                       position, win, top3, fraction, _number(row.get("distance_m")))
            updates.append((row, run))
        # Validate every outcome before mutating any aggregate.
        for row, run in updates:
            horse = _identifier(row["horse_id"])
            self.horses.setdefault(horse, _Horse()).add(run)
            self.horse_groups.setdefault((group, horse), _Aggregate()).add(run.win, run.top3)
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
        course = row.get("course")
        if course == "帯広":
            course = "帯広(ば)"
        sex, surface = row.get("sex"), row.get("surface")
        jockey_id, trainer_id = _identifier(row.get("jockey_id")), _identifier(row.get("trainer_id"))
        jockey = histories.jockeys.get((group, jockey_id), empty) if jockey_id else empty
        trainer = histories.trainers.get((group, trainer_id), empty) if trainer_id else empty
        vector = [
            _COURSES.get(course), _SURFACES.get(surface), _SEX.get(sex),
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
