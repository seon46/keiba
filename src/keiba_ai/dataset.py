"""Validate exported result CSVs and stage them without loading the dataset."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
import csv
from datetime import date
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import sqlite3


RESULT_HEADERS = {
    "race_id", "region", "page", "race_date", "horse_id", "horse_number",
    "finish_position", "finish_status", "race_course", "race_distance_m",
}
STATUSES = {"finished", "scratched", "excluded", "did_not_finish", "disqualified"}
GROUPS = {"central_flat", "central_jump", "local_flat", "local_banei"}
_NULLS = {"", "null", "none"}
_NUMBER = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?")
_HORSE = re.compile(r"(?:[0-9]{10}|B[0-9]{9})")


class RowValidationError(ValueError):
    """A row is invalid but its containing race can still be identified."""


def _text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.casefold() in _NULLS else text


def _required(value, name: str) -> str:
    text = _text(value)
    if text is None:
        raise RowValidationError(f"{name}: required value is missing")
    return text


def _day(value, name="race_date") -> str:
    text = _required(value, name)
    try:
        if date.fromisoformat(text).isoformat() != text:
            raise ValueError(text)
    except ValueError as exc:
        raise RowValidationError(f"{name}: expected YYYY-MM-DD") from exc
    return text


def _numeric(value, name: str, *, integer=False, required=False, positive=False):
    text = _text(value)
    if text is None:
        if required:
            raise RowValidationError(f"{name}: required number is missing")
        return None
    if isinstance(value, bool) or not _NUMBER.fullmatch(text):
        raise RowValidationError(f"{name}: expected a finite number")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise RowValidationError(f"{name}: invalid number") from exc
    if not number.is_finite() or (positive and number <= 0):
        raise RowValidationError(f"{name}: expected a finite positive number")
    if integer:
        if number != number.to_integral_value():
            raise RowValidationError(f"{name}: expected an integer")
        return int(number)
    result = float(number)
    if result in (float("inf"), float("-inf")) or (positive and result <= 0):
        raise RowValidationError(f"{name}: number is outside the supported range")
    return result


def _boolean(value, name: str) -> bool | None:
    text = _text(value)
    if text is None:
        return None
    values = {"true": True, "false": False, "1": True, "0": False}
    if text.casefold() not in values:
        raise RowValidationError(f"{name}: expected True or False")
    return values[text.casefold()]


def _identity(raw: Mapping) -> tuple[str, str]:
    region, identifier = _text(raw.get("region")), _text(raw.get("race_id"))
    if region not in ("central", "local") or not identifier or not re.fullmatch(r"[0-9]{12}", identifier):
        raise ValueError("Cannot identify a domestic race: require region=central/local and a 12-digit race_id")
    return region, identifier


def normalize_row(raw: Mapping, page: str = "results") -> dict:
    """Normalize one exported row; malformed values are never guessed.

    Entries may omit result labels. Result errors are raised so ingestion can
    quarantine the entire race, rather than train on its remaining horses.
    """
    if page not in ("results", "entries"):
        raise ValueError("page must be results or entries")
    region, identifier = _identity(raw)
    if _text(raw.get("page", page)) != page:
        raise RowValidationError(f"page: expected {page}")
    if "race_id_verified" in raw and _boolean(raw["race_id_verified"], "race_id_verified") is False:
        raise RowValidationError("race_id_verified: source identity is unverified")
    horse_id = _required(raw.get("horse_id"), "horse_id")
    if not _HORSE.fullmatch(horse_id):
        raise RowValidationError("horse_id: expected 10 digits or B followed by 9 digits")
    race_number = int(identifier[-2:])
    if not race_number:
        raise RowValidationError("race_id: race-number suffix must be positive")
    supplied_number = _numeric(raw.get("race_number"), "race_number", integer=True, positive=True)
    if supplied_number is not None and supplied_number != race_number:
        raise RowValidationError("race_number: conflicts with the race_id suffix")
    day = _day(raw.get("race_date"))
    course = _required(raw.get("race_course"), "race_course")
    surface = _text(raw.get("race_surface"))
    surface = {"ダート": "ダ"}.get(surface, surface)
    race_type = _text(raw.get("race_type"))
    if race_type not in (None, "banei"):
        raise RowValidationError("race_type: unsupported race type")
    banei_course = "帯広" in course or "ばんえい" in course
    if race_type == "banei" or banei_course:
        if region != "local" or not banei_course:
            raise RowValidationError("race_type: banei conflicts with region/course")
        if surface not in (None, "直"):
            raise RowValidationError("race_surface: banei must not use a flat/jump surface")
        group, surface = "local_banei", None
    elif surface in ("芝", "ダ"):
        group = region + "_flat"
    elif region == "central" and surface in ("障", "障芝", "障ダ"):
        group = "central_jump"
    else:
        raise RowValidationError("race_surface: missing or unsupported surface")
    distance = _numeric(raw.get("race_distance_m"), "race_distance_m", integer=True,
                        required=True, positive=True)
    if group == "local_banei" and distance != 200:
        raise RowValidationError("race_distance_m: supported banei races are 200 metres")
    status = _text(raw.get("finish_status"))
    position = _numeric(raw.get("finish_position"), "finish_position", integer=True, positive=True)
    if status not in STATUSES and not (page == "entries" and status is None):
        raise RowValidationError("finish_status: missing, unknown, abandoned, or unsupported result")
    if page == "entries" and (status not in (None, "scratched", "excluded") or position is not None):
        raise RowValidationError("entries must not contain finish ranks or post-race result statuses")
    if page == "results" and status == "finished" and position is None:
        raise RowValidationError("finish_position: finished runner must have a positive integer rank")
    if status != "finished" and position is not None:
        raise RowValidationError("finish_position: non-finisher must not have a numeric rank")
    if _text(raw.get("race_status")) not in (None, "held"):
        raise RowValidationError("race_status: cancelled/unsupported races cannot contain runner rows")
    sex = _text(raw.get("sex"))
    if sex not in (None, "牡", "牝", "セ", "騸"):
        raise RowValidationError("sex: unsupported value")
    identifiers = {}
    for key in ("jockey_id", "trainer_id"):
        value = _text(raw.get(key))
        if value is not None and not re.fullmatch(r"[A-Za-z0-9]+", value):
            raise RowValidationError(f"{key}: expected an alphanumeric identifier")
        identifiers[key] = value
    horse_number = _numeric(raw.get("horse_number"), "horse_number", integer=True, required=True, positive=True)
    if horse_number > 99:
        raise RowValidationError("horse_number: expected a domestic number from 1 to 99")
    # These are outcomes, never attributes of an entries row. Non-completions
    # can contain provisional times which must not enter performance history.
    completed = page == "results" and status == "finished"
    times = {
        key: _numeric(raw.get(key), key, positive=True) if completed else None
        for key in ("time_seconds", "last_3f_seconds")
    }
    return {
        "race_id": identifier, "region": region, "day": day, "horse_id": horse_id,
        "horse_number": horse_number,
        "frame": _numeric(raw.get("frame"), "frame", integer=True, positive=True),
        "sex": "セ" if sex == "騸" else sex,
        "age": _numeric(raw.get("age"), "age", integer=True, positive=True),
        **identifiers,
        "carried_weight_kg": _numeric(raw.get("carried_weight_kg"), "carried_weight_kg", positive=True),
        "finish_position": position, "finish_status": status, "course": course,
        **times,
        "surface": surface, "distance_m": distance,
        "declared_field_size": _numeric(raw.get("race_declared_field_size"), "race_declared_field_size",
                                         integer=True, positive=True),
        "group_name": group,
        "odds": _numeric(raw.get("odds"), "odds", positive=True),
        "popularity": _numeric(raw.get("popularity"), "popularity", integer=True, positive=True),
        "snapshot_id": _text(raw.get("snapshot_id")),
    }


def _race_problems(rows: list[dict]) -> list[str]:
    reasons = []
    for key in ("day", "snapshot_id", "course", "surface", "distance_m", "declared_field_size", "group_name"):
        if len({row[key] for row in rows}) != 1:
            reasons.append(f"inconsistent race field: {key}")
    numbers = [row["horse_number"] for row in rows]
    if len(numbers) != len(set(numbers)):
        reasons.append("duplicate horse_number within race")
    sizes = {row["declared_field_size"] for row in rows if row["declared_field_size"] is not None}
    if sizes and sizes != {len(rows)}:
        reasons.append("declared_field_size does not match all CSV rows, including scratched/excluded")
    active = sum(row["finish_status"] not in ("scratched", "excluded") for row in rows)
    if any(row["finish_position"] is not None and row["finish_position"] > active for row in rows):
        reasons.append("finish_position exceeds the active race field size")
    if not any(row["finish_status"] == "finished" and row["finish_position"] == 1 for row in rows):
        reasons.append("race has no finished runner in first place")
    return reasons


def ingest_csv(source: Path, database: Path, report=None) -> dict:
    """Stream a results CSV into a new SQLite DB, quarantining invalid races.

    Duplicate runner keys and unidentifiable races abort the whole import.
    The caller owns publication of the completed staging database.
    """
    source, database = Path(source).resolve(), Path(database).resolve()
    if source == database:
        raise ValueError("source CSV and destination database must differ")
    if database.exists():
        raise FileExistsError(f"Destination database already exists: {database}")
    database.parent.mkdir(parents=True, exist_ok=True)
    db = None
    try:
        db = sqlite3.connect(database)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE input_rows (
                region TEXT NOT NULL,race_id TEXT NOT NULL,day TEXT NOT NULL,
                horse_id TEXT NOT NULL,horse_number INTEGER NOT NULL,group_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,PRIMARY KEY(region,race_id,horse_id)
            );
            CREATE TABLE race_issues (
                region TEXT NOT NULL,race_id TEXT NOT NULL,reason TEXT NOT NULL,
                PRIMARY KEY(region,race_id,reason)
            );
            CREATE TEMP TABLE seen_keys (
                region TEXT,race_id TEXT,horse_id TEXT,PRIMARY KEY(region,race_id,horse_id)
            );
            CREATE TEMP TABLE source_races (
                region TEXT,race_id TEXT,PRIMARY KEY(region,race_id)
            );
        """)
        total, imported = 0, 0
        regions, groups, statuses, missing = Counter(), Counter(), Counter(), Counter()
        batch, issues, seen, races = [], [], [], set()

        def flush():
            try:
                db.executemany("INSERT INTO seen_keys VALUES (?,?,?)", seen)
                db.executemany("INSERT INTO input_rows VALUES (?,?,?,?,?,?,?)", batch)
            except sqlite3.IntegrityError as exc:
                raise ValueError("Duplicate (region,race_id,horse_id): export only one latest snapshot per race") from exc
            db.executemany("INSERT OR IGNORE INTO race_issues VALUES (?,?,?)", issues)
            db.executemany("INSERT OR IGNORE INTO source_races VALUES (?,?)", races)
            batch.clear(); issues.clear(); seen.clear(); races.clear()

        db.execute("BEGIN")
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            header = reader.fieldnames
            if header is None or not RESULT_HEADERS <= set(header):
                absent = sorted(RESULT_HEADERS - set(header or []))
                raise ValueError("Missing required CSV columns: " + ", ".join(absent))
            if len(header) != len(set(header)):
                raise ValueError("Duplicate CSV column names")
            for raw in reader:
                total += 1
                region, identifier = _identity(raw)
                regions[region] += 1
                races.add((region, identifier))
                raw_status = _text(raw.get("finish_status"))
                statuses[raw_status if raw_status in STATUSES | {"unknown", "abandoned"} else "<missing/invalid>"] += 1
                horse = _text(raw.get("horse_id"))
                if horse and _HORSE.fullmatch(horse):
                    seen.append((region, identifier, horse))
                try:
                    if None in raw:
                        raise RowValidationError("CSV row has more cells than its header")
                    normalized = normalize_row(raw)
                except RowValidationError as exc:
                    issues.append((region, identifier, str(exc)))
                else:
                    imported += 1
                    groups[normalized["group_name"]] += 1
                    for key, value in normalized.items():
                        if value is None:
                            missing[key] += 1
                    batch.append((region, identifier, normalized["day"], normalized["horse_id"],
                                  normalized["horse_number"], normalized["group_name"],
                                  json.dumps(normalized, ensure_ascii=False, allow_nan=False)))
                if total % 1000 == 0:
                    flush()
                    if report:
                        report(event="ingest_progress", rows=total, imported_rows=imported)
            flush()
        db.execute("CREATE INDEX input_rows_race ON input_rows(region,race_id,horse_number)")
        db.execute("CREATE INDEX input_rows_day ON input_rows(day,region,race_id,horse_number)")
        # Quarantine repeated horse numbers in SQL before collecting one race.
        # Together with the number bound this keeps each validation group small,
        # even when a malformed CSV repeats a race millions of times.
        db.execute("INSERT OR IGNORE INTO race_issues SELECT region,race_id,"
                   "'duplicate horse_number within race' FROM input_rows "
                   "GROUP BY region,race_id HAVING COUNT(*)!=COUNT(DISTINCT horse_number)")
        cursor = db.execute(
            "SELECT region,race_id,payload_json FROM input_rows i WHERE NOT EXISTS "
            "(SELECT 1 FROM race_issues e WHERE e.region=i.region AND e.race_id=i.race_id) "
            "ORDER BY region,race_id,horse_number")
        active, rows = None, []

        def validate(key, values):
            if key is not None:
                db.executemany("INSERT OR IGNORE INTO race_issues VALUES (?,?,?)",
                               ((key[0], key[1], reason) for reason in _race_problems(values)))

        for row in cursor:
            key = row["region"], row["race_id"]
            if active is not None and key != active:
                validate(active, rows)
                rows = []
            active = key
            rows.append(json.loads(row["payload_json"]))
        validate(active, rows)
        cursor.close()
        usable = db.execute(
            "SELECT COUNT(*) FROM input_rows i WHERE NOT EXISTS "
            "(SELECT 1 FROM race_issues e WHERE e.region=i.region AND e.race_id=i.race_id)").fetchone()[0]
        usable_races = db.execute(
            "SELECT COUNT(*) FROM (SELECT region,race_id FROM input_rows i WHERE NOT EXISTS "
            "(SELECT 1 FROM race_issues e WHERE e.region=i.region AND e.race_id=i.race_id) "
            "GROUP BY region,race_id)").fetchone()[0]
        race_count = db.execute("SELECT COUNT(*) FROM source_races").fetchone()[0]
        excluded = []
        for row in db.execute("SELECT region,race_id,GROUP_CONCAT(reason,char(10)) AS reasons "
                              "FROM race_issues GROUP BY region,race_id ORDER BY region,race_id LIMIT 30"):
            excluded.append({"region": row["region"], "race_id": row["race_id"],
                             "reasons": row["reasons"].split("\n")})
        period = db.execute("SELECT MIN(day),MAX(day),COUNT(DISTINCT day) FROM input_rows").fetchone()
        month_group_counts = [dict(row) for row in db.execute(
            "SELECT substr(day,1,7) AS month,group_name,COUNT(*) AS rows,"
            "COUNT(DISTINCT region||':'||race_id) AS races FROM input_rows i WHERE NOT EXISTS "
            "(SELECT 1 FROM race_issues e WHERE e.region=i.region AND e.race_id=i.race_id) "
            "GROUP BY substr(day,1,7),group_name ORDER BY month,group_name")]
        result = {
            "source": str(source), "database": str(database), "input_rows": total,
            "imported_rows": imported, "invalid_rows": total-imported,
            "usable_rows": usable, "excluded_rows": total-usable,
            "races": race_count, "usable_races": usable_races,
            "excluded_race_count": race_count-usable_races,
            "excluded_races": excluded, "excluded_races_truncated": race_count-usable_races > 30,
            "period": {"start_date": period[0], "end_date": period[1], "days": period[2]},
            "regions": dict(regions), "groups": dict(groups), "finish_status": dict(statuses),
            "missing_values": dict(missing), "month_group_counts": month_group_counts,
        }
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return result
    except BaseException:
        if db is not None:
            db.rollback()
            db.close()
            db = None
        # Only this call's new staging files are removed; existing files were refused.
        for path in (database, Path(str(database)+"-wal"), Path(str(database)+"-shm")):
            path.unlink(missing_ok=True)
        raise
    finally:
        if db is not None:
            db.close()


def iter_races(database: Path | sqlite3.Connection, before_date: str | None = None) -> Iterator[list[dict]]:
    """Yield usable races in day order; before_date excludes the specified day."""
    if before_date is not None:
        before_date = _day(before_date, "before_date")
    borrowed = isinstance(database, sqlite3.Connection)
    db = database if borrowed else sqlite3.connect(Path(database).resolve().as_uri()+"?mode=ro", uri=True)
    arguments = () if before_date is None else (before_date,)
    cursor = None
    try:
        cursor = db.execute(
            "SELECT i.day,i.region,i.race_id,i.payload_json FROM input_rows i WHERE NOT EXISTS "
            "(SELECT 1 FROM race_issues e WHERE e.region=i.region AND e.race_id=i.race_id) "
            + ("AND i.day<? " if before_date is not None else "")
            + "ORDER BY i.day,i.region,i.race_id,i.horse_number", arguments)
        active, rows = None, []
        for row in cursor:
            key = row[0], row[1], row[2]
            if active is not None and key != active:
                yield rows
                rows = []
            active = key
            rows.append(json.loads(row[3]))
        if rows:
            yield rows
    finally:
        if cursor is not None:
            cursor.close()
        if not borrowed:
            db.close()
