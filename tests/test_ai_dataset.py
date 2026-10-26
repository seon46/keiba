import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from keiba_ai.dataset import ingest_csv, iter_races, normalize_row


def row(**changes):
    result = {
        "race_id": "202405010101", "region": "central", "page": "results",
        "race_date": "2024-01-01", "race_number": "1", "race_course": "東京",
        "race_surface": "芝", "race_distance_m": "1600", "race_type": "",
        "race_declared_field_size": "2", "race_id_verified": "True",
        "horse_id": "2021100001", "horse_number": "1", "frame": "1",
        "sex": "牡", "age": "3", "jockey_id": "00123", "trainer_id": "a024c",
        "carried_weight_kg": "57.0", "finish_position": "1", "finish_status": "finished",
        "odds": "2.5", "popularity": "1", "snapshot_id": "snapshot-one",
    }
    result.update(changes)
    return result


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.csv = self.directory / "runners.csv"
        self.db = self.directory / "learning.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows, *, encoding="utf-8", headers=None):
        fields = headers or list(dict.fromkeys(key for item in rows for key in item))
        with self.csv.open("w", encoding=encoding, newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def pair(self, **changes):
        return [row(**changes), row(horse_id="2021100002", horse_number="2", finish_position="2", **changes)]

    def test_normalization_preserves_identifiers_and_strict_types(self):
        normalized = normalize_row(row(jockey_id="a03fd", sex="騸", frame="1.0", age="3.0",
                                       odds="null", popularity="None"))
        self.assertEqual(normalized["jockey_id"], "a03fd")
        self.assertEqual(normalized["trainer_id"], "a024c")
        self.assertEqual(normalized["horse_id"], "2021100001")
        self.assertEqual(normalized["frame"], 1)
        self.assertIsInstance(normalized["frame"], int)
        self.assertEqual(normalized["carried_weight_kg"], 57.0)
        self.assertIsInstance(normalized["carried_weight_kg"], float)
        self.assertEqual(normalized["sex"], "セ")
        self.assertIsNone(normalized["odds"])
        self.assertIsNone(normalized["popularity"])
        self.assertEqual(normalized["group_name"], "central_flat")
        json.dumps(normalized, allow_nan=False)
        opaque = normalize_row(row(jockey_id="AbCdEf", trainer_id="abcde"))
        self.assertEqual((opaque["jockey_id"], opaque["trainer_id"]), ("AbCdEf", "abcde"))

    def test_groups_include_jump_local_and_banei_without_flat_weight_limits(self):
        self.assertEqual(normalize_row(row(race_surface="障芝"))["group_name"], "central_jump")
        local = normalize_row(row(region="local", race_course="盛岡", race_surface="ダート"))
        self.assertEqual((local["group_name"], local["surface"]), ("local_flat", "ダ"))
        for label in ("", "banei"):
            banei = normalize_row(row(region="local", race_course="帯広(ば)", race_type=label,
                                      race_surface="", race_distance_m="200", carried_weight_kg="620",
                                      horse_id="B201800249", jockey_id="B0059"))
            self.assertEqual(banei["group_name"], "local_banei")
            self.assertIsNone(banei["surface"])
            self.assertEqual(banei["carried_weight_kg"], 620.0)
            self.assertEqual(banei["horse_id"], "B201800249")

    def test_entries_can_omit_labels(self):
        source = row(page="entries")
        del source["finish_status"]
        del source["finish_position"]
        normalized = normalize_row(source, page="entries")
        self.assertIsNone(normalized["finish_position"])
        self.assertIsNone(normalized["finish_status"])
        with self.assertRaises(ValueError):
            normalize_row(source)
        for status, position in (("finished", "1"), ("finished", ""), ("did_not_finish", ""),
                                 ("disqualified", ""), ("", "1")):
            with self.subTest(status=status, position=position), self.assertRaises(ValueError):
                normalize_row(row(page="entries", finish_status=status, finish_position=position), page="entries")
        scratched = normalize_row(row(page="entries", finish_status="scratched", finish_position=""), page="entries")
        self.assertEqual(scratched["finish_status"], "scratched")

    def test_completed_times_are_optional_positive_seconds_as_floats(self):
        result = normalize_row(row(time_seconds="145.0", last_3f_seconds="34.4"))
        self.assertEqual((result["time_seconds"], result["last_3f_seconds"]), (145.0, 34.4))
        self.assertIsInstance(result["time_seconds"], float)
        self.assertIsInstance(result["last_3f_seconds"], float)
        legacy = normalize_row(row())
        self.assertIsNone(legacy["time_seconds"])
        self.assertIsNone(legacy["last_3f_seconds"])
        explicit_missing = normalize_row(row(time_seconds="null", last_3f_seconds=""))
        self.assertIsNone(explicit_missing["time_seconds"])
        self.assertIsNone(explicit_missing["last_3f_seconds"])

    def test_invalid_completed_times_do_not_silently_become_history_values(self):
        for field in ("time_seconds", "last_3f_seconds"):
            for value in ("NaN", "Inf", "0", "-1", "1:25.6", "0." + "0" * 400 + "1"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    normalize_row(row(**{field: value}))

    def test_noncompletions_never_keep_raw_result_times(self):
        for status in ("did_not_finish", "disqualified", "scratched", "excluded"):
            for time, closing in (("85.6", "36.8"), ("NaN", "not-a-result")):
                with self.subTest(status=status, time=time):
                    normalized = normalize_row(row(finish_status=status, finish_position="",
                                                   time_seconds=time, last_3f_seconds=closing))
                    self.assertIsNone(normalized["time_seconds"])
                    self.assertIsNone(normalized["last_3f_seconds"])

    def test_entries_ignore_current_result_times_even_when_values_are_malformed(self):
        source = row(page="entries", finish_status="", finish_position="")
        baseline = normalize_row(source, page="entries")
        for time, closing in (("85.6", "36.8"), ("NaN", "not-a-result")):
            with self.subTest(time=time):
                with_results = normalize_row(dict(source, time_seconds=time, last_3f_seconds=closing), page="entries")
                self.assertEqual(with_results, baseline)

    def test_csv_times_roundtrip_and_missing_counts_include_legacy_and_dnf(self):
        first = row(time_seconds="145.0", last_3f_seconds="34.4", race_declared_field_size="3")
        legacy = row(horse_id="2021100002", horse_number="2", finish_position="2", race_declared_field_size="3")
        dnf = row(horse_id="2021100003", horse_number="3", finish_status="did_not_finish", finish_position="",
                  time_seconds="150.0", last_3f_seconds="37.2", race_declared_field_size="3")
        self.write([first, legacy, dnf])
        report = ingest_csv(self.csv, self.db)
        self.assertEqual(report["usable_rows"], 3)
        self.assertEqual(report["missing_values"]["time_seconds"], 2)
        self.assertEqual(report["missing_values"]["last_3f_seconds"], 2)
        runners = next(iter_races(self.db))
        self.assertEqual((runners[0]["time_seconds"], runners[0]["last_3f_seconds"]), (145.0, 34.4))
        self.assertIsNone(runners[1]["time_seconds"])
        self.assertIsNone(runners[2]["time_seconds"])

    def test_nonfinite_fractional_ranks_invalid_booleans_and_false_identity_are_rejected(self):
        for changes in ({"odds": "NaN"}, {"carried_weight_kg": "inf"}, {"age": "-1"},
                        {"finish_position": "1.2"}, {"finish_position": ""},
                        {"race_id_verified": "False"}, {"race_id_verified": "truthy"},
                        {"race_number": "2"}, {"race_date": "2024-02-30"},
                        {"race_surface": ""}, {"jockey_id": "00123.0"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_row(row(**changes))

    def test_ingestion_is_bom_aware_and_leaves_source_unchanged(self):
        self.write(self.pair(), encoding="utf-8-sig")
        digest = hashlib.sha256(self.csv.read_bytes()).hexdigest()
        report = ingest_csv(self.csv, self.db)
        self.assertEqual((report["input_rows"], report["usable_rows"], report["usable_races"]), (2, 2, 1))
        self.assertEqual(report["period"], {"start_date": "2024-01-01", "end_date": "2024-01-01", "days": 1})
        self.assertEqual(report["groups"], {"central_flat": 2})
        self.assertEqual(report["finish_status"], {"finished": 2})
        self.assertEqual(hashlib.sha256(self.csv.read_bytes()).hexdigest(), digest)
        json.dumps(report, allow_nan=False)
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_invalid_one_horse_quarantines_the_whole_race(self):
        for changes in ({"horse_id": ""}, {"finish_status": "unknown"}, {"finish_status": "abandoned"},
                        {"finish_position": "1.5"}, {"race_date": "bad date"}):
            with self.subTest(changes=changes):
                rows = self.pair()
                rows[0].update(changes)
                self.write(rows)
                destination = self.directory / ("case-" + str(len(list(self.directory.glob('case-*')))) + ".sqlite3")
                report = ingest_csv(self.csv, destination)
                self.assertEqual(report["usable_rows"], 0)
                self.assertEqual(report["excluded_race_count"], 1)
                self.assertTrue(report["excluded_races"][0]["reasons"])
                self.assertEqual(list(iter_races(destination)), [])

    def test_race_consistency_and_duplicate_horse_number_are_enforced(self):
        for changes, reason in (({"snapshot_id": "other"}, "snapshot_id"),
                                ({"race_date": "2024-01-02"}, "day"),
                                ({"race_course": "中山"}, "course"),
                                ({"race_distance_m": "1800"}, "distance_m"),
                                ({"horse_number": "1"}, "horse_number"),
                                ({"race_declared_field_size": "3"}, "declared_field_size")):
            with self.subTest(changes=changes):
                rows = self.pair();rows[1].update(changes)
                self.write(rows)
                destination = self.directory / (reason + ".sqlite3")
                report = ingest_csv(self.csv, destination)
                self.assertEqual(report["usable_races"], 0)
                self.assertTrue(any(reason in item for item in report["excluded_races"][0]["reasons"]))

    def test_scratches_and_exclusions_count_toward_declared_rows(self):
        rows = self.pair(race_declared_field_size="4")
        for number, status in ((3, "scratched"), (4, "excluded")):
            rows.append(row(horse_id=f"202110000{number}", horse_number=str(number), finish_position="",
                            finish_status=status, race_declared_field_size="4", odds="", popularity=""))
        self.write(rows)
        report = ingest_csv(self.csv, self.db)
        self.assertEqual(report["usable_rows"], 4)
        self.assertEqual(len(next(iter_races(self.db))), 4)
        self.assertEqual(report["missing_values"]["finish_position"], 2)

    def test_dnf_dq_are_retained_and_top_three_dead_heats_can_exceed_three_horses(self):
        rows = []
        for number, (status, position) in enumerate([
            ("finished", "1"), ("finished", "3"), ("finished", "3"), ("finished", "3"),
            ("did_not_finish", ""), ("disqualified", ""),
        ], 1):
            rows.append(row(horse_id=f"202110000{number}", horse_number=str(number), finish_position=position,
                            finish_status=status, race_declared_field_size="6"))
        self.write(rows)
        report = ingest_csv(self.csv, self.db)
        self.assertEqual(report["usable_rows"], 6)
        race = next(iter_races(self.db))
        self.assertEqual(sum(h["finish_position"] is not None and h["finish_position"] <= 3 for h in race), 4)

    def test_no_first_place_and_out_of_range_positions_exclude_races(self):
        self.write([row(finish_position="2"), row(horse_id="2021100002", horse_number="2", finish_position="3")])
        report = ingest_csv(self.csv, self.db)
        reasons = report["excluded_races"][0]["reasons"]
        self.assertTrue(any("no finished runner in first place" in reason for reason in reasons))
        self.assertTrue(any("exceeds" in reason for reason in reasons))

    def test_rank_limit_excludes_scratched_and_excluded_horses(self):
        rows = [row(race_declared_field_size="3"),
                row(horse_id="2021100002", horse_number="2", finish_position="3", race_declared_field_size="3"),
                row(horse_id="2021100003", horse_number="3", finish_position="", finish_status="scratched",
                    race_declared_field_size="3")]
        self.write(rows)
        report = ingest_csv(self.csv, self.db)
        self.assertEqual(report["usable_races"], 0)
        self.assertTrue(any("active race field size" in item for item in report["excluded_races"][0]["reasons"]))

    def test_duplicate_horse_keys_are_fatal_even_across_snapshots(self):
        self.write([row(), row(snapshot_id="second-snapshot")])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ingest_csv(self.csv, self.db)
        self.assertFalse(self.db.exists())

    def test_streaming_batches_and_duplicate_detection_across_batch_boundaries(self):
        rows = [row(race_id=f"{2024000000+index:010d}01", race_declared_field_size="1") for index in range(1001)]
        self.write(rows)
        events = []
        report = ingest_csv(self.csv, self.db, report=lambda **event: events.append(event))
        self.assertEqual(report["usable_races"], 1001)
        self.assertEqual(events[0]["rows"], 1000)
        duplicate = self.directory / "duplicate.sqlite3"
        self.write(rows + [dict(rows[0], snapshot_id="later")])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ingest_csv(self.csv, duplicate)
        self.assertFalse(duplicate.exists())

    def test_race_iterator_sorts_and_date_cutoff_excludes_the_cutoff_day(self):
        rows = [row(race_id="202405010201", race_date="2024-01-02", race_declared_field_size="1")]
        rows += list(reversed(self.pair()))
        self.write(rows)
        ingest_csv(self.csv, self.db)
        races = list(iter_races(self.db))
        self.assertEqual([r[0]["day"] for r in races], ["2024-01-01", "2024-01-02"])
        self.assertEqual([h["horse_number"] for h in races[0]], [1, 2])
        self.assertEqual(len(list(iter_races(self.db, before_date="2024-01-02"))), 1)
        db = sqlite3.connect(self.db)
        try:
            self.assertEqual(len(list(iter_races(db))), 2)
            self.assertEqual(db.execute("SELECT 1").fetchone()[0], 1)
        finally:
            db.close()
        with self.assertRaises(ValueError):
            list(iter_races(self.db, before_date="20240102"))

    def test_missing_header_unidentifiable_races_and_existing_outputs_are_protected(self):
        self.write([{"race_id": "202405010101"}], headers=["race_id"])
        with self.assertRaisesRegex(ValueError, "columns"):
            ingest_csv(self.csv, self.db)
        self.assertFalse(self.db.exists())
        self.write([row(race_id="")])
        with self.assertRaisesRegex(ValueError, "identify"):
            ingest_csv(self.csv, self.db)
        self.assertFalse(self.db.exists())
        self.db.write_bytes(b"existing database")
        with self.assertRaises(FileExistsError):
            ingest_csv(self.csv, self.db)
        self.assertEqual(self.db.read_bytes(), b"existing database")

    def test_reason_preview_is_limited_and_all_exclusions_remain_queryable(self):
        self.write([row(race_id=f"{2024000000+index:010d}01", finish_status="unknown") for index in range(31)])
        report = ingest_csv(self.csv, self.db)
        self.assertEqual(report["excluded_race_count"], 31)
        self.assertEqual(len(report["excluded_races"]), 30)
        self.assertTrue(report["excluded_races_truncated"])
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM race_issues").fetchone()[0], 31)

    def test_month_group_counts_only_include_usable_races(self):
        rows = self.pair()
        rows += self.pair(region="local", race_course="盛岡", race_surface="ダ")
        rows += self.pair(race_id="202405010201", race_date="2024-02-01", race_surface="障芝")
        rows += [row(race_id="202405010301", race_date="2024-03-01", finish_status="unknown")]
        self.write(rows)
        report = ingest_csv(self.csv, self.db)
        self.assertEqual(report["month_group_counts"], [
            {"month": "2024-01", "group_name": "central_flat", "rows": 2, "races": 1},
            {"month": "2024-01", "group_name": "local_flat", "rows": 2, "races": 1},
            {"month": "2024-02", "group_name": "central_jump", "rows": 2, "races": 1},
        ])
        self.assertEqual(sum(item["rows"] for item in report["month_group_counts"]), report["usable_rows"])
        self.assertEqual(sum(item["races"] for item in report["month_group_counts"]), report["usable_races"])


if __name__ == "__main__":
    unittest.main()
