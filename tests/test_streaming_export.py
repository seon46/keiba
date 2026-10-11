from __future__ import annotations

import csv
from dataclasses import replace
import gzip
import hashlib
import io
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from keiba_scraper.storage import Store, StorageError


class StreamingExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.serial = 0

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def race(self, identifier="202405021211", *, day="2024-05-26", region="central",
             page="results", runners=None, **metadata):
        self.serial += 1
        url = f"https://example.test/{region}/{identifier}/{page}"
        observation = self.store.observe(url, f"race {self.serial}".encode(), encoding="utf-8")
        race = {"race_id": identifier, "region": region, "page": page, **metadata}
        if day is not None:
            race["race_date"] = day
        if runners is None:
            runners = [{"horse_id": "2021100001", "horse_number": 1}]
            if page == "results":
                runners[0]["finish_position"] = 1
        self.store.save_race(observation, {"race": race, "runners": runners})
        return observation

    def horse(self, runs, identifier="2021100001"):
        self.serial += 1
        observation = self.store.observe(
            f"https://example.test/horse/{identifier}",
            f"horse {self.serial}".encode(), encoding="utf-8",
        )
        self.store.save_horse(observation, {
            "horse_id": identifier, "horse_name": "テスト馬", "runs": runs,
        })
        return observation

    def test_inclusive_dates_filter_race_days_and_preserve_page_and_region(self):
        self.race("202405021201", day="2024-05-25")
        lower = self.race("202405021202", day="2024-05-26")
        upper = self.race("202405021203", day="2024-05-27")
        self.race("202405021204", day="2024-05-28")
        self.race("202405021205", day=None)
        self.race("202405021202", day="2024-05-26", page="entries")
        self.race("202405021202", day="2024-05-26", region="local")
        filters = dict(dataset="runners", page="results", region="central",
                       start_date="2024-05-26", end_date="2024-05-27")
        rows = list(self.store.iter_export_rows(**filters))
        self.assertEqual([row["snapshot_id"] for row in rows], [lower.id, upper.id])
        self.assertEqual([row["race_date"] for row in rows], ["2024-05-26", "2024-05-27"])
        self.assertEqual(rows[0]["retrieved_at"], lower.retrieved_at)
        self.assertEqual(rows[0]["source_url"], lower.url)
        self.assertEqual(rows[0]["source"], "netkeiba")
        self.assertNotIn("race_race_date", rows[0])
        self.assertEqual(self.store.export_rows(**filters), rows)
        entries = list(self.store.iter_export_rows(**{**filters, "page": "entries"}))
        self.assertEqual(len(entries), 1)
        self.assertNotIn("finish_position", entries[0])
        self.assertEqual(len(list(self.store.iter_export_rows(dataset="runners"))), 7)
        self.assertEqual(len(list(self.store.iter_export_rows(
            **{**filters, "start_date": None, "end_date": "2024-05-26"}))), 2)

    def test_date_filter_never_revives_an_old_corrected_snapshot(self):
        old = self.race(day="2024-05-26", old_only=True)
        new = self.race(day="2024-05-28", new_only=True)
        filters = dict(dataset="runners", start_date="2024-05-26", end_date="2024-05-27")
        self.assertEqual(list(self.store.iter_export_rows(**filters)), [])
        self.assertEqual(self.store.export_fields(**filters), [])
        historical = list(self.store.iter_export_rows(**filters, all_snapshots=True))
        self.assertEqual([row["snapshot_id"] for row in historical], [old.id])
        latest = list(self.store.iter_export_rows(dataset="runners"))
        self.assertEqual([row["snapshot_id"] for row in latest], [new.id])

    def test_horse_runs_filter_each_run_after_selecting_latest_horse(self):
        old = self.horse([{"race_id": "202405021211", "race_date": "2024-05-26", "old_only": 1}])
        latest = self.horse([
            {"race_id": "202405021211", "race_date": "2024-05-26", "finish_position": 1},
            {"race_id": "2024J0010109", "race_date": "2024-05-27", "finish_position": 2},
            {"race_id": "202405021212", "race_date": "2024-05-28", "outside_only": 1},
            {"race_id": "202405021213", "race_date": None},
        ])
        filters = dict(dataset="horse-runs", start_date="2024-05-26", end_date="2024-05-27")
        rows = list(self.store.iter_export_rows(**filters))
        self.assertEqual([row["race_id"] for row in rows], ["202405021211", "2024J0010109"])
        self.assertTrue(all(row["snapshot_id"] == latest.id for row in rows))
        self.assertTrue(all(row["horse_id"] == "2021100001" for row in rows))
        self.assertNotIn("old_only", self.store.export_fields(**filters))
        self.assertNotIn("outside_only", self.store.export_fields(**filters))
        snapshots = list(self.store.iter_export_rows(**filters, all_snapshots=True))
        self.assertEqual([row["snapshot_id"] for row in snapshots], [old.id, latest.id, latest.id])
        self.assertEqual(len(list(self.store.iter_export_rows(dataset="horse-runs"))), 4)

    def test_csv_fields_include_later_rows_and_only_the_selected_period(self):
        self.race("202405021211", day="2024-05-26", race_name="テスト賞", runners=[
            {"horse_id": "2021100001", "finish_position": 1},
            {"horse_id": "2021100002", "finish_position": 2, "weight_kg": 480},
        ])
        self.race("202405021212", day="2024-05-27", track_condition="良", runners=[
            {"horse_id": "2021100003", "finish_position": 1, "corner_order": "3-2-1"},
        ])
        self.race("202405021213", day="2024-05-28", outside_column="excluded")
        filters = dict(dataset="runners", start_date="2024-05-26", end_date="2024-05-27")
        fields = self.store.export_fields(**filters)
        self.assertTrue({"weight_kg", "corner_order", "race_track_condition", "race_name"}.issubset(fields))
        self.assertNotIn("race_outside_column", fields)
        self.assertEqual(len(fields), len(set(fields)))
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(self.store.iter_export_rows(**filters))
        parsed = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[1]["weight_kg"], "480")
        self.assertEqual(parsed[2]["corner_order"], "3-2-1")

    def test_iterator_is_lazy_and_decodes_only_one_requested_row(self):
        self.race(runners=[{"horse_id": str(2021100000 + index), "finish_position": index}
                           for index in range(1, 101)])
        original_loads = json.loads
        with patch("keiba_scraper.storage.json.loads", wraps=original_loads) as decoder:
            iterator = self.store.iter_export_rows(dataset="runners")
            self.assertEqual(decoder.call_count, 0)
            first = next(iterator)
            self.assertEqual(first["horse_id"], "2021100001")
            self.assertEqual(decoder.call_count, 2)
            second = next(iterator)
            self.assertEqual(second["horse_id"], "2021100002")
            self.assertEqual(decoder.call_count, 4)
            iterator.close()
        # The header scan must use streaming rows, not the compatibility list.
        with patch.object(self.store, "export_rows", side_effect=AssertionError("must stream")):
            self.assertIn("horse_id", self.store.export_fields(dataset="runners"))

    def test_empty_date_validation_and_existing_database_reopening(self):
        self.assertEqual(list(self.store.iter_export_rows(dataset="runners")), [])
        self.assertEqual(self.store.export_fields(dataset="horse-runs"), [])
        for start, end in (("20240526", None), ("2024-02-30", None), (None, "invalid"),
                           ("2024-05-28", "2024-05-26")):
            with self.subTest(start=start, end=end), self.assertRaises(StorageError):
                self.store.export_fields(dataset="runners", start_date=start, end_date=end)
        observation = self.race()
        self.store.close()
        self.store = Store(self.temp.name)
        rows = list(self.store.iter_export_rows(dataset="runners"))
        self.assertEqual(rows[0]["snapshot_id"], observation.id)
        self.assertEqual(self.store.body(observation), b"race 1")

    def test_new_gzip_and_legacy_html_preserve_original_bytes_and_observations(self):
        original = '<meta charset="EUC-JP"><p>東京競馬場</p>'.encode("euc-jp")
        compressed = self.store.observe("https://example.test/new", original, encoding="euc-jp")
        self.assertTrue(compressed.raw_path.endswith(".html.gz"))
        compressed_path = self.store.directory / compressed.raw_path
        first_archive = compressed_path.read_bytes()
        self.assertEqual(gzip.decompress(first_archive), original)
        self.assertEqual(compressed.sha256, hashlib.sha256(original).hexdigest())
        self.assertEqual(self.store.body(compressed), original)
        repeated = self.store.observe("https://example.test/new", original, encoding="euc-jp")
        self.assertEqual(repeated.raw_path, compressed.raw_path)
        self.assertNotEqual(repeated.id, compressed.id)
        self.assertEqual(compressed_path.read_bytes(), first_archive)
        self.assertEqual(first_archive[4:8], b"\x00\x00\x00\x00")

        # Reproduce a pre-compression row without migrating its original file.
        legacy = replace(compressed, id="legacy-observation", url="https://example.test/legacy",
                         raw_path=compressed.raw_path.removesuffix(".gz"))
        legacy_path = self.store.directory / legacy.raw_path
        legacy_path.write_bytes(original)
        with self.store.db:
            self.store.db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?)",
                                  tuple(legacy.__dict__.values()))
        self.store.close()
        self.store = Store(self.temp.name)
        self.assertEqual(self.store.cached(legacy.url), legacy)
        self.assertEqual(self.store.body(legacy), original)
        self.assertEqual(self.store.body(compressed), original)
        self.assertEqual(legacy_path.read_bytes(), original)
        self.assertEqual(compressed_path.read_bytes(), first_archive)
        self.assertEqual(self.store.cached(legacy.url).retrieved_at, legacy.retrieved_at)

    def test_broken_and_wrong_content_gzip_raise_storage_errors(self):
        observation = self.store.observe("https://example.test/raw", b"original", encoding="utf-8")
        path = self.store.directory / observation.raw_path
        valid_archive = path.read_bytes()
        for damaged in (b"not gzip", valid_archive[:-4], gzip.compress(b"different", mtime=0)):
            with self.subTest(damaged=damaged[:10]):
                path.write_bytes(damaged)
                with self.assertRaises(StorageError):
                    self.store.body(observation)


if __name__ == "__main__":
    unittest.main()
