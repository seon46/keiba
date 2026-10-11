from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from keiba_scraper.cli import main
from keiba_scraper.storage import Store


class ProductionCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.store = Store(self.directory)
        for number, day, runner in ((1, '2021-10-11', {'horse_id': '2019100001'}),
                                    (2, '2021-10-12', {'horse_id': '2019100002', 'later_field': 'later'})):
            obs = self.store.observe(f'https://race.netkeiba.com/race/result.html?race_id=20210504030{number}',
                                     str(number).encode(), encoding='utf-8')
            self.store.save_race(obs, {'race': {'race_id': f'20210504030{number}', 'region': 'central',
                                              'page': 'results', 'race_date': day}, 'runners': [runner]})
        self.store.close()

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args):
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            return main(['--data-dir', str(self.directory), *args])

    def test_streaming_csv_includes_later_columns_and_inclusive_date_filters(self):
        output = self.directory / 'runners.csv'
        self.assertEqual(self.run_cli('export', '--output', str(output)), 0)
        with output.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['later_field'], 'later')
        selected = self.directory / 'one.jsonl'
        self.assertEqual(self.run_cli('export', '--format', 'jsonl', '--start-date', '2021-10-12',
                                     '--end-date', '2021-10-12', '--output', str(selected)), 0)
        self.assertEqual(json.loads(selected.read_text())['horse_id'], '2019100002')

    def test_failed_export_never_replaces_existing_output_or_leaves_partial_files(self):
        output = self.directory / 'existing.jsonl'
        output.write_text('original user file')
        def interrupted_rows(*args, **kwargs):
            yield {'horse_id': '2019100001'}
            raise OSError('simulated full disk')
        with patch.object(Store, 'iter_export_rows', interrupted_rows):
            self.assertEqual(self.run_cli('export', '--format', 'jsonl', '--output', str(output), '--overwrite'), 2)
        self.assertEqual(output.read_text(), 'original user file')
        self.assertFalse(list(self.directory.glob('.keiba-export-*')))

    def test_csv_header_and_rows_see_same_snapshot_while_another_collector_writes(self):
        original = Store.export_fields
        def concurrent_insert(store, **kwargs):
            fields = original(store, **kwargs)
            other = Store(self.directory)
            try:
                obs = other.observe('https://race.netkeiba.com/race/new', b'new row', encoding='utf-8')
                other.save_race(obs, {'race': {'race_id': '202105040303', 'region': 'central', 'page': 'results'},
                                      'runners': [{'horse_id': '2019100003', 'new_column': 'arrived during export'}]})
            finally:
                other.close()
            return fields
        output = self.directory / 'snapshot.csv'
        with patch.object(Store, 'export_fields', concurrent_insert):
            self.assertEqual(self.run_cli('export', '--output', str(output)), 0)
        with output.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        store = Store(self.directory)
        try:
            self.assertEqual(len(store.export_rows(dataset='runners')), 3)
        finally:
            store.close()

    def test_outputs_cannot_replace_database_sidecars_or_raw_originals(self):
        for suffix in ('', '-wal', '-shm', '-journal'):
            self.assertEqual(self.run_cli('export', '--overwrite', '--output',
                                         str(self.directory / ('keiba.sqlite3' + suffix))), 2)
        self.assertEqual(self.run_cli('export', '--overwrite', '--output', str(self.directory / 'raw' / 'data.csv')), 2)
        store = Store(self.directory)
        try:
            self.assertEqual(store.db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
        finally:
            store.close()

    def test_resume_scope_changes_and_abbreviated_flags_are_rejected(self):
        for option in ('--start-date', '--sta', '--years', '--ye'):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(['backfill', '--resume', 'example', option,
                      '5' if option.startswith('--ye') else '2022-01-01', '--plan-only'])
            self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
