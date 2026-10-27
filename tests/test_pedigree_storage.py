from contextlib import redirect_stderr, redirect_stdout
import copy
import csv
from itertools import product
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from keiba_scraper.cli import main
from keiba_scraper.storage import Store, StorageError


def tree(identifier='2019102383', *, partial=False):
    paths = [''.join(path) for length in range(1, 6) for path in product('SD', repeat=length)]
    missing = ['DDDDD'] if partial else []
    ancestors = [dict(path=path, generation=len(path),
                      horse_id=None if path in missing else f'{index:010d}',
                      name=None if path in missing else f'祖先{index}', source_url=None)
                 for index, path in enumerate(paths, 1)]
    by_path = {ancestor['path']: ancestor for ancestor in ancestors}
    result = dict(horse_id=identifier, horse_name='血統テスト馬', ancestors=ancestors,
                  missing_paths=missing, generations=5, parse_warnings=[])
    for prefix, path in (('sire', 'S'), ('dam', 'D'), ('damsire', 'DS')):
        result[prefix + '_id'] = by_path[path]['horse_id']
        result[prefix + '_name'] = by_path[path]['name']
    return result


class PedigreeStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.store = Store(self.directory)
        self.addCleanup(self.store.close)

    def save(self, parsed=None, body=b'pedigree'):
        parsed = parsed or tree()
        observation = self.store.observe(f"https://db.netkeiba.com/horse/ped/{parsed['horse_id']}/",
                                         body, encoding='utf-8')
        self.store.save_pedigree(observation, parsed)
        return observation

    def cli(self, *args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(['--data-dir', str(self.directory), *args])

    def test_full_and_partial_trees_keep_paths_ids_and_provenance(self):
        first = self.save()
        partial = self.save(tree('B202100001', partial=True), body=b'partial pedigree')
        rows = list(self.store.iter_export_rows(dataset='pedigrees'))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['snapshot_id'], first.id)
        self.assertEqual(rows[0]['retrieved_at'], first.retrieved_at)
        self.assertEqual(rows[0]['source_url'], first.url)
        self.assertEqual(rows[0]['known_ancestor_count'], 62)
        self.assertEqual(rows[1]['known_ancestor_count'], 61)
        self.assertEqual(rows[1]['identified_ancestor_count'], 61)
        self.assertEqual(rows[1]['missing_paths'], ['DDDDD'])
        self.assertIsNone(rows[1]['ancestors'][-1]['name'])
        self.assertEqual(self.store.body(partial), b'partial pedigree')

    def test_save_is_idempotent_and_latest_export_preserves_all_observations(self):
        first = self.save()
        original = self.store.body(first)
        self.store.save_pedigree(first, tree())
        second = self.save(tree(partial=True), body=b'new pedigree')
        rows = self.store.export_rows(dataset='pedigrees')
        self.assertEqual([row['snapshot_id'] for row in rows], [second.id])
        self.assertEqual(len(self.store.export_rows(dataset='pedigrees', all_snapshots=True)), 2)
        self.assertEqual(self.store.body(first), original)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM races').fetchone()[0], 0)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM horses').fetchone()[0], 0)

    def test_mismatched_observation_and_broken_tree_never_replace_saved_tree(self):
        original = self.save()
        wrong = self.store.observe('https://db.netkeiba.com/horse/ped/2019100000/', b'wrong', encoding='utf-8')
        with self.assertRaises(StorageError):
            self.store.save_pedigree(wrong, tree())
        for mutation in ('short', 'duplicate', 'generation', 'ancestor_id', 'empty'):
            parsed = copy.deepcopy(tree())
            if mutation == 'short': parsed['ancestors'].pop()
            elif mutation == 'duplicate': parsed['ancestors'][1] = parsed['ancestors'][0]
            elif mutation == 'generation': parsed['ancestors'][0]['generation'] = 5
            elif mutation == 'ancestor_id': parsed['ancestors'][0]['horse_id'] = '../unsafe'
            else:
                for ancestor in parsed['ancestors']:
                    ancestor.update(horse_id=None, name=None)
            with self.subTest(mutation=mutation), self.assertRaises(StorageError):
                self.store.save_pedigree(original, parsed)
        self.assertEqual(self.store.export_rows(dataset='pedigrees')[0]['known_ancestor_count'], 62)

    def test_cli_csv_and_jsonl_exports_need_no_http_and_preserve_nested_tree(self):
        self.save(tree(partial=True))
        csv_path = self.directory/'pedigrees.csv'
        jsonl_path = self.directory/'pedigrees.jsonl'
        with patch('keiba_scraper.cli.Client', side_effect=AssertionError('export must not make HTTP calls')):
            self.assertEqual(self.cli('export', '--dataset', 'pedigrees', '--output', str(csv_path)), 0)
            self.assertEqual(self.cli('export', '--dataset', 'pedigrees', '--format', 'jsonl',
                                      '--output', str(jsonl_path)), 0)
        with csv_path.open(encoding='utf-8', newline='') as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row['sire_id'], '0000000001')
        self.assertEqual(len(json.loads(row['ancestors'])), 62)
        self.assertEqual(json.loads(jsonl_path.read_text())['missing_paths'], ['DDDDD'])
        self.assertEqual(self.cli('export', '--dataset', 'pedigrees', '--output', str(csv_path)), 2)

    def test_pedigrees_reject_race_filters_and_original_file_destinations(self):
        self.save()
        for filters in (dict(page='results'), dict(region='central'), dict(start_date='2021-01-01')):
            with self.subTest(filters=filters), self.assertRaises(StorageError):
                list(self.store.iter_export_rows(dataset='pedigrees', **filters))
        self.assertEqual(self.cli('export', '--dataset', 'pedigrees', '--overwrite',
                                  '--output', str(self.directory/'keiba.sqlite3')), 2)
        self.assertEqual(self.store.db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')


if __name__ == '__main__':
    unittest.main()
