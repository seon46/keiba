from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import numpy as np


def pedigree_store(directory, horses_by_strength):
    """Scraper store whose pedigrees tie stronger horses to the same sires."""
    from keiba_ai.pedigree import PATHS
    from keiba_scraper.storage import Store
    Store(directory).close()
    connection = sqlite3.connect(Path(directory) / 'keiba.sqlite3')
    for rank, horse in enumerate(horses_by_strength):
        sire, damsire = f'10{rank // 8:08d}', f'20{rank // 5:08d}'
        ancestors = []
        for path in PATHS:
            if path == 'S':
                identifier = sire
            elif path == 'DS':
                identifier = damsire
            elif path == 'D':
                identifier = f'30{rank:08d}'
            else:
                identifier = f'4{len(path)}{(rank * 7 + len(path)) % 30:08d}'
            ancestors.append(dict(path=path, generation=len(path), name=f'馬{identifier}', horse_id=identifier))
        snapshot = f'ped-{horse}'
        connection.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?)',
                           (snapshot, f'https://db.netkeiba.com/horse/ped/{horse}/', '2026-11-01T00:00:00', 200,
                            'utf-8', '0' * 64, 'raw/none'))
        connection.execute('INSERT INTO pedigrees VALUES (?,?,?)',
                           (snapshot, horse, json.dumps(dict(horse_id=horse, ancestors=ancestors), ensure_ascii=False)))
    connection.commit()
    connection.close()


class PedigreeBlockTests(unittest.TestCase):
    def test_sire_theta_is_causal_and_learns(self):
        from keiba_ai.pedigree import PATHS, PEDIGREE_FEATURES, pedigree_block
        days = np.asarray(['2024-01-01'] * 4 + ['2024-01-02'] * 4)
        races = np.asarray(['r1'] * 4 + ['r2'] * 4)
        horses = np.asarray(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8'], dtype=object)
        finished = np.ones(8, dtype=bool)
        positions = np.asarray([1, 2, 3, 4, 1, 2, 3, 4])
        strength = np.zeros(8)
        pedigrees = {h: tuple(('SIREA' if h in ('h1', 'h5') else 'SIREB') if path == 'S' else None for path in PATHS)
                     for h in horses}
        block = pedigree_block(days, races, horses, finished, positions, strength, np.ones(8), np.full(8, 1600.),
                               pedigrees)
        theta = PEDIGREE_FEATURES.index('sire_theta')
        runs = PEDIGREE_FEATURES.index('sire_runs')
        self.assertEqual(block[0, runs], 0)           # nothing before day 1
        self.assertEqual(block[0, theta], 0)
        self.assertEqual(block[4, runs], 1)            # SIREA ran once before day 2
        self.assertGreater(block[4, theta], block[5, theta])  # SIREA's winner beat SIREB's runners


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy', 'torch')),
                     'Install requirements-ml.txt and PyTorch for the v3j experiment tests')
class ExperimentJTests(unittest.TestCase):
    def test_experiment_j_runs_every_variant(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3j import VARIANTS, main
        from test_ai_experiment_v3d import jockey_csv
        with TemporaryDirectory() as name:
            root = Path(name)
            jockey_csv(root / 'source.csv', years=(2021, 2022, 2023, 2024, 2025, 2026))
            with (root / 'source.csv').open(encoding='utf-8') as file:
                horses = sorted({row['horse_id'] for row in csv.DictReader(file)})
            pedigree_store(root / 'data', horses)
            with redirect_stdout(io.StringIO()):
                database = prepare(root / 'source.csv', root / 'prepared')
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['--dataset', str(database), '--race-db', str(root / 'data' / 'keiba.sqlite3'),
                             '--output-dir', str(root / 'experiment'), '--validation-years', '2025', '--threads', '1',
                             '--rounds', '200', '--patience', '20', '--evaluation-samples', '64', '--device', 'cpu',
                             '--nn-epochs', '3', '--nn-width', '32', '--nn-batch-races', '32'])
            self.assertEqual(code, 0, output.getvalue()[-3000:])
            report = json.loads((root / 'experiment' / 'experiment.json').read_text(encoding='utf-8'))
            self.assertFalse(report['test_period_used'])
            for group in ('central_flat', 'local_flat'):
                result = report['groups'][group]
                self.assertEqual(result['pedigree_coverage'], 1.0)
                self.assertEqual(set(result['summary']), set(VARIANTS))
                for name in VARIANTS:
                    self.assertTrue(np.isfinite(result['summary'][name]['race_outcome_log_loss']))
                fold = result['folds'][0]['variants']
                self.assertEqual(len(fold['D_blend']['weights']), 2)
                self.assertGreaterEqual(fold['C_nn']['best_epoch'], 1)


if __name__ == '__main__':
    unittest.main()
