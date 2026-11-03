from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy', 'torch')),
                     'Install requirements-ml.txt and PyTorch for the v3k experiment tests')
class ExperimentKTests(unittest.TestCase):
    def test_variants_run_and_the_cache_is_reused(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3k import main
        from test_ai_experiment_v3d import jockey_csv
        from test_ai_experiment_v3j import pedigree_store
        with TemporaryDirectory() as name:
            root = Path(name)
            jockey_csv(root / 'source.csv', years=(2021, 2022, 2023, 2024, 2025, 2026))
            with (root / 'source.csv').open(encoding='utf-8') as file:
                horses = sorted({row['horse_id'] for row in csv.DictReader(file)})
            pedigree_store(root / 'data', horses)
            with redirect_stdout(io.StringIO()):
                database = prepare(root / 'source.csv', root / 'prepared')
            common = ['--dataset', str(database), '--race-db', str(root / 'data' / 'keiba.sqlite3'),
                      '--cache-dir', str(root / 'cache'), '--validation-years', '2025', '--threads', '1',
                      '--rounds', '200', '--patience', '20', '--evaluation-samples', '64', '--device', 'cpu',
                      '--seeds', '1', '2']
            for run, extra in (('first', ['--search-trials', '2']), ('second', ['--variants', 'nn_sched'])):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(common + ['--output-dir', str(root / run)] + extra)
                self.assertEqual(code, 0, output.getvalue()[-3000:])
            first = json.loads((root / 'first' / 'experiment.json').read_text(encoding='utf-8'))
            second = json.loads((root / 'second' / 'experiment.json').read_text(encoding='utf-8'))
            for group in ('central_flat', 'local_flat'):
                summary = first['groups'][group]['summary']
                for key in ('A_lgbm', 'B_lgbm_ped', 'nn_j', 'D_nn_j', 'nn_reg_x5', 'D_nn_search_x5'):
                    self.assertTrue(np.isfinite(summary[key]['race_outcome_log_loss']), key)
                self.assertIn('vs_D_nn_j', summary['D_nn_reg_x5'])
                fold = first['groups'][group]['folds'][0]
                self.assertFalse(fold['cached'])
                self.assertEqual(len(fold['variants']['nn_search_x5']['search']), 2)
                self.assertTrue(second['groups'][group]['folds'][0]['cached'])
                # the cached LightGBM scores are identical between runs
                self.assertAlmostEqual(second['groups'][group]['summary']['A_lgbm']['race_outcome_log_loss'],
                                       summary['A_lgbm']['race_outcome_log_loss'], places=9)


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy', 'torch')),
                     'Install requirements-ml.txt and PyTorch for the v3l experiment tests')
class ExperimentLTests(unittest.TestCase):
    def test_tabm_variants_run(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3l import DEFAULT_VARIANTS, main
        from test_ai_experiment_v3d import jockey_csv
        from test_ai_experiment_v3j import pedigree_store
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
                             '--cache-dir', str(root / 'cache'), '--output-dir', str(root / 'l'),
                             '--validation-years', '2025', '--threads', '1', '--rounds', '200', '--patience', '20',
                             '--evaluation-samples', '64', '--device', 'cpu'])
            self.assertEqual(code, 0, output.getvalue()[-3000:])
            report = json.loads((root / 'l' / 'experiment.json').read_text(encoding='utf-8'))
            for group in ('central_flat', 'local_flat'):
                summary = report['groups'][group]['summary']
                for variant in DEFAULT_VARIANTS:
                    self.assertTrue(np.isfinite(summary[variant]['race_outcome_log_loss']), variant)
                    self.assertIn(f'D_{variant}', summary)
                self.assertEqual(report['groups'][group]['folds'][0]['variants']['tabm']['config']['ensemble'], 32)
