from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy', 'joblib', 'torch')),
                     'Install requirements-ml.txt and torch for the v4 pipeline tests')
class PipelineV4Tests(unittest.TestCase):
    def test_fit_then_predict_matches_fit_side_scores(self):
        from dataclasses import replace
        from keiba_ai.cli import prepare
        from keiba_ai.pipeline_v4 import NN_CONFIG, fit, load_predictor, main
        from test_ai_experiment_v3d import jockey_csv
        from test_ai_experiment_v3j import pedigree_store
        from test_ai_pipeline_v3 import entries_from_results
        with TemporaryDirectory() as name:
            root = Path(name)
            jockey_csv(root / 'source.csv', years=(2021, 2022, 2023, 2024, 2025))
            with (root / 'source.csv').open(encoding='utf-8') as file:
                horses = sorted({row['horse_id'] for row in csv.DictReader(file)})
            pedigree_store(root / 'data', horses[:-3])  # a few horses without a pedigree
            race_db = root / 'data' / 'keiba.sqlite3'
            with redirect_stdout(io.StringIO()):
                database = prepare(root / 'source.csv', root / 'prepared')
            small = replace(NN_CONFIG, width=32, ensemble=4, max_epochs=2, batch_races=32, n_frequencies=8,
                            numeric_dim=4, embedding=4, pedigree_dim=4, entity_min_count=2, ancestor_min_count=1)
            summary = fit(database, race_db, root / 'model', train_end='2025-07-01', evaluate_until='2025-12-31',
                          threads=1, device='cpu', seeds=(42, 7), nn_config=small, rounds=200, patience=20,
                          evaluation_samples=64)
            self.assertEqual(summary['pipeline'], 'v4-pipeline-1')
            for group in ('central_flat', 'local_flat'):
                result = summary['groups'][group]
                self.assertEqual(len(result['networks']), 2)
                self.assertEqual(len(result['blend_weights']), 2)
                evaluation = result['evaluation']
                self.assertEqual(set(evaluation), {'model', 'lightgbm_pedigree', 'network', 'uniform'})
                self.assertLess(evaluation['model']['race_outcome_log_loss'],
                                evaluation['uniform']['race_outcome_log_loss'] + .05)
            self.assertEqual(type(load_predictor(database, root / 'model', race_db)).__module__, 'keiba_ai.pipeline_v4')

            with (root / 'model' / 'evaluation_local_flat.csv').open(encoding='utf-8') as file:
                evaluated = list(csv.DictReader(file))
            day = sorted({row['race_date'] for row in evaluated})[3]
            self.assertEqual(entries_from_results(root / 'source.csv', root / 'entries.csv', day), 20)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['predict', '--dataset', str(database), '--race-db', str(race_db), '--model-dir',
                             str(root / 'model'), '--entries', str(root / 'entries.csv'), '--output',
                             str(root / 'prediction.csv'), '--samples', '256', '--threads', '1'])
            self.assertEqual(code, 0, output.getvalue()[-3000:])
            with (root / 'prediction.csv').open(encoding='utf-8') as file:
                predicted = list(csv.DictReader(file))
            self.assertEqual(len(predicted), 20)
            for group in ('central_flat', 'local_flat'):
                with (root / 'model' / f'evaluation_{group}.csv').open(encoding='utf-8') as file:
                    reference = {(row['race_id'], row['horse_id']): row for row in csv.DictReader(file)
                                 if row['race_date'] == day}
                rows = [row for row in predicted if row['group'] == group]
                self.assertEqual(len(rows), 10)
                for row in rows:
                    expected = reference[(row['race_id'], row['horse_id'])]
                    self.assertAlmostEqual(float(row['strength_score']), float(expected['strength_score']), places=4)
                    self.assertAlmostEqual(float(row['finish_probability']), float(expected['finish_probability']),
                                           places=4)
                    total = sum(float(row[f'rank_{rank}_probability']) for rank in range(1, 11))
                    self.assertAlmostEqual(total + float(row['nonfinish_probability']), 1.0, places=6)

            entries_from_results(root / 'source.csv', root / 'early.csv', '2025-01-03')
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['predict', '--dataset', str(database), '--race-db', str(race_db), '--model-dir',
                             str(root / 'model'), '--entries', str(root / 'early.csv'), '--output',
                             str(root / 'early_prediction.csv')])
            self.assertEqual(code, 2)
            self.assertIn('学習期間内', output.getvalue())


if __name__ == '__main__':
    unittest.main()
