from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ENTRY_DROP = ('finish_position', 'finish_status', 'time_seconds', 'last_3f_seconds')


def entries_from_results(source, destination, day):
    with source.open(encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader if row['race_date'] == day]
        fields = [field for field in reader.fieldnames if field not in ENTRY_DROP]
    with destination.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: ('entries' if field == 'page' else row[field]) for field in fields})
    return len(rows)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy', 'joblib')),
                     'Install requirements-ml.txt for the v3 pipeline tests')
class PipelineTests(unittest.TestCase):
    def test_fit_then_predict_matches_fit_side_scores(self):
        from keiba_ai.cli import prepare
        from keiba_ai.pipeline_v3 import main
        from test_ai_experiment_v3d import jockey_csv
        with TemporaryDirectory() as name:
            root = Path(name)
            jockey_csv(root / 'source.csv', years=(2021, 2022, 2023, 2024, 2025))
            with redirect_stdout(io.StringIO()):
                database = prepare(root / 'source.csv', root / 'prepared')
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['fit', '--dataset', str(database), '--output-dir', str(root / 'model'),
                             '--train-end', '2025-07-01', '--evaluate-until', '2025-12-31', '--threads', '1',
                             '--rounds', '200', '--patience', '20', '--evaluation-samples', '64'])
            self.assertEqual(code, 0, output.getvalue())
            summary = json.loads((root / 'model' / 'model.json').read_text(encoding='utf-8'))
            self.assertFalse(summary['evaluated_test_period'])
            for group in ('central_flat', 'local_flat'):
                evaluation = summary['groups'][group]['evaluation']
                self.assertLess(evaluation['model']['race_outcome_log_loss'],
                                evaluation['uniform']['race_outcome_log_loss'] + .05)

            # A day after train_end with two earlier post-train_end days in the history.
            with (root / 'model' / 'evaluation_local_flat.csv').open(encoding='utf-8') as file:
                evaluated = list(csv.DictReader(file))
            day = sorted({row['race_date'] for row in evaluated})[3]
            self.assertEqual(entries_from_results(root / 'source.csv', root / 'entries.csv', day), 20)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['predict', '--dataset', str(database), '--model-dir', str(root / 'model'),
                             '--entries', str(root / 'entries.csv'), '--output', str(root / 'prediction.csv'),
                             '--samples', '256', '--threads', '1'])
            self.assertEqual(code, 0, output.getvalue())
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
                for rank in range(1, 11):
                    column = sum(float(row[f'rank_{rank}_probability']) for row in rows)
                    self.assertLessEqual(column, 1.0 + 1e-6)

            # evaluate on the saved bundle reproduces the fit-time evaluation.
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['evaluate', '--dataset', str(database), '--model-dir', str(root / 'model'),
                             '--output-dir', str(root / 'evaluation'), '--end', '2025-12-31', '--samples', '64',
                             '--threads', '1'])
            self.assertEqual(code, 0, output.getvalue())
            evaluation = json.loads((root / 'evaluation' / 'evaluation.json').read_text(encoding='utf-8'))
            for group in ('central_flat', 'local_flat'):
                self.assertAlmostEqual(evaluation['groups'][group]['model']['race_outcome_log_loss'],
                                       summary['groups'][group]['evaluation']['model']['race_outcome_log_loss'], places=5)
                self.assertTrue(evaluation['groups'][group]['by_week'])
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['evaluate', '--dataset', str(database), '--model-dir', str(root / 'model'),
                             '--output-dir', str(root / 'evaluation_early'), '--start', '2025-01-01'])
            self.assertEqual(code, 2)

            # Days inside the training period are refused.
            entries_from_results(root / 'source.csv', root / 'early.csv', '2025-01-03')
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['predict', '--dataset', str(database), '--model-dir', str(root / 'model'),
                             '--entries', str(root / 'early.csv'), '--output', str(root / 'early_prediction.csv')])
            self.assertEqual(code, 2)
            self.assertIn('学習期間内', output.getvalue())

    def test_test_period_evaluation_needs_explicit_flag(self):
        from keiba_ai.pipeline_v3 import fit
        with TemporaryDirectory() as name:
            with self.assertRaises(ValueError):
                fit(Path(name) / 'missing.sqlite3', Path(name) / 'out', train_end='2025-12-01',
                    evaluate_until='2026-06-01')


if __name__ == '__main__':
    unittest.main()
