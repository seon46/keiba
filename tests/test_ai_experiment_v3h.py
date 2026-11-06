from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from keiba_ai.experiment_v3 import PLStructure


def _sample(true_temperatures, races=3000, size=10, seed=4):
    rng = np.random.default_rng(seed)
    keys, positions, scores = [], [], []
    for race in range(races):
        strength = rng.normal(size=size)
        remaining, order = list(range(size)), []
        for stage in range(size):
            temperature = true_temperatures[min(stage, len(true_temperatures) - 1)]
            logits = strength[remaining] / temperature
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            order.append(remaining.pop(rng.choice(len(remaining), p=probabilities)))
        rank = np.empty(size, dtype=int)
        rank[order] = np.arange(1, size + 1)
        keys.extend([race] * size)
        positions.extend(rank.tolist())
        scores.extend(strength.tolist())
    return PLStructure(np.asarray(keys), np.asarray(positions)), np.asarray(scores)


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'scipy is required')
class PowerTemperatureTests(unittest.TestCase):
    def test_power_law_is_recovered(self):
        from keiba_ai.experiment_v3h import fit_power_temperatures
        true = [.7 * (stage ** .25) for stage in range(1, 11)]
        structure, scores = _sample(true)
        fitted, _, shape = fit_power_temperatures(structure, scores)
        self.assertAlmostEqual(shape['t1'], .7, delta=.08)
        self.assertAlmostEqual(shape['beta'], .25, delta=.07)
        self.assertEqual(len(fitted), 18)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy')),
                     'Install requirements-ml.txt for the v3h experiment tests')
class ExperimentHTests(unittest.TestCase):
    def test_experiment_h_runs_every_method(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3h import METHODS, main
        from test_ai_experiment_v3d import jockey_csv
        with TemporaryDirectory() as name:
            root = Path(name)
            jockey_csv(root / 'source.csv', years=(2021, 2022, 2023, 2024, 2025, 2026))
            with redirect_stdout(io.StringIO()):
                database = prepare(root / 'source.csv', root / 'prepared')
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['--dataset', str(database), '--output-dir', str(root / 'experiment'),
                             '--validation-years', '2025', '--threads', '1', '--rounds', '200', '--patience', '20',
                             '--evaluation-samples', '64'])
            self.assertEqual(code, 0, output.getvalue())
            report = json.loads((root / 'experiment' / 'experiment.json').read_text(encoding='utf-8'))
            self.assertFalse(report['test_period_used'])
            for group in ('central_flat', 'local_flat'):
                self.assertEqual(set(report['groups'][group]['summary']), set(METHODS))
                fold = report['groups'][group]['folds'][0]
                self.assertGreater(fold['oof_races'], fold['stop_races'])
                self.assertEqual(len(fold['methods']['oof_6']['temperatures']), 6)


if __name__ == '__main__':
    unittest.main()
