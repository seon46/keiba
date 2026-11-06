from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from keiba_ai.experiment_v3f import RELATIVE_FEATURES, race_relative


class RaceRelativeTests(unittest.TestCase):
    def test_z_and_rank_are_within_race(self):
        values = np.asarray([[1.], [2.], [3.], [np.nan], [5.], [5.]])
        races = np.asarray(['a', 'a', 'a', 'a', 'b', 'b'])
        out = race_relative(values, races)
        np.testing.assert_allclose(out[:3, 0], [-1.2247449, 0, 1.2247449], atol=1e-6)
        np.testing.assert_allclose(out[:3, 1], [0, .5, 1])
        self.assertTrue(np.isnan(out[3]).all())
        np.testing.assert_allclose(out[4:, 0], [0, 0])
        np.testing.assert_allclose(out[4:, 1], [.5, .5])

    def test_single_known_value_stays_missing(self):
        out = race_relative(np.asarray([[1.], [np.nan]]), np.asarray(['a', 'a']))
        self.assertTrue(np.isnan(out).all())

    def test_feature_names_are_unique(self):
        self.assertEqual(len(RELATIVE_FEATURES), len(set(RELATIVE_FEATURES)))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy')),
                     'Install requirements-ml.txt for the v3f experiment tests')
class ExperimentFTests(unittest.TestCase):
    def test_experiment_f_runs(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3f import main
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
            self.assertEqual(set(report['groups']['central_flat']['summary']), {'speed', 'speed_rel'})
            self.assertEqual(set(report['groups']['local_flat']['summary']),
                             {'speed', 'speed_rel', 'speed_theta', 'speed_theta_rel'})
            fold = report['groups']['local_flat']['folds'][0]
            self.assertIn('relative_feature_gain_rank', fold['variants']['speed_theta_rel'])
            self.assertGreater(fold['theta_sd_validation'], 0)


if __name__ == '__main__':
    unittest.main()
