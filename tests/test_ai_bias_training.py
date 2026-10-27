from contextlib import redirect_stdout
import csv
from datetime import date, timedelta
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from keiba_ai.cli import prepare
from keiba_ai.model import load_bias_variant, load_group
from keiba_ai.rank_model import load_rank_group, train_rank
from test_ai_model import fixture_csv


def bias_fixture(path, *, years=(2023,2025,2026), sessions=8, surface='芝'):
    """Three races per day, with pre-existing market/corner observations."""
    fixture_csv(path, years=years, days=sessions*3)
    with path.open(encoding='utf-8',newline='') as file:
        reader=csv.DictReader(file)
        fields=reader.fieldnames+['popularity','odds','corner_order','race_track_condition']
        rows=list(reader)
    for row in rows:
        original=date.fromisoformat(row['race_date'])
        ordinal=(original-date(original.year,1,1)).days
        session, slot=divmod(ordinal,3)
        number=(1,5,9)[slot]
        horse=int(row['horse_number'])
        row.update(race_date=(date(original.year,1,1)+timedelta(days=session)).isoformat(),
                   race_id=f'{original.year}0501{session+1:02d}{number:02d}',race_surface=surface,
                   popularity=str(horse),odds=str(1.5+horse),corner_order=str(horse),
                   finish_position=str(((horse+slot-1)%10)+1),race_track_condition='良',
                   snapshot_id=f'{original.year}-{session}-{number}')
    with path.open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


@unittest.skipUnless(importlib.util.find_spec('sklearn'),'Install requirements-ml.txt for bias training tests')
class BiasTrainingTests(unittest.TestCase):
    def test_variant_loader_replaces_only_bias_columns_and_refuses_partial_coverage(self):
        import numpy as np
        from keiba_ai.bias import BIAS_CONFIGS
        from keiba_ai.features import CONTEXT_FEATURE_NAMES
        with TemporaryDirectory() as name,redirect_stdout(io.StringIO()):
            root=Path(name)
            bias_fixture(root/'source.csv',years=(2023,),sessions=2)
            database=prepare(root/'source.csv',root/'prepared')
            data=load_group(database,'central_flat')
            original=data['x'].copy()
            variant=BIAS_CONFIGS[0]
            load_bias_variant(database,data,variant.identifier)
            np.testing.assert_array_equal(data['x'][:,:len(CONTEXT_FEATURE_NAMES)],
                                          original[:,:len(CONTEXT_FEATURE_NAMES)])
            self.assertTrue(np.isfinite(data['race_numbers']).all())
            with sqlite3.connect(database) as connection:
                row=connection.execute('SELECT region,race_id,horse_id FROM bias_variants '
                                       'WHERE config_id=? LIMIT 1',(variant.identifier,)).fetchone()
                connection.execute('DELETE FROM bias_variants WHERE config_id=? AND region=? AND race_id=? AND horse_id=?',
                                   (variant.identifier,*row))
            before=data['x'].copy()
            with self.assertRaisesRegex(ValueError,'不足'):
                load_bias_variant(database,data,variant.identifier)
            np.testing.assert_array_equal(data['x'],before)

    def test_bias_config_is_selected_on_validation_and_band_metrics_reuse_all_test_races(self):
        import joblib
        import numpy as np
        from keiba_ai.bias import BIAS_CONFIGS
        from keiba_ai.features import FEATURE_NAMES
        with TemporaryDirectory() as name,redirect_stdout(io.StringIO()):
            root=Path(name)
            bias_fixture(root/'source.csv')
            database=prepare(root/'source.csv',root/'prepared')
            report=train_rank(database,root/'models',threads=1,iterations=2,feature_set='bias',
                              validation_samples=8,evaluation_samples=16)
            group=report['groups']['central_flat']
            bundle=joblib.load(root/'models/central_flat.joblib')
            self.assertEqual(bundle['feature_set'],'bias')
            self.assertEqual(bundle['rank_estimator'].n_features_in_,104)
            self.assertEqual(group['selected_bias_config'],bundle['bias_config'])
            self.assertEqual(set(group['bias_config_comparison']),{config.identifier for config in BIAS_CONFIGS})
            self.assertEqual(group['validation']['race_outcome_log_loss'],min(
                value['metrics']['race_outcome_log_loss'] for value in group['bias_config_comparison'].values()))
            test=group['test']
            self.assertEqual(test['races'],24)
            for key in ('by_race_number','by_intraday_support','by_first_race'):
                self.assertEqual(sum(value['races'] for value in test[key].values()),test['races'])
                self.assertEqual(sum(value['rows'] for value in test[key].values()),test['rows'])
                weighted=sum(value['race_outcome_log_loss']*value['races'] for value in test[key].values()
                             if value['races'])/test['races']
                self.assertAlmostEqual(weighted,test['race_outcome_log_loss'])
            self.assertEqual(test['by_first_race']['first']['races'],8)
            self.assertEqual(test['by_intraday_support']['positive']['races'],16)
            with (root/'models'/group['test_predictions_file']).open(encoding='utf-8',newline='') as file:
                predictions=list(csv.DictReader(file))
            for row in predictions:
                self.assertAlmostEqual(sum(float(row[f'rank_{rank}_probability']) for rank in range(1,11))+
                                       float(row['nonfinish_probability']),1,places=10)
            with sqlite3.connect(database) as connection:
                connection.execute("UPDATE input_rows SET payload_json=json_set(payload_json,'$.finish_position',"
                                   "11-json_extract(payload_json,'$.finish_position')) WHERE day>='2026-01-01'")
                connection.execute("UPDATE features SET target=1-target WHERE day>='2026-01-01'")
            changed=train_rank(database,root/'changed',threads=1,iterations=2,feature_set='bias',
                               validation_samples=8,evaluation_samples=16)
            alternative=joblib.load(root/'changed/central_flat.joblib')
            self.assertEqual(group['selected_bias_config'],changed['groups']['central_flat']['selected_bias_config'])
            self.assertEqual(group['bias_config_comparison'],changed['groups']['central_flat']['bias_config_comparison'])
            self.assertNotEqual(group['test']['race_outcome_log_loss'],changed['groups']['central_flat']['test']['race_outcome_log_loss'])
            probe=np.zeros((2,len(FEATURE_NAMES)),dtype=np.float32)
            np.testing.assert_array_equal(bundle['rank_estimator'].predict(probe),
                                          alternative['rank_estimator'].predict(probe))

    def test_rank_auto_compares_legacy_context_and_bias_without_forcing_promotion(self):
        with TemporaryDirectory() as name,redirect_stdout(io.StringIO()):
            root=Path(name)
            bias_fixture(root/'source.csv')
            database=prepare(root/'source.csv',root/'prepared')
            result=train_rank(database,root/'models',threads=1,iterations=1,feature_set='auto',
                              validation_samples=8,evaluation_samples=8)
            group=result['groups']['central_flat']
            self.assertEqual(set(group['validation_comparison']),{'enhanced','context','bias'})
            self.assertEqual(group['validation']['race_outcome_log_loss'],min(
                value['metrics']['race_outcome_log_loss'] for value in group['validation_comparison'].values()))

    def test_jump_group_retains_legacy_candidates_and_rejects_forced_bias(self):
        with TemporaryDirectory() as name,redirect_stdout(io.StringIO()):
            root=Path(name)
            bias_fixture(root/'source.csv',surface='障')
            database=prepare(root/'source.csv',root/'prepared')
            report=train_rank(database,root/'auto',threads=1,iterations=1,feature_set='auto',
                              validation_samples=8,evaluation_samples=8)
            group=report['groups']['central_jump']
            self.assertEqual(set(group['validation_comparison']),{'enhanced','context'})
            self.assertIsNone(group['selected_bias_config'])
            with self.assertRaisesRegex(ValueError,'学習できる区分'):
                train_rank(database,root/'forced',threads=1,iterations=1,feature_set='bias',
                           validation_samples=8,evaluation_samples=8)
            skipped=json.loads((root/'forced/evaluation.json').read_text())['groups']['central_jump']
            self.assertEqual(skipped['state'],'skipped')


if __name__=='__main__':
    unittest.main()
