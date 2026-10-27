from contextlib import closing, redirect_stdout
import csv
from datetime import date, timedelta
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from keiba_ai.cli import prepare, predict
from keiba_ai.model import evaluate, select_feature_columns, train


def fixture_csv(path, *, years=(2023,2025,2026), days=40, page='results'):
    fields = ['race_id','region','page','race_date','horse_id','horse_number','frame','sex','age',
              'jockey_id','trainer_id','carried_weight_kg','finish_position','finish_status',
              'race_course','race_surface','race_distance_m','race_declared_field_size','snapshot_id','race_id_verified',
              'time_seconds','last_3f_seconds']
    with path.open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        for year in years:
            for number in range(days):
                for horse in range(1,11):
                    writer.writerow(dict(race_id=f'{year}05{number:04d}01',region='central',page=page,
                        race_date=(date(year,1,1)+timedelta(days=number)).isoformat(),
                        horse_id=f'20190000{horse:02d}',horse_number=horse,frame=(horse+1)//2,
                        sex='牡',age=4,jockey_id=f'{horse:05d}',trainer_id=f'{horse:05d}',carried_weight_kg=57,
                        finish_position=horse if page=='results' else '',finish_status='finished' if page=='results' else '',
                        race_course='東京',race_surface='芝',race_distance_m=1600,race_declared_field_size=10,
                        snapshot_id=f'snapshot-{year}-{number}',race_id_verified='True',
                        time_seconds=95+horse if page=='results' else '',
                        last_3f_seconds=33+horse/5 if page=='results' else ''))


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Install requirements-ml.txt for model tests')
class ModelTests(unittest.TestCase):
    def test_preparation_closes_sqlite_handles_for_windows_directory_move(self):
        real_connect=sqlite3.connect
        opened=[]
        def connect(*args,**kwargs):
            connection=real_connect(*args,**kwargs)
            opened.append(connection)
            return connection
        with TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            root=Path(name)
            fixture_csv(root/'source.csv',years=(2023,),days=1)
            with patch('sqlite3.connect',side_effect=connect):
                prepare(root/'source.csv',root/'prepared')
            self.assertGreater(len(opened),0)
            for connection in opened:
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute('SELECT 1')

    def test_calibration_counts_all_boundary_probabilities_once(self):
        probabilities=[i/10 for i in range(11)]+[.6,.6]
        result=evaluate([0]*len(probabilities),probabilities,[(str(1),str(i),i) for i in range(len(probabilities))])
        self.assertEqual(sum(row['rows'] for row in result['calibration']),len(probabilities))
        self.assertEqual(next(row for row in result['calibration'] if row['lower']==.6)['rows'],3)

    def test_noncanonical_split_dates_are_rejected_before_reading_data(self):
        with self.assertRaisesRegex(ValueError,'YYYY-MM-DD'):
            train(Path('missing.sqlite3'),Path('unused'),validation_start='20250101')

    def test_top3_probabilities_are_not_forced_to_one_and_ties_are_valid(self):
        result = evaluate([1,1,1,1,0],[.8,.8,.8,.8,.1],[(str(1),str(i),i) for i in range(5)])
        self.assertEqual(result['positives'],4)
        self.assertEqual(result['precision_at_3'],1)
        self.assertLess(result['brier'],.05)

    def test_fit_predict_and_test_labels_do_not_change_the_model(self):
        import joblib
        import numpy as np
        from keiba_ai.features import FEATURE_NAMES
        with TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            root=Path(name)
            source=root/'results.csv'
            fixture_csv(source)
            database=prepare(source,root/'prepared')
            report=train(database,root/'model',threads=1,iterations=8)
            group=report['groups']['central_flat']
            self.assertEqual(group['state'],'trained')
            self.assertLess(group['trained_through'],'2026-01-01')
            self.assertEqual(group['test']['rows'],400)
            self.assertEqual(group['test']['races'],40)
            comparison=group['validation_comparison']
            self.assertEqual(set(comparison),{'baseline','form','enhanced','context'})
            self.assertLessEqual(group['validation']['race_log_loss'],comparison['baseline']['metrics']['race_log_loss'])
            self.assertEqual(group['validation']['race_log_loss'],min(item['metrics']['race_log_loss'] for item in comparison.values()))
            original=joblib.load(root/'model/central_flat.joblib')
            with closing(sqlite3.connect(database)) as db, db:
                # Alter only held-out labels. The fitted estimator and selection
                # must remain identical; the final evaluation must change.
                db.execute("UPDATE features SET target=1-target WHERE day>='2026-01-01'")
            changed=train(database,root/'changed',threads=1,iterations=8)
            alternative=joblib.load(root/'changed/central_flat.joblib')
            probes=np.zeros((2,len(FEATURE_NAMES)),dtype=np.float32)
            np.testing.assert_array_equal(original['estimator'].predict_proba(probes[:,select_feature_columns(original)]),
                                          alternative['estimator'].predict_proba(probes[:,select_feature_columns(alternative)]))
            self.assertEqual(group['selected_max_leaf_nodes'],changed['groups']['central_flat']['selected_max_leaf_nodes'])
            self.assertEqual(group['selected_feature_set'],changed['groups']['central_flat']['selected_feature_set'])
            self.assertEqual(group['validation_comparison'],changed['groups']['central_flat']['validation_comparison'])
            self.assertNotEqual(group['test']['log_loss'],changed['groups']['central_flat']['test']['log_loss'])
            entries=root/'entries.csv'
            fixture_csv(entries,years=(2027,),days=1,page='entries')
            predictions=root/'predictions.csv'
            predict(database,root/'model',entries,predictions)
            with predictions.open(encoding='utf-8',newline='') as file:
                rows=list(csv.DictReader(file))
            self.assertEqual(len(rows),10)
            self.assertTrue(all(0<=float(row['top3_probability'])<=1 for row in rows))
            self.assertTrue(all(row['history_last_date']<'2027-01-01' for row in rows))
            wrong=dict(original,group='local_flat')
            joblib.dump(wrong,root/'model/central_flat.joblib')
            with self.assertRaisesRegex(ValueError,'区分'):
                predict(database,root/'model',entries,root/'wrong-group.csv')
            joblib.dump(original,root/'model/central_flat.joblib')
            fixture_csv(entries,years=(2025,),days=1,page='entries')
            with self.assertRaisesRegex(ValueError,'学習期間'):
                predict(database,root/'model',entries,root/'old-predictions.csv')
            with self.assertRaises(ValueError):
                prepare(source,root/'prepared')

    def test_each_explicit_profile_predicts_using_its_selected_columns(self):
        import joblib
        from keiba_ai.features import BASE_FEATURE_NAMES, RANK_FEATURE_NAMES, FEATURE_NAMES
        with TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            root=Path(name)
            fixture_csv(root/'source.csv',days=20)
            fixture_csv(root/'entries.csv',years=(2027,),days=1,page='entries')
            database=prepare(root/'source.csv',root/'prepared')
            for profile,names in (('baseline',BASE_FEATURE_NAMES),('enhanced',RANK_FEATURE_NAMES),('context',FEATURE_NAMES)):
                directory=root/profile
                result=train(database,directory,threads=1,iterations=3,feature_set=profile)
                bundle=joblib.load(directory/'central_flat.joblib')
                self.assertEqual(bundle['selected_feature_names'],list(names))
                self.assertEqual(bundle['estimator'].n_features_in_,len(names))
                self.assertEqual(result['groups']['central_flat']['selected_feature_set'],profile)
                predict(database,directory,root/'entries.csv',root/(profile+'.csv'))
                self.assertTrue((root/(profile+'.csv')).is_file())

    def test_auto_keeps_baseline_when_new_features_offer_no_validation_gain(self):
        from keiba_ai.features import BASE_FEATURE_NAMES
        with TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            root=Path(name)
            fixture_csv(root/'source.csv',days=20)
            database=prepare(root/'source.csv',root/'prepared')
            # Equivalent candidates represent an extension with no additional
            # information. A newer profile alone must not justify promotion.
            profiles={key:list(BASE_FEATURE_NAMES) for key in ('baseline','enhanced')}
            with patch('keiba_ai.model.feature_sets',return_value=profiles):
                report=train(database,root/'models',threads=1,iterations=3)
            group=report['groups']['central_flat']
            self.assertEqual(group['selected_feature_set'],'baseline')
            self.assertEqual(group['validation_comparison']['baseline']['metrics'],
                             group['validation_comparison']['enhanced']['metrics'])


if __name__=='__main__':
    unittest.main()
