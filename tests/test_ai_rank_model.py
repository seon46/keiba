from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from keiba_ai.cli import prepare,predict
from keiba_ai.rank_model import load_rank_group,train_rank
from test_ai_model import fixture_csv


@unittest.skipUnless(importlib.util.find_spec('sklearn'),'Install requirements-ml.txt for rank model tests')
class RankModelTests(unittest.TestCase):
    def test_nonfinish_model_has_separate_probability_and_no_invented_last_rank(self):
        import joblib
        with TemporaryDirectory() as name,redirect_stdout(io.StringIO()):
            root=Path(name)
            fixture_csv(root/'source.csv',days=20)
            with (root/'source.csv').open(encoding='utf-8',newline='') as file:
                reader=csv.DictReader(file);fields=reader.fieldnames;rows=list(reader)
            for row in rows:
                if row['horse_number']=='10':
                    row.update(finish_status='did_not_finish',finish_position='',time_seconds='',last_3f_seconds='')
            with (root/'source.csv').open('w',encoding='utf-8',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            database=prepare(root/'source.csv',root/'prepared')
            report=train_rank(database,root/'models',threads=1,iterations=3,feature_set='enhanced',
                              validation_samples=16,evaluation_samples=32)
            bundle=joblib.load(root/'models/central_flat.joblib')
            self.assertIsNotNone(bundle['finish_estimator'])
            self.assertEqual(report['groups']['central_flat']['test']['nonfinished_rows'],20)
            fixture_csv(root/'entries.csv',years=(2027,),days=1,page='entries')
            predict(database,root/'models',root/'entries.csv',root/'predictions.csv',samples=64)
            with (root/'predictions.csv').open(encoding='utf-8',newline='') as file:
                output=list(csv.DictReader(file))
            self.assertTrue(any(float(row['nonfinish_probability'])>0 for row in output))
            self.assertLess(sum(float(row['rank_10_probability']) for row in output),1)
            for row in output:
                ranks=sum(float(row[f'rank_{rank}_probability']) for rank in range(1,11))
                self.assertAlmostEqual(ranks+float(row['nonfinish_probability']),1,places=10)

    def test_ranking_excludes_ties_but_retains_nonfinish_and_history(self):
        with TemporaryDirectory() as name,redirect_stdout(io.StringIO()):
            root=Path(name)
            fixture_csv(root/'source.csv',years=(2023,),days=3)
            database=prepare(root/'source.csv',root/'prepared')
            connection=sqlite3.connect(database)
            try:
                with connection:
                    connection.execute("UPDATE input_rows SET payload_json=json_set(payload_json,'$.finish_position',3) "
                                       "WHERE day='2023-01-01' AND horse_number=4")
                    connection.execute("UPDATE input_rows SET payload_json=json_set(payload_json,'$.finish_position',NULL,"
                                       "'$.finish_status','did_not_finish') WHERE day='2023-01-02' AND horse_number=10")
                data=load_rank_group(database,'central_flat')
                self.assertEqual(data['excluded_rows'],10)
                self.assertEqual(len(data['identities']),20)
                self.assertEqual(int((~data['finished']).sum()),1)
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM features').fetchone()[0],30)
            finally:
                connection.close()

    def test_rank_fit_probabilities_and_test_outcomes_cannot_select_model(self):
        import joblib
        import numpy as np
        from keiba_ai.model import select_feature_columns
        from keiba_ai.features import FEATURE_NAMES
        with TemporaryDirectory() as name,redirect_stdout(io.StringIO()):
            root=Path(name)
            fixture_csv(root/'source.csv',days=20)
            database=prepare(root/'source.csv',root/'prepared')
            report=train_rank(database,root/'models',threads=1,iterations=3,feature_set='enhanced',
                              validation_samples=16,evaluation_samples=32)
            group=report['groups']['central_flat']
            self.assertEqual(group['state'],'trained')
            self.assertEqual(group['test']['rows'],200)
            self.assertEqual(group['test']['top3']['races'],20)
            with (root/'models'/group['test_predictions_file']).open(encoding='utf-8',newline='') as file:
                historical=list(csv.DictReader(file))
            self.assertEqual(len(historical),200)
            self.assertTrue(all(row['race_date']>='2026-01-01' for row in historical))
            self.assertTrue(all(abs(sum(float(row[f'rank_{rank}_probability']) for rank in range(1,11))+
                                   float(row['nonfinish_probability'])-1)<1e-10 for row in historical))
            original=joblib.load(root/'models/central_flat.joblib')
            self.assertEqual(original['target'],'rank')
            connection=sqlite3.connect(database)
            try:
                with connection:
                    connection.execute("UPDATE input_rows SET payload_json=json_set(payload_json,'$.finish_position',"
                                       "11-json_extract(payload_json,'$.finish_position')) WHERE day>='2026-01-01'")
                    connection.execute("UPDATE features SET target=CASE WHEN horse_number>=8 THEN 1 ELSE 0 END WHERE day>='2026-01-01'")
            finally:
                connection.close()
            changed=train_rank(database,root/'changed',threads=1,iterations=3,feature_set='enhanced',
                               validation_samples=16,evaluation_samples=32)
            other=joblib.load(root/'changed/central_flat.joblib')
            self.assertEqual(group['selected_temperature'],changed['groups']['central_flat']['selected_temperature'])
            self.assertEqual(group['validation_comparison'],changed['groups']['central_flat']['validation_comparison'])
            probes=np.zeros((2,len(FEATURE_NAMES)))[:,select_feature_columns(original)]
            np.testing.assert_array_equal(original['rank_estimator'].predict(probes),other['rank_estimator'].predict(probes))
            self.assertNotEqual(group['test']['race_outcome_log_loss'],changed['groups']['central_flat']['test']['race_outcome_log_loss'])
            fixture_csv(root/'entries.csv',years=(2027,),days=2,page='entries')
            with (root/'entries.csv').open(encoding='utf-8',newline='') as file:
                reader=csv.DictReader(file)
                fields=reader.fieldnames
                rows=list(reader)
            rows=[row for row in rows if row['race_date']=='2027-01-01' or int(row['horse_number'])<=6]
            for row in rows:
                if row['race_date']=='2027-01-02': row['race_declared_field_size']='6'
            with (root/'entries.csv').open('w',encoding='utf-8',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            predict(database,root/'models',root/'entries.csv',root/'predictions.csv',samples=128)
            with (root/'predictions.csv').open(encoding='utf-8',newline='') as file:
                output=list(csv.DictReader(file))
            self.assertEqual(len(output),16)
            for row in output:
                ranks=[float(row[f'rank_{rank}_probability']) for rank in range(1,11)]
                self.assertAlmostEqual(sum(ranks)+float(row['nonfinish_probability']),1,places=10)
                self.assertAlmostEqual(sum(ranks[:3]),float(row['top3_probability']),places=10)
                if row['starters']=='6': self.assertEqual(ranks[6:],[0]*4)
            for day in ('2027-01-01','2027-01-02'):
                race=[row for row in output if row['race_date']==day]
                for rank in range(1,len(race)+1):
                    self.assertAlmostEqual(sum(float(row[f'rank_{rank}_probability']) for row in race),1,places=10)


if __name__=='__main__':
    unittest.main()
