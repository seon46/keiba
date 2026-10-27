from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from keiba_ai.metadata import RaceMetadata
from keiba_ai.cli import prepare
from keiba_scraper.cli import main as export
from keiba_scraper.storage import Store


RACE='202405010101'
HTML='''<html><head><title>架空賞 結果 | 2024年1月1日 東京1R</title></head>
<body><h1 class="RaceName">架空賞</h1><div class="RaceData02">
サラ系3歳 1勝クラス 馬齢 10頭 本賞金:1000,400,250万円</div>
<table><tr><td>関係のない後続本文 C3 ハンデ</td></tr></table></body></html>'''


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/'data'
        store=Store(self.source)
        observation=store.observe(f'https://race.netkeiba.com/race/result.html?race_id={RACE}',HTML.encode(),encoding='utf-8')
        self.observation=observation
        self.race=dict(race_id=RACE,region='central',page='results',race_date='2024-01-01',
            race_name='架空賞',title='架空賞 結果 | 2024年1月1日 東京1R',course='東京',
            surface='芝',distance_m=1600,declared_field_size=10,race_id_verified=True,
            weather='晴',track_condition='良')
        self.runners=[dict(horse_id=f'20190000{i:02}',horse_number=i,sex='牡',age=3,
            finish_status='finished',finish_position=i,weight_kg=480+i,weight_change_kg=-2,
            odds=float(i+1),popularity=i) for i in range(1,11)]
        store.save_race(observation,dict(race=self.race,runners=self.runners))
        store.close()
        self.raw=dict(snapshot_id=observation.id,race_id=RACE,region='central',race_date='2024-01-01')

    def test_html_is_enriched_once_without_modifying_input_or_original_files(self):
        database=self.source/'keiba.sqlite3'
        original=database.read_bytes()
        raw=(self.source/self.observation.raw_path).read_bytes()
        with RaceMetadata(database) as provider:
            first=provider(self.raw)
            self.assertEqual(first['race_class'],'1win')
            self.assertEqual(first['race_weight_rule'],'馬齢')
            self.assertEqual(first['race_first_prize_man'],1000)
            self.assertEqual(first['race_age_min'],3)
            for _ in range(9): self.assertEqual(provider(self.raw),first)
            self.assertEqual(provider.summary()['from_html'],1)
            self.assertEqual(provider.summary()['cache_hits'],9)
        self.assertNotIn('race_class',self.raw)
        self.assertEqual(database.read_bytes(),original)
        self.assertEqual((self.source/self.observation.raw_path).read_bytes(),raw)

    def test_missing_or_mismatched_snapshot_never_guesses_another_race(self):
        with RaceMetadata(self.source/'keiba.sqlite3') as provider:
            self.assertEqual(provider(dict(self.raw,snapshot_id='missing')), {})
            self.assertEqual(provider(dict(self.raw,race_id='202405010102')), {})
            self.assertEqual(provider(dict(self.raw,region='local')), {})
            self.assertEqual(provider(dict(self.raw,snapshot_id='')), {})
            stats=provider.summary()
            self.assertEqual(stats['missing_snapshot'],1)
            self.assertEqual(stats['identity_mismatch'],2)
            self.assertEqual(stats['rows_without_snapshot'],1)

    def test_corrupt_original_is_reported_without_becoming_conditions(self):
        (self.source/self.observation.raw_path).write_bytes(b'corrupted gzip')
        with RaceMetadata(self.source/'keiba.sqlite3') as provider:
            self.assertEqual(provider(self.raw),{})
            self.assertEqual(provider.summary()['unusable_raw'],1)

    def test_already_structured_metadata_avoids_loading_html(self):
        store=Store(self.source)
        store.save_race(self.observation,dict(race=dict(self.race,conditions_raw='3歳 1勝クラス 馬齢 10頭',
                                                       race_class='1win'),runners=self.runners))
        store.close()
        (self.source/self.observation.raw_path).unlink()
        with RaceMetadata(self.source/'keiba.sqlite3') as provider:
            self.assertEqual(provider(self.raw)['race_class'],'1win')
            self.assertEqual(provider.summary()['from_metadata'],1)

    def test_real_export_and_prepare_attach_static_conditions_to_ai_only(self):
        csv=self.root/'results.csv'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(export(['--data-dir',str(self.source),'export','--page','results','--output',str(csv)]),0)
            digest=hashlib.sha256(csv.read_bytes()).hexdigest()
            database=prepare(csv,self.root/'prepared',race_db=self.source/'keiba.sqlite3')
        import sqlite3
        connection=sqlite3.connect(database)
        try:
            row=json.loads(connection.execute('SELECT payload_json FROM input_rows ORDER BY horse_number LIMIT 1').fetchone()[0])
            self.assertEqual(row['race_class'],'1win')
            self.assertEqual(row['weight_kg'],481)
            self.assertEqual(row['weather'],'晴')
            quality=json.loads((self.root/'prepared/quality_report.json').read_text())['quality']
            self.assertEqual(quality['metadata_enrichment']['from_html'],1)
        finally:
            connection.close()
        self.assertEqual(hashlib.sha256(csv.read_bytes()).hexdigest(),digest)


if __name__=='__main__':
    unittest.main()
