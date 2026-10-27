"""Live-result validation, per-target causal prefixes and train/serve parity."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from keiba_ai.bias import DEFAULT_BIAS_CONFIG
from keiba_ai.cli import main, prepare
from keiba_ai.dataset import iter_races
from keiba_ai.features import FEATURE_NAMES, HistoryState, race_features
from keiba_ai.same_day import load_same_day_results, rebuild_prediction_states
from test_ai_bias_pipeline import source_race


def known_race(day, number, *, course='東京', surface='ダ'):
    race = source_race(day, number, course=course, surface=surface)
    code = {'東京': '05', '京都': '08'}[course]
    identifier = f'{day[:4]}{code}01{int(day[-2:]):02d}{number:02d}'
    for row in race:
        row.update(race_id=identifier, horse_id=f'20190000{row["horse_number"]:02d}',
                   jockey_id=f'{row["horse_number"]:05d}', trainer_id=f'{row["horse_number"]:05d}',
                   snapshot_id='snapshot-'+identifier)
    return race


def write_results(path, races, *, page='results'):
    rows = []
    for race in races:
        for item in race:
            row = dict(race_id=item['race_id'], region=item['region'], page=page,
                       race_date=item['day'], horse_id=item['horse_id'], horse_number=item['horse_number'],
                       frame=item['frame'], sex=item['sex'], age=item['age'],
                       jockey_id=item['jockey_id'], trainer_id=item['trainer_id'],
                       carried_weight_kg=item['carried_weight_kg'],
                       finish_position=item['finish_position'] if page == 'results' else None,
                       finish_status=item['finish_status'] if page == 'results' else None,
                       race_course=item['course'], race_surface=item['surface'],
                       race_distance_m=item['distance_m'], race_declared_field_size=item['declared_field_size'],
                       snapshot_id=item['snapshot_id'], race_id_verified=True,
                       odds=item['odds'], popularity=item['popularity'],
                       race_track_condition=item['track_condition'],
                       time_seconds=item.get('time_seconds') if page == 'results' else None,
                       last_3f_seconds=item.get('last_3f_seconds') if page == 'results' else None,
                       corner_positions=json.dumps(item.get('corner_positions', {})) if page == 'results' else None,
                       corner_sequence=json.dumps(item.get('corner_sequence', [])) if page == 'results' else None,
                       corner_labels=json.dumps(item.get('corner_labels')) if page == 'results' else None)
            rows.append(row)
    with Path(path).open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SameDayResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_only_same_day_course_surface_and_lower_races_are_selected(self):
        day = '2027-01-02'
        target = known_race(day, 7)
        sources = [known_race(day, number) for number in (1, 6, 7, 9)]
        sources += [known_race(day, 3, course='京都'), known_race(day, 2, surface='芝'),
                    known_race('2027-01-01', 2), known_race('2027-01-03', 2)]
        csv_path = self.root/'today.csv'
        write_results(csv_path, sources)
        with load_same_day_results(csv_path) as observed:
            state = observed.for_race(target)
            self.assertEqual([record.number for record in state.records], [1, 6])
            self.assertTrue(all(record.day.isoformat() == day for record in state.records))
            self.assertEqual({record.track for record in state.records}, {('central', '東京', 'ダ')})

    def test_target_and_later_result_changes_leave_its_vectors_unchanged(self):
        day = '2027-01-02'
        original = [known_race(day, number) for number in (1, 6, 7, 9)]
        changed = deepcopy(original)
        for race in changed[2:]:
            for row in race:
                row.update(finish_position=9-row['horse_number'], popularity=row['horse_number'],
                           track_condition='不良', odds=9999)
        first, second = self.root/'first.csv', self.root/'second.csv'
        write_results(first, original)
        write_results(second, changed)
        target = known_race(day, 7)
        with load_same_day_results(first) as a, load_same_day_results(second) as b:
            left = race_features(target, HistoryState(), intraday=a.for_race(target), full_race=target)
            right = race_features(target, HistoryState(), intraday=b.for_race(target), full_race=target)
            self.assertEqual(left, right)

    def test_each_target_gets_a_fresh_state_even_when_prediction_order_is_reversed(self):
        day = '2027-01-02'
        csv_path = self.root/'today.csv'
        write_results(csv_path, [known_race(day, number) for number in (1, 6, 7, 9)])
        with load_same_day_results(csv_path) as observed:
            ninth = observed.for_race(known_race(day, 9))
            seventh = observed.for_race(known_race(day, 7))
            seventh_again = observed.for_race(known_race(day, 7))
            self.assertEqual([record.number for record in ninth.records], [1, 6, 7])
            self.assertEqual(seventh.records, seventh_again.records)
            self.assertEqual([record.number for record in seventh.records], [1, 6])
            self.assertEqual(observed.for_race(known_race('2027-01-03', 7)).valid_races, 0)

    def test_partial_result_csv_is_rejected_before_it_can_be_used(self):
        race = known_race('2027-01-02', 1)
        csv_path = self.root/'incomplete.csv'
        write_results(csv_path, [race[:-1]])
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, '不正・不完全'):
            with load_same_day_results(csv_path):
                self.fail('An incomplete result must not be exposed to prediction')
        self.assertEqual(hashlib.sha256(csv_path.read_bytes()).hexdigest(), digest)

    def test_temporary_database_is_removed_and_source_is_immutable_on_success_and_exception(self):
        csv_path = self.root/'today.csv'
        write_results(csv_path, [known_race('2027-01-02', 1)])
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        with load_same_day_results(csv_path) as observed:
            database = observed.database
            connection = observed.connection
            self.assertEqual(observed.sha256, digest)
            self.assertTrue(database.is_file())
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute('DELETE FROM input_rows')
        self.assertFalse(database.exists())
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with load_same_day_results(csv_path) as observed:
                interrupted_database = observed.database
                raise RuntimeError('interrupted')
        self.assertFalse(interrupted_database.exists())
        self.assertEqual(hashlib.sha256(csv_path.read_bytes()).hexdigest(), digest)
        with load_same_day_results(None) as absent:
            self.assertIsNone(absent)

    def test_prediction_rebuild_matches_training_for_all_104_columns(self):
        races = [known_race(day, number)
                 for day in ('2024-01-01', '2024-01-02', '2024-01-03') for number in (1, 2, 3)]
        source = self.root/'history.csv'
        write_results(source, races)
        with redirect_stdout(io.StringIO()):
            database = prepare(source, self.root/'prepared')
        original_database = hashlib.sha256(database.read_bytes()).hexdigest()
        history, priors, last_date = rebuild_prediction_states(database, '2024-01-03')
        self.assertEqual(last_date, '2024-01-02')
        self.assertEqual(history.asof_day, '2024-01-02')
        self.assertLess(priors.latest_day.isoformat(), '2024-01-03')
        with load_same_day_results(source) as observed:
            # The source deliberately includes prior days and future/current R.
            for target in [race for race in iter_races(database) if race[0]['day'] == '2024-01-03']:
                actual = race_features(target, history, intraday=observed.for_race(target), priors=priors,
                                       full_race=target, bias_config=DEFAULT_BIAS_CONFIG)
                with sqlite3.connect(database) as connection:
                    expected = [json.loads(row[0]) for row in connection.execute(
                        'SELECT feature_json FROM features WHERE region=? AND race_id=? ORDER BY horse_number',
                        (target[0]['region'], target[0]['race_id']))]
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual[0]), len(FEATURE_NAMES))
        self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), original_database)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Install requirements-ml.txt for rank smoke test')
    def test_cli_bias_fit_predict_uses_live_results_and_preserves_rank_probability_mass(self):
        source = self.root/'history.csv'
        races = [known_race(f'{year}-01-{day:02d}', number)
                 for year in (2023, 2025, 2026) for day in range(1, 5) for number in range(1, 8)]
        write_results(source, races)
        directory = self.root/'fit'
        today, entries, output = self.root/'today.csv', self.root/'entries.csv', self.root/'predictions.csv'
        write_results(today, [known_race('2027-01-01', number) for number in range(1, 9)])
        write_results(entries, [known_race('2027-01-01', 9)], page='entries')
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, today, entries)}
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['fit', '--input', str(source), '--output-dir', str(directory),
                                   '--target', 'rank', '--feature-set', 'bias', '--threads', '1', '--iterations', '2',
                                   '--validation-samples', '8', '--evaluation-samples', '8']), 0)
            self.assertEqual(main(['predict', '--dataset', str(directory/'features.sqlite3'),
                                   '--model-dir', str(directory/'models'), '--entries', str(entries),
                                   '--same-day-results', str(today), '--output', str(output), '--samples', '32']), 0)
        with output.open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 8)
        for row in rows:
            ranks = [float(row[f'rank_{rank}_probability']) for rank in range(1, 9)]
            self.assertAlmostEqual(sum(ranks)+float(row['nonfinish_probability']), 1, places=10)
            self.assertAlmostEqual(sum(ranks[:3]), float(row['top3_probability']), places=10)
            self.assertEqual(float(row['day_bias_races']), 8)
            self.assertEqual(row['intraday_bias_used'], 'True')
            self.assertEqual(row['feature_set'], 'bias')
            self.assertEqual(row['same_day_results_sha256'], before[today])
        self.assertEqual({path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}, before)


if __name__ == '__main__':
    unittest.main()
