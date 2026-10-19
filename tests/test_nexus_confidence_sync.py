from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nexus_confidence_sync import annotate_saved_outputs
from nexus_model.forecast_confidence import from_joint


STAMP = '2026-10-19T18:00:00+09:00'
SAVED = '2026-10-19T11:00:00+09:00'


class ConfidenceMetadataSync(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.outputs = Path(self.temp.name)

    def write(self, path, value):
        target = self.outputs / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def read(self, path):
        return json.loads((self.outputs / path).read_text(encoding='utf-8'))

    def fixture(self):
        prediction = dict(model_id='saved-model', captured_at=SAVED, source_sha256='original',
                          tickets=['1-2-3', '2-1-3'], odds_at=SAVED,
                          allocations={'5000': {'available': True, 'stakes': {'1-2-3': 3000, '2-1-3': 2000}}})
        race = dict(race_id='012320261019', date='2026-10-19', prediction=prediction,
                    winners=['1-2-3'], payouts={'1-2-3': 1500}, result_complete=True,
                    result_captured_at='2026-10-19T11:15:00+09:00', result_source_sha256='result-source',
                    refunded=[], settlement={'flat': {'stake': 200, 'returned': 1500}})
        rows = [race, dict(race_id='022320261019', date='2026-10-19')]
        live = dict(updated_at=SAVED, results_updated_at='2026-10-19T17:00:00+09:00', races=rows)
        self.write('nexus_live.json', live)
        self.write('nexus_forecasts/2026-10-19.json', rows)
        conditional = dict(updated_at='2026-10-19T17:01:00+09:00', races=[
            dict(id='cond', shadow=dict(tickets=['1-2-3', '2-1-3'], snapshot_at=SAVED),
                 actual=['1-2-3'], payouts={'1-2-3': 1500}, budget={'stake': 5000}),
            dict(id='no-forecast', shadow=None)])
        self.write('conditional_live.json', conditional)
        saved_row = dict(race_id='cond', top12=['1-2-3', '2-1-3'], probabilities=[.4, .3],
                         snapshot_at_jst=SAVED, close_at='2026-10-19T11:10:00+09:00')
        ledger_path = self.outputs / 'company/conditional_live_ledger.jsonl'
        ledger_path.parent.mkdir()
        ledger_path.write_text(json.dumps(saved_row) + '\n', encoding='utf-8')
        return deepcopy(live), deepcopy(conditional), ledger_path.read_bytes()

    def test_annotation_changes_only_confidence_metadata_and_preserves_recent_results(self):
        original, conditional, ledger_bytes = self.fixture()
        report = annotate_saved_outputs(self.outputs, annotated_at=STAMP)
        self.assertTrue(report['changed'])
        live = self.read('nexus_live.json')
        after_conditional = self.read('conditional_live.json')
        self.assertEqual(live['confidence_updated_at'], STAMP)
        self.assertEqual(after_conditional['confidence_updated_at'], STAMP)
        self.assertEqual(live['races'][0]['prediction']['confidence']['status'], 'insufficient')
        self.assertEqual(after_conditional['races'][0]['shadow']['confidence']['grade'], 'S')
        self.assertEqual(self.read('nexus_forecasts/2026-10-19.json'), live['races'])
        del live['confidence_updated_at']
        del live['races'][0]['prediction']['confidence']
        del after_conditional['confidence_updated_at']
        del after_conditional['races'][0]['shadow']['confidence']
        self.assertEqual(live, original)
        self.assertEqual(after_conditional, conditional)
        self.assertEqual((self.outputs / 'company/conditional_live_ledger.jsonl').read_bytes(), ledger_bytes)

    def test_repeat_run_is_idempotent_even_with_a_later_annotation_clock(self):
        self.fixture()
        annotate_saved_outputs(self.outputs, annotated_at=STAMP)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.outputs.rglob('*') if p.is_file()}
        with patch('nexus_confidence_sync._write', side_effect=AssertionError('must not rewrite')):
            report = annotate_saved_outputs(self.outputs, annotated_at='2026-10-20T18:00:00+09:00')
        self.assertFalse(report['changed'])
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_matching_frozen_ledger_confidence_is_copied_without_replacing_prediction(self):
        original, _, _ = self.fixture()
        rows = self.read('nexus_forecasts/2026-10-19.json')
        rows[0]['prediction']['confidence'] = from_joint([1, 2, 3], [.7, .04, .06, .08, .08, .04],
                                                       ['1-2-3', '2-1-3'], annotated_at=SAVED)
        self.write('nexus_forecasts/2026-10-19.json', rows)
        annotate_saved_outputs(self.outputs, annotated_at=STAMP)
        prediction = self.read('nexus_live.json')['races'][0]['prediction']
        self.assertEqual(prediction['confidence']['status'], 'assessed')
        self.assertEqual(prediction['confidence']['annotated_at'], SAVED)
        self.assertEqual({k:v for k,v in prediction.items() if k != 'confidence'}, original['races'][0]['prediction'])

    def test_wrong_conditional_snapshot_is_never_given_a_supported_grade(self):
        self.fixture()
        feed = self.read('conditional_live.json')
        feed['races'][0]['shadow']['snapshot_at'] = '2026-10-19T11:01:00+09:00'
        self.write('conditional_live.json', feed)
        annotate_saved_outputs(self.outputs, annotated_at=STAMP)
        confidence = self.read('conditional_live.json')['races'][0]['shadow']['confidence']
        self.assertEqual(confidence['status'], 'insufficient')
        self.assertEqual(confidence['reason_code'], 'frozen_snapshot_mismatch')

    def test_corrupt_other_feed_does_not_partially_write_nexus(self):
        self.fixture()
        path = self.outputs / 'nexus_live.json'
        before = path.read_bytes()
        (self.outputs / 'conditional_live.json').write_text('{', encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError):
            annotate_saved_outputs(self.outputs, annotated_at=STAMP)
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
