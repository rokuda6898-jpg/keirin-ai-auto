from copy import deepcopy
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nexus_model.forecast_confidence import (
    annotate_conditional_view, annotate_prediction, from_joint, from_saved,
)


STAMP = '2026-10-19T18:00:00+09:00'
SAVED = '2026-10-19T11:00:00+09:00'
CLOSE = '2026-10-19T11:10:00+09:00'


def saved(probabilities=(.4, .3)):
    return dict(top12=['1-2-3', '2-1-3'], probabilities=list(probabilities),
                snapshot_at_jst=SAVED, close_at=CLOSE,
                model_sha256='frozen-model', budget={'stake': 5000})


class ForecastConfidence(unittest.TestCase):
    def test_thresholds_use_selected_mass_without_normalizing_selected_tickets(self):
        for mass, grade in ((.65, 'S'), (.649999, 'A'), (.45, 'A'), (.449999, 'B'), (0, 'B')):
            with self.subTest(mass=mass):
                row = saved((mass, 0))
                answer = from_saved(row, annotated_at=STAMP)
                self.assertEqual(answer['grade'], grade)
                self.assertEqual(answer['status'], 'assessed')
                self.assertEqual(answer['model_mass'], mass)
                self.assertFalse(answer['empirical_grade_validation'])
                self.assertEqual(answer['annotated_at'], STAMP)
                self.assertEqual(answer['source_captured_at'], SAVED)

    def test_final_tickets_and_car_order_determine_support(self):
        cars = [3, 1, 2]
        orders = list(itertools.permutations(cars, 3))
        q = [.7 if order == (3, 1, 2) else .06 for order in orders]
        strong = from_joint(cars, q, ['3-1-2'], annotated_at=STAMP)
        weak = from_joint(cars, q, ['1-2-3'], annotated_at=STAMP)
        self.assertEqual(strong['grade'], 'S')
        self.assertEqual(weak['grade'], 'B')
        self.assertAlmostEqual(strong['model_mass'], .7)
        self.assertAlmostEqual(weak['model_mass'], .06)

    def test_same_distribution_and_tickets_ignore_favorite_category_odds_and_result(self):
        original = saved()
        altered = dict(original, category='HIGH', marks={'1': '◎'},
                       budget={'stake': 10000}, odds={'1-2-3': 999.9}, actual=['3-1-2'])
        self.assertEqual(from_saved(original, annotated_at=STAMP), from_saved(altered, annotated_at=STAMP))

    def test_entropy_describes_complete_distribution(self):
        uniform = from_joint([1, 2, 3], [1/6] * 6, ['1-2-3'], annotated_at=STAMP)
        concentrated = from_joint([1, 2, 3], [1, 0, 0, 0, 0, 0], ['1-2-3'], annotated_at=STAMP)
        self.assertAlmostEqual(uniform['normalized_entropy'], 1)
        self.assertEqual(concentrated['normalized_entropy'], 0)
        self.assertEqual(concentrated['model_mass'], 1)
        self.assertEqual(concentrated['grade'], 'S')
        self.assertNotIn('normalized_entropy', from_saved(saved(), annotated_at=STAMP))

    def test_duplicate_tickets_never_inflate_confidence(self):
        answer = from_joint([1, 2, 3], [1/6]*6, ['1-2-3', '1-2-3'], annotated_at=STAMP)
        self.assertEqual((answer['grade'], answer['status']), ('B', 'insufficient'))
        self.assertIsNone(answer['model_mass'])

    def test_malformed_full_distribution_is_insufficient(self):
        for cars, values in (([1, 2, 3], [.2] * 6), ([1, 2, 3], [.5, .5]),
                             ([1, 2, 3], [float('nan')] * 6), ([1, 2, 3], [float('inf')] * 6),
                             ([1, 2, 3], [-.1, .3, .2, .2, .2, .2]),
                             ([1, 1, 3], [1/6] * 6), (None, None)):
            with self.subTest(cars=cars, values=values):
                answer = from_joint(cars, values, ['1-2-3'], annotated_at=STAMP)
                self.assertEqual(answer['status'], 'insufficient')
                self.assertIsNone(answer['model_mass'])

    def test_saved_probabilities_require_alignment_and_preclose_receipt(self):
        cases = [dict(probabilities=[.7]), dict(probabilities=[.8, .8]),
                 dict(probabilities=[.5, float('nan')]), dict(snapshot_at_jst=CLOSE),
                 dict(snapshot_at_jst='2026-10-19T11:00:00'), dict(close_at=None)]
        for changes in cases:
            with self.subTest(changes=changes):
                answer = from_saved(dict(saved(), **changes), annotated_at=STAMP)
                self.assertEqual((answer['grade'], answer['status']), ('B', 'insufficient'))
                self.assertIsNone(answer['model_mass'])
        mismatch = from_saved(saved(), tickets=['3-2-1'], annotated_at=STAMP)
        self.assertEqual(mismatch['reason_code'], 'frozen_snapshot_mismatch')
        wrong_time = from_saved(saved(), snapshot_at='2026-10-19T11:01:00+09:00', annotated_at=STAMP)
        self.assertEqual(wrong_time['reason_code'], 'frozen_snapshot_mismatch')

    def test_frozen_snapshot_can_be_annotated_after_the_race_without_reprediction(self):
        row = saved()
        before = deepcopy(row)
        view = dict(tickets=row['top12'][:], snapshot_at=SAVED, marks=['◎', '○'])
        answer = annotate_conditional_view(view, row, annotated_at=STAMP)
        self.assertEqual(answer['grade'], 'S')
        self.assertEqual(row, before)
        self.assertEqual(view['tickets'], before['top12'])
        self.assertEqual(view['snapshot_at'], SAVED)
        self.assertEqual(view['confidence'], answer)

    def test_existing_nexus_prediction_gets_insufficient_metadata_only(self):
        prediction = dict(tickets=['1-2-3'], captured_at=SAVED,
                          first_probabilities={'1': .99, '2': .01}, category='FAVORITE',
                          allocations={'5000': {'stakes': {'1-2-3': 5000}}})
        before = deepcopy(prediction)
        answer = annotate_prediction(prediction, annotated_at=STAMP)
        self.assertEqual((answer['grade'], answer['status']), ('B', 'insufficient'))
        self.assertEqual({key:value for key,value in prediction.items() if key != 'confidence'}, before)
        self.assertIs(annotate_prediction(prediction, annotated_at='2026-10-20T00:00:00+09:00'), answer)
        self.assertEqual(answer['annotated_at'], STAMP)

    def test_assessed_engine_metadata_is_preserved_and_receipt_added(self):
        confidence = from_joint([1, 2, 3], [1, 0, 0, 0, 0, 0], ['1-2-3'], annotated_at=STAMP)
        prediction = dict(tickets=['1-2-3'], captured_at=SAVED, confidence=confidence)
        self.assertIs(annotate_prediction(prediction), confidence)
        self.assertEqual(confidence['grade'], 'S')
        self.assertEqual(confidence['source_captured_at'], SAVED)
        self.assertEqual(confidence['annotated_at'], STAMP)

    def test_nonexistent_forecast_gets_no_confidence(self):
        for value in (None, {}, {'tickets': []}):
            with self.subTest(value=value):
                self.assertIsNone(annotate_prediction(value))
                self.assertIsNone(annotate_conditional_view(value, saved()))
                if value is not None:
                    self.assertNotIn('confidence', value)


if __name__ == '__main__':
    unittest.main()
