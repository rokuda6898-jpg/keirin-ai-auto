"""50x shadow selection: no synthetic claims about ROI or accuracy."""
import unittest
from itertools import permutations
from conditional_high_value import select, unavailable

KEYS = ['-'.join(map(str, p)) for p in permutations(range(1, 8), 3)]
WEIGHTS = [max(.05, 15 - i / 12) for i in range(len(KEYS))]
TOTAL = sum(WEIGHTS)
PROBS = [p / TOTAL for p in WEIGHTS]
BASE = KEYS[:12]


def quote(high, missing=(), capped=()):
    prices = {key: 10 for key in KEYS if key not in missing}
    prices.update(high)
    prices.update({key: '9999.9' for key in capped})
    return {'provider': 'fixture', 'bet_type': 'trifecta',
            'source_url': 'https://winticket.jp/test',
            'captured_at_jst': '2026-10-13T12:00:00+09:00',
            'odds': prices}


class TestConditionalHighValue(unittest.TestCase):
    def test_all_combinations_eligible_including_outside_top12(self):
        # High-priced tickets ranked 13+ are not lost before looking up odds.
        picks = select(KEYS, PROBS, BASE, quote({KEYS[28]: 250, KEYS[45]: 450,
                                                 KEYS[90]: 700, KEYS[160]: 850}))
        self.assertEqual(len(picks['tickets']), 12)
        self.assertTrue(picks['new_high'])
        self.assertTrue(all(int(x['model_rank']) > 12 for x in picks['new_high']))
        self.assertTrue(set(x['buy'] for x in picks['new_high']).isdisjoint(BASE))
        self.assertEqual(picks['baseline12'], BASE)
        self.assertEqual(picks['tickets'][:12-len(picks['new_high'])],
                         BASE[:12-len(picks['new_high'])])
        self.assertFalse(picks['validated_roi'])

    def test_50_to_99_separate_from_100_plus(self):
        p = select(KEYS, PROBS, BASE, quote({KEYS[12]: 99, KEYS[13]: 100}))
        bands = {x['odds_band'] for x in p['new_high']}
        self.assertEqual(bands, {'50_to_99', '100x_plus'})

    def test_below_50_ineligible_and_no_forced_fill(self):
        p = select(KEYS, PROBS, BASE, quote({KEYS[19]: 49.9}))
        self.assertEqual(p['tickets'], BASE)
        self.assertEqual(p['new_high'], [])

    def test_market_incomplete_or_censored_baseline_is_not_repaired(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            select(KEYS, PROBS, BASE, quote({}, missing=[KEYS[-1]]))
        with self.assertRaisesRegex(ValueError, 'baseline quote is capped'):
            select(KEYS, PROBS, BASE, quote({}, capped=[BASE[0]]))
        self.assertEqual(unavailable('late')['status'], 'unavailable')

    def test_50x_selected_even_if_uncalibrated_proxy_is_below_one(self):
        # The original problem: a rank-13 outcome at 50x is discarded before
        # market inspection. Here it is a prospective RESEARCH candidate, not
        # a claim of positive EV or an authorized bet.
        p = select(KEYS, PROBS, BASE, quote({KEYS[12]: 50}))
        self.assertEqual(p['new_high'][0]['buy'], KEYS[12])
        self.assertLess(p['new_high'][0]['probability_x_odds_proxy'], 1.0)
        self.assertFalse(p['new_high'][0]['proxy_above_one'])
        self.assertFalse(p['validated_roi'])

    def test_high_censored_never_selected(self):
        p = select(KEYS, PROBS, BASE, quote({KEYS[19]: 110}, capped=[KEYS[20]]))
        self.assertEqual([x['buy'] for x in p['new_high']], [KEYS[19]])

    def test_candidate_distribution_and_baseline_cannot_be_falsified(self):
        for probs, base in (([0] * len(PROBS), BASE), (PROBS, BASE[::-1])):
            with self.assertRaises(ValueError):
                select(KEYS, probs, base, quote({}))
