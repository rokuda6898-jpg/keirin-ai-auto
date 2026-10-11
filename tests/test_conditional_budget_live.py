"""Integration with the existing conditional forecast and official normalizer."""
import json
from pathlib import Path
import tempfile
import unittest
from conditional_live import LEDGER, atomic_json
from test_conditional_live import fixture, execute, NOW


def quotes(r, clock):
    return {k:r[k] for k in ('race_id','date','venue','race_no')} | {
        'provider':'fixture', 'source_url':'https://example.test/odds',
        'bet_type':'trifecta', 'captured_at_jst':clock().isoformat(),
        'odds':dict.fromkeys(r['top12'], 20)}

def race(feed):
    return next(r for r in feed['races'] if r['id'] == 'open')

class ConditionalBudgetLiveTests(unittest.TestCase):
    def test_allocation_is_saved_and_result_does_not_reallocate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture(root)
            feed,_=execute(root, odds_loader=quotes)
            initial=race(feed); plan=initial['budget']
            self.assertEqual(plan['status'], 'allocated')
            self.assertIsNone(initial['budget_settlement']['roi'])
            buy=initial['shadow']['tickets'][0]
            atomic_json(root/'outputs/latest_results.json', [dict(race_id='open',
                official_result_available=True, actual_trifecta=buy,
                payouts_trifecta_json={buy:1100})])
            def forbidden(*args): self.fail('refetched after prediction')
            feed,_=execute(root, odds_loader=forbidden, predict_new=False)
            final=race(feed)
            self.assertEqual(final['budget'], plan)
            self.assertEqual(final['shadow']['tickets'], initial['shadow']['tickets'])
            self.assertEqual(final['budget_settlement']['payout_yen'], plan['allocations'][0]['stake_yen']//100*1100)
            settled=final['budget_settlement']
            atomic_json(root/'outputs/latest_results.json', [])
            feed,_=execute(root, odds_loader=forbidden, predict_new=False)
            self.assertEqual(race(feed)['budget_settlement'], settled)
            atomic_json(root/'outputs/latest_results.json', [dict(race_id='open', official_result_available=False)])
            feed,_=execute(root, odds_loader=forbidden, predict_new=False)
            self.assertEqual(race(feed)['budget_settlement']['status'], 'pending')

    def test_legacy_predictions_never_receive_later_odds(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture(root); execute(root, odds_loader=quotes)
            saved=json.loads((root/LEDGER).read_text())
            for key in list(saved):
                if key.startswith('budget'): del saved[key]
            (root/LEDGER).write_text(json.dumps(saved)+'\n')
            def forbidden(*args): self.fail('legacy was backfilled')
            feed,_=execute(root, odds_loader=forbidden)
            self.assertIsNone(race(feed)['budget'])
            self.assertEqual(race(feed)['shadow']['tickets'], saved['top12'])

    def test_missing_odds_preserves_forecast_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture(root)
            def fail(*args): raise ValueError('missing odds')
            feed,_=execute(root, odds_loader=fail)
            self.assertEqual(len(race(feed)['shadow']['tickets']), 12)
            self.assertEqual(race(feed)['budget']['status'], 'unavailable')
            feed,_=execute(root, odds_loader=quotes)
            self.assertEqual(race(feed)['budget']['status'], 'unavailable')

if __name__ == '__main__':
    unittest.main()
