import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import fusion_shadow_live
import settle_results


class SettleResultsTests(unittest.TestCase):
    def test_tied_result_settles_both_winners_and_keeps_both_payoffs(self):
        entries = pd.DataFrame([{'race_id':'test','car_no':car} for car in (1,2,3,4)])
        page = 'レースが確定しました 1着 1 1 2着 2 2 3着 3 3 3着 4 4 ３連単 1>2>3 10,000円 1>2>4 30,000円 ３連複'
        result = settle_results.parse_netkeirin_result_html(page,entries,'https://example.test/result')
        self.assertEqual(result['actual_trifecta_buys'],['1-2-3','1-2-4'])
        frame = settle_results.official_trifecta_frame(pd.DataFrame([result]))
        self.assertEqual(frame.iloc[0].actual_trifecta,'1-2-3|1-2-4')
        import json
        self.assertEqual(json.loads(result['payouts_trifecta_json']),{'1-2-3':10000,'1-2-4':30000})
        row = pd.DataFrame([{'actual_trifecta':'1-2-3|2-1-3','actual_winner_car_no':1,
                             'predicted_winner_car_no':2}])
        self.assertTrue(settle_results.top1_hits(row,'predicted_winner_car_no').iloc[0])

    def test_day_rollover_preserves_confirmation_and_accepts_new_results(self):
        prior = pd.DataFrame([
            {"race_id":"old", "bet_type":"trifecta", "buy":"1-2-3", "is_decided":True, "actual_return_yen":500},
            {"race_id":"new", "bet_type":"trifecta", "buy":"2-1-3", "is_decided":False, "actual_return_yen":0},
        ])
        current = pd.DataFrame([
            {"race_id":"old", "bet_type":"trifecta", "buy":"1-2-3", "is_decided":False, "actual_return_yen":0},
            {"race_id":"new", "bet_type":"trifecta", "buy":"2-1-3", "is_decided":True, "actual_return_yen":800},
        ])
        kept = settle_results.preserve_decided_settlements(prior, current).set_index("race_id")
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept.loc["old", "actual_return_yen"], 500)
        self.assertEqual(kept.loc["new", "actual_return_yen"], 800)
        self.assertTrue(kept.is_decided.all())

    def test_empty_bet_file_writes_zero_summary_without_fetching_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            latest_bets = output_dir / "latest_bets.csv"
            settled_bets = output_dir / "settled_bets.csv"
            purchase_plan = output_dir / "purchase_plan.csv"
            summary_csv = output_dir / "settlement_summary.csv"
            report_md = output_dir / "japanese_report.md"
            prediction_ledger = output_dir / "prediction_ledger.csv"
            prediction_history = output_dir / "prediction_history.csv"
            history_html = output_dir / "history.html"
            latest_results = output_dir / "latest_results.json"
            pd.DataFrame(
                columns=["date", "venue", "race_no", "race_id", "bet_type", "buy"]
            ).to_csv(latest_bets, index=False)

            with mock.patch.multiple(
                settle_results,
                LATEST_BETS_CSV=latest_bets,
                SETTLED_BETS_CSV=settled_bets,
                PURCHASE_PLAN_CSV=purchase_plan,
                SETTLEMENT_SUMMARY_CSV=summary_csv,
                REPORT_MD=report_md,
                PREDICTION_LEDGER_CSV=prediction_ledger,
                PREDICTION_HISTORY_CSV=prediction_history,
                HISTORY_HTML=history_html,
                LATEST_RESULTS_JSON=latest_results,
            ), mock.patch.object(settle_results, "ensure_dirs"), mock.patch.object(
                settle_results, "fetch_results"
            ) as fetch_results, mock.patch.object(
                fusion_shadow_live, "has_overdue_unsettled", return_value=False
            ):
                settle_results.run_settlement(argparse.Namespace(min_expected_profit=0.0))

            fetch_results.assert_not_called()
            summary = pd.read_csv(summary_csv)
            self.assertEqual(summary["bets"].sum(), 0)
            self.assertEqual(summary["stake_yen"].sum(), 0)
            self.assertTrue(settled_bets.exists())
            self.assertTrue(purchase_plan.exists())
            self.assertTrue(report_md.exists())


    def test_only_official_finish_order_marks_race_decided(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            latest_bets = output_dir / "latest_bets.csv"
            settled_bets = output_dir / "settled_bets.csv"
            purchase_plan = output_dir / "purchase_plan.csv"
            summary_csv = output_dir / "settlement_summary.csv"
            report_md = output_dir / "japanese_report.md"
            prediction_ledger = output_dir / "prediction_ledger.csv"
            prediction_history = output_dir / "prediction_history.csv"
            history_html = output_dir / "history.html"
            latest_results = output_dir / "latest_results.json"
            pd.DataFrame([{
                "date": "2026-09-24", "venue": "test", "race_no": 1,
                "race_id": "010120260924", "bet_type": "exacta", "buy": "1-2",
                "expected_profit_yen": 100, "stake_yen": 100, "return_if_hit_yen": 500,
            }]).to_csv(latest_bets, index=False)
            pending = pd.DataFrame([{
                "race_id": "010120260924", "actual_trifecta": "",
                "actual_exacta": "", "official_result_available": False,
            }])

            with mock.patch.multiple(
                settle_results,
                LATEST_BETS_CSV=latest_bets,
                SETTLED_BETS_CSV=settled_bets,
                PURCHASE_PLAN_CSV=purchase_plan,
                SETTLEMENT_SUMMARY_CSV=summary_csv,
                REPORT_MD=report_md,
                PREDICTION_LEDGER_CSV=prediction_ledger,
                PREDICTION_HISTORY_CSV=prediction_history,
                HISTORY_HTML=history_html,
                LATEST_RESULTS_JSON=latest_results,
            ), mock.patch.object(settle_results, "ensure_dirs"), mock.patch.object(
                settle_results, "fetch_results", return_value=pending
            ):
                settle_results.run_settlement(argparse.Namespace(min_expected_profit=0.0))

            settled = pd.read_csv(settled_bets)
            self.assertFalse(bool(settled.loc[0, "is_decided"]))
            self.assertEqual(settled.loc[0, "display_status"], "結果取得待ち")


    def test_netkeirin_fallback_parses_confirmed_top3_and_trifecta_payout(self):
        race_entries = pd.DataFrame([{
            "race_id": "067320261004",
            "date": "2026-10-04",
            "venue": "小松島",
            "race_no": 6,
            "car_no": 1,
            "bracket_no": 1,
            "start_at": 1791122580,
            "close_at": 1791122280,
            "source_url": "https://www.winticket.jp/keirin/komatsushima/racecard/2026100373/2/6",
        }])
        html = """
        <html><body>
        <div>レースが確定しました</div>
        <table>
          <tr><td>1着</td><td>1</td><td>1</td><td>橋本智昭</td></tr>
          <tr><td>2着</td><td>4</td><td>4</td><td>平石浩之</td></tr>
          <tr><td>3着</td><td>5</td><td>5</td><td>山田義彦</td></tr>
        </table>
        <div>３連単 1&gt;4&gt;5 3,070円 8人気</div>
        </body></html>
        """
        result = settle_results.parse_netkeirin_result_html(
            html,
            race_entries,
            "https://keirin.netkeiba.com/race/result/?race_id=202610047306",
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["official_result_available"])
        self.assertEqual(result["actual_trifecta"], "1-4-5")
        self.assertEqual(result["actual_trifecta_odds"], 30.7)
        self.assertEqual(result["result_source"], "netkeirin_fallback")

    def test_netkeirin_fallback_rejects_unconfirmed_result_page(self):
        race_entries = pd.DataFrame([{
            "race_id": "078420261005",
            "date": "2026-10-05",
            "venue": "武雄",
            "race_no": 7,
            "car_no": 1,
            "bracket_no": 1,
            "source_url": "https://www.winticket.jp/keirin/takeo/racecard/2026100384/3/7",
        }])
        html = "<html><body><div>レース結果決定後に公開されます</div></body></html>"
        result = settle_results.parse_netkeirin_result_html(
            html,
            race_entries,
            "https://keirin.netkeiba.com/race/result/?race_id=202610058407",
        )
        self.assertIsNone(result)

    def test_netkeirin_url_uses_race_date_venue_code_and_race_number(self):
        entry = pd.Series({
            "date": "2026-10-05",
            "race_no": 7,
            "source_url": "https://www.winticket.jp/keirin/takeo/racecard/2026100384/3/7",
        })
        self.assertEqual(
            settle_results.netkeirin_result_url(entry),
            "https://keirin.netkeiba.com/race/result/?race_id=202610058407",
        )


if __name__ == "__main__":
    unittest.main()
