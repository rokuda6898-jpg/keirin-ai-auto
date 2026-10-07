import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from betting_logic import (score_riders, race_plan, generate_formations, select_race,
                           main_limit, hole_limit, divergence_points, clean_odds)
import predict


def fixture(strong=False):
    return pd.DataFrame({
        "car_no": range(1, 8), "race_id": ["fixture"] * 7,
        "fixed_axis_calibration_passed": [True] * 7,
        "date": ["2026-10-06"] * 7, "venue": ["テスト"] * 7, "race_no": [1] * 7,
        "p_win": [.65, .10, .08, .06, .05, .04, .02] if strong else [.18, .17, .16, .15, .13, .11, .10],
        "place2_rate": [.12, .14, .16, .18, .20, .22, .55],
        "place3_rate": [.15, .17, .19, .21, .23, .50, .25],
        "line_id": [1, 1, 2, 2, 3, 3, 4],
        "popularity_rank": [1, 6, 2, 3, 4, 5, 7],
    })


def fair_odds(riders, ev=1.4):
    candidates = generate_formations(riders, race_plan(riders))
    return candidates[["buy", "bet_type"]].assign(odds_used=ev / candidates.prob)


class BettingLogicTests(unittest.TestCase):
    def test_fixed_axis_challenger_keeps_lower_pool_and_allows_other_heads(self):
        riders=score_riders(fixture(True))
        baseline,_=select_race(riders,fair_odds(riders))
        spread,plan=select_race(riders,fair_odds(riders),force_no_fixed=True)
        self.assertEqual(baseline['head'].nunique(),1)
        self.assertGreater(spread['head'].nunique(),1)
        self.assertFalse(plan['first_fixed'])
        self.assertEqual(len(plan['formation']['second']),len(riders))
        self.assertEqual(len(plan['formation']['third']),len(riders))
    def test_quote_metadata_survives_into_candidate_portfolio(self):
        race=fixture().assign(close_at=1000)
        odds=fair_odds(score_riders(race)).assign(race_id='fixture',odds_sources='winticket',
                                               odds_captured_at_jst='1970-01-01T00:08:20+00:00',
                                               odds_verification_status='verified')
        _,candidates,_=predict.build_strategy_outputs(race,odds,600,3600)
        chosen=candidates[candidates.is_selected]
        self.assertFalse(chosen.empty)
        self.assertTrue(chosen.odds_sources.eq('winticket').all())
        self.assertTrue(chosen.odds_captured_at_jst.eq('1970-01-01T00:08:20+00:00').all())
    def test_validated_position_models_take_priority_over_fallback_rates(self):
        race = fixture().assign(position_model_source="position_specialists",
                                p_second=[.05, .1, .15, .2, .25, .2, .05],
                                p_third=[.04, .06, .1, .15, .2, .25, .2])
        riders = score_riders(race)
        self.assertEqual(riders.position_score_source.iloc[0], "validated_position_specialists")
        self.assertAlmostEqual(riders.score_second.sum(), 100)
        self.assertEqual(riders.loc[riders.car_no.eq(5), "rank_second"].iloc[0], 1)
        self.assertEqual(riders.loc[riders.car_no.eq(6), "rank_third"].iloc[0], 1)

    def test_card_summary_shows_groups_fixed_axis_and_skip(self):
        riders = score_riders(fixture(True))
        _, plan = select_race(riders, None)
        card = predict.render_strategy_summary(plan, False)
        for label in ["本線", "穴", "1着固定", "荒れ指数", "購入停止中", "期待値条件"]:
            self.assertIn(label, card)

    def test_top10_history_is_subset_of_new_selected_portfolio(self):
        race = fixture().assign(close_at=1000)
        odds = fair_odds(score_riders(race)).assign(race_id="fixture")
        pred, candidates, _ = predict.build_strategy_outputs(race, odds, 600, 3600)
        selected = set(candidates.loc[candidates.is_selected, "buy"])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(predict, "OUTPUT_DIR", Path(directory)), \
             patch.object(predict, "score_trifecta_reranker", return_value=None):
            ledger = predict.save_trifecta_top10_ledger(
                pred, predict.datetime.now(predict.ZoneInfo("Asia/Tokyo")), candidates)
        current = ledger[ledger.variant.eq("production_top10")]
        self.assertGreater(len(current), 0)
        self.assertLessEqual(len(current), 10)
        self.assertTrue(set(current.buy).issubset(selected))
        self.assertTrue(current.ev.ge(1.10).all())
        self.assertTrue(current.strategy_version.eq(predict.STRATEGY_VERSION).all())

    def test_main_boundaries_and_third_expansion(self):
        for margin, expected in [(10, 6), (9, 8), (7, 8), (6, 10), (4, 10), (3, 12), (0, 12)]:
            self.assertEqual(main_limit(margin, 3), expected)
            self.assertEqual(main_limit(margin, 2), min(expected + 2, 12))

    def test_position_specialists_survive_low_win_rank(self):
        riders = score_riders(fixture())
        self.assertEqual(int(riders.loc[riders.car_no.eq(7), "rank_second"].iloc[0]), 1)
        self.assertEqual(int(riders.loc[riders.car_no.eq(6), "rank_third"].iloc[0]), 1)
        plan = race_plan(riders)
        tickets = generate_formations(riders, plan)
        self.assertIn(7, plan["formation"]["second"])
        self.assertIn(6, plan["formation"]["third"])
        self.assertTrue(tickets.loc[tickets.buy.eq("1-7-6"), "main_formation"].iloc[0])

    def test_full_probability_mass_and_fixed_axis_no_reinflation(self):
        riders = score_riders(fixture())
        self.assertAlmostEqual(generate_formations(riders, race_plan(riders)).prob.sum(), 1)
        riders = score_riders(fixture(True))
        tickets = generate_formations(riders, race_plan(riders))
        self.assertEqual(set(tickets["head"]), {1})
        self.assertAlmostEqual(tickets.prob.sum(), .65)
        selected, plan = select_race(riders, fair_odds(riders))
        self.assertTrue(plan["first_fixed"])
        self.assertEqual(set(selected.loc[selected.is_selected, "head"]), {1})

    def test_axis_requires_absolute_probability_in_addition_to_gap(self):
        race = pd.DataFrame({"car_no": [1, 2, 3, 4, 5], "p_win": [.4, .3, .15, .1, .05]})
        race["fixed_axis_calibration_passed"] = True
        plan = race_plan(score_riders(race))
        self.assertEqual(plan["first_gap"], 10)
        self.assertFalse(plan["first_fixed"])
        for top, expected in [(.59, False), (.6, True)]:
            race["p_win"] = [top, (1-top)/2, (1-top)/4, (1-top)/8, (1-top)/8]
            self.assertEqual(race_plan(score_riders(race))["first_fixed"], expected)

    def test_ev_compression_disjoint_and_chaos_gate(self):
        riders = score_riders(fixture())
        # Close exact-place scores create an intentionally chaotic test race.
        riders["score_second"] = 50.
        riders["score_third"] = 50.
        selected, plan = select_race(riders, fair_odds(riders, ev=1.6))
        bets = selected[selected.is_selected]
        self.assertFalse(bets.buy.duplicated().any())
        self.assertLessEqual(plan["main_count"], plan["main_limit"])
        self.assertEqual(plan["hole_limit"], 12)
        self.assertGreaterEqual(plan["chaos_index"], 60)
        self.assertEqual(plan["hole_count"], 12)
        self.assertTrue(bets.loc[bets.ticket_group.eq("本線"), "ev"].ge(1.10).all())
        self.assertTrue(bets.loc[bets.ticket_group.eq("穴"), "ev"].ge(1.25).all())
        self.assertEqual(sum(plan["chaos_components"].values()), plan["chaos_index"])
        self.assertLessEqual(plan["chaos_index"], 100)

    def test_do_not_fill_and_skip_missing_or_bad_odds(self):
        riders = score_riders(fixture(True))
        odds = fair_odds(riders, 1.09)
        chosen, plan = select_race(riders, odds)
        self.assertEqual(plan["main_count"] + plan["hole_count"], 0)
        odds = fair_odds(riders).sort_values('odds_used').head(2)
        chosen, plan = select_race(riders, odds)
        self.assertEqual(int(chosen.is_selected.sum()), 2)
        for value in [np.nan, np.inf, -1, 0]:
            chosen, plan = select_race(riders, odds.assign(odds_used=value))
            self.assertFalse(chosen.is_selected.any())

    def test_exact_ev_boundaries_and_probability_first_main_sort(self):
        riders = score_riders(fixture(True))
        generated = generate_formations(riders, race_plan(riders))
        probabilities = generated.set_index("buy").prob
        top4 = generated.sort_values("prob", ascending=False).head(4)[["buy", "bet_type"]].copy()
        top4["odds_used"] = [1.10 / probabilities[buy] for buy in top4.buy]
        self.assertTrue(top4.odds_used.lt(100).all())
        chosen, _ = select_race(riders, top4)
        main = chosen[chosen.ticket_group.eq("本線")]
        self.assertEqual(len(main), 4)
        self.assertTrue(main.ev.ge(1.10).all())
        expected = top4.assign(prob=[probabilities[buy] for buy in top4.buy]).sort_values(
            ["prob", "buy"], ascending=[False, True]
        ).buy.tolist()
        actual = main.sort_values(["prob", "ev", "buy"], ascending=[False, False, True]).buy.tolist()
        self.assertEqual(actual, expected)

    def test_main_is_under_100x_and_hole_is_100x_or_more(self):
        riders = score_riders(fixture())
        candidates = generate_formations(riders, race_plan(riders))
        odds = candidates[["buy", "bet_type", "prob"]].copy()
        odds["odds_used"] = np.where(odds["prob"].rank(ascending=False, method="first").le(20), 50.0, 150.0)
        odds = odds.drop(columns="prob")
        chosen, _ = select_race(riders, odds)
        main = chosen[chosen.ticket_group.eq("本線")]
        holes = chosen[chosen.ticket_group.eq("穴")]
        self.assertFalse(main.empty)
        self.assertTrue(main.odds_used.lt(100).all())
        if not holes.empty:
            self.assertTrue(holes.odds_used.ge(100).all())
        main_sorted = main.sort_values(["prob", "ev", "buy"], ascending=[False, False, True])
        self.assertEqual(main.buy.tolist(), main_sorted.buy.tolist())

    def test_popularity_direction_and_incomplete_proxy(self):
        self.assertEqual([divergence_points(x) for x in range(7)], [0, 0, 3, 5, 7, 10, 10])
        riders = score_riders(fixture())
        self.assertEqual(riders.loc[1, "undervalued_points"], 7)
        self.assertEqual(riders.loc[2, "undervalued_points"], 0)
        race = fixture().drop(columns="popularity_rank")
        self.assertEqual(score_riders(race, fair_odds(riders).head(10)).popularity_source.iloc[0], "unavailable")
        self.assertEqual(score_riders(race, fair_odds(riders)).popularity_source.iloc[0], "full_trifecta_market_proxy")

    def test_hole_switches_and_12_requires_enough_ev_tickets(self):
        riders = score_riders(fixture())
        riders["score_second"] = [100, 80, 60, 40, 30, 20, 10]
        riders["score_third"] = [100, 80, 60, 40, 30, 20, 10]
        one = pd.DataFrame({"head": [4] * 20})
        two = pd.DataFrame({"head": [4, 5] * 10})
        plan = {"chaos_index": 59, "first_fixed": False}
        self.assertEqual(hole_limit(one.head(0), plan, riders), 0)
        self.assertEqual(hole_limit(one, plan, riders), 6)
        riders["score_second"] = 50.
        self.assertEqual(hole_limit(one, plan, riders), 8)
        self.assertEqual(hole_limit(two, plan, riders), 10)
        plan["chaos_index"] = 60
        self.assertEqual(hole_limit(two.head(11), plan, riders), 10)
        self.assertEqual(hole_limit(two.head(12), plan, riders), 12)

    def test_duplicate_odds_conservative_and_invalid_input(self):
        odds = pd.DataFrame({"buy": ["1-2-3"] * 2, "odds_used": [100, 50]})
        self.assertEqual(clean_odds(odds).odds_used.iloc[0], 50)
        with self.assertRaises(ValueError):
            score_riders(pd.concat([fixture(), fixture().head(1)]))
        with self.assertRaises(ValueError):
            select_race(score_riders(fixture()), odds, main_ev=float("nan"))

    def test_timing_and_other_ticket_compatibility(self):
        race = fixture().assign(close_at=1000)
        odds = fair_odds(score_riders(race)).assign(race_id="fixture")
        for epoch in [700, 1000, -3000]:
            _, candidates, plans = predict.build_strategy_outputs(race, odds, epoch, 3600)
            self.assertFalse(candidates.is_selected.any())
            self.assertFalse(plans[0]["timing_eligible"])
        _, candidates, plans = predict.build_strategy_outputs(race, odds, 600, 3600)
        self.assertTrue(candidates.is_selected.any())
        self.assertTrue(plans[0]["timing_eligible"])


    def test_future_race_has_display_only_picks_without_entering_validation(self):
        race = fixture().assign(close_at=5000)
        odds = fair_odds(score_riders(race)).assign(race_id="fixture")
        _, candidates, plans = predict.build_strategy_outputs(race, odds, 0, 3600)
        self.assertFalse(candidates.is_selected.any())
        self.assertFalse(plans[0]["timing_eligible"])

        display = predict.provisional_display_bets(candidates, "fixture", 5000)
        self.assertFalse(display.empty)
        self.assertTrue(display.ticket_group.isin(["本線", "穴"]).all())
        self.assertEqual(int(display.stake_yen.sum()), 10000)
        self.assertTrue(display.display_only.all())

    def test_provisional_display_can_use_stale_raw_quote_without_authorizing_purchase(self):
        race = fixture().assign(close_at=5000)
        odds = fair_odds(score_riders(race)).assign(race_id="fixture")
        odds["odds_captured_at_jst"] = "2026-01-01T00:00:00+09:00"
        odds["odds_verification_status"] = "verified"
        _, candidates, plans = predict.build_strategy_outputs(race, odds, 2_000_000_000, 3600)
        self.assertFalse(candidates.is_selected.any())

        display = predict.provisional_display_bets(candidates, "fixture", 5000, odds)
        self.assertFalse(display.empty)
        self.assertTrue(display.ticket_group.isin(["本線", "穴"]).all())
        self.assertTrue(display.display_only.all())
        self.assertTrue(display.display_quote_fallback.any())

    def test_purchase_gate_binds_strategy_model_and_120_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.write_bytes(b"fixture")
            gate = root / "gate.json"
            result = {"target_passed": True, "strategy_version": predict.STRATEGY_VERSION,
                      "model_sha256": predict.file_sha256(model), "roi": .20}
            with patch.multiple(predict, PROFIT_GATE_PATH=gate, MODEL_PATH=model):
                for changes, expected in [({}, True), ({"roi": .19}, False),
                                          ({"strategy_version": "old"}, False),
                                          ({"model_sha256": "different"}, False)]:
                    gate.write_text(json.dumps({**result, **changes}), encoding="utf-8")
                    self.assertEqual(predict.load_profit_gate()["target_passed"], expected)

    def test_small_fields_have_json_safe_plans(self):
        for size in [3, 4]:
            riders = score_riders(fixture().head(size))
            _, plan = select_race(riders, fair_odds(riders))
            json.dumps(plan, allow_nan=False)


    def test_risk_department_vetoes_fixed_axis_and_widens_flow(self):
        riders = score_riders(fixture(True))
        riders["score_second"] = 50.0
        riders["score_third"] = 50.0
        plan = race_plan(riders)
        self.assertGreaterEqual(plan["risk_score"], 60)
        self.assertTrue(plan["risk_veto_fixed"])
        self.assertFalse(plan["first_fixed"])
        self.assertEqual(plan["main_limit"], main_limit(plan["first_gap"], plan["third_boundary_gap"]))
        generate_formations(riders, plan)
        self.assertGreater(len(plan["formation"]["first"]), 1)
        self.assertGreaterEqual(len(plan["formation"]["second"]), 5)
        self.assertGreaterEqual(len(plan["formation"]["third"]), 6)

    def test_high_risk_main_spreads_second_and_third_places(self):
        riders = score_riders(fixture(True))
        riders["score_second"] = 50.0
        riders["score_third"] = 50.0
        generated = generate_formations(riders, race_plan(riders))
        odds = generated[["buy", "bet_type", "prob"]].copy()
        odds["odds_used"] = 1.4 / odds["prob"]
        odds = odds.drop(columns="prob")
        chosen, plan = select_race(riders, odds)
        main = chosen[chosen.ticket_group.eq("本線")]
        self.assertGreaterEqual(plan["risk_score"], 60)
        self.assertLessEqual(len(main), plan["main_limit"])
        self.assertGreaterEqual(main["second"].nunique(), 3)
        self.assertGreaterEqual(main["third"].nunique(), 3)
        self.assertIn("本命展開", set(main["scenario"]) | set(plan["selected_scenarios"]))


if __name__ == "__main__":
    unittest.main()
