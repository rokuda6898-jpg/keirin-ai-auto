"""Synthetic contract/boundary tests; these fixtures are NOT performance data."""
import copy
from datetime import date, datetime, timedelta
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nexus_value as value
from nexus_model.forecast_confidence import from_joint
from nexus_model.replay_evidence import bind_prediction, from_inference
from nexus_source_replay import model_input_hash

JST = ZoneInfo("Asia/Tokyo")
MANIFEST = dict(id="synthetic-value-contract-fixture",train=["2023-10-09","2025-10-08"])
MANIFEST_RAW = json.dumps(MANIFEST)
MODEL_HASH = hashlib.sha256(MANIFEST_RAW.encode()).hexdigest()
FIT_AT = datetime(2025,11,9,12,tzinfo=JST)
PERIOD = dict(calibration_start="2025-10-09",calibration_end="2025-11-08",
              evaluation_start="2025-11-10",evaluation_end="2026-11-09")


def frame(day="2025-11-10", number=1, price=100):
    ds = date.fromisoformat(day)
    close = datetime.combine(ds,datetime.min.time(),JST)+timedelta(hours=15)
    rid = f"{number:02d}46{ds.strftime('%Y%m%d')}"
    cars = [1,2,3,4]
    triples = list(itertools.permutations(cars,3))
    selected = [t for t in triples if t[0] == 1 and t[1] in (2,3)]
    block = dict(first=[1],second=[2,3],third=[2,3,4],count=4)
    q = [(.80/4 if t in selected else .20/20) for t in triples]
    data = dict(race=dict(id=rid,number=number,closeAt=close.timestamp(),
        startAt=(close+timedelta(minutes=1)).timestamp(),cancel=False,decidedAt=0,distance=1600),
        schedule=dict(date=ds.strftime("%Y%m%d"),cupId="2025111046",index=1,day=1),
        entries=[dict(number=c,playerId=str(100+c),raceId=rid,absent=False) for c in cars],
        players=[dict(id=str(100+c),name=f"選手{c}") for c in cars],
        records=[],linePrediction={},results=[],finalOdds=False,trifectaWinningOddsIds=[],
        trifecta=[dict(key=list(t),raceId=rid,unitPrice=100,absent=False,
                       odds=price if price else 20+i*2,payoffUnitPrice=0) for i,t in enumerate(triples)])
    snap = dict(schema=1,source_kind="existing_public_page_response",
        captured_at=(close-timedelta(seconds=240)).isoformat(),sha256="a"*64,
        meta=dict(race_id=rid,date=day,race_number=number,cancel=False,
                  close_at=close.timestamp(),start_at=(close+timedelta(minutes=1)).timestamp()),data=data)
    snap["input_sha256"] = model_input_hash(snap)
    tickets = ["-".join(map(str,t)) for t in selected]
    prediction = dict(model_id=MANIFEST["id"],policy_id="flexible-first-v1",tickets=tickets,
        blocks=[block],count=4,marks={"1":"◎","2":"○","3":"▲","4":"△"},
        captured_at=(close-timedelta(seconds=200)).isoformat(),input_captured_at=snap["captured_at"],
        source_sha256=snap["sha256"],input_sha256=snap["input_sha256"],
        history_through=(ds-timedelta(days=1)).isoformat(),
        confidence=from_joint(cars,q,tickets,annotated_at=(close-timedelta(seconds=200)).isoformat()))
    prediction["replay_evidence"] = from_inference(cars,q,dict(tickets=selected,blocks=[block]),MANIFEST,MODEL_HASH)
    row = dict(race_id=rid,date=day,race_no=number,venue="synthetic",session="day",
        start_at=(close+timedelta(minutes=1)).isoformat(),close_at=close.isoformat(),
        riders=[dict(car=c,pid=str(100+c),name=f"選手{c}") for c in cars],
        prediction=prediction,closed=False,cancelled=False,result_complete=False)
    bind_prediction(prediction,race_id=rid,race_date=day,close_at=row["close_at"])
    market = copy.deepcopy(snap)
    market.update(captured_at=(close-timedelta(seconds=180)).isoformat(),sha256="b"*64)
    return dict(race=row,input_snapshot=snap,market_snapshot=market,
                decision_at=(close-timedelta(seconds=120)).isoformat(),
                published_at=(close-timedelta(seconds=100)).isoformat())


def outcome(context, payouts=None, **kwargs):
    row = context["race"]
    return dict(race_id=row["race_id"],date=row["date"],complete=True,
                captured_at=(datetime.fromisoformat(row["close_at"])+timedelta(minutes=5)).isoformat(),
                source_sha256="c"*64,payouts={"1-2-3":2000} if payouts is None else payouts,
                refunded=[],refund_all=False,**kwargs)


def artifact():
    # A deliberately synthetic receipt for validator/unit arithmetic tests.
    # Only test_fit_uses_two_coefficients exercises real fitting; none of these
    # dates or labels assert a genuine source publication or achieved return.
    rid = "014620251010"
    return value._seal(dict(schema=1,kind="independent_two_coefficient_calibrator",
        policy_id=value.POLICY_ID,policy_sha256=value.POLICY_SHA256,
        implementation_sha256=value.implementation_sha256(),model_id=MANIFEST["id"],
        model_manifest_sha256=MODEL_HASH,training_cutoff_exclusive="2025-10-09",
        **PERIOD,fit_objective="multinomial_log_loss",coefficients=dict(q=1.0,market=0.0),
        calibration_race_ids=[rid],calibration_all_race_ids=[rid],calibration_races=1,
        calibration_exclusions=[],calibration_dataset_sha256="d"*64,
        calibration_outcomes_latest_at="2025-10-10T15:05:00+09:00",
        calibration_log_loss=1.0,raw_q_log_loss=1.0,
        fit_started_at=FIT_AT.isoformat(),fitted_at=FIT_AT.isoformat(),
        reliability_validated=False,probability_lower_bound=False,
        performance_status="unvalidated_research_hypothesis"),"artifact_sha256")


def assess(context=None, calibration=None):
    context = frame() if context is None else context
    return value.assess(context,artifact() if calibration is None else calibration,MANIFEST,MODEL_HASH,
                        clock=lambda:datetime(2026,10,20,15,tzinfo=JST))


def inventory(contexts):
    first,last = map(date.fromisoformat,(PERIOD["evaluation_start"],PERIOD["evaluation_end"]))
    bydate = {}
    for context in contexts:
        row = context["race"]
        bydate.setdefault(row["date"],[]).append(row["race_id"])
    days = [dict(date=(first+timedelta(days=n)).isoformat(),
                 race_ids=bydate.get((first+timedelta(days=n)).isoformat(),[]),
                 source_sha256="e"*64,complete=True) for n in range((last-first).days+1)]
    return value._seal(dict(schema=1,start=first.isoformat(),end=last.isoformat(),days=days),"inventory_sha256")


class ValueMath(unittest.TestCase):
    def test_transform_does_not_rewrite_q_or_renormalize_selected_tickets(self):
        context = frame()
        q = value.validate_frame(context,MANIFEST,MODEL_HASH)["q"]
        original = copy.deepcopy(q)
        quotes = {key:float(i+2) for i,key in enumerate(q)}
        raw = value.purchase_probabilities(q,quotes,dict(q=1.0,market=0.0))
        market = value.purchase_probabilities(q,quotes,dict(q=0.0,market=1.0))
        self.assertAlmostEqual(sum(raw.values()),1)
        for key in q:
            self.assertAlmostEqual(raw[key],q[key],places=12)
            self.assertAlmostEqual(market[key],(1/quotes[key])/sum(1/o for o in quotes.values()))
        self.assertEqual(q,original)
        self.assertAlmostEqual(sum(raw[t] for t in context["race"]["prediction"]["tickets"]),.8)

    def test_stress_rounds_down_and_ev_uses_total_payout(self):
        quotes = value.stressed_odds({"x":3.1,"y":10.0})
        self.assertEqual(quotes,{"x":2.4,"y":8.0})
        self.assertAlmostEqual(value.expected_return({"x":.3,"y":.1},{"x":100,"y":100},quotes),.76)
        with self.assertRaises(ValueError):
            value.expected_return({"x":.3},{"x":True},{"x":3})

    def test_all_formations_marks_and_confidence_preserved_and_all_budgets_separate(self):
        context = frame()
        original = copy.deepcopy(context)
        report = assess(context)
        self.assertEqual(report["status"],"assessed")
        self.assertEqual(report["flat"]["status"],"buy")
        self.assertAlmostEqual(report["flat"]["expected_return_multiple"],16.0)
        self.assertEqual(len(report["allocated"]),51)
        for name,mode in report["allocated"].items():
            self.assertEqual(set(mode["stakes"]),set(context["race"]["prediction"]["tickets"]))
            self.assertEqual(sum(mode["stakes"].values()),int(name))
            self.assertGreaterEqual(mode["minimum_stress_return_yen"],int(name))
            self.assertAlmostEqual(mode["expected_return_multiple"],16.0)
        for key in ("tickets","blocks","count","marks","confidence"):
            self.assertEqual(report["forecast"][key],context["race"]["prediction"][key])
        self.assertEqual(context,original)
        self.assertFalse(report["is_live_purchase_recommendation"])
        self.assertIsNone(report["annual_100_achieved"])

    def test_portfolio_below_ev_skips_and_infeasible_allocation_has_no_stakes(self):
        report = assess(frame(price=2))
        self.assertEqual(report["flat"]["status"],"skip")
        self.assertAlmostEqual(report["flat"]["expected_return_multiple"],.32)
        self.assertEqual(report["allocated"]["5000"]["reason_code"],"no_break_even")
        self.assertIsNone(report["allocated"]["5000"]["stakes"])


class ValueEvidence(unittest.TestCase):
    def test_missing_calibration_never_falls_back_to_raw_q(self):
        report = value.assess(frame(),None,MANIFEST,MODEL_HASH)
        self.assertEqual(report["reason_code"],"missing_independent_calibrator")
        self.assertIsNone(report["p_bet"])
        self.assertTrue(all(m["status"] == "unavailable" for m in report["allocated"].values()))

    def test_future_declared_decision_cannot_be_used_as_an_observed_record(self):
        report = value.assess(frame(),artifact(),MANIFEST,MODEL_HASH,clock=lambda:FIT_AT)
        self.assertEqual(report["reason_code"],"future_decision_context")
        self.assertIsNone(report["p_bet"])

    def test_quote_age_close_window_and_publication_are_distinct_gates(self):
        base = frame()
        close = datetime.fromisoformat(base["race"]["close_at"])
        boundary = copy.deepcopy(base)
        boundary["market_snapshot"]["captured_at"] = (close-timedelta(seconds=240)).isoformat()
        boundary["published_at"] = (close-timedelta(seconds=60)).isoformat()
        self.assertEqual(assess(boundary)["status"],"assessed")
        for kind in ("old_early","late_publish","future_quote","late_decision"):
            context = copy.deepcopy(base)
            if kind == "old_early":
                # Recent at decision, but fetched too early relative to close.
                context["market_snapshot"]["captured_at"] = (close-timedelta(seconds=301)).isoformat()
                context["input_snapshot"]["captured_at"] = (close-timedelta(seconds=350)).isoformat()
                prediction = context["race"]["prediction"]
                prediction["input_captured_at"] = context["input_snapshot"]["captured_at"]
                evidence = prediction["replay_evidence"]
                evidence["input_captured_at"] = prediction["input_captured_at"]
                value._seal(evidence,"evidence_sha256")
            elif kind == "late_publish":
                context["published_at"] = (close-timedelta(seconds=59)).isoformat()
            elif kind == "future_quote":
                context["market_snapshot"]["captured_at"] = (close-timedelta(seconds=80)).isoformat()
            else:
                context["decision_at"] = (close-timedelta(seconds=59)).isoformat()
            with self.subTest(kind=kind):
                self.assertEqual(assess(context)["status"],"unavailable")

    def test_outcome_markers_rejected_before_probability_or_allocation(self):
        for kind in ("winning_ids","positive_payoff","final","results","cancelled","top_result"):
            context = frame()
            data = context["market_snapshot"]["data"]
            if kind == "winning_ids": data["trifectaWinningOddsIds"] = ["some-id"]
            if kind == "positive_payoff": data["trifecta"][0]["payoffUnitPrice"] = 2270
            if kind == "final": data["finalOdds"] = True
            if kind == "results": data["results"] = [dict(order=1)]
            if kind == "cancelled": context["race"]["cancelled"] = True
            if kind == "top_result": context["outcome"] = outcome(context)
            with self.subTest(kind=kind), patch.object(value,"purchase_probabilities",side_effect=AssertionError("must not score")):
                self.assertEqual(assess(context)["status"],"unavailable")

    def test_q_hash_card_identity_and_censored_or_partial_books_fail_closed(self):
        for kind in ("q_hash","car","model","input","partial","ceiling","duplicate"):
            context = frame()
            if kind == "q_hash": context["race"]["prediction"]["replay_evidence"]["q"][0] += .01
            if kind == "car": context["race"]["riders"][0]["pid"] = "999"
            if kind == "model": context["race"]["prediction"]["model_id"] = "other"
            if kind == "input": context["input_snapshot"]["data"]["race"]["distance"] = 2000
            if kind == "partial": context["market_snapshot"]["data"]["trifecta"].pop()
            if kind == "ceiling": context["market_snapshot"]["data"]["trifecta"][0]["odds"] = 9999.9
            if kind == "duplicate": context["market_snapshot"]["data"]["trifecta"].append(copy.deepcopy(context["market_snapshot"]["data"]["trifecta"][0]))
            with self.subTest(kind=kind):
                self.assertEqual(assess(context)["status"],"unavailable")

    def test_calibrator_cannot_include_model_training_or_evaluation_year(self):
        for field,bad in (("calibration_start","2025-10-08"),("calibration_end","2025-11-10"),
                          ("fitted_at","2025-11-10T00:00:00+09:00"),("fit_objective","return_on_investment")):
            calibration = artifact()
            calibration[field] = bad
            value._seal(calibration,"artifact_sha256")
            with self.subTest(field=field):
                self.assertEqual(assess(calibration=calibration)["status"],"unavailable")
        calibration = artifact()
        calibration["coefficients"]["q"] = 2.0
        self.assertEqual(assess(calibration=calibration)["status"],"unavailable")

    def test_fit_uses_two_coefficients_and_never_optimizes_roi(self):
        records = []
        for n in range(3):
            context = frame(f"2025-10-{10+n:02d}",price=0)
            records.append(dict(frame=context,outcome=outcome(context)))
        before = copy.deepcopy(records)
        calibration = value.fit_calibrator(records,MANIFEST,MODEL_HASH,**PERIOD,clock=lambda:FIT_AT)
        self.assertEqual(set(calibration["coefficients"]),{"q","market"})
        self.assertEqual(calibration["fit_objective"],"multinomial_log_loss")
        self.assertLessEqual(calibration["calibration_log_loss"],calibration["raw_q_log_loss"]+1e-9)
        self.assertFalse(calibration["reliability_validated"])
        self.assertEqual(records,before)
        self.assertIs(value.validate_calibrator(calibration,MANIFEST,MODEL_HASH),calibration)
        with self.assertRaises(value.EvidenceError):
            value.fit_calibrator(records,MANIFEST,MODEL_HASH,**PERIOD,
                clock=lambda:datetime(2025,11,10,tzinfo=JST))
        clocks = iter([FIT_AT,datetime(2025,11,10,tzinfo=JST)])
        with self.assertRaises(value.EvidenceError):
            value.fit_calibrator(records,MANIFEST,MODEL_HASH,**PERIOD,clock=lambda:next(clocks))


class ValueAnnual(unittest.TestCase):
    def test_exact_year_anniversary_and_leap_days_not_calendar_month_labels(self):
        first,last = value.annual_period("2025-10-09","2026-10-08")
        self.assertEqual((last-first).days+1,365)
        first,last = value.annual_period("2023-10-09","2024-10-08")
        self.assertEqual((last-first).days+1,366)
        with self.assertRaises(ValueError): value.annual_period("2025-10-09","2026-10-09")

    def test_partial_inventory_or_missing_decision_never_claims_annual_return(self):
        context = frame()
        complete = inventory([context])
        partial = copy.deepcopy(complete)
        partial["days"].pop()
        value._seal(partial,"inventory_sha256")
        for schedule,assessments in ((partial,[assess(context)]),(complete,[])):
            result = value.annual_audit(schedule,assessments,[outcome(context)],artifact(),MANIFEST,MODEL_HASH)
            self.assertEqual(result["status"],"unverified")
            self.assertIsNone(result["annual_roi"])
            self.assertIsNone(result["annual_100_achieved"])

    def test_full_q_missing_is_unevaluated_not_a_profitable_skip(self):
        contexts = [frame(),frame("2025-11-11")]
        assessments = [assess(c) for c in contexts]
        bad = copy.deepcopy(contexts[1])
        bad["race"]["prediction"].pop("replay_evidence")
        assessments[1] = assess(bad)
        report = value.annual_audit(inventory(contexts),assessments,[outcome(c) for c in contexts],artifact(),MANIFEST,MODEL_HASH)
        self.assertEqual(report["status"],"incomplete")
        self.assertIsNone(report["summary"]["flat"]["annual_roi"])
        self.assertEqual(report["summary"]["flat"]["unavailable_race_ids"],[contexts[1]["race"]["race_id"]])

    def test_dead_heats_partial_refunds_and_voids_use_total_yen(self):
        contexts = [frame("2025-11-10"),frame("2025-12-10"),frame("2026-01-10")]
        outcomes = [outcome(contexts[0],{"1-2-3":2000,"1-2-4":3000}),
                    outcome(contexts[1],{"1-2-3":1000}),outcome(contexts[2],{})]
        outcomes[1]["refunded"] = ["1-2-4"]
        outcomes[2]["refund_all"] = True
        report = value.annual_audit(inventory(contexts),[assess(c) for c in contexts],outcomes,artifact(),MANIFEST,MODEL_HASH)
        self.assertEqual(report["status"],"complete_hypothetical")
        flat = report["summary"]["flat"]
        self.assertEqual(flat["observed"]["races"],2)
        self.assertEqual(flat["observed"]["hits"],2)
        self.assertEqual(flat["observed"]["stake_yen"],700)
        self.assertEqual(flat["observed"]["payout_yen"],6000)
        self.assertEqual(flat["observed"]["refund_yen"],500)
        self.assertAlmostEqual(flat["annual_roi"],6000/700)
        self.assertEqual(len(flat["monthly"]),12)
        self.assertAlmostEqual(sum(x["payout_yen"] for x in flat["monthly"])/sum(x["stake_yen"] for x in flat["monthly"]),flat["annual_roi"])
        self.assertIsNone(report["annual_100_achieved"])
        self.assertTrue(flat["historical_roi_exceeds_100"])
        self.assertEqual(len(report["race_records"]),3)

    def test_independent_mode_purchase_coverage_and_unchanged_all_race_forecast_hits(self):
        contexts = [frame(),frame("2025-11-11",price=2)]
        # Uniform 0.2 probabilities on the four original tickets: flat EV is
        # 1.32, while maximin allocation buys more of the low-return ticket and
        # has EV near 1.01. The same race legitimately passes only flat.
        tickets = contexts[0]["race"]["prediction"]["tickets"]
        for quote in contexts[0]["market_snapshot"]["data"]["trifecta"]:
            key = "-".join(map(str,quote["key"]))
            if key in tickets:
                quote["odds"] = 3 if key == tickets[0] else 10
        reports = [assess(c) for c in contexts]
        report = value.annual_audit(inventory(contexts),reports,[outcome(c) for c in contexts],artifact(),MANIFEST,MODEL_HASH)
        self.assertEqual(report["all_race_forecast_metrics"]["hits"],2)
        self.assertEqual(report["summary"]["flat"]["observed"]["hits"],1)
        self.assertEqual(report["summary"]["flat"]["purchase_coverage"],.5)
        self.assertEqual(report["summary"]["allocated"]["5000"]["purchase_coverage"],0)
        self.assertEqual(report["summary"]["allocated"]["5000"]["purchase_race_ids"],[])
        self.assertEqual(report["summary"]["allocated"]["5000"]["skip_race_ids"],[c["race"]["race_id"] for c in contexts])

    def test_changed_portfolio_or_calibrator_cannot_pass_annual_audit(self):
        context = frame()
        for kind in ("ticket","fit"):
            report = assess(context)
            if kind == "ticket":
                ticket = next(iter(report["allocated"]["5000"]["stakes"]))
                del report["allocated"]["5000"]["stakes"][ticket]
            else:
                report["calibrator_sha256"] = "f"*64
            value._seal(report,"receipt_sha256")
            result = value.annual_audit(inventory([context]),[report],[outcome(context)],artifact(),MANIFEST,MODEL_HASH)
            self.assertEqual(result["status"],"unverified")
            self.assertIsNone(result["annual_100_achieved"])


class ValueCLI(unittest.TestCase):
    def test_cli_missing_calibration_writes_unavailable_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source,output,model = root/"input.json",root/"report.json",root/"manifest.json"
            source.write_text(json.dumps(dict(schema=1,policy_id=value.POLICY_ID,frame=frame(),calibrator=None)))
            model.write_text(MANIFEST_RAW)
            args = ["assess","--input",str(source),"--output",str(output),"--manifest",str(model)]
            with patch("builtins.print"):
                value.main(args)
            original = output.read_bytes()
            self.assertEqual(json.loads(original)["status"],"unavailable")
            with self.assertRaises(value.EvidenceError),patch("builtins.print"):
                value.main(args)
            self.assertEqual(output.read_bytes(),original)


if __name__ == "__main__":
    unittest.main()
