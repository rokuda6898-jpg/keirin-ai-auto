"""Exercise real input adapters and publication boundaries without source HTTP."""
import copy
import itertools
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# The production source module imports requests for its separate PC collector.
# Hermetic tests only use its pure payoff parser and deny every HTTP call.
try:
    import requests
except ModuleNotFoundError:
    requests = types.ModuleType("requests")
    requests.get = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no HTTP in offline tests"))
    sys.modules["requests"] = requests
import nexus_input_snapshots as capture
import nexus_publish_saved_inputs as publisher
from nexus_model.inputs import History, build


NOW = datetime.fromisoformat("2026-10-19T07:00:00+09:00")


class Clock:
    def __init__(self, now=NOW):
        self.value = now
    def __call__(self):
        return self.value


def page(number=1, size=7, close=None):
    """The existing collectors' TanStack race/cup/odds contract, source units intact."""
    close = close or NOW + timedelta(hours=2, minutes=number)
    rid = f"{number:02d}1220261019"
    race = dict(id=rid, number=number, class_="A級", raceType="予選", isGradeRace=False,
                cancel=False, entriesNumber=size, startAt=int((close + timedelta(minutes=5)).timestamp()),
                closeAt=int(close.timestamp()), distance=1625, decidedAt=None)
    race["class"] = race.pop("class_")
    data = dict(race=race, schedule=dict(date="20261019", cupId="2026101912", index=1, day=1),
                entries=[dict(number=car, playerId=f"{car:06d}", raceId=rid, absent=False)
                         for car in range(1, size + 1)],
                players=[dict(id=f"{car:06d}", name=f"選手{car}", birthday="19961020", age=999,
                              gender="男", term=100) for car in range(1, size + 1)],
                records=[dict(playerId=f"{car:06d}", racePoint=90-car, firstRate=12.5,
                              secondRate=20.0, thirdRate=30.0, first=5, second=3, third=2, others=8,
                              standing=1, back=2, frontRunner=3, stalker=4, deepCloser=5,
                              marker=6, gearRatio=3.92, style="両") for car in range(1, size + 1)],
                linePrediction=dict(lines=[dict(entries=[dict(numbers=[1]), dict(numbers=[2, 3])])]
                                    + [dict(entries=[dict(numbers=[c])]) for c in range(4, size + 1)]))
    cup = dict(cup=dict(id="2026101912", venueId=12, grade=3), venue=dict(id=12, name="青森"))
    rows = [("FETCH_KEIRIN_RACE", data), ("FETCH_KEIRIN_CUP_RACES", cup),
            ("FETCH_KEIRIN_RACE_ODDS", {"trifecta": [], "finalOdds": False})]
    state = {"tanStackQuery": {"queries": [{"queryKey": [key], "state": {"data": value}} for key, value in rows]}}
    return state, f"https://www.winticket.jp/keirin/aomori/racecard/2026101912/1/{number}"


def prediction(cars=(1, 2, 3)):
    tickets = ["-".join(map(str, t)) for t in itertools.permutations(cars, 3)]
    return dict(model_id="fixture", policy_id="flexible-first-v1", count=6,
                blocks=[dict(first=list(cars), second=list(cars), third=list(cars), count=6)],
                tickets=tickets, marks={str(c): "◎" if i == 0 else "○" for i, c in enumerate(cars)},
                category="FAVORITE", confidence=dict(grade="A", status="assessed"),
                q=[0.1] * 6)


class FakeEngine:
    def __init__(self, model_path):
        pass
    def predict(self, matrix):
        return prediction(tuple(matrix["rows"][0]["cars"][:3]))


class SavedInputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        capture.save_json(self.root / "models/nexus53/manifest.json", dict(id="fixture", name="fixture", counts={}))
        self.history = History()
        self.history.data["through"] = "2026-10-18"
        self.clock = Clock()

    def snapshot(self, number=1, size=7, close=None):
        state, url = page(number, size, close)
        snap = capture.capture_existing_page(state, url, json.dumps(state), NOW.isoformat(), self.root)
        self.assertIsNotNone(snap)
        return snap, state

    def schedule(self, snaps):
        rows = [dict(race_id=s["meta"]["race_id"], date=s["meta"]["date"], venue=s["meta"]["venue"],
                     race_no=s["meta"]["race_number"], start_at=s["meta"]["start_at"],
                     close_at=s["meta"]["close_at"], source_url=s["meta"]["source_url"])
                for s in snaps]
        capture.capture_schedule(rows, NOW.isoformat(), self.root)

    def run_model(self, engine=FakeEngine):
        with patch("requests.get", side_effect=AssertionError("no network is allowed")):
            return publisher.run(self.root, self.clock, engine, self.history)

    def live(self):
        return capture.read_json(self.root / "outputs/nexus_live.json")

    def test_exact_fields_seven_and_nine_and_contested_slots_reach_real_adapter(self):
        for size in (7, 9):
            snap, _ = self.snapshot(size=size)
            data, meta = snap["data"], snap["meta"]
            self.assertEqual(data["records"][0]["firstRate"], 12.5)
            self.assertEqual(data["records"][0]["first"], 5)
            self.assertEqual(data["players"][0]["birthday"], "19961020")
            self.assertNotIn("age", data["players"][0])
            self.assertEqual(data["linePrediction"]["lines"][0]["entries"][1]["numbers"], [2, 3])
            matrix, lines = build(self.history, data, meta)
            self.assertEqual(matrix["rows"][0]["active_cars"], size)
            self.assertEqual(lines["competition_groups"], [[2, 3]])
            self.assertIn(29.0, matrix["riders"][0, 0])

    def test_invalid_identity_unknown_absence_and_missing_cup_are_rejected(self):
        for mutation in ("identity", "absence", "cup"):
            state, url = page()
            data = state["tanStackQuery"]["queries"][0]["state"]["data"]
            if mutation == "identity":
                data["schedule"]["cupId"] = "2026102012"
            elif mutation == "absence":
                data["entries"][0]["absent"] = None
            else:
                state["tanStackQuery"]["queries"].pop(1)
            with self.assertRaises(ValueError):
                capture.snapshot_from_state(state, url, "response", NOW.isoformat())

    def test_all_scheduled_races_predict_without_odds_or_confidence_filter(self):
        a, _ = self.snapshot(1)
        b, _ = self.snapshot(2, size=9)
        self.schedule([a, b])
        report = self.run_model()
        self.assertEqual(report["predicted"], 2)
        self.assertEqual(report["missing_open"], [])
        for row in self.live()["races"]:
            self.assertEqual(row["prediction"]["count"], 6)
            self.assertEqual(row["prediction"]["allocations"], {})
            self.assertNotIn("q", row["prediction"])

    def test_source_index_and_raw_inputs_prove_coverage_without_market_csv(self):
        a, _ = self.snapshot(1)
        b, _ = self.snapshot(2)
        capture.capture_expected_schedule([a["url"], b["url"]], "2026-10-19", NOW.isoformat(), self.root)
        report = self.run_model()
        self.assertTrue(report["schedule_complete"])
        self.assertEqual(report["scheduled"], 2)
        self.assertEqual(report["predicted"], 2)

    def test_missing_index_race_is_reported_despite_partial_saved_csv(self):
        a, _ = self.snapshot(1)
        _, missing_url = page(2)
        self.schedule([a])
        capture.capture_expected_schedule([a["url"], missing_url], "2026-10-19", NOW.isoformat(), self.root)
        report = self.run_model()
        self.assertFalse(report["schedule_complete"])
        self.assertEqual(report["missing_schedule_urls"], [missing_url])
        self.assertEqual(report["predicted"], 1)

    def test_closed_and_stale_missing_races_are_not_backfilled(self):
        a, _ = self.snapshot(1, close=NOW - timedelta(minutes=1))
        b, _ = self.snapshot(2)
        self.schedule([a, b])
        self.clock.value += timedelta(minutes=31)
        report = self.run_model()
        self.assertEqual(report["predicted"], 0)
        self.assertEqual(report["missing_closed"], [a["meta"]["race_id"]])
        self.assertEqual(report["missing_open"], [b["meta"]["race_id"]])
        self.assertTrue((self.root / capture.SNAPSHOTS / "2026-10-19/outcomes" / (a["meta"]["race_id"] + ".json")).exists())

    def test_crossing_deadline_during_inference_never_publishes_prediction(self):
        a, _ = self.snapshot(1, close=NOW + timedelta(minutes=1))
        self.schedule([a])
        clock = self.clock
        class Slow(FakeEngine):
            def predict(self, matrix):
                clock.value += timedelta(minutes=2)
                return super().predict(matrix)
        report = self.run_model(Slow)
        self.assertEqual(report["predicted"], 0)
        self.assertNotIn("prediction", self.live()["races"][0])

    def test_checkpoint_exists_before_next_race_but_closed_private_pick_is_removed(self):
        a, _ = self.snapshot(1, close=NOW + timedelta(minutes=1))
        b, _ = self.snapshot(2)
        self.schedule([a, b])
        clock, root = self.clock, self.root
        class Queue(FakeEngine):
            calls = 0
            def predict(self, matrix):
                self.calls += 1
                if self.calls == 2:
                    rows = capture.read_json(root / "outputs/nexus_forecasts/2026-10-19.json")
                    self_test.assertTrue(any(r.get("prediction") for r in rows))
                    clock.value += timedelta(minutes=2)
                return super().predict(matrix)
        self_test = self
        report = self.run_model(Queue)
        rows = {r["race_id"]: r for r in self.live()["races"]}
        self.assertNotIn("prediction", rows[a["meta"]["race_id"]])
        self.assertIn("prediction", rows[b["meta"]["race_id"]])
        self.assertEqual(report["not_published_before_deadline"], [a["meta"]["race_id"]])

    def test_changed_inputs_refresh_and_preserve_previous_forecast(self):
        snap, state = self.snapshot()
        self.schedule([snap])
        self.run_model()
        first = copy.deepcopy(self.live()["races"][0]["prediction"])
        state["tanStackQuery"]["queries"][0]["state"]["data"]["records"][0]["racePoint"] += 1
        self.clock.value += timedelta(minutes=2)
        capture.capture_existing_page(state, snap["url"], json.dumps(state), self.clock().isoformat(), self.root)
        self.run_model()
        row = self.live()["races"][0]
        self.assertEqual(row["previous_predictions"], [first])
        self.assertNotEqual(row["prediction"]["input_sha256"], first["input_sha256"])

    def test_latest_forecast_and_complete_result_merge_without_regression(self):
        p = prediction()
        p.pop("q")
        p.update(captured_at=NOW.isoformat(), input_captured_at=NOW.isoformat(), allocations={})
        old = dict(race_id="011220261019", date="2026-10-19", close_at=(NOW+timedelta(hours=2)).isoformat(),
                   prediction=p, result_complete=True, winners=["1-2-3"], payouts={"1-2-3": 800},
                   result_captured_at="2026-10-19T10:00:00+09:00", results_updated_at="2026-10-19T10:01:00+09:00")
        new = copy.deepcopy(old)
        new["prediction"]["captured_at"] = "2026-10-18T23:00:00+00:00"  # 08:00 JST, newer.
        new["result_complete"] = False
        new["payouts"] = {}
        new["results_updated_at"] = "2026-10-19T11:00:00+09:00"
        merged = publisher.merge_saved_row(old, new)
        self.assertEqual(merged["prediction"]["captured_at"], new["prediction"]["captured_at"])
        self.assertTrue(merged["result_complete"])
        self.assertEqual(merged["payouts"], {"1-2-3": 800})
        self.assertEqual(merged["settlement"]["flat"]["returned"], 800)

    def test_top_level_metadata_and_settled_rows_survive_new_publication(self):
        snap, _ = self.snapshot()
        self.schedule([snap])
        settled = dict(race_id="011220261018", date="2026-10-18", result_complete=True,
                       payouts={"1-2-3": 1000}, settlement={"flat": {"returned": 1000}})
        baseline = dict(date="2026-10-19", updated_at=NOW.isoformat(), results_updated_at="preserve-result-time",
                        policy_updated_at="preserve-policy-time", races=[settled])
        capture.save_json(self.root / "outputs/nexus_live.json", baseline)
        self.run_model()
        live = self.live()
        self.assertEqual(live["results_updated_at"], baseline["results_updated_at"])
        self.assertEqual(live["policy_updated_at"], baseline["policy_updated_at"])
        self.assertEqual(live["races"][0], settled)

    def test_lagging_history_is_reported_and_not_a_network_dependency(self):
        snap, _ = self.snapshot()
        self.schedule([snap])
        self.history.data["through"] = "2026-10-17"
        report = self.run_model()
        self.assertEqual(report["predicted"], 1)
        self.assertEqual(report["history"]["lag_days"], 1)
        self.assertTrue(report["history"]["errors"])

    def test_same_day_history_is_never_used_for_inference(self):
        snap, _ = self.snapshot()
        self.schedule([snap])
        self.history.data["through"] = "2026-10-19"
        report = self.run_model()
        self.assertEqual(report["predicted"], 0)
        self.assertIn("history must end before", report["history"]["errors"][0]["reason"])

    def test_copy_rejects_older_capture_from_concurrent_collector(self):
        snap, _ = self.snapshot()
        with tempfile.TemporaryDirectory() as other:
            destination = Path(other)
            path = capture.SNAPSHOTS / "2026-10-19/preclose" / (snap["meta"]["race_id"] + ".json")
            newer = copy.deepcopy(snap)
            newer["captured_at"] = (NOW + timedelta(minutes=5)).isoformat()
            newer["input_sha256"] = "newer-source"
            capture.save_json(destination / path, newer)
            publisher.copy_newer_inputs(self.root, destination)
            self.assertEqual(capture.read_json(destination / path), newer)

    def test_publication_gate_restores_the_previous_public_forecast(self):
        snap, _ = self.snapshot(close=NOW + timedelta(minutes=2))
        self.schedule([snap])
        self.run_model()
        previous = copy.deepcopy(self.live()["races"][0])
        live = self.live()
        live["races"][0]["prediction"]["captured_at"] = (NOW + timedelta(seconds=30)).isoformat()
        live["races"][0]["prediction"]["confidence"]["grade"] = "S"
        capture.save_json(self.root / "outputs/nexus_live.json", live)
        capture.save_json(self.root / "outputs/nexus_forecasts/2026-10-19.json", live["races"])
        self.clock.value = NOW + timedelta(seconds=75)
        removed = publisher.gate_saved_publication(self.root, {previous["race_id"]: previous}, self.clock)
        self.assertEqual(removed, [previous["race_id"]])
        self.assertEqual(self.live()["races"][0]["prediction"], previous["prediction"])

    def test_publication_gate_never_reactivates_a_confirmed_withdrawn_rider(self):
        old = dict(race_id="011220261019", close_at=(NOW + timedelta(seconds=30)).isoformat(),
                   prediction=prediction((1, 2, 3)))
        current = dict(old, prediction=prediction((2, 3, 4)), riders=[{"car": c} for c in (2, 3, 4)])
        rows = {old["race_id"]: current}
        publisher.discard_late_changes(rows, {old["race_id"]: old}, self.clock, 60)
        self.assertNotIn("prediction", current)
        self.assertEqual(current["previous_predictions"], [old["prediction"]])

    def test_confidence_annotation_is_allowed_after_close_without_changing_bets(self):
        old = dict(race_id="011220261019", close_at=(NOW-timedelta(minutes=1)).isoformat(),
                   prediction=prediction((1, 2, 3)))
        current = copy.deepcopy(old)
        current["prediction"]["confidence"] = {"grade": "B", "status": "insufficient"}
        rows = {old["race_id"]: current}
        self.assertEqual(publisher.discard_late_changes(rows, {old["race_id"]: old}, self.clock, 60), [])
        self.assertEqual(current["prediction"]["tickets"], old["prediction"]["tickets"])
        self.assertEqual(current["prediction"]["confidence"]["status"], "insufficient")

    def test_remerge_does_not_score_postclose_allocation_and_preserves_cancellation(self):
        p = prediction()
        p.pop("q")
        p.update(captured_at=NOW.isoformat(), input_captured_at=NOW.isoformat(),
                 allocations={"5000": {"available": True, "stakes": {"1-2-3": 5000}}},
                 allocation_saved_at=(NOW+timedelta(hours=3)).isoformat())
        old = dict(race_id="011220261019", date="2026-10-19", close_at=(NOW+timedelta(hours=2)).isoformat(),
                   prediction=p, result_complete=True, winners=["1-2-3"], payouts={"1-2-3": 800},
                   result_captured_at=(NOW+timedelta(hours=3)).isoformat())
        new = copy.deepcopy(old)
        new["prediction"]["captured_at"] = (NOW+timedelta(minutes=1)).isoformat()
        new["result_complete"] = False
        new["payouts"] = {}
        merged = publisher.merge_saved_row(old, new)
        self.assertEqual(set(merged["settlement"]), {"flat"})
        old.update(cancelled=True, refund_all=True, payouts={}, winners=[])
        merged = publisher.merge_saved_row(old, new)
        self.assertTrue(merged["cancelled"])
        self.assertEqual(merged["settlement"]["flat"]["refund"], 600)


if __name__ == "__main__":
    unittest.main()
