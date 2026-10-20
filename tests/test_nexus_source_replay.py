"""Local Git provenance, exact replay, and unchanged prospective deadlines."""
import copy
from datetime import timedelta
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import test_nexus_forward as fixtures
import nexus_forward as forward
import nexus_source_replay as replay
from nexus_input_snapshots import SNAPSHOTS, canonical, read_json, save_json

Clock, NOW = fixtures.Clock, fixtures.NOW


class SourceReplayTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ForwardTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.manifest = fixture.root, fixture.manifest
        self.clock = Clock()
        self.row, self.cars, self.q = fixture.row, fixture.cars, fixture.q
        self.row["lines"] = []
        p = self.row["prediction"]
        p.pop("replay_evidence")
        p.update(captured_at=(NOW - timedelta(hours=2)).isoformat(),
                 input_captured_at=(NOW - timedelta(hours=2, minutes=2)).isoformat(),
                 source_kind="existing_public_page_response", plan_changed=False, search="fixture",
                 band_probabilities={"MAIN": .5, "MEDIUM": .3, "HIGH": .2})
        p["confidence"]["annotated_at"] = p["captured_at"]
        p["confidence"]["source_captured_at"] = p["captured_at"]
        close = forward._stamp(self.row["close_at"]).timestamp()
        start = forward._stamp(self.row["start_at"]).timestamp()
        rid = self.row["race_id"]
        self.snapshot = dict(schema=1, source_kind="existing_public_page_response",
            captured_at=p["input_captured_at"], sha256=p["source_sha256"],
            meta=dict(race_id=rid, date=self.row["date"], close_at=close, start_at=start, cancel=False),
            data=dict(race=dict(id=rid, number=1, closeAt=close, startAt=start, distance=2000, cancel=False),
                schedule=dict(date="20261020", cupId="fixture", index=1, day=1),
                entries=[dict(number=c, playerId=str(100+c), raceId=rid, absent=False) for c in self.cars],
                players=[dict(id=str(100+c), name="選手"+str(c)) for c in self.cars],
                records=[dict(playerId=str(100+c), racePoint=100+c) for c in self.cars],
                linePrediction={}, results=[], finalOdds=False,
                trifecta=[dict(key=list(t), odds=100, raceId=rid, unitPrice=100, absent=False)
                          for t in itertools.permutations(self.cars, 3)]))
        p["input_sha256"] = self.snapshot["input_sha256"] = replay.model_input_hash(self.snapshot)
        self.snapshot_path = self.root / SNAPSHOTS / self.row["date"] / "preclose" / (rid+".json")
        save_json(self.snapshot_path, self.snapshot)
        self.history_path = self.root / replay.HISTORY
        self.history_path.parent.mkdir(parents=True)
        self.history = dict(through="2026-10-19", form={}, tactics={}, tactic_roles={}, pairs={}, pooled={}, roles={})
        self.save_history(self.history)
        self.history_raw = self.history_path.read_bytes()
        original = {k: copy.deepcopy(p[k]) for k in replay.FORECAST_FIELDS}
        original["q"] = self.q
        original["confidence"].pop("source_captured_at")
        # This tiny immutable package isolates provenance/deadline behavior from
        # numerical model accuracy; a real-model replay is verified separately.
        sources = {
            "__init__.py": "",
            "payload.py": "PREDICTION = " + repr(original) + "\n",
            "engine.py": "import copy\nfrom .payload import PREDICTION\nclass Engine:\n def __init__(self, root): pass\n def predict(self, matrix): return copy.deepcopy(PREDICTION)\n",
            "inputs.py": "import gzip,json\nclass History:\n def __init__(self,data): self.data=data\n @classmethod\n def load(cls,path):\n  with gzip.open(path,'rt') as f: return cls(json.load(f))\ndef build(history,data,meta):\n assert not any(k in data for k in ('results','trifecta','finalOdds','trifectaWinningOddsIds','trifectaPayoffStatus'))\n return {'rows':[{'cars':[e['number'] for e in data['entries'] if not e['absent']]}]}, {'known_unchanged_groups':[]}\n"
        }
        package = self.root / "src/nexus_model"
        package.mkdir(parents=True)
        for name, text in sources.items():
            (package / name).write_text(text)
        publisher = self.root / replay.PUBLISHER
        publisher.write_text("# fixture audited original publisher\n")
        profile = patch.object(replay, "AUDITED_PUBLISHERS", {replay.blob_hash(publisher.read_bytes())})
        profile.start()
        self.addCleanup(profile.stop)
        self.save_live()
        save_json(self.root / "outputs/nexus_forecasts/2026-10-20.json", [self.row])
        self.git("init", "-q")
        self.git("config", "user.name", "Replay Fixture")
        self.git("config", "user.email", "fixture@example.test")
        self.commit()

    def git(self, *args):
        published = (NOW - timedelta(hours=2) + timedelta(seconds=10)).isoformat()
        env = dict(os.environ, GIT_AUTHOR_DATE=published, GIT_COMMITTER_DATE=published)
        return subprocess.run(["git", *args], cwd=self.root, env=env, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()

    def commit(self, amend=False):
        self.git("add", ".")
        self.git("commit", "-q", *( ["--amend"] if amend else [] ), "-m", replay.PUBLICATION_MESSAGE)
        self.origin = self.git("rev-parse", "HEAD")

    def save_history(self, value):
        with gzip.open(self.history_path, "wt") as stream:
            json.dump(value, stream)

    def save_live(self):
        save_json(self.root / "outputs/nexus_live.json", dict(schema=1, date=self.row["date"], races=[self.row]))

    def run_forward(self, **kwargs):
        return forward.run(self.root, clock=self.clock, replay_published=True, **kwargs)

    def clear_feed(self):
        (self.root / forward.FEED).unlink(missing_ok=True)

    def test_original_git_history_and_forecast_replay_without_freshening(self):
        before = canonical(self.row)
        # A newer local history is not an acceptable substitute for the actual
        # original, even when original forecast probabilities would match.
        self.save_history(dict(self.history, through="2026-10-20", form={"new": [[999999, []]]}))
        report = self.run_forward()
        row = report["races"][0]
        record = row["source_replay"]
        self.assertEqual(record["original_publication_commit"], self.origin)
        self.assertEqual(record["history_sha256"], hashlib.sha256(self.history_raw).hexdigest())
        self.assertEqual(record["input_captured_at"], self.row["prediction"]["input_captured_at"])
        self.assertEqual(record["original_prediction_at"], self.row["prediction"]["captured_at"])
        self.assertEqual(record["replayed_at"], NOW.isoformat())
        self.assertEqual(record["max_preserved_numeric_error"], 0)
        self.assertIs(record["original_full_q_observed"], False)
        self.assertEqual(row["saved_at"], NOW.isoformat())
        self.assertEqual(row["candidate"]["count"], 10)
        self.assertEqual(row["candidate"]["allocations"], {})
        self.assertIn("30分", row["allocation_reason"])
        self.assertEqual(report["summary"]["allocated"]["races"], 0)
        self.assertEqual(before, canonical(read_json(self.root / "outputs/nexus_live.json")["races"][0]))
        self.assertNotIn("replay_evidence", self.row["prediction"])
        with patch.object(replay.PublishedReplay, "replay", side_effect=AssertionError("frozen records must not rerun")):
            repeated = self.run_forward()
        self.assertEqual(repeated["races"][0]["forecast_sha256"], row["forecast_sha256"])

    def test_opt_in_and_settle_only_do_not_reconstruct(self):
        with patch.object(replay.PublishedReplay, "replay", side_effect=AssertionError("must not replay")):
            report = forward.run(self.root, clock=self.clock)
            self.assertNotIn("candidate", report["races"][0])
            self.clear_feed()
            self.assertIsNone(self.run_forward(settle_only=True))

    def test_missing_original_history_and_unreviewed_publisher_are_unavailable(self):
        real_read = replay.PublishedReplay.read
        def missing_history(engine, ref, path):
            if path == replay.HISTORY:
                raise ValueError("original history missing")
            return real_read(engine, ref, path)
        with patch.object(replay.PublishedReplay, "read", missing_history):
            report = self.run_forward()
        self.assertNotIn("candidate", report["races"][0])
        self.clear_feed()
        with patch.object(replay, "AUDITED_PUBLISHERS", set()):
            report = self.run_forward()
        self.assertNotIn("candidate", report["races"][0])

    def test_original_input_hash_identity_and_result_are_not_repaired(self):
        for case in ("input_hash", "race", "car", "result", "winning_ids", "payoff", "future_history", "manifest"):
            with self.subTest(case=case):
                self.clear_feed()
                real_read = replay.PublishedReplay.read
                def bad_origin(engine, ref, path):
                    raw = real_read(engine, ref, path)
                    if path.endswith(self.row["race_id"] + ".json"):
                        value = json.loads(raw)
                        if case == "input_hash": value["input_sha256"] = "0" * 64
                        elif case == "race": value["data"]["race"]["id"] = "024620261020"
                        elif case == "car": value["data"]["entries"][0]["playerId"] = "wrong"
                        elif case == "result": value["data"]["results"] = [{"rank": 1}]
                        elif case == "winning_ids":
                            value["data"]["race"]["decidedAt"] = 0
                            value["data"]["trifectaWinningOddsIds"] = ["official-winner"]
                        elif case == "payoff":
                            value["data"]["race"]["decidedAt"] = 0
                            value["data"]["trifecta"][0]["payoffUnitPrice"] = 2270
                        return canonical(value).encode()
                    if case == "future_history" and path == replay.HISTORY:
                        value = dict(self.history, form={"future": [[999999, []]]})
                        return gzip.compress(canonical(value).encode())
                    if case == "manifest" and path == "models/nexus53/manifest.json":
                        return canonical(dict(self.manifest, id="different")).encode()
                    return raw
                with patch.object(replay.PublishedReplay, "read", bad_origin):
                    report = self.run_forward()
                self.assertNotIn("candidate", report["races"][0])

    def test_current_card_cancel_schedule_or_new_input_rejects_replay(self):
        original_row = copy.deepcopy(self.row)
        for case in ("rider", "lines", "schedule", "cancel", "input"):
            with self.subTest(case=case):
                self.clear_feed()
                self.row = copy.deepcopy(original_row)
                latest = copy.deepcopy(self.snapshot)
                if case == "rider": self.row["riders"][0]["pid"] = "changed"
                elif case == "lines": self.row["lines"] = [[[1], [2]]]
                elif case == "schedule": self.row["close_at"] = (NOW + timedelta(minutes=6)).isoformat()
                elif case == "cancel": latest["data"]["race"]["cancel"] = True
                else: latest["data"]["records"][0]["racePoint"] += 1
                save_json(self.snapshot_path, latest)
                self.save_live()
                report = self.run_forward()
                self.assertNotIn("candidate", report["races"][0])

    def test_replayed_probability_difference_is_strict_and_never_tuned(self):
        real_engine = replay.PublishedReplay._engine
        def changed_engine(instance, *args):
            engine, adapter, implementation = real_engine(instance, *args)
            predict = engine.predict
            def changed(matrix):
                result = predict(matrix)
                result["first_probabilities"]["1"] += 2e-12
                return result
            engine.predict = changed
            return engine, adapter, implementation
        with patch.object(replay.PublishedReplay, "_engine", changed_engine):
            report = self.run_forward()
        self.assertNotIn("candidate", report["races"][0])
        self.assertIn("再現確率", report["races"][0]["reason"])

    def test_close_result_and_delayed_replay_never_backfill(self):
        with patch.object(replay.PublishedReplay, "replay", side_effect=AssertionError("must not replay after outcome")):
            self.row.update(result_complete=True, payouts={"1-2-3": 1000})
            self.save_live()
            self.assertNotIn("candidate", self.run_forward()["races"][0])
            self.clear_feed()
            self.row.update(result_complete=False, payouts={})
            self.clock.value = NOW + timedelta(minutes=4)
            self.save_live()
            self.assertNotIn("candidate", self.run_forward()["races"][0])
        self.clear_feed()
        self.clock.value = NOW
        real_engine = replay.PublishedReplay._engine
        def delayed(instance, *args):
            values = real_engine(instance, *args)
            self.clock.value = NOW + timedelta(minutes=4)
            return values
        with patch.object(replay.PublishedReplay, "_engine", delayed):
            self.assertNotIn("candidate", self.run_forward()["races"][0])

    def test_final_publication_gate_checks_exact_source_and_latest_card(self):
        for case in ("probability", "rider", "lines", "deadline"):
            with self.subTest(case=case):
                self.clear_feed()
                self.clock.value = NOW
                original = copy.deepcopy(self.row)
                report = self.run_forward()
                self.assertIn("candidate", report["races"][0])
                if case == "probability": self.row["prediction"]["first_probabilities"]["1"] += .01
                elif case == "rider": self.row["riders"][0]["pid"] = "changed"
                elif case == "lines": self.row["lines"] = [[[1], [2]]]
                else: self.clock.value = NOW + timedelta(minutes=4)
                self.save_live()
                self.assertEqual(forward.gate_publication(self.root, {}, self.clock), [self.row["race_id"]])
                self.assertNotIn("candidate", read_json(self.root / forward.FEED)["races"][0])
                self.row = original
                self.save_live()

    def test_fresh_quotes_can_allocate_but_do_not_refresh_old_source_times(self):
        latest = copy.deepcopy(self.snapshot)
        latest["captured_at"] = (NOW - timedelta(seconds=15)).isoformat()
        latest["sha256"] = "c" * 64
        save_json(self.snapshot_path, latest)
        row = self.run_forward()["races"][0]
        self.assertEqual(row["candidate"]["auto_budget"], row["baseline"]["auto_budget"])
        self.assertEqual(row["candidate"]["auto_budget"], 5000)
        self.assertEqual(row["candidate"]["odds_at"], latest["captured_at"])
        self.assertEqual(row["source_replay"]["input_captured_at"], self.snapshot["captured_at"])

    def test_latest_positive_payoff_without_result_flags_is_not_a_preclose_quote(self):
        latest = copy.deepcopy(self.snapshot)
        latest["captured_at"] = (NOW - timedelta(seconds=15)).isoformat()
        latest["data"]["race"]["decidedAt"] = 0
        latest["data"]["trifecta"][0]["payoffUnitPrice"] = 2270
        save_json(self.snapshot_path, latest)
        report = self.run_forward()
        self.assertNotIn("candidate", report["races"][0])
        with self.assertRaises(ValueError):
            forward._quotes(self.root, self.row, self.row["prediction"], self.cars, NOW)


if __name__ == "__main__":
    unittest.main()
