"""Reproduce an already public forecast from its original local Git objects.

This is a separately dated reconstruction, never evidence that full scores were
saved with the old forecast. No source HTTP, model training, timestamp refresh,
or allocation occurs here. Missing provenance leaves the comparison unavailable.
"""
import copy
from datetime import date, datetime, timedelta
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from nexus_input_snapshots import SNAPSHOTS, canonical, read_json

KIND = "preclose_replay_of_published_forecast"
HISTORY = "data/nexus/history.json.gz"
PUBLISHER = "src/nexus_publish_saved_inputs.py"
# This audited publisher reads HISTORY before inference and commits any history
# advance together with the new forecast. Unreviewed historical writers fail
# closed; a shared through-date alone is never sufficient provenance.
AUDITED_PUBLISHERS = {"e85e3e901f392fcce786be7570f0ce1adf9de9d7"}
PUBLICATION_MESSAGE = "Publish NEXUS from existing pre-close source snapshots"
FLOAT_TOLERANCE = 1e-12
ALLOCATION_FIELDS = {"allocations", "auto_budget", "evaluated_budgets", "feasible_budgets",
    "allocation_policy", "allocation_saved_at", "auto_saved_at", "odds_at", "odds",
    "odds_source_sha256", "comparison_saved_at"}
FORECAST_FIELDS = {"model_id", "policy_id", "plan_changed", "search", "category", "count",
    "blocks", "tickets", "marks", "band_probabilities", "first_probabilities", "confidence"}
CARD_FIELDS = ("race_id", "date", "race_no", "start_at", "close_at", "riders", "lines")


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def blob_hash(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware replay timestamp required")
    return result


def forecast_core(prediction):
    """The exact public source forecast, excluding independently updated odds."""
    if not isinstance(prediction, dict):
        raise ValueError("published source forecast required")
    return {key: copy.deepcopy(value) for key, value in prediction.items()
            if key not in ALLOCATION_FIELDS and key not in ("replay_evidence", "replay_evidence_error", "q")}


def card_digest(row):
    return digest({key: row.get(key) for key in CARD_FIELDS})


def model_input_hash(snapshot):
    data, meta = snapshot["data"], snapshot["meta"]
    model_data = {key: data[key] for key in ("players", "entries", "records", "linePrediction") if key in data}
    model_data["race"] = {key: data["race"].get(key) for key in ("id", "distance", "startAt", "closeAt")}
    model_data["schedule"] = {key: data["schedule"].get(key) for key in ("date", "cupId", "index", "day")}
    model_meta = {key: value for key, value in meta.items() if key not in ("source_url", "session")}
    return digest([model_meta, model_data])


def has_market_result(data):
    """Some official outcomes carry winning IDs/payouts without result flags."""
    if data.get("trifectaWinningOddsIds"):
        return True
    for quote in data.get("trifecta", []) or []:
        if not isinstance(quote, dict):
            raise ValueError("invalid source market record")
        value = quote.get("payoffUnitPrice")
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("invalid source payoff value")
        if number > 0:
            return True
    return False


def _equal(actual, expected, path="prediction"):
    """Every preserved field must match; finite float differences <= 1e-12."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        actual, expected = ({str(k): v for k, v in value.items()} for value in (actual, expected))
        if actual.keys() != expected.keys():
            raise ValueError("再現項目が原本と一致しません: " + path)
        return max((_equal(actual[k], expected[k], path + "." + k) for k in actual), default=0.0)
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("再現項目数が原本と一致しません: " + path)
        return max((_equal(a, b, path + "." + str(i)) for i, (a, b) in enumerate(zip(actual, expected))), default=0.0)
    if (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float)) and not isinstance(expected, bool)):
        delta = abs(actual - expected)
        if not math.isfinite(actual) or not math.isfinite(expected) or delta > FLOAT_TOLERANCE:
            raise ValueError("再現確率が原本と一致しません: " + path)
        return delta
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError("再現予想が原本と一致しません: " + path)
    return 0.0


def _open(row, now, lead):
    close = stamp(row["close_at"])
    if (now + timedelta(seconds=lead) >= close or row.get("closed") or row.get("cancelled")
            or row.get("result_complete") or row.get("winners") or row.get("payouts")
            or row.get("refunded") or row.get("refund_all") or row["date"] != now.astimezone(close.tzinfo).date().isoformat()):
        raise ValueError("締切前の未確定レースだけ保存済み予想を再現できます")


def _snapshot(snapshot, row, prediction, now, *, original):
    data, meta = snapshot["data"], snapshot["meta"]
    race, schedule = data["race"], data["schedule"]
    captured, close, start = stamp(snapshot["captured_at"]), stamp(row["close_at"]), stamp(row["start_at"])
    if (snapshot.get("schema") != 1 or snapshot.get("source_kind") != "existing_public_page_response"
            or meta.get("race_id") != row["race_id"] or meta.get("date") != row["date"]
            or str(race.get("id")) != row["race_id"] or race.get("number") != row["race_no"]
            or schedule.get("date") != row["date"].replace("-", "")
            or race.get("cancel") is not False or race.get("decidedAt") or data.get("results")
            or data.get("finalOdds") or has_market_result(data) or meta.get("cancel") is not False
            or float(race["closeAt"]) != close.timestamp() or float(meta["close_at"]) != close.timestamp()
            or float(race["startAt"]) != start.timestamp() or float(meta["start_at"]) != start.timestamp()
            or not captured <= now < close or model_input_hash(snapshot) != snapshot.get("input_sha256")
            or snapshot.get("input_sha256") != prediction.get("input_sha256")):
        raise ValueError("元入力の原本・日時・レース情報が一致しません")
    entries = data["entries"]
    if (any(type(e.get("absent")) is not bool or str(e.get("raceId")) != row["race_id"] for e in entries)
            or sorted((e["number"], str(e["playerId"])) for e in entries if not e["absent"])
            != sorted((r["car"], str(r["pid"])) for r in row["riders"])):
        raise ValueError("原本から出走選手が変わっています")
    if original:
        saved = stamp(prediction["captured_at"])
        if (captured != stamp(prediction["input_captured_at"])
                or not 0 <= (saved - captured).total_seconds() <= 1800
                or snapshot.get("sha256") != prediction.get("source_sha256")):
            raise ValueError("元予想の入力時刻・出典を確認できません")


def verify_record(record, row=None):
    if not isinstance(record, dict) or record.get("schema") != 1 or record.get("evidence_kind") != KIND:
        raise ValueError("unexpected source replay evidence")
    if record.get("evidence_sha256") != digest({k: v for k, v in record.items() if k != "evidence_sha256"}):
        raise ValueError("source replay integrity mismatch")
    if record.get("original_full_q_observed") is not False or record.get("exact_replay_match") is not True:
        raise ValueError("original score availability must not be rewritten")
    error = record.get("max_preserved_numeric_error")
    if isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(error) or not 0 <= error <= FLOAT_TOLERANCE:
        raise ValueError("invalid original forecast equality proof")
    for key in ("model_manifest_sha256", "input_sha256", "source_sha256", "history_sha256",
                "implementation_sha256", "source_baseline_sha256", "input_snapshot_sha256", "source_card_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(record.get(key))):
            raise ValueError("complete replay SHA-256 required")
    for key in ("original_publication_commit", "history_git_blob", "input_snapshot_git_blob",
                "publisher_git_blob", "implementation_git_tree", "original_forecast_git_blob"):
        if not re.fullmatch(r"[a-f0-9]{40}", str(record.get(key))):
            raise ValueError("complete replay Git object required")
    if record["publisher_git_blob"] not in AUDITED_PUBLISHERS:
        raise ValueError("unreviewed historical publisher")
    inputs, predicted, published, replayed, close = [stamp(record[k]) for k in (
        "input_captured_at", "original_prediction_at", "original_published_at", "replayed_at", "close_at")]
    if not inputs <= predicted <= published <= replayed < close or (predicted - inputs).total_seconds() > 1800:
        raise ValueError("invalid replay provenance timestamps")
    if date.fromisoformat(record["feature_history_through"]) >= date.fromisoformat(record["date"]):
        raise ValueError("replayed history reaches race day")
    cars, q = record["cars"], record["q"]
    if (not isinstance(cars, list) or not 4 <= len(cars) <= 9 or len(cars) != len(set(cars))
            or any(type(c) is not int or not 1 <= c <= 9 for c in cars)
            or record.get("q_order") != "itertools.permutations(cars,3)"
            or not isinstance(q, list) or len(q) != math.perm(len(cars), 3)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in q)
            or abs(math.fsum(q) - 1) > 1e-8):
        raise ValueError("invalid complete reconstructed scores")
    if row is not None:
        if (record["race_id"] != row["race_id"] or record["date"] != row["date"]
                or close != stamp(row["close_at"]) or replayed > stamp(row["saved_at"])
                or set(cars) != {r["car"] for r in row["riders"]}
                or digest(forecast_core(row["baseline"])) != record["source_baseline_sha256"]
                or record["evidence_sha256"] != row["source_evidence_sha256"]):
            raise ValueError("replay source differs from frozen comparison")


class PublishedReplay:
    """Per-run cache of verified local historical packages and immutable inputs."""
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}
        self.packages = {}
        self.directory = tempfile.TemporaryDirectory(prefix="nexus-source-replay-")

    def close(self):
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in self.packages):
                sys.modules.pop(name, None)
        self.directory.cleanup()

    def git(self, *args):
        try:
            # Also disable implicit network access from partial-clone lazy fetch.
            env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_ALLOW_PROTOCOL="", GIT_TERMINAL_PROMPT="0")
            return subprocess.run(["git", *args], cwd=self.root, check=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=20, env=env).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("元の公開 Git オブジェクトを確認できません") from exc

    def read(self, ref, path):
        key = (ref, path)
        if key not in self.cache:
            self.cache[key] = self.git("show", ref + ":" + path)
        return self.cache[key]

    def _origin(self, row, prediction, current):
        captured = stamp(prediction["captured_at"])
        path = "outputs/nexus_forecasts/" + row["date"] + ".json"
        refs = self.git("log", "--format=%H", "--max-count=16",
                        "--since=" + (captured - timedelta(seconds=300)).isoformat(),
                        "--until=" + (captured + timedelta(seconds=300)).isoformat(), "--", path).decode().splitlines()
        for ref in reversed(refs):
            if not re.fullmatch(r"[a-f0-9]{40}", ref):
                continue
            info = self.git("show", "-s", "--format=%cI%n%s", ref).decode().splitlines()
            published = stamp(info[0])
            if (len(info) != 2 or info[1] != PUBLICATION_MESSAGE
                    or not captured <= published <= current or (published - captured).total_seconds() > 300
                    or published >= stamp(row["close_at"])):
                continue
            publisher = self.read(ref, PUBLISHER)
            if blob_hash(publisher) not in AUDITED_PUBLISHERS:
                continue
            raw = self.read(ref, path)
            rows = json.loads(raw)
            if not isinstance(rows, list) or len({r["race_id"] for r in rows}) != len(rows):
                raise ValueError("元の公開予想一覧を確認できません")
            source = next((r for r in rows if r.get("race_id") == row["race_id"]), None)
            if source is None or digest(forecast_core(source.get("prediction"))) != digest(forecast_core(prediction)):
                continue
            if card_digest(source) != card_digest(row):
                raise ValueError("元の公開から出走表・ライン・締切が変わっています")
            if source.get("closed") or source.get("cancelled") or source.get("result_complete") or source.get("payouts"):
                raise ValueError("元の公開が締切前・結果未確定と確認できません")
            return ref, published, raw, publisher
        raise ValueError("元予想と同じ履歴を証明する公開原本がありません")

    def _engine(self, ref, manifest_hash):
        listing = self.git("ls-tree", "-r", ref, "--", "src/nexus_model").decode().splitlines()
        entries = {}
        for line in listing:
            header, path = line.split("\t", 1)
            mode, kind, oid = header.split()
            if path.endswith(".py"):
                if mode != "100644" or kind != "blob" or Path(path).parent.as_posix() != "src/nexus_model":
                    raise ValueError("unsupported original model implementation")
                entries[path] = oid
        if "src/nexus_model/engine.py" not in entries or "src/nexus_model/inputs.py" not in entries:
            raise ValueError("original implementation missing")
        implementation = digest(entries)
        name = "_nexus_source_replay_" + implementation[:20]
        manifest_raw = self.read(ref, "models/nexus53/manifest.json")
        if hashlib.sha256(manifest_raw).hexdigest() != manifest_hash:
            raise ValueError("現在の固定モデルと原本モデルが一致しません")
        if name not in self.packages:
            folder = Path(self.directory.name) / name
            folder.mkdir()
            for path, oid in entries.items():
                raw = self.read(ref, path)
                if blob_hash(raw) != oid:
                    raise ValueError("original implementation Git hash mismatch")
                (folder / Path(path).name).write_bytes(raw)
            spec = importlib.util.spec_from_file_location(name, folder / "__init__.py", submodule_search_locations=[str(folder)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[name] = package
            # Register before importing so cleanup also covers partial imports.
            self.packages[name] = None
            spec.loader.exec_module(package)
            engine_module = importlib.import_module(name + ".engine")
            inputs_module = importlib.import_module(name + ".inputs")
            self.packages[name] = (engine_module.Engine(self.root / "models/nexus53"), inputs_module)
        engine, inputs_module = self.packages[name]
        return engine, inputs_module, implementation

    def replay(self, row, manifest, manifest_hash, clock, lead=60):
        current = clock()
        _open(row, current, lead)
        prediction = row["prediction"]
        if prediction.get("replay_evidence") or prediction.get("model_id") != manifest["id"]:
            raise ValueError("legacy original forecast required")
        ref, published, original_raw, publisher = self._origin(row, prediction, current)
        snapshot_path = (SNAPSHOTS / row["date"] / "preclose" / (row["race_id"] + ".json")).as_posix()
        snapshot_raw = self.read(ref, snapshot_path)
        snapshot = json.loads(snapshot_raw)
        _snapshot(snapshot, row, prediction, current, original=True)
        latest = read_json(self.root / snapshot_path)
        if latest is not None:
            _snapshot(latest, row, prediction, current, original=False)
        engine, adapter, implementation = self._engine(ref, manifest_hash)
        history_raw = self.read(ref, HISTORY)
        history_path = Path(self.directory.name) / (blob_hash(history_raw) + ".json.gz")
        if not history_path.exists():
            history_path.write_bytes(history_raw)
        history = adapter.History.load(history_path)
        through = history.data["through"]
        if through != prediction["history_through"] or date.fromisoformat(through) >= date.fromisoformat(row["date"]):
            raise ValueError("元の学習履歴の原本・期間が一致しません")
        cutoff = date.fromisoformat(through).toordinal()
        for key in ("form", "tactics", "tactic_roles", "pairs", "pooled", "roles"):
            if any(type(event[0]) is not int or event[0] > cutoff
                   for events in history.data.get(key, {}).values() for event in events):
                raise ValueError("原本履歴に対象期間より後のデータがあります")
        _open(row, clock(), lead)
        clean = copy.deepcopy(snapshot["data"])
        for key in ("results", "trifecta", "finalOdds", "trifectaWinningOddsIds", "trifectaPayoffStatus"):
            clean.pop(key, None)
        matrix, line = adapter.build(history, clean, snapshot["meta"])
        actual = engine.predict(matrix)
        expected = {k: copy.deepcopy(prediction[k]) for k in FORECAST_FIELDS}
        assessed = {k: copy.deepcopy(actual[k]) for k in FORECAST_FIELDS}
        # The original publisher appended this receipt after inference. The
        # annotation wall clock is not a model output and is never backdated.
        assessed["confidence"]["source_captured_at"] = prediction["captured_at"]
        expected["confidence"].pop("annotated_at", None)
        assessed["confidence"].pop("annotated_at", None)
        error = _equal(assessed, expected)
        if line["known_unchanged_groups"] != row.get("lines"):
            raise ValueError("再現したラインが保存原本と一致しません")
        finished = clock()
        _open(row, finished, lead)
        record = dict(schema=1, evidence_kind=KIND, original_full_q_observed=False,
            exact_replay_match=True, max_preserved_numeric_error=error,
            original_publication_commit=ref, original_published_at=published.isoformat(),
            original_prediction_at=prediction["captured_at"], input_captured_at=prediction["input_captured_at"],
            replayed_at=finished.isoformat(), race_id=row["race_id"], date=row["date"], close_at=row["close_at"],
            model_id=manifest["id"], model_manifest_sha256=manifest_hash,
            source_baseline_sha256=digest(forecast_core(prediction)), source_card_sha256=card_digest(row), input_sha256=prediction["input_sha256"],
            source_sha256=prediction["source_sha256"], input_snapshot_sha256=hashlib.sha256(snapshot_raw).hexdigest(),
            input_snapshot_git_blob=blob_hash(snapshot_raw), original_forecast_git_blob=blob_hash(original_raw),
            publisher_git_blob=blob_hash(publisher), history_git_blob=blob_hash(history_raw),
            history_sha256=hashlib.sha256(history_raw).hexdigest(), feature_history_through=through,
            implementation_sha256=implementation,
            implementation_git_tree=self.git("rev-parse", ref + ":src/nexus_model").decode().strip(),
            runtime=dict(python=sys.version.split()[0], numpy=importlib.import_module("numpy").__version__,
                         lightgbm=importlib.import_module("lightgbm").__version__),
            cars=[int(c) for c in matrix["rows"][0]["cars"]], q=[float(v) for v in actual["q"]],
            q_order="itertools.permutations(cars,3)")
        record["evidence_sha256"] = digest(record)
        verify_record(record)
        return record
