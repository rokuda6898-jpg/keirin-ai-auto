"""Offline, gated research for the original 27,351-race Nexus benchmark.

This module never fetches data, trains a model, writes a live forecast, or places
a bet.  A manifest total is a reference value, not a substitute for a replay.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "nexus-formation-5385-v1"
PROTOCOL_ID = "nexus-annual-flat100-points-v2"
MODEL_FINGERPRINT = "d7ad899a9280e673608f1fed533423b22b79e04b8ab63be1ff2766b6c22e1ed5"
POINTS = (6, 8, 10, 12)
UNIT_YEN = 100
MAX_FORMATIONS = 4
EXCLUDED_REFUND_RACE_IDS = ("031120260425", "098720260529")
HUNDREDFOLD_DEFINITION = "Race with at least one selected official winning ticket paying >=10,000 yen per 100 yen; a dead-heat race counts once and refunds do not count."
JST = timezone(timedelta(hours=9))
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
MODEL_KEYS = ("id", "train", "historical_period", "historical", "counts",
              "max_formations", "score_selection", "formation_selection",
              "team_selection", "alignment", "v2_calibration")


class ReplayError(ValueError):
    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code, self.details = code, details

    def report(self):
        return dict(status="blocked", error=self.code, message=str(self),
                    details=self.details, research_only=True,
                    annual_candidate_result_generated=False)


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str = "nexus-27351-2025-10-09-2026-10-08"
    first: str = "2025-10-09"
    development_last: str = "2026-04-08"
    last: str = "2026-10-08"
    races: int = 27351
    hits: int = 14729
    stake_yen: int = 43596000
    payout_yen: int = 33783680
    model_fingerprint: str | None = MODEL_FINGERPRINT
    hundredfold_hit_races: int | None = 265
    refund_accounting: str = "established_only"


ANNUAL = Benchmark()


@dataclass(frozen=True)
class Race:
    race_id: str
    day: str
    cars: tuple[int, ...]
    q: tuple[float, ...] | None
    original: tuple[tuple[int, int, int], ...]
    payouts: tuple[tuple[tuple[int, int, int], int], ...]
    refunds: frozenset[tuple[int, int, int]]
    refund_all: bool
    timing_unverified: bool


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _object_pairs(pairs):
    answer = {}
    for key, value in pairs:
        if key in answer:
            raise ReplayError("duplicate_json_key", "JSON object contains a duplicate key", key=key)
        answer[key] = value
    return answer


def read_json(path):
    try:
        return json.loads(Path(path).read_text("utf-8"), object_pairs_hook=_object_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError("input_unreadable", "Cannot read required JSON input", path=str(path)) from exc


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ReplayError("invalid_integer", "Expected a non-boolean integer", field=name, value=value)
    return value


def _day(value, name):
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("noncanonical date")
    except (TypeError, ValueError) as exc:
        raise ReplayError("invalid_date", "Expected YYYY-MM-DD", field=name, value=value) from exc
    return value


def _identity(race_id, day, spec):
    _day(day, "date")
    if (not isinstance(race_id, str) or not re.fullmatch(r"\d{12}", race_id)
            or not 1 <= int(race_id[:2]) <= 12
            or race_id[4:] != day.replace("-", "")):
        raise ReplayError("race_identity", "Race id and date do not identify the same race", race_id=race_id, date=day)
    if not spec.first <= day <= spec.last:
        raise ReplayError("outside_period", "Race is outside the frozen annual period", race_id=race_id, date=day)
    if spec.refund_accounting == "established_only" and race_id in EXCLUDED_REFUND_RACE_IDS:
        raise ReplayError("benchmark_refund_exclusion", "The original annual study explicitly excluded this refunded race; it must not enter the 27,351-race cohort", race_id=race_id)


def _hash(value, name):
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ReplayError("invalid_hash", "Expected a lowercase SHA-256 digest", field=name)
    return value


def _ticket(value, cars):
    if isinstance(value, str):
        if not re.fullmatch(r"[1-9]-[1-9]-[1-9]", value):
            raise ReplayError("invalid_ticket", "Ticket must use canonical car numbers", ticket=value)
        value = tuple(map(int, value.split("-")))
    elif isinstance(value, (list, tuple)):
        value = tuple(value)
    else:
        raise ReplayError("invalid_ticket", "Expected a triple or 1-2-3 string")
    if (len(value) != 3 or any(type(x) is not int for x in value)
            or len(set(value)) != 3 or not set(value) <= set(cars)):
        raise ReplayError("invalid_ticket", "Ticket is inconsistent with the frozen rider field", ticket=value)
    return value


def ticket_text(ticket):
    return "-".join(map(str, ticket))


def expand_formations(blocks, cars):
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= MAX_FORMATIONS:
        raise ReplayError("formation_count", "Every plan must contain one to four complete formations")
    tickets = set()
    for block in blocks:
        if not isinstance(block, dict):
            raise ReplayError("formation_shape", "Formation must be a JSON object")
        pools = []
        for name in ("first", "second", "third"):
            pool = block.get(name)
            if (not isinstance(pool, list) or not pool
                    or any(type(c) is not int or c not in cars for c in pool)
                    or len(set(pool)) != len(pool)):
                raise ReplayError("formation_pool", "Formation contains invalid or duplicate car choices", pool=name)
            pools.append(pool)
        generated = {t for t in itertools.product(*pools) if len(set(t)) == 3}
        if "count" in block and _integer(block["count"], "formation.count") != len(generated):
            raise ReplayError("formation_expansion", "Formation count differs from its complete expansion")
        # The recovered original audit expands every complete block, then buys
        # each ticket in their union once. Overlap is valid, not extra stake.
        tickets.update(generated)
    if not tickets:
        raise ReplayError("formation_expansion", "The complete formation union contains no valid ticket")
    return tuple(sorted(tickets))


def _instant(value, name):
    try:
        if type(value) in (int, float) and math.isfinite(value):
            return datetime.fromtimestamp(value, timezone.utc)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone missing")
        return parsed
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise ReplayError("invalid_timestamp", "Timestamp must include an explicit timezone", field=name) from exc


def _parse_race(obj, model_sha, spec):
    if not isinstance(obj, dict) or obj.get("schema") != 1:
        raise ReplayError("ledger_schema", "Expected one canonical schema=1 race per line")
    rid, day = obj.get("race_id"), obj.get("date")
    _identity(rid, day, spec)
    cars = obj.get("cars")
    if (not isinstance(cars, list) or not 3 <= len(cars) <= 9
            or any(type(c) is not int or not 1 <= c <= 9 for c in cars)
            or len(cars) != len(set(cars))):
        raise ReplayError("rider_field", "cars must preserve the distinct original model order", race_id=rid)
    if obj.get("model_id") != MODEL_ID:
        raise ReplayError("model_identity", "Forecast is not the fixed Nexus model", race_id=rid)
    if _hash(obj.get("model_manifest_sha256"), "model_manifest_sha256") != model_sha:
        raise ReplayError("model_hash_mismatch", "Forecast model manifest does not match the frozen source", race_id=rid)
    if obj.get("training_cutoff_exclusive") != spec.first:
        raise ReplayError("training_leakage", "Original annual model must be frozen before the entire annual period", race_id=rid)
    history_day = obj.get("feature_history_through")
    if history_day is not None and _day(history_day, "feature_history_through") >= day:
        raise ReplayError("history_leakage", "History includes the current race day or a future day", race_id=rid)
    snapshot, close = obj.get("snapshot_at"), obj.get("close_at")
    input_captured = obj.get("input_captured_at")
    if (snapshot is None) != (close is None):
        raise ReplayError("incomplete_timing", "snapshot_at and close_at must be supplied together", race_id=rid)
    if snapshot is not None:
        saved, closes = _instant(snapshot, "snapshot_at"), _instant(close, "close_at")
        if (saved >= closes or closes.astimezone(JST).date().isoformat() != day
                or saved.astimezone(JST).date().isoformat() != day):
            raise ReplayError("forecast_timing", "A declared saved forecast must precede its own race close", race_id=rid)
    if input_captured is not None:
        if snapshot is None:
            raise ReplayError("incomplete_timing", "input_captured_at requires an original saved forecast timestamp", race_id=rid)
        captured = _instant(input_captured, "input_captured_at")
        if captured > saved:
            raise ReplayError("input_timing", "Input was captured after the declared forecast was saved", race_id=rid)
    for name in ("source_sha256", "input_sha256"):
        if obj.get(name) is not None:
            _hash(obj[name], name)
    listed = obj.get("original_tickets")
    if not isinstance(listed, list):
        raise ReplayError("original_missing", "Original annual tickets are required", race_id=rid)
    original = tuple(_ticket(t, cars) for t in listed)
    if len(original) not in (6, 15, 24) or len(original) != len(set(original)):
        raise ReplayError("original_count", "Original plan must have 6, 15 or 24 unique tickets", race_id=rid)
    if tuple(sorted(original)) != expand_formations(obj.get("original_formations"), cars):
        raise ReplayError("original_formation_mismatch", "Original tickets differ from their complete formations", race_id=rid)
    raw_q = obj.get("q")
    q = None
    if raw_q is not None:
        if (not isinstance(raw_q, list) or len(raw_q) != math.perm(len(cars), 3)
                or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in raw_q)
                or abs(math.fsum(raw_q) - 1.0) > 1e-6):
            raise ReplayError("incomplete_q", "q must contain every permutation, finite and summing to one", race_id=rid)
        q = tuple(float(x) for x in raw_q)
    outcome = obj.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("status") not in ("settled", "refunded"):
        raise ReplayError("missing_result", "Every original race needs a finalized official outcome; no subset is allowed", race_id=rid)
    raw_payouts, raw_refunds = outcome.get("payouts_per_100_yen"), outcome.get("refund_tickets")
    refund_all = outcome.get("refund_all")
    if not isinstance(raw_payouts, dict) or not isinstance(raw_refunds, list) or type(refund_all) is not bool:
        raise ReplayError("outcome_schema", "Official payouts and explicit refund status are required", race_id=rid)
    payouts = tuple((_ticket(t, cars), _integer(v, "payout_per_100_yen", 1)) for t, v in raw_payouts.items())
    refunds = tuple(_ticket(t, cars) for t in raw_refunds)
    if len(refunds) != len(set(refunds)) or set(refunds) & {t for t, _ in payouts}:
        raise ReplayError("refund_conflict", "A ticket cannot be both a winner and a refund", race_id=rid)
    if (refund_all and (payouts or outcome["status"] != "refunded")) or (not refund_all and (not payouts or outcome["status"] != "settled")):
        raise ReplayError("outcome_incomplete", "Incomplete or inconsistent official result", race_id=rid)
    if spec.refund_accounting == "established_only" and (refund_all or refunds):
        raise ReplayError("annual_refund_accounting", "The original annual benchmark contains established races and winning payouts only; refund credits are not part of this comparison", race_id=rid)
    return Race(rid, day, tuple(cars), q, tuple(sorted(original)), payouts,
                frozenset(refunds), refund_all,
                snapshot is None or input_captured is None or history_day is None)


def settle(race, tickets):
    unique = tuple(tickets)
    if not unique or len(unique) != len(set(unique)):
        raise ReplayError("candidate_duplicates", "Each ticket must be bought exactly once")
    selected = set(unique)
    payouts = dict(race.payouts)
    won = selected & payouts.keys()
    refund_yen = UNIT_YEN * (len(selected) if race.refund_all else len(selected & race.refunds))
    return dict(race_id=race.race_id, date=race.day, races=1, hits=int(bool(won)),
                tickets=len(selected), stake_yen=UNIT_YEN * len(selected),
                payout_yen=sum(payouts[t] for t in won) + refund_yen,
                refund_yen=refund_yen,
                hundredfold_hit_races=int(any(payouts[t] >= 10000 for t in won)))


def summarize(rows):
    rows = list(rows)
    keys = ("races", "hits", "tickets", "stake_yen", "payout_yen", "refund_yen", "hundredfold_hit_races")
    answer = {k: sum(r[k] for r in rows) for k in keys}
    n, stake = answer["races"], answer["stake_yen"]
    return answer | dict(hit_rate=answer["hits"] / n if n else None,
                         roi=answer["payout_yen"] / stake if stake else None,
                         profit_yen=answer["payout_yen"] - stake,
                         average_points=answer["tickets"] / n if n else None)


def _model_info(path, spec):
    if spec.refund_accounting not in ("established_only", "gross_with_refunds"):
        raise ReplayError("benchmark_accounting", "Unknown frozen benchmark refund convention")
    model = read_json(path)
    try:
        frozen = {k: model[k] for k in MODEL_KEYS}
        frozen["assets"] = {k: v["sha"] for k, v in model["assets"].items()}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ReplayError("model_manifest_shape", "The complete original model manifest is required") from exc
    actual_fingerprint = fingerprint(frozen)
    if spec.model_fingerprint is not None and actual_fingerprint != spec.model_fingerprint:
        raise ReplayError("model_policy_changed", "Training, model assets, calibration or original policy differ from the annual benchmark")
    expected_train_end = (date.fromisoformat(spec.first) - timedelta(days=1)).isoformat()
    if (model.get("id") != MODEL_ID or model.get("train", [None])[-1] != expected_train_end
            or model.get("historical_period") != [spec.first, spec.last]):
        raise ReplayError("model_period", "Model must be frozen before the whole reference annual period")
    return dict(manifest_sha256=file_sha(path), frozen_model_fingerprint=actual_fingerprint,
                model_id=MODEL_ID, train_end=expected_train_end,
                declared_assets=len(frozen["assets"]),
                proof_scope="Manifest and ledger consistency; not a re-inference or historical input-availability proof")


def audit(ledger_path, cohort_path, model_path, *, spec=ANNUAL):
    model = _model_info(model_path, spec)
    cohort = read_json(cohort_path)
    if (not isinstance(cohort, dict) or cohort.get("schema") != 1
            or cohort.get("benchmark_id") != spec.benchmark_id
            or cohort.get("model_manifest_sha256") != model["manifest_sha256"]):
        raise ReplayError("cohort_identity", "The original cohort manifest must identify this exact benchmark and model")
    if not isinstance(cohort.get("source_description"), str) or not cohort["source_description"].strip():
        raise ReplayError("cohort_provenance", "Describe the original study source used to obtain the race-id inventory")
    _hash(cohort.get("source_sha256"), "cohort.source_sha256")
    ids = cohort.get("race_ids")
    if (not isinstance(ids, list) or any(not isinstance(r, str) for r in ids)
            or len(ids) != spec.races or len(set(ids)) != len(ids)):
        raise ReplayError("cohort_count", "Original cohort must contain every unique reference race id", expected=spec.races)
    for rid in ids:
        day = f"{rid[4:8]}-{rid[8:10]}-{rid[10:12]}"
        _identity(rid, day, spec)
    races, seen = [], set()
    try:
        opener = gzip.open if str(ledger_path).endswith(".gz") else open
        with opener(ledger_path, "rt", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    raise ReplayError("blank_ledger_line", "Blank rows are not silently skipped", line=line_no)
                obj = json.loads(line, object_pairs_hook=_object_pairs)
                race = _parse_race(obj, model["manifest_sha256"], spec)
                if race.race_id in seen:
                    raise ReplayError("duplicate_race", "Ledger repeats a race", race_id=race.race_id)
                races.append(race)
                seen.add(race.race_id)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError("ledger_unreadable", "Cannot read the complete canonical JSONL ledger", path=str(ledger_path)) from exc
    expected = set(ids)
    if seen != expected:
        missing, extra = sorted(expected - seen), sorted(seen - expected)
        raise ReplayError("cohort_mismatch", "Ledger must contain exactly the original annual cohort",
                          missing_count=len(missing), extra_count=len(extra), missing=missing[:20], extra=extra[:20])
    races.sort(key=lambda r: (r.day, r.race_id))
    baseline = summarize(settle(r, r.original) for r in races)
    wanted = dict(races=spec.races, hits=spec.hits, stake_yen=spec.stake_yen, payout_yen=spec.payout_yen)
    if spec.hundredfold_hit_races is not None:
        wanted["hundredfold_hit_races"] = spec.hundredfold_hit_races
    mismatch = {k: dict(expected=v, actual=baseline[k]) for k, v in wanted.items() if baseline[k] != v}
    if mismatch:
        raise ReplayError("benchmark_mismatch", "Original benchmark was not reproduced; candidate research is blocked", mismatches=mismatch)
    report = dict(schema=1, status="baseline_reproduced", research_only=True,
                  benchmark=asdict(spec), baseline=baseline, model=model,
                  ledger_sha256=file_sha(ledger_path), cohort_sha256=file_sha(cohort_path),
                  cohort_race_ids_sha256=fingerprint(sorted(ids)),
                  full_q_races=sum(r.q is not None for r in races),
                  historical_timing_unverified_races=sum(r.timing_unverified for r in races),
                  historical_input_availability_proven=False,
                  formation_accounting="Union of one to four fully expanded blocks; overlapping tickets are bought once. Per-block count, when present, counts the complete block before cross-block deduplication.",
                  excluded_refund_race_ids=list(EXCLUDED_REFUND_RACE_IDS) if spec.refund_accounting == "established_only" else [],
                  refund_accounting=("Original source convention: exclude the two named fully refunded races; 27,351 established races only. Stake is 100 yen per unique selected ticket and return is the sum of official winning payouts, without refund credits."
                                     if spec.refund_accounting == "established_only" else
                                     "Explicit alternate fixture convention: gross stake includes refunded tickets and returns include refund credits; this is not the original annual benchmark."),
                  hundredfold_definition=HUNDREDFOLD_DEFINITION,
                  source_definition_scope="Formation union, the two refund exclusions and the 265-race hundredfold definition are supported by the recovered original audit code. Code inspection alone does not reproduce annual row-level results.",
                  limitations=["The annual period was repeatedly studied; the later period is a chronological test, not an untouched holdout.",
                               "Matching totals and declared hashes does not prove historical feature availability or reproduce every model inference.",
                               "Uniform 100-yen-per-ticket returns are separate from any 5,000-10,000-yen allocation simulation."])
    return races, report


def candidate_specs():
    return [dict(id=f"q-{k}-form4-v1", points=k, max_formations=MAX_FORMATIONS,
                 selector="frozen optimizer.select on original full q; exact top set if packable, otherwise declared formation search") for k in POINTS]


def implementation_hashes():
    paths = [Path(__file__)] + [ROOT / "src" / "nexus_model" / (n + ".py")
                               for n in ("optimizer", "forms", "shape", "exact")]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def _write_once(path, value, *, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="research-", suffix=".tmp", delete=False) as stream:
            name = stream.name
            if jsonl:
                for row in value:
                    stream.write(canonical(row) + b"\n")
            else:
                stream.write(canonical(value) + b"\n")
        try:
            os.link(name, path)
        except FileExistsError as exc:
            raise ReplayError("artifact_already_exists", "A frozen research artifact cannot be overwritten", path=str(path)) from exc
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def _require_q(races):
    missing = [r.race_id for r in races if r.q is None]
    if missing:
        raise ReplayError("full_q_required", "All original races require full q before any candidate is evaluated",
                          missing_count=len(missing), missing=missing[:20])
    small = [r.race_id for r in races if math.perm(len(r.cars), 3) < max(POINTS)]
    if small:
        raise ReplayError("candidate_field_too_small", "A frozen candidate cannot cover the full cohort; do not silently reduce or skip it", race_ids=small[:20])


def prepare(ledger_path, cohort_path, model_path, run_dir, *, spec=ANNUAL):
    races, report = audit(ledger_path, cohort_path, model_path, spec=spec)
    _require_q(races)
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ReplayError("run_directory_not_empty", "Use a new empty directory; never overwrite a declared study")
    if not any(r.day <= spec.development_last for r in races) or not any(r.day > spec.development_last for r in races):
        raise ReplayError("empty_time_split", "Both frozen time periods require races")
    protocol = dict(schema=1, protocol_id=PROTOCOL_ID, research_only=True,
                    declared_at=datetime.now(timezone.utc).isoformat(), benchmark=asdict(spec),
                    development=[spec.first, spec.development_last],
                    chronological_validation=[(date.fromisoformat(spec.development_last) + timedelta(days=1)).isoformat(), spec.last],
                    candidates=candidate_specs(), unit_yen=UNIT_YEN, all_races_required=True,
                    source_paths=dict(ledger=str(Path(ledger_path).resolve()), cohort=str(Path(cohort_path).resolve()), model_manifest=str(Path(model_path).resolve())),
                    source_hashes=dict(ledger=report["ledger_sha256"], cohort=report["cohort_sha256"], model_manifest=report["model"]["manifest_sha256"]),
                    implementation_hashes=implementation_hashes(),
                    selection_rule="Highest pooled development ROI; ties: fewer tickets, then higher hit rate, then candidate id.",
                    validation_rule="Evaluate only the development-selected candidate once; no parameter change after viewing it.",
                    untouched_holdout=False, baseline_reproduction=report)
    protocol["protocol_sha256"] = fingerprint(protocol)
    _write_once(run_dir / "protocol.json", protocol)
    return protocol


def _locked_inputs(run_dir, spec):
    protocol = read_json(Path(run_dir) / "protocol.json")
    digest = protocol.get("protocol_sha256")
    if digest != fingerprint({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ReplayError("protocol_changed", "The declared protocol fingerprint does not match")
    if (protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("benchmark") != asdict(spec)
            or protocol.get("candidates") != candidate_specs()
            or protocol.get("implementation_hashes") != implementation_hashes()):
        raise ReplayError("protocol_changed", "Protocol, candidate family or implementation changed after declaration")
    paths = protocol["source_paths"]
    for key, path in paths.items():
        if file_sha(path) != protocol["source_hashes"][key]:
            raise ReplayError("source_changed", "An input changed after the study was frozen", input=key)
    races, _ = audit(paths["ledger"], paths["cohort"], paths["model_manifest"], spec=spec)
    _require_q(races)
    return races, protocol


def select_tickets(cars, q, points):
    """The selector receives no result, date, venue, payout, or market odds."""
    from nexus_model import optimizer
    plan = optimizer.select(list(cars), q, points)
    tickets = tuple(_ticket(t, cars) for t in plan["tickets"])
    if len(tickets) != points or tuple(sorted(tickets)) != expand_formations(plan["blocks"], cars):
        raise ReplayError("candidate_formation_mismatch", "Candidate must expand to the exact declared number of unique tickets")
    return tuple(sorted(tickets)), plan["blocks"]


def _evaluate(races, candidate, *, selector=select_tickets, progress=None):
    answer = []
    for i, race in enumerate(races, 1):
        tickets, blocks = selector(race.cars, race.q, candidate["points"])
        tickets = tuple(_ticket(t, race.cars) for t in tickets)
        if len(tickets) != candidate["points"] or tuple(sorted(tickets)) != expand_formations(blocks, race.cars):
            raise ReplayError("candidate_formation_mismatch", "Candidate ticket count or exact formation changed", race_id=race.race_id)
        q_map = dict(zip(itertools.permutations(race.cars, 3), race.q))
        answer.append(settle(race, tickets) | dict(candidate_id=candidate["id"],
                      selected_tickets=[ticket_text(t) for t in tickets], formations=blocks,
                      covered_probability=math.fsum(q_map[t] for t in tickets)))
        if progress is not None and (i % 100 == 0 or i == len(races)):
            progress(dict(candidate=candidate["id"], races_complete=i, total=len(races)))
    return answer


def _winner(reports):
    return min(reports, key=lambda r: (-Fraction(r["summary"]["payout_yen"], r["summary"]["stake_yen"]),
                                      r["candidate"]["points"],
                                      -Fraction(r["summary"]["hits"], r["summary"]["races"]), r["candidate"]["id"]))


def develop(run_dir, *, spec=ANNUAL, selector=select_tickets, progress=None):
    run_dir = Path(run_dir)
    if (run_dir / "development.json").exists() or (run_dir / "selection.json").exists():
        raise ReplayError("development_locked", "Development results are already frozen; no repeated candidate selection")
    races, protocol = _locked_inputs(run_dir, spec)
    development = [r for r in races if r.day <= spec.development_last]
    predictions, reports = [], []
    for candidate in candidate_specs():
        rows = _evaluate(development, candidate, selector=selector, progress=progress)
        predictions.extend(rows)
        reports.append(dict(candidate=candidate, summary=summarize(rows)))
    report = dict(schema=1, phase="development", protocol_sha256=protocol["protocol_sha256"],
                  baseline=summarize(settle(r, r.original) for r in development), candidates=reports,
                  chronological_candidate_results_computed=False)
    selected = _winner(reports)
    _write_once(run_dir / "development_predictions.jsonl", predictions, jsonl=True)
    _write_once(run_dir / "development.json", report)
    selection = dict(schema=1, protocol_sha256=protocol["protocol_sha256"],
                     selected_candidate=selected["candidate"],
                     development_sha256=file_sha(run_dir / "development.json"),
                     predictions_sha256=file_sha(run_dir / "development_predictions.jsonl"),
                     chronological_results_seen_in_this_protocol=False)
    selection["selection_sha256"] = fingerprint(selection)
    _write_once(run_dir / "selection.json", selection)
    return report | dict(selected_candidate=selected["candidate"])


def _read_predictions(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line, object_pairs_hook=_object_pairs) for line in stream]


def _comparison(baseline, candidate):
    baseline, candidate = list(baseline), list(candidate)
    old_ids, new_ids = [r["race_id"] for r in baseline], [r["race_id"] for r in candidate]
    if len(set(old_ids)) != len(old_ids) or len(set(new_ids)) != len(new_ids) or set(old_ids) != set(new_ids):
        raise ReplayError("comparison_cohort", "Comparison must retain the exact same unique race ids")
    a, b = summarize(baseline), summarize(candidate)
    if a["races"] != b["races"]:
        raise ReplayError("comparison_cohort", "Comparison must retain exactly the same races")
    return dict(baseline=a, candidate=b,
                hit_rate_delta_pp=100 * (b["hit_rate"] - a["hit_rate"]) if a["races"] else None,
                roi_delta_pp=100 * (b["roi"] - a["roi"]) if a["races"] else None)


def validate(run_dir, *, spec=ANNUAL, selector=select_tickets, progress=None):
    run_dir = Path(run_dir)
    races, protocol = _locked_inputs(run_dir, spec)
    selection = read_json(run_dir / "selection.json")
    if (selection.get("selection_sha256") != fingerprint({k: v for k, v in selection.items() if k != "selection_sha256"})
            or selection.get("protocol_sha256") != protocol["protocol_sha256"]
            or selection.get("development_sha256") != file_sha(run_dir / "development.json")
            or selection.get("predictions_sha256") != file_sha(run_dir / "development_predictions.jsonl")):
        raise ReplayError("selection_changed", "The development selection or its evidence changed")
    development = read_json(run_dir / "development.json")
    candidate = _winner(development["candidates"])["candidate"]
    if candidate != selection.get("selected_candidate") or candidate not in candidate_specs():
        raise ReplayError("selection_changed", "Validation must use the sole candidate selected by the frozen development rule")
    receipt = dict(protocol_sha256=protocol["protocol_sha256"], selection_sha256=selection["selection_sha256"])
    receipt_path, result_path = run_dir / "validation_started.json", run_dir / "validation.json"
    prediction_path = run_dir / "validation_predictions.jsonl"
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise ReplayError("validation_locked", "Cannot resume validation with a different frozen study or candidate")
    else:
        _write_once(receipt_path, receipt)
    if result_path.exists():
        saved_result = read_json(result_path)
        if (saved_result.get("report_sha256") != fingerprint({k: v for k, v in saved_result.items() if k != "report_sha256"})
                or saved_result.get("protocol_sha256") != protocol["protocol_sha256"]
                or saved_result.get("selection_sha256") != selection["selection_sha256"]
                or saved_result.get("selected_candidate") != candidate
                or saved_result.get("validation_predictions_sha256") != file_sha(prediction_path)):
            raise ReplayError("validation_result_changed", "Completed validation evidence is inconsistent; it cannot be replaced")
        return saved_result | dict(reused_completed_result=True)
    later = [r for r in races if r.day > spec.development_last]
    new_later = _evaluate(later, candidate, selector=selector, progress=progress)
    old_later = [settle(r, r.original) for r in later]
    new_dev = [r for r in _read_predictions(run_dir / "development_predictions.jsonl") if r["candidate_id"] == candidate["id"]]
    annual_new = new_dev + new_later
    if len(annual_new) != len(races) or {r["race_id"] for r in annual_new} != {r.race_id for r in races}:
        raise ReplayError("comparison_cohort", "Selected candidate does not contain every annual race exactly once")
    annual_old = [settle(r, r.original) for r in races]
    largest = min(annual_new, key=lambda r: (-r["payout_yen"], r["race_id"]))["race_id"]
    sensitivity = _comparison([r for r in annual_old if r["race_id"] != largest],
                              [r for r in annual_new if r["race_id"] != largest])
    annual, chronological = _comparison(annual_old, annual_new), _comparison(old_later, new_later)
    dev_best = _winner(development["candidates"])["summary"]
    gates = dict(development_improves=dev_best["roi"] > development["baseline"]["roi"],
                 chronological_improves=chronological["roi_delta_pp"] > 0,
                 annual_over_100_percent=annual["candidate"]["roi"] > 1,
                 chronological_over_100_percent=chronological["candidate"]["roi"] > 1,
                 largest_payout_removed_over_100_percent=(sensitivity["candidate"]["roi"] or 0) > 1)
    report = dict(schema=1, phase="chronological_validation", research_only=True,
                  protocol_sha256=protocol["protocol_sha256"], selection_sha256=selection["selection_sha256"],
                  selected_candidate=candidate, chronological=chronological, annual=annual,
                  sensitivity=dict(excluded_race_id=largest, rule="Remove the single largest candidate payout race from both policies without retuning", comparison=sensitivity),
                  gates=gates, meets_this_research_gate=all(gates.values()),
                  untouched_holdout=False, future_performance_proven=False,
                  live_policy_changed=False)
    if prediction_path.exists():
        if _read_predictions(prediction_path) != new_later:
            raise ReplayError("validation_predictions_changed", "Saved predictions differ from the resumed identical calculation")
    else:
        _write_once(prediction_path, new_later, jsonl=True)
    report["validation_predictions_sha256"] = file_sha(prediction_path)
    report["report_sha256"] = fingerprint(report)
    _write_once(result_path, report)
    return report


def input_contract():
    return dict(schema=1, protocol_id=PROTOCOL_ID, benchmark=asdict(ANNUAL),
                ledger_format="UTF-8 JSONL or JSONL.gz; one unique original annual race per line; no missing or extra rows",
                required_columns=["schema", "race_id", "date", "cars", "original_tickets", "original_formations",
                                  "model_id", "model_manifest_sha256", "training_cutoff_exclusive", "outcome"],
                q="Optional only for baseline audit; candidate research requires every probability in itertools.permutations(cars, 3) order for every original race. Never normalize a saved top-k list as full q.",
                original_tickets="Annual original 6/15/24 tickets before the later open_formation modification; not conditional12 or conditional8 tickets",
                original_formations="1 to 4 {first:[car],second:[car],third:[car],count:int} complete blocks. Buy their deduplicated union once; overlaps are allowed. count is optional and, if supplied, must equal that block's full expansion before deduplication across blocks.",
                outcome=dict(status="settled; fully refunded races do not belong to this established-race benchmark", payouts_per_100_yen="Map of every official winning triple to its actual integer yen payoff per 100 yen; includes dead heats, excludes refunds", refund_tickets="Explicit empty list: refund credits were not part of the original annual comparison", refund_all="Explicit false"),
                optional_audit_columns=["snapshot_at", "input_captured_at", "close_at", "feature_history_through", "source_url", "source_sha256", "input_sha256"],
                timing_note="snapshot_at, if supplied, means a genuinely saved pre-close forecast on the race date. input_captured_at must be <= snapshot_at < close_at. Retrospective source capture timestamps belong in separate provenance fields. Missing event-time proof remains unproven.",
                refund_accounting="The recovered original audit excludes the two named fully refunded races. For its 27,351 established races: stake = 100 yen times unique selected tickets; return = winning payouts only, without refund credits.",
                excluded_refund_race_ids=list(EXCLUDED_REFUND_RACE_IDS),
                hundredfold_definition=HUNDREDFOLD_DEFINITION,
                cohort_required=dict(schema=1, benchmark_id=ANNUAL.benchmark_id,
                                     model_manifest_sha256="SHA-256 of the exact supplied model manifest", source_description="Original study source for its established-race inventory and the two source-defined refund exclusions", source_sha256="SHA-256 of that original source", race_ids="All 27,351 original established race ids, once each, excluding 031120260425 and 098720260529"),
                example_kind="Schema instructions only. No annual forecasts, payouts, or profitable result are generated by this command.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    schema = sub.add_parser("schema", help="Print the exact minimum input contract; no results are generated")
    schema.add_argument("--output")
    for name in ("audit", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("--ledger", required=True)
        p.add_argument("--cohort", required=True)
        p.add_argument("--model-manifest", default=str(ROOT / "models" / "nexus53" / "manifest.json"))
        p.add_argument("--output-dir" if name == "prepare" else "--output", required=name == "prepare")
    for name in ("develop", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    progress = lambda row: print(json.dumps(dict(progress=row)), file=sys.stderr, flush=True)
    try:
        if args.command == "schema":
            result = input_contract()
        elif args.command == "audit":
            _, result = audit(args.ledger, args.cohort, args.model_manifest)
        elif args.command == "prepare":
            result = prepare(args.ledger, args.cohort, args.model_manifest, args.output_dir)
        elif args.command == "develop":
            result = develop(args.run_dir, progress=progress)
        else:
            result = validate(args.run_dir, progress=progress)
        output = getattr(args, "output", None)
        if output:
            _write_once(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except ReplayError as exc:
        print(json.dumps(exc.report(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except OSError as exc:
        print(json.dumps(ReplayError("file_unavailable", "Required input or output file is unavailable", path=exc.filename).report(), ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
