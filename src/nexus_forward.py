"""Freeze a ten-point NEXUS comparison before close, then settle official results.

The selector reads only previously saved joint model scores.  Historical research
is never imported and this module neither fetches source data nor places bets.
"""
import argparse
import copy
from collections import Counter
from contextlib import ExitStack
from datetime import date, datetime, timedelta
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

from nexus_input_snapshots import JST, ROOT, SNAPSHOTS, canonical, now_jst, read_json, save_json
from nexus_model.budget import allocate
from nexus_model.forecast_confidence import from_joint
from nexus_model.publication import allocation_snapshot, settle_prediction
from nexus_model.replay_evidence import verify
from nexus_roi_research import select_tickets
from nexus_source_replay import PublishedReplay, card_digest, digest as replay_digest, forecast_core, has_market_result, verify_record

FEED = Path("outputs/nexus_forward.json")
POLICY_ID = "nexus-forward-ten-v1"
POINTS = 10
MIN_LEAD_SECONDS = 60
SIDES = ("baseline", "candidate")
META = ("race_id", "date", "venue", "race_no", "session", "start_at", "close_at", "riders", "source_url")
FROZEN = META + ("saved_at", "baseline", "candidate", "source_prediction_sha256",
                 "source_evidence_sha256", "allocation_status", "allocation_reason")
ALLOCATION_KEYS = ("allocations", "auto_budget", "evaluated_budgets", "feasible_budgets",
                   "allocation_policy", "allocation_saved_at", "auto_saved_at", "odds_at",
                   "odds", "odds_source_sha256")


def _stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return result


def _now(clock):
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timezone-aware clock required")
    return value.astimezone(JST)


def _lead(value):
    if isinstance(value, bool):
        raise ValueError("finite publication lead required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("finite publication lead required")
    return max(MIN_LEAD_SECONDS, result)


def _digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("complete SHA-256 required")
    return value


def _index(values):
    if not isinstance(values, list):
        raise ValueError("race list required")
    result = {}
    for row in values:
        if not isinstance(row, dict) or not isinstance(row.get("race_id"), str):
            raise ValueError("race identity required")
        if row["race_id"] in result:
            raise ValueError("duplicate race identity")
        result[row["race_id"]] = row
    return result


def _metadata(row):
    rid, ds = row.get("race_id"), row.get("date")
    day = date.fromisoformat(ds)
    if (not isinstance(rid, str) or not re.fullmatch(r"\d{12}", rid)
            or not 1 <= int(rid[:2]) <= 12 or rid[4:] != ds.replace("-", "")
            or type(row.get("race_no")) is not int or row["race_no"] != int(rid[:2])):
        raise ValueError("race/date/number mismatch")
    close, start = _stamp(row["close_at"]), _stamp(row["start_at"])
    if close > start or close.astimezone(JST).date() != day or start.astimezone(JST).date() != day:
        raise ValueError("race schedule mismatch")
    return {key: copy.deepcopy(row.get(key)) for key in META}


def _tickets(prediction, cars, expected=None):
    values = prediction.get("tickets")
    if (not isinstance(values, list) or not values or len(values) != len(set(values))
            or type(prediction.get("count")) is not int or prediction["count"] != len(values)
            or expected is not None and len(values) != expected):
        raise ValueError("ticket count mismatch")
    for ticket in values:
        if not isinstance(ticket, str) or not re.fullmatch(r"[1-9]-[1-9]-[1-9]", ticket):
            raise ValueError("invalid ticket")
        triple = tuple(map(int, ticket.split("-")))
        if len(set(triple)) != 3 or not set(triple).issubset(cars):
            raise ValueError("ticket car mismatch")
    blocks, expanded = prediction.get("blocks"), set()
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 4:
        raise ValueError("one to four complete formations required")
    for block in blocks:
        groups = [block[key] for key in ("first", "second", "third")]
        if any(not isinstance(group, list) or not group or len(group) != len(set(group))
               or any(type(car) is not int or car not in cars for car in group) for group in groups):
            raise ValueError("invalid formation")
        part = {"-".join(map(str, t)) for t in itertools.product(*groups) if len(set(t)) == 3}
        if "count" in block and (type(block["count"]) is not int or block["count"] != len(part)):
            raise ValueError("formation count mismatch")
        expanded.update(part)
    if set(values) != expanded:
        raise ValueError("formation expansion differs from tickets")
    return values


def _source_signature(prediction):
    keys = ("model_id", "captured_at", "input_captured_at", "input_sha256", "source_sha256",
            "history_through", "tickets", "blocks", "count")
    identity = {key: prediction.get(key) for key in keys}
    identity["evidence_sha256"] = (prediction.get("replay_evidence") or {}).get("evidence_sha256")
    return _digest(identity)


def _frozen_signature(row):
    value = {key: row.get(key) for key in FROZEN}
    if "source_replay" in row:
        value["source_replay"] = row["source_replay"]
    return _digest(value)


def _source(row, manifest, manifest_hash, current):
    prediction = row.get("prediction")
    if not isinstance(prediction, dict) or not isinstance(prediction.get("replay_evidence"), dict):
        raise ValueError("締切前の全着順評価を待っています")
    evidence = prediction["replay_evidence"]
    verify(evidence)
    cars, q = evidence.get("cars"), evidence.get("q")
    if (not isinstance(cars, list) or not 4 <= len(cars) <= 9 or len(cars) != len(set(cars))
            or any(type(car) is not int or not 1 <= car <= 9 for car in cars)
            or not isinstance(q, list) or len(q) != math.perm(len(cars), 3)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in q)
            or abs(math.fsum(q) - 1) > 1e-8):
        raise ValueError("invalid complete joint scores")
    active = row.get("riders")
    if (not isinstance(active, list) or len(active) != len(cars)
            or {r.get("car") for r in active} != set(cars)):
        raise ValueError("active cars differ from model evidence")
    close = _stamp(row["close_at"])
    captured, inputs_at = _stamp(prediction["captured_at"]), _stamp(prediction["input_captured_at"])
    if not inputs_at <= captured <= current < close:
        raise ValueError("invalid source prediction timestamps")
    if captured.astimezone(close.tzinfo).date().isoformat() != row["date"]:
        raise ValueError("source prediction is for another day")
    through = date.fromisoformat(prediction["history_through"])
    if through >= date.fromisoformat(row["date"]):
        raise ValueError("feature history reaches the race day")
    required = dict(race_id=row["race_id"], date=row["date"], model_id=manifest["id"],
                    model_manifest_sha256=manifest_hash, evidence_kind="prospective_preclose_inference",
                    q_order="itertools.permutations(cars,3)",
                    feature_history_through=through.isoformat(),
                    source_sha256=_hash(prediction["source_sha256"]))
    if prediction.get("input_sha256"):
        required["input_sha256"] = _hash(prediction["input_sha256"])
    elif evidence.get("input_sha256"):
        raise ValueError("source input hash missing")
    if any(evidence.get(key) != value for key, value in required.items()):
        raise ValueError("bound model evidence identity mismatch")
    if (prediction.get("model_id") != manifest["id"]
            or _stamp(evidence["snapshot_at"]) != captured
            or _stamp(evidence["input_captured_at"]) != inputs_at
            or _stamp(evidence["close_at"]) != close):
        raise ValueError("bound prediction timing/model mismatch")
    cutoff = (date.fromisoformat(manifest["train"][1]) + timedelta(days=1)).isoformat()
    if evidence.get("training_cutoff_exclusive") != cutoff:
        raise ValueError("training cutoff mismatch")
    _tickets(prediction, cars)
    if prediction.get("first_probabilities"):
        marginal = dict.fromkeys(cars, 0.0)
        for triple, probability in zip(itertools.permutations(cars, 3), q):
            marginal[triple[0]] += probability
        if any(abs(marginal[car] - prediction["first_probabilities"].get(str(car),
                       prediction["first_probabilities"].get(car, -1))) > 1e-10 for car in cars):
            raise ValueError("model score ordering differs from source marginals")
    return prediction, evidence


def _quotes(root, row, prediction, cars, current):
    snap = read_json(root / SNAPSHOTS / row["date"] / "preclose" / (row["race_id"] + ".json"))
    if not isinstance(snap, dict):
        raise ValueError("同じ時点の締切前オッズを確認できません")
    meta, data = snap.get("meta", {}), snap.get("data", {})
    race = data.get("race", {})
    close = _stamp(row["close_at"])
    quote_at = _stamp(snap["captured_at"])
    if (current - quote_at).total_seconds() > 1800:
        raise ValueError("オッズ取得から30分を超えているため、配分は未保存です")
    if (snap.get("schema") != 1 or snap.get("source_kind") != "existing_public_page_response"
            or meta.get("race_id") != row["race_id"] or meta.get("date") != row["date"]
            or str(race.get("id")) != row["race_id"]
            or race.get("number") != row["race_no"]
            or str(data.get("schedule", {}).get("date")) != row["date"].replace("-", "")
            or race.get("cancel") or race.get("decidedAt") or data.get("finalOdds")
            or data.get("results") or has_market_result(data)
            or snap.get("input_sha256") != prediction.get("input_sha256")
            or not quote_at <= current < close or not 0 <= (current - quote_at).total_seconds() <= 1800
            or quote_at < _stamp(prediction["input_captured_at"])
            or float(race["closeAt"]) != close.timestamp()
            or float(meta["close_at"]) != close.timestamp()):
        raise ValueError("同じ入力・締切前オッズの一致を確認できません")
    active = [entry["number"] for entry in data.get("entries", []) if entry.get("absent") is False]
    if len(active) != len(cars) or set(active) != set(cars):
        raise ValueError("オッズ取得時の出走選手が変わっています")
    quotes = {}
    for value in data.get("trifecta", []):
        key = value.get("key", [])
        if value.get("unitPrice") != 100 or value.get("absent") is True:
            continue
        if (str(value.get("raceId")) != row["race_id"] or len(key) != 3 or len(set(key)) != 3
                or any(type(car) is not int or car not in cars for car in key)):
            raise ValueError("オッズのレース・車番を確認できません")
        price = value.get("odds")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            raise ValueError("オッズの値を確認できません")
        ticket = "-".join(map(str, key))
        if ticket in quotes:
            raise ValueError("duplicate quote")
        quotes[ticket] = price
    return quotes, quote_at.isoformat(), _hash(snap["sha256"])


def _new_forecast(root, source_row, manifest, manifest_hash, clock, min_lead, replayer=None):
    current, row = _now(clock), _metadata(source_row)
    close = _stamp(row["close_at"])
    if (current + timedelta(seconds=min_lead) >= close or source_row.get("closed")
            or _known_outcome(source_row) or row["date"] != current.astimezone(close.tzinfo).date().isoformat()):
        raise ValueError("締切前に保存できる未確定レースが必要です")
    reconstructed = None
    if replayer is not None and isinstance(source_row.get("prediction"), dict) and "replay_evidence" not in source_row["prediction"]:
        reconstructed = replayer.replay(source_row, manifest, manifest_hash, clock, min_lead)
        original, evidence = source_row["prediction"], reconstructed
        _tickets(original, evidence["cars"])
    else:
        original, evidence = _source(source_row, manifest, manifest_hash, current)
    triples, blocks = select_tickets(evidence["cars"], evidence["q"], POINTS)
    tickets = ["-".join(map(str, triple)) for triple in triples]
    candidate = dict(model_id=manifest["id"], policy_id=POLICY_ID, count=POINTS, tickets=tickets,
                     blocks=copy.deepcopy(blocks), marks=copy.deepcopy(original.get("marks", {})),
                     category=original.get("category"), captured_at=current.isoformat(),
                     input_captured_at=original["input_captured_at"],
                     source_sha256=original["source_sha256"], input_sha256=original.get("input_sha256"),
                     history_through=original["history_through"],
                     confidence=from_joint(evidence["cars"], evidence["q"], tickets, annotated_at=current.isoformat()),
                     allocations={}, odds={}, auto_budget=None)
    _tickets(candidate, evidence["cars"], POINTS)
    baseline = {key: copy.deepcopy(value) for key, value in original.items()
                if key not in ALLOCATION_KEYS and key not in ("replay_evidence", "q")}
    baseline.update(allocations={}, odds={}, auto_budget=None)
    allocation_status, allocation_reason = "missing_quotes", "共通の締切前オッズを確認できないため配分は未保存です"
    try:
        quotes, odds_at, source_hash = _quotes(root, row, original, evidence["cars"], _now(clock))
        if not set(baseline["tickets"] + candidate["tickets"]).issubset(quotes):
            raise ValueError("両方式すべての買い目のオッズが必要です")
        prepared = [allocation_snapshot(plan, quotes, odds_at, current.isoformat(), row["close_at"])
                    for plan in (baseline, candidate)]
        common = sorted(set(prepared[0]["feasible_budgets"]) & set(prepared[1]["feasible_budgets"]))
        chosen = common[0] if common else None
        for plan, values in zip((baseline, candidate), prepared):
            values["auto_budget"] = chosen
            if chosen is not None and str(chosen) not in values["allocations"]:
                values["allocations"][str(chosen)] = allocate(plan["tickets"], quotes, chosen)
            values["odds_source_sha256"] = source_hash
            plan.update(values)
        allocation_status = "saved" if chosen is not None else "no_common_break_even"
        allocation_reason = ("両方式に同じ締切前オッズ・同じ予算を使う比較用配分です" if chosen is not None else
                             "両方式ともトリガミを避けられる共通予算が5,000〜10,000円にありません")
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        allocation_reason = str(exc)[:240]
    finished = _now(clock)
    if finished + timedelta(seconds=min_lead) >= close:
        raise ValueError("計算中に締切が近づいたため新規予想を保存しません")
    candidate["captured_at"] = finished.isoformat()
    candidate["confidence"]["annotated_at"] = finished.isoformat()
    for plan in (baseline, candidate):
        plan["comparison_saved_at"] = finished.isoformat()
        if plan.get("allocation_saved_at"):
            plan["allocation_saved_at"] = plan["auto_saved_at"] = finished.isoformat()
    row.update(saved_at=finished.isoformat(), baseline=baseline, candidate=candidate,
               source_prediction_sha256=_source_signature(original),
               source_evidence_sha256=evidence["evidence_sha256"],
               allocation_status=allocation_status, allocation_reason=allocation_reason)
    if reconstructed is not None:
        row["source_replay"] = reconstructed
    row["forecast_sha256"] = _frozen_signature(row)
    return row


def _known_outcome(row):
    return bool(row.get("result_complete") or row.get("payouts") or row.get("winners")
                or row.get("refund_all") or row.get("refunded") or row.get("cancelled"))


def _validate_saved(row, activated):
    _metadata(row)
    if not row.get("candidate") and not row.get("baseline"):
        return
    if not row.get("candidate") or not row.get("baseline"):
        raise ValueError("both frozen forecasts are required")
    saved, close = _stamp(row["saved_at"]), _stamp(row["close_at"])
    if not activated <= saved < close or row["candidate"].get("policy_id") != POLICY_ID:
        raise ValueError("invalid prospective forecast timestamp/policy")
    if _stamp(row["candidate"]["captured_at"]) != saved:
        raise ValueError("candidate save timestamp mismatch")
    cars = [r["car"] for r in row["riders"]]
    if len(cars) != len(set(cars)):
        raise ValueError("duplicate frozen rider")
    for side in SIDES:
        plan = row[side]
        _tickets(plan, cars, POINTS if side == "candidate" else None)
        if not _stamp(plan["input_captured_at"]) <= _stamp(plan["captured_at"]) <= saved:
            raise ValueError("frozen source time mismatch")
        if plan.get("allocation_saved_at") and not _stamp(plan["odds_at"]) <= _stamp(plan["allocation_saved_at"]) <= saved:
            raise ValueError("frozen allocation time mismatch")
    if row.get("forecast_sha256") != _frozen_signature(row):
        raise ValueError("frozen forecast integrity mismatch")
    _hash(row["source_prediction_sha256"])
    _hash(row["source_evidence_sha256"])
    if "source_replay" in row:
        verify_record(row["source_replay"], row)


def _validate_payload(payload, manifest_hash=None):
    if not isinstance(payload, dict) or type(payload.get("schema")) is not int or payload["schema"] != 1:
        raise ValueError("unexpected forward comparison schema")
    policy = payload.get("policy", {})
    if (policy.get("id") != POLICY_ID or policy.get("points") != POINTS
            or policy.get("max_formations") != 4 or policy.get("unit_yen") != 100
            or manifest_hash is not None and policy.get("model_manifest_sha256") != manifest_hash):
        raise ValueError("unexpected frozen forward policy/model")
    activated = _stamp(policy["activated_at"])
    for row in _index(payload.get("races")).values():
        _validate_saved(row, activated)


def _whole_payout(value):
    if isinstance(value, bool):
        raise ValueError("official payout must be whole yen")
    number = Fraction(str(value))
    if number.denominator != 1 or number <= 0:
        raise ValueError("official payout must be positive whole yen")
    return number.numerator


def _result_receipt(source_row):
    """Keep a verified page result's SHA and observation time together.

    The result synchronizer preserves the first receipt and records its strict
    same-race page verification in result_sync_* fields. Never combine one
    receipt's hash with the other receipt's timestamp or infer a missing hash.
    """
    if source_row.get("result_sync_source") == "nexus_page_snapshot":
        return dict(captured_at=source_row.get("result_sync_captured_at"),
                    source_sha256=source_row.get("result_sync_source_sha256"))
    return dict(captured_at=source_row.get("result_captured_at"),
                source_sha256=source_row.get("result_source_sha256"))


def _update_result(row, source_row, current):
    prior_result = row.get("result") or {}
    if row.get("candidate") and row.get("status") in ("settled", "excluded"):
        # A partial later feed must not erase already received official results.
        if source_row is None or not _known_outcome(source_row):
            return
        old_at, new_at = prior_result.get("captured_at"), _result_receipt(source_row)["captured_at"]
        if old_at and new_at:
            try:
                if _stamp(new_at) < _stamp(old_at):
                    return
            except (ValueError, TypeError):
                return
    row["settlement"] = {}
    row["result"] = {}
    if not row.get("candidate"):
        if source_row is not None:
            row["result"] = {key: copy.deepcopy(source_row.get(key)) for key in
                             ("winners", "payouts", "refunded", "refund_all", "cancelled", "result_complete")}
            row["result"].update(_result_receipt(source_row))
        return
    close = _stamp(row["close_at"])
    row.update(status="saved" if current < close else "result_pending", reason_code="", reason="")
    if source_row is None:
        return
    try:
        meta = _metadata(source_row)
        if any(meta[key] != row[key] for key in ("race_id", "date", "race_no", "start_at", "close_at")):
            raise ValueError("official result schedule/identity changed")
        raw = dict(winners=copy.deepcopy(source_row.get("winners", [])),
                   payouts=copy.deepcopy(source_row.get("payouts", {})),
                   refunded=copy.deepcopy(source_row.get("refunded", [])),
                   refund_all=bool(source_row.get("refund_all")), cancelled=bool(source_row.get("cancelled")),
                    result_complete=bool(source_row.get("result_complete")),
                    **_result_receipt(source_row))
        row["result"] = raw
        if raw["refunded"] or raw["refund_all"] or raw["cancelled"]:
            row.update(status="excluded", reason_code="refund_or_cancellation", reason="取消・返還のあるレースは両方式の比較成績から除外します")
            return
        if not raw["result_complete"]:
            return
        if current < close or not _stamp(raw["captured_at"]) >= close or _stamp(raw["captured_at"]) > current:
            raise ValueError("official result timestamp is not observable after close")
        _hash(raw["source_sha256"])
        if (not isinstance(raw["payouts"], dict) or not raw["payouts"]
                or not isinstance(raw["winners"], list) or len(raw["winners"]) != len(set(raw["winners"]))
                or set(raw["winners"]) != set(raw["payouts"])):
            raise ValueError("official winners and payouts differ")
        payouts = {ticket: _whole_payout(value) for ticket, value in raw["payouts"].items()}
        cars = {r["car"] for r in row["riders"]}
        for ticket in payouts:
            if (not isinstance(ticket, str) or not re.fullmatch(r"[1-9]-[1-9]-[1-9]", ticket)
                    or len(set(ticket.split("-"))) != 3 or not set(map(int, ticket.split("-"))).issubset(cars)):
                raise ValueError("official winning ticket has invalid car identity")
        row["settlement"] = {side: settle_prediction(row[side], payouts) for side in SIDES}
        row.update(status="settled", reason_code="", reason="")
    except (ValueError, KeyError, TypeError, OverflowError, ZeroDivisionError) as exc:
        row.update(status="invalid_result", reason_code="official_evidence_invalid", reason=str(exc)[:240], settlement={})


def _metrics(rows, side, key):
    values = [row["settlement"][side][key] for row in rows]
    stake, returned = sum(v["stake"] for v in values), sum(v["returned"] for v in values)
    return dict(races=len(rows), hits=sum(v["hit"] for v in values),
                hit_rate=sum(v["hit"] for v in values) / len(rows) if rows else None,
                stake_yen=stake, payout_yen=returned, profit_yen=returned - stake,
                roi=returned / stake if stake else None)


def _summarize(payload):
    settled = [row for row in payload["races"] if row.get("status") == "settled"
               and all(row.get("settlement", {}).get(side, {}).get("flat") for side in SIDES)]
    allocated = [row for row in settled if all(row["settlement"][side].get("auto") for side in SIDES)
                 and row["baseline"].get("auto_budget") == row["candidate"].get("auto_budget")]
    payload["summary"] = {}
    for name, rows, key in (("flat", settled, "flat"), ("allocated", allocated, "auto")):
        payload["summary"][name] = dict(race_ids=[r["race_id"] for r in rows], races=len(rows),
                                         **{side: _metrics(rows, side, key) for side in SIDES})
    statuses = Counter(row.get("status", "unknown") for row in payload["races"])
    payload["coverage"] = dict(races=len(payload["races"]),
                                saved=sum(bool(r.get("candidate")) for r in payload["races"]),
                                settled=len(settled), allocated_common=len(allocated), statuses=dict(statuses))


def _write(root, payload, current):
    payload["updated_at"] = current.isoformat()
    payload["races"] = sorted(payload["races"], key=lambda r: (r["date"], r["race_id"]))
    _summarize(payload)
    save_json(root / FEED, payload)


def run(root=ROOT, clock=now_jst, min_lead_seconds=MIN_LEAD_SECONDS, settle_only=False, replay_published=False):
    root, current = Path(root), _now(clock)
    min_lead = _lead(min_lead_seconds)
    baseline = read_json(root / FEED)
    if settle_only and baseline is None:
        return None
    manifest_bytes = (root / "models/nexus53/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if baseline is not None:
        _validate_payload(baseline, manifest_hash)
    live = read_json(root / "outputs/nexus_live.json", {})
    sources = _index(live.get("races", []))
    payload = copy.deepcopy(baseline) if baseline is not None else dict(schema=1, policy=dict(
        id=POLICY_ID, name="NEXUS 10点・導入後の比較", activated_at=current.isoformat(),
        points=POINTS, max_formations=4, unit_yen=100, minimum_lead_seconds=MIN_LEAD_SECONDS,
        model_id=manifest["id"], model_manifest_sha256=manifest_hash,
        selection="same_joint_scores_exact_complete_formations_10_max4",
        allocation="smallest_common_feasible_budget_same_preclose_quotes_once",
        mode="simulation", backfill=False, annual_target_verified=False), races=[])
    rows = _index(payload["races"])
    with ExitStack() as cleanup:
        replayer = PublishedReplay(root) if replay_published and not settle_only else None
        if replayer is not None:
            cleanup.callback(replayer.close)
        for rid, source_row in sources.items():
            if settle_only:
                if rid in rows:
                    _update_result(rows[rid], source_row, _now(clock))
                continue
            # Current schedules are complete even while score evidence is missing.
            # Prior dates appear only when this policy genuinely saved a forecast.
            if source_row.get("date") != current.date().isoformat() and rid not in rows:
                continue
            row = rows.get(rid)
            if row is None or not row.get("candidate"):
                try:
                    row = _metadata(source_row)
                except (ValueError, KeyError, TypeError, OverflowError):
                    # Invalid metadata cannot identify a user-facing race reliably.
                    continue
                now, close = _now(clock), _stamp(row["close_at"])
                closed = now + timedelta(seconds=min_lead) >= close or bool(source_row.get("closed")) or _known_outcome(source_row)
                row.update(status="closed_without_record" if closed else "missing_preclose_evidence",
                           reason_code="closed_without_preclose_record" if closed else "missing_score_evidence",
                           reason="締切前の10点保存がないため比較成績に含めません" if closed else "締切前の全着順評価を待っています",
                           result={}, settlement={})
                if not closed:
                    try:
                        row = _new_forecast(root, source_row, manifest, manifest_hash, clock, min_lead, replayer)
                    except (ValueError, KeyError, TypeError, OverflowError, OSError, ImportError) as exc:
                        row["reason"] = str(exc)[:240]
                rows[rid] = row
            _update_result(row, source_row, _now(clock))
    for rid, row in rows.items():
        if rid not in sources:
            _update_result(row, None, _now(clock))
    payload["date"] = current.date().isoformat()
    payload["races"] = list(rows.values())
    _write(root, payload, _now(clock))
    if not settle_only:
        gate_publication(root, baseline or {}, clock=clock, lead_seconds=min_lead)
    return read_json(root / FEED)


def gate_publication(root, baseline, clock=now_jst, lead_seconds=MIN_LEAD_SECONDS):
    """Revert private new plans after delay or canonical forecast rollback."""
    root, current = Path(root), _now(clock)
    lead_seconds = _lead(lead_seconds)
    payload = read_json(root / FEED)
    if payload is None:
        return []
    _validate_payload(payload)
    if baseline:
        _validate_payload(baseline)
        if payload["policy"] != baseline["policy"]:
            raise ValueError("forward policy cannot change during publication")
    before = _index((baseline or {}).get("races", []))
    sources = _index(read_json(root / "outputs/nexus_live.json", {}).get("races", []))
    dropped, replacements = [], []
    for row in payload["races"]:
        old = before.get(row["race_id"], {})
        changed = bool(row.get("candidate")) and row.get("forecast_sha256") != old.get("forecast_sha256")
        if not changed:
            replacements.append(row)
            continue
        source = sources.get(row["race_id"], {})
        candidates = [source.get("prediction")] + list(source.get("previous_predictions", []))
        source_exists = any(isinstance(p, dict) and _source_signature(p) == row["source_prediction_sha256"] for p in candidates)
        if "source_replay" in row:
            source_exists = any(isinstance(p, dict) and _source_signature(p) == row["source_prediction_sha256"]
                and replay_digest(forecast_core(p)) == row["source_replay"]["source_baseline_sha256"] for p in candidates)
            source_exists = source_exists and card_digest(source) == row["source_replay"]["source_card_sha256"]
        try:
            meta = _metadata(source)
            source_unchanged = (all(meta[key] == row[key] for key in ("race_id", "date", "race_no", "start_at", "close_at"))
                                and meta["riders"] == row["riders"]
                                and len(meta["riders"]) == len(row["riders"])
                                and not source.get("closed") and not _known_outcome(source))
        except (ValueError, KeyError, TypeError, OverflowError):
            source_unchanged = False
        too_late = current + timedelta(seconds=lead_seconds) >= _stamp(row["close_at"])
        if old.get("candidate") or too_late or not source_exists or not source_unchanged:
            if old.get("candidate"):
                replacement = copy.deepcopy(old)
                _update_result(replacement, source or None, current)
            else:
                replacement = {key: copy.deepcopy(row.get(key)) for key in META}
                replacement.update(status="not_published_before_close", reason_code="publication_gate",
                                   reason="締切前の公開または元予想の公開を確認できず新規10点を破棄しました", result={}, settlement={})
            replacements.append(replacement)
            dropped.append(row["race_id"])
        else:
            replacements.append(row)
    if dropped:
        payload["races"] = replacements
        _write(root, payload, current)
    return dropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--min-lead-seconds", type=float, default=MIN_LEAD_SECONDS)
    parser.add_argument("--settle-only", action="store_true")
    parser.add_argument("--replay-published", action="store_true", help="Require exact original Git provenance for legacy score reconstruction")
    args = parser.parse_args()
    report = run(args.root, min_lead_seconds=args.min_lead_seconds, settle_only=args.settle_only, replay_published=args.replay_published)
    print(json.dumps(report["coverage"] if report else {"status": "not_started"}, ensure_ascii=False))
