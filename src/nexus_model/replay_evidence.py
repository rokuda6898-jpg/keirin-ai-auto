"""Preserve complete model scores for later, explicitly dated ROI research.

The record is evidence from one inference, not a backtest or a betting policy.
No result, payout, market quote, or inferred probability enters this module.
"""
import copy
import hashlib
import itertools
import json
import math
import re
from datetime import date, datetime, timedelta


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("a complete SHA-256 is required")
    return value


def _stamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("evidence timestamps must include a timezone")
    return parsed


def verify(record):
    """Check integrity without repairing or normalizing any saved evidence."""
    payload = {k: v for k, v in record.items() if k != "evidence_sha256"}
    if record.get("evidence_sha256") != _digest(payload):
        raise ValueError("replay evidence hash mismatch")
    return True


def from_inference(cars, q, original_plan, manifest, manifest_sha256):
    """Keep exact q ordering and the formation before the live policy repair."""
    cars = list(cars)
    if (not 3 <= len(cars) <= 9 or len(set(cars)) != len(cars)
            or any(type(car) is not int or not 1 <= car <= 9 for car in cars)):
        raise ValueError("distinct active car numbers are required")
    triples = list(itertools.permutations(cars, 3))
    values = list(q)
    if (len(values) != len(triples)
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or abs(math.fsum(values) - 1.0) > 1e-8):
        raise ValueError("complete, finite, normalized joint scores are required")
    tickets = [tuple(t) for t in original_plan["tickets"]]
    blocks = copy.deepcopy(original_plan["blocks"])
    if (not 1 <= len(blocks) <= 4 or len(tickets) != len(set(tickets))
            or not tickets or not set(tickets).issubset(triples)):
        raise ValueError("invalid original formation tickets")
    expanded = set()
    for block in blocks:
        sets = [block[k] for k in ("first", "second", "third")]
        if any(not group or len(set(group)) != len(group)
               or any(type(car) is not int or car not in cars for car in group)
               for group in sets):
            raise ValueError("invalid original formation car numbers")
        part = {t for t in itertools.product(*sets) if len(set(t)) == 3}
        if "count" in block and (type(block["count"]) is not int or block["count"] != len(part)):
            raise ValueError("original formation count differs from its expansion")
        # Preserve source blocks exactly; their deduplicated union is bought once.
        expanded.update(part)
    if expanded != set(tickets):
        raise ValueError("original formations do not match original tickets")
    cutoff = (date.fromisoformat(manifest["train"][1]) + timedelta(days=1)).isoformat()
    record = dict(schema=1, cars=cars, q=values,
                  q_order="itertools.permutations(cars,3)",
                  original_tickets=["-".join(map(str, t)) for t in sorted(tickets)],
                  original_formations=blocks,
                  original_count=len(tickets),
                  model_id=manifest["id"],
                  model_manifest_sha256=_hash(manifest_sha256),
                  training_cutoff_exclusive=cutoff,
                  evidence_kind="inference_without_outcomes")
    record["evidence_sha256"] = _digest(record)
    return record


def bind_prediction(prediction, *, race_id, race_date, close_at):
    """Bind a new inference to its existing preclose publication timestamps.

    A legacy prediction without score evidence stays unchanged. This function
    must only be called at creation, never to reconstruct a closed forecast.
    The normal publisher deadline gate also applies to this nested evidence.
    """
    record = prediction.get("replay_evidence")
    if record is None:
        return False
    verify(record)
    if "race_id" in record:
        raise ValueError("replay evidence was already bound to a prediction")
    if (not isinstance(race_id, str) or not re.fullmatch(r"\d{12}", race_id)
            or not 1 <= int(race_id[:2]) <= 12
            or race_id[4:] != race_date.replace("-", "")):
        raise ValueError("explicit race identity is required")
    ds = date.fromisoformat(race_date)
    saved = _stamp(prediction["captured_at"])
    captured = _stamp(prediction["input_captured_at"])
    close = _stamp(close_at)
    if not captured <= saved < close:
        raise ValueError("replay evidence must be saved before close")
    if saved.astimezone(close.tzinfo).date() != ds or close.date() != ds:
        raise ValueError("replay evidence race date mismatch")
    through = date.fromisoformat(prediction["history_through"])
    if through >= ds:
        raise ValueError("feature history must end before the race day")
    bound = copy.deepcopy(record)
    bound.pop("evidence_sha256")
    bound.update(race_id=race_id, date=race_date,
                 snapshot_at=saved.isoformat(),
                 input_captured_at=captured.isoformat(),
                 close_at=close.isoformat(),
                 feature_history_through=through.isoformat(),
                 source_sha256=_hash(prediction["source_sha256"]),
                 evidence_kind="prospective_preclose_inference")
    if prediction.get("input_sha256"):
        bound["input_sha256"] = _hash(prediction["input_sha256"])
    bound["evidence_sha256"] = _digest(bound)
    prediction["replay_evidence"] = bound
    return True
