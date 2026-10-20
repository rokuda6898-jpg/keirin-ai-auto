"""Publish the saved NEXUS model from existing collector snapshots, without HTTP.

The --publish mode is for the existing GitHub collector workflows. It uses an
isolated latest-main worktree so slow legacy jobs cannot overwrite newer picks.
"""
import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from nexus_input_snapshots import (ROOT, SNAPSHOTS, JST, canonical, now_jst,
                                   read_json, save_json, timestamp)
from nexus_model import inputs, source, update_history
from nexus_model.forecast_confidence import annotate_prediction
from nexus_model.publication import allocation_snapshot, is_open_prediction
from nexus_model.replay_evidence import bind_prediction


def csv_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def saved_schedule(root, day):
    saved = read_json(root / SNAPSHOTS / day / "schedule.json", {})
    rows = [r for r in saved.get("races", []) if r.get("date") == day]
    current = [r for r in csv_rows(root / "outputs/latest_race_schedule.csv") if r.get("date") == day]
    current_at = max((r.get("schedule_fetched_at_jst", "") for r in current), default="")
    if current and (not rows or _time_key(current_at) > _time_key(saved.get("captured_at"))):
        rows = current
    ids = [str(r.get("race_id", "")) for r in rows]
    if rows and (any(not rid.isdigit() for rid in ids) or len(ids) != len(set(ids))):
        raise ValueError("duplicate or invalid complete schedule")
    byid = {str(r["race_id"]): r for r in rows}
    authoritative = bool(rows)
    # Market verification can fail after a complete original response has
    # already been captured. Build its schedule directly from that response,
    # while proving full coverage against the collector's original index.
    index = read_json(root / SNAPSHOTS / day / "index.json", {})
    expected = set(index.get("urls", [])) if index.get("date") == day else set()
    for kind in ("preclose", "outcomes"):
        for path in sorted((root / SNAPSHOTS / day / kind).glob("*.json")):
            try:
                snap = read_json(path)
                meta = snap.get("meta", {})
                rid = str(meta.get("race_id", ""))
                if (snap.get("source_kind") != "existing_public_page_response" or path.stem != rid
                        or not rid.isdigit() or meta.get("date") != day):
                    continue
                if rid not in byid:
                    byid[rid] = dict(race_id=rid, date=day, venue=meta["venue"],
                                     race_no=meta["race_number"], start_at=meta["start_at"],
                                     close_at=meta["close_at"], source_url=meta["source_url"])
                    authoritative = False
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                # The input loop reports invalid pre-close files individually.
                continue
    for row in byid.values():
        for field in ("start_at", "close_at"):
            if not math.isfinite(float(row[field])):
                raise ValueError("invalid scheduled time")
    if expected:
        authoritative = expected.issubset({r.get("source_url") for r in byid.values()})
    return byid, authoritative


def _time_key(value):
    try:
        return timestamp(value).timestamp()
    except (ValueError, TypeError):
        return float("-inf")


def _prediction_key(row):
    prediction = row.get("prediction") or {}
    capture = _time_key(prediction.get("captured_at"))
    close = _time_key(row.get("close_at"))
    if capture >= close or capture == float("-inf"):
        return (float("-inf"),) * 4
    return (capture, _time_key(prediction.get("input_captured_at")),
            _time_key(prediction.get("allocation_saved_at")),
            _time_key(prediction.get("confidence", {}).get("annotated_at")))


def merge_saved_row(old, incoming):
    """Choose forecast and result evidence independently after interrupted writes."""
    newest = incoming if _prediction_key(incoming) > _prediction_key(old) else old
    merged = copy.deepcopy(newest)
    complete = lambda r: bool(r.get("result_complete") or r.get("refund_all") or r.get("cancelled"))
    result_key = lambda r: (_time_key(r.get("results_updated_at")), _time_key(r.get("result_captured_at")))
    if complete(old) != complete(incoming):
        result_row = old if complete(old) else incoming
    else:
        result_row = incoming if result_key(incoming) > result_key(old) else old
    result_fields = lambda key: (key.startswith("result") or key in (
        "winners", "payouts", "refunded", "refund_all", "cancelled", "settlement",
        "settlement_unavailable_reason", "payout_pending"))
    for key in list(merged):
        if result_fields(key):
            merged.pop(key)
    for key, value in result_row.items():
        if result_fields(key):
            merged[key] = copy.deepcopy(value)
    if complete(result_row):
        merged["closed"] = True
    previous = []
    for record in (old, incoming):
        for prediction in record.get("previous_predictions", []):
            if prediction != merged.get("prediction") and prediction not in previous:
                previous.append(copy.deepcopy(prediction))
        prediction = record.get("prediction")
        if (prediction and prediction != merged.get("prediction")
                and _prediction_key(record)[0] != float("-inf") and prediction not in previous):
            previous.append(copy.deepcopy(prediction))
    if previous:
        merged["previous_predictions"] = previous
    if merged.get("prediction") != result_row.get("prediction"):
        merged.pop("settlement", None)
        if merged.get("prediction") and complete(result_row):
            from nexus_results_sync import _settle_saved
            _settle_saved(merged)
    return merged


def load_saved_rows(root, live):
    existing = {}
    for path in sorted((root / "outputs/nexus_forecasts").glob("*.json")):
        for row in read_json(path, []):
            existing[row["race_id"]] = row
    for row in live.get("races", []):
        old = existing.get(row["race_id"])
        if old is None:
            existing[row["race_id"]] = row
        else:
            existing[row["race_id"]] = merge_saved_row(old, row)
    return copy.deepcopy(existing)


def advance_saved_history(root, history, today):
    """Append only complete prior days. A missing result never blocks today's run."""
    errors = []
    cursor = date.fromisoformat(history.data["through"]) + timedelta(days=1)
    while cursor < date.fromisoformat(today):
        ds = cursor.isoformat()
        try:
            schedule, authoritative = saved_schedule(root, ds)
            if not authoritative:
                raise ValueError("complete saved schedule missing")
            pending = []
            for rid in schedule:
                snap = read_json(root / SNAPSHOTS / ds / "outcomes" / (rid + ".json"))
                if not snap or snap["meta"]["race_id"] != rid or snap["meta"]["date"] != ds:
                    raise ValueError("saved individual outcome missing: " + rid)
                _, _, void, complete = source.outcomes(snap["data"])
                if not complete and not void:
                    raise ValueError("complete original outcome missing: " + rid)
                pending.append(update_history.event(history, snap["data"], snap["meta"]))
            candidate = inputs.History(copy.deepcopy(history.data))
            for event in pending:
                candidate.add(event)
            candidate.data["through"] = ds
            candidate.save(root / "data/nexus/history.json.gz")
            history.data = candidate.data
        except Exception as exc:
            errors.append(dict(date=ds, reason=f"{type(exc).__name__}: {exc}"))
            break
        cursor += timedelta(days=1)
    required = (date.fromisoformat(today) - timedelta(days=1)).isoformat()
    return dict(through=history.data["through"], required_through=required,
                lag_days=max(0, (date.fromisoformat(required) - date.fromisoformat(history.data["through"])).days),
                errors=errors, source="saved_prior_day_outcomes")


def session_for(row, schedule):
    # The meeting's first race determines its session for the whole day.
    url = row.get("source_url", "")
    meeting = url.rsplit("/", 2)[0]
    times = [float(r["start_at"]) for r in schedule.values()
             if r.get("source_url", "").rsplit("/", 2)[0] == meeting]
    if not times:
        return row.get("session")
    first = datetime.fromtimestamp(min(times), JST)
    hour = first.hour + first.minute / 60
    return "midnight" if hour >= 20 else "night" if hour >= 14 else "morning" if hour < 10.5 else "day"


def _iso(epoch):
    return datetime.fromtimestamp(float(epoch), JST).isoformat()


def discard_late_changes(rows, previously_published, clock, lead_seconds=0):
    """A private checkpoint does not make a newly closed forecast public evidence."""
    removed = []
    for rid, row in rows.items():
        old = previously_published.get(rid, {})
        def forecast_fields(prediction):
            return {key: value for key, value in (prediction or {}).items() if key != "confidence"}
        if forecast_fields(row.get("prediction")) == forecast_fields(old.get("prediction")) or not row.get("prediction"):
            continue
        close = timestamp(row["close_at"]).timestamp()
        if clock().timestamp() + lead_seconds < close:
            continue
        active = {r["car"] for r in row.get("riders", [])}
        prior = old.get("prediction")
        withdrawn = bool(prior and active and any(int(car) not in active
                         for ticket in prior.get("tickets", []) for car in ticket.split("-")))
        for key in ("prediction", "previous_predictions", "lines"):
            if key in old:
                row[key] = copy.deepcopy(old[key])
            else:
                row.pop(key, None)
        if withdrawn:
            # A publication delay must not reactivate a ticket whose rider is
            # now explicitly absent. Keep only the earlier public evidence.
            row.setdefault("previous_predictions", []).append(copy.deepcopy(prior))
            row.pop("prediction", None)
            row.pop("lines", None)
        row["closed"] = clock().timestamp() >= close
        if not row.get("prediction"):
            row["message"] = ("欠場確認後の買い目を締切前に公開できませんでした。以前の予想は履歴に保存しています。"
                              if withdrawn else "締切前に公開できた予想の保存なし。成績には含めません。")
        removed.append(rid)
    return removed


def checkpoint_day(out, existing, today):
    rows = sorted((r for r in existing.values() if r["date"] == today), key=lambda r: r["race_id"])
    save_json(out / "nexus_forecasts" / (today + ".json"), rows)


def run(root=ROOT, clock=now_jst, engine_factory=None, history=None,
        max_input_age=1800, min_lead_seconds=0):
    root = Path(root)
    now = clock()
    if now.tzinfo is None:
        raise ValueError("timezone-aware clock required")
    today = now.date().isoformat()
    out = root / "outputs"
    live = read_json(out / "nexus_live.json", {})
    existing = load_saved_rows(root, live)
    before = copy.deepcopy(existing)
    schedule, authoritative = saved_schedule(root, today)
    snaps = {}
    invalid = []
    for path in sorted((root / SNAPSHOTS / today / "preclose").glob("*.json")):
        try:
            snap = read_json(path)
            rid = snap["meta"]["race_id"]
            if path.stem != rid or snap["meta"]["date"] != today or snap.get("source_kind") != "existing_public_page_response":
                raise ValueError("saved source identity mismatch")
            snaps[rid] = snap
            if rid not in schedule:
                meta = snap["meta"]
                schedule[rid] = dict(race_id=rid, date=today, venue=meta["venue"],
                                     race_no=meta["race_number"], start_at=meta["start_at"],
                                     close_at=meta["close_at"], source_url=meta["source_url"])
                # A partial set of successful cards cannot certify a full day.
                authoritative = False
        except Exception as exc:
            invalid.append(dict(file=path.name, reason=f"{type(exc).__name__}: {exc}"))
    if not schedule:
        for row in existing.values():
            if row.get("date") == today:
                schedule[row["race_id"]] = dict(row, start_at=timestamp(row["start_at"]).timestamp(),
                                               close_at=timestamp(row["close_at"]).timestamp())
    model_path = root / "models/nexus53"
    manifest = read_json(model_path / "manifest.json")
    history_status = dict(through=None, required_through=(now.date() - timedelta(days=1)).isoformat(), errors=[])
    try:
        if history is None:
            hp = root / "data/nexus/history.json.gz"
            history = inputs.History.load(hp if hp.exists() else model_path / "history.json.gz")
        if history.data["through"] >= today:
            raise ValueError("history must end before the prediction day")
        history_status = advance_saved_history(root, history, today)
    except Exception as exc:
        history = None
        history_status["errors"].append(dict(reason=f"{type(exc).__name__}: {exc}"))
    engine = None
    generated, quotes_updated, stale, errors = [], [], [], []
    source_changed = False
    for rid, item in sorted(schedule.items(), key=lambda pair: (float(pair[1]["close_at"]), pair[0])):
        row = existing.setdefault(rid, dict(race_id=rid, date=today))
        # Officially settled records and their original stakes are immutable here.
        if row.get("result_complete") or row.get("refund_all") or row.get("cancelled"):
            continue
        close = float(item["close_at"])
        if row.get("close_at"):
            close = min(close, timestamp(row["close_at"]).timestamp())
        row.update(venue=item.get("venue", row.get("venue", "")),
                   race_no=int(item.get("race_no", row.get("race_no", 0))),
                   source_url=item.get("source_url", row.get("source_url", "")),
                   start_at=_iso(item["start_at"]), close_at=_iso(close))
        row["session"] = session_for(row, schedule)
        row["closed"] = clock().timestamp() >= close
        snap = snaps.get(rid)
        if row["closed"]:
            if not row.get("prediction"):
                row["message"] = "締切前予想の保存なし。公式結果のみ表示し、成績には含めません。"
            continue
        if not snap:
            stale.append(dict(race_id=rid, reason="missing_original_snapshot"))
            continue
        try:
            captured = timestamp(snap["captured_at"])
            data, meta = snap["data"], snap["meta"]
            close = min(close, float(data["race"]["closeAt"]))
            row["close_at"] = _iso(close)
            row["closed"] = clock().timestamp() >= close
            if data["race"].get("cancel") is True:
                row["cancelled"] = True
            age = clock().timestamp() - captured.timestamp()
            if (not 0 <= age <= max_input_age or captured.timestamp() >= close
                    or clock().timestamp() + min_lead_seconds >= close
                    or data["race"].get("decidedAt")):
                stale.append(dict(race_id=rid, reason="stale_or_postclose_source", age_seconds=round(age)))
                continue
            row.update(close_at=_iso(close), start_at=_iso(data["race"]["startAt"]),
                       class_name=meta["race_class"], stage=meta["race_type"],
                       cancelled=bool(data["race"]["cancel"]))
            if row["cancelled"]:
                continue
            players = {str(p["id"]): p for p in data["players"]}
            row["riders"] = [dict(car=e["number"], name=players[str(e["playerId"])].get("name", ""), pid=str(e["playerId"]))
                             for e in data["entries"] if e["absent"] is False]
            old = row.get("prediction")
            active_cars = {r["car"] for r in row["riders"]}
            withdrawn_ticket = bool(old and any(int(car) not in active_cars
                                                for ticket in old["tickets"] for car in ticket.split("-")))
            if withdrawn_ticket:
                row.setdefault("previous_predictions", []).append(copy.deepcopy(old))
                row.pop("prediction", None)
            needs_forecast = (not old or not is_open_prediction(old)
                              or old.get("input_sha256") != snap["input_sha256"]
                              or old.get("confidence", {}).get("status") != "assessed"
                              or (history is not None and old.get("history_through") != history.data["through"]))
            if needs_forecast:
                if history is None:
                    raise ValueError("verified prior-day history unavailable")
                if engine is None:
                    if engine_factory is None:
                        from nexus_model.engine import Engine
                        engine_factory = Engine
                    engine = engine_factory(model_path)
                matrix, line = inputs.build(history, data, meta)
                candidate = engine.predict(matrix)
                saved_at = clock()
                if saved_at.timestamp() + min_lead_seconds >= close:
                    row["closed"] = saved_at.timestamp() >= close
                    errors.append(dict(race_id=rid, reason="deadline_reached_during_inference"))
                    continue
                candidate.pop("q", None)
                candidate.update(captured_at=saved_at.isoformat(), input_captured_at=snap["captured_at"],
                                 source_sha256=snap["sha256"], input_sha256=snap["input_sha256"],
                                 source_kind=snap["source_kind"], history_through=history.data["through"],
                                 allocations={}, odds={})
                # Only new preclose inferences receive evidence; legacy rows stay as saved.
                try:
                    bind_prediction(candidate, race_id=rid, race_date=today, close_at=row["close_at"])
                except Exception as exc:
                    candidate.pop("replay_evidence", None)
                    candidate["replay_evidence_error"] = f"{type(exc).__name__}: {exc}"
                if candidate.get("replay_evidence_error"):
                    errors.append(dict(race_id=rid, reason="replay evidence: " + candidate["replay_evidence_error"]))
                annotate_prediction(candidate, annotated_at=saved_at.isoformat())
                if old and not withdrawn_ticket:
                    row.setdefault("previous_predictions", []).append(copy.deepcopy(old))
                row["prediction"] = candidate
                row["lines"] = line["known_unchanged_groups"]
                row.pop("message", None)
                generated.append(rid)
                source_changed = True
                # Persist immediately; later races/allocations must not delay
                # the evidence timestamp of an earlier generated forecast.
                if clock().timestamp() + min_lead_seconds < close:
                    checkpoint_day(out, existing, today)
                else:
                    discard_late_changes({rid: row}, before, clock, min_lead_seconds)
                    errors.append(dict(race_id=rid, reason="deadline_reached_before_checkpoint"))
                    continue
            prediction = row.get("prediction")
            if prediction and not data.get("finalOdds") and clock().timestamp() + min_lead_seconds < close:
                quotes = {"-".join(map(str, q["key"])): q["odds"] for q in data.get("trifecta", [])
                          if q.get("unitPrice") == 100 and q.get("absent") is not True
                          and isinstance(q.get("odds"), (int, float)) and q["odds"] > 0
                          and len(q.get("key", [])) == 3 and len(set(q["key"])) == 3}
                if all(t in quotes for t in prediction["tickets"]) and (
                        not prediction.get("odds_at") or timestamp(prediction["odds_at"]) < captured):
                    allocation_time = clock().isoformat()
                    prepared = allocation_snapshot(prediction, quotes, captured.isoformat(), allocation_time, row["close_at"])
                    if clock().timestamp() + min_lead_seconds < close:
                        prediction.update(prepared)
                        prediction["odds_source_sha256"] = snap["sha256"]
                        quotes_updated.append(rid)
                        source_changed = True
                        checkpoint_day(out, existing, today)
        except Exception as exc:
            errors.append(dict(race_id=rid, reason=f"{type(exc).__name__}: {exc}"))
    late = discard_late_changes(existing, before, clock, min_lead_seconds)
    generated = [rid for rid in generated if rid not in late and existing[rid].get("prediction")]
    quotes_updated = [rid for rid in quotes_updated if rid not in late]
    current = [existing[rid] for rid in schedule]
    missing_open = [r["race_id"] for r in current if not r.get("prediction") and not r.get("cancelled")
                    and timestamp(r["close_at"]).timestamp() > clock().timestamp()]
    missing_closed = [r["race_id"] for r in current if not r.get("prediction")
                      and timestamp(r["close_at"]).timestamp() <= clock().timestamp()]
    expected = read_json(root / SNAPSHOTS / today / "index.json", {})
    missing_urls = sorted(set(expected.get("urls", [])) - {r.get("source_url") for r in schedule.values()})
    report = dict(date=today, checked_at=clock().isoformat(), scheduled=len(schedule),
                  schedule_complete=authoritative, predicted=sum(bool(r.get("prediction")) for r in current),
                  generated=generated, odds_updated=quotes_updated, missing_open=missing_open,
                  missing_closed=missing_closed, missing_schedule_urls=missing_urls,
                  stale_inputs=stale, invalid_inputs=invalid,
                  errors=errors, not_published_before_deadline=late, history=history_status)
    feed = copy.deepcopy(live)
    feed.update(schema=1, date=today, races=sorted(existing.values(), key=lambda r: (r["date"], r["race_id"])),
                coverage=dict(scheduled=len(schedule), predicted=report["predicted"],
                              missing_open=len(missing_open), schedule_complete=authoritative),
                history_freshness=history_status)
    if manifest:
        feed["model"] = {k: manifest[k] for k in ("id", "name", "historical", "train", "counts", "live_policy") if k in manifest}
    if source_changed or not feed.get("updated_at"):
        feed["updated_at"] = clock().isoformat()
        if history is not None:
            feed["history_through"] = history.data["through"]
    if missing_open or not authoritative:
        feed["notice"] = "全レースの予想入力を確認中です。確認できたレースから掲載しています。"
    elif not errors:
        feed["notice"] = ""
    # Top-level results_updated_at/policy_updated_at and older records survive.
    for ds in sorted({r["date"] for r in existing.values()}):
        path = out / "nexus_forecasts" / (ds + ".json")
        rows = [r for r in feed["races"] if r["date"] == ds]
        if rows != read_json(path, []):
            save_json(path, rows)
    if feed != live:
        save_json(out / "nexus_live.json", feed)
    save_json(out / "nexus_coverage_status.json", report)
    print("NEXUS_SAVED_INPUT_PUBLICATION " + canonical(report), flush=True)
    return report


def copy_newer_inputs(source_root, destination_root):
    """Do not replace a newer collector response committed by another writer."""
    for path in sorted((source_root / SNAPSHOTS).glob("*/*.json")) + sorted((source_root / SNAPSHOTS).glob("*/*/*.json")):
        target = destination_root / path.relative_to(source_root)
        incoming, existing = read_json(path), read_json(target)
        if not existing or timestamp(incoming["captured_at"]) > timestamp(existing["captured_at"]):
            save_json(target, incoming)


def gate_saved_publication(root, baseline, clock=now_jst, lead_seconds=60):
    """Recheck private changes immediately before commit/push."""
    feed = read_json(root / "outputs/nexus_live.json", {})
    rows = load_saved_rows(root, feed)
    late = discard_late_changes(rows, baseline, clock, lead_seconds)
    if not late:
        return []
    feed["races"] = sorted(rows.values(), key=lambda r: (r["date"], r["race_id"]))
    today = [r for r in feed["races"] if r["date"] == feed["date"]]
    missing = [r["race_id"] for r in today if not r.get("prediction") and not r.get("cancelled")
               and timestamp(r["close_at"]).timestamp() > clock().timestamp()]
    feed.setdefault("coverage", {}).update(predicted=sum(bool(r.get("prediction")) for r in today), missing_open=len(missing))
    for ds in {rows[rid]["date"] for rid in late}:
        checkpoint_day(root / "outputs", rows, ds)
    save_json(root / "outputs/nexus_live.json", feed)
    report = read_json(root / "outputs/nexus_coverage_status.json", {})
    report["not_published_before_deadline"] = sorted(set(report.get("not_published_before_deadline", [])) | set(late))
    report["generated"] = [rid for rid in report.get("generated", []) if rid not in late]
    report["odds_updated"] = [rid for rid in report.get("odds_updated", []) if rid not in late]
    report["predicted"] = feed["coverage"]["predicted"]
    report["missing_open"] = missing
    report["missing_closed"] = [r["race_id"] for r in today if not r.get("prediction")
                                and timestamp(r["close_at"]).timestamp() <= clock().timestamp()]
    save_json(root / "outputs/nexus_coverage_status.json", report)
    return late


def update_near_close_comparison(root, baseline, generate=False):
    """An optional comparison failure must not block canonical forecasts."""
    path = Path(root) / "outputs/conditional_near_close.json"
    try:
        if generate:
            subprocess.run([sys.executable, str(Path(root) / "src/conditional_budget_near_close.py")],
                           cwd=root, check=True)
        from conditional_budget_near_close import gate_publication
        return True, gate_publication(root, baseline)
    except Exception as exc:
        # A failing process may have written a partial/new file already.
        if baseline:
            save_json(path, baseline)
        else:
            path.unlink(missing_ok=True)
        status_path = Path(root) / "outputs/nexus_coverage_status.json"
        status = read_json(status_path, {})
        status["near_close_comparison"] = dict(status="error", reason=f"{type(exc).__name__}: {exc}")
        save_json(status_path, status)
        print("::warning::Near-close comparison failed; original forecasts continue: " + str(exc), flush=True)
        return False, ["comparison_error"]


def record_roi_comparison_error(root, exc):
    status_path = Path(root) / "outputs/nexus_coverage_status.json"
    status = read_json(status_path, {})
    status["roi_comparison"] = dict(status="error", reason=f"{type(exc).__name__}: {exc}")
    save_json(status_path, status)
    print("::warning::ROI comparison failed; original forecasts continue: " + str(exc), flush=True)


def update_roi_comparison(root, baseline, generate=False):
    """Publish the optional eight-ticket record without risking original picks."""
    path = Path(root) / "outputs/conditional_roi.json"
    try:
        if generate:
            subprocess.run([sys.executable, str(Path(root) / "src/conditional_roi.py")],
                           cwd=root, check=True)
        from conditional_roi import gate_publication
        return True, gate_publication(root, copy.deepcopy(baseline), lead_seconds=60)
    except Exception as exc:
        # Generation and final gates may fail after a partial write. Only the
        # original public baseline is allowed to survive a failed attempt.
        if baseline:
            save_json(path, baseline)
        else:
            path.unlink(missing_ok=True)
        record_roi_comparison_error(root, exc)
        return False, ["comparison_error"]


def publish_ci(root=ROOT):
    """Recompute on latest main for each bounded push attempt, without HTTP data."""
    root = Path(root)
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("--publish is reserved for the existing authorized GitHub collector workflow")
    def git(directory, *args, check=True):
        return subprocess.run(["git", *args], cwd=directory, check=check,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with tempfile.TemporaryDirectory(prefix="nexus-existing-inputs-") as directory:
        checkout = Path(directory) / "checkout"
        git(root, "fetch", "origin", "main")
        git(root, "worktree", "add", "--detach", str(checkout), "origin/main")
        try:
            for attempt in range(4):
                if attempt:
                    git(root, "fetch", "origin", "main")
                    git(checkout, "reset", "--hard", "origin/main")
                    git(checkout, "clean", "-fd")
                baseline = load_saved_rows(checkout, read_json(checkout / "outputs/nexus_live.json", {}))
                near_close_baseline = read_json(checkout / "outputs/conditional_near_close.json", {})
                roi_baseline_error = None
                try:
                    roi_baseline = read_json(checkout / "outputs/conditional_roi.json", {})
                    if not isinstance(roi_baseline, dict):
                        raise ValueError("ROI comparison baseline must be a JSON object")
                except Exception as exc:
                    # An invalid optional feed must not block the main forecast
                    # publisher or be replaced by a fresh retrospective record.
                    roi_baseline, roi_baseline_error = None, exc
                copy_newer_inputs(root, checkout)
                result = subprocess.run([sys.executable, str(checkout / "src/nexus_publish_saved_inputs.py"),
                                         "--min-lead-seconds", "60"], cwd=checkout)
                if result.returncode not in (0, 1):
                    raise RuntimeError("offline publication failed before writing coverage status")
                # Confidence annotations use only frozen scores and must also
                # reach previously saved forecasts without changing their bets.
                subprocess.run([sys.executable, str(checkout / "src/nexus_confidence_sync.py")],
                               cwd=checkout, check=True)
                # Keep the original conditional allocation immutable. Compare
                # the same tickets and budget using already captured later
                # quotes, in a separate prospective record with no source HTTP.
                near_close_healthy, _ = update_near_close_comparison(checkout, near_close_baseline, generate=True)
                if roi_baseline_error is None:
                    roi_healthy, _ = update_roi_comparison(checkout, roi_baseline, generate=True)
                else:
                    record_roi_comparison_error(checkout, roi_baseline_error)
                    roi_healthy = False
                gate_saved_publication(checkout, baseline)
                if near_close_healthy:
                    near_close_healthy, _ = update_near_close_comparison(checkout, near_close_baseline)
                if roi_healthy:
                    roi_healthy, _ = update_roi_comparison(checkout, roi_baseline)
                for name in (str(SNAPSHOTS), "outputs/nexus_live.json", "outputs/nexus_forecasts",
                             "outputs/conditional_live.json", "outputs/conditional_near_close.json",
                             "outputs/conditional_roi.json",
                             "outputs/nexus_coverage_status.json", "data/nexus/history.json.gz"):
                    if (checkout / name).exists():
                        git(checkout, "add", "-f", "--", name)
                if git(checkout, "diff", "--cached", "--quiet", check=False).returncode == 0:
                    return result.returncode
                git(checkout, "-c", "user.name=github-actions[bot]", "-c",
                    "user.email=41898282+github-actions[bot]@users.noreply.github.com", "commit", "-m",
                    "Publish NEXUS from existing pre-close source snapshots")
                late_forecasts = gate_saved_publication(checkout, baseline)
                _, late_allocations = (update_near_close_comparison(checkout, near_close_baseline)
                                       if near_close_healthy else (False, []))
                _, late_roi = (update_roi_comparison(checkout, roi_baseline)
                               if roi_healthy else (False, []))
                if late_forecasts or late_allocations or late_roi:
                    git(checkout, "add", "--", "outputs/nexus_live.json", "outputs/nexus_forecasts", "outputs/nexus_coverage_status.json")
                    if near_close_healthy:
                        git(checkout, "add", "--all", "--", "outputs/conditional_near_close.json")
                    if roi_healthy:
                        git(checkout, "add", "--all", "--", "outputs/conditional_roi.json")
                    git(checkout, "-c", "user.name=github-actions[bot]", "-c",
                        "user.email=41898282+github-actions[bot]@users.noreply.github.com", "commit", "--amend", "--no-edit")
                if git(checkout, "push", "origin", "HEAD:main", check=False).returncode == 0:
                    report = read_json(checkout / "outputs/nexus_coverage_status.json", {})
                    return 0 if report.get("schedule_complete") and not report.get("missing_open") else 1
            raise RuntimeError("NEXUS publish raced with newer main on all four attempts")
        finally:
            git(root, "worktree", "remove", "--force", str(checkout), check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--min-lead-seconds", type=float, default=0)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.publish:
        return publish_ci()
    if args.verify:
        report = read_json(ROOT / "outputs/nexus_coverage_status.json", {})
    else:
        report = run(min_lead_seconds=args.min_lead_seconds)
    return 0 if report.get("schedule_complete") and not report.get("missing_open") else 1


if __name__ == "__main__":
    raise SystemExit(main())
