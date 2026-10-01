import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common import TODAY_CSV, OUTPUT_DIR, RAW_DIR

STATUS_PATH = OUTPUT_DIR / "manager_status.json"
INCIDENT_HISTORY_PATH = OUTPUT_DIR / "manager_incident_history.jsonl"
LAST_GOOD_DIR = RAW_DIR / "last_good"
LAST_GOOD_ENTRIES = LAST_GOOD_DIR / "today_entries.csv"
LAST_GOOD_ODDS = LAST_GOOD_DIR / "today_odds.csv"
RACE_SCHEDULE_PATH = OUTPUT_DIR / "latest_race_schedule.csv"
MAX_REPAIR_ATTEMPTS = 3
RETRY_SECONDS = 5


def snapshot_last_good():
    """Persist only an already-audited source snapshot as rollback material."""
    LAST_GOOD_DIR.mkdir(parents=True, exist_ok=True)
    if TODAY_CSV.exists():
        shutil.copy2(TODAY_CSV, LAST_GOOD_ENTRIES)
    odds = RAW_DIR / "today_odds.csv"
    if odds.exists():
        shutil.copy2(odds, LAST_GOOD_ODDS)


def rollback_last_good():
    if not LAST_GOOD_ENTRIES.exists():
        return False
    shutil.copy2(LAST_GOOD_ENTRIES, TODAY_CSV)
    if LAST_GOOD_ODDS.exists():
        shutil.copy2(LAST_GOOD_ODDS, RAW_DIR / "today_odds.csv")
    return True


def validate_source_gate():
    """Publication gate: source data must pass both rider and whole-race audits."""
    entry_problems, stats = audit_entries()
    coverage_problems = audit_race_coverage()
    identity_problems = []
    try:
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
        if "player_id" in entries.columns:
            for race_id, group in entries.groupby("race_id"):
                ids = group["player_id"].fillna("").astype(str).str.strip()
                if ids.eq("").any() or ids.duplicated().any():
                    identity_problems.append({"type": "source_identity_gate_failed", "race_id": str(race_id)})
    except Exception as exc:
        identity_problems.append({"type": "source_identity_gate_unreadable", "detail": str(exc)})
    problems = entry_problems + coverage_problems + identity_problems
    return not problems, problems, stats


def audit_entries():
    problems = []
    try:
        df = pd.read_csv(TODAY_CSV, dtype={"race_id": str})
    except Exception as exc:
        return [{"type": "entries_unreadable", "detail": str(exc)}], {}

    if df.empty:
        return [{"type": "entries_empty"}], {}

    required = {"race_id", "venue", "race_no", "car_no"}
    missing = sorted(required - set(df.columns))
    if missing:
        return [{"type": "missing_columns", "columns": missing}], {}

    stats = {}
    for race_id, group in df.groupby("race_id"):
        cars = sorted(pd.to_numeric(group["car_no"], errors="coerce").dropna().astype(int).unique().tolist())
        expected_values = pd.to_numeric(group.get("entries_number"), errors="coerce").dropna()
        expected = int(expected_values.max()) if len(expected_values) else 0
        # Never infer the official field size from max(car_no): if tail riders
        # (for example 8/9) disappear together, max(car_no) makes a 9-car race
        # look like a valid 7-car race. entries_number is source-declared.
        missing_cars = sorted(set(range(1, expected + 1)) - set(cars)) if expected > 0 else []
        duplicate_cars = sorted(group.loc[group.duplicated("car_no", keep=False), "car_no"].dropna().astype(int).unique().tolist())
        stats[str(race_id)] = {"cars": cars, "count": len(cars), "expected_entries": expected}
        if expected <= 0:
            problems.append({"type": "missing_expected_field_size", "race_id": str(race_id), "cars": cars})
        elif len(cars) != expected or missing_cars:
            problems.append({"type": "missing_riders", "race_id": str(race_id), "expected": expected, "missing_cars": missing_cars, "cars": cars})
        if duplicate_cars:
            problems.append({"type": "duplicate_riders", "race_id": str(race_id), "cars": duplicate_cars})
        if len(cars) < 5 or len(cars) > 9:
            problems.append({"type": "implausible_rider_count", "race_id": str(race_id), "count": len(cars), "cars": cars})
    return problems, stats


def audit_race_coverage():
    """Detect entire races disappearing from the daily snapshot."""
    problems = []
    if not RACE_SCHEDULE_PATH.exists():
        return [{"type": "race_schedule_missing"}]
    try:
        schedule = pd.read_csv(RACE_SCHEDULE_PATH, dtype={"race_id": str})
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str})
        today = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
        if "date" in schedule.columns:
            schedule = schedule[schedule["date"].astype(str).eq(today)]
        expected_ids = set(schedule["race_id"].dropna().astype(str))
        actual_ids = set(entries["race_id"].dropna().astype(str))
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if missing:
            problems.append({
                "type": "missing_entire_races",
                "expected_races": len(expected_ids),
                "actual_races": len(actual_ids),
                "race_ids": missing,
            })
        if extra:
            problems.append({"type": "unexpected_races", "race_ids": extra})
    except Exception as exc:
        problems.append({"type": "race_coverage_audit_failed", "detail": str(exc)})
    return problems


def audit_budget():
    problems = []
    path = OUTPUT_DIR / "latest_shadow_bets.csv"
    if not path.exists() or path.stat().st_size == 0:
        return problems
    try:
        bets = pd.read_csv(path, dtype={"race_id": str})
        if bets.empty or "stake_yen" not in bets.columns:
            return problems
        for race_id, group in bets.groupby("race_id"):
            total = int(pd.to_numeric(group["stake_yen"], errors="coerce").fillna(0).sum())
            if total != 10000:
                problems.append({"type": "race_budget_mismatch", "race_id": str(race_id), "total_yen": total})
    except Exception as exc:
        problems.append({"type": "budget_audit_failed", "detail": str(exc)})
    return problems


def audit_live_bets():
    """Detect the failure mode where predictions/bet candidates exist but the live bet output is empty."""
    problems = []
    shadow_path = OUTPUT_DIR / "latest_shadow_bets.csv"
    live_path = OUTPUT_DIR / "latest_bets.csv"
    if not shadow_path.exists() or shadow_path.stat().st_size == 0:
        return problems
    try:
        shadow = pd.read_csv(shadow_path, dtype={"race_id": str})
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return problems
    if shadow.empty:
        return problems
    try:
        live = pd.read_csv(live_path, dtype={"race_id": str}) if live_path.exists() else pd.DataFrame()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        live = pd.DataFrame()
    # The site intentionally renders shadow recommendations even while the
    # external-profit purchase gate is closed. Empty latest_bets must therefore
    # never masquerade as a prediction/data failure.
    if live.empty and "purchase_authorized" in shadow.columns:
        authorized = shadow["purchase_authorized"].astype(str).str.lower().isin({"true", "1", "yes"}).any()
        if authorized:
            problems.append({"type": "authorized_bets_missing", "shadow_rows": int(len(shadow))})
    return problems


def audit_site_output():
    problems = []
    html = OUTPUT_DIR / "index.html"
    if not html.exists() or html.stat().st_size < 1000:
        return [{"type": "site_output_missing_or_too_small"}]
    try:
        content = html.read_text(encoding="utf-8")
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str})
        # The published HTML must contain every race/car currently present in
        # the validated prediction input. This catches stale pages that are
        # large enough to pass the old file-size-only health check.
        for race_id, group in entries.groupby("race_id"):
            race_id = str(race_id)
            if race_id not in content:
                problems.append({"type": "site_missing_race", "race_id": race_id})
                continue
            cars = sorted(pd.to_numeric(group["car_no"], errors="coerce").dropna().astype(int).unique().tolist())
            # Prediction audit is the authoritative car-level check; here require
            # the race itself to be represented in the actual published artifact.
            if not cars:
                problems.append({"type": "site_race_has_no_valid_cars", "race_id": race_id})
    except Exception as exc:
        problems.append({"type": "site_output_unreadable", "detail": str(exc)})
    return problems


def audit_results():
    problems = []
    path = OUTPUT_DIR / "latest_results.json"
    try:
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str})
    except Exception:
        return problems
    now = datetime.now(ZoneInfo("Asia/Tokyo")).timestamp()
    overdue = set()
    if "close_at" in entries.columns:
        close_at = pd.to_numeric(entries["close_at"], errors="coerce")
        overdue = set(entries.loc[close_at.notna() & (close_at < now - 30 * 60), "race_id"].astype(str))
    if not overdue:
        return problems
    if not path.exists() or path.stat().st_size == 0:
        return [{"type": "results_missing", "overdue_races": sorted(overdue)}]
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        decided = {str(x.get("race_id")) for x in rows if x.get("official_result_available")}
        missing = sorted(overdue - decided)
        if missing:
            problems.append({"type": "results_overdue", "race_ids": missing})
    except Exception as exc:
        problems.append({"type": "results_unreadable", "detail": str(exc)})
    return problems


def audit_prediction_outputs():
    problems = []
    latest = OUTPUT_DIR / "latest_predictions.csv"
    if not latest.exists() or latest.stat().st_size == 0:
        problems.append({"type": "prediction_missing"})
        return problems
    try:
        pred = pd.read_csv(latest, dtype={"race_id": str})
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str})
        for race_id, group in entries.groupby("race_id"):
            source_cars = set(pd.to_numeric(group["car_no"], errors="coerce").dropna().astype(int))
            p = pred[pred["race_id"].astype(str).eq(str(race_id))]
            pred_cars = set(pd.to_numeric(p["car_no"], errors="coerce").dropna().astype(int)) if "car_no" in p else set()
            missing = sorted(source_cars - pred_cars)
            if missing:
                problems.append({"type": "prediction_missing_riders", "race_id": str(race_id), "missing_cars": missing})
    except Exception as exc:
        problems.append({"type": "prediction_unreadable", "detail": str(exc)})
    return problems


def audit_identity_and_prediction_quality():
    problems = []
    try:
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
    except Exception:
        return problems

    # A car must map to exactly one rider and a rider must not occupy multiple
    # cars in the same race. This catches malformed snapshots that still have
    # the correct row count.
    if "player_id" in entries.columns:
        for race_id, group in entries.groupby("race_id"):
            valid = group.dropna(subset=["player_id"]).copy()
            if valid["player_id"].astype(str).duplicated().any():
                dup = sorted(valid.loc[valid["player_id"].astype(str).duplicated(keep=False), "player_id"].astype(str).unique().tolist())
                problems.append({"type": "duplicate_player_identity", "race_id": str(race_id), "player_ids": dup})
            if valid["player_id"].astype(str).isin({"", "nan", "None"}).any() or len(valid) != len(group):
                problems.append({"type": "missing_player_identity", "race_id": str(race_id)})

    latest = OUTPUT_DIR / "latest_predictions.csv"
    if latest.exists() and latest.stat().st_size:
        try:
            pred = pd.read_csv(latest, dtype={"race_id": str, "player_id": str})
            if "p_win" in pred.columns:
                pred["p_win"] = pd.to_numeric(pred["p_win"], errors="coerce")
                for race_id, group in pred.groupby("race_id"):
                    probs = group["p_win"]
                    if probs.isna().any() or (~probs.between(0, 1)).any():
                        problems.append({"type": "invalid_prediction_probability", "race_id": str(race_id)})
                    total = float(probs.sum(skipna=True))
                    if len(group) and abs(total - 1.0) > 0.02:
                        problems.append({"type": "prediction_probability_not_normalized", "race_id": str(race_id), "sum": total})
        except Exception as exc:
            problems.append({"type": "prediction_quality_audit_failed", "detail": str(exc)})
    return problems


def audit_freshness():
    problems = []
    now = datetime.now(ZoneInfo("Asia/Tokyo")).timestamp()
    # During active racing, stale prediction artifacts can look structurally
    # perfect while actually serving old information.
    latest = OUTPUT_DIR / "latest_predictions.csv"
    try:
        schedule = pd.read_csv(RACE_SCHEDULE_PATH, dtype={"race_id": str})
        close_at = pd.to_numeric(schedule.get("close_at"), errors="coerce")
        active = close_at.notna() & close_at.gt(now) & close_at.le(now + 60 * 60)
        if active.any() and latest.exists():
            age = now - latest.stat().st_mtime
            if age > 45 * 60:
                problems.append({"type": "stale_predictions", "age_seconds": int(age)})
        elif active.any() and not latest.exists():
            problems.append({"type": "prediction_missing"})
    except Exception:
        pass
    return problems


def append_incident_history(problems, action=None):
    if not problems:
        return
    record = {
        "at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "problem_types": sorted({str(p.get("type")) for p in problems}),
        "problems": problems,
        "repair_action": action,
    }
    with INCIDENT_HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_health(problems, stats):
    kinds = [str(p.get("type")) for p in problems]
    severity = "critical" if any(k in {
        "missing_riders", "missing_entire_races", "entries_unreadable",
        "entries_empty", "prediction_missing", "prediction_missing_riders",
    } for k in kinds) else ("warning" if kinds else "ok")
    expected_riders = sum(int(v.get("expected_entries", 0) or 0) for v in stats.values())
    actual_riders = sum(int(v.get("count", 0) or 0) for v in stats.values())
    return {
        "severity": severity,
        "problem_types": sorted(set(kinds)),
        "race_count": len(stats),
        "expected_riders": expected_riders,
        "actual_riders": actual_riders,
        "rider_integrity_ok": bool(stats) and expected_riders == actual_riders and not any(
            k in {"missing_riders", "missing_entire_races", "duplicate_riders"} for k in kinds
        ),
    }


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def repair(problems=None):
    """Choose the smallest safe repair for the detected failure class."""
    problems = problems or []
    kinds = {str(p.get("type")) for p in problems}

    data_kinds = {
        "entries_unreadable", "entries_empty", "missing_columns",
        "missing_expected_field_size", "missing_riders", "duplicate_riders",
        "implausible_rider_count", "missing_entire_races", "unexpected_races",
        "race_schedule_missing", "race_coverage_audit_failed",
        "duplicate_player_identity", "missing_player_identity",
    }
    prediction_kinds = {
        "prediction_missing", "prediction_missing_riders", "prediction_unreadable",
        "invalid_prediction_probability", "prediction_probability_not_normalized",
        "prediction_quality_audit_failed", "stale_predictions",
    }
    result_kinds = {"results_missing", "results_overdue", "results_unreadable"}
    site_kinds = {"site_output_missing_or_too_small", "site_missing_race", "site_race_has_no_valid_cars", "site_output_unreadable"}
    bet_kinds = {"race_budget_mismatch", "budget_audit_failed", "authorized_bets_missing"}

    actions = []
    # Any source-data integrity problem gets the strongest repair: an atomic
    # full-day rebuild. Never mask it with a near-close-only snapshot.
    if kinds & data_kinds:
        # Keep the last known-good source before touching production input.
        pre_ok, _, _ = validate_source_gate()
        if pre_ok:
            snapshot_last_good()
        if run([sys.executable, "src/fetch_today_entries.py"]) != 0:
            rollback_last_good()
            return False, "full_day_data_rebuild_failed_rolled_back"
        gate_ok, gate_problems, _ = validate_source_gate()
        if not gate_ok:
            rolled_back = rollback_last_good()
            append_incident_history(gate_problems, "publication_gate_rejected")
            return False, "publication_gate_rejected_rolled_back" if rolled_back else "publication_gate_rejected_no_backup"
        snapshot_last_good()
        actions.append("full_day_data_rebuild_gated")

    # Prediction/site/bet faults are regenerated from the already validated
    # source snapshot. A successful command is not enough; the next audit loop
    # must independently prove every race/rider is present.
    if (kinds & (data_kinds | prediction_kinds | site_kinds | bet_kinds)) or not (OUTPUT_DIR / "latest_predictions.csv").exists():
        if run([sys.executable, "src/predict.py"]) != 0:
            return False, "predict_regeneration_failed"
        actions.append("prediction_regeneration")

    if kinds & (data_kinds | result_kinds):
        if run([sys.executable, "src/settle_results.py"]) != 0:
            return False, "settlement_refresh_failed"
        actions.append("settlement_refresh")

    if not actions:
        # Unknown failures get a conservative full integrity rebuild rather than
        # being silently ignored.
        if run([sys.executable, "src/fetch_today_entries.py"]) != 0:
            return False, "conservative_full_rebuild_fetch_failed"
        if run([sys.executable, "src/predict.py"]) != 0:
            return False, "conservative_full_rebuild_predict_failed"
        actions.extend(["conservative_full_day_rebuild", "prediction_regeneration"])
    return True, "+".join(actions)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    repaired = False
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        entry_problems, stats = audit_entries()
        prediction_problems = audit_prediction_outputs()
        problems = (
            entry_problems
            + audit_race_coverage()
            + prediction_problems
            + audit_identity_and_prediction_quality()
            + audit_freshness()
            + audit_budget()
            + audit_live_bets()
            + audit_site_output()
            + audit_results()
        )
        history.append({
            "attempt": attempt,
            "problems": problems,
            "health": summarize_health(problems, stats),
        })
        if not problems:
            # Only audited source snapshots are eligible to become rollback
            # checkpoints. This prevents a corrupt fetch from replacing backup.
            snapshot_last_good()
            status = "healthy" if not repaired else "repaired"
            break
        if attempt >= MAX_REPAIR_ATTEMPTS:
            status = "unhealthy"
            break
        ok, action = repair(problems)
        append_incident_history(problems, action)
        repaired = repaired or ok
        history[-1]["repair_action"] = action
        if not ok:
            time.sleep(RETRY_SECONDS)
            continue
        time.sleep(RETRY_SECONDS)

    payload = {
        "checked_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "status": status,
        "repaired": repaired,
        "attempts": history,
        "race_stats": stats,
        "health": summarize_health(history[-1]["problems"] if history else [], stats),
    }
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status == "unhealthy":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
