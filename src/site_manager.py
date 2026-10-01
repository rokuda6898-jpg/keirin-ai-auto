import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common import TODAY_CSV, OUTPUT_DIR, RAW_DIR

STATUS_PATH = OUTPUT_DIR / "manager_status.json"
MAX_REPAIR_ATTEMPTS = 3
RETRY_SECONDS = 5


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
        problems.append({"type": "site_output_missing_or_too_small"})
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


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def repair():
    # Rebuild the entire daily snapshot. Partial patching is deliberately avoided:
    # it was the source of stale/missing rider states.
    if run([sys.executable, "src/fetch_today_entries.py"]) != 0:
        # Full-day fetching is atomic and can fail when even one source race is
        # temporarily incomplete. Fall back to the persisted daily schedule and
        # rebuild the active near-close races instead of leaving TODAY_CSV absent.
        fallback = [sys.executable, "src/fetch_upcoming_entries.py", "--min-minutes", "5", "--max-minutes", "40", "--retry-sec", "5"]
        if run(fallback) != 0 or not TODAY_CSV.exists():
            return False, "fetch_failed_full_and_upcoming"
    if run([sys.executable, "src/predict.py"]) != 0:
        return False, "predict_failed"
    # Also refresh settlement; this is idempotent and closes stale result gaps.
    if run([sys.executable, "src/settle_results.py"]) != 0:
        return False, "settlement_refresh_failed"
    return True, "rebuilt_snapshot_and_results"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    repaired = False
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        entry_problems, stats = audit_entries()
        prediction_problems = audit_prediction_outputs()
        problems = entry_problems + prediction_problems + audit_budget() + audit_live_bets() + audit_site_output() + audit_results()
        history.append({"attempt": attempt, "problems": problems})
        if not problems:
            status = "healthy" if not repaired else "repaired"
            break
        if attempt >= MAX_REPAIR_ATTEMPTS:
            status = "unhealthy"
            break
        ok, action = repair()
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
    }
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status == "unhealthy":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
