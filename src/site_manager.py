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
        expected = max(cars) if cars else 0
        # Japanese keirin fields are normally contiguous 1..N. A missing number is
        # a stronger corruption signal than merely having 5/7/9 rows.
        missing_cars = sorted(set(range(1, expected + 1)) - set(cars))
        duplicate_cars = sorted(group.loc[group.duplicated("car_no", keep=False), "car_no"].dropna().astype(int).unique().tolist())
        stats[str(race_id)] = {"cars": cars, "count": len(cars), "expected_contiguous": expected}
        if missing_cars:
            problems.append({"type": "missing_riders", "race_id": str(race_id), "missing_cars": missing_cars, "cars": cars})
        if duplicate_cars:
            problems.append({"type": "duplicate_riders", "race_id": str(race_id), "cars": duplicate_cars})
        if len(cars) < 5 or len(cars) > 9:
            problems.append({"type": "implausible_rider_count", "race_id": str(race_id), "count": len(cars), "cars": cars})
    return problems, stats


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
        return False, "fetch_failed"
    if run([sys.executable, "src/predict.py"]) != 0:
        return False, "predict_failed"
    return True, "rebuilt_daily_snapshot"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    repaired = False
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        entry_problems, stats = audit_entries()
        prediction_problems = audit_prediction_outputs()
        problems = entry_problems + prediction_problems
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
