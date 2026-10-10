"""Fetch racecards for the isolated conditional trifecta forecast.

This job must not modify company inputs, prediction snapshots, or official results.
Only pre-close, source-verified entry rows are persisted. Missing races stay missing.
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from fetch_today_entries import _entry_rows_complete, parse_race_page
from fixed_year_study import input_reason
from public_live import frozen, read_csv

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path("data/raw/conditional_today_entries.csv")
LEDGER = Path("outputs/company/conditional_live_ledger.jsonl")
MIN_SECONDS_TO_CLOSE = 300
MAX_FETCHES = 90
MAX_WALL_SECONDS = 720


def now_jst():
    return datetime.now(ZoneInfo("Asia/Tokyo"))


def valid_race(frame, race_id, date, close_at):
    """Require complete and consistent starters; no partial/cross-date fallback."""
    if frame.empty or not {"race_id", "date", "car_no", "player_id", "close_at"}.issubset(frame):
        return False
    if frame["race_id"].astype(str).nunique() != 1 or str(frame["race_id"].iloc[0]) != str(race_id):
        return False
    if not frame["date"].astype(str).eq(str(date)).all():
        return False
    complete, _, _ = _entry_rows_complete(frame.to_dict("records"))
    if not complete or input_reason(frame):
        return False
    closes = pd.to_numeric(frame["close_at"], errors="coerce")
    if closes.isna().any() or closes.nunique() != 1:
        return False
    # A changed deadline needs independent confirmation; do not freeze using
    # conflicting provider and schedule snapshots.
    return abs(float(closes.iloc[0]) - float(close_at)) <= 300


def read_valid_cache(path, schedule, date):
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, dtype={"race_id": str, "player_id": str}, low_memory=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return {}
    if frame.empty or "race_id" not in frame:
        return {}
    cached = {}
    for race_id, race in frame.groupby("race_id", sort=False):
        target = schedule.get(str(race_id))
        if target and target.get("date") == date and valid_race(
            race, race_id, date, target["close_at"]
        ):
            cached[str(race_id)] = race
    return cached


def refresh(root=ROOT, clock=now_jst, fetcher=parse_race_page,
            min_seconds_to_close=MIN_SECONDS_TO_CLOSE,
            max_fetches=MAX_FETCHES, max_wall_seconds=MAX_WALL_SECONDS):
    now = clock()
    if now.tzinfo is None:
        raise ValueError("timezone-aware clock required")
    schedule_rows = read_csv(root / "outputs/latest_race_schedule.csv")
    today = now.date().isoformat()
    schedule = {str(x["race_id"]): x for x in schedule_rows if x.get("date") == today}
    if not schedule or len(schedule) != sum(x.get("date") == today for x in schedule_rows):
        raise ValueError("missing or duplicate current-day schedule")
    cached = read_valid_cache(root / INPUT, schedule, today)
    ledger = frozen(root / LEDGER)
    upcoming = sorted(
        (r for r in schedule.values()
         if float(r["close_at"]) > now.timestamp() + min_seconds_to_close
         and str(r["race_id"]) not in ledger
         and str(r["race_id"]) not in cached),
        key=lambda r: (float(r["close_at"]), str(r["race_id"]))
    )
    attempted, failures, fetched = 0, [], 0
    started = time.monotonic()
    for item in upcoming:
        if attempted >= max_fetches or time.monotonic() - started >= max_wall_seconds:
            break
        if float(item["close_at"]) <= clock().timestamp() + min_seconds_to_close:
            continue
        attempted += 1
        rid = str(item["race_id"])
        url = item.get("source_url", "")
        if not str(url).startswith("https://www.winticket.jp/keirin/"):
            failures.append({"race_id": rid, "reason": "invalid_source_url"})
            continue
        try:
            rows, _ = fetcher(url, completeness_attempts=1)
            frame = pd.DataFrame(rows)
            if not valid_race(frame, rid, today, item["close_at"]):
                raise ValueError("mismatched_or_incomplete_racecard")
            if clock().timestamp() + min_seconds_to_close >= min(
                float(item["close_at"]), float(frame["close_at"].iloc[0])
            ):
                raise ValueError("too_close_to_deadline")
            cached[rid] = frame
            fetched += 1
        except Exception as error:
            failures.append({"race_id": rid, "reason": f"{type(error).__name__}: {error}"[:250]})
    path = root / INPUT
    if cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        full = pd.concat(list(cached.values()), ignore_index=True, sort=False)
        full = full.sort_values(["race_id", "car_no"], kind="stable")
        temp = path.with_suffix(".csv.tmp")
        full.to_csv(temp, index=False)
        os.replace(temp, path)
    elif path.exists():
        # Never leave yesterday's cache labeled as today's input.
        path.unlink()
    report = {"date": today, "scheduled": len(schedule), "already_frozen": sum(
        rid in ledger for rid in schedule), "valid_cached": len(cached),
        "newly_fetched": fetched, "attempted": attempted,
        "failed": failures, "unfetched": max(0, len(upcoming) - attempted)}
    print("CONDITIONAL_ENTRY_REFRESH " + json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-fetches", type=int, default=MAX_FETCHES)
    args = parser.parse_args()
    refresh(max_fetches=args.max_fetches)


if __name__ == "__main__":
    main()
