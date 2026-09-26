import argparse
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from common import RAW_DIR, ensure_dirs
from fetch_today_entries import RACE_SCHEDULE_CSV, parse_race_page, save_today_frames

UPCOMING_COUNT_FILE = RAW_DIR / "upcoming_races_count.txt"
UPCOMING_METADATA_FILE = RAW_DIR / "upcoming_entries_metadata.json"


def select_upcoming_races(schedule, now_epoch, min_minutes=5, max_minutes=40):
    frame = schedule.copy()
    frame["close_at"] = pd.to_numeric(frame.get("close_at"), errors="coerce")
    seconds_to_close = frame["close_at"] - float(now_epoch)
    return frame[
        seconds_to_close.ge(float(min_minutes) * 60)
        & seconds_to_close.le(float(max_minutes) * 60)
    ].copy()


def _race_entry_set(frame, race_id):
    cars = pd.to_numeric(
        frame.loc[frame["race_id"].astype(str).eq(str(race_id)), "car_no"],
        errors="coerce",
    ).dropna().astype(int)
    return set(cars.tolist())


def _entries_complete(frame, race_id):
    """A fetched race must contain a contiguous official car-number field."""
    cars = _race_entry_set(frame, race_id)
    if not cars:
        return False
    return cars == set(range(1, max(cars) + 1))


def fetch_upcoming(min_minutes=5, max_minutes=40, sleep_sec=0.2, retry_sec=5.0):
    ensure_dirs()
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    if not RACE_SCHEDULE_CSV.exists():
        UPCOMING_COUNT_FILE.write_text("0", encoding="ascii")
        print(f"schedule not found: {RACE_SCHEDULE_CSV}")
        return 0

    schedule = pd.read_csv(RACE_SCHEDULE_CSV, dtype={"race_id": str})
    schedule = schedule[schedule["date"].astype(str).eq(now.strftime("%Y-%m-%d"))]
    upcoming = select_upcoming_races(schedule, now.timestamp(), min_minutes, max_minutes)
    UPCOMING_COUNT_FILE.write_text(str(len(upcoming)), encoding="ascii")
    if upcoming.empty:
        print("no races in the near-close window")
        return 0

    all_entries = []
    all_odds = []
    failures = []
    for row in upcoming.to_dict("records"):
        url = row.get("source_url")
        race_id = str(row["race_id"])
        last_error = None
        # Do not pass a partially fetched field (for example 1,2,5,6,7)
        # into prediction. Retry every 5 seconds until the official field is
        # complete or the race is too close to safely refresh.
        while True:
            try:
                entries, odds = parse_race_page(url)
                race_entries = [item for item in entries if str(item.get("race_id")) == race_id]
                check = pd.DataFrame(race_entries)
                if len(check) and _entries_complete(check, race_id):
                    all_entries.extend(race_entries)
                    all_odds.extend(item for item in odds if str(item.get("race_id")) == race_id)
                    break
                last_error = f"incomplete field: {sorted(_race_entry_set(check, race_id)) if len(check) else []}"
            except Exception as error:
                last_error = str(error)

            seconds_left = float(row.get("close_at", 0) or 0) - datetime.now(ZoneInfo("Asia/Tokyo")).timestamp()
            if seconds_left <= 300:
                failures.append({"race_id": race_id, "url": url, "error": last_error or "incomplete field"})
                break
            time.sleep(retry_sec)
        time.sleep(sleep_sec)

    if not all_entries:
        UPCOMING_COUNT_FILE.write_text("0", encoding="ascii")
        raise ValueError(f"failed to fetch upcoming entries: {failures[:3]}")

    entries, odds, _ = save_today_frames(all_entries, all_odds)
    fetched_races = int(entries["race_id"].nunique())
    UPCOMING_COUNT_FILE.write_text(str(fetched_races), encoding="ascii")
    metadata = {
        "fetched_at_jst": now.isoformat(timespec="seconds"),
        "min_minutes_to_close": min_minutes,
        "max_minutes_to_close": max_minutes,
        "scheduled_races": int(len(upcoming)),
        "fetched_races": fetched_races,
        "entry_rows": int(len(entries)),
        "odds_rows": int(len(odds)),
        "failures": failures,
    }
    UPCOMING_METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return fetched_races


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-minutes", type=float, default=5)
    parser.add_argument("--max-minutes", type=float, default=40)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument("--retry-sec", type=float, default=5.0)
    args = parser.parse_args()
    return fetch_upcoming(args.min_minutes, args.max_minutes, args.sleep_sec, args.retry_sec)


if __name__ == "__main__":
    main()
