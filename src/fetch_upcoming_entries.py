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
ODDS_HISTORY_FILE = RAW_DIR / "win_odds_history.csv"


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
    """Require every official starter, including missing tail cars such as 8/9."""
    race = frame.loc[frame["race_id"].astype(str).eq(str(race_id))].copy()
    cars = _race_entry_set(frame, race_id)
    if not cars or race.empty:
        return False

    expected_values = pd.to_numeric(race.get("entries_number"), errors="coerce").dropna()
    if len(expected_values):
        expected_count = int(expected_values.max())
    else:
        # Fallback only when the source omits entriesNumber.
        expected_count = max(cars)

    return cars == set(range(1, expected_count + 1))


def append_win_odds_history(entries, captured_at):
    frame = pd.DataFrame(entries)
    needed = ["race_id", "car_no", "player_id", "odds_win", "close_at"]
    if frame.empty or not {"race_id", "car_no", "odds_win"}.issubset(frame.columns):
        return
    for col in needed:
        if col not in frame.columns:
            frame[col] = pd.NA
    snap = frame[needed].copy()
    snap["odds_win"] = pd.to_numeric(snap["odds_win"], errors="coerce")
    snap = snap[snap["odds_win"].gt(0)].copy()
    if snap.empty:
        return
    snap["captured_at"] = float(captured_at)
    snap["seconds_to_close"] = pd.to_numeric(snap["close_at"], errors="coerce") - float(captured_at)
    if ODDS_HISTORY_FILE.exists():
        try:
            old = pd.read_csv(ODDS_HISTORY_FILE, dtype={"race_id": str, "player_id": str})
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            old = pd.DataFrame()
        snap = pd.concat([old, snap], ignore_index=True, sort=False)
    snap = snap.drop_duplicates(["race_id","car_no","captured_at"], keep="last")
    snap.to_csv(ODDS_HISTORY_FILE, index=False)


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

    # Atomic near-close snapshot: never replace today's input with only the
    # subset that happened to fetch successfully. That previously made whole
    # races disappear from prediction/site output during transient source gaps.
    fetched_ids = {str(item.get("race_id")) for item in all_entries}
    scheduled_ids = {str(item.get("race_id")) for item in upcoming.to_dict("records")}
    missing_ids = sorted(scheduled_ids - fetched_ids)
    if failures or missing_ids:
        UPCOMING_COUNT_FILE.write_text("0", encoding="ascii")
        raise ValueError(
            f"refusing partial upcoming snapshot: scheduled={len(scheduled_ids)} "
            f"fetched={len(fetched_ids)} missing={missing_ids} failures={failures[:3]}"
        )

    # Merge refreshed near-close races into the existing full-day snapshot.
    # Never replace TODAY_CSV with only the current 5-40 minute window; doing so
    # made entire races (and therefore their riders) disappear from the site.
    from common import TODAY_CSV, TODAY_ODDS_CSV
    from fetch_today_entries import ODDS_COLUMNS
    fresh_entries = pd.DataFrame(all_entries)
    fresh_odds = pd.DataFrame(all_odds, columns=ODDS_COLUMNS)
    append_win_odds_history(all_entries, datetime.now(ZoneInfo("Asia/Tokyo")).timestamp())
    target_ids = set(fresh_entries["race_id"].astype(str))
    if TODAY_CSV.exists():
        try:
            base_entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str})
            # The near-close refresher is only allowed to patch an already
            # complete daily snapshot. If whole races are absent, rebuilding
            # from a small window would preserve the corruption indefinitely.
            scheduled_day_ids = set(schedule["race_id"].dropna().astype(str))
            base_ids = set(base_entries["race_id"].dropna().astype(str))
            missing_base_ids = sorted(scheduled_day_ids - base_ids)
            if missing_base_ids:
                raise ValueError(
                    f"base daily snapshot incomplete; refusing near-close merge: "
                    f"missing_races={missing_base_ids}"
                )
            base_entries = base_entries[~base_entries["race_id"].astype(str).isin(target_ids)]
            merged_entries = pd.concat([base_entries, fresh_entries], ignore_index=True, sort=False)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"base daily snapshot unreadable; refusing partial rebuild: {exc}") from exc
    else:
        raise ValueError("base daily snapshot missing; refusing near-close-only rebuild")
    if TODAY_ODDS_CSV.exists():
        try:
            base_odds = pd.read_csv(TODAY_ODDS_CSV, dtype={"race_id": str})
            base_odds = base_odds[~base_odds["race_id"].astype(str).isin(target_ids)]
            merged_odds = pd.concat([base_odds, fresh_odds], ignore_index=True, sort=False)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            merged_odds = fresh_odds
    else:
        merged_odds = fresh_odds
    entries, odds, _ = save_today_frames(merged_entries.to_dict("records"), merged_odds.to_dict("records"))
    fetched_races = int(fresh_entries["race_id"].nunique())
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
