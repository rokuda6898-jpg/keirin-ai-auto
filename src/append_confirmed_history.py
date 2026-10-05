import argparse
import json
import time
from pathlib import Path

import pandas as pd

from build_history import parse_history_race
from common import HISTORY_CSV, OUTPUT_DIR, RAW_DIR, ensure_dirs


TODAY_CSV = RAW_DIR / "today_entries.csv"


def complete_race_ids(history):
    if history.empty or "race_id" not in history.columns or "official_finish_pos" not in history.columns:
        return set()

    work = history[["race_id", "official_finish_pos"]].copy()
    work["race_id"] = work["race_id"].astype(str)
    work["official_finish_pos"] = pd.to_numeric(work["official_finish_pos"], errors="coerce")

    complete = set()
    for race_id, group in work.groupby("race_id", sort=False):
        orders = set(group["official_finish_pos"].dropna().astype(int).tolist())
        if {1, 2, 3}.issubset(orders):
            complete.add(str(race_id))
    return complete


def append_confirmed_today(history_path=HISTORY_CSV, today_path=TODAY_CSV, sleep_sec=0.0):
    ensure_dirs()
    history_path = Path(history_path)
    today_path = Path(today_path)

    if not today_path.exists():
        raise FileNotFoundError(f"today entries not found: {today_path}")

    today = pd.read_csv(today_path, dtype={"race_id": str, "player_id": str})
    required = {"race_id", "source_url"}
    missing = required.difference(today.columns)
    if missing:
        raise ValueError(f"today entries missing columns: {sorted(missing)}")

    races = (
        today[["race_id", "source_url"]]
        .dropna(subset=["race_id", "source_url"])
        .astype({"race_id": str, "source_url": str})
        .drop_duplicates("race_id", keep="last")
    )

    try:
        history = pd.read_csv(history_path, dtype={"race_id": str, "player_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        history = pd.DataFrame()

    already_complete = complete_race_ids(history)
    candidates = races[~races["race_id"].isin(already_complete)].copy()

    confirmed_frames = []
    pending = []
    failed = []

    for race_id, source_url in candidates.itertuples(index=False):
        try:
            rows, _ = parse_history_race(source_url, use_cache=False)
            race = pd.DataFrame(rows)
            if race.empty:
                pending.append(str(race_id))
                continue

            race["race_id"] = race["race_id"].astype(str)
            orders = pd.to_numeric(race.get("official_finish_pos"), errors="coerce")
            official_top3 = set(orders.dropna().astype(int).tolist())
            if not {1, 2, 3}.issubset(official_top3):
                pending.append(str(race_id))
                continue

            # Source data is authoritative for the race id returned by the race page.
            # Replace any previously incomplete copy of that race atomically below.
            confirmed_frames.append(race)
            print(f"central history confirmed: race_id={race_id} rows={len(race)}")
        except Exception as exc:
            failed.append({"race_id": str(race_id), "source_url": source_url, "error": str(exc)})
            print(f"central history failed: race_id={race_id} error={exc}")

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if confirmed_frames:
        incoming = pd.concat(confirmed_frames, ignore_index=True, sort=False)
        incoming["race_id"] = incoming["race_id"].astype(str)
        confirmed_ids = set(incoming["race_id"].dropna().astype(str))

        if not history.empty:
            history["race_id"] = history["race_id"].astype(str)
            history = history[~history["race_id"].isin(confirmed_ids)]

        combined = pd.concat([history, incoming], ignore_index=True, sort=False)
        if {"race_id", "player_id"}.issubset(combined.columns):
            combined["player_id"] = combined["player_id"].astype(str)
            combined = combined.drop_duplicates(["race_id", "player_id"], keep="last")

        sort_cols = [c for c in ["date", "venue", "race_no", "car_no"] if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols, kind="mergesort")

        history_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = history_path.with_suffix(history_path.suffix + ".tmp")
        combined.to_csv(tmp_path, index=False)
        tmp_path.replace(history_path)
        history = combined

    summary = {
        "history_path": str(history_path),
        "today_races": int(len(races)),
        "already_complete": int(len(already_complete.intersection(set(races["race_id"])))),
        "checked": int(len(candidates)),
        "appended_races": int(
            pd.concat(confirmed_frames, ignore_index=True)["race_id"].astype(str).nunique()
            if confirmed_frames
            else 0
        ),
        "pending_races": len(pending),
        "failed_races": len(failed),
        "history_races": int(history["race_id"].nunique()) if not history.empty and "race_id" in history.columns else 0,
        "history_rows": int(len(history)),
        "pending_race_ids": pending,
        "failures": failed,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "central_history_append_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Append only today's officially confirmed races to the central history.csv."
    )
    parser.add_argument("--history", default=str(HISTORY_CSV))
    parser.add_argument("--today", default=str(TODAY_CSV))
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    args = parser.parse_args()
    append_confirmed_today(
        history_path=args.history,
        today_path=args.today,
        sleep_sec=args.sleep_sec,
    )


if __name__ == "__main__":
    main()
