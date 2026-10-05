import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import HISTORY_CSV, OUTPUT_DIR, RAW_DIR, TODAY_CSV, ensure_dirs

SNAPSHOT_CSV = RAW_DIR / "live_snapshot_history.csv"
SUMMARY_JSON = OUTPUT_DIR / "live_snapshot_learning_summary.json"

TARGET_SECONDS = {
    "morning": 4 * 3600,
    "40m": 40 * 60,
    "20m": 20 * 60,
    "10m": 10 * 60,
}


def snapshot_bucket(seconds_to_close):
    value = pd.to_numeric(seconds_to_close, errors="coerce")
    if pd.isna(value) or value <= 300:
        return None
    value = float(value)
    if value >= 7200:
        return "morning"
    if 1800 <= value <= 3000:
        return "40m"
    if 900 <= value <= 1500:
        return "20m"
    if 420 <= value <= 780:
        return "10m"
    return None


def _read_snapshots():
    try:
        return pd.read_csv(SNAPSHOT_CSV, dtype={"race_id": str, "player_id": str})
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _distance_to_target(row):
    bucket = str(row.get("snapshot_bucket", ""))
    target = TARGET_SECONDS.get(bucket)
    seconds = pd.to_numeric(row.get("seconds_to_close"), errors="coerce")
    if target is None or pd.isna(seconds):
        return float("inf")
    return abs(float(seconds) - float(target))


def capture():
    ensure_dirs()
    if not TODAY_CSV.exists():
        return 0
    today = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
    if today.empty or "race_id" not in today.columns:
        return 0

    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    close_at = pd.to_numeric(today.get("close_at"), errors="coerce")
    today["seconds_to_close"] = close_at - now.timestamp()
    today["snapshot_bucket"] = today["seconds_to_close"].map(snapshot_bucket)
    snap = today[today["snapshot_bucket"].notna()].copy()
    if snap.empty:
        print("no races in live learning snapshot windows")
        return 0

    snap["captured_at_jst"] = now.isoformat(timespec="seconds")
    snap["captured_at_epoch"] = now.timestamp()
    snap["snapshot_source"] = "real_pre_race_feed"
    if "finish_pos" in snap.columns:
        snap = snap.drop(columns=["finish_pos"])
    snap["label_finish_pos"] = np.nan
    snap["label_available"] = False
    snap["_distance"] = snap.apply(_distance_to_target, axis=1)

    old = _read_snapshots()
    if len(old):
        if "_distance" not in old.columns:
            old["_distance"] = old.apply(_distance_to_target, axis=1)
        combined = pd.concat([old, snap], ignore_index=True, sort=False)
    else:
        combined = snap

    keys = ["race_id", "player_id", "snapshot_bucket"]
    combined["race_id"] = combined["race_id"].astype(str)
    combined["player_id"] = combined["player_id"].astype(str)
    combined = combined.sort_values(
        ["race_id", "player_id", "snapshot_bucket", "_distance", "captured_at_epoch"],
        ascending=[True, True, True, True, False],
        kind="mergesort",
    )
    combined = combined.drop_duplicates(keys, keep="first")
    combined = combined.drop(columns=["_distance"], errors="ignore")
    combined.to_csv(SNAPSHOT_CSV, index=False)
    print(
        f"captured live training snapshots rows={len(snap)} "
        f"stored_rows={len(combined)} stored_races={combined['race_id'].nunique()}"
    )
    return int(len(snap))


def label_from_history():
    ensure_dirs()
    snapshots = _read_snapshots()
    if snapshots.empty or not HISTORY_CSV.exists():
        write_summary(snapshots)
        return 0

    history = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    if history.empty or not {"race_id", "player_id", "finish_pos"}.issubset(history.columns):
        write_summary(snapshots)
        return 0

    labels = history[["race_id", "player_id", "finish_pos"]].copy()
    labels["race_id"] = labels["race_id"].astype(str)
    labels["player_id"] = labels["player_id"].astype(str)
    labels["finish_pos"] = pd.to_numeric(labels["finish_pos"], errors="coerce")
    labels = labels[labels["finish_pos"].notna()].drop_duplicates(
        ["race_id", "player_id"], keep="last"
    )

    base = snapshots.drop(columns=["label_finish_pos", "label_available"], errors="ignore")
    merged = base.merge(labels, on=["race_id", "player_id"], how="left")
    merged = merged.rename(columns={"finish_pos": "label_finish_pos"})
    merged["label_available"] = pd.to_numeric(
        merged["label_finish_pos"], errors="coerce"
    ).notna()
    merged.to_csv(SNAPSHOT_CSV, index=False)
    labeled = int(merged["label_available"].sum())
    write_summary(merged)
    print(
        f"labeled live snapshots rows={labeled}/{len(merged)} "
        f"labeled_races={merged.loc[merged['label_available'], 'race_id'].nunique()}"
    )
    return labeled


def write_summary(frame=None):
    if frame is None:
        frame = _read_snapshots()
    if frame.empty:
        payload = {
            "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "rows": 0,
            "races": 0,
            "labeled_rows": 0,
            "labeled_races": 0,
            "snapshot_buckets": {},
        }
    else:
        available = (
            frame["label_available"].fillna(False).astype(bool)
            if "label_available" in frame.columns
            else pd.Series(False, index=frame.index)
        )
        payload = {
            "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "rows": int(len(frame)),
            "races": int(frame["race_id"].astype(str).nunique()),
            "labeled_rows": int(available.sum()),
            "labeled_races": int(frame.loc[available, "race_id"].astype(str).nunique()),
            "snapshot_buckets": {
                str(k): int(v)
                for k, v in frame["snapshot_bucket"].fillna("unknown").astype(str).value_counts().items()
            },
            "policy": "one closest real pre-race snapshot per race/player/time bucket; labels joined only after official history is available",
        }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--label", action="store_true")
    args = parser.parse_args()

    if not args.capture and not args.label:
        args.capture = True
        args.label = True
    if args.capture:
        capture()
    if args.label:
        label_from_history()
    else:
        write_summary()


if __name__ == "__main__":
    main()
