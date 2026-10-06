import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from common import FEATURE_COLS, HISTORY_CSV, MODEL_PATH, OUTPUT_DIR, RAW_DIR, TODAY_CSV, ensure_dirs, prepare_features

SNAPSHOT_CSV = RAW_DIR / "live_snapshot_history.csv"
FEATURE_BASE_CSV = RAW_DIR / "live_feature_base.csv"
FEATURE_BASE_META = RAW_DIR / "live_feature_base.meta.json"
SUMMARY_JSON = OUTPUT_DIR / "live_snapshot_learning_summary.json"

HISTORY_FEATURE_COLUMNS = [
    "player_prior_races",
    "player_prior_win_rate",
    "player_prior_place2_rate",
    "player_prior_place3_rate",
    "player_prior_avg_finish",
    "player_recent5_avg_finish",
    "player_recent10_avg_finish",
    "player_recent5_win_rate",
    "player_recent10_win_rate",
    "player_form_trend_5_vs_10",
    "meeting_prior_races",
    "meeting_prior_avg_finish",
    "meeting_form_delta",
    "meeting_finish_trend",
    "player_prior_days_since_last_race",
    "player_prior_strength",
    "player_prior_strength_rank",
    "player_prior_strength_gap_to_best",
    "player_prior_strength_vs_field",
    "player_elo",
    "player_recent_weighted_finish",
    "player_elo_rank",
    "player_elo_vs_field",
    "h2h_prior_meetings",
    "h2h_prior_win_share",
    "line_pair_prior_races",
    "line_pair_second_win_rate",
]

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


def _feature_key(frame):
    work = frame.copy()
    return (
        work["race_id"].astype(str)
        + "::" + work["player_id"].astype(str)
        + "::" + pd.to_numeric(work.get("car_no"), errors="coerce").fillna(-1).astype(int).astype(str)
    )


def _feature_fingerprint(frame):
    # These are the inputs consumed by the chronological history enrichers.
    text = ["date", "venue", "race_id", "player_id", "meeting_id"]
    numeric = ["race_no", "car_no", "line_id", "line_position", "finish_pos"]
    normalized = frame.reindex(columns=text + numeric).copy()
    for col in text:
        normalized[col] = normalized[col].fillna("").astype(str)
    for col in numeric:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce").astype(float)
    normalized = normalized.sort_values(["race_id", "car_no", "player_id"], kind="mergesort")
    digest = hashlib.sha256(pd.util.hash_pandas_object(normalized, index=False).values.tobytes())
    for path in [HISTORY_CSV, Path(__file__).with_name("common.py"), Path(__file__).with_name("predict.py")]:
        if not path.exists():
            return None
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _load_feature_base(today):
    try:
        meta = json.loads(FEATURE_BASE_META.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != _feature_fingerprint(today):
            return None
        base = pd.read_csv(
            FEATURE_BASE_CSV,
            dtype={"race_id": str, "player_id": str},
        )
    except (FileNotFoundError, OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    required = {"race_id", "player_id", "car_no", *HISTORY_FEATURE_COLUMNS}
    if base.empty or not required.issubset(base.columns):
        return None
    current_keys = set(_feature_key(today))
    base_keys = set(_feature_key(base))
    if not current_keys.issubset(base_keys):
        return None
    cols = ["race_id", "player_id", "car_no", *HISTORY_FEATURE_COLUMNS]
    return base[cols].drop_duplicates(["race_id", "player_id", "car_no"], keep="last")


def _apply_feature_base(today, base):
    out = today.copy()
    for col in HISTORY_FEATURE_COLUMNS:
        out = out.drop(columns=[col], errors="ignore")
    out["race_id"] = out["race_id"].astype(str)
    out["player_id"] = out["player_id"].astype(str)
    out["_car_merge"] = pd.to_numeric(out.get("car_no"), errors="coerce")
    base = base.copy()
    base["race_id"] = base["race_id"].astype(str)
    base["player_id"] = base["player_id"].astype(str)
    base["_car_merge"] = pd.to_numeric(base.get("car_no"), errors="coerce")
    base = base.drop(columns=["car_no"], errors="ignore")
    return out.merge(
        base,
        on=["race_id", "player_id", "_car_merge"],
        how="left",
    ).drop(columns=["_car_merge"], errors="ignore")


def _save_feature_base(enriched):
    cols = ["race_id", "player_id", "car_no", *[
        col for col in HISTORY_FEATURE_COLUMNS if col in enriched.columns
    ]]
    if not {"race_id", "player_id", "car_no"}.issubset(enriched.columns):
        return
    enriched[cols].drop_duplicates(
        ["race_id", "player_id", "car_no"], keep="last"
    ).to_csv(FEATURE_BASE_CSV, index=False)
    dates=pd.to_datetime(enriched.get('date'),errors='coerce')
    cutoff=dates.min() if isinstance(dates,pd.Series) else pd.NaT
    FEATURE_BASE_META.write_text(json.dumps({"fingerprint": _feature_fingerprint(enriched),
        "history_policy":"strictly_before_earliest_prediction_date",
        "history_cutoff_exclusive":str(cutoff.date()) if pd.notna(cutoff) else None}), encoding="utf-8")


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


def capture_frame(frame, prepared_features=None, captured_at=None):
    """Persist the exact pre-race feature state used by prediction.

    prepared_features should be the numeric matrix returned by prepare_features
    before it is narrowed to an older production model schema. These values are
    preferred over legacy raw snapshots so future live-only training sees the
    same feature state that production inference saw.
    """
    ensure_dirs()
    if frame is None or len(frame) == 0 or "race_id" not in frame.columns:
        return 0

    now = captured_at or datetime.now(ZoneInfo("Asia/Tokyo"))
    snap = frame.copy()
    close_at = pd.to_numeric(snap.get("close_at"), errors="coerce")
    snap["seconds_to_close"] = close_at - now.timestamp()
    snap["snapshot_bucket"] = snap["seconds_to_close"].map(snapshot_bucket)
    snap = snap[snap["snapshot_bucket"].notna()].copy()
    if snap.empty:
        print("no races in live learning snapshot windows")
        return 0

    if prepared_features is not None and len(prepared_features) == len(frame):
        prepared = prepared_features.copy()
        prepared = prepared.reindex(frame.index)
        prepared = prepared.loc[snap.index]
        for col in FEATURE_COLS:
            if col in prepared.columns:
                snap[col] = pd.to_numeric(prepared[col], errors="coerce")
        snap["model_input_ready"] = True
        snap["model_feature_version"] = "prepared_feature_cols_v1"
    else:
        snap["model_input_ready"] = False
        snap["model_feature_version"] = "raw_feed_only"

    snap["snapshot_bucket_code"] = snap["snapshot_bucket"].map(
        {"morning": 0.0, "40m": 1.0, "20m": 2.0, "10m": 3.0}
    )
    snap["captured_at_jst"] = now.isoformat(timespec="seconds")
    snap["captured_at_epoch"] = now.timestamp()
    snap["snapshot_source"] = "real_pre_race_feed"
    if "finish_pos" in snap.columns:
        snap = snap.drop(columns=["finish_pos"])
    if "official_finish_pos" in snap.columns:
        snap = snap.drop(columns=["official_finish_pos"])
    snap["label_finish_pos"] = np.nan
    snap["label_available"] = False
    snap["_distance"] = snap.apply(_distance_to_target, axis=1)
    snap["_feature_priority"] = snap["model_input_ready"].fillna(False).astype(bool).astype(int)

    old = _read_snapshots()
    if len(old):
        if "_distance" not in old.columns:
            old["_distance"] = old.apply(_distance_to_target, axis=1)
        if "model_input_ready" not in old.columns:
            old["model_input_ready"] = False
        old["_feature_priority"] = old["model_input_ready"].fillna(False).astype(bool).astype(int)
        combined = pd.concat([old, snap], ignore_index=True, sort=False)
    else:
        combined = snap

    keys = ["race_id", "player_id", "snapshot_bucket"]
    combined["race_id"] = combined["race_id"].astype(str)
    combined["player_id"] = combined["player_id"].astype(str)
    combined = combined.sort_values(
        ["race_id", "player_id", "snapshot_bucket", "_feature_priority", "_distance", "captured_at_epoch"],
        ascending=[True, True, True, False, True, False],
        kind="mergesort",
    )
    combined = combined.drop_duplicates(keys, keep="first")
    combined = combined.drop(columns=["_distance", "_feature_priority"], errors="ignore")
    combined.to_csv(SNAPSHOT_CSV, index=False)
    ready = combined.get("model_input_ready", pd.Series(False, index=combined.index))
    print(
        f"captured live training snapshots rows={len(snap)} "
        f"stored_rows={len(combined)} stored_races={combined['race_id'].nunique()} "
        f"model_ready_rows={int(ready.fillna(False).astype(bool).sum())}"
    )
    return int(len(snap))


def capture():
    """Legacy-compatible capture; enrich to the production input state when possible."""
    ensure_dirs()
    if not TODAY_CSV.exists():
        return 0
    today = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
    if today.empty or "race_id" not in today.columns:
        return 0

    try:
        # Lazy import avoids a module cycle: predict does not need this module
        # for ordinary inference, while snapshot collection reuses the exact
        # same live enrichment functions.
        from predict import add_live_odds_movement, add_today_prior_features
        cached = _load_feature_base(today)
        if cached is not None:
            enriched = _apply_feature_base(today, cached)
            print("using cached daily historical live features")
        else:
            enriched = add_today_prior_features(today)
            _save_feature_base(enriched)
            print(f"built live historical feature base: {FEATURE_BASE_CSV}")
        enriched = add_live_odds_movement(enriched)
        fill_values = None
        if MODEL_PATH.exists():
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict):
                fill_values = bundle.get("fill_values") or None
        X, _ = prepare_features(enriched, fill_values)
        return capture_frame(enriched, X)
    except Exception as exc:
        print(f"enriched live snapshot capture unavailable; raw fallback: {exc}")
        return capture_frame(today, None)


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
        ready = (
            frame["model_input_ready"].fillna(False).astype(bool)
            if "model_input_ready" in frame.columns
            else pd.Series(False, index=frame.index)
        )
        labeled_ready = available & ready
        bucket_races = (
            frame.groupby(frame["snapshot_bucket"].fillna("unknown"))["race_id"]
            .nunique()
            .to_dict()
            if "snapshot_bucket" in frame.columns else {}
        )
        labeled_dates = pd.to_datetime(
            frame.loc[labeled_ready, "date"], errors="coerce"
        ).dropna()
        labeled_ready_races = int(
            frame.loc[labeled_ready, "race_id"].astype(str).nunique()
        )
        if labeled_ready_races >= 1500 and labeled_dates.dt.date.nunique() >= 45:
            readiness = "external_validation_ready"
        elif labeled_ready_races >= 500 and labeled_dates.dt.date.nunique() >= 20:
            readiness = "shadow_training_ready"
        else:
            readiness = "collecting"
        payload = {
            "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "rows": int(len(frame)),
            "races": int(frame["race_id"].astype(str).nunique()),
            "model_input_ready_rows": int(ready.sum()),
            "model_input_ready_races": int(frame.loc[ready, "race_id"].astype(str).nunique()),
            "labeled_rows": int(available.sum()),
            "labeled_races": int(frame.loc[available, "race_id"].astype(str).nunique()),
            "labeled_model_ready_rows": int(labeled_ready.sum()),
            "labeled_model_ready_races": labeled_ready_races,
            "labeled_model_ready_dates": int(labeled_dates.dt.date.nunique()),
            "readiness": readiness,
            "shadow_training_floor_races": 500,
            "external_validation_floor_races": 1500,
            "snapshot_buckets": {
                str(k): int(v)
                for k, v in frame["snapshot_bucket"].fillna("unknown").astype(str).value_counts().items()
            },
            "snapshot_bucket_races": {str(k): int(v) for k, v in bucket_races.items()},
            "policy": "one closest model-ready real pre-race snapshot per race/player/time bucket; labels joined only after official history is available",
        }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--capture-raw", action="store_true")
    parser.add_argument("--label", action="store_true")
    args = parser.parse_args()

    if args.capture_raw:
        if TODAY_CSV.exists():
            capture_frame(pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str}), None)
        write_summary()
        return

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
