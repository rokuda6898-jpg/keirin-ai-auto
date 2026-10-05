import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from common import MODEL_PATH as PRODUCTION_MODEL_PATH, MODEL_DIR, RAW_DIR, normalize_race_prob
from live_snapshot_model import (
    LIVE_FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    build_live_matrix_from_snapshot_rows,
)

SNAPSHOT_CSV = RAW_DIR / "live_snapshot_history.csv"

SHADOW_TRAIN_RACES = 500
EXTERNAL_VALIDATION_RACES = 1500
MIN_SHADOW_DATES = 20
MIN_EXTERNAL_DATES = 45
MIN_TEST_CONTEXTS = 100
MIN_EXTERNAL_TEST_CONTEXTS = 300
MIN_DELTA_RATE = 0.015
MIN_NET_HITS = 5


def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y"}
    )


def load_ready_snapshots():
    df = pd.read_csv(SNAPSHOT_CSV, dtype={"race_id": str, "player_id": str})
    required = {
        "race_id", "player_id", "date", "snapshot_bucket",
        "label_finish_pos", "label_available", "model_input_ready",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"live snapshot store missing columns: {sorted(missing)}")

    df["label_available"] = bool_series(df["label_available"])
    df["model_input_ready"] = bool_series(df["model_input_ready"])
    df = df[df["label_available"] & df["model_input_ready"]].copy()
    df["label_finish_pos"] = pd.to_numeric(df["label_finish_pos"], errors="coerce")
    df = df[df["label_finish_pos"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["snapshot_id"] = (
        df["race_id"].astype(str) + "::" + df["snapshot_bucket"].astype(str)
    )

    keep_ids = []
    for snapshot_id, g in df.groupby("snapshot_id", sort=False):
        players = g["player_id"].astype(str).nunique()
        expected = pd.to_numeric(g.get("entries_number"), errors="coerce").dropna()
        expected_count = int(expected.max()) if len(expected) else players
        if players == expected_count and players >= 4:
            keep_ids.append(snapshot_id)
    return df[df["snapshot_id"].isin(keep_ids)].copy()


def split_dates(df):
    dates = sorted(df["date"].dropna().dt.normalize().unique())
    if len(dates) < MIN_SHADOW_DATES:
        return None
    d1 = dates[max(1, min(len(dates)-2, int(len(dates) * 0.70)))]
    d2 = dates[max(2, min(len(dates)-1, int(len(dates) * 0.85)))]
    fit = df[df["date"] < d1].copy()
    calib = df[(df["date"] >= d1) & (df["date"] < d2)].copy()
    test = df[df["date"] >= d2].copy()
    return fit, calib, test, pd.Timestamp(d1), pd.Timestamp(d2)


def model():
    return HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.04,
        random_state=90210,
    )


def score_top1(frame, prob):
    scored = frame[["snapshot_id", "race_id", "snapshot_bucket", "label_finish_pos"]].copy()
    scored["p_raw"] = np.clip(np.asarray(prob, dtype=float), 1e-9, 1.0)
    # normalize_race_prob groups by race_id, so temporarily make each time
    # bucket its own race context.
    scored["_race_id_original"] = scored["race_id"].astype(str)
    scored["race_id"] = scored["snapshot_id"].astype(str)
    scored = normalize_race_prob(scored, "p_raw", "p_win")
    scored["rank"] = scored.groupby("race_id")["p_win"].rank(
        ascending=False, method="first"
    )
    top = scored[scored["rank"].eq(1)].copy()
    top["is_hit"] = pd.to_numeric(top["label_finish_pos"], errors="coerce").eq(1)
    return top


def score_production(frame):
    if not PRODUCTION_MODEL_PATH.exists():
        return None
    bundle = joblib.load(PRODUCTION_MODEL_PATH)
    prod_model = bundle.get("model")
    prod_calibrator = bundle.get("calibrator")
    features = list(bundle.get("features") or [])
    fills = bundle.get("fill_values") or {}
    if prod_model is None or not features:
        return None

    X = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    for col in features:
        X[col] = X[col].fillna(float(fills.get(col, 0.0)))
    raw = prod_model.predict_proba(X)[:, 1]
    prob = prod_calibrator.predict(raw) if prod_calibrator is not None else raw
    return score_top1(frame, prob)


def bucket_metrics(top):
    rows = {}
    if top is None or top.empty:
        return rows
    for bucket, g in top.groupby("snapshot_bucket", sort=False):
        rows[str(bucket)] = {
            "contexts": int(len(g)),
            "hits": int(g["is_hit"].sum()),
            "hit_rate": float(g["is_hit"].mean()),
        }
    return rows


def write_collecting_metrics(df, reason):
    races = int(df["race_id"].nunique()) if len(df) else 0
    dates = int(df["date"].dt.date.nunique()) if len(df) else 0
    payload = {
        "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "status": "collecting",
        "reason": reason,
        "labeled_model_ready_races": races,
        "labeled_model_ready_dates": dates,
        "shadow_training_floor_races": SHADOW_TRAIN_RACES,
        "external_validation_floor_races": EXTERNAL_VALIDATION_RACES,
        "production_auto_promotion": False,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-races", type=int, default=SHADOW_TRAIN_RACES)
    args = parser.parse_args()

    if not SNAPSHOT_CSV.exists() or not SNAPSHOT_CSV.stat().st_size:
        write_collecting_metrics(pd.DataFrame(), "snapshot store unavailable")
        return

    df = load_ready_snapshots()
    races = int(df["race_id"].nunique()) if len(df) else 0
    dates = int(df["date"].dt.date.nunique()) if len(df) else 0
    minimum = max(int(args.min_races), SHADOW_TRAIN_RACES)
    if races < minimum or dates < MIN_SHADOW_DATES:
        write_collecting_metrics(
            df,
            f"need >= {minimum} labeled model-ready races and >= {MIN_SHADOW_DATES} dates",
        )
        return

    split = split_dates(df)
    if split is None:
        write_collecting_metrics(df, "insufficient chronological dates")
        return
    fit, calib, test, calibration_start, test_start = split

    fit_contexts = int(fit["snapshot_id"].nunique())
    calib_contexts = int(calib["snapshot_id"].nunique())
    test_contexts = int(test["snapshot_id"].nunique())
    if test_contexts < MIN_TEST_CONTEXTS:
        write_collecting_metrics(
            df, f"need >= {MIN_TEST_CONTEXTS} held-out snapshot contexts"
        )
        return

    X_fit, fills = build_live_matrix_from_snapshot_rows(fit)
    X_calib, _ = build_live_matrix_from_snapshot_rows(calib, fills)
    X_test, _ = build_live_matrix_from_snapshot_rows(test, fills)
    y_fit = pd.to_numeric(fit["label_finish_pos"], errors="coerce").eq(1).astype(int)
    y_calib = pd.to_numeric(calib["label_finish_pos"], errors="coerce").eq(1).astype(int)

    live_model = model()
    live_model.fit(X_fit, y_fit)

    calibrator = None
    raw_calib = live_model.predict_proba(X_calib)[:, 1] if len(X_calib) else np.array([])
    if len(raw_calib) >= 100 and y_calib.nunique() >= 2 and np.isfinite(raw_calib).all():
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_calib, y_calib)

    raw_test = live_model.predict_proba(X_test)[:, 1]
    live_prob = calibrator.predict(raw_test) if calibrator is not None else raw_test
    live_top = score_top1(test, live_prob)
    prod_top = score_production(test)

    production_rate = float(prod_top["is_hit"].mean()) if prod_top is not None and len(prod_top) else None
    live_rate = float(live_top["is_hit"].mean()) if len(live_top) else None
    production_hits = int(prod_top["is_hit"].sum()) if prod_top is not None and len(prod_top) else 0
    live_hits = int(live_top["is_hit"].sum()) if len(live_top) else 0
    comparable_contexts = min(len(live_top), len(prod_top)) if prod_top is not None else 0
    delta = (
        float(live_rate - production_rate)
        if live_rate is not None and production_rate is not None else None
    )
    net_hits = live_hits - production_hits

    # Fit the shadow artifact on every currently labeled real snapshot. The
    # held-out metrics above remain untouched and govern eligibility.
    X_full, full_fills = build_live_matrix_from_snapshot_rows(df)
    y_full = pd.to_numeric(df["label_finish_pos"], errors="coerce").eq(1).astype(int)
    final_model = model()
    final_model.fit(X_full, y_full)

    joblib.dump(
        {
            "model": final_model,
            "calibrator": calibrator,
            "features": LIVE_FEATURES,
            "fill_values": full_fills,
            "training_scope": "real_pre_race_model_ready_snapshots",
            "trained_races": races,
            "trained_dates": dates,
        },
        MODEL_PATH,
    )

    external_ready = (
        races >= EXTERNAL_VALIDATION_RACES
        and dates >= MIN_EXTERNAL_DATES
        and test_contexts >= MIN_EXTERNAL_TEST_CONTEXTS
        and delta is not None
        and delta >= MIN_DELTA_RATE
        and net_hits >= MIN_NET_HITS
    )
    status = (
        "eligible_for_prospective_audit"
        if external_ready else "shadow_model_ready"
    )

    payload = {
        "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "status": status,
        "labeled_model_ready_races": races,
        "labeled_model_ready_dates": dates,
        "snapshot_contexts": int(df["snapshot_id"].nunique()),
        "fit_contexts": fit_contexts,
        "calibration_contexts": calib_contexts,
        "test_contexts": test_contexts,
        "calibration_start": str(calibration_start.date()),
        "test_start": str(test_start.date()),
        "probability_calibration": {
            "active": calibrator is not None,
            "method": "isotonic_regression" if calibrator is not None else "none",
        },
        "production_test_hits": production_hits,
        "production_test_hit_rate": production_rate,
        "live_snapshot_test_hits": live_hits,
        "live_snapshot_test_hit_rate": live_rate,
        "delta_hit_rate": delta,
        "delta_pp": None if delta is None else delta * 100.0,
        "net_hits": net_hits,
        "comparable_test_contexts": int(comparable_contexts),
        "live_bucket_metrics": bucket_metrics(live_top),
        "production_bucket_metrics": bucket_metrics(prod_top),
        "shadow_training_floor_races": SHADOW_TRAIN_RACES,
        "external_validation_floor_races": EXTERNAL_VALIDATION_RACES,
        "external_validation_floor_test_contexts": MIN_EXTERNAL_TEST_CONTEXTS,
        "promotion_rule": "never auto-promote; after >=1500 real races and retrospective gain, require >=300 prospective same-race official comparisons plus external validation",
        "production_auto_promotion": False,
    }
    METRICS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
