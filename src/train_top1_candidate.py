import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from common import (
    HISTORY_CSV,
    MODEL_PATH,
    FEATURE_COLS,
    add_player_prior_features,
    add_player_elo_features,
    add_pair_history_features,
    ensure_dirs,
    prepare_features,
)


def main():
    ensure_dirs()

    if not HISTORY_CSV.exists() or not HISTORY_CSV.stat().st_size:
        raise FileNotFoundError("data/raw/history.csv is missing or empty")

    df = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    required = {"race_id", "date", "finish_pos", "car_no"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history.csv missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    finish = pd.to_numeric(df["finish_pos"], errors="coerce")
    df = df[finish.notna()].copy()
    df["target_win"] = finish[finish.notna()].eq(1).astype(int)
    df = df.sort_values(["date", "race_id", "car_no"])

    # Same leakage-safe pre-race feature pipeline used by production training.
    df = add_player_prior_features(df)
    df = add_player_elo_features(df)
    df = add_pair_history_features(df)

    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 20:
        raise ValueError("not enough chronological dates for candidate calibration")
    split_idx = max(1, min(len(dates) - 1, int(len(dates) * 0.85)))
    calibration_start = pd.Timestamp(dates[split_idx])

    fit_df = df[df["date"] < calibration_start].copy()
    calibration_df = df[df["date"] >= calibration_start].copy()
    if fit_df["race_id"].nunique() < 1000 or calibration_df["race_id"].nunique() < 200:
        raise ValueError(
            "insufficient chronological races for leakage-safe candidate calibration"
        )

    X_fit, fill_values = prepare_features(fit_df)
    y_fit = fit_df["target_win"].astype(int)
    X_calibration, _ = prepare_features(calibration_df, fill_values)
    y_calibration = calibration_df["target_win"].astype(int)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )
    model.fit(X_fit, y_fit)

    calibration_raw = model.predict_proba(X_calibration)[:, 1]
    calibrator = None
    if y_calibration.nunique() >= 2 and np.isfinite(calibration_raw).all():
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calibration_raw, y_calibration)

    metrics = {
        "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "training_scope": "chronological_fit_plus_holdout_calibration_top1_candidate",
        "n_rows": int(len(df)),
        "n_races": int(df["race_id"].nunique()),
        "fit_rows": int(len(fit_df)),
        "fit_races": int(fit_df["race_id"].nunique()),
        "calibration_rows": int(len(calibration_df)),
        "calibration_races": int(calibration_df["race_id"].nunique()),
        "calibration_start": str(calibration_start.date()),
        "probability_calibration": {
            "active": calibrator is not None,
            "method": "isotonic_regression" if calibrator is not None else "none",
            "fit_scope": "latest_15pct_chronological_dates",
        },
        "features": FEATURE_COLS,
        "note": "Fast winner candidate with chronology-safe held-out probability calibration. Promotion still requires the later external/live-parity gate.",
    }

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "fill_values": fill_values,
        "features": FEATURE_COLS,
        "metrics": metrics,
    }

    candidate_path = MODEL_PATH.parent / "candidate_win_model.joblib"
    metrics_path = MODEL_PATH.parent / "candidate_fast_metrics.json"
    joblib.dump(bundle, candidate_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved fast top1 candidate: {candidate_path}")


if __name__ == "__main__":
    main()
