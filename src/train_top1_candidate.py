import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

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

    X_full, fill_values = prepare_features(df)
    y_full = df["target_win"].astype(int)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )
    model.fit(X_full, y_full)

    metrics = {
        "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "training_scope": "full_history_top1_candidate_fast",
        "n_rows": int(len(df)),
        "n_races": int(df["race_id"].nunique()),
        "features": FEATURE_COLS,
        "note": "Fast full-history winner candidate. Promotion still requires the later external/live-parity gate.",
    }

    bundle = {
        "model": model,
        "calibrator": None,
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
