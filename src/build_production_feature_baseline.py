import json
from pathlib import Path

import joblib
import pandas as pd

from common import (
    HISTORY_CSV,
    METRICS_PATH,
    MODEL_PATH,
    add_pair_history_features,
    add_player_elo_features,
    add_player_prior_features,
    prepare_features,
)
from model_drift_audit import build_feature_baseline


def main():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError("production model or metrics missing")

    bundle = joblib.load(MODEL_PATH)
    features = list(bundle.get("features") or [])
    fill_values = bundle.get("fill_values") or {}
    metrics = json.loads(Path(METRICS_PATH).read_text(encoding="utf-8"))

    trained_at = pd.Timestamp(metrics.get("trained_at_jst"))
    cutoff = trained_at.tz_localize(None).normalize()

    df = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[df["date"] <= cutoff].copy()
    df = df.sort_values(["date", "race_id", "car_no"], kind="mergesort")
    if df.empty:
        raise ValueError("no history rows at or before production training cutoff")

    df = add_player_prior_features(df)
    df = add_player_elo_features(df)
    df = add_pair_history_features(df)
    X, _ = prepare_features(df, fill_values)
    for col in features:
        if col not in X.columns:
            X[col] = float(fill_values.get(col, 0.0))
    X = X.reindex(columns=features)

    payload = build_feature_baseline(
        X,
        df,
        metadata={
            "baseline_for": "current_production_model",
            "production_trained_at_jst": metrics.get("trained_at_jst"),
            "history_cutoff": str(cutoff.date()),
            "history_races": int(df["race_id"].nunique()),
            "model_feature_count": int(len(features)),
        },
    )
    print(json.dumps({
        "baseline_for": payload["metadata"]["baseline_for"],
        "history_races": payload["metadata"]["history_races"],
        "features": len(payload["features"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
