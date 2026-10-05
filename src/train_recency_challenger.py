import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (
    FEATURE_COLS,
    HISTORY_CSV,
    MODEL_DIR,
    add_pair_history_features,
    add_player_elo_features,
    add_player_prior_features,
    ensure_dirs,
    normalize_race_prob,
    prepare_features,
)

RECENCY_MODEL_PATH = MODEL_DIR / "recency_win_model.joblib"
RECENCY_METRICS_PATH = MODEL_DIR / "recency_model_metrics.json"


def race_top1_rate(frame, prob):
    scored = frame[["race_id", "finish_pos"]].copy()
    scored["p_raw"] = np.clip(np.asarray(prob, dtype=float), 1e-9, 1.0)
    scored = normalize_race_prob(scored)
    scored["rank"] = scored.groupby("race_id")["p_win"].rank(ascending=False, method="first")
    top = scored[scored["rank"].eq(1)]
    return (
        float(pd.to_numeric(top["finish_pos"], errors="coerce").eq(1).mean())
        if len(top) else None
    )


def recency_weights(dates, half_life_days):
    dates = pd.to_datetime(dates)
    newest = dates.max()
    age_days = (newest - dates).dt.total_seconds() / 86400.0
    weights = np.power(0.5, age_days / float(half_life_days))
    return np.clip(weights, 0.05, 1.0)


def build_model(max_iter=300, random_state=84):
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=random_state,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--half-life-days", type=float, default=120.0)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()

    ensure_dirs()
    df = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    needed = {"race_id", "date", "finish_pos"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"history.csv missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "race_id", "car_no"], kind="mergesort")
    df = add_player_prior_features(df)
    df = add_player_elo_features(df)
    df = add_pair_history_features(df)
    df["target_win"] = pd.to_numeric(df["finish_pos"], errors="coerce").eq(1).astype(int)

    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 30:
        raise ValueError("not enough chronological dates for recency challenger")
    split_idx = max(1, min(len(dates) - 1, int(len(dates) * (1.0 - args.test_fraction))))
    cutoff = dates[split_idx]
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    if train_df["race_id"].nunique() < 1000 or test_df["race_id"].nunique() < 300:
        raise ValueError("insufficient races for recency challenger validation")

    X_train, fill_values = prepare_features(train_df)
    X_test, _ = prepare_features(test_df, fill_values)
    y_train = train_df["target_win"].astype(int)

    weighted = build_model(random_state=84)
    w = recency_weights(train_df["date"], args.half_life_days)
    weighted.fit(X_train, y_train, sample_weight=w)
    weighted_prob = weighted.predict_proba(X_test)[:, 1]
    weighted_rate = race_top1_rate(test_df, weighted_prob)

    baseline = build_model(random_state=84)
    baseline.fit(X_train, y_train)
    baseline_prob = baseline.predict_proba(X_test)[:, 1]
    baseline_rate = race_top1_rate(test_df, baseline_prob)

    X_full, full_fill = prepare_features(df)
    full_weights = recency_weights(df["date"], args.half_life_days)
    final_model = build_model(random_state=84)
    final_model.fit(X_full, df["target_win"].astype(int), sample_weight=full_weights)

    bundle = {
        "model": final_model,
        "calibrator": None,
        "fill_values": full_fill,
        "features": FEATURE_COLS,
        "training_scope": "full_history_recency_weighted_shadow",
        "half_life_days": float(args.half_life_days),
        "trained_through": str(pd.Timestamp(df["date"].max()).date()),
    }
    joblib.dump(bundle, RECENCY_MODEL_PATH)

    metrics = {
        "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "role": "recency_weighted_shadow_challenger",
        "half_life_days": float(args.half_life_days),
        "train_races": int(train_df["race_id"].nunique()),
        "test_races": int(test_df["race_id"].nunique()),
        "test_start": str(pd.Timestamp(test_df["date"].min()).date()),
        "test_end": str(pd.Timestamp(test_df["date"].max()).date()),
        "weighted_top1_hit_rate": weighted_rate,
        "same_split_unweighted_top1_hit_rate": baseline_rate,
        "delta_hit_rate": (
            float(weighted_rate - baseline_rate)
            if weighted_rate is not None and baseline_rate is not None else None
        ),
        "full_history_races": int(df["race_id"].nunique()),
        "production_auto_promotion": False,
        "next_gate": "prospective_same-race shadow audit then external validation",
    }
    RECENCY_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
