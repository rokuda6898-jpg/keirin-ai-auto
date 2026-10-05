import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from common import MODEL_DIR, OUTPUT_DIR
from trifecta_reranker import FEATURES, MODEL_PATH

CANDIDATES_CSV = OUTPUT_DIR / "production_replay_trifecta_candidates.csv"
METRICS_JSON = MODEL_DIR / "trifecta_reranker_metrics.json"


def top10_hit_rate(frame, score_col):
    if frame.empty:
        return None, 0, 0
    hits = 0
    races = 0
    for _, g in frame.groupby("race_id", sort=False):
        top = g.sort_values(score_col, ascending=False, kind="mergesort").head(10)
        hits += int(pd.to_numeric(top["target"], errors="coerce").fillna(0).astype(int).eq(1).any())
        races += 1
    return (hits / races if races else None), hits, races


def prepare_x(frame, fills=None):
    X = frame.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce")
    if fills is None:
        fills = {}
        for col in FEATURES:
            med = X[col].median()
            fills[col] = 0.0 if pd.isna(med) else float(med)
    for col in FEATURES:
        X[col] = X[col].fillna(float(fills.get(col, 0.0)))
    return X, fills


def fit_model(X, y):
    model = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=314,
    )
    positive = int(y.sum())
    negative = int(len(y) - positive)
    pos_weight = max(1.0, min(negative / max(positive, 1), 100.0))
    weights = np.where(y.to_numpy(dtype=int) == 1, pos_weight, 1.0)
    model.fit(X, y, sample_weight=weights)
    return model, pos_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-races", type=int, default=2000)
    args = parser.parse_args()

    df = pd.read_csv(CANDIDATES_CSV, dtype={"race_id": str, "buy": str})
    required = {"race_id", "date", "buy", "target", "base_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"replay candidate file missing: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    race_dates = (
        df.groupby("race_id", as_index=False)["date"].min()
        .sort_values(["date", "race_id"], kind="mergesort")
    )
    if len(race_dates) < int(args.min_races):
        raise ValueError(
            f"need >= {args.min_races} replay races for trifecta reranker; have {len(race_dates)}"
        )

    split_idx = max(1, min(len(race_dates) - 1, int(len(race_dates) * 0.75)))
    split_date = pd.Timestamp(race_dates.iloc[split_idx]["date"])
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    if train.empty or test.empty:
        raise ValueError("trifecta reranker chronological split is empty")

    X_train, fills = prepare_x(train)
    y_train = train["target"]
    if y_train.nunique() < 2:
        metrics = {
            "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "role": "shadow_trifecta_combination_reranker",
            "status": "insufficient_label_diversity",
            "training_races": int(train["race_id"].nunique()),
            "production_auto_promotion": False,
        }
        METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    model, pos_weight = fit_model(X_train, y_train)

    X_test, _ = prepare_x(test, fills)
    test["reranker_score"] = model.predict_proba(X_test)[:, 1]

    base_rate, base_hits, test_races = top10_hit_rate(test, "base_prob")
    rerank_rate, rerank_hits, _ = top10_hit_rate(test, "reranker_score")
    delta = (
        float(rerank_rate - base_rate)
        if base_rate is not None and rerank_rate is not None else None
    )

    X_full, full_fills = prepare_x(df)
    final_model, final_pos_weight = fit_model(X_full, df["target"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
            "fill_values": full_fills,
            "training_scope": "out_of_sample_production_replay_candidates",
        },
        MODEL_PATH,
    )

    metrics = {
        "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "role": "shadow_trifecta_combination_reranker",
        "training_candidate_rows": int(len(train)),
        "training_races": int(train["race_id"].nunique()),
        "test_candidate_rows": int(len(test)),
        "test_races": int(test_races),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "base_top10_hits": int(base_hits),
        "base_top10_hit_rate": base_rate,
        "reranker_top10_hits": int(rerank_hits),
        "reranker_top10_hit_rate": rerank_rate,
        "delta_hit_rate": delta,
        "delta_pp": None if delta is None else delta * 100.0,
        "training_positive_weight": float(pos_weight),
        "final_positive_weight": float(final_pos_weight),
        "production_auto_promotion": False,
        "promotion_rule": "prospective live Top10 shadow audit plus external validation before any production use",
    }
    METRICS_JSON.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
