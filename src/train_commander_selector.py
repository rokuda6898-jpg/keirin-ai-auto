import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from commander_selector import FEATURES, MODEL_PATH, race_candidate_features
from common import MODEL_DIR, OUTPUT_DIR

VARIANT_RESULTS = OUTPUT_DIR / "top1_variant_results.csv"
MISS_ANALYSIS = OUTPUT_DIR / "top1_miss_analysis.csv"
METRICS_PATH = MODEL_DIR / "commander_selector_metrics.json"


def _prepare_x(frame, fills=None):
    X = frame.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce")
    if fills is None:
        fills = {}
        for col in FEATURES:
            med = X[col].median()
            fills[col] = 0.0 if pd.isna(med) else float(med)
    for col in FEATURES:
        X[col] = X[col].fillna(float(fills.get(col, 0.0)))
    return X, fills


def _build_training_frame(variants, base):
    if variants.empty or base.empty:
        return pd.DataFrame()
    variants = variants[variants["variant"].astype(str).ne("commander_selector")].copy()
    rows = []
    base_index = base.drop_duplicates("race_id", keep="last").set_index("race_id")

    for race_id, group in variants.groupby("race_id", sort=False):
        rid = str(race_id)
        if rid not in base_index.index:
            continue
        base_row = base_index.loc[rid]
        if isinstance(base_row, pd.DataFrame):
            base_row = base_row.iloc[-1]
        actual = pd.to_numeric(group.get("actual_winner_car_no"), errors="coerce").dropna()
        if actual.empty:
            continue
        actual_winner = int(actual.iloc[-1])
        feat = race_candidate_features(base_row, group)
        if feat.empty:
            continue
        feat["race_id"] = rid
        feat["date"] = str(base_row.get("date", ""))
        feat["target"] = pd.to_numeric(
            feat["predicted_winner_car_no"], errors="coerce"
        ).eq(actual_winner).astype(int)
        rows.append(feat)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _selector_hit_rate(frame, score_col):
    if frame.empty:
        return None, 0, 0
    hits = 0
    races = 0
    for _, g in frame.groupby("race_id", sort=False):
        best = g.sort_values(score_col, ascending=False, kind="mergesort").iloc[0]
        hits += int(best["target"] == 1)
        races += 1
    return (hits / races if races else None), hits, races


def _production_hit_rate(frame):
    if frame.empty:
        return None, 0, 0
    prod = frame[frame["variant"].astype(str).eq("production")].copy()
    if prod.empty:
        return None, 0, 0
    prod = prod.drop_duplicates("race_id", keep="last")
    hits = int(pd.to_numeric(prod["target"], errors="coerce").fillna(0).astype(int).sum())
    races = int(len(prod))
    return (hits / races if races else None), hits, races


def main():
    try:
        variants = pd.read_csv(VARIANT_RESULTS, dtype={"race_id": str})
        base = pd.read_csv(MISS_ANALYSIS, dtype={"race_id": str})
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        metrics = {
            "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "status": "insufficient_data",
            "reason": "settled variant/base ledgers unavailable",
            "production_auto_promotion": False,
        }
        METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    data = _build_training_frame(variants, base)
    race_count = int(data["race_id"].nunique()) if len(data) else 0
    if race_count < 250:
        metrics = {
            "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "status": "insufficient_data",
            "races": race_count,
            "required_races": 250,
            "production_auto_promotion": False,
        }
        METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    race_dates = (
        data.groupby("race_id", as_index=False)["date"].min()
        .sort_values(["date", "race_id"], kind="mergesort")
    )
    split_idx = max(1, min(len(race_dates) - 1, int(len(race_dates) * 0.70)))
    split_date = pd.Timestamp(race_dates.iloc[split_idx]["date"])
    train = data[data["date"] < split_date].copy()
    test = data[data["date"] >= split_date].copy()
    if train.empty or test.empty:
        raise ValueError("commander chronological split is empty")

    X_train, fills = _prepare_x(train)
    y_train = train["target"].astype(int)
    if y_train.nunique() < 2:
        metrics = {
            "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "status": "insufficient_label_diversity",
            "training_races": int(train["race_id"].nunique()),
            "production_auto_promotion": False,
        }
        METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=23,
        l2_regularization=0.08,
        random_state=2718,
    )
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    pos_weight = max(1.0, min(negative / max(positive, 1), 20.0))
    weights = np.where(y_train.to_numpy() == 1, pos_weight, 1.0)
    model.fit(X_train, y_train, sample_weight=weights)

    X_test, _ = _prepare_x(test, fills)
    test["commander_score"] = model.predict_proba(X_test)[:, 1]
    selector_rate, selector_hits, test_races = _selector_hit_rate(test, "commander_score")
    production_rate, production_hits, production_races = _production_hit_rate(test)
    delta = (
        float(selector_rate - production_rate)
        if selector_rate is not None and production_rate is not None else None
    )

    X_full, full_fills = _prepare_x(data)
    y_full = data["target"].astype(int)
    final_model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=23,
        l2_regularization=0.08,
        random_state=2718,
    )
    positive_full = int(y_full.sum())
    negative_full = int(len(y_full) - positive_full)
    full_weight = max(1.0, min(negative_full / max(positive_full, 1), 20.0))
    final_model.fit(
        X_full,
        y_full,
        sample_weight=np.where(y_full.to_numpy() == 1, full_weight, 1.0),
    )
    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
            "fill_values": full_fills,
            "training_scope": "settled_live_shadow_department_candidates",
        },
        MODEL_PATH,
    )

    metrics = {
        "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "status": "shadow_model_ready",
        "training_races": int(train["race_id"].nunique()),
        "test_races": int(test_races),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "production_hits": int(production_hits),
        "production_hit_rate": production_rate,
        "commander_hits": int(selector_hits),
        "commander_hit_rate": selector_rate,
        "delta_hit_rate": delta,
        "delta_pp": None if delta is None else delta * 100.0,
        "production_auto_promotion": False,
        "promotion_rule": "commander remains shadow until prospective >=300-race same-race audit and external validation",
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
