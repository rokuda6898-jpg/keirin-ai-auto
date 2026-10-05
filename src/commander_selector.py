import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import MODEL_DIR

MODEL_PATH = MODEL_DIR / "commander_selector.joblib"

FEATURES = [
    "predicted_win_prob",
    "top1_top2_margin",
    "second_pick_prob",
    "prob_gap",
    "line_position",
    "line_size",
    "second_pick_line_position",
    "race_attack_pressure",
    "other_line_attack_pressure",
    "variant_code",
    "differs_from_production",
    "candidate_is_second_pick",
    "candidate_vote_share",
    "variant_count",
    "unique_pick_count",
]


def variant_code(value):
    text = str(value or "")
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16) % 1000)


def _num(value, default=0.0):
    value = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(value) else float(value)


def race_candidate_features(base_row, variant_rows):
    if variant_rows is None or len(variant_rows) == 0:
        return pd.DataFrame(columns=["variant", "predicted_winner_car_no", *FEATURES])

    work = variant_rows.copy()
    work["predicted_winner_car_no"] = pd.to_numeric(
        work["predicted_winner_car_no"], errors="coerce"
    )
    work = work[work["predicted_winner_car_no"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["variant", "predicted_winner_car_no", *FEATURES])

    prod = work[work["variant"].astype(str).eq("production")]
    production_pick = (
        int(prod.iloc[-1]["predicted_winner_car_no"])
        if len(prod) else int(work.iloc[0]["predicted_winner_car_no"])
    )
    second_pick = pd.to_numeric(base_row.get("second_pick_car_no"), errors="coerce")
    vote_counts = work["predicted_winner_car_no"].astype(int).value_counts()
    variant_count = int(len(work))
    unique_pick_count = int(work["predicted_winner_car_no"].nunique())

    rows = []
    for _, row in work.iterrows():
        pick = int(row["predicted_winner_car_no"])
        rows.append({
            "variant": str(row.get("variant", "")),
            "predicted_winner_car_no": pick,
            "predicted_win_prob": _num(base_row.get("predicted_win_prob")),
            "top1_top2_margin": _num(base_row.get("top1_top2_margin")),
            "second_pick_prob": _num(base_row.get("second_pick_prob")),
            "prob_gap": _num(base_row.get("predicted_win_prob")) - _num(base_row.get("second_pick_prob")),
            "line_position": _num(base_row.get("line_position")),
            "line_size": _num(base_row.get("line_size")),
            "second_pick_line_position": _num(base_row.get("second_pick_line_position")),
            "race_attack_pressure": _num(base_row.get("race_attack_pressure")),
            "other_line_attack_pressure": _num(base_row.get("other_line_attack_pressure")),
            "variant_code": variant_code(row.get("variant")),
            "differs_from_production": float(pick != production_pick),
            "candidate_is_second_pick": float(pd.notna(second_pick) and pick == int(second_pick)),
            "candidate_vote_share": float(vote_counts.get(pick, 0) / max(variant_count, 1)),
            "variant_count": float(variant_count),
            "unique_pick_count": float(unique_pick_count),
        })
    return pd.DataFrame(rows)


def choose_variant(base_row, variant_rows, model_path=MODEL_PATH):
    path = Path(model_path)
    if not path.exists() or variant_rows is None or len(variant_rows) == 0:
        return None
    try:
        bundle = joblib.load(path)
        model = bundle.get("model")
        features = list(bundle.get("features") or FEATURES)
        fills = bundle.get("fill_values") or {}
        if model is None:
            return None
        frame = race_candidate_features(base_row, variant_rows)
        if frame.empty:
            return None
        X = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
        for col in features:
            X[col] = X[col].fillna(float(fills.get(col, 0.0)))
        frame["commander_score"] = model.predict_proba(X)[:, 1]
        frame = frame.sort_values(
            ["commander_score", "candidate_vote_share"],
            ascending=[False, False],
            kind="mergesort",
        )
        return frame.iloc[0].to_dict()
    except Exception as exc:
        print(f"commander selector shadow scoring skipped: {exc}", flush=True)
        return None
