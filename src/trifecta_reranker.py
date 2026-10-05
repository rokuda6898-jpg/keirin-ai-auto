import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import MODEL_DIR

MODEL_PATH = MODEL_DIR / "trifecta_reranker.joblib"

FEATURES = [
    "base_prob",
    "log_base_prob",
    "first_p_win", "first_p_second", "first_p_third",
    "second_p_win", "second_p_second", "second_p_third",
    "third_p_win", "third_p_second", "third_p_third",
    "first_strength", "second_strength", "third_strength",
    "first_score", "second_score", "third_score",
    "first_line_position", "second_line_position", "third_line_position",
    "first_line_size", "second_line_size", "third_line_size",
    "same_line_12", "same_line_13", "same_line_23",
    "first_is_leader", "second_is_second_wheel",
    "number_of_lines", "entries_number",
    "first_win_gap_second", "second_place_gap_third",
]


def _num(value, default=np.nan):
    value = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(value) else float(value)


def _car_map(race_df):
    work = race_df.copy()
    work["car_no"] = pd.to_numeric(work["car_no"], errors="coerce")
    return {
        int(row["car_no"]): row
        for _, row in work[work["car_no"].notna()].iterrows()
    }


def _same_line(a, b):
    la = _num(a.get("line_id"))
    lb = _num(b.get("line_id"))
    return float(pd.notna(la) and pd.notna(lb) and int(la) >= 0 and int(la) == int(lb))


def build_candidate_features(race_df, candidates):
    if candidates is None or len(candidates) == 0:
        return pd.DataFrame(columns=["buy", *FEATURES])

    by_car = _car_map(race_df)
    if not by_car:
        return pd.DataFrame(columns=["buy", *FEATURES])

    pwin = pd.to_numeric(race_df.get("p_win"), errors="coerce").dropna().sort_values(ascending=False)
    first_second_gap = float(pwin.iloc[0] - pwin.iloc[1]) if len(pwin) >= 2 else 0.0

    psecond = pd.to_numeric(
        race_df.get("p_second", race_df.get("p_win")), errors="coerce"
    ).dropna().sort_values(ascending=False)
    second_third_gap = float(psecond.iloc[1] - psecond.iloc[2]) if len(psecond) >= 3 else 0.0

    base = race_df.iloc[0]
    rows = []
    for _, cand in candidates.iterrows():
        try:
            first, second, third = [int(x) for x in str(cand.get("buy", "")).split("-")]
        except (TypeError, ValueError):
            continue
        if first not in by_car or second not in by_car or third not in by_car:
            continue

        a, b, c = by_car[first], by_car[second], by_car[third]
        base_prob = max(_num(cand.get("prob"), 0.0), 1e-12)
        row = {
            "buy": f"{first}-{second}-{third}",
            "base_prob": base_prob,
            "log_base_prob": math.log(base_prob),
            "first_p_win": _num(a.get("p_win"), 0.0),
            "first_p_second": _num(a.get("p_second", a.get("p_win")), 0.0),
            "first_p_third": _num(a.get("p_third", a.get("p_win")), 0.0),
            "second_p_win": _num(b.get("p_win"), 0.0),
            "second_p_second": _num(b.get("p_second", b.get("p_win")), 0.0),
            "second_p_third": _num(b.get("p_third", b.get("p_win")), 0.0),
            "third_p_win": _num(c.get("p_win"), 0.0),
            "third_p_second": _num(c.get("p_second", c.get("p_win")), 0.0),
            "third_p_third": _num(c.get("p_third", c.get("p_win")), 0.0),
            "first_strength": _num(a.get("rider_strength"), 0.0),
            "second_strength": _num(b.get("rider_strength"), 0.0),
            "third_strength": _num(c.get("rider_strength"), 0.0),
            "first_score": _num(a.get("score"), 0.0),
            "second_score": _num(b.get("score"), 0.0),
            "third_score": _num(c.get("score"), 0.0),
            "first_line_position": _num(a.get("line_position"), 0.0),
            "second_line_position": _num(b.get("line_position"), 0.0),
            "third_line_position": _num(c.get("line_position"), 0.0),
            "first_line_size": _num(a.get("line_size"), 0.0),
            "second_line_size": _num(b.get("line_size"), 0.0),
            "third_line_size": _num(c.get("line_size"), 0.0),
            "same_line_12": _same_line(a, b),
            "same_line_13": _same_line(a, c),
            "same_line_23": _same_line(b, c),
            "first_is_leader": float(_num(a.get("line_position"), 0.0) == 1.0),
            "second_is_second_wheel": float(_num(b.get("line_position"), 0.0) == 2.0),
            "number_of_lines": _num(base.get("number_of_lines"), 0.0),
            "entries_number": _num(base.get("entries_number"), len(race_df)),
            "first_win_gap_second": first_second_gap,
            "second_place_gap_third": second_third_gap,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def score_candidates(race_df, candidates, model_path=MODEL_PATH):
    path = Path(model_path)
    if not path.exists() or candidates is None or len(candidates) == 0:
        return None
    try:
        bundle = joblib.load(path)
        model = bundle.get("model")
        features = list(bundle.get("features") or FEATURES)
        fills = bundle.get("fill_values") or {}
        if model is None:
            return None
        frame = build_candidate_features(race_df, candidates)
        if frame.empty:
            return None
        X = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
        for col in features:
            fill = float(fills.get(col, 0.0))
            X[col] = X[col].fillna(fill)
        scores = model.predict_proba(X)[:, 1]
        out = frame[["buy", "base_prob"]].copy()
        out["reranker_score"] = np.asarray(scores, dtype=float)
        return out.sort_values(
            ["reranker_score", "base_prob"], ascending=[False, False], kind="mergesort"
        )
    except Exception as exc:
        print(f"trifecta reranker shadow scoring skipped: {exc}", flush=True)
        return None
