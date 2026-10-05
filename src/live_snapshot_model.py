from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import FEATURE_COLS, MODEL_DIR, MODEL_PATH as PRODUCTION_MODEL_PATH, normalize_race_prob, prepare_features

MODEL_PATH = MODEL_DIR / "live_snapshot_win_model.joblib"
METRICS_PATH = MODEL_DIR / "live_snapshot_model_metrics.json"

EXTRA_FEATURES = [
    "snapshot_bucket_code",
    "seconds_to_close",
    "odds_move_pct",
    "odds_move_last_pct",
    "odds_snapshot_count",
]
LIVE_FEATURES = [*FEATURE_COLS, *EXTRA_FEATURES]

BUCKET_CODE = {"morning": 0.0, "40m": 1.0, "20m": 2.0, "10m": 3.0}


def add_snapshot_context(frame, now_epoch=None):
    out = frame.copy()
    if "snapshot_bucket_code" not in out.columns:
        bucket = out.get("snapshot_bucket", pd.Series("", index=out.index)).astype(str)
        out["snapshot_bucket_code"] = bucket.map(BUCKET_CODE)
    if "seconds_to_close" not in out.columns and now_epoch is not None:
        close_at = pd.to_numeric(out.get("close_at"), errors="coerce")
        out["seconds_to_close"] = close_at - float(now_epoch)
    for col in EXTRA_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return out


def build_live_matrix_from_snapshot_rows(frame, fills=None):
    work = add_snapshot_context(frame)
    X = work.reindex(columns=LIVE_FEATURES).apply(pd.to_numeric, errors="coerce")
    if fills is None:
        fills = {}
        for col in LIVE_FEATURES:
            med = X[col].median()
            fills[col] = 0.0 if pd.isna(med) else float(med)
    for col in LIVE_FEATURES:
        X[col] = X[col].fillna(float(fills.get(col, 0.0)))
    return X, fills


def build_live_matrix_from_current(frame, now_epoch, fills=None, source_fill_values=None):
    work = frame.copy()
    X_base, _ = prepare_features(work, source_fill_values)
    enriched = pd.DataFrame(index=work.index)
    for col in FEATURE_COLS:
        enriched[col] = pd.to_numeric(X_base[col], errors="coerce") if col in X_base.columns else np.nan
    current = add_snapshot_context(work, now_epoch=now_epoch)
    for col in EXTRA_FEATURES:
        enriched[col] = pd.to_numeric(current[col], errors="coerce")
    X = enriched.reindex(columns=LIVE_FEATURES)
    if fills is None:
        fills = {}
        for col in LIVE_FEATURES:
            med = X[col].median()
            fills[col] = 0.0 if pd.isna(med) else float(med)
    for col in LIVE_FEATURES:
        X[col] = X[col].fillna(float(fills.get(col, 0.0)))
    return X, fills


def score_current(frame, now_epoch, model_path=MODEL_PATH):
    path = Path(model_path)
    if not path.exists() or frame is None or len(frame) == 0:
        return None
    try:
        bundle = joblib.load(path)
        model = bundle.get("model")
        calibrator = bundle.get("calibrator")
        features = list(bundle.get("features") or LIVE_FEATURES)
        fills = bundle.get("fill_values") or {}
        if model is None:
            return None
        source_fill_values = None
        if PRODUCTION_MODEL_PATH.exists():
            try:
                production_bundle = joblib.load(PRODUCTION_MODEL_PATH)
                if isinstance(production_bundle, dict):
                    source_fill_values = production_bundle.get("fill_values") or None
            except Exception:
                source_fill_values = None
        X, _ = build_live_matrix_from_current(
            frame, now_epoch, fills, source_fill_values=source_fill_values
        )
        X = X.reindex(columns=features)
        raw = model.predict_proba(X)[:, 1]
        prob = calibrator.predict(raw) if calibrator is not None else raw
        work = frame[["race_id"]].copy()
        work["p_live_raw"] = np.clip(prob, 1e-9, 1.0)
        work = normalize_race_prob(work, "p_live_raw", "p_live_snapshot")
        return pd.Series(work["p_live_snapshot"].to_numpy(), index=frame.index)
    except Exception as exc:
        print(f"live snapshot model shadow scoring skipped: {exc}", flush=True)
        return None
