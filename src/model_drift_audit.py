import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import MODEL_DIR, OUTPUT_DIR

FEATURE_BASELINE_PATH = MODEL_DIR / "feature_baseline.json"
LIVE_DRIFT_PATH = OUTPUT_DIR / "live_drift_audit.json"


def _finite_series(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _stats(series):
    s = _finite_series(series).dropna()
    if s.empty:
        return {
            "count": 0, "mean": None, "std": None, "median": None,
            "q01": None, "q10": None, "q90": None, "q99": None,
        }
    q = s.quantile([0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "median": float(q.loc[0.50]),
        "q01": float(q.loc[0.01]),
        "q10": float(q.loc[0.10]),
        "q90": float(q.loc[0.90]),
        "q99": float(q.loc[0.99]),
    }


def build_feature_baseline(prepared_features, source_df, metadata=None, path=FEATURE_BASELINE_PATH):
    payload = {
        "created_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "rows": int(len(prepared_features)),
        "metadata": metadata or {},
        "features": {},
    }
    for col in prepared_features.columns:
        item = _stats(prepared_features[col])
        source_present = col in source_df.columns
        item["source_present"] = bool(source_present)
        item["source_missing_rate"] = (
            float(pd.to_numeric(source_df[col], errors="coerce").isna().mean())
            if source_present and len(source_df) else None
        )
        payload["features"][col] = item
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def audit_live_drift(prepared_features, source_df, baseline_path=FEATURE_BASELINE_PATH, output_path=LIVE_DRIFT_PATH):
    try:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        payload = {
            "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "status": "baseline_unavailable",
            "rows": int(len(prepared_features)),
            "flagged_features": 0,
            "features": [],
        }
        Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    details = []
    flagged = 0
    for col, ref in (baseline.get("features") or {}).items():
        if col not in prepared_features.columns:
            continue
        live = _finite_series(prepared_features[col])
        live_nonnull = live.dropna()
        if live_nonnull.empty:
            continue

        live_stats = _stats(live_nonnull)
        q10 = ref.get("q10")
        q90 = ref.get("q90")
        q01 = ref.get("q01")
        q99 = ref.get("q99")
        ref_median = ref.get("median")
        span = None
        if q10 is not None and q90 is not None:
            span = max(float(q90) - float(q10), 1e-6)
        median_shift = (
            abs(float(live_stats["median"]) - float(ref_median)) / span
            if span is not None and ref_median is not None else None
        )
        outlier_share = None
        if q01 is not None and q99 is not None:
            outlier_share = float(((live_nonnull < float(q01)) | (live_nonnull > float(q99))).mean())

        live_missing = (
            float(pd.to_numeric(source_df[col], errors="coerce").isna().mean())
            if ref.get("source_present") and col in source_df.columns and len(source_df)
            else None
        )
        missing_increase = (
            live_missing - float(ref.get("source_missing_rate"))
            if live_missing is not None and ref.get("source_missing_rate") is not None
            else None
        )

        reasons = []
        if median_shift is not None and median_shift >= 1.5:
            reasons.append("median_shift")
        if outlier_share is not None and outlier_share >= 0.25:
            reasons.append("outlier_share")
        if missing_increase is not None and missing_increase >= 0.20:
            reasons.append("missingness_jump")
        is_flagged = bool(reasons)
        flagged += int(is_flagged)
        details.append({
            "feature": col,
            "flagged": is_flagged,
            "reasons": reasons,
            "median_iqr_shift": median_shift,
            "outlier_share_vs_training_1_99pct": outlier_share,
            "live_source_missing_rate": live_missing,
            "training_source_missing_rate": ref.get("source_missing_rate"),
            "missing_rate_increase": missing_increase,
            "live_median": live_stats.get("median"),
            "training_median": ref_median,
        })

    total = len(details)
    ratio = flagged / total if total else 0.0
    if flagged >= 10 or ratio >= 0.15:
        status = "red"
    elif flagged >= 4 or ratio >= 0.05:
        status = "amber"
    else:
        status = "green"

    details = sorted(
        details,
        key=lambda x: (
            int(x["flagged"]),
            x["median_iqr_shift"] or 0.0,
            x["outlier_share_vs_training_1_99pct"] or 0.0,
        ),
        reverse=True,
    )
    payload = {
        "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "status": status,
        "rows": int(len(prepared_features)),
        "features_checked": total,
        "flagged_features": flagged,
        "flagged_ratio": ratio,
        "top_flags": [x for x in details if x["flagged"]][:20],
        "features": details,
    }
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
