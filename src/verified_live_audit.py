"""Separate timestamp-confirmed evidence from legacy post-close overwritten rows."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from common import OUTPUT_DIR


def build_verified_live_audit(output_dir=OUTPUT_DIR):
    paths = [output_dir / "top1_prediction_ledger.csv", output_dir / "top1_settled_results.csv"]
    report = {"updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
              "legacy_settled_rows": 0, "timestamp_verified_races": 0, "unverified_rows": 0,
              "hits": 0, "hit_rate": None, "scope": "prediction timestamp strictly before known close time",
              "limitations": "Legacy rows are timestamp checks, not a reconstructed immutable archive."}
    if all(p.exists() for p in paths):
        ledger, actual = [pd.read_csv(p, dtype={"race_id": str}) for p in paths]
        if not ledger.empty and not actual.empty:
            merged = ledger.merge(actual, on="race_id")
            merged = merged[merged.official_result_available.astype(str).str.lower().isin(["true", "1"])]
            report["legacy_settled_rows"] = len(merged)
            created = pd.to_datetime(merged.get("prediction_created_at_jst"), utc=True, errors="coerce")
            close = pd.to_datetime(pd.to_numeric(merged.get("close_at"), errors="coerce"), unit="s", utc=True, errors="coerce")
            if isinstance(created, pd.Series) and isinstance(close, pd.Series):
                verified = merged[created.notna() & close.notna() & created.lt(close)].copy()
                actual_winner = pd.to_numeric(verified.actual_trifecta.astype(str).str.split("-").str[0], errors="coerce")
                hits = int(pd.to_numeric(verified.predicted_winner_car_no, errors="coerce").eq(actual_winner).sum())
                report.update(timestamp_verified_races=len(verified), hits=hits,
                              hit_rate=hits / len(verified) if len(verified) else None)
            report["unverified_rows"] = report["legacy_settled_rows"] - report["timestamp_verified_races"]
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "verified_live_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
