import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import HISTORY_CSV, OUTPUT_DIR, TODAY_CSV, ensure_dirs
from model_audit_office import build_model_audit

COMPANY_DIR = OUTPUT_DIR / "company"
ARCHIVE_CSV = COMPANY_DIR / "settled_race_archive.csv"
TOP1_LEDGER = OUTPUT_DIR / "top1_prediction_ledger.csv"
TOP1_SETTLED = OUTPUT_DIR / "top1_settled_results.csv"
TOP1_MISS = OUTPUT_DIR / "top1_miss_analysis.csv"
TOP1_ACCURACY = OUTPUT_DIR / "top1_accuracy.json"
HISTORY_INTEGRITY = OUTPUT_DIR / "history_integrity_report.json"
CROSS_SOURCE_SUMMARY = OUTPUT_DIR / "cross_source_audit_summary.json"
LIVE_DRIFT_AUDIT = OUTPUT_DIR / "live_drift_audit.json"
TRIFECTA_TOP10_ACCURACY = OUTPUT_DIR / "trifecta_top10_accuracy.json"


def now_jst():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")


def read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def write_json(name, payload):
    (COMPANY_DIR / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def num(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def integer(value, default=None):
    value = num(value, None)
    return default if value is None else int(value)


def rate(hits, races):
    return None if not races else round(float(hits) / float(races), 6)


def margin_band(value):
    value = num(value, None)
    if value is None:
        return "unknown"
    if value <= 0.03:
        return "near_tie_<=0.03"
    if value < 0.10:
        return "close_0.03-0.10"
    if value < 0.25:
        return "normal_0.10-0.25"
    return "clear_>=0.25"


def distribution(df, column, topn=30):
    if df.empty or column not in df.columns:
        return []
    counts = df[column].fillna("unknown").astype(str).value_counts().head(topn)
    total = int(counts.sum())
    return [
        {"segment": str(key), "races": int(value), "share": rate(value, total)}
        for key, value in counts.items()
    ]


def grouped_hit_rate(df, column):
    if df.empty or column not in df.columns or "is_hit" not in df.columns:
        return []
    work = df[[column, "is_hit"]].copy()
    work[column] = work[column].fillna("unknown").astype(str)
    work["is_hit"] = work["is_hit"].fillna(False).astype(bool)
    out = (
        work.groupby(column, as_index=False)
        .agg(races=("is_hit", "size"), hits=("is_hit", "sum"))
        .sort_values("races", ascending=False)
    )
    out["hit_rate"] = out["hits"] / out["races"]
    return [
        {
            "segment": str(row[column]),
            "races": int(row["races"]),
            "hits": int(row["hits"]),
            "hit_rate": round(float(row["hit_rate"]), 6),
        }
        for _, row in out.iterrows()
    ]


def settled_results():
    df = read_csv(TOP1_SETTLED, dtype={"race_id": str})
    if df.empty:
        return df
    available = df["official_result_available"] if "official_result_available" in df.columns else pd.Series(False, index=df.index)
    df = df[available.fillna(False).astype(bool)].copy()
    tri = df["actual_trifecta"] if "actual_trifecta" in df.columns else pd.Series("", index=df.index)
    parts = tri.fillna("").astype(str).str.split("-")
    df["winner_car_no"] = pd.to_numeric(parts.str[0], errors="coerce")
    df["second_car_no"] = pd.to_numeric(parts.str[1], errors="coerce")
    df["third_car_no"] = pd.to_numeric(parts.str[2], errors="coerce")
    return df[df["winner_car_no"].notna()].drop_duplicates("race_id", keep="last")


def current_archive_rows():
    today = read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
    actual = settled_results()
    if today.empty or actual.empty:
        return pd.DataFrame()

    top1 = read_csv(TOP1_LEDGER, dtype={"race_id": str, "predicted_winner_player_id": str})
    if not top1.empty:
        top1["race_id"] = top1["race_id"].astype(str)
        top1 = top1.drop_duplicates("race_id", keep="last").set_index("race_id")

    actual = actual.set_index("race_id")
    rows = []
    for race_id, race in today.groupby(today["race_id"].astype(str), sort=False):
        if race_id not in actual.index:
            continue
        result = actual.loc[race_id]
        winner = integer(result.get("winner_car_no"))
        if winner is None:
            continue
        car = pd.to_numeric(race["car_no"], errors="coerce")
        winner_rows = race[car.eq(winner)]
        winner_row = winner_rows.iloc[-1] if len(winner_rows) else pd.Series(dtype=object)
        base = race.iloc[-1]
        row = {
            "race_id": race_id,
            "date": str(base.get("date", "")),
            "venue": str(base.get("venue", "")),
            "race_no": integer(base.get("race_no")),
            "race_class": str(base.get("race_class", "")),
            "race_type": str(base.get("race_type", "")),
            "field_size": integer(base.get("entries_number"), len(race)),
            "number_of_lines": integer(base.get("number_of_lines")),
            "line_type": str(base.get("line_type", "")),
            "winner_car_no": winner,
            "second_car_no": integer(result.get("second_car_no")),
            "third_car_no": integer(result.get("third_car_no")),
            "winner_style": str(winner_row.get("style", "")),
            "winner_line_position": integer(winner_row.get("line_position")),
            "winner_line_size": integer(winner_row.get("line_size")),
            "winner_is_line_leader": integer(winner_row.get("is_line_leader")),
            "source_provider": str(base.get("source_provider", "")),
            "archived_at_jst": now_jst(),
        }
        if not top1.empty and race_id in top1.index:
            pred = top1.loc[race_id]
            pred_car = integer(pred.get("predicted_winner_car_no"))
            row.update(
                {
                    "predicted_winner_car_no": pred_car,
                    "predicted_win_prob": num(pred.get("predicted_win_prob")),
                    "top1_top2_margin": num(pred.get("top1_top2_margin")),
                    "top1_confidence_class": str(pred.get("top1_confidence_class", "")),
                    "predicted_line_position": integer(pred.get("line_position")),
                    "predicted_line_size": integer(pred.get("line_size")),
                    "predicted_style": str(pred.get("style", "")),
                    "second_pick_car_no": integer(pred.get("second_pick_car_no")),
                    "top1_hit": None if pred_car is None else bool(pred_car == winner),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def update_archive():
    current = current_archive_rows()
    old = read_csv(ARCHIVE_CSV, dtype={"race_id": str})
    if current.empty and old.empty:
        return pd.DataFrame()
    out = pd.concat([old, current], ignore_index=True, sort=False)
    out["race_id"] = out["race_id"].astype(str)
    out = out.drop_duplicates("race_id", keep="last")
    sort_cols = [x for x in ["date", "venue", "race_no"] if x in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort")
    out.to_csv(ARCHIVE_CSV, index=False)
    return out


def bool_series(series):
    if series is None:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def load_auxiliary_race_knowledge():
    meta = {
        "history_loaded": False,
        "history_rows": 0,
        "historical_races": 0,
        "providers": [],
    }
    if not HISTORY_CSV.exists():
        return pd.DataFrame(), meta

    history = read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    if history.empty or "race_id" not in history.columns:
        return pd.DataFrame(), meta

    history["race_id"] = history["race_id"].astype(str)
    meta["history_loaded"] = True
    meta["history_rows"] = int(len(history))
    meta["historical_races"] = int(history["race_id"].nunique())
    if "source_provider" in history.columns:
        meta["providers"] = sorted(
            x for x in history["source_provider"].dropna().astype(str).unique() if x
        )

    finish_col = None
    for candidate in ["official_finish_pos", "finish_pos"]:
        if candidate in history.columns:
            values = pd.to_numeric(history[candidate], errors="coerce")
            if values.notna().any():
                finish_col = candidate
                break
    if finish_col is None:
        return pd.DataFrame(), meta

    finish = pd.to_numeric(history[finish_col], errors="coerce")
    winners = history[finish.eq(1)].copy()
    if winners.empty:
        return pd.DataFrame(), meta
    winners = winners.drop_duplicates("race_id", keep="last")

    base_cols = [
        "race_id", "date", "venue", "race_no", "race_class", "race_type",
        "entries_number", "number_of_lines", "line_type", "weather",
        "source_provider",
    ]
    base_cols = [x for x in base_cols if x in history.columns]
    base = history[base_cols].drop_duplicates("race_id", keep="last").copy()
    rename = {"entries_number": "field_size"}
    base = base.rename(columns=rename)

    winner_cols = [
        "race_id", "car_no", "style", "line_position", "line_size",
        "is_line_leader", "score", "rider_strength",
    ]
    winner_cols = [x for x in winner_cols if x in winners.columns]
    winner = winners[winner_cols].copy().rename(
        columns={
            "car_no": "winner_car_no",
            "style": "winner_style",
            "line_position": "winner_line_position",
            "line_size": "winner_line_size",
            "is_line_leader": "winner_is_line_leader",
            "score": "winner_score",
            "rider_strength": "winner_rider_strength",
        }
    )
    out = base.merge(winner, on="race_id", how="left")
    out["knowledge_origin"] = "auxiliary_history"
    return out, meta


def history_manifest(archive, aux_meta, aux_races):
    integrity = read_json(HISTORY_INTEGRITY)
    previous = read_json(COMPANY_DIR / "central_history_manifest.json")

    observed_history_races = int(aux_meta.get("historical_races", 0) or 0)
    observed_history_rows = int(aux_meta.get("history_rows", 0) or 0)
    integrity_races = int(integrity.get("history_races", 0) or 0)
    integrity_rows = int(integrity.get("history_rows", 0) or 0)
    previous_history_races = int(previous.get("historical_races", 0) or 0)
    previous_history_rows = int(previous.get("history_rows", 0) or 0)

    history_races = max(observed_history_races, integrity_races, previous_history_races)
    history_rows = max(observed_history_rows, integrity_rows, previous_history_rows)
    archive_races = int(archive["race_id"].nunique()) if not archive.empty else 0

    providers = set(aux_meta.get("providers", []))
    providers.update(previous.get("known_source_providers", []) or [])
    if not archive.empty and "source_provider" in archive.columns:
        providers.update(archive["source_provider"].dropna().astype(str).unique())

    history_loaded = bool(aux_meta.get("history_loaded"))
    previous_central = int(previous.get("central_races", 0) or 0)
    previous_counted_archive = int(previous.get("archive_races_counted_in_central", 0) or 0)

    if history_loaded:
        history_ids = set(aux_races["race_id"].astype(str)) if not aux_races.empty else set()
        archive_ids = set(archive["race_id"].astype(str)) if not archive.empty else set()
        central_races = max(history_races, len(history_ids | archive_ids))
        central_count_exact = len(history_ids) == observed_history_races
        archive_counted = archive_races
        pending_archive = 0
    else:
        newly_seen_archive = max(archive_races - previous_counted_archive, 0)
        central_races = max(history_races, previous_central + newly_seen_archive)
        central_count_exact = False
        archive_counted = archive_races
        pending_archive = newly_seen_archive

    top1 = read_json(TOP1_ACCURACY)
    cross_source = read_json(CROSS_SOURCE_SUMMARY)
    multi_source_verified = max(
        int(previous.get("multi_source_verified_races", 0) or 0),
        int(cross_source.get("matched_races", 0) or 0),
    )
    return {
        "updated_at_jst": now_jst(),
        "history_rows": history_rows,
        "historical_races": history_races,
        "settled_race_archive_races": archive_races,
        "archive_races_counted_in_central": archive_counted,
        "new_archive_races_since_last_company_update": pending_archive,
        "central_races": central_races,
        "central_count_exact_this_run": central_count_exact,
        "history_cache_loaded_this_run": history_loaded,
        "official_prediction_eval_races": int(top1.get("races", 0) or 0),
        "known_source_providers": sorted(x for x in providers if x),
        "multi_source_verified_races": multi_source_verified,
        "cross_source_audit": {
            "audited_races": int(cross_source.get("audited_races", 0) or 0),
            "matched_races": int(cross_source.get("matched_races", 0) or 0),
            "mismatched_races": int(cross_source.get("mismatched_races", 0) or 0),
            "agreement_rate": num(cross_source.get("agreement_rate")),
            "secondary_source": cross_source.get("secondary_source"),
        },
        "history_integrity": {
            "complete_races": max(
                int(integrity.get("complete_races", 0) or 0),
                int(previous.get("history_integrity", {}).get("complete_races", 0) or 0),
            ),
            "broken_races": int(integrity.get("broken_races", 0) or 0),
            "broken_rate": num(integrity.get("broken_rate")),
            "missing_entry_rows": int(integrity.get("missing_entry_rows", 0) or 0),
        },
        "policy": {
            "historical_races_are_auxiliary_until_cross_source_verified": True,
            "pre_race_predictions_and_official_results_are_separate": True,
            "departments_use_compacted_specialist_knowledge_not_full_history_each_prediction": True,
            "owner_has_final_authority": True,
            "no_subordinate_can_override_ceo": True,
        },
    }


def merged_race_knowledge(aux_races, archive):
    if aux_races.empty and archive.empty:
        return pd.DataFrame()
    out = pd.concat([aux_races, archive], ignore_index=True, sort=False)
    if "race_id" in out.columns:
        out["race_id"] = out["race_id"].astype(str)
        out = out.drop_duplicates("race_id", keep="last")
    return out


def risk_report(archive):
    scored = read_csv(TOP1_MISS, dtype={"race_id": str})
    if not scored.empty:
        hit = scored["is_hit"] if "is_hit" in scored.columns else pd.Series(False, index=scored.index)
        scored["is_hit"] = bool_series(hit)
    elif not archive.empty and "top1_hit" in archive.columns:
        scored = archive[archive["top1_hit"].notna()].copy()
        scored["is_hit"] = bool_series(scored["top1_hit"])

    if scored.empty:
        return {
            "updated_at_jst": now_jst(),
            "role": "risk_department",
            "evaluated_races": 0,
            "hits": 0,
            "top1_hit_rate": None,
            "recent_50_hit_rate": None,
            "hit_rate_by_margin_band": [],
            "hit_rate_by_confidence_class": [],
            "hit_rate_by_predicted_line_position": [],
            "miss_reason_distribution": [],
            "trifecta_top10": read_json(TRIFECTA_TOP10_ACCURACY),
            "live_drift": read_json(LIVE_DRIFT_AUDIT),
        }

    margin = scored["top1_top2_margin"] if "top1_top2_margin" in scored.columns else pd.Series(np.nan, index=scored.index)
    scored["margin_band"] = margin.map(margin_band)
    races = int(len(scored))
    hits = int(scored["is_hit"].sum())
    recent = scored.tail(min(50, races))
    misses = scored[~scored["is_hit"]]
    return {
        "updated_at_jst": now_jst(),
        "role": "risk_department",
        "evaluated_races": races,
        "hits": hits,
        "top1_hit_rate": rate(hits, races),
        "recent_50_hit_rate": round(float(recent["is_hit"].mean()), 6) if len(recent) else None,
        "hit_rate_by_margin_band": grouped_hit_rate(scored, "margin_band"),
        "hit_rate_by_confidence_class": grouped_hit_rate(scored, "top1_confidence_class"),
        "hit_rate_by_predicted_line_position": grouped_hit_rate(scored, "line_position"),
        "miss_reason_distribution": distribution(misses, "miss_reason"),
        "trifecta_top10": read_json(TRIFECTA_TOP10_ACCURACY),
        "live_drift": read_json(LIVE_DRIFT_AUDIT),
    }


def strategist_report(risk, manifest, model_audit=None):
    actions = []
    races = int(risk.get("evaluated_races", 0) or 0)
    overall = num(risk.get("top1_hit_rate"))
    recent = num(risk.get("recent_50_hit_rate"))
    warnings = []

    model_audit = model_audit or {}
    audit = model_audit.get("audit", {}) or {}
    promotion_board = model_audit.get("promotion_board", {}) or {}
    second_pick = model_audit.get("second_pick_specialist", {}) or {}
    live_drift = risk.get("live_drift", {}) or {}
    trifecta_top10 = risk.get("trifecta_top10", {}) or {}

    if live_drift.get("status") in {"amber", "red"}:
        actions.append(
            {
                "priority": 1,
                "type": "live_feature_drift",
                "action": "freeze automatic model promotion and investigate shifted/missing live features before changing production",
                "evidence": {
                    "status": live_drift.get("status"),
                    "flagged_features": live_drift.get("flagged_features"),
                    "features_checked": live_drift.get("features_checked"),
                    "top_flags": (live_drift.get("top_flags") or [])[:5],
                },
            }
        )

    if int(trifecta_top10.get("races", 0) or 0) >= 50:
        actions.append(
            {
                "priority": 2,
                "type": "trifecta_top10_kpi",
                "action": "optimize ticket logic against live 10-point hit rate as a separate KPI from Top1 accuracy",
                "evidence": {
                    "races": trifecta_top10.get("races"),
                    "hit_rate": trifecta_top10.get("hit_rate"),
                    "recent_50_hit_rate": trifecta_top10.get("recent_50_hit_rate"),
                },
            }
        )

    if audit.get("drift_status") == "material_live_gap":
        actions.append(
            {
                "priority": 1,
                "type": "live_validation_gap",
                "action": "freeze broad heuristic expansion; diagnose live-vs-validation Top1 gap before promoting new production logic",
                "evidence": {
                    "production_hit_rate": audit.get("production_hit_rate"),
                    "validation_top1_final_rate": audit.get("validation_top1_final_rate"),
                    "live_vs_validation_final_gap_pp": audit.get("live_vs_validation_final_gap_pp"),
                },
            }
        )

    eligible = promotion_board.get("eligible_for_external_validation", []) or []
    if eligible:
        actions.append(
            {
                "priority": 1,
                "type": "challenger_validation",
                "action": "send the leading shadow challenger to external validation; do not auto-promote",
                "evidence": eligible[0],
            }
        )
    else:
        best = audit.get("best_shadow_challenger") or {}
        if best.get("net_hits", 0) > 0:
            actions.append(
                {
                    "priority": 2,
                    "type": "shadow_leader_watch",
                    "action": "keep the current best challenger in shadow until the 300-race promotion floor is reached",
                    "evidence": best,
                }
            )

    second_rule = second_pick.get("best_shadow_rule")
    if second_rule:
        actions.append(
            {
                "priority": 2,
                "type": "second_pick_reversal_research",
                "action": "continue prospective shadow testing of the best second-pick reversal rule; no live switch until promotion criteria pass",
                "evidence": second_rule,
            }
        )

    if races < 40:
        warnings.append("insufficient_official_prediction_sample_for_weight_changes")
    if manifest.get("multi_source_verified_races", 0) == 0:
        warnings.append("auxiliary_history_not_cross_source_verified")
    if races >= 40 and overall is not None and recent is not None and recent + 0.05 < overall:
        actions.append(
            {
                "priority": 1,
                "type": "deterioration_watch",
                "action": "freeze automatic weight promotion; require shadow validation before production changes",
            }
        )

    misses = {x["segment"]: x for x in risk.get("miss_reason_distribution", [])}
    pressure = misses.get("leader_overrated_under_pressure")
    if pressure and pressure["races"] >= 10:
        actions.append(
            {
                "priority": 2,
                "type": "candidate_overlay",
                "action": "shadow-test a penalty for line leaders under high attack pressure",
                "evidence": pressure,
            }
        )

    close = next(
        (x for x in risk.get("hit_rate_by_margin_band", []) if x["segment"] == "near_tie_<=0.03"),
        None,
    )
    if close and close["races"] >= 10 and overall is not None and close["hit_rate"] + 0.08 < overall:
        actions.append(
            {
                "priority": 2,
                "type": "ticket_breadth",
                "action": "keep multi-head coverage for near-tie races; do not collapse to a single head",
                "evidence": close,
            }
        )

    if not actions:
        actions.append(
            {
                "priority": 3,
                "type": "hold_course",
                "action": "collect more official pre-race evaluations; use auxiliary history for hypotheses, not automatic production changes",
            }
        )

    return {
        "updated_at_jst": now_jst(),
        "role": "strategist",
        "central_races_available": int(manifest.get("central_races", 0) or 0),
        "official_evaluated_races": races,
        "warnings": warnings,
        "recommended_actions": sorted(actions, key=lambda x: x["priority"]),
        "model_audit_status": audit.get("drift_status"),
        "promotion_candidates": promotion_board.get("eligible_for_external_validation", []),
        "production_change_authority": "ceo_only_after_validation",
    }


def preserve_or_build_department(filename, role, history_loaded, fresh_payload, archive_races):
    previous = read_json(COMPANY_DIR / filename)
    if history_loaded or not previous:
        payload = dict(fresh_payload)
        payload["updated_at_jst"] = now_jst()
        payload["role"] = role
        payload["knowledge_refresh_mode"] = (
            "full_history_plus_live_archive" if history_loaded else "live_archive_bootstrap"
        )
        payload["live_settled_archive_races"] = archive_races
        return payload

    payload = dict(previous)
    payload["updated_at_jst"] = now_jst()
    payload["role"] = role
    payload["knowledge_refresh_mode"] = "preserved_full_history_knowledge_plus_live_archive"
    payload["live_settled_archive_races"] = archive_races
    return payload


def run(mode):
    ensure_dirs()
    COMPANY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        model_audit = build_model_audit()
    except Exception as exc:
        model_audit = {
            "audit": {
                "updated_at_jst": now_jst(),
                "role": "ceo_direct_model_audit_office",
                "drift_status": "audit_error",
                "error": str(exc),
            },
            "department_league": {"standings": []},
            "second_pick_specialist": {"best_shadow_rule": None},
            "promotion_board": {"eligible_for_external_validation": []},
            "difficulty_routing": {"segments": []},
        }

    archive = update_archive()
    aux_races, aux_meta = load_auxiliary_race_knowledge()
    manifest = history_manifest(archive, aux_meta, aux_races)
    knowledge = merged_race_knowledge(aux_races, archive)
    history_loaded = bool(aux_meta.get("history_loaded"))
    archive_races = int(manifest.get("settled_race_archive_races", 0) or 0)

    fresh_data = {
        "history": manifest,
        "warnings": (
            (["historical_source_not_cross_verified"] if manifest["multi_source_verified_races"] == 0 else [])
            + (["central_history_below_30000"] if manifest["central_races"] < 30000 else [])
            + (["history_integrity_broken"] if manifest["history_integrity"]["broken_races"] else [])
        ),
        "knowledge_races": int(knowledge["race_id"].nunique()) if not knowledge.empty and "race_id" in knowledge.columns else 0,
        "field_size_distribution": distribution(knowledge, "field_size"),
        "venue_distribution": distribution(knowledge, "venue", topn=50),
        "race_type_distribution": distribution(knowledge, "race_type", topn=30),
    }
    data_dept = preserve_or_build_department(
        "data_department.json", "data_department", history_loaded, fresh_data, archive_races
    )
    data_dept["history"] = manifest
    data_dept["warnings"] = fresh_data["warnings"]

    fresh_pace = {
        "knowledge_races": int(knowledge["race_id"].nunique()) if not knowledge.empty and "race_id" in knowledge.columns else 0,
        "winner_style_distribution": distribution(knowledge, "winner_style"),
        "number_of_lines_distribution": distribution(knowledge, "number_of_lines"),
        "winner_line_leader_distribution": distribution(knowledge, "winner_is_line_leader"),
        "race_type_distribution": distribution(knowledge, "race_type", topn=30),
    }
    pace_dept = preserve_or_build_department(
        "pace_department.json", "pace_department", history_loaded, fresh_pace, archive_races
    )

    scored_archive = (
        archive[archive["top1_hit"].notna()].copy()
        if not archive.empty and "top1_hit" in archive.columns
        else pd.DataFrame()
    )
    if not scored_archive.empty:
        scored_archive["is_hit"] = bool_series(scored_archive["top1_hit"])
    fresh_line = {
        "knowledge_races": int(knowledge["race_id"].nunique()) if not knowledge.empty and "race_id" in knowledge.columns else 0,
        "winner_line_position_distribution": distribution(knowledge, "winner_line_position"),
        "winner_line_size_distribution": distribution(knowledge, "winner_line_size"),
        "top1_hit_rate_by_predicted_line_position": grouped_hit_rate(scored_archive, "predicted_line_position"),
        "top1_hit_rate_by_number_of_lines": grouped_hit_rate(scored_archive, "number_of_lines"),
    }
    line_dept = preserve_or_build_department(
        "line_department.json", "line_department", history_loaded, fresh_line, archive_races
    )
    # Official prediction evaluation is live data; refresh these fields even when
    # the 30k+ auxiliary history cache is not mounted in an intraday job.
    line_dept["top1_hit_rate_by_predicted_line_position"] = fresh_line["top1_hit_rate_by_predicted_line_position"]
    line_dept["top1_hit_rate_by_number_of_lines"] = fresh_line["top1_hit_rate_by_number_of_lines"]

    risk = risk_report(archive)
    strategist = strategist_report(risk, manifest, model_audit)

    findings = []
    status = "green"

    audit_payload = model_audit.get("audit", {}) or {}
    if audit_payload.get("drift_status") == "material_live_gap":
        findings.append(
            {
                "severity": "amber",
                "finding": "live Top1 accuracy is materially below chronological validation",
                "challenge_to_ceo": "Which production-stage adjustment or live-data shift explains the gap before any new override is promoted?",
                "evidence": {
                    "production_hit_rate": audit_payload.get("production_hit_rate"),
                    "validation_top1_final_rate": audit_payload.get("validation_top1_final_rate"),
                    "gap_pp": audit_payload.get("live_vs_validation_final_gap_pp"),
                },
            }
        )
        status = "amber"
    cross = manifest.get("cross_source_audit", {}) or {}
    if int(cross.get("mismatched_races", 0) or 0) > 0:
        findings.append(
            {
                "severity": "red",
                "finding": "cross-source historical result mismatch detected",
                "challenge_to_ceo": "Which races differ across providers, and should those races be quarantined before retraining?",
                "evidence": cross,
            }
        )
        status = "red"

    if manifest["multi_source_verified_races"] == 0:
        findings.append(
            {
                "severity": "amber",
                "finding": "historical auxiliary data is not cross-source verified",
                "challenge_to_ceo": "Which proposed change is supported by official pre-race validation rather than auxiliary history alone?",
            }
        )
        status = "amber"
    if manifest["history_integrity"]["broken_races"]:
        findings.append(
            {
                "severity": "red",
                "finding": "history integrity contains broken races",
                "challenge_to_ceo": "Why should retraining proceed before the broken history is repaired?",
            }
        )
        status = "red"
    if risk["evaluated_races"] < 100:
        findings.append(
            {
                "severity": "amber",
                "finding": "official pre-race evaluation sample is still limited",
                "challenge_to_ceo": "Would the same conclusion survive another independent block of settled races?",
            }
        )
        if status == "green":
            status = "amber"

    third_party = {
        "updated_at_jst": now_jst(),
        "role": "independent_audit_and_ceo_coach",
        "overall_status": status,
        "findings": findings,
        "ceo_coaching_questions": [x["challenge_to_ceo"] for x in findings],
        "may_override_ceo": False,
    }
    ceo = {
        "updated_at_jst": now_jst(),
        "role": "ceo_decision_support",
        "owner_is_final_authority": True,
        "ceo_is_highest_ai_authority": True,
        "subordinates_may_override_ceo": False,
        "production_change_blocked": status == "red",
        "central_races_available": manifest["central_races"],
        "historical_races_available": manifest["historical_races"],
        "official_prediction_eval_races": risk["evaluated_races"],
        "official_top1_hit_rate": risk["top1_hit_rate"],
        "trifecta_top10_kpi": risk.get("trifecta_top10"),
        "live_feature_drift": risk.get("live_drift"),
        "cross_source_audit": manifest.get("cross_source_audit"),
        "model_audit_status": audit_payload.get("drift_status"),
        "best_shadow_challenger": audit_payload.get("best_shadow_challenger"),
        "promotion_candidates": (model_audit.get("promotion_board", {}) or {}).get("eligible_for_external_validation", []),
        "second_pick_shadow_rule": (model_audit.get("second_pick_specialist", {}) or {}).get("best_shadow_rule"),
        "department_league_leader": (
            ((model_audit.get("department_league", {}) or {}).get("standings") or [None])[0]
        ),
        "strategist_recommendations": strategist["recommended_actions"],
        "third_party_status": status,
        "decision_rule": "production changes require validation evidence; auxiliary history creates hypotheses but cannot promote itself",
    }
    executive = {
        "updated_at_jst": now_jst(),
        "role": "executive_officer",
        "source": "ceo_decision_support",
        "status": "dissemination_package_pending_ceo_or_owner_decision",
        "directives": [
            {
                "target": "model_freshness_department",
                "instruction": "run the recency-weighted challenger in shadow and report same-race gains without automatic promotion",
            },
            {
                "target": "data_quality_audit",
                "instruction": "track live feature drift and cross-source history agreement; quarantine mismatches before retraining",
            },
            {
                "target": "ticket_audit",
                "instruction": "track live 3連単10点 hit rate independently from Top1 accuracy",
            },
            {
                "target": "model_audit_office",
                "instruction": "compare production and every shadow challenger on identical settled races; block unsupported promotion",
            },
            {
                "target": "second_pick_specialist",
                "instruction": "measure Top1-to-second reversal opportunities prospectively and remain shadow-only until validation passes",
            },
            {
                "target": "risk_department",
                "instruction": "keep official pre-race validation separate and report deterioration immediately",
            },
            {
                "target": "data_department",
                "instruction": "retain cumulative history, append settled races, and never label single-source history fully verified",
            },
            {
                "target": "pace_department",
                "instruction": "maintain compacted pace/style knowledge from the full auxiliary history",
            },
            {
                "target": "line_department",
                "instruction": "maintain compacted line-position knowledge and compare it with official prediction misses",
            },
        ],
        "strategist_actions_to_disseminate_after_approval": strategist["recommended_actions"],
        "may_change_production_without_ceo": False,
    }
    state = {
        "updated_at_jst": now_jst(),
        "mode": mode,
        "governance": {
            "owner": "final_decision",
            "ceo": "highest_ai_executive",
            "strategist": "strategy_advice",
            "executive_officer": "dissemination_and_execution_control",
            "departments": "specialist_analysis",
            "secretary": "briefing_and_coordination",
            "independent_audit": "challenge_and_ceo_coaching_without_override",
            "model_audit_office": "same_race_model_comparison_and_promotion_control",
            "second_pick_specialist": "top1_to_second_reversal_shadow_research",
            "model_freshness_department": "recency_weighted_shadow_model_competition",
            "data_quality_audit": "live_feature_drift_and_cross_source_validation",
            "ticket_audit": "live_trifecta_top10_product_kpi",
        },
    }

    write_json("central_history_manifest.json", manifest)
    write_json("data_department.json", data_dept)
    write_json("pace_department.json", pace_dept)
    write_json("line_department.json", line_dept)
    write_json("risk_department.json", risk)
    write_json("strategist_brief.json", strategist)
    write_json("third_party_audit.json", third_party)
    write_json("ceo_decision_support.json", ceo)
    write_json("executive_directive.json", executive)
    write_json("company_state.json", state)

    digest = [
        "# 競輪AI 会社運用ダイジェスト",
        "",
        f"- 更新: {now_jst()}",
        f"- 中央戦史: {manifest['central_races']:,}レース",
        f"- 履歴キャッシュ: {manifest['historical_races']:,}レース",
        f"- 正式な発走前予想検証: {risk['evaluated_races']:,}レース",
        f"- Top1的中率: {risk['top1_hit_rate'] if risk['top1_hit_rate'] is not None else '未算出'}",
        f"- モデル監査: {audit_payload.get('drift_status', '未算出')}",
        f"- ライブ入力ドリフト: {(risk.get('live_drift') or {}).get('status', '未算出')}",
        f"- 3連単10点的中率: {(risk.get('trifecta_top10') or {}).get('hit_rate', '未算出')}",
        f"- 戦史クロスソース一致: {(manifest.get('cross_source_audit') or {}).get('agreement_rate', '未算出')}",
        f"- 昇格候補: {len((model_audit.get('promotion_board', {}) or {}).get('eligible_for_external_validation', []))}",
        f"- 第三者監査: {status}",
        "",
        "## 軍師提言",
    ]
    digest += [f"- P{x['priority']}: {x['action']}" for x in strategist["recommended_actions"]]
    digest += [
        "",
        "## 専務の周知",
        "- モデル鮮度部: 直近重視チャレンジャーを影運用し、本番モデルの経年劣化を監視する。",
        "- データ品質監査: 学習時との特徴量ドリフトと戦史クロスソース一致率を監視する。",
        "- 券種監査: Top1とは別に3連単10点の実戦的中率を正式KPIとして監視する。",
        "- モデル監査室: 本番と全チャレンジャーを同一確定レースで比較し、純増のない補正を昇格させない。",
        "- 2番手逆転班: Top1→2番手入替候補を影運用し、300R以上の前向き検証まで本番変更しない。",
        "- データ部: 累積戦史を保持し、新規確定結果を差分追加する。",
        "- 展開部: 全戦史から圧縮した脚質・展開知識を維持する。",
        "- ライン部: 全戦史のライン傾向と本番予想の失敗を分離して評価する。",
        "- リスク部: 発走前固定予想だけを正式成績として監視する。",
        "",
        "## 権限",
        "- オーナーが最終決裁者。",
        "- AI組織内ではCEOが最上位。軍師・専務・各部署・秘書・第三者監査はCEOを上書きしない。",
        "- 第三者監査はCEOへ反証質問を返し、判断力を鍛えるが決裁権は持たない。",
        "- 本番変更は検証を通過した候補だけを採用する。",
        "",
    ]
    (COMPANY_DIR / "secretary_digest.md").write_text("\n".join(digest), encoding="utf-8")
    print(
        f"company loop updated: central={manifest['central_races']} "
        f"history={manifest['historical_races']} "
        f"archive={manifest['settled_race_archive_races']} "
        f"official_eval={risk['evaluated_races']}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["settle", "weekly", "manual"], default="manual")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
