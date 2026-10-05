import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, MODEL_DIR

COMPANY_DIR = OUTPUT_DIR / "company"
VARIANT_RESULTS = OUTPUT_DIR / "top1_variant_results.csv"
MISS_ANALYSIS = OUTPUT_DIR / "top1_miss_analysis.csv"
V4_SUMMARY = MODEL_DIR / "v4_validation_summary.json"

MIN_PROMOTION_RACES = 300
MIN_CHANGED_RACES = 30
MIN_DELTA_RATE = 0.015
MIN_NET_HITS = 5


DEPARTMENT_BY_VARIANT = {
    "production": "production_baseline",
    "core_model": "model_audit_office",
    "second_wheel": "second_pick_specialist",
    "pace_conflict": "pace_department",
    "race_scenario": "pace_department",
    "variable_line": "line_department",
    "chaser_guard": "risk_department",
    "second_pick_reversal_live": "second_pick_specialist",
    "difficulty_router": "model_audit_office",
    "recency_model": "model_freshness_department",
}


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


def bool_series(series):
    if series is None:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y"}
    )


def safe_rate(hits, races):
    return None if not races else float(hits) / float(races)


def margin_band(value):
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return "unknown"
    if value <= 0.03:
        return "near_tie_<=0.03"
    if value < 0.10:
        return "close_0.03-0.10"
    if value < 0.25:
        return "normal_0.10-0.25"
    return "clear_>=0.25"


def prepare_variant_results(path=VARIANT_RESULTS):
    df = read_csv(path, dtype={"race_id": str})
    required = {"race_id", "variant", "predicted_winner_car_no", "actual_winner_car_no"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    if "official_result_available" in df.columns:
        df = df[bool_series(df["official_result_available"])].copy()
    df["race_id"] = df["race_id"].astype(str)
    df["predicted_winner_car_no"] = pd.to_numeric(df["predicted_winner_car_no"], errors="coerce")
    df["actual_winner_car_no"] = pd.to_numeric(df["actual_winner_car_no"], errors="coerce")
    df = df[df["predicted_winner_car_no"].notna() & df["actual_winner_car_no"].notna()].copy()
    return df.drop_duplicates(["race_id", "variant"], keep="last")


def compare_variant_to_production(df, variant):
    production = df[df["variant"].eq("production")][
        ["race_id", "predicted_winner_car_no", "actual_winner_car_no"]
    ].rename(columns={"predicted_winner_car_no": "production_pick"})
    challenger = df[df["variant"].eq(variant)][
        ["race_id", "predicted_winner_car_no"]
    ].rename(columns={"predicted_winner_car_no": "challenger_pick"})
    merged = production.merge(challenger, on="race_id", how="inner")
    if merged.empty:
        return {
            "variant": variant,
            "department": DEPARTMENT_BY_VARIANT.get(variant, "experimental_department"),
            "races": 0,
            "production_hits": 0,
            "challenger_hits": 0,
            "production_hit_rate": None,
            "challenger_hit_rate": None,
            "delta_hit_rate": None,
            "delta_pp": None,
            "changed_races": 0,
            "flip_gains": 0,
            "flip_losses": 0,
            "net_hits": 0,
            "promotion_status": "no_data",
        }

    merged["production_hit"] = merged["production_pick"].eq(merged["actual_winner_car_no"])
    merged["challenger_hit"] = merged["challenger_pick"].eq(merged["actual_winner_car_no"])
    merged["changed"] = merged["production_pick"].ne(merged["challenger_pick"])
    merged["gain"] = merged["changed"] & ~merged["production_hit"] & merged["challenger_hit"]
    merged["loss"] = merged["changed"] & merged["production_hit"] & ~merged["challenger_hit"]

    races = int(len(merged))
    production_hits = int(merged["production_hit"].sum())
    challenger_hits = int(merged["challenger_hit"].sum())
    prod_rate = safe_rate(production_hits, races)
    chall_rate = safe_rate(challenger_hits, races)
    delta = None if prod_rate is None or chall_rate is None else chall_rate - prod_rate
    changed_races = int(merged["changed"].sum())
    gains = int(merged["gain"].sum())
    losses = int(merged["loss"].sum())
    net_hits = gains - losses

    if races < MIN_PROMOTION_RACES or changed_races < MIN_CHANGED_RACES:
        status = "insufficient_sample"
    elif delta is not None and delta >= MIN_DELTA_RATE and net_hits >= MIN_NET_HITS:
        status = "eligible_for_external_validation"
    elif net_hits > 0:
        status = "promising_shadow"
    elif net_hits == 0:
        status = "neutral"
    else:
        status = "underperforming"

    return {
        "variant": variant,
        "department": DEPARTMENT_BY_VARIANT.get(variant, "experimental_department"),
        "races": races,
        "production_hits": production_hits,
        "challenger_hits": challenger_hits,
        "production_hit_rate": prod_rate,
        "challenger_hit_rate": chall_rate,
        "delta_hit_rate": delta,
        "delta_pp": None if delta is None else delta * 100.0,
        "changed_races": changed_races,
        "flip_gains": gains,
        "flip_losses": losses,
        "net_hits": net_hits,
        "promotion_status": status,
    }


def second_pick_specialist(miss_path=MISS_ANALYSIS):
    scored = read_csv(miss_path, dtype={"race_id": str})
    required = {
        "race_id",
        "actual_winner_car_no",
        "predicted_winner_car_no",
        "second_pick_car_no",
        "top1_top2_margin",
    }
    if scored.empty or not required.issubset(scored.columns):
        return {
            "updated_at_jst": now_jst(),
            "role": "second_pick_reversal_specialist",
            "races": 0,
            "threshold_tests": [],
            "best_shadow_rule": None,
            "production_use": "shadow_only",
        }

    if "official_result_available" in scored.columns:
        scored = scored[bool_series(scored["official_result_available"])].copy()
    scored = scored.drop_duplicates("race_id", keep="last")
    for col in ["actual_winner_car_no", "predicted_winner_car_no", "second_pick_car_no", "top1_top2_margin"]:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")
    scored = scored[
        scored["actual_winner_car_no"].notna()
        & scored["predicted_winner_car_no"].notna()
        & scored["second_pick_car_no"].notna()
    ].copy()
    if scored.empty:
        return {
            "updated_at_jst": now_jst(),
            "role": "second_pick_reversal_specialist",
            "races": 0,
            "threshold_tests": [],
            "best_shadow_rule": None,
            "production_use": "shadow_only",
        }

    scored["production_hit"] = scored["predicted_winner_car_no"].eq(scored["actual_winner_car_no"])
    scored["second_hit"] = scored["second_pick_car_no"].eq(scored["actual_winner_car_no"])
    total_races = int(len(scored))
    baseline_hits = int(scored["production_hit"].sum())

    tests = []
    for threshold in [0.03, 0.05, 0.10, 0.15, 0.25]:
        mask = scored["top1_top2_margin"].le(threshold)
        subset = scored[mask]
        races = int(len(subset))
        if not races:
            continue
        prod_hits = int(subset["production_hit"].sum())
        second_hits = int(subset["second_hit"].sum())
        switched_total_hits = baseline_hits - prod_hits + second_hits
        net_hits = second_hits - prod_hits
        tests.append({
            "rule": f"switch_to_second_when_margin<={threshold:.2f}",
            "margin_threshold": threshold,
            "switched_races": races,
            "production_hits_in_switched_races": prod_hits,
            "second_pick_hits_in_switched_races": second_hits,
            "net_hits": net_hits,
            "baseline_total_hits": baseline_hits,
            "hypothetical_total_hits": switched_total_hits,
            "baseline_hit_rate": safe_rate(baseline_hits, total_races),
            "hypothetical_hit_rate": safe_rate(switched_total_hits, total_races),
            "global_delta_pp": (
                (switched_total_hits - baseline_hits) / total_races * 100.0
                if total_races else None
            ),
        })

    eligible = [x for x in tests if x["switched_races"] >= 30 and x["net_hits"] > 0]
    best = max(
        eligible,
        key=lambda x: (x["global_delta_pp"], x["net_hits"], x["switched_races"]),
        default=None,
    )
    if best:
        best = dict(best)
        best["status"] = "shadow_candidate_only"
        best["promotion_requirement"] = (
            "must beat production prospectively on >=300 settled races and pass external validation"
        )

    second_wins = int(scored["second_hit"].sum())
    misses = int((~scored["production_hit"]).sum())
    return {
        "updated_at_jst": now_jst(),
        "role": "second_pick_reversal_specialist",
        "races": total_races,
        "production_hits": baseline_hits,
        "production_hit_rate": safe_rate(baseline_hits, total_races),
        "second_pick_wins": second_wins,
        "second_pick_share_of_production_misses": (
            second_wins / misses if misses else None
        ),
        "threshold_tests": tests,
        "best_shadow_rule": best,
        "production_use": "shadow_only",
    }


def difficulty_routing(miss_path=MISS_ANALYSIS, variant_path=VARIANT_RESULTS):
    scored = read_csv(miss_path, dtype={"race_id": str})
    if scored.empty or "top1_top2_margin" not in scored.columns:
        return []
    if "official_result_available" in scored.columns:
        scored = scored[bool_series(scored["official_result_available"])].copy()
    scored = scored.drop_duplicates("race_id", keep="last")
    scored["top1_top2_margin"] = pd.to_numeric(scored["top1_top2_margin"], errors="coerce")
    scored["production_hit"] = (
        pd.to_numeric(scored.get("predicted_winner_car_no"), errors="coerce")
        .eq(pd.to_numeric(scored.get("actual_winner_car_no"), errors="coerce"))
    )
    scored["margin_band"] = scored["top1_top2_margin"].map(margin_band)

    core = prepare_variant_results(variant_path)
    core = core[core["variant"].eq("core_model")][
        ["race_id", "predicted_winner_car_no", "actual_winner_car_no"]
    ].copy()
    if not core.empty:
        core["core_hit"] = core["predicted_winner_car_no"].eq(core["actual_winner_car_no"])
        scored = scored.merge(core[["race_id", "core_hit"]], on="race_id", how="left")
    else:
        scored["core_hit"] = np.nan

    rows = []
    order = ["clear_>=0.25", "normal_0.10-0.25", "close_0.03-0.10", "near_tie_<=0.03", "unknown"]
    for band in order:
        g = scored[scored["margin_band"].eq(band)]
        if g.empty:
            continue
        core_known = g["core_hit"].notna()
        if band == "clear_>=0.25":
            policy = "preserve_primary_ranking; challengers need strong evidence before override"
        elif band == "near_tie_<=0.03":
            policy = "treat as high-ambiguity; keep multi-head coverage and shadow-test reversals"
        else:
            policy = "allow specialist shadow competition; no automatic production override"
        rows.append({
            "segment": band,
            "races": int(len(g)),
            "production_hits": int(g["production_hit"].sum()),
            "production_hit_rate": float(g["production_hit"].mean()),
            "core_sample_races": int(core_known.sum()),
            "core_hits": int(g.loc[core_known, "core_hit"].sum()) if core_known.any() else 0,
            "core_hit_rate": float(g.loc[core_known, "core_hit"].mean()) if core_known.any() else None,
            "policy": policy,
        })
    return rows


def build_model_audit(
    output_dir=OUTPUT_DIR,
    company_dir=COMPANY_DIR,
    variant_path=VARIANT_RESULTS,
    miss_path=MISS_ANALYSIS,
    validation_summary_path=V4_SUMMARY,
):
    company_dir = Path(company_dir)
    company_dir.mkdir(parents=True, exist_ok=True)

    variant_df = prepare_variant_results(variant_path)
    variants = sorted(x for x in variant_df.get("variant", pd.Series(dtype=str)).dropna().astype(str).unique() if x != "production")
    comparisons = [compare_variant_to_production(variant_df, variant) for variant in variants]
    comparisons = sorted(
        comparisons,
        key=lambda x: (
            -999 if x["delta_hit_rate"] is None else x["delta_hit_rate"],
            x["net_hits"],
            x["races"],
        ),
        reverse=True,
    )

    production_rows = variant_df[variant_df["variant"].eq("production")].copy()
    production_races = int(len(production_rows))
    production_hits = int(
        production_rows["predicted_winner_car_no"].eq(production_rows["actual_winner_car_no"]).sum()
    ) if production_races else 0
    production_rate = safe_rate(production_hits, production_races)

    validation = read_json(validation_summary_path)
    validation_final = validation.get("top1_final", {}) if isinstance(validation, dict) else {}
    validation_core = validation.get("top1_core", {}) if isinstance(validation, dict) else {}
    validation_final_rate = validation_final.get("rate")
    validation_core_rate = validation_core.get("rate")
    live_gap_pp = (
        (production_rate - float(validation_final_rate)) * 100.0
        if production_rate is not None and validation_final_rate is not None
        else None
    )
    drift_status = (
        "material_live_gap"
        if live_gap_pp is not None and live_gap_pp <= -5.0
        else "watch"
        if live_gap_pp is not None and live_gap_pp <= -2.0
        else "stable_or_unknown"
    )

    league = []
    for row in comparisons:
        item = dict(row)
        item["league_status"] = row["promotion_status"]
        league.append(item)

    specialist = second_pick_specialist(miss_path)
    routing = difficulty_routing(miss_path, variant_path)
    eligible = [x for x in comparisons if x["promotion_status"] == "eligible_for_external_validation"]
    best = comparisons[0] if comparisons else None

    audit = {
        "updated_at_jst": now_jst(),
        "role": "ceo_direct_model_audit_office",
        "production_races": production_races,
        "production_hits": production_hits,
        "production_hit_rate": production_rate,
        "validation_top1_final_rate": validation_final_rate,
        "validation_top1_core_rate": validation_core_rate,
        "live_vs_validation_final_gap_pp": live_gap_pp,
        "drift_status": drift_status,
        "comparisons": comparisons,
        "best_shadow_challenger": best,
        "policy": {
            "production_changes_are_automatic": False,
            "minimum_same_race_sample": MIN_PROMOTION_RACES,
            "minimum_changed_races": MIN_CHANGED_RACES,
            "minimum_delta_hit_rate": MIN_DELTA_RATE,
            "minimum_net_hits": MIN_NET_HITS,
            "eligible_candidates_go_to": "external_validation_then_ceo_owner_decision",
        },
    }

    promotion_board = {
        "updated_at_jst": now_jst(),
        "role": "model_promotion_board",
        "eligible_for_external_validation": eligible,
        "watchlist": [
            x for x in comparisons
            if x["promotion_status"] in {"promising_shadow", "insufficient_sample"}
            and x["net_hits"] > 0
        ],
        "production_auto_promotion": False,
        "final_authority": "owner_after_ceo_review",
    }

    league_payload = {
        "updated_at_jst": now_jst(),
        "role": "department_shadow_league",
        "baseline": {
            "variant": "production",
            "races": production_races,
            "hits": production_hits,
            "hit_rate": production_rate,
        },
        "standings": league,
        "rule": "departments compete on same-race official results; only validated net improvement can advance",
    }

    routing_payload = {
        "updated_at_jst": now_jst(),
        "role": "race_difficulty_routing",
        "segments": routing,
        "production_change": "none; routing is governance/shadow guidance until validated",
    }

    outputs = {
        "model_audit_office.json": audit,
        "department_league.json": league_payload,
        "second_pick_specialist.json": specialist,
        "promotion_board.json": promotion_board,
        "difficulty_routing.json": routing_payload,
    }
    for name, payload in outputs.items():
        (company_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    pd.DataFrame(comparisons).to_csv(company_dir / "model_audit_comparisons.csv", index=False)

    return {
        "audit": audit,
        "department_league": league_payload,
        "second_pick_specialist": specialist,
        "promotion_board": promotion_board,
        "difficulty_routing": routing_payload,
    }


def main():
    payload = build_model_audit()
    print(json.dumps({
        "production_hit_rate": payload["audit"].get("production_hit_rate"),
        "best_shadow_challenger": payload["audit"].get("best_shadow_challenger"),
        "eligible_for_external_validation": len(
            payload["promotion_board"].get("eligible_for_external_validation", [])
        ),
        "second_pick_best_shadow_rule": payload["second_pick_specialist"].get("best_shadow_rule"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
