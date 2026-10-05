import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import HISTORY_CSV, OUTPUT_DIR, TODAY_CSV, ensure_dirs

COMPANY_DIR = OUTPUT_DIR / "company"
ARCHIVE_CSV = COMPANY_DIR / "settled_race_archive.csv"
TOP1_LEDGER = OUTPUT_DIR / "top1_prediction_ledger.csv"
TOP1_SETTLED = OUTPUT_DIR / "top1_settled_results.csv"
TOP1_MISS = OUTPUT_DIR / "top1_miss_analysis.csv"
TOP1_ACCURACY = OUTPUT_DIR / "top1_accuracy.json"
HISTORY_INTEGRITY = OUTPUT_DIR / "history_integrity_report.json"


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


def history_manifest(archive):
    integrity = read_json(HISTORY_INTEGRITY)
    history_races = int(integrity.get("history_races", 0) or 0)
    history_rows = int(integrity.get("history_rows", 0) or 0)
    providers = set()

    if HISTORY_CSV.exists():
        history = read_csv(HISTORY_CSV, dtype={"race_id": str})
        if not history.empty:
            history_rows = int(len(history))
            history_races = int(history["race_id"].nunique())
            if "source_provider" in history.columns:
                providers.update(history["source_provider"].dropna().astype(str).unique())

    if not archive.empty and "source_provider" in archive.columns:
        providers.update(archive["source_provider"].dropna().astype(str).unique())

    top1 = read_json(TOP1_ACCURACY)
    return {
        "updated_at_jst": now_jst(),
        "history_rows": history_rows,
        "historical_races": history_races,
        "settled_race_archive_races": int(archive["race_id"].nunique()) if not archive.empty else 0,
        "official_prediction_eval_races": int(top1.get("races", 0) or 0),
        "known_source_providers": sorted(x for x in providers if x),
        "multi_source_verified_races": 0,
        "history_integrity": {
            "complete_races": int(integrity.get("complete_races", 0) or 0),
            "broken_races": int(integrity.get("broken_races", 0) or 0),
            "broken_rate": num(integrity.get("broken_rate")),
            "missing_entry_rows": int(integrity.get("missing_entry_rows", 0) or 0),
        },
        "policy": {
            "historical_races_are_auxiliary_until_cross_source_verified": True,
            "pre_race_predictions_and_official_results_are_separate": True,
            "owner_has_final_authority": True,
            "no_subordinate_can_override_ceo": True,
        },
    }


def risk_report(archive):
    scored = read_csv(TOP1_MISS, dtype={"race_id": str})
    if not scored.empty:
        hit = scored["is_hit"] if "is_hit" in scored.columns else pd.Series(False, index=scored.index)
        scored["is_hit"] = hit.fillna(False).astype(bool)
    elif not archive.empty and "top1_hit" in archive.columns:
        scored = archive[archive["top1_hit"].notna()].copy()
        scored["is_hit"] = scored["top1_hit"].astype(bool)

    if scored.empty:
        return {
            "updated_at_jst": now_jst(),
            "evaluated_races": 0,
            "hits": 0,
            "top1_hit_rate": None,
            "recent_50_hit_rate": None,
            "hit_rate_by_margin_band": [],
            "miss_reason_distribution": [],
        }

    margin = scored["top1_top2_margin"] if "top1_top2_margin" in scored.columns else pd.Series(np.nan, index=scored.index)
    scored["margin_band"] = margin.map(margin_band)
    races = int(len(scored))
    hits = int(scored["is_hit"].sum())
    recent = scored.tail(min(50, races))
    misses = scored[~scored["is_hit"]]
    return {
        "updated_at_jst": now_jst(),
        "evaluated_races": races,
        "hits": hits,
        "top1_hit_rate": rate(hits, races),
        "recent_50_hit_rate": round(float(recent["is_hit"].mean()), 6) if len(recent) else None,
        "hit_rate_by_margin_band": grouped_hit_rate(scored, "margin_band"),
        "hit_rate_by_confidence_class": grouped_hit_rate(scored, "top1_confidence_class"),
        "hit_rate_by_predicted_line_position": grouped_hit_rate(scored, "line_position"),
        "miss_reason_distribution": distribution(misses, "miss_reason"),
    }


def strategist_report(risk):
    actions = []
    races = int(risk.get("evaluated_races", 0) or 0)
    overall = num(risk.get("top1_hit_rate"))
    recent = num(risk.get("recent_50_hit_rate"))
    warnings = []

    if races < 40:
        warnings.append("insufficient_official_prediction_sample_for_weight_changes")
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
                "action": "collect more official pre-race evaluations; do not alter production from auxiliary history alone",
            }
        )

    return {
        "updated_at_jst": now_jst(),
        "role": "strategist",
        "evaluated_races": races,
        "warnings": warnings,
        "recommended_actions": sorted(actions, key=lambda x: x["priority"]),
        "production_change_authority": "ceo_only_after_validation",
    }


def run(mode):
    ensure_dirs()
    COMPANY_DIR.mkdir(parents=True, exist_ok=True)

    archive = update_archive()
    manifest = history_manifest(archive)
    risk = risk_report(archive)
    strategist = strategist_report(risk)

    data_dept = {
        "updated_at_jst": now_jst(),
        "role": "data_department",
        "history": manifest,
        "warnings": (
            (["historical_source_not_cross_verified"] if manifest["multi_source_verified_races"] == 0 else [])
            + (["repository_visible_history_below_30000"] if manifest["historical_races"] < 30000 else [])
            + (["history_integrity_broken"] if manifest["history_integrity"]["broken_races"] else [])
        ),
        "field_size_distribution": distribution(archive, "field_size"),
        "venue_distribution": distribution(archive, "venue", topn=50),
    }
    pace_dept = {
        "updated_at_jst": now_jst(),
        "role": "pace_department",
        "winner_style_distribution": distribution(archive, "winner_style"),
        "number_of_lines_distribution": distribution(archive, "number_of_lines"),
        "winner_line_leader_distribution": distribution(archive, "winner_is_line_leader"),
    }

    scored_archive = archive[archive["top1_hit"].notna()].copy() if not archive.empty and "top1_hit" in archive.columns else pd.DataFrame()
    if not scored_archive.empty:
        scored_archive["is_hit"] = scored_archive["top1_hit"].astype(bool)
    line_dept = {
        "updated_at_jst": now_jst(),
        "role": "line_department",
        "winner_line_position_distribution": distribution(archive, "winner_line_position"),
        "winner_line_size_distribution": distribution(archive, "winner_line_size"),
        "top1_hit_rate_by_predicted_line_position": grouped_hit_rate(scored_archive, "predicted_line_position"),
        "top1_hit_rate_by_number_of_lines": grouped_hit_rate(scored_archive, "number_of_lines"),
    }

    findings = []
    status = "green"
    if manifest["multi_source_verified_races"] == 0:
        findings.append({"severity": "amber", "finding": "historical auxiliary data is not cross-source verified"})
        status = "amber"
    if manifest["history_integrity"]["broken_races"]:
        findings.append({"severity": "red", "finding": "history integrity contains broken races"})
        status = "red"
    if risk["evaluated_races"] < 100:
        findings.append({"severity": "amber", "finding": "official pre-race evaluation sample is still limited"})
        if status == "green":
            status = "amber"

    third_party = {
        "updated_at_jst": now_jst(),
        "role": "independent_audit",
        "overall_status": status,
        "findings": findings,
        "may_override_ceo": False,
    }
    ceo = {
        "updated_at_jst": now_jst(),
        "role": "ceo_decision_support",
        "owner_is_final_authority": True,
        "ceo_is_highest_ai_authority": True,
        "subordinates_may_override_ceo": False,
        "production_change_blocked": status == "red",
        "historical_races_available": manifest["historical_races"],
        "official_prediction_eval_races": risk["evaluated_races"],
        "official_top1_hit_rate": risk["top1_hit_rate"],
        "strategist_recommendations": strategist["recommended_actions"],
        "third_party_status": status,
    }
    state = {
        "updated_at_jst": now_jst(),
        "mode": mode,
        "authority_chain": ["owner", "ceo", "strategist", "executives", "departments", "secretary", "independent_audit"],
        "note": "independent audit challenges but cannot override CEO",
    }

    write_json("central_history_manifest.json", manifest)
    write_json("data_department.json", data_dept)
    write_json("pace_department.json", pace_dept)
    write_json("line_department.json", line_dept)
    write_json("risk_department.json", risk)
    write_json("strategist_brief.json", strategist)
    write_json("third_party_audit.json", third_party)
    write_json("ceo_decision_support.json", ceo)
    write_json("company_state.json", state)

    digest = [
        "# 競輪AI 会社運用ダイジェスト",
        "",
        f"- 更新: {now_jst()}",
        f"- 中央戦史: {manifest['historical_races']:,}レース",
        f"- 正式な発走前予想検証: {risk['evaluated_races']:,}レース",
        f"- Top1的中率: {risk['top1_hit_rate'] if risk['top1_hit_rate'] is not None else '未算出'}",
        f"- 第三者監査: {status}",
        "",
        "## 軍師提言",
    ]
    digest += [f"- P{x['priority']}: {x['action']}" for x in strategist["recommended_actions"]]
    digest += [
        "",
        "## 権限",
        "- オーナーが最終決裁者。",
        "- AI組織内ではCEOが最上位。軍師・役員・各部署・秘書・第三者監査はCEOを上書きしない。",
        "- 本番変更は検証を通過した候補だけを採用する。",
        "",
    ]
    (COMPANY_DIR / "secretary_digest.md").write_text("\n".join(digest), encoding="utf-8")
    print(
        f"company loop updated: history={manifest['historical_races']} "
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
