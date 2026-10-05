import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common import HISTORY_CSV, OUTPUT_DIR
from fetch_today_entries import http_get
from settle_results import parse_netkeirin_result_html

RESULTS_CSV = OUTPUT_DIR / "cross_source_audit_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "cross_source_audit_summary.json"


def netkeirin_url_from_race_id(race_id):
    rid = str(race_id or "").strip()
    if len(rid) != 12 or not rid.isdigit():
        return ""
    race_no = rid[:2]
    venue_code = rid[2:4]
    date = rid[4:12]
    return f"https://keirin.netkeiba.com/race/result/?race_id={date}{venue_code}{race_no}"


def expected_top3(group):
    finish_col = None
    for candidate in ["official_finish_pos", "finish_pos"]:
        if candidate in group.columns:
            values = pd.to_numeric(group[candidate], errors="coerce")
            if values.isin([1, 2, 3]).sum() >= 3:
                finish_col = candidate
                break
    if finish_col is None or "car_no" not in group.columns:
        return ""
    work = group[["car_no", finish_col]].copy()
    work["car_no"] = pd.to_numeric(work["car_no"], errors="coerce")
    work["finish"] = pd.to_numeric(work[finish_col], errors="coerce")
    work = work[work["finish"].isin([1, 2, 3])].sort_values("finish")
    if len(work) != 3 or work["finish"].tolist() != [1, 2, 3]:
        return ""
    cars = work["car_no"].dropna().astype(int).tolist()
    return "-".join(str(x) for x in cars) if len(cars) == 3 else ""


def deterministic_order(race_id):
    return hashlib.sha256(str(race_id).encode("utf-8")).hexdigest()


def load_history_candidates(history):
    rows = []
    for race_id, group in history.groupby(history["race_id"].astype(str), sort=False):
        tri = expected_top3(group)
        if not tri:
            continue
        base = group.iloc[-1]
        date = str(base.get("date", ""))
        rows.append({
            "race_id": str(race_id),
            "date": date,
            "venue": str(base.get("venue", "")),
            "race_no": base.get("race_no", ""),
            "expected_trifecta": tri,
            "sort_key": deterministic_order(race_id),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["_date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values(["_date", "sort_key"], ascending=[False, True], kind="mergesort")
    return frame.drop(columns=["_date"])


def summarize(results):
    if results.empty:
        return {
            "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "audited_races": 0,
            "matched_races": 0,
            "mismatched_races": 0,
            "unavailable_races": 0,
            "error_races": 0,
            "agreement_rate": None,
            "secondary_source": "netkeirin",
        }
    status = results["status"].fillna("").astype(str)
    matched = int(status.eq("match").sum())
    mismatched = int(status.eq("mismatch").sum())
    comparable = matched + mismatched
    return {
        "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "audited_races": int(len(results)),
        "matched_races": matched,
        "mismatched_races": mismatched,
        "unavailable_races": int(status.eq("unavailable").sum()),
        "error_races": int(status.eq("error").sum()),
        "agreement_rate": (matched / comparable if comparable else None),
        "comparable_races": comparable,
        "secondary_source": "netkeirin",
        "policy": "matched_races are counted as cross-source verified history",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-races", type=int, default=200)
    parser.add_argument("--candidate-pool", type=int, default=5000)
    parser.add_argument("--sleep-sec", type=float, default=0.25)
    args = parser.parse_args()

    history = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    if history.empty or "race_id" not in history.columns:
        raise ValueError("history.csv is unavailable for cross-source audit")

    candidates = load_history_candidates(history)
    if candidates.empty:
        raise ValueError("no settled history races available for cross-source audit")
    candidates = candidates.head(max(int(args.candidate_pool), int(args.max_races)))

    try:
        old = pd.read_csv(RESULTS_CSV, dtype={"race_id": str})
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        old = pd.DataFrame()

    completed = set()
    if len(old):
        completed = set(
            old.loc[old["status"].astype(str).isin(["match", "mismatch"]), "race_id"].astype(str)
        )
    todo = candidates[~candidates["race_id"].astype(str).isin(completed)].head(int(args.max_races))

    audit_rows = []
    history_by_race = {str(rid): g.copy() for rid, g in history.groupby(history["race_id"].astype(str), sort=False)}
    for item in todo.to_dict("records"):
        race_id = str(item["race_id"])
        url = netkeirin_url_from_race_id(race_id)
        status = "error"
        observed = ""
        detail = ""
        try:
            page = http_get(url, attempts=3)
            parsed = parse_netkeirin_result_html(page, history_by_race[race_id], url)
            if parsed and parsed.get("official_result_available"):
                observed = str(parsed.get("actual_trifecta", "") or "")
                status = "match" if observed == item["expected_trifecta"] else "mismatch"
            else:
                status = "unavailable"
        except Exception as exc:
            detail = str(exc)
            status = "error"

        audit_rows.append({
            "race_id": race_id,
            "date": item.get("date", ""),
            "venue": item.get("venue", ""),
            "race_no": item.get("race_no", ""),
            "expected_trifecta": item.get("expected_trifecta", ""),
            "secondary_trifecta": observed,
            "status": status,
            "secondary_source": "netkeirin",
            "secondary_url": url,
            "checked_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "detail": detail,
        })
        print(f"cross-source race={race_id} status={status} expected={item['expected_trifecta']} observed={observed}")
        time.sleep(max(float(args.sleep_sec), 0.0))

    current = pd.DataFrame(audit_rows)
    combined = pd.concat([old, current], ignore_index=True, sort=False)
    if len(combined):
        combined["race_id"] = combined["race_id"].astype(str)
        priority = {"match": 4, "mismatch": 4, "unavailable": 2, "error": 1}
        combined["_priority"] = combined["status"].astype(str).map(priority).fillna(0)
        combined = combined.sort_values(["race_id", "_priority", "checked_at_jst"], kind="mergesort")
        combined = combined.drop_duplicates("race_id", keep="last").drop(columns=["_priority"])
        combined = combined.sort_values(["date", "race_id"], kind="mergesort")
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RESULTS_CSV, index=False)

    payload = summarize(combined)
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
