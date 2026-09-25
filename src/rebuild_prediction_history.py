import glob
import re
from pathlib import Path

import pandas as pd

from fetch_today_entries import fetch_today_entries
from predict import allocate_daily_budget
from settle_results import (
    PREDICTION_HISTORY_CSV,
    build_history_html,
    HISTORY_HTML,
    normalize_bets_for_settlement,
    parse_result_page,
    settled_return_yen,
)

OUTPUT_DIR = Path("outputs")


def snapshot_time(path):
    m = re.search(r"bets_(\d{8})_(\d{6})\.csv$", path.name)
    return m.group(1) + m.group(2) if m else ""


def load_final_snapshots(start="20260919", end="20260925"):
    frames = []
    for name in glob.glob("outputs/bets_????????_??????.csv"):
        path = Path(name)
        m = re.search(r"bets_(\d{8})_(\d{6})\.csv$", path.name)
        if not m or not (start <= m.group(1) <= end):
            continue
        try:
            df = pd.read_csv(path, dtype={"race_id": str, "buy": str})
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df.empty or "race_id" not in df.columns:
            continue
        df["_snapshot"] = snapshot_time(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_bets = pd.concat(frames, ignore_index=True, sort=False)
    all_bets = normalize_bets_for_settlement(all_bets)
    all_bets["buy"] = all_bets["buy"].fillna("").astype(str).str.strip()
    all_bets = all_bets[all_bets["buy"].ne("")].copy()
    if all_bets.empty:
        return all_bets

    # Pick the last snapshot that contained an published prediction for each race.
    last = all_bets.groupby(["date", "race_id"], dropna=False)["_snapshot"].transform("max")
    final = all_bets[all_bets["_snapshot"].eq(last)].copy()
    key = [c for c in ["date", "race_id", "bet_type", "buy"] if c in final.columns]
    final = final.drop_duplicates(key, keep="last")

    # Reconstruct missing bankroll allocations with the same production allocator.
    # Allocation is daily: total 10,000 yen, in 100-yen units, max 2,000 yen per ticket.
    final["stake_yen"] = pd.to_numeric(final.get("stake_yen", 0), errors="coerce")
    rebuilt = []
    for _, day in final.groupby("date", sort=False):
        existing = day["stake_yen"].fillna(0)
        if existing.gt(0).any():
            rebuilt.append(day)
            continue
        required = {"expected_profit_100yen", "prob"}
        if required.issubset(day.columns):
            rebuilt.append(allocate_daily_budget(day, budget_yen=10000, max_per_bet_yen=2000))
        else:
            day = day.copy()
            day["stake_yen"] = 0
            rebuilt.append(day)
    return pd.concat(rebuilt, ignore_index=True, sort=False)


def main():
    bets = load_final_snapshots()
    if bets.empty:
        raise SystemExit("no recoverable prediction snapshots found")

    # Re-fetch each historical racecard date to recover the canonical source URL.
    schedule_parts = []
    for race_date in sorted(bets["date"].dropna().astype(str).unique()):
        try:
            entries = fetch_today_entries(race_date=race_date)
            schedule_parts.append(entries[["race_id", "source_url"]].drop_duplicates("race_id"))
        except Exception as exc:
            print(f"historical racecard fetch failed: {race_date} {exc}")
    if schedule_parts:
        schedule = pd.concat(schedule_parts, ignore_index=True).drop_duplicates("race_id", keep="last")
        bets = bets.drop(columns=["source_url"], errors="ignore").merge(schedule, on="race_id", how="left")

    results = []
    for url in bets.get("source_url", pd.Series(dtype=str)).dropna().drop_duplicates():
        try:
            results.append(parse_result_page(url))
        except Exception as exc:
            print(f"result fetch failed: {url} {exc}")

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        bets = bets.merge(result_df, on="race_id", how="left")

    official = bets.get("official_result_available", pd.Series(False, index=bets.index)).fillna(False).astype(bool)
    bets["actual_for_bet_type"] = bets.apply(
        lambda row: row.get(f"actual_{row.get('bet_type', 'trifecta')}", row.get("actual_trifecta", "")), axis=1
    )
    bets["is_decided"] = official & bets["actual_for_bet_type"].fillna("").astype(str).str.len().gt(0)
    bets["display_status"] = bets["is_decided"].map({True: "確定", False: "結果取得待ち"})
    bets["is_hit"] = bets.apply(
        lambda row: bool(row["is_decided"]) and str(row["buy"]) in str(row.get("actual_for_bet_type", "")).split("|"), axis=1
    )
    bets["actual_return_yen"] = [
        settled_return_yen(row, 0) for _, row in bets.iterrows()
    ]
    stake = pd.to_numeric(bets["stake_yen"], errors="coerce").fillna(0)
    bets["actual_profit_yen"] = bets["actual_return_yen"] - stake
    bets.drop(columns=["_snapshot"], errors="ignore").to_csv(PREDICTION_HISTORY_CSV, index=False)
    HISTORY_HTML.write_text(build_history_html(bets), encoding="utf-8")
    print(f"recovered history tickets={len(bets)} races={bets['race_id'].nunique()}")


if __name__ == "__main__":
    main()
