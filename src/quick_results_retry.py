"""Replay already collected official results after a Git race, without HTTP."""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re

from nexus_input_snapshots import ROOT, JST, canonical, read_json, save_json
from nexus_results_sync import _outcome, _timestamp

RESULTS = Path("outputs/latest_results.json")


def today_jst():
    return datetime.now(JST).date().isoformat()


def race_date(row):
    if not isinstance(row, dict) or not re.fullmatch(r"\d{12}", str(row.get("race_id", ""))):
        raise ValueError("valid result race_id required")
    day = datetime.strptime(str(row["race_id"])[-8:], "%Y%m%d").date().isoformat()
    if row.get("date") and row["date"] != day:
        raise ValueError("result date/race identity mismatch")
    return day


def result_rows(path, missing=False):
    rows = read_json(path, [] if missing else None)
    if not isinstance(rows, list):
        raise ValueError("latest_results must be a race list")
    ids = []
    for row in rows:
        race_date(row)
        ids.append(str(row["race_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate official result race")
    return rows


def quality(row):
    outcome = _outcome(row)
    void = row.get("refund_all") is True or row.get("cancelled") is True
    level = 3 if void else 2 if outcome and outcome["payout_complete"] else 1 if outcome else 0
    # Partial tied payoffs also must not lose already known winning amounts.
    partial_payoffs = len(outcome["payouts"]) if level == 1 else 0
    observed = _timestamp(row.get("result_observed_at_jst"))
    return level, partial_payoffs, observed.timestamp() if observed is not None else float("-inf")


def merge_results(current, cached, day):
    """Keep each complete/latest original row intact; never invent observations."""
    merged = {str(row["race_id"]): deepcopy(row) for row in current if race_date(row) == day}
    accepted = []
    for row in cached:
        if race_date(row) != day:
            raise ValueError("cached results contain another date")
        rid = str(row["race_id"])
        if rid not in merged or quality(row) > quality(merged[rid]):
            merged[rid] = deepcopy(row)
            accepted.append(rid)
    return [merged[rid] for rid in sorted(merged)], accepted


def cache_path(cache, root):
    cache, root = Path(cache).resolve(), Path(root).resolve()
    if cache.is_relative_to(root):
        raise ValueError("retry cache must be outside the checkout")
    return cache, root


def snapshot(cache, root=ROOT):
    cache, root = cache_path(cache, root)
    rows = result_rows(root / RESULTS)
    days = {race_date(row) for row in rows}
    if len(days) > 1:
        raise ValueError("collected latest results span multiple dates")
    day = next(iter(days), today_jst())
    from nexus_publish_saved_inputs import copy_newer_inputs
    save_json(cache / RESULTS, rows)
    copy_newer_inputs(root, cache)
    save_json(cache / "retry_manifest.json", dict(schema=1, date=day))
    report = dict(action="snapshot", cache_date=day, races=len(rows))
    print("QUICK_RESULTS_RETRY " + canonical(report), flush=True)
    return report


def apply_offline(root, rows):
    """Use only the existing HTML patch and frozen-forecast result sync hooks."""
    import pandas as pd
    import settle_results
    previous_outputs = settle_results.OUTPUT_DIR
    try:
        settle_results.OUTPUT_DIR = root / "outputs"
        patched = settle_results.patch_index_with_official_results(pd.DataFrame(rows))
        synced = settle_results.sync_nexus_results_after_settlement()
    finally:
        settle_results.OUTPUT_DIR = previous_outputs
    if synced.get("status") != "ok":
        raise RuntimeError("offline NEXUS result sync failed: " + str(synced.get("error", synced)))
    return patched, synced


def replay(cache, root=ROOT):
    cache, root = cache_path(cache, root)
    manifest = read_json(cache / "retry_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("valid retry snapshot required")
    day = manifest["date"]
    cached = result_rows(cache / RESULTS)
    if any(race_date(row) != day for row in cached):
        raise ValueError("retry manifest/result date mismatch")
    current = result_rows(root / RESULTS, missing=True)
    current_days = {race_date(row) for row in current}
    skipped = None
    accepted = []
    if day != today_jst() or any(value > day for value in current_days):
        # A midnight push race may restore a newer day's main. Keep that feed.
        merged = current
        skipped = "cached_result_date_is_not_current"
    else:
        merged, accepted = merge_results(current, cached, day)
        if merged != current:
            save_json(root / RESULTS, merged)
    # Historical raw outcomes can still be retained for the next history update.
    # This never copies a live feed, ledger, prediction, odds or allocation.
    from nexus_publish_saved_inputs import copy_newer_inputs
    copy_newer_inputs(cache, root)
    patched, synced = apply_offline(root, merged)
    report = dict(action="replay", cache_date=day, merged_races=len(merged),
                  accepted_cached_races=accepted, skipped_reason=skipped,
                  index_patched=patched, nexus_sync=synced)
    print("QUICK_RESULTS_RETRY " + canonical(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "replay"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    (snapshot if args.action == "snapshot" else replay)(args.cache, args.root)


if __name__ == "__main__":
    main()
