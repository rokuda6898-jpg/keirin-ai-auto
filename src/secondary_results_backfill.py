"""Backfill last night's unposted trio/exacta/ROI results from original race pages.

The current-day latest_results.json is not overwritten by older races.
Only historical frozen forecasts that still lack an official settlement are
eligible. Never reconstruct payoffs from 3連単 dividends or prediction odds.
"""
import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas as pd

from secondary_results_sync import FEEDS, official_outcome

ROOT = Path(__file__).resolve().parents[1]
JST = ZoneInfo("Asia/Tokyo")
ARCHIVE = Path("outputs/official_results_archive")


def _capture_quality(row):
    if not row or not row.get("official_result_available"):
        return -1
    return sum(int(official_outcome(row, pool) is not None)
               for pool in ("trifecta", "trio", "exacta"))


def _timestamp(value):
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return t if t.tzinfo else None
    except (ValueError, TypeError):
        return None


def _accept(old, new):
    if old is None:
        return True
    old_quality, new_quality = _capture_quality(old), _capture_quality(new)
    if new_quality != old_quality:
        return new_quality > old_quality
    later, earlier = _timestamp(new.get("result_observed_at_jst")), _timestamp(old.get("result_observed_at_jst"))
    return bool(later and (not earlier or later > earlier))


def _safe_url(row):
    url = str(row.get("source_url") or "")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "www.winticket.jp":
        return None
    match = re.fullmatch(r"/keirin/[^/]+/racecard/(\d{10})/(\d+)/(\d+)/?", parsed.path)
    if not match or parsed.query or parsed.fragment:
        return None
    cup, index, race = match.groups()
    if not (str(row.get("cup_id")) == cup and str(row.get("index")) == index
            and str(row.get("race_number")) == race):
        return None
    return url


def pending_races(root=ROOT, now=None, lookback_days=2):
    root = Path(root)
    now = now or datetime.now(JST)
    pending = {}
    for name, (pool, _) in FEEDS.items():
        path = root / "outputs" / (name + "_live.json")
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("races", []):
            if not isinstance(row, dict) or not isinstance(row.get("forecast"), dict):
                continue
            if row.get("settlement", {}).get("status") in ("hit", "miss", "void") if isinstance(row.get("settlement"), dict) else False:
                continue
            try:
                date = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
                close = datetime.fromtimestamp(float(row["close_at"]), JST)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not 0 < (now.date() - date).days <= lookback_days or close + timedelta(minutes=10) > now:
                continue
            url = _safe_url(row)
            if url is None or str(row.get("race_id", ""))[-8:] != date.strftime("%Y%m%d"):
                continue
            race_id = str(row["race_id"])
            if race_id in pending and pending[race_id]["url"] != url:
                raise ValueError("conflicting public race source URL for " + race_id)
            slot = pending.setdefault(race_id, {"date": date.isoformat(), "url": url,
                                               "close_at": close.timestamp(), "pools": set()})
            slot["pools"].add(pool)
    return pending


def backfill(root=ROOT, *, now=None, lookback_days=2, limit=30, parse=None):
    root = Path(root)
    now = now or datetime.now(JST)
    pending = pending_races(root, now=now, lookback_days=lookback_days)
    if parse is None:
        from settle_results import parse_result_page
        parse = parse_result_page
    processed, confirmed, failures = [], [], []
    archives = {}
    for race_id, detail in sorted(pending.items(), key=lambda v: (v[1]["close_at"], v[0])):
        day = detail["date"]
        path = root / ARCHIVE / (day + ".json")
        if day not in archives:
            content = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            if not isinstance(content, list):
                raise ValueError("official archive must contain a race list")
            archives[day] = {str(row["race_id"]): row for row in content if isinstance(row, dict)}
        rows = archives[day]
        previous = rows.get(race_id)
        if previous and all(official_outcome(previous, pool) is not None for pool in detail["pools"]):
            continue
        if len(processed) >= limit:
            break
        processed.append(race_id)
        try:
            outcome = parse(detail["url"])
            if not isinstance(outcome, dict) or str(outcome.get("race_id")) != race_id:
                raise ValueError("fetched race identity does not match the immutable forecast")
            if not outcome.get("official_result_available"):
                continue
            # A provider result may confirm a finish before publishing all
            # winning dividends; only store complete, useful official evidence.
            if not any(official_outcome(outcome, pool) is not None for pool in detail["pools"]):
                continue
            captured = json.loads(pd.DataFrame([outcome]).to_json(orient="records", force_ascii=False))[0]
            if _accept(previous, captured):
                rows[race_id] = captured
                confirmed.append(race_id)
        except Exception as exc:
            failures.append({"race_id": race_id, "reason": type(exc).__name__})
    for day, records in archives.items():
        path = root / ARCHIVE / (day + ".json")
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        result = [records[k] for k in sorted(records)]
        if result != old:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp.replace(path)
    report = {"pending_source_races": len(pending), "attempted": len(processed),
              "saved_official": len(confirmed), "failures": failures[:15]}
    print("SECONDARY_RESULT_BACKFILL " + json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=2)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    backfill(lookback_days=args.lookback_days, limit=args.limit)
