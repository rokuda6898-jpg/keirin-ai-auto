import argparse
import json
import os
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from common import ensure_dirs, OUTPUT_DIR, RAW_DIR, TODAY_CSV, TODAY_ODDS_CSV
from race_features import build_entry_rows

BASE_URL = "https://www.winticket.jp"
RACECARD_URL = f"{BASE_URL}/keirin/racecard"
TRIFECTA_ODDS_CSV = RAW_DIR / "today_trifecta_odds.csv"
RACE_SCHEDULE_CSV = OUTPUT_DIR / "latest_race_schedule.csv"
RACE_CACHE_DIR = RAW_DIR / "race_cache"
ODDS_COLUMNS = [
    "date",
    "venue",
    "race_no",
    "race_id",
    "bet_type",
    "buy",
    "odds",
    "min_odds",
    "max_odds",
    "odds_used",
    "popularity_order",
    "source_url",
]
TRIFECTA_ODDS_COLUMNS = ["date", "venue", "race_no", "race_id", "buy", "trifecta_odds", "popularity_order", "source_url"]
BET_SPECS = {
    # Keep this only as an optional source: the current WINTICKET race odds
    # payload often omits the win pool entirely. Never derive it from other pools.
    "win": {"source": "win", "key_len": 1, "ordered": True},
    "trifecta": {"source": "trifecta", "key_len": 3, "ordered": True},
    "trio": {"source": "trio", "key_len": 3, "ordered": False},
    "exacta": {"source": "exacta", "key_len": 2, "ordered": True},
    "quinella": {"source": "quinella", "key_len": 2, "ordered": False},
    "quinella_place": {"source": "quinellaPlace", "key_len": 2, "ordered": False},
    "bracket_exacta": {"source": "bracketExacta", "key_len": 2, "ordered": True},
    "bracket_quinella": {"source": "bracketQuinella", "key_len": 2, "ordered": False},
}


_HTTP_RATE_LOCK = threading.Lock()
_HTTP_NEXT_REQUEST_AT = 0.0


def wait_for_http_slot():
    global _HTTP_NEXT_REQUEST_AT
    try:
        interval = max(float(os.getenv("WINTICKET_MIN_REQUEST_INTERVAL_SEC", "0.25")), 0.0)
    except ValueError:
        interval = 0.25
    if interval == 0:
        return
    with _HTTP_RATE_LOCK:
        now = time.monotonic()
        request_at = max(now, _HTTP_NEXT_REQUEST_AT)
        _HTTP_NEXT_REQUEST_AT = request_at + interval
    delay = request_at - now
    if delay > 0:
        time.sleep(delay)


def http_get(url, attempts=7):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; keirin-ai-auto/1.0)",
        "Accept-Language": "ja,en;q=0.8",
    }
    attempts = max(int(attempts), 1)
    for attempt in range(attempts):
        wait_for_http_slot()
        response = requests.get(url, headers=headers, timeout=30)
        retryable = response.status_code == 429 or 500 <= response.status_code < 600
        if retryable and attempt + 1 < attempts:
            retry_after = response.headers.get("Retry-After", "")
            try:
                wait_seconds = max(float(retry_after), 0.0)
            except ValueError:
                wait_seconds = 0.0
            wait_seconds = max(wait_seconds, min(2**attempt, 30)) + random.uniform(0.0, 0.5)
            print(
                f"retry http {response.status_code}: attempt={attempt + 1}/{attempts} "
                f"wait={wait_seconds:.1f}s url={url}",
                flush=True,
            )
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text
    raise RuntimeError(f"unreachable HTTP retry state: {url}")


def extract_preloaded_state(html):
    m = re.search(r"window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});\s*window\.__CONFIG__", html, re.S)
    if not m:
        raise ValueError("WINTICKET preloaded state was not found")
    return json.loads(m.group(1))


def tanstack_queries(state):
    return state.get("tanStackQuery", {}).get("queries", [])


def find_query_data(state, marker):
    for q in tanstack_queries(state):
        if marker in json.dumps(q.get("queryKey"), ensure_ascii=False):
            data = q.get("state", {}).get("data")
            if isinstance(data, dict):
                return data
    return {}


def collect_race_links(index_html, race_date):
    links = set()
    for href in re.findall(r'href="([^"]+)"', index_html):
        if re.fullmatch(r"/keirin/[^/]+/racecard/\d{10}/\d+/\d+", href):
            links.add(urljoin(BASE_URL, href))
    return sorted(links)


def normalize_date(yyyymmdd):
    s = str(yyyymmdd)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def recent_avg_finish(record):
    orders = []
    for key in ["currentCupResults", "previousCupResults"]:
        for result in record.get(key, []) or []:
            order = result.get("order")
            if isinstance(order, (int, float)) and order > 0:
                orders.append(order)
            if len(orders) >= 10:
                break
        if len(orders) >= 10:
            break
    return float(np.mean(orders)) if orders else np.nan


def days_since_last_race(record, race_date):
    dates = []
    for key in ["currentCupResults", "previousCupResults"]:
        for result in record.get(key, []) or []:
            m = re.search(r"(20\d{6})", str(result.get("raceId", "")))
            if m:
                dates.append(datetime.strptime(m.group(1), "%Y%m%d").date())
    if not dates:
        return np.nan
    current = datetime.strptime(race_date, "%Y-%m-%d").date()
    before = [d for d in dates if d <= current]
    if not before:
        return np.nan
    return max((current - max(before)).days, 0)


def get_venue_name(state, race_data):
    cup_data = find_query_data(state, "FETCH_KEIRIN_CUP_RACES")
    venue = cup_data.get("venue")
    if isinstance(venue, dict) and venue.get("name"):
        return venue["name"]

    venue_id = None
    schedule = race_data.get("schedule", {})
    cup_id = schedule.get("cupId")
    for cup in race_data.get("cups", []):
        if cup.get("id") == cup_id:
            venue_id = cup.get("venueId")
            break
    venue_list = find_query_data(state, "FETCH_KEIRIN_VENUE_LIST")
    for venue in venue_list.get("venues", []):
        if str(venue.get("id")) == str(venue_id):
            return venue.get("name", "")
    for region in venue_list.get("regions", []):
        for venue in region.get("venues", []):
            if str(venue.get("id")) == str(venue_id):
                return venue.get("name", "")
    return ""


def key_to_buy(key, ordered=True):
    values = [int(x) for x in key]
    if not ordered:
        values = sorted(values)
    return "-".join(str(x) for x in values)


def build_odds_rows(odds_data, race_date, venue, race_no, race_id, url):
    rows = []
    for bet_type, spec in BET_SPECS.items():
        for item in odds_data.get(spec["source"], []) or []:
            key = item.get("key", [])
            if len(key) != spec["key_len"] or item.get("absent"):
                continue
            if bet_type == "trifecta":
                if str(item.get("raceId", "")) != str(race_id):
                    continue
                if any(not isinstance(x, int) or not 1 <= x <= 9 for x in key) or len(set(key)) != 3:
                    continue
            odds = pd.to_numeric(item.get("odds"), errors="coerce")
            min_odds = pd.to_numeric(item.get("minOdds"), errors="coerce")
            max_odds = pd.to_numeric(item.get("maxOdds"), errors="coerce")
            odds_used = odds
            if bet_type == "quinella_place" and (pd.isna(odds_used) or odds_used <= 0):
                odds_used = min_odds
            rows.append(
                {
                    "date": race_date,
                    "venue": venue,
                    "race_no": race_no,
                    "race_id": race_id,
                    "bet_type": bet_type,
                    "buy": key_to_buy(key, ordered=spec.get("ordered", True)),
                    "odds": odds,
                    "min_odds": min_odds,
                    "max_odds": max_odds,
                    "odds_used": odds_used,
                    "popularity_order": item.get("popularityOrder", np.nan),
                    "source_url": url,
                }
            )
    return rows


def _entry_rows_complete(rows):
    """Validate fetched starters against the source-declared field size."""
    if not rows:
        return False, [], 0
    frame = pd.DataFrame(rows)
    cars = sorted(pd.to_numeric(frame.get("car_no"), errors="coerce").dropna().astype(int).unique().tolist())
    expected_values = pd.to_numeric(frame.get("entries_number"), errors="coerce").dropna()
    expected = int(expected_values.max()) if len(expected_values) else 0
    if expected <= 0:
        return False, cars, expected
    if "declared_entries_number" in frame.columns:
        declared_values = pd.to_numeric(frame["declared_entries_number"], errors="coerce").dropna()
    else:
        declared_values = pd.Series(dtype=float)
    declared = int(declared_values.max()) if len(declared_values) else expected
    cancelled = set()
    if "cancelled_car_numbers" in frame.columns:
        for value in frame["cancelled_car_numbers"].dropna().astype(str).unique():
            cancelled.update(int(x) for x in re.findall(r"\d+", value))
    expected_active = set(range(1, declared + 1)) - cancelled
    complete = (
        expected > 0
        and len(cars) == expected
        and set(cars) == expected_active
        and expected + len(cancelled) == declared
    )
    return complete, cars, expected


def parse_race_page(url, completeness_attempts=5, completeness_retry_sec=5.0):
    last_incomplete = None
    for completeness_attempt in range(max(int(completeness_attempts), 1)):
        html = http_get(url)
        state = extract_preloaded_state(html)
        race_data = find_query_data(state, "FETCH_KEIRIN_RACE")
        if not race_data:
            raise ValueError(f"race data not found: {url}")

        schedule = race_data["schedule"]
        race = race_data["race"]
        race_date = normalize_date(schedule["date"])
        venue = get_venue_name(state, race_data)
        race_no = int(race["number"])
        race_id = str(race["id"])

        raw_entries = race_data.get("entries", []) or []
        declared = int(race.get("entriesNumber") or len(raw_entries) or 0)
        raw_cars = sorted(
            int(entry.get("number")) for entry in raw_entries
            if entry.get("number") is not None
        )
        if declared <= 0 or len(raw_cars) != declared or set(raw_cars) != set(range(1, declared + 1)):
            raise ValueError(
                f"incomplete raw race field: race_id={race_id} declared={declared} "
                f"raw_cars={raw_cars} url={url}"
            )

        entry_rows = build_entry_rows(
            race_data,
            race_date=race_date,
            venue=venue,
            race_no=race_no,
            race_id=race_id,
            source_url=url,
            include_results=False,
        )
        complete, cars, expected = _entry_rows_complete(entry_rows)
        if complete:
            odds_data = find_query_data(state, "FETCH_KEIRIN_RACE_ODDS")
            odds_rows = build_odds_rows(odds_data, race_date, venue, race_no, race_id, url)
            from market_odds_sources import verify_market
            odds_rows = verify_market(entry_rows, odds_rows)
            return entry_rows, odds_rows

        last_incomplete = (
            f"incomplete race field: race_id={race_id} expected={expected} "
            f"cars={cars} attempt={completeness_attempt + 1}/{completeness_attempts} url={url}"
        )
        if completeness_attempt + 1 < completeness_attempts:
            time.sleep(completeness_retry_sec)

    raise ValueError(last_incomplete or f"incomplete race field: {url}")



def _atomic_csv_write(frame, path):
    """Write a CSV atomically so a killed refresh cannot leave a half-written cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def cache_complete_races(entries_df, odds_df):
    """Persist each complete race independently so one bad refresh cannot erase it."""
    RACE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for race_id, group in entries_df.groupby("race_id", sort=False):
        expected = pd.to_numeric(group.get("entries_number"), errors="coerce").dropna()
        expected_n = int(expected.iloc[0]) if len(expected) else len(group)
        cars = sorted(pd.to_numeric(group["car_no"], errors="coerce").dropna().astype(int).unique().tolist())
        complete, _, _ = _entry_rows_complete(group.to_dict("records"))
        if not complete:
            continue
        rid = str(race_id)
        _atomic_csv_write(group, RACE_CACHE_DIR / f"{rid}_entries.csv")
        race_odds = odds_df[odds_df["race_id"].astype(str).eq(rid)] if len(odds_df) else odds_df
        _atomic_csv_write(race_odds, RACE_CACHE_DIR / f"{rid}_odds.csv")


def cache_complete_race_rows(entry_rows, odds_rows):
    """Checkpoint one validated race immediately, even if another race later fails."""
    if not entry_rows:
        return
    entries_df = pd.DataFrame(entry_rows)
    odds_df = pd.DataFrame(odds_rows, columns=ODDS_COLUMNS)
    cache_complete_races(entries_df, odds_df)


def load_cached_race(race_id):
    entry_path = RACE_CACHE_DIR / f"{race_id}_entries.csv"
    odds_path = RACE_CACHE_DIR / f"{race_id}_odds.csv"
    if not entry_path.exists():
        return None, None
    try:
        entries = pd.read_csv(entry_path, dtype={"race_id": str})
        complete, _, _ = _entry_rows_complete(entries.to_dict("records"))
        if not complete:
            return None, None
        odds = pd.read_csv(odds_path, dtype={"race_id": str}) if odds_path.exists() else pd.DataFrame(columns=ODDS_COLUMNS)
        return entries, odds
    except Exception:
        return None, None


def guard_against_destructive_shrink(entries_df):
    """Refuse a snapshot that would catastrophically shrink an existing same-day set."""
    if not TODAY_CSV.exists():
        return
    try:
        old = pd.read_csv(TODAY_CSV, dtype={"race_id": str})
    except Exception:
        return
    if old.empty or "race_id" not in old.columns or entries_df.empty:
        return
    old_races = old["race_id"].astype(str).nunique()
    new_races = entries_df["race_id"].astype(str).nunique()
    # A full-day replacement should never collapse a healthy snapshot by more
    # than 20%. This is a final write barrier in addition to fetch completeness.
    if old_races >= 10 and new_races < max(1, int(old_races * 0.80)):
        raise ValueError(f"destructive snapshot shrink blocked: old_races={old_races} new_races={new_races}")




def merge_existing_same_day_odds(entries_df, fresh_odds_df, existing_path=None):
    """Keep last-known same-day odds when a refresh returns no odds for a race."""
    path = Path(existing_path) if existing_path is not None else TODAY_ODDS_CSV
    fresh = fresh_odds_df.copy()
    if not path.exists():
        return fresh
    try:
        old = pd.read_csv(path, dtype={"race_id": str})
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return fresh
    if old.empty:
        return fresh

    if "date" in entries_df.columns and "date" in old.columns:
        current_dates = set(entries_df["date"].dropna().astype(str))
        old = old[old["date"].astype(str).isin(current_dates)].copy()
    if old.empty:
        return fresh
    if fresh.empty:
        return old.reset_index(drop=True)

    if "race_id" in fresh.columns and "race_id" in old.columns:
        fresh_ids = set(fresh["race_id"].dropna().astype(str))
        old = old[~old["race_id"].astype(str).isin(fresh_ids)].copy()
    merged = pd.concat([old, fresh], ignore_index=True, sort=False)
    key = [c for c in ["race_id", "bet_type", "buy"] if c in merged.columns]
    if key:
        merged = merged.drop_duplicates(key, keep="last")
    return merged.reset_index(drop=True)


def save_today_frames(all_entries, all_odds):
    entries_df = pd.DataFrame(all_entries).sort_values(["date", "venue", "race_no", "car_no"])
    odds_df = pd.DataFrame(all_odds, columns=ODDS_COLUMNS)
    odds_df = merge_existing_same_day_odds(entries_df, odds_df)
    if len(odds_df):
        odds_df = odds_df.sort_values(["date", "venue", "race_no", "bet_type", "popularity_order", "buy"])
    odds_df.to_csv(TODAY_ODDS_CSV, index=False)

    # A genuine single-win quote may be present in the same odds endpoint.
    # If absent, leave odds_win missing; do not infer it from quinella/etc.
    if len(odds_df):
        win_odds = odds_df.loc[odds_df["bet_type"].eq("win"), ["race_id", "buy", "odds_used"]].copy()
        if len(win_odds):
            win_odds["car_no"] = pd.to_numeric(win_odds["buy"], errors="coerce")
            win_odds["odds_win"] = pd.to_numeric(win_odds["odds_used"], errors="coerce")
            win_odds = win_odds.dropna(subset=["car_no", "odds_win"]).drop_duplicates(["race_id", "car_no"], keep="last")
            entries_df = entries_df.drop(columns=["odds_win"], errors="ignore").merge(
                win_odds[["race_id", "car_no", "odds_win"]], on=["race_id", "car_no"], how="left"
            )
    if "odds_win" not in entries_df.columns:
        entries_df["odds_win"] = np.nan

    guard_against_destructive_shrink(entries_df)
    cache_complete_races(entries_df, odds_df)
    entries_df.to_csv(TODAY_CSV, index=False)

    trifecta_df = odds_df[odds_df["bet_type"].eq("trifecta")].copy() if len(odds_df) else pd.DataFrame(columns=ODDS_COLUMNS)
    if len(trifecta_df):
        trifecta_df["trifecta_odds"] = trifecta_df["odds_used"]
        trifecta_df = trifecta_df[TRIFECTA_ODDS_COLUMNS]
    else:
        trifecta_df = pd.DataFrame(columns=TRIFECTA_ODDS_COLUMNS)
    trifecta_df.to_csv(TRIFECTA_ODDS_CSV, index=False)
    return entries_df, odds_df, trifecta_df


def _race_id_from_link(link):
    m = re.search(r"/racecard/(\\d{10})/\\d+/\\d+", str(link))
    return m.group(1) if m else ""


def write_source_attempt_log(race_date, attempts):
    """Keep provider outcomes so fallback order can adapt to real failures."""
    path = RAW_DIR / "source_attempts.jsonl"
    checked_at = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as fh:
        for item in attempts:
            row = {"checked_at_jst": checked_at, "date": race_date, **item}
            fh.write(json.dumps(row, ensure_ascii=False) + "\\n")


def cached_race_ids_for_date(race_date):
    ids = set()
    if not RACE_CACHE_DIR.exists():
        return ids
    compact_date = race_date.replace("-", "")
    for path in RACE_CACHE_DIR.glob("*_entries.csv"):
        rid = path.name.removesuffix("_entries.csv")
        if compact_date not in rid:
            continue
        entries, _ = load_cached_race(rid)
        if entries is not None:
            ids.add(rid)
    return ids


def fetch_today_entries(race_date=None, max_races=None, sleep_sec=0.2):
    ensure_dirs()
    race_date = race_date or datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    index_html = http_get(RACECARD_URL)
    links = collect_race_links(index_html, race_date)
    if max_races:
        links = links[:max_races]
    if not links:
        raise ValueError(f"no WINTICKET racecard links found for {race_date}")

    all_entries = []
    all_odds = []
    failures = []
    source_attempts = []
    for i, link in enumerate(links, start=1):
        try:
            # One provider attempt per race. If it is incomplete, keep the
            # already-good cache and hand this race to the next provider layer.
            entries, odds = parse_race_page(link, completeness_attempts=1)
            entries = [r for r in entries if r.get("date") == race_date]
            odds = [r for r in odds if r.get("date") == race_date]
            if not entries:
                print(f"skipped {i}/{len(links)}: {link} date mismatch")
                continue
            # Checkpoint each good race immediately. A later failure must not
            # throw away the races already obtained successfully.
            cache_complete_race_rows(entries, odds)
            all_entries.extend(entries)
            all_odds.extend(odds)
            source_attempts.append({"provider": "winticket", "race_id": str(entries[0].get("race_id", "")), "status": "complete"})
            print(f"fetched {i}/{len(links)}: {link} entries={len(entries)} odds={len(odds)}")
        except Exception as e:
            race_id = _race_id_from_link(link)
            failures.append({"url": link, "race_id": race_id, "error": str(e)})
            source_attempts.append({"provider": "winticket", "race_id": race_id, "status": "incomplete", "error": str(e)})
            print(f"failed {i}/{len(links)}: {link} error={e}")
        time.sleep(sleep_sec)

    write_source_attempt_log(race_date, source_attempts)
    if failures:
        cached = cached_race_ids_for_date(race_date)
        unresolved = [f for f in failures if str(f.get("race_id", "")) not in cached]
        print(
            f"fallback required: provider=winticket failures={len(failures)} "
            f"already_cached={len(failures) - len(unresolved)} unresolved={len(unresolved)}",
            flush=True,
        )

    if not all_entries:
        raise ValueError(f"failed to fetch any entries for {race_date}: {failures[:3]}")
    # Atomic daily snapshot: never publish a partial day. Previously a single
    # incomplete race could fail while all other races were saved, silently
    # turning a rider-level defect into a whole-race omission.
    if failures:
        raise ValueError(
            f"refusing partial daily snapshot for {race_date}: "
            f"{len(failures)}/{len(links)} races failed; first={failures[:3]}"
        )

    entries_df, odds_df, trifecta_df = save_today_frames(all_entries, all_odds)
    schedule_columns = ["date", "venue", "race_no", "race_id", "start_at", "close_at", "source_url"]
    schedule = entries_df[schedule_columns].drop_duplicates("race_id").sort_values(["date", "close_at", "venue", "race_no"])
    schedule["schedule_fetched_at_jst"] = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
    schedule.to_csv(RACE_SCHEDULE_CSV, index=False)

    metadata = {
        "source": "WINTICKET racecard",
        "source_url": RACECARD_URL,
        "date": race_date,
        "fetched_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "races": len(links),
        "entry_rows": int(len(entries_df)),
        "odds_rows": int(len(odds_df)),
        "bet_types": sorted(odds_df["bet_type"].dropna().unique().tolist()) if len(odds_df) else [],
        "trifecta_odds_rows": int(len(trifecta_df)),
        "failures": failures,
        "race_schedule": str(RACE_SCHEDULE_CSV),
    }
    (RAW_DIR / "today_entries_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return entries_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Race date in YYYY-MM-DD. Defaults to today in Asia/Tokyo.")
    parser.add_argument("--max-races", type=int, default=None)
    args = parser.parse_args()
    fetch_today_entries(args.date, args.max_races)


if __name__ == "__main__":
    main()
