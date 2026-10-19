"""Keep exact NEXUS inputs from page responses the existing collectors obtained.

This module makes no network requests. It must never be used to retry a denied
source through another route. CSV summaries are not accepted as model inputs.
"""
import copy
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = Path("data/raw/nexus_page_snapshots")
JST = ZoneInfo("Asia/Tokyo")
RECORD_FIELDS = ("racePoint firstRate secondRate thirdRate standing back "
                 "frontRunner stalker deepCloser marker first second third others gearRatio style").split()


def now_jst():
    return datetime.now(JST)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def timestamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("timezone-aware capture required")
    return stamp


def query_data(state):
    if not isinstance(state, dict):
        raise ValueError("page state is not an object")
    rows = state.get("tanStackQuery", {}).get("queries", [])
    return [(json.dumps(q.get("queryKey"), ensure_ascii=False), q.get("state", {}).get("data"))
            for q in rows if isinstance(q, dict)]


def entity(data_rows, singular, plural, identifier):
    for _, data in data_rows:
        if not isinstance(data, dict):
            continue
        candidates = [data, data.get(singular)] + list(data.get(plural, []) or [])
        if plural == "venues":
            for region in data.get("regions", []) or []:
                candidates += list(region.get("venues", []) or [])
        for item in candidates:
            if isinstance(item, dict) and str(item.get("id")) == str(identifier):
                if singular == "cup" and "grade" not in item:
                    continue
                if singular == "venue" and "name" not in item:
                    continue
                return item
    raise ValueError("source " + singular + " metadata missing")


def snapshot_from_state(state, source_url, response_text, captured_at):
    """Validate identity and preserve source units, contested slots and absences."""
    captured = timestamp(captured_at)
    parsed = urlsplit(source_url)
    match = re.fullmatch(r"/keirin/[^/]+/racecard/(\d{10})/(\d+)/(\d+)/?", parsed.path)
    if parsed.scheme != "https" or parsed.hostname != "www.winticket.jp" or not match:
        raise ValueError("unrecognized existing racecard URL")
    cup_id, index, number = match.groups()
    rows = query_data(state)
    candidates = [d for marker, d in rows if "FETCH_KEIRIN_RACE" in marker
                  and isinstance(d, dict) and isinstance(d.get("race"), dict)
                  and isinstance(d.get("schedule"), dict) and "entries" in d]
    if len(candidates) != 1:
        raise ValueError("unique original race data required")
    original = candidates[0]
    race, schedule = original["race"], original["schedule"]
    rid = str(race.get("id", ""))
    ds = str(schedule.get("date", ""))
    if not re.fullmatch(r"\d{8}", ds):
        raise ValueError("invalid source race date")
    race_date = datetime.strptime(ds, "%Y%m%d").date().isoformat()
    if (not rid.isdigit() or str(schedule.get("cupId")) != cup_id
            or schedule.get("index") != int(index) or race.get("number") != int(number)):
        raise ValueError("page/race/meeting identity mismatch")
    close, start = race.get("closeAt"), race.get("startAt")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in (close, start)) or close > start:
        raise ValueError("valid source start and close times required")
    if datetime.fromtimestamp(start, JST).date().isoformat() != race_date:
        raise ValueError("source start date mismatch")
    cup = entity(rows + [("race", original)], "cup", "cups", cup_id)
    venue = entity(rows + [("race", original)], "venue", "venues", cup.get("venueId"))
    if not isinstance(race.get("class"), str) or not isinstance(race.get("raceType"), str):
        raise ValueError("source race class/stage missing")
    if type(race.get("isGradeRace")) is not bool or type(race.get("cancel")) is not bool:
        raise ValueError("explicit race grade/cancellation flags required")
    if isinstance(cup.get("grade"), bool) or not isinstance(cup.get("grade"), (int, float)):
        raise ValueError("source cup grade missing")
    meta = dict(race_id=rid, date=race_date, cup_id=cup_id, index=int(index),
                venue_id=str(cup["venueId"]).zfill(2), venue=venue["name"],
                race_class=race["class"], race_type=race["raceType"],
                graded=race["isGradeRace"], cup_grade=cup["grade"], cancel=race["cancel"],
                race_number=int(number), start_at=start, close_at=close,
                source_url=source_url, session=None)
    entries = original.get("entries")
    players, records = original.get("players"), original.get("records")
    if not all(isinstance(x, list) for x in (entries, players, records)):
        raise ValueError("original entries, players and records required")
    # Existing collectors explicitly recognize this non-rider placeholder.
    entries = [e for e in entries if not (e.get("number") == 0 and e.get("absent") is True)]
    declared = race.get("entriesNumber")
    cars = [e.get("number") for e in entries]
    if (type(declared) is not int or not 3 <= declared <= 9
            or any(type(c) is not int for c in cars)
            or sorted(cars) != list(range(1, declared + 1))):
        raise ValueError("incomplete original field")
    if any(type(e.get("absent")) is not bool or str(e.get("raceId")) != rid
           or not str(e.get("playerId", "")).isdigit() for e in entries):
        raise ValueError("invalid entry identity or unknown absence")
    active = [e for e in entries if e["absent"] is False]
    pids = [str(p.get("id")) for p in players]
    record_ids = [str(r.get("playerId")) for r in records]
    active_ids = [str(e["playerId"]) for e in active]
    if (len(pids) != len(set(pids)) or len(record_ids) != len(set(record_ids))
            or len(active_ids) != len(set(active_ids))
            or not set(active_ids).issubset(pids) or not set(active_ids).issubset(record_ids)):
        raise ValueError("missing or duplicate original player/record identity")
    if not race["cancel"] and len(active) < 3:
        raise ValueError("fewer than three active riders")
    # Retain only fields consumed by the frozen adapter/history updater. Values
    # remain exactly as supplied; no rates/counts/ages or lineup are inferred.
    data = {k: copy.deepcopy(original[k]) for k in ("race", "schedule", "linePrediction", "results")
            if k in original}
    data["entries"] = copy.deepcopy(entries)
    data["players"] = [{k: p[k] for k in ("id", "name", "birthday", "gender", "term") if k in p}
                       for p in players]
    data["records"] = [{k: r[k] for k in ["playerId"] + RECORD_FIELDS if k in r} for r in records]
    odds_candidates = [d for marker, d in rows if "FETCH_KEIRIN_RACE_ODDS" in marker and isinstance(d, dict)]
    odds_data = odds_candidates[0] if len(odds_candidates) == 1 else original
    for field in ("trifecta", "trifectaWinningOddsIds", "trifectaPayoffStatus", "finalOdds"):
        if field in odds_data:
            data[field] = copy.deepcopy(odds_data[field])
        elif field in original:
            data[field] = copy.deepcopy(original[field])
    for q in data.get("trifecta", []):
        if str(q.get("raceId")) != rid:
            raise ValueError("odds race identity mismatch")
    from nexus_model.raw import extract
    if not race["cancel"]:
        extract(data, meta)
    model_data = {k: data[k] for k in ("players", "entries", "records", "linePrediction") if k in data}
    model_data["race"] = {k: race.get(k) for k in ("id", "distance", "startAt", "closeAt")}
    model_data["schedule"] = {k: schedule.get(k) for k in ("date", "cupId", "index", "day")}
    model_meta = {k: v for k, v in meta.items() if k not in ("source_url", "session")}
    input_hash = hashlib.sha256(canonical([model_meta, model_data]).encode()).hexdigest()
    payload = response_text.encode("utf-8") if isinstance(response_text, str) else response_text
    return dict(schema=1, source_kind="existing_public_page_response", captured_at=captured.isoformat(),
                sha256=hashlib.sha256(payload).hexdigest(), input_sha256=input_hash,
                url=source_url, meta=meta, data=data)


def capture_existing_page(state, source_url, response_text, captured_at, root=ROOT):
    """A capture failure is visible without breaking the established collector."""
    try:
        snap = snapshot_from_state(state, source_url, response_text, captured_at)
        meta = snap["meta"]
        preclose = timestamp(snap["captured_at"]).timestamp() < meta["close_at"]
        kind = "preclose" if preclose and not snap["data"]["race"].get("decidedAt") else "outcomes"
        path = Path(root) / SNAPSHOTS / meta["date"] / kind / (meta["race_id"] + ".json")
        old = read_json(path)
        if not old or timestamp(old["captured_at"]) < timestamp(snap["captured_at"]):
            save_json(path, snap)
        return snap
    except Exception as exc:
        print(f"::warning::NEXUS page snapshot not accepted: {source_url}: {type(exc).__name__}: {exc}", flush=True)
        return None


def capture_schedule(rows, captured_at, root=ROOT):
    """Keep the complete collector schedule separately from successful inputs."""
    stamp = timestamp(captured_at)
    bydate = {}
    for row in rows:
        item = dict(row)
        bydate.setdefault(item["date"], []).append(item)
    for ds, items in bydate.items():
        ids = [str(r["race_id"]) for r in items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate scheduled race")
        save_json(Path(root) / SNAPSHOTS / ds / "schedule.json",
                  dict(schema=1, date=ds, captured_at=stamp.isoformat(), races=items))


def capture_expected_schedule(urls, race_date, captured_at, root=ROOT):
    """Record the existing index response's full list before race/market failures."""
    stamp = timestamp(captured_at)
    day = datetime.strptime(race_date, "%Y-%m-%d").date().isoformat()
    unique = sorted(set(urls))
    if not unique or any(not re.fullmatch(
            r"https://www\.winticket\.jp/keirin/[^/]+/racecard/\d{10}/\d+/\d+", url)
            for url in unique):
        raise ValueError("complete existing racecard URL list required")
    path = Path(root) / SNAPSHOTS / day / "index.json"
    old = read_json(path)
    if not old or timestamp(old["captured_at"]) < stamp:
        save_json(path, dict(schema=1, date=day, captured_at=stamp.isoformat(),
                             source_kind="existing_public_index_response", urls=unique))
