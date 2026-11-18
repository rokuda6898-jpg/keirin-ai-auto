"""Validated, explicitly configured racecard CSV source.

This is a transport for a user-provided/licensed source, not an attempt to
work around a provider's HTTP 403 or to scrape third-party pages. Never treat
a previous-day snapshot or partial field as today's forecast input.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from common import RAW_DIR, OUTPUT_DIR, TODAY_CSV, TODAY_ODDS_CSV, ensure_dirs
from fetch_today_entries import _entry_rows_complete, ODDS_COLUMNS, TRIFECTA_ODDS_COLUMNS

JST = ZoneInfo("Asia/Tokyo")
SCHEDULE = OUTPUT_DIR / "latest_race_schedule.csv"
TRIFECTA = RAW_DIR / "today_trifecta_odds.csv"
MAX_BYTES = 20 * 1024 * 1024
REQUIRED = {
    "race_id", "date", "venue", "race_no", "player_id", "car_no",
    "entries_number", "start_at", "close_at", "source_url", "player_name",
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate(frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if now.tzinfo is None:
        raise ValueError("timezone-aware clock required")
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"authorized today CSV missing columns: {missing}")
    if frame.empty:
        raise ValueError("authorized today CSV is empty")
    today = now.astimezone(JST).date().isoformat()
    if not frame["date"].astype(str).eq(today).all():
        raise ValueError(f"authorized today CSV is not exclusively today's data ({today})")
    if frame["race_id"].isna().any() or not frame["race_id"].astype(str).str.fullmatch(r"\d{12}").all():
        raise ValueError("invalid race_id")
    if frame["player_id"].isna().any() or not frame["player_id"].astype(str).str.fullmatch(r"\d{6}").all():
        raise ValueError("missing or invalid player_id")
    if frame["player_name"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("missing player name")
    if frame["venue"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("missing venue")
    if not frame["source_url"].fillna("").astype(str).str.startswith("https://").all():
        raise ValueError("missing HTTPS provenance for authorized racecard")
    if "finish_pos" in frame.columns and pd.to_numeric(frame["finish_pos"], errors="coerce").notna().any():
        raise ValueError("result-bearing source cannot be used as pre-race input")
    if "official_finish_pos" in frame.columns and pd.to_numeric(frame["official_finish_pos"], errors="coerce").notna().any():
        raise ValueError("official result-bearing source cannot be used as pre-race input")
    if "result_available" in frame.columns:
        flags = frame["result_available"].fillna("").astype(str).str.lower().str.strip()
        if not flags.isin(["", "0", "0.0", "false", "nan"]).all():
            raise ValueError("already-settled data cannot be used as pre-race input")
    for name in ("car_no", "race_no", "entries_number", "start_at", "close_at"):
        nums = pd.to_numeric(frame[name], errors="coerce")
        if nums.isna().any():
            raise ValueError(f"invalid {name}")
    if frame.duplicated(["race_id", "car_no"]).any() or frame.duplicated(["race_id", "player_id"]).any():
        raise ValueError("duplicate racer in authorized today CSV")

    for race_id, group in frame.groupby("race_id", sort=False):
        if group["venue"].nunique() != 1 or group["race_no"].nunique() != 1:
            raise ValueError(f"inconsistent race identity: {race_id}")
        if group["close_at"].nunique() != 1 or group["start_at"].nunique() != 1:
            raise ValueError(f"inconsistent race time: {race_id}")
        start = float(group["start_at"].iloc[0])
        close = float(group["close_at"].iloc[0])
        if start <= close or start - close > 3600:
            raise ValueError(f"invalid deadline: {race_id}")
        if datetime.fromtimestamp(close, JST).date().isoformat() != today:
            raise ValueError(f"deadline/date mismatch: {race_id}")
        if not (1 <= int(group["race_no"].iloc[0]) <= 12):
            raise ValueError(f"invalid race number: {race_id}")
        complete, _, declared = _entry_rows_complete(group.to_dict("records"))
        if not complete or len(group) != declared or len(group) < 5 or len(group) > 9:
            raise ValueError(f"partial rider list: {race_id}")
    return frame


def load(url: str, now: datetime | None = None, requester=requests.get) -> pd.DataFrame:
    """Import a complete race day from an owner-configured HTTPS CSV URL.

    The operator must have the right to use this feed in a commercial product.
    A failed validation never overwrites a previously committed snapshot.
    """
    ensure_dirs()
    now = now or datetime.now(JST)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("authorized CSV URL must be HTTPS with no embedded credentials")
    response = requester(url, timeout=45)
    response.raise_for_status()
    payload = response.content
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError("authorized CSV is empty or exceeds 20 MB")
    frame = pd.read_csv(io.BytesIO(payload), dtype={"race_id": str, "player_id": str}, low_memory=False)
    frame = validate(frame, now)
    # Never recycle previous-day odds as today's live quotes.
    odds = pd.DataFrame(columns=ODDS_COLUMNS)
    trifecta = pd.DataFrame(columns=TRIFECTA_ODDS_COLUMNS)
    columns = ["date", "venue", "race_no", "race_id", "start_at", "close_at", "source_url"]
    schedule = frame[columns].drop_duplicates("race_id")
    schedule = schedule.sort_values(["close_at", "venue", "race_no"]).reset_index(drop=True)
    schedule["schedule_fetched_at_jst"] = now.astimezone(JST).isoformat(timespec="seconds")

    _atomic_csv(frame, TODAY_CSV)
    _atomic_csv(odds, TODAY_ODDS_CSV)
    _atomic_csv(trifecta, TRIFECTA)
    _atomic_csv(schedule, SCHEDULE)
    metadata = {
        "source": "operator_configured_authorized_csv",
        "source_host": parsed.hostname,
        "date": now.astimezone(JST).date().isoformat(),
        "fetched_at_jst": now.astimezone(JST).isoformat(timespec="seconds"),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "races": int(frame["race_id"].nunique()),
        "entry_rows": len(frame),
        "odds_rows": 0,
        "market_status": "unavailable; no odds inferred",
    }
    (RAW_DIR / "today_entries_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("AUTHORIZED_RACECARD " + json.dumps(metadata, ensure_ascii=False), flush=True)
    return frame
