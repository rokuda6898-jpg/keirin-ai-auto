"""Freeze market-based first-position hypotheses; advisory only."""
import json
import math
from datetime import datetime
from itertools import permutations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common import OUTPUT_DIR

VERSION = "market_first_shadow_v1"
LEDGER = "market_first_shadow_ledger.jsonl"


def first_from_quotes(race, odds, now, close):
    """Return a market favorite only for complete, recent trifecta quotes."""
    needed = {"race_id", "bet_type", "buy", "odds_used", "odds_captured_at_jst"}
    if odds is None or odds.empty or not needed.issubset(odds.columns):
        return None
    cars = sorted(int(c) for c in race["car_no"])
    if len(cars) < 3 or len(cars) != len(set(cars)):
        return None
    market = odds.loc[
        odds["race_id"].astype(str).eq(str(race.iloc[0]["race_id"]))
        & odds["bet_type"].astype(str).eq("trifecta")
    ]
    expected = {"-".join(map(str, x)) for x in permutations(cars, 3)}
    if len(market) != len(expected) or set(market["buy"].astype(str)) != expected:
        return None
    if market["buy"].astype(str).duplicated().any():
        return None
    prices = pd.to_numeric(market["odds_used"], errors="coerce")
    if prices.isna().any() or not prices.between(1, 9999.9, inclusive="left").all():
        return None
    if ("odds_verification_status" in market
            and market["odds_verification_status"].astype(str).str.startswith("excluded").any()):
        return None
    stamps = pd.to_datetime(market["odds_captured_at_jst"], utc=True, errors="coerce")
    if stamps.isna().any():
        return None
    seconds = stamps.map(lambda t: t.timestamp())
    if ((seconds > now.timestamp()) | (seconds >= close)
            | ((now.timestamp() - seconds) > 300)).any():
        return None
    weights = {car: 0.0 for car in cars}
    for buy, odd in zip(market["buy"], prices):
        weights[int(str(buy).split("-")[0])] += 1 / float(odd)
    return min(cars, key=lambda car: (-weights[car], car))


def freeze_market_axis(pred, odds, now, reference, output_dir=OUTPUT_DIR):
    """Persist one pre-close market axis per race, independent from model scores."""
    folder = Path(output_dir) / "company"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LEDGER
    saved = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                saved.add(str(json.loads(line)["race_id"]))
            except (KeyError, ValueError, TypeError):
                continue
    additions = []
    for race_id, race in pred.groupby("race_id", sort=False):
        rid = str(race_id)
        if rid in saved or rid not in reference:
            continue
        close = float(race.iloc[0]["close_at"])
        if not math.isfinite(close) or close - now.timestamp() <= 300:
            continue
        axis = first_from_quotes(race, odds, now, close)
        if axis is None:
            continue
        base = reference[rid]
        additions.append({
            "version": VERSION, "race_id": rid,
            "venue": str(race.iloc[0].get("venue", "")),
            "snapshot_at": now.isoformat(timespec="seconds"),
            "close_at": close, "market_first": int(axis),
            "current_hole": int(base["variants"]["current_hole"]),
            "consensus": int(base["variants"]["six_department_consensus"]),
            "basis": "full_recent_trifecta_market_marginal_not_win_probability",
            "purchase_authorized": False,
        })
    if additions:
        with path.open("a", encoding="utf-8") as f:
            for row in additions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return additions
