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


def build_market_report(output_dir=OUTPUT_DIR):
    """Compare preserved pre-close market axis with official winner only."""
    folder = Path(output_dir) / "company"
    folder.mkdir(parents=True, exist_ok=True)
    ledger = folder / LEDGER
    frozen = {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                stamp = datetime.fromisoformat(row["snapshot_at"])
                if (row["version"] != VERSION or stamp.tzinfo is None
                        or float(row["close_at"]) - stamp.timestamp() <= 300):
                    continue
                rid = str(row["race_id"])
                if rid not in frozen:
                    frozen[rid] = row
            except (KeyError, TypeError, ValueError):
                continue
    settled_path = folder / "market_first_shadow_settled.json"
    try:
        prior = json.loads(settled_path.read_text(encoding="utf-8")) if settled_path.exists() else []
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("Prior market comparison unreadable; refusing overwrite") from exc
    if not isinstance(prior, list):
        raise ValueError("Prior market comparison invalid; refusing overwrite")
    settled = {str(r["race_id"]): r for r in prior
               if isinstance(r, dict) and r.get("version") == VERSION and "race_id" in r}
    try:
        results = json.loads((Path(output_dir) / "latest_results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        results = []
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, dict):
            continue
        rid = str(result.get("race_id", ""))
        if (rid in settled or rid not in frozen
                or str(result.get("official_result_available", "")).lower() not in {"true", "1"}):
            continue
        actual = str(result.get("actual_trifecta", "")).split("-")
        if (len(actual) != 3 or len(set(actual)) != 3
                or not all(x.isdigit() and 1 <= int(x) <= 9 for x in actual)):
            continue
        row = frozen[rid]
        winner = int(actual[0])
        settled[rid] = {
            **row, "official_winner": winner,
            "hits": {name: winner == int(row[name]) for name in
                     ("market_first", "current_hole", "consensus")},
        }
    rows = sorted(settled.values(), key=lambda r: r["race_id"])
    settled_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    variants = {}
    for name in ("market_first", "current_hole", "consensus"):
        wins = sum(int(r["hits"][name]) for r in rows)
        variants[name] = {"paired_races": len(rows), "winner_hits": wins,
                          "hit_rate": wins / len(rows) if rows else None}
    report = {
        "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "version": VERSION, "frozen_races": len(frozen), "settled_races": len(rows),
        "variants": variants,
        "market_only": sum(r["hits"]["market_first"] and not r["hits"]["current_hole"] for r in rows),
        "hole_only": sum(r["hits"]["current_hole"] and not r["hits"]["market_first"] for r in rows),
        "different_axes": sum(r["market_first"] != r["current_hole"] for r in rows),
        "review_ready": len(rows) >= 300,
        "automatic_promotion": False, "purchase_authorized": False,
        "limitation": "Trifecta market-derived first-place support, not actual win-pool odds, verified 100x tickets or ROI.",
    }
    (folder / "market_first_shadow_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    table = "".join(
        "<tr><td>" + label + "</td><td>" + str(variants[name]["winner_hits"])
        + "/" + str(len(rows)) + "</td></tr>"
        for name, label in (
            ("market_first", "市場の1着支持"),
            ("current_hole", "高配当部の1着仮説"),
            ("consensus", "他6部署の多数支持"),
        )
    )
    (folder / "market_first_shadow_report.html").write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>市場支持の1着軸検証</title><main>'
        '<p><a href="high_payout_axis_shadow_report.html">既存4方式の検証へ戻る</a></p>'
        '<h1>市場の1着候補と高配当部の比較</h1>'
        f'<p>締切前記録 {len(frozen)}レース、公式結果照合 {len(rows)}レース</p>'
        '<table><tr><th>仮説</th><th>1着正解</th></tr>' + table + '</table>'
        '<p>全3連単組み合わせの300秒以内の取得時刻付き実オッズが揃うレースのみ。'
        '単勝の実オッズではなく、3連単の逆数を1着ごとに集計した市場支持指標。'
        '100倍以上の穴車券・的中率・回収率とは異なる。'
        '300レース以上の事前検証までロジックと買い目は変更しない。</p>'
        '</main></html>', encoding="utf-8")
    return report
