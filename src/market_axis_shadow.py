"""Freeze market-based first-position hypotheses; advisory only."""
import argparse
import html
import json
import math
from collections import Counter
from datetime import datetime
from itertools import permutations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common import OUTPUT_DIR

VERSION = "market_first_shadow_v1"
LEDGER = "market_first_shadow_ledger.jsonl"
CAPTURE_STATUS = "market_first_capture_status.json"


def inspect_quotes(race, odds, now, close):
    """Explain rejection without weakening completeness or the 300-second limit."""
    details = {"expected_quotes": 0, "observed_quotes": 0}
    if race is None or race.empty or not {"race_id", "car_no"}.issubset(race.columns):
        return None, "invalid_entries", details
    numbers = pd.to_numeric(race["car_no"], errors="coerce")
    if (numbers.isna().any() or not numbers.between(1, 9).all()
            or not numbers.eq(numbers.round()).all() or numbers.duplicated().any()
            or len(numbers) < 3 or race["race_id"].astype(str).nunique() != 1):
        return None, "invalid_entries", details
    cars = sorted(int(c) for c in numbers)
    expected = {"-".join(map(str, x)) for x in permutations(cars, 3)}
    details["expected_quotes"] = len(expected)
    needed = {"race_id", "bet_type", "buy", "odds_used", "odds_captured_at_jst"}
    if odds is None or odds.empty:
        return None, "odds_empty", details
    if not needed.issubset(odds.columns):
        details["missing_columns"] = sorted(needed - set(odds.columns))
        return None, "odds_schema_missing", details
    market = odds.loc[
        odds["race_id"].astype(str).eq(str(race.iloc[0]["race_id"]))
        & odds["bet_type"].astype(str).eq("trifecta")
    ]
    buys = market["buy"].astype(str)
    details["observed_quotes"] = len(market)
    if buys.duplicated().any():
        return None, "duplicate_quotes", details
    if len(market) != len(expected) or set(buys) != expected:
        details["missing_combinations"] = len(expected - set(buys))
        details["unexpected_combinations"] = len(set(buys) - expected)
        return None, "incomplete_quotes", details
    prices = pd.to_numeric(market["odds_used"], errors="coerce")
    if prices.isna().any() or not prices.between(1, 9999.9, inclusive="left").all():
        return None, "invalid_prices", details
    if ("odds_verification_status" in market
            and market["odds_verification_status"].astype(str).str.startswith("excluded").any()):
        return None, "excluded_quotes", details
    stamps = pd.to_datetime(market["odds_captured_at_jst"], utc=True, errors="coerce")
    if stamps.isna().any():
        return None, "invalid_capture_time", details
    seconds = stamps.map(lambda t: t.timestamp())
    details["oldest_quote_age_seconds"] = float(now.timestamp() - seconds.min())
    if (seconds > now.timestamp()).any():
        return None, "future_capture_time", details
    if (seconds >= close).any():
        return None, "quote_at_or_after_close", details
    if ((now.timestamp() - seconds) > 300).any():
        return None, "stale_quotes", details
    weights = {car: 0.0 for car in cars}
    for buy, odd in zip(buys, prices):
        weights[int(buy.split("-")[0])] += 1 / float(odd)
    return min(cars, key=lambda car: (-weights[car], car)), "eligible", details


def first_from_quotes(race, odds, now, close):
    """Backward-compatible selector; diagnostic details are available separately."""
    return inspect_quotes(race, odds, now, close)[0]


def freeze_market_axis(pred, odds, now, reference, output_dir=OUTPUT_DIR, *, phase="prediction", dry_run=False):
    """Persist pre-close axes and account for every saved or rejected race."""
    if phase not in {"prediction", "ingestion", "diagnostic"}:
        raise ValueError("Unknown market capture phase")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("market capture clock must include a timezone")
    folder = Path(output_dir) / "company"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LEDGER
    saved = set()
    malformed_ledger_lines = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                saved.add(str(json.loads(line)["race_id"]))
            except (KeyError, ValueError, TypeError):
                malformed_ledger_lines += 1
    additions, decisions = [], []
    input_valid = pred is not None and "race_id" in pred.columns
    groups = pred.groupby("race_id", sort=False) if input_valid else []
    for race_id, race in groups:
        rid = str(race_id)
        decision = {"race_id": rid, "reason": None}
        decisions.append(decision)
        if malformed_ledger_lines:
            decision["reason"] = "invalid_existing_ledger"
            continue
        if rid in saved:
            decision["reason"] = "already_saved"
            continue
        try:
            close = float(race.iloc[0]["close_at"])
        except (KeyError, TypeError, ValueError, OverflowError):
            close = float("nan")
        if not math.isfinite(close):
            decision["reason"] = "invalid_close_time"
            continue
        if close - now.timestamp() <= 300:
            decision["reason"] = "near_or_after_close"
            continue
        if rid not in reference:
            decision["reason"] = "reference_missing"
            continue
        close_values = pd.to_numeric(race.get("close_at"), errors="coerce")
        if close_values.isna().any() or close_values.nunique() != 1:
            decision["reason"] = "inconsistent_close_time"
            continue
        axis, reason, details = inspect_quotes(race, odds, now, close)
        decision.update(details)
        decision["reason"] = reason
        if axis is None:
            continue
        try:
            base = reference[rid]
            base_stamp = datetime.fromisoformat(base["snapshot_at"])
            hole = int(base["variants"]["current_hole"])
            consensus = int(base["variants"]["six_department_consensus"])
            cars = {int(c) for c in race["car_no"]}
            if (base_stamp.tzinfo is None or base_stamp.timestamp() > now.timestamp()
                    or base_stamp.timestamp() >= close or hole not in cars or consensus not in cars):
                raise ValueError("reference not verified before capture")
        except (KeyError, TypeError, ValueError, OverflowError):
            decision["reason"] = "invalid_reference"
            continue
        additions.append({
            "version": VERSION, "race_id": rid,
            "venue": str(race.iloc[0].get("venue", "")),
            "snapshot_at": now.isoformat(timespec="seconds"),
            "close_at": close, "market_first": int(axis),
            "current_hole": hole, "consensus": consensus,
            "reference_snapshot_at": base_stamp.isoformat(timespec="seconds"),
            "source_car_numbers": sorted(cars), "capture_phase": phase,
            "quote_count": details["observed_quotes"],
            "oldest_quote_age_seconds": details["oldest_quote_age_seconds"],
            "basis": "full_recent_trifecta_market_marginal_not_win_probability",
            "purchase_authorized": False,
        })
        decision["reason"] = "eligible_not_saved" if dry_run else "saved"
    if additions and not dry_run:
        previous_text = path.read_text(encoding="utf-8") if path.exists() else ""
        if previous_text and not previous_text.endswith("\n"):
            previous_text += "\n"
        pending = path.with_suffix(".jsonl.tmp")
        pending.write_text(previous_text + "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in additions
        ), encoding="utf-8")
        pending.replace(path)
    counts = dict(sorted(Counter(d["reason"] for d in decisions).items()))
    audit = {
        "checked_at_jst": now.isoformat(timespec="seconds"),
        "version": VERSION, "input_valid": input_valid, "phase": phase, "dry_run": dry_run,
        "considered_races": len(decisions), "saved_this_run": 0 if dry_run else len(additions),
        "eligible_this_run": len(additions),
        "already_saved_races": counts.get("already_saved", 0),
        "reason_counts": counts, "decisions": decisions,
        "malformed_existing_ledger_lines": malformed_ledger_lines,
        "max_quote_age_seconds": 300, "minimum_seconds_before_close": 300,
        "purchase_authorized": False,
    }
    temporary = folder / (CAPTURE_STATUS + ".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(folder / CAPTURE_STATUS)
    (folder / f"market_first_capture_{phase}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("market_axis_capture " + json.dumps({"phase": phase, "races": len(decisions), "saved": audit["saved_this_run"],
                                              "reasons": counts}, ensure_ascii=False), flush=True)
    return [] if dry_run else additions


def capture_after_fetch(entries, odds, output_dir=OUTPUT_DIR, now=None):
    """Freeze the observed market immediately, before slow feature/model work.

    Failure of this optional experiment is logged, never converted into a
    purchase decision and never allowed to discard a valid racecard refresh.
    """
    now = now or datetime.now(ZoneInfo("Asia/Tokyo"))
    folder = Path(output_dir) / "company"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        from department_coverage import _frozen_axis_experiments
        references = _frozen_axis_experiments(folder)
        return freeze_market_axis(pd.DataFrame(entries), pd.DataFrame(odds), now,
                                  references, output_dir, phase="ingestion")
    except Exception as exc:
        issue = {"checked_at_jst": now.isoformat(timespec="seconds"),
                 "phase": "ingestion", "status": "error",
                 "error": f"{type(exc).__name__}: {exc}", "purchase_authorized": False}
        (folder / "market_first_capture_error.json").write_text(
            json.dumps(issue, ensure_ascii=False, indent=2), encoding="utf-8")
        print("market_axis_capture_error " + json.dumps(issue, ensure_ascii=False), flush=True)
        return []


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
                        or not math.isfinite(float(row["close_at"]))
                        or float(row["close_at"]) - stamp.timestamp() <= 300
                        or not all(str(row.get(k, "")).isdigit() and 1 <= int(row[k]) <= 9
                                   for k in ("market_first", "current_hole", "consensus"))):
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
    audits = {}
    for phase in ("ingestion", "prediction", "diagnostic"):
        audit_path = folder / f"market_first_capture_{phase}.json"
        try:
            if audit_path.exists():
                value = json.loads(audit_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    audits[phase] = value
        except (OSError, ValueError, TypeError):
            audits[phase] = {"status": "unreadable"}
    report = {
        "capture_audits": audits,
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
    capture_html = "".join(
        "<p>" + html.escape(phase) + "："
        + html.escape(json.dumps({k: v for k, v in audit.items() if k != "decisions"}, ensure_ascii=False))
        + "</p>" for phase, audit in audits.items()
    )
    (folder / "market_first_shadow_report.html").write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>市場支持の1着軸検証</title><main>'
        '<p><a href="high_payout_axis_shadow_report.html">既存4方式の検証へ戻る</a></p>'
        '<h1>市場の1着候補と高配当部の比較</h1>'
        f'<p>締切前記録 {len(frozen)}レース、公式結果照合 {len(rows)}レース</p>'
        '<table><tr><th>仮説</th><th>1着正解</th></tr>' + table + '</table>'
        + '<h2>取得・見送りの診断</h2>' + (capture_html or '<p>診断記録はまだありません。</p>') +
        '<p>全3連単組み合わせの300秒以内の取得時刻付き実オッズが揃うレースのみ。'
        '単勝の実オッズではなく、3連単の逆数を1着ごとに集計した市場支持指標。'
        '100倍以上の穴車券・的中率・回収率とは異なる。'
        '300レース以上の事前検証までロジックと買い目は変更しない。</p>'
        '</main></html>', encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Diagnose the actual checked-in market snapshot without inventing forecasts.")
    parser.add_argument("--diagnose", action="store_true", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from common import TODAY_CSV, TODAY_ODDS_CSV
    from department_coverage import _frozen_axis_experiments
    entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str})
    quotes = pd.read_csv(TODAY_ODDS_CSV, dtype={"race_id": str, "buy": str})
    refs = _frozen_axis_experiments(OUTPUT_DIR / "company")
    freeze_market_axis(entries, quotes, datetime.now(ZoneInfo("Asia/Tokyo")), refs,
                       args.output_dir, phase="diagnostic", dry_run=True)
    build_market_report(args.output_dir)


if __name__ == "__main__":
    main()
