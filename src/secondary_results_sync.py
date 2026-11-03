"""Settle frozen high-ROI, trio and exacta forecasts from captured official payouts.

No predictions, odds, model parameters, release dates, or pre-close records are changed.
A published winner without a verified official payout stays pending.
"""
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from official_outcomes import normalize_outcome

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
JST = ZoneInfo("Asia/Tokyo")
FEEDS = {
    "high_roi": ("trifecta", "ROI9891_fixed_EX_20261102"),
    "trio": ("trio", "trio_selected9891_454_20261103"),
    "exacta": ("exacta", "exacta_native_highroi_345_20261103"),
}


def _timestamp(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp if stamp.tzinfo else None
    except (ValueError, TypeError):
        return None


def _payoffs(value):
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    result = {}
    for ticket, amount in payload.items():
        try:
            yen = float(amount)
        except (ValueError, TypeError):
            continue
        if math.isfinite(yen) and yen > 0 and yen.is_integer():
            result[str(ticket)] = int(yen)
    return result


def _winners(result, pool):
    """Expand official tied 3-car tickets, rather than taking only one order."""
    normalized = normalize_outcome(result)
    if normalized is None:
        return []
    triples = normalized["winning_buys"]
    if pool == "trifecta":
        return triples
    if pool == "trio":
        return sorted({"-".join(sorted(buy.split("-"), key=int)) for buy in triples})
    return sorted({"-".join(buy.split("-")[:2]) for buy in triples})


def official_outcome(result, pool):
    if result.get("refund_all") is True or result.get("cancelled") is True:
        return {"void": True, "winners": [], "payoffs": {},
                "observed_at": result.get("result_observed_at_jst")}
    winners = _winners(result, pool)
    if not winners:
        return None
    payouts = _payoffs(result.get(f"payouts_{pool}_json"))
    # Do not treat a missing winning payoff as a known loss or as 0 yen.
    if any(ticket not in payouts for ticket in winners):
        return None
    return {"void": False, "winners": winners,
            "payoffs": {ticket: payouts[ticket] for ticket in winners},
            "observed_at": result.get("result_observed_at_jst")}


def settlement(row, outcome):
    prediction = row.get("forecast")
    if not isinstance(prediction, dict) or not isinstance(prediction.get("tickets"), list):
        return None
    stake = int(prediction["count"]) * 100
    if (prediction.get("unit_yen") != 100 or
            stake != prediction.get("total_yen") or
            len(set(prediction["tickets"])) != prediction["count"]):
        raise ValueError(f"invalid frozen forecast amount: {row.get('race_id')}")
    if outcome["void"]:
        return dict(status="void", winning_tickets=[], payoffs={}, matched=[],
                    HIGH_hit=False, stake=stake, returned=stake, refund_yen=stake)
    matched = sorted(set(prediction["tickets"]) & set(outcome["winners"]))
    returned = sum(outcome["payoffs"][ticket] for ticket in matched)
    return dict(status="hit" if matched else "miss", winning_tickets=outcome["winners"],
                payoffs=outcome["payoffs"], matched=matched,
                HIGH_hit=any(outcome["payoffs"][ticket] >= 10000 for ticket in matched),
                stake=stake, returned=returned, refund_yen=0)


def summary(feed):
    model = feed["model_id"]
    s = dict(races=0, hits=0, HIGH_hits=0, stake=0, returned=0,
             hit_pct=None, roi_pct=None, void=0, pending=0, model_id=model)
    for row in feed["races"]:
        pred = row.get("forecast")
        if not isinstance(pred, dict) or pred.get("model_id") != model or row.get("performance_eligible") is not True:
            continue
        settled = row.get("settlement")
        if not isinstance(settled, dict) or settled.get("status") not in ("hit", "miss", "void"):
            s["pending"] += 1
            continue
        if settled["status"] == "void":
            s["void"] += 1
            continue
        s["races"] += 1
        s["hits"] += int(settled["status"] == "hit")
        s["HIGH_hits"] += int(bool(settled.get("HIGH_hit")))
        s["stake"] += int(settled["stake"])
        s["returned"] += int(settled["returned"])
    if s["races"]:
        s["hit_pct"] = 100 * s["hits"] / s["races"]
    if s["stake"]:
        s["roi_pct"] = 100 * s["returned"] / s["stake"]
    return s


def sync(outputs=OUTPUTS):
    outputs = Path(outputs)
    source = outputs / "latest_results.json"
    if not source.is_file():
        raise FileNotFoundError(f"Official results feed is missing: {source}")
    fetched = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(fetched, list):
        raise ValueError("Official results feed must be a race list")
    official = {str(x["race_id"]): x for x in fetched
                if isinstance(x, dict) and "race_id" in x}
    report = {}
    for name, (pool, model) in FEEDS.items():
        path = outputs / f"{name}_live.json"
        if not path.exists():
            report[name] = {"error": "missing_feed"}
            continue
        feed = json.loads(path.read_text(encoding="utf-8"))
        if (feed.get("schema") != 1 or feed.get("model_id") != model or
                not isinstance(feed.get("races"), list)):
            raise ValueError(f"invalid or unexpected model feed: {name}")
        if name != "high_roi" and feed.get("pool") != pool:
            raise ValueError(f"unexpected pool in {name}")
        updated = 0
        last_stamp = _timestamp(feed.get("results_updated_at"))
        for row in feed["races"]:
            if not isinstance(row, dict) or not isinstance(row.get("forecast"), dict):
                continue
            record = official.get(str(row.get("race_id")))
            if not record:
                continue
            outcome = official_outcome(record, pool)
            if outcome is None:
                continue
            decided = settlement(row, outcome)
            if decided is None:
                continue
            previous = row.get("settlement")
            if previous and previous.get("status") == "void" and not outcome["void"]:
                continue
            observed = _timestamp(outcome["observed_at"])
            prior_stamp = _timestamp(previous.get("settled_at")) if isinstance(previous, dict) else None
            if prior_stamp and observed and observed < prior_stamp:
                continue
            if isinstance(previous, dict) and all(previous.get(k) == v for k, v in decided.items()):
                continue
            decided["settled_at"] = outcome["observed_at"] or (previous or {}).get("settled_at")
            row["settlement"] = decided
            updated += 1
            if observed and (last_stamp is None or observed > last_stamp):
                last_stamp = observed
        expected = summary(feed)
        changed_summary = feed.get("observed") != expected
        if changed_summary:
            feed["observed"] = expected
        if updated or changed_summary:
            if last_stamp:
                feed["results_updated_at"] = last_stamp.isoformat()
            # generated_at is deliberately NOT touched: the prediction feed
            # must never masquerade as freshly generated from a settlement.
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(feed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp.replace(path)
        report[name] = {"official_records": len(official), "newly_settled": updated,
                        "performance": expected, "changed": bool(updated or changed_summary)}
    print("SECONDARY_RESULT_SYNC " + json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    sync()
