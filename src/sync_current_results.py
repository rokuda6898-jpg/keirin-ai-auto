"""Refresh the home page's settled bets from already-captured official results.

Never rewrites forecasts, pre-close snapshots, or publication eligibility.
A completed official result is necessary before a bet can be settled.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from official_outcomes import normalize_outcome, ticket
from nexus_model.budget import settle as settle_stakes

ROOT = Path(__file__).resolve().parents[1]


def _stamp(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else None
    except (TypeError, ValueError):
        return None


def _yen(value):
    try:
        f = Fraction(str(value))
        return int(f) if f.denominator == 1 and f >= 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _nexus_outcome(row):
    if not isinstance(row, dict) or not (row.get("result_complete") or row.get("refund_all") or row.get("cancelled")):
        return None
    refund_all = bool(row.get("refund_all") or row.get("cancelled"))
    winners = row.get("winners") or []
    payouts = row.get("payouts") or {}
    if not isinstance(winners, list) or not isinstance(payouts, dict):
        return None
    winners = list(dict.fromkeys(winners))
    if not refund_all and (not winners or any(not ticket(t) for t in winners)):
        return None
    values = {t: _yen(payouts.get(t)) for t in winners}
    if not refund_all and any(v is None or v == 0 for v in values.values()):
        return None
    refund = row.get("refunded") or []
    if not isinstance(refund, list) or any(not ticket(t) for t in refund):
        return None
    return {
        "winners": winners,
        "payouts": {} if refund_all else values,
        "refunded": refund,
        "refund_all": refund_all,
        "observed_at": row.get("results_updated_at") or row.get("result_sync_captured_at") or row.get("result_captured_at"),
    }


def _compact_outcome(row):
    if not isinstance(row, dict):
        return None
    outcome = normalize_outcome(row)
    if not outcome or not outcome["payout_complete"]:
        return None
    payouts = {t: _yen(v) for t, v in outcome["payouts"].items()}
    if any(v is None or v == 0 for v in payouts.values()):
        return None
    return {
        "winners": outcome["winning_buys"], "payouts": payouts,
        "refunded": [], "refund_all": False,
        "observed_at": outcome["source_observed_at"],
    }


def _settlement(forecast, outcome):
    tickets = forecast.get("tickets")
    if not isinstance(tickets, list) or not tickets or len(set(tickets)) != len(tickets) or any(not ticket(t) for t in tickets):
        return None
    stakes = {t: 100 for t in tickets}
    payout = settle_stakes(stakes, outcome["payouts"], outcome["refunded"], outcome["refund_all"])
    winners = outcome["winners"]
    matched = [t for t in tickets if t in winners and t not in outcome["refunded"]]
    status = "void" if payout["void"] else "hit" if payout["hit"] else "miss"
    return {
        "status": status,
        "winning_tickets": winners,
        "payoffs": outcome["payouts"],
        "matched": matched,
        "HIGH_hit": any(outcome["payouts"].get(t, 0) >= 10000 for t in matched),
        "stake": payout["stake"],
        "returned": payout["returned"],
        "refund_yen": payout["refund"],
        "settled_at": outcome["observed_at"] or datetime.now(timezone.utc).isoformat(),
    }


def _performance(feed):
    model_id = feed["model_id"]
    result = dict(races=0, hits=0, HIGH_hits=0, stake=0, returned=0, void=0,
                  pending=0, unverified_publication=0, model_id=model_id)
    for race in feed["races"]:
        if (race.get("forecast") or {}).get("model_id") != model_id:
            continue
        if not race.get("performance_eligible"):
            result["unverified_publication"] += 1
            continue
        settled = race.get("settlement")
        if not isinstance(settled, dict) or settled.get("status") not in ("hit", "miss", "void"):
            result["pending"] += 1
            continue
        if settled["status"] == "void":
            result["void"] += 1
        else:
            result["races"] += 1
            result["hits"] += int(settled["status"] == "hit")
            result["HIGH_hits"] += int(bool(settled.get("HIGH_hit")))
        result["stake"] += int(settled["stake"])
        result["returned"] += int(settled["returned"])
    result["hit_pct"] = result["hits"] / result["races"] * 100 if result["races"] else None
    result["roi_pct"] = result["returned"] / result["stake"] * 100 if result["stake"] else None
    return result


def sync(outputs=ROOT / "outputs"):
    outputs = Path(outputs)
    path = outputs / "current_live.json"
    if not path.exists():
        return {"changed": False, "reason": "no_current_feed"}
    feed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(feed, dict) or not isinstance(feed.get("races"), list) or not feed.get("model_id"):
        raise ValueError("current feed has an invalid schema")
    compact_path = outputs / "latest_results.json"
    nexus_path = outputs / "nexus_live.json"
    compact = json.loads(compact_path.read_text(encoding="utf-8")) if compact_path.exists() else []
    nexus = json.loads(nexus_path.read_text(encoding="utf-8")) if nexus_path.exists() else {"races": []}
    if not isinstance(compact, list) or not isinstance(nexus, dict) or not isinstance(nexus.get("races"), list):
        raise ValueError("result feed has an invalid schema")
    outcomes = {}
    for row in compact:
        if not isinstance(row, dict):
            continue
        result = _compact_outcome(row)
        if result:
            outcomes[str(row.get("race_id"))] = result
    for row in nexus["races"]:
        result = _nexus_outcome(row)
        if result:
            # Certified NEXUS snapshots retain refunds and tied payouts.
            outcomes[str(row.get("race_id"))] = result
    changes = 0
    new_today = 0
    updated_stamps = []
    for race in feed["races"]:
        result = outcomes.get(str(race.get("race_id")))
        if not result:
            continue
        forecast = race.get("forecast")
        if not isinstance(forecast, dict):
            continue
        if not result["refund_all"] and race.get("status") == "cancelled":
            continue
        old = race.get("settlement")
        incoming = _stamp(result["observed_at"])
        previous = _stamp((old or {}).get("settled_at"))
        if old and previous and (incoming is None or incoming < previous):
            continue
        settled = _settlement(forecast, result)
        if not settled:
            continue
        if old and all(old.get(k) == v for k, v in settled.items() if k != "settled_at"):
            continue
        if not old and race.get("date") == feed.get("today"):
            new_today += 1
        race["settlement"] = settled
        changes += 1
        if incoming:
            updated_stamps.append(incoming)
    if not changes:
        return {"changed": False, "newly_settled_today": 0, "outcomes": len(outcomes)}
    feed["observed"] = _performance(feed)
    existing = _stamp(feed.get("results_updated_at"))
    all_stamps = updated_stamps + ([existing] if existing else [])
    if all_stamps:
        feed["results_updated_at"] = max(all_stamps).isoformat()
    new = json.dumps(feed, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(new, encoding="utf-8")
    temporary.replace(path)
    return {"changed": True, "updated_races": changes, "newly_settled_today": new_today,
            "observed_races": feed["observed"]["races"], "outcomes": len(outcomes),
            "results_updated_at": feed.get("results_updated_at")}


if __name__ == "__main__":
    print("CURRENT_RESULT_SYNC " + json.dumps(sync(), ensure_ascii=False), flush=True)
