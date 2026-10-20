"""Reproduce short, explicitly limited NEXUS ROI comparisons from saved evidence.

This is an exploratory diagnostic, not the original 27,351-race annual test.
It never edits forecasts, trains a model, fetches odds, or places bets.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path

from nexus_model.budget import settle
from nexus_roi_research import select_tickets

POINTS = (6, 8, 10, 12)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("a timezone-aware evidence timestamp is required")
    return result


def valid_result(row):
    p = row.get("prediction") or {}
    if not p:
        return "missing_preclose_prediction"
    if not row.get("result_complete") or not row.get("payouts"):
        return "official_result_pending"
    if row.get("refund_all") or row.get("refunded") or row.get("cancelled"):
        return "refund_or_cancellation"
    try:
        if not stamp(p["input_captured_at"]) <= stamp(p["captured_at"]) < stamp(row["close_at"]):
            return "forecast_timestamp_invalid"
        if p["history_through"] >= row["date"]:
            return "history_after_cutoff"
        ts = p["tickets"]
        if not ts or len(ts) != len(set(ts)) or p["count"] != len(ts):
            return "ticket_identity_invalid"
        if set(row["winners"]) != set(row["payouts"]):
            return "official_winner_payout_mismatch"
        for payout in row["payouts"].values():
            if isinstance(payout, bool) or Fraction(str(payout)).denominator != 1 or payout <= 0:
                return "official_payout_invalid"
    except (ValueError, KeyError, TypeError, AttributeError):
        return "evidence_invalid"
    return None


def outcome(row, tickets):
    value = settle({t: 100 for t in tickets}, row["payouts"])
    return dict(race_id=row["race_id"], date=row["date"], tickets=len(tickets),
                stake_yen=value["stake"], payout_yen=value["returned"],
                hit=value["hit"], torigami=bool(value["hit"] and value["returned"] < value["stake"]),
                high_payout_hit=any(t in tickets and amount >= 10000 for t, amount in row["payouts"].items()))


def totals(rows):
    stake = sum(r["stake_yen"] for r in rows)
    payout = sum(r["payout_yen"] for r in rows)
    return dict(races=len(rows), hits=sum(r["hit"] for r in rows),
                hit_rate=sum(r["hit"] for r in rows) / len(rows) if rows else None,
                stake_yen=stake, payout_yen=payout, profit_yen=payout - stake,
                roi=payout / stake if stake else None,
                torigami=sum(r["torigami"] for r in rows),
                high_payout_hits=sum(r["high_payout_hit"] for r in rows),
                mean_tickets=sum(r["tickets"] for r in rows) / len(rows) if rows else None)


def validate_replay(record, row):
    if record.get("exact_replay_match") is not True or not record.get("match_flags") or not all(record["match_flags"].values()):
        raise ValueError("replay must pass the original forecast equality audit")
    cars, q = record["cars"], record["q"]
    if (not 3 <= len(cars) <= 9 or len(cars) != len(set(cars))
            or any(type(c) is not int or not 1 <= c <= 9 for c in cars)):
        raise ValueError("invalid replay car identity")
    if (len(q) != math.perm(len(cars), 3)
            or any(isinstance(p, bool) or not isinstance(p, (int, float))
                   or not math.isfinite(p) or not 0 <= p <= 1 for p in q)
            or abs(math.fsum(q) - 1) > 1e-8):
        raise ValueError("full normalized joint distribution required")
    if record["date"] != row["date"]:
        raise ValueError("replay date mismatch")
    original = row["prediction"]
    if sorted(record["live_tickets"]) != sorted(original["tickets"]):
        raise ValueError("reproduced live tickets differ from original publication")
    if record["input_sha256"] != original.get("input_sha256"):
        raise ValueError("replay input is not the original published input")
    if record["history_through"] != original["history_through"]:
        raise ValueError("replay history differs from the original history")
    if set(cars) != {r["car"] for r in row["riders"]}:
        raise ValueError("active rider set differs")
    if record.get("q_order", "itertools.permutations(cars,3)") != "itertools.permutations(cars,3)":
        raise ValueError("replay q ordering differs")
    # Marginals independently protect the cars/q ordering used in selection.
    marginals = dict.fromkeys(cars, 0.0)
    for t, p in zip(itertools.permutations(cars, 3), q):
        marginals[t[0]] += p
    if any(abs(marginals[c] - original["first_probabilities"][str(c)]) > 1e-10 for c in cars):
        raise ValueError("replayed first-place probabilities differ from publication")


def quote_diagnostic(rows):
    records, excluded = [], Counter()
    for row in rows:
        p = row["prediction"]
        try:
            if not stamp(p["odds_at"]) < stamp(row["close_at"]):
                raise ValueError("late quote")
            odds = {t: Fraction(str(p["odds"][t])) for t in p["tickets"]}
            if any(price <= 0 for price in odds.values()):
                raise ValueError("invalid quote")
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            excluded["missing_complete_preclose_quotes"] += 1
            continue
        decisions = {"published": list(p["tickets"])}
        for name, buffer in (("remove_flat_torigami", Fraction(1)),
                             ("remove_flat_torigami_buffer10", Fraction(11, 10))):
            selected = sorted(p["tickets"], key=lambda t: (odds[t], t))
            while len(selected) > 1 and odds[selected[0]] < len(selected) * buffer:
                selected.pop(0)
            decisions[name] = selected
        # All decisions are fixed before any official payoff is read.
        records.append({name: outcome(row, tickets) for name, tickets in decisions.items()})
    return dict(scope="All settled saved forecasts with complete preclose quotes; research only",
                exclusions=dict(excluded),
                summaries={name: totals([r[name] for r in records]) for name in (
                    "published", "remove_flat_torigami", "remove_flat_torigami_buffer10")})


def run(feed_path, replay_path, audit_path=None):
    feed = json.loads(Path(feed_path).read_text())
    replays = [json.loads(line) for line in Path(replay_path).read_text().splitlines() if line.strip()]
    if not replays:
        raise ValueError("a nonempty verified replay cohort is required")
    manifest_path = Path(__file__).resolve().parents[1] / "models/nexus53/manifest.json"
    manifest_sha256 = digest(manifest_path)
    if any(r.get("model_manifest_sha256") != manifest_sha256 for r in replays):
        raise ValueError("replayed model differs from the frozen NEXUS manifest")
    audit = json.loads(Path(audit_path).read_text()) if audit_path else None
    if audit and (audit["eligible_sha256"] != digest(replay_path)
                  or set(audit["eligible_race_ids"]) != {r["race_id"] for r in replays}
                  or audit.get("selection_used_outcomes") is not False):
        raise ValueError("reconstruction audit does not match the fixed replay cohort")
    ids = [r["race_id"] for r in feed["races"]]
    replay_ids = [r["race_id"] for r in replays]
    if len(ids) != len(set(ids)) or len(replay_ids) != len(set(replay_ids)):
        raise ValueError("duplicate race identity")
    saved, excluded = {}, Counter()
    for row in feed["races"]:
        reason = valid_result(row)
        if reason:
            excluded[reason] += 1
        else:
            saved[row["race_id"]] = row
            flat = outcome(row, row["prediction"]["tickets"])
            settled = row.get("settlement", {}).get("flat")
            if settled and any(flat[k] != settled[v] for k, v in (
                    ("stake_yen", "stake"), ("payout_yen", "returned"), ("hit", "hit"))):
                raise ValueError("published official flat settlement mismatch")
    candidates = ("published",) + tuple(f"formation_{n}" for n in POINTS)
    records = []
    for replay in sorted(replays, key=lambda r: (r["date"], r["race_id"])):
        row = saved[replay["race_id"]]
        validate_replay(replay, row)
        decisions = {"published": dict(tickets=row["prediction"]["tickets"],
                                       formations=row["prediction"]["blocks"])}
        for n in POINTS:
            tickets, formations = select_tickets(replay["cars"], replay["q"], n)
            decisions[f"formation_{n}"] = dict(tickets=["-".join(map(str, t)) for t in tickets],
                                                formations=formations)
        records.append(dict(race_id=row["race_id"], date=row["date"],
                            comparisons={name: outcome(row, plan["tickets"]) | plan
                                         for name, plan in decisions.items()}))
    # Keep numeric ticket counts separate from the exact, inspectable ticket lists.
    for record in records:
        for value in record["comparisons"].values():
            value["selected_tickets"] = value.pop("tickets")
            value["tickets"] = len(value["selected_tickets"])
    dates = sorted({r["date"] for r in records})
    summary = {name: totals([r["comparisons"][name] for r in records]) for name in candidates}
    sensitivity = {}
    for name in candidates:
        largest = max(records, key=lambda r: r["comparisons"][name]["payout_yen"])
        remaining = [r for r in records if r["race_id"] != largest["race_id"]]
        sensitivity[name] = dict(removed_race_id=largest["race_id"],
            removed_candidate_return_yen=largest["comparisons"][name]["payout_yen"],
            candidate=totals([r["comparisons"][name] for r in remaining]),
            baseline=totals([r["comparisons"]["published"] for r in remaining]))
    return dict(schema=1, protocol_id="nexus-recent-flat100-diagnostic-v1",
                generated_at=datetime.now(timezone.utc).isoformat(), research_only=True,
                annual_benchmark_reproduced=False, annual_target_achieved=False,
                original_annual_races=27351, unit_yen=100,
                source_sha256=dict(feed=digest(feed_path), replay=digest(replay_path),
                                   model_manifest=manifest_sha256,
                                   replay_audit=digest(audit_path) if audit_path else None),
                candidates=list(candidates), all_replayed_races_in_every_candidate=True,
                untouched_holdout=False, production_adopted=False,
                limitation="Small reconstructed recent cohort, selected by evidence availability. Previously inspected outcomes. No claim of annual improvement or future profitability.",
                inference_scope="Same frozen model, same preclose inputs and prior-day history; only complete formation ticket counts change.",
                all_saved_result_exclusions=dict(excluded),
                all_saved_result_baseline=totals([outcome(r, r["prediction"]["tickets"]) for r in saved.values()]),
                reconstruction_audit=(dict(source_rows=audit["source_rows"],
                    eligible_rows=audit["eligible_rows"], exclusions=audit["exclusions"],
                    selection_used_outcomes=False, numeric_tolerance=audit["numeric_tolerance"])
                    if audit else None),
                replay_cohort_dates=dates, summary=summary,
                remove_largest_candidate_return=sensitivity,
                by_date={ds: {name: totals([r["comparisons"][name] for r in records if r["date"] == ds])
                              for name in candidates} for ds in dates},
                quote_diagnostic=quote_diagnostic(list(saved.values())), records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.feed, args.replay, args.replay_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))
