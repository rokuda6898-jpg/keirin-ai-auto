"""Prospective same-ticket, same-budget comparison using existing page captures.

No network access, prediction, ledger edits, or retrospective allocation. A
complete eligible quote is accepted once, including an infeasible allocation.
"""
import argparse
import copy
import json
import math
import re
from datetime import datetime
from pathlib import Path

from conditional_budget import allocate, check_saved_plan, number, settle
from nexus_input_snapshots import ROOT, JST, SNAPSHOTS, now_jst, read_json, save_json, timestamp
from nexus_results_sync import read_saved_page_outcome
from official_outcomes import normalize_outcome

FEED = Path("outputs/conditional_near_close.json")
LEDGER = Path("outputs/company/conditional_live_ledger.jsonl")
POLICY_ID = "conditional-near-close-v1"
MIN_LEAD, MAX_LEAD, MAX_AGE = 300, 2400, 600


def _now(clock):
    value = clock()
    if value.tzinfo is None:
        raise ValueError("timezone-aware clock required")
    return value.astimezone(JST)


def _epoch(value):
    if isinstance(value, bool):
        raise ValueError("invalid closing time")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("invalid closing time")
    return value


def _true(value):
    return value is True or str(value).lower() in ("1", "true")


def _refund(row):
    if not isinstance(row, dict):
        return False
    return (any(_true(row.get(k)) for k in ("refund_all", "cancelled", "cancel"))
            or any(row.get(k) not in (None, "", "[]", [], {}, False)
                   for k in ("cancelled_cars", "cancelled_car_numbers", "refunded", "returned_buys")))


def _known(row):
    return bool(isinstance(row, dict) and (
        _true(row.get("official_result_available")) or row.get("actual")
        or row.get("payouts") or row.get("budget_official_outcome")
        or row.get("decided_at") or row.get("actual_trifecta")
        or row.get("actual_trifecta_buys")
        and row.get("actual_trifecta_buys") != "[]"))


def _identity(row):
    rid, day = row["id"], row["date"]
    if (not re.fullmatch(r"\d{12}", rid) or rid[4:] != day.replace("-", "")
            or int(rid[:2]) != row["number"]
            or datetime.fromtimestamp(row["close_at"], JST).date().isoformat() != day):
        raise ValueError("レース ID・日付・締切が不一致")
    initial = row["initial_budget"]
    if initial.get("status") != "allocated" or not check_saved_plan(initial, row["tickets"]):
        raise ValueError("初回配分の保存内容が不正")
    stamps = [timestamp(row["prediction_snapshot_at_jst"]),
              timestamp(initial["allocation_frozen_at_jst"]),
              timestamp(initial["odds_captured_at_jst"])]
    if any(t.timestamp() >= row["close_at"] or t.astimezone(JST).date().isoformat() != day
           for t in stamps):
        raise ValueError("初回保存が締切後または別日")
    return max(stamps)


def _quotes(snapshot, row, current):
    initial_saved = _identity(row)
    if snapshot.get("schema") != 1 or snapshot.get("source_kind") != "existing_public_page_response":
        raise ValueError("既存の公開ページ取得記録が必要")
    captured = timestamp(snapshot["captured_at"])
    close, now = row["close_at"], current.timestamp()
    if (captured <= initial_saved or captured.astimezone(JST).date().isoformat() != row["date"]
            or not 0 <= now - captured.timestamp() <= MAX_AGE
            or not MIN_LEAD <= close - captured.timestamp() <= MAX_LEAD
            or not MIN_LEAD <= close - now <= MAX_LEAD):
        raise ValueError("新しいオッズ待ち（締切 5〜40 分前・取得後 10 分以内）")
    meta, data = snapshot["meta"], snapshot["data"]
    race, schedule = data["race"], data["schedule"]
    rid = row["id"]
    if (str(meta.get("race_id")) != rid or str(race.get("id")) != rid
            or meta.get("date") != row["date"] or str(schedule.get("date")) != rid[4:]
            or str(meta.get("venue_id")).zfill(2) != rid[2:4]
            or meta.get("venue") != row["venue"]
            or meta.get("race_number") != row["number"] or race.get("number") != row["number"]
            or _epoch(meta.get("close_at")) != close or _epoch(race.get("closeAt")) != close):
        raise ValueError("取得オッズのレース・日付・締切が不一致")
    if (race.get("cancel") is not False or meta.get("cancel") is not False
            or race.get("decidedAt") or data.get("finalOdds") or data.get("results")):
        raise ValueError("中止・確定後のオッズは利用しません")
    entries = data.get("entries", [])
    declared = race.get("entriesNumber")
    if (type(declared) is not int or not 3 <= declared <= 9 or not isinstance(entries, list)
            or len(entries) != declared or any(not isinstance(e, dict) for e in entries)
            or sorted(e.get("number", -1) for e in entries) != list(range(1, declared + 1))
            or any(e.get("absent") is not False or str(e.get("raceId")) != rid for e in entries)
            or any(int(c) > declared for t in row["tickets"] for c in t.split("-"))):
        raise ValueError("欠場・出走車の不足があるため配分を保留")
    quotes = {}
    for item in data.get("trifecta", []):
        if str(item.get("raceId")) != rid:
            raise ValueError("オッズ ID が不一致")
        key = item.get("key", [])
        ticket = "-".join(map(str, key)) if isinstance(key, list) else ""
        if ticket not in row["tickets"]:
            continue
        if (ticket in quotes or item.get("unitPrice") != 100 or item.get("absent") is True):
            raise ValueError("重複・返還・単位不明のオッズ")
        price = number(item.get("odds"))
        if price >= number("9999.9"):
            raise ValueError("上限表示のオッズは利用しません")
        quotes[ticket] = str(price)
    if set(quotes) != set(row["tickets"]):
        raise ValueError("全買い目のオッズ取得待ち")
    if not re.fullmatch(r"[a-f0-9]{64}", str(snapshot.get("sha256", ""))) or not snapshot.get("url"):
        raise ValueError("オッズ出典の保存情報が不足")
    return quotes, captured


def _settlements(row, original, result, nexus=None, evidence=None):
    if _refund(original) or _refund(result) or _refund(nexus) or _refund(evidence):
        row["refund_pending"] = True
    candidates = [o for o in (row.get("official_outcome"), (evidence or {}).get("outcome"),
                  (original or {}).get("budget_official_outcome"), normalize_outcome(result or {})) if o]
    if candidates:
        # Prefer explicit official payoff IDs over compact results, including
        # after a day rollover. Preserve a complete/newer accepted payoff.
        def rank(value):
            try:
                observed = timestamp(value.get("source_observed_at", "")).timestamp()
            except (ValueError, TypeError):
                observed = float("-inf")
            return (bool(value.get("payout_complete")),
                    value.get("evidence") == "existing_public_page_response", observed)
        row["official_outcome"] = copy.deepcopy(max(candidates, key=rank))
    for plan_key, settlement_key in (("initial_budget", "initial_budget_settlement"),
                                      ("budget", "budget_settlement")):
        plan = row.get(plan_key)
        settled = settle(plan, row["tickets"], row.get("official_outcome"))
        if row.get("refund_pending") and plan and plan.get("status") == "allocated":
            settled = settle(plan, row["tickets"], None)
            settled["status"] = "refund_pending"
        row[settlement_key] = settled


def _summary(rows):
    paired = [r for r in rows if all(r.get(k, {}).get("status") == "settled"
              for k in ("initial_budget_settlement", "budget_settlement"))]
    summary = {s: sum(r["status"] == s for r in rows)
               for s in ("waiting", "allocated", "infeasible", "missed", "excluded")}
    summary.update(registered=len(rows), paired_races=len(paired),
                   scope="both_allocated_and_officially_settled")
    for name, key in (("initial", "initial_budget_settlement"), ("near_close", "budget_settlement")):
        invested = sum(r[key]["invested_yen"] for r in paired)
        payout = sum(r[key]["payout_yen"] for r in paired)
        hits = sum(bool(r[key]["hit"]) for r in paired)
        summary[name] = dict(races=len(paired), hits=hits, invested_yen=invested,
                             payout_yen=payout, profit_yen=payout-invested,
                             roi=payout/invested if invested else None,
                             hit_rate=hits/len(paired) if paired else None)
    return summary


def _write(root, payload, current):
    payload["updated_at"] = current.isoformat()
    payload["races"].sort(key=lambda r: (r["close_at"], r["id"]))
    payload["comparison"] = _summary(payload["races"])
    save_json(Path(root) / FEED, payload)


def run(root=ROOT, clock=now_jst):
    root, current = Path(root), _now(clock)
    baseline = read_json(root / FEED, {})
    if baseline and (baseline.get("schema") != 1 or baseline.get("policy", {}).get("id") != POLICY_ID):
        raise ValueError("unexpected near-close feed policy")
    payload = copy.deepcopy(baseline) or dict(schema=1, policy=dict(
        id=POLICY_ID, activated_at=current.isoformat(), mode="simulation",
        same_tickets=True, same_budget=True, quote_selection="first_eligible_complete_once",
        minimum_lead_seconds=MIN_LEAD, maximum_lead_seconds=MAX_LEAD,
        maximum_quote_age_seconds=MAX_AGE, backfill=False), races=[])
    rows = {r["id"]: r for r in payload["races"]}
    if len(rows) != len(payload["races"]):
        raise ValueError("duplicate near-close race")
    originals = {}
    path = root / LEDGER
    for line in path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []:
        if line.strip():
            value = json.loads(line)
            rid = str(value["race_id"])
            if rid in originals:
                raise ValueError("duplicate original forecast")
            originals[rid] = value
    results = {str(r["race_id"]): r for r in read_json(root / "outputs/latest_results.json", [])}
    nexus_rows = {str(r.get("race_id")): r for r in
                  read_json(root / "outputs/nexus_live.json", {}).get("races", [])}
    for rid, original in originals.items():
        if rid in rows or original.get("date") != current.date().isoformat():
            continue
        plan = original.get("budget")
        try:
            close = _epoch(original.get("close_at"))
            if close <= current.timestamp() or not plan or plan.get("status") != "allocated":
                continue
            row = dict(id=rid, date=original["date"], venue=original.get("venue", ""),
                       number=int(float(original["race_no"])), close_at=close,
                       prediction_snapshot_at_jst=original["snapshot_at_jst"],
                       tickets=copy.deepcopy(original["top12"]), initial_budget=copy.deepcopy(plan),
                       budget=None, status="waiting", reason="締切 5〜40 分前の新しいオッズを待っています",
                       registered_at_jst=current.isoformat())
            rows[rid] = row
            _identity(row)
        except (ValueError, TypeError, KeyError) as exc:
            if rid in rows:
                rows[rid].update(status="excluded", reason=str(exc))
    for rid, row in rows.items():
        original, result = originals.get(rid, {}), results.get(rid, {})
        nexus = nexus_rows.get(rid, {})
        if nexus.get("date") != row["date"] or nexus.get("race_no") != row["number"]:
            nexus = {}
        if not row.get("budget") and row["status"] not in ("excluded", "missed"):
            current = _now(clock)
            lead = row["close_at"] - current.timestamp()
            if (_known(original) or _known(result) or _refund(original)
                    or _refund(result) or _refund(nexus)):
                row.update(status="excluded", reason="結果・返還情報があるため新しい配分は作成しません")
            elif lead < MIN_LEAD:
                row.update(status="missed", reason="受付時間を経過・後付け配分はしません")
            elif lead <= MAX_LEAD:
                try:
                    path = root / SNAPSHOTS / row["date"] / "preclose" / (rid + ".json")
                    snapshot = read_json(path)
                    if not snapshot:
                        raise ValueError("既存収集の新しいオッズ取得待ち")
                    quotes, captured = _quotes(snapshot, row, current)
                    plan = allocate(row["tickets"], quotes, budget=row["initial_budget"]["budget_yen"])
                    saved = _now(clock)
                    if not MIN_LEAD <= row["close_at"] - saved.timestamp() <= MAX_LEAD:
                        raise ValueError("配分計算中に受付時間を経過")
                    plan.update(version=row["initial_budget"]["version"],
                                budget_selection="same_as_initial", comparison_policy_id=POLICY_ID,
                                odds_captured_at_jst=captured.isoformat(),
                                allocation_frozen_at_jst=saved.isoformat(),
                                allocation_attempted_at_jst=current.isoformat(),
                                source_url=snapshot["url"], provider="winticket",
                                source_sha256=snapshot["sha256"])
                    if plan["status"] == "allocated" and not check_saved_plan(plan, row["tickets"]):
                        raise ValueError("近締切配分の整合性エラー")
                    row.update(budget=plan, status=plan["status"], reason=plan["reason"],
                               source_sha256=snapshot["sha256"])
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    row["reason"] = str(exc)[:240]
        evidence = read_saved_page_outcome(root, dict(race_id=rid, date=row["date"],
                      race_no=row["number"], close_at=row["close_at"]))
        _settlements(row, original, result, nexus, evidence)
    payload["races"] = list(rows.values())
    _write(root, payload, _now(clock))
    gate_publication(root, baseline, clock=clock, lead_seconds=0)
    return read_json(root / FEED)


def gate_publication(root, baseline, clock=now_jst, lead_seconds=60):
    """Drop only unpublished allocations that no longer have publication lead."""
    root, current = Path(root), _now(clock)
    payload = read_json(root / FEED)
    if not payload:
        return []
    before = {r["id"]: r for r in (baseline or {}).get("races", [])}
    late = []
    for row in payload["races"]:
        if (row.get("budget") and not before.get(row["id"], {}).get("budget")
                and current.timestamp() + lead_seconds >= row["close_at"]):
            row.update(budget=None, status="missed", reason="公開前に締切が近づいたため新規配分を破棄")
            row.pop("source_sha256", None)
            row["budget_settlement"] = settle(None, row["tickets"], None)
            late.append(row["id"])
    if late:
        _write(root, payload, current)
    return late


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    report = run(parser.parse_args().root)
    print(json.dumps(report["comparison"], ensure_ascii=False))
