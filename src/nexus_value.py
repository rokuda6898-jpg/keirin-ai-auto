"""Offline, gated portfolio-value research; never a live betting instruction.

The original NEXUS q, formations, marks and confidence are immutable inputs.
Only a separately fitted two-coefficient purchase probability sees a complete
preclose market.  Neither final odds nor outcomes enter ``assess``.  This module
does not fetch data, train the base model, publish forecasts, or place bets.

The 0.80 odds stress and 1.10 expected-return threshold are fixed, UNVALIDATED
initial hypotheses.  They are not probability bounds or a promise of profit.
"""
import argparse
import calendar
import copy
from datetime import date, datetime, timedelta
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

from nexus_input_snapshots import JST, ROOT, canonical, now_jst, read_json, save_json
from nexus_model.budget import allocate, settle
# Reuse the existing strict source/card/binding checks, never its ten-point
# selector or publisher.  The source-snapshot validator checks original input
# age (1800 s), identity, result markers, active rider IDs and input hashes.
from nexus_forward import _metadata, _source, _tickets
from nexus_source_replay import _snapshot, digest, forecast_core, stamp

POLICY_ID = "nexus-value-ev-v1"
BUDGETS = tuple(range(5000, 10001, 100))
POLICY = dict(
    policy_id=POLICY_ID, status="unvalidated_research_hypothesis",
    portfolio="all_original_formation_tickets_or_skip",
    probability="softmax(a*log(max(q,1e-15))+b*log(market_inverse_odds))",
    fit_objective="multinomial_log_loss", coefficient_bounds=[0.0, 4.0],
    odds_stress=0.80, expected_return_threshold=1.10,
    max_quote_age_seconds=300, max_quote_to_close_seconds=300,
    min_publication_lead_seconds=60, odds_ceiling_exclusive=9999.9,
    stress_rounding="floor_to_0.1_decimal_odds",
    allocated_budgets=list(BUDGETS), flat_yen_per_ticket=100,
    probability_lower_bound=False, prospective_performance_validated=False,
)
POLICY_SHA256 = digest(POLICY)
MODES = ("flat",) + tuple(str(b) for b in BUDGETS)
IMPLEMENTATION_FILES = ("nexus_value.py", "nexus_model/budget.py", "nexus_forward.py",
                        "nexus_source_replay.py", "nexus_model/replay_evidence.py", "nexus_input_snapshots.py")


class EvidenceError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _require(condition, code, message):
    if not condition:
        raise EvidenceError(code, message)


def _hash(value):
    _require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value),
             "invalid_hash", "完全な SHA-256 が必要です")
    return value


def _seal(value, field):
    value[field] = digest({k: v for k, v in value.items() if k != field})
    return value


def _verify(value, field):
    _require(isinstance(value, dict) and value.get(field) ==
             digest({k: v for k, v in value.items() if k != field}),
             "integrity_mismatch", "保存済み研究記録のハッシュが一致しません")


def _now(clock):
    value = clock()
    _require(isinstance(value, datetime) and value.tzinfo is not None,
             "invalid_clock", "実時刻にはタイムゾーンが必要です")
    return value.astimezone(JST)


def _add_months(day, count):
    year, month = divmod(day.year * 12 + day.month - 1 + count, 12)
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


def annual_period(start, end):
    """A calendar year, including leap days, with twelve anniversary folds."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    _require(last == _add_months(first, 12) - timedelta(days=1),
             "invalid_annual_period", "評価期間は開始日から完全な1年間が必要です")
    return first, last


def _model(manifest, manifest_sha256):
    _hash(manifest_sha256)
    first, last = map(date.fromisoformat, manifest["train"])
    _require(first <= last and isinstance(manifest.get("id"), str),
             "invalid_model_provenance", "学習期間を明示した固定モデルが必要です")
    return last + timedelta(days=1)


def implementation_sha256():
    source = Path(__file__).resolve().parent
    return digest({name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in IMPLEMENTATION_FILES})


def _ticket_keys(cars):
    return ["-".join(map(str, t)) for t in itertools.permutations(cars, 3)]


def validate_frame(frame, manifest, manifest_sha256):
    """Validate a result-free, already recorded decision context, without repair.

    ``decision_at`` and ``published_at`` are recorded source context, not the
    current execution time. The CLI labels the latter separately. Reconstructed
    after-close research q is deliberately not accepted as prospective evidence.
    """
    _require(isinstance(frame, dict) and set(frame) == {
        "race", "input_snapshot", "market_snapshot", "decision_at", "published_at"},
        "invalid_frame_schema", "結果を含まない固定の判断入力が必要です")
    row = frame["race"]
    _metadata(row)
    _require(not any(row.get(k) for k in (
        "closed", "cancelled", "result_complete", "winners", "payouts", "result",
        "refunded", "refund_all", "result_captured_at")),
        "outcome_observed", "結果取得済み・取消・締切済みの入力は購入判断に使えません")
    decision, published, close = map(stamp, (frame["decision_at"], frame["published_at"], row["close_at"]))
    _require(decision <= published <= close - timedelta(seconds=60),
             "publication_deadline", "判断と公開の両方が締切60秒前までに必要です")
    cutoff = _model(manifest, manifest_sha256)
    _require(cutoff <= date.fromisoformat(row["date"]),
             "base_model_leakage", "評価レースを学習済みのモデルは使用できません")
    prediction, evidence = _source(row, manifest, manifest_sha256, decision)
    _require(bool(prediction.get("input_sha256")), "missing_input_hash", "元入力の一致証明が必要です")
    original, market = frame["input_snapshot"], frame["market_snapshot"]
    _snapshot(original, row, prediction, decision, original=True)
    _snapshot(market, row, prediction, decision, original=False)
    quote_at = stamp(market["captured_at"])
    _hash(market["sha256"])
    _require(stamp(prediction["input_captured_at"]) <= quote_at <= decision,
             "quote_sequence", "元入力・オッズ取得・判断の時刻が整合しません")
    _require(0 <= (decision - quote_at).total_seconds() <= 300,
             "stale_quote", "オッズ取得から5分を超えているため購入判断は未評価です")
    _require(0 < (close - quote_at).total_seconds() <= 300,
             "quote_outside_close_window", "締切5分以内の全オッズ保存記録が必要です")
    keys = _ticket_keys(evidence["cars"])
    quotes = {}
    for value in market["data"].get("trifecta", []):
        _require(isinstance(value, dict), "invalid_market", "オッズ原本を確認できません")
        if value.get("absent") is True:
            continue
        key = value.get("key")
        _require(isinstance(key, list) and len(key) == 3 and len(set(key)) == 3
                 and all(type(c) is int and c in evidence["cars"] for c in key)
                 and str(value.get("raceId")) == row["race_id"]
                 and value.get("unitPrice") == 100 and value.get("absent") is False,
                 "invalid_market_identity", "同じ出走者・レースの全オッズが必要です")
        ticket = "-".join(map(str, key))
        odds = value.get("odds")
        _require(not isinstance(odds, bool) and isinstance(odds, (int, float))
                 and math.isfinite(odds) and 1 <= odds < POLICY["odds_ceiling_exclusive"],
                 "censored_or_invalid_quote", "欠測・上限表示を含むオッズでは市場確率を計算できません")
        _require(ticket not in quotes, "duplicate_quote", "同じ買い目のオッズが重複しています")
        quotes[ticket] = odds
    _require(set(quotes) == set(keys), "incomplete_market", "市場確率には全着順の同時点オッズが必要です")
    return dict(row=row, prediction=prediction, evidence=evidence,
                q=dict(zip(keys, evidence["q"])), quotes=quotes, quote_at=quote_at,
                decision_at=decision, published_at=published)


def purchase_probabilities(q, quotes, coefficients):
    """Pure two-feature transform, not a validated betting recommendation."""
    _require(isinstance(q, dict) and q and set(q) == set(quotes),
             "probability_market_mismatch", "全qと全オッズの買い目が一致する必要があります")
    a, b = coefficients["q"], coefficients["market"]
    _require(all(not isinstance(v, bool) and isinstance(v, (int, float))
                 and math.isfinite(v) and 0 <= v <= 4 for v in (a, b)),
             "invalid_coefficients", "校正係数は固定範囲内の有限値が必要です")
    _require(all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                 and 0 <= v <= 1 for v in q.values()) and abs(math.fsum(q.values()) - 1) <= 1e-8,
             "invalid_q", "元qは完全で正規化済みである必要があります")
    inverse = {}
    for k, odds in quotes.items():
        _require(not isinstance(odds, bool) and isinstance(odds, (int, float))
                 and math.isfinite(odds) and odds > 0, "invalid_quote", "有限の正のオッズが必要です")
        inverse[k] = 1 / odds
    total = math.fsum(inverse.values())
    score = {k: a * math.log(max(q[k], 1e-15)) + b * math.log(inverse[k] / total) for k in q}
    peak = max(score.values())
    exp = {k: math.exp(v - peak) for k, v in score.items()}
    normalizer = math.fsum(exp.values())
    return {k: v / normalizer for k, v in exp.items()}


def stressed_odds(quotes):
    # Payouts are per 100 yen. Round down once to a 10-yen payout tick before
    # allocation; Fraction avoids an optimistic floating-point rounding boundary.
    result = {}
    for key, odds in quotes.items():
        ticks = Fraction(str(odds)) * Fraction(4, 5) * 10
        result[key] = float(Fraction(ticks.numerator // ticks.denominator, 10))
    return result


def expected_return(probability, stakes, quotes):
    total = sum(stakes.values())
    _require(total > 0 and all(type(v) is int and v > 0 and v % 100 == 0 for v in stakes.values()),
             "invalid_stakes", "各買い目は100円単位の正の金額が必要です")
    # Decimal odds are the TOTAL return multiplier; no second subtraction of
    # the stake or takeout belongs in this expression.
    return math.fsum(probability[t] * amount * quotes[t] for t, amount in stakes.items()) / total


def _outcome(raw, row, *, latest=None):
    _require(isinstance(raw, dict) and raw.get("complete") is True
             and raw.get("race_id") == row["race_id"] and raw.get("date") == row["date"],
             "missing_official_outcome", "同じレースの確定結果原本が必要です")
    _hash(raw.get("source_sha256"))
    captured = stamp(raw["captured_at"])
    _require(captured >= stamp(row["close_at"]) and (latest is None or captured <= latest),
             "outcome_timing", "結果を利用可能になった時刻が整合しません")
    cars = [r["car"] for r in row["riders"]]
    keys = set(_ticket_keys(cars))
    payouts, refunded = raw.get("payouts"), raw.get("refunded", [])
    _require(isinstance(payouts, dict) and isinstance(refunded, list)
             and len(refunded) == len(set(refunded)) and set(refunded) <= keys
             and set(payouts) <= keys and not set(refunded) & set(payouts)
             and type(raw.get("refund_all", False)) is bool
             and all(type(p) is int and p > 0 for p in payouts.values())
             and not (raw.get("refund_all") and payouts)
             and (bool(payouts) or raw.get("refund_all")),
             "invalid_official_outcome", "確定払戻・返還の原本を確認できません")
    return payouts, refunded, raw.get("refund_all", False)


def fit_calibrator(records, manifest, manifest_sha256, *, calibration_start,
                   calibration_end, evaluation_start, evaluation_end, clock=now_jst):
    """Fit two bounded coefficients by race-level NLL on independent records.

    The actual fit must finish before the evaluation year begins. There is no
    raw-q fallback, ROI optimization, annual-period fitting or hidden warmup.
    Dead heats and refunds are retained as explicit exclusions from fitting;
    evaluation settlement still handles all their winning/refunded tickets.
    """
    import numpy as np
    from scipy.optimize import minimize

    first, last = annual_period(evaluation_start, evaluation_end)
    cal_first, cal_last = map(date.fromisoformat, (calibration_start, calibration_end))
    cutoff, started = _model(manifest, manifest_sha256), _now(clock)
    evaluation_at = datetime.combine(first, datetime.min.time(), JST)
    _require(cutoff <= cal_first <= cal_last < first,
             "calibration_period_leakage", "モデル学習後・評価年より前の独立した校正期間が必要です")
    _require(started < evaluation_at and cal_last < started.date(),
             "fit_not_before_evaluation", "校正は対象年の開始前に実時刻で凍結する必要があります")
    _require(isinstance(records, list) and records, "missing_calibration", "独立した校正記録がありません")
    matrices, lengths, winners, ids, exclusions, seen, latest = [], [], [], [], [], set(), None
    for record in records:
        _require(isinstance(record, dict) and set(record) == {"frame", "outcome"},
                 "invalid_calibration_schema", "校正の判断入力と結果を分離してください")
        context = validate_frame(record["frame"], manifest, manifest_sha256)
        row, rid = context["row"], context["row"]["race_id"]
        _require(rid not in seen, "duplicate_calibration_race", "校正レースが重複しています")
        seen.add(rid)
        _require(cal_first <= date.fromisoformat(row["date"]) <= cal_last,
                 "calibration_outside_period", "校正期間外のレースが含まれています")
        payouts, refunded, refund_all = _outcome(record["outcome"], row, latest=started)
        observed = stamp(record["outcome"]["captured_at"])
        _require(observed < evaluation_at, "calibration_label_leakage", "評価年開始後の結果は校正に使えません")
        latest = max(latest, observed) if latest else observed
        if len(payouts) != 1 or refunded or refund_all:
            exclusions.append(dict(race_id=rid, reason_code="dead_heat_or_refund_not_fit"))
            continue
        keys = list(context["q"])
        inverse = np.array([1 / context["quotes"][k] for k in keys], dtype=float)
        features = np.column_stack((np.log([max(context["q"][k], 1e-15) for k in keys]),
                                    np.log(inverse / inverse.sum())))
        features -= features.mean(axis=0)
        matrices.append(features)
        lengths.append(len(keys))
        winners.append(keys.index(next(iter(payouts))))
        ids.append(rid)
    _require(bool(ids), "no_fit_labels", "校正に使える単独的中・返還なしの結果がありません")
    matrix = np.vstack(matrices)
    _require(np.linalg.matrix_rank(matrix) == 2, "unidentifiable_calibration", "2係数を区別できる校正入力が必要です")
    lengths = np.asarray(lengths)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    target = starts + np.asarray(winners)

    def objective(theta):
        logits = matrix @ theta
        peak = np.maximum.reduceat(logits, starts)
        exp = np.exp(logits - np.repeat(peak, lengths))
        sums = np.add.reduceat(exp, starts)
        probability = exp / np.repeat(sums, lengths)
        loss = (peak + np.log(sums) - logits[target]).mean()
        gradient = ((matrix * probability[:, None]).sum(axis=0) - matrix[target].sum(axis=0)) / len(ids)
        return float(loss), gradient

    fitted = minimize(objective, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                      bounds=[(0.0, 4.0), (0.0, 4.0)], options={"maxiter":500, "ftol":1e-13, "gtol":1e-9})
    _require(fitted.success and math.isfinite(float(fitted.fun)),
             "fit_failed", "校正の最適化が収束しませんでした")
    finished = _now(clock)
    _require(started <= finished < evaluation_at, "fit_not_before_evaluation",
             "計算中に対象年が始まったため、この校正は採用できません")
    artifact = dict(schema=1, kind="independent_two_coefficient_calibrator", policy_id=POLICY_ID,
        policy_sha256=POLICY_SHA256, implementation_sha256=implementation_sha256(),
        model_id=manifest["id"], model_manifest_sha256=manifest_sha256,
        training_cutoff_exclusive=cutoff.isoformat(), calibration_start=calibration_start,
        calibration_end=calibration_end, evaluation_start=evaluation_start, evaluation_end=evaluation_end,
        fit_objective="multinomial_log_loss", coefficients=dict(q=float(fitted.x[0]), market=float(fitted.x[1])),
        calibration_race_ids=ids, calibration_all_race_ids=sorted(seen), calibration_races=len(ids),
        calibration_exclusions=exclusions, calibration_dataset_sha256=digest(records),
        calibration_outcomes_latest_at=latest.isoformat(), calibration_log_loss=float(fitted.fun),
        raw_q_log_loss=objective(np.array([1.0, 0.0]))[0],
        optimizer=dict(name="L-BFGS-B", bounds=[[0.0,4.0],[0.0,4.0]], maxiter=500,
                       ftol=1e-13, gtol=1e-9, iterations=int(fitted.nit)),
        fit_started_at=started.isoformat(), fitted_at=finished.isoformat(),
        reliability_validated=False, probability_lower_bound=False,
        performance_status="unvalidated_research_hypothesis")
    return _seal(artifact, "artifact_sha256")


def validate_calibrator(artifact, manifest, manifest_sha256):
    _verify(artifact, "artifact_sha256")
    first, _ = annual_period(artifact["evaluation_start"], artifact["evaluation_end"])
    cal_first, cal_last = map(date.fromisoformat, (artifact["calibration_start"], artifact["calibration_end"]))
    cutoff = _model(manifest, manifest_sha256)
    eval_at = datetime.combine(first, datetime.min.time(), JST)
    _require(artifact.get("schema") == 1 and artifact.get("kind") == "independent_two_coefficient_calibrator"
             and artifact.get("policy_id") == POLICY_ID and artifact.get("policy_sha256") == POLICY_SHA256
             and artifact.get("implementation_sha256") == implementation_sha256()
             and artifact.get("model_id") == manifest["id"]
             and artifact.get("model_manifest_sha256") == manifest_sha256
             and artifact.get("training_cutoff_exclusive") == cutoff.isoformat()
             and artifact.get("fit_objective") == "multinomial_log_loss",
             "calibrator_policy_mismatch", "固定モデル・方式に一致する校正原本が必要です")
    _require(cutoff <= cal_first <= cal_last < first
             and stamp(artifact["calibration_outcomes_latest_at"]) <= stamp(artifact["fit_started_at"])
             <= stamp(artifact["fitted_at"]) < eval_at
             and cal_last < stamp(artifact["fit_started_at"]).astimezone(JST).date(),
             "calibrator_period_leakage", "校正期間・結果利用時刻が評価年と独立していません")
    ids, all_ids = artifact.get("calibration_race_ids"), artifact.get("calibration_all_race_ids")
    _require(isinstance(ids, list) and ids and len(set(ids)) == len(ids)
             and isinstance(all_ids, list) and len(set(all_ids)) == len(all_ids)
             and set(ids) <= set(all_ids) and artifact.get("calibration_races") == len(ids),
             "invalid_calibration_receipt", "独立した校正レース一覧が必要です")
    for rid in all_ids:
        _require(isinstance(rid, str) and re.fullmatch(r"\d{12}", rid)
                 and cal_first <= datetime.strptime(rid[4:], "%Y%m%d").date() <= cal_last,
                 "invalid_calibration_receipt", "校正レースの期間が一致しません")
    _hash(artifact.get("calibration_dataset_sha256"))
    purchase_probabilities({"one":0.5,"two":0.5}, {"one":2,"two":2}, artifact["coefficients"])
    _require(artifact.get("reliability_validated") is False and artifact.get("probability_lower_bound") is False
             and artifact.get("performance_status") == "unvalidated_research_hypothesis",
             "unsupported_validation_claim", "校正fitの成功を性能検証済みとは扱えません")
    return artifact


def _mode(probability, tickets, quotes, budget=None):
    if budget is None:
        plan = dict(available=True, budget=len(tickets) * 100, stakes=dict.fromkeys(tickets,100))
    elif len(tickets) * 100 > budget:
        plan = dict(available=False, budget=budget, reason_code="ticket_count_exceeds_budget")
    else:
        plan = allocate(tickets, quotes, budget)
    if not plan["available"]:
        return dict(status="skip", reason_code=plan["reason_code"], reason=plan.get("reason", "全買い目を保つ配分が成立しません"),
                    budget=plan["budget"], stakes=None, expected_return_multiple=None,
                    minimum_stress_return_yen=plan.get("best_possible_return"))
    ev = expected_return(probability, plan["stakes"], quotes)
    minimum = min(Fraction(str(quotes[t])) * amount for t, amount in plan["stakes"].items())
    minimum_yen = minimum.numerator // minimum.denominator
    _require(budget is None or minimum >= budget, "allocation_below_budget", "ストレス後払戻の予算割れ配分です")
    buy = ev >= POLICY["expected_return_threshold"]
    return dict(status="buy" if buy else "skip", reason_code="expected_value_pass" if buy else "expected_value_below_threshold",
                reason="未検証の期待回収倍率条件に適合" if buy else "未検証の期待回収倍率条件に届かず見送り",
                budget=plan["budget"], stakes=plan["stakes"], expected_return_multiple=ev,
                minimum_stress_return_yen=minimum_yen, stress_quote_break_even=minimum >= plan["budget"])


def assess(frame, calibrator, manifest, manifest_sha256, *, clock=now_jst):
    """Return separate offline buy/skip/unavailable hypotheses for all budgets."""
    row = frame.get("race", {}) if isinstance(frame, dict) else {}
    prediction = row.get("prediction") or {}
    report = dict(schema=1, kind="offline_portfolio_assessment", policy_id=POLICY_ID,
        policy_sha256=POLICY_SHA256, policy=copy.deepcopy(POLICY), computed_at=_now(clock).isoformat(),
        race_id=row.get("race_id"), date=row.get("date"), close_at=row.get("close_at"),
        riders=copy.deepcopy(row.get("riders", [])), decision_at=frame.get("decision_at"),
        published_at=frame.get("published_at"), is_live_purchase_recommendation=False,
        forecast={k:copy.deepcopy(prediction.get(k)) for k in ("model_id", "tickets", "blocks", "count", "marks", "confidence")},
        p_bet=None, stress_odds=None, annual_100_achieved=None,
        source_forecast_sha256=digest(forecast_core(prediction)), frame_sha256=digest(frame))
    try:
        context = validate_frame(frame, manifest, manifest_sha256)
        _require(context["published_at"] <= stamp(report["computed_at"]),
                 "future_decision_context", "実行時点より未来の保存・公開記録は使用できません")
        _require(calibrator is not None, "missing_independent_calibrator", "独立期間の校正が未検証のため購入判断は保留です")
        validate_calibrator(calibrator, manifest, manifest_sha256)
        _require(calibrator["evaluation_start"] <= row["date"] <= calibrator["evaluation_end"]
                 and stamp(calibrator["fitted_at"]) <= context["decision_at"],
                 "calibrator_outside_evaluation", "事前に凍結した校正の対象年ではありません")
        probability = purchase_probabilities(context["q"], context["quotes"], calibrator["coefficients"])
        quotes = stressed_odds(context["quotes"])
        tickets = context["prediction"]["tickets"]
        report.update(status="assessed", p_bet=probability, stress_odds={t:quotes[t] for t in tickets},
            quote_at=context["quote_at"].isoformat(), calibrator_sha256=calibrator["artifact_sha256"],
            source_evidence_sha256=context["evidence"]["evidence_sha256"],
            market_snapshot_sha256=digest(frame["market_snapshot"]),
            flat=_mode(probability, tickets, quotes),
            allocated={str(b):_mode(probability,tickets,quotes,b) for b in BUDGETS})
    except (EvidenceError, KeyError, TypeError, ValueError, OverflowError) as exc:
        unavailable = dict(status="unavailable", reason_code=getattr(exc,"code","invalid_preclose_evidence"),
            reason=str(exc), stakes=None, expected_return_multiple=None, minimum_stress_return_yen=None)
        report.update(status="unavailable", reason_code=unavailable["reason_code"], reason=unavailable["reason"],
                      flat=copy.deepcopy(unavailable), allocated={str(b):dict(unavailable,budget=b) for b in BUDGETS})
    return _seal(report, "receipt_sha256")


def _inventory(inventory):
    _verify(inventory, "inventory_sha256")
    first, last = annual_period(inventory["start"], inventory["end"])
    _require(inventory.get("schema") == 1 and isinstance(inventory.get("days"), list),
             "invalid_inventory", "年間の全日程原本が必要です")
    days, races = set(), {}
    for item in inventory["days"]:
        day = date.fromisoformat(item["date"])
        _require(first <= day <= last and day not in days and item.get("complete") is True,
                 "incomplete_inventory", "年間の日程に重複・欠測があります")
        days.add(day)
        _hash(item.get("source_sha256"))
        _require(isinstance(item.get("race_ids"), list), "invalid_inventory", "開催なしの日も全日付の記録が必要です")
        for rid in item["race_ids"]:
            _require(isinstance(rid,str) and re.fullmatch(r"\d{12}",rid) and 1 <= int(rid[:2]) <= 12
                     and rid[4:] == day.strftime("%Y%m%d") and rid not in races,
                     "invalid_inventory", "日程原本のレースIDが整合しません")
            races[rid] = day.isoformat()
    _require(len(days) == (last-first).days+1, "incomplete_inventory", "年間の全日付の原本が揃っていません")
    return first, last, races


def _metrics(items):
    nonvoid = [(rid, s) for rid, s in items if not s["void"]]
    stake = sum(s["net_stake"] for _,s in nonvoid)
    payout = sum(s["winnings"] for _,s in nonvoid)
    hits = sum(s["hit"] for _,s in nonvoid)
    return dict(races=len(nonvoid), hits=hits, hit_rate=hits/len(nonvoid) if nonvoid else None,
                stake_yen=stake, payout_yen=payout, profit_yen=payout-stake,
                gross_stake_yen=sum(s["stake"] for _,s in items), refund_yen=sum(s["refund"] for _,s in items),
                roi=payout/stake if stake else None, race_ids=[rid for rid,_ in nonvoid])


def _validate_assessment(report, calibrator):
    _verify(report, "receipt_sha256")
    _require(report.get("schema") == 1 and report.get("kind") == "offline_portfolio_assessment"
             and report.get("policy_id") == POLICY_ID and report.get("policy_sha256") == POLICY_SHA256
             and report.get("is_live_purchase_recommendation") is False
             and report.get("status") in ("assessed", "unavailable"),
             "invalid_assessment", "固定方式の研究判定原本が必要です")
    if report["status"] == "unavailable":
        _require(report.get("p_bet") is None and all(
            (report["flat"] if name == "flat" else report["allocated"][name]).get("status") == "unavailable"
            and (report["flat"] if name == "flat" else report["allocated"][name]).get("stakes") is None for name in MODES),
            "invalid_assessment", "未評価レースを見送りや購入へ置き換えることはできません")
        return
    _require(report.get("calibrator_sha256") == calibrator["artifact_sha256"],
             "mixed_calibrators", "同じ年内で校正係数を差し替えることはできません")
    _require(stamp(calibrator["fitted_at"]) <= stamp(report["decision_at"])
             <= stamp(report["published_at"]) <= stamp(report["close_at"])-timedelta(seconds=60),
             "assessment_deadline", "判定原本の時系列が一致しません")
    cars = [r["car"] for r in report["riders"]]
    _require(3 <= len(cars) <= 9 and len(set(cars)) == len(cars), "invalid_assessment", "有効な出走者が必要です")
    tickets = _tickets(report["forecast"],cars)
    probability, quotes = report["p_bet"], report["stress_odds"]
    _require(isinstance(probability,dict) and set(probability) == set(_ticket_keys(cars))
             and all(not isinstance(p,bool) and isinstance(p,(int,float)) and math.isfinite(p) and 0 <= p <= 1
                     for p in probability.values()) and abs(math.fsum(probability.values())-1) <= 1e-8
             and isinstance(quotes,dict) and set(quotes) == set(tickets)
             and all(not isinstance(p,bool) and isinstance(p,(int,float)) and math.isfinite(p) and p > 0 for p in quotes.values()),
             "invalid_assessment", "全購入確率と保存済みストレスオッズが必要です")
    for name in MODES:
        mode = report["flat"] if name == "flat" else report["allocated"][name]
        expected = _mode(probability,tickets,quotes,None if name == "flat" else int(name))
        _require(canonical(mode) == canonical(expected), "changed_ev_decision",
                 "保存した全買い目・配分・期待回収倍率・見送り根拠が固定方式と一致しません")


def annual_audit(inventory, assessments, outcomes, calibrator, manifest, manifest_sha256):
    """Audit a fixed-year cohort, preserving missing data and mode-specific IDs.

    Complete hypothetical settlement is reported separately from real purchased
    performance. No confidence interval or adoption claim is inferred from ROI.
    """
    report = dict(schema=1, policy_id=POLICY_ID, policy_sha256=POLICY_SHA256,
        status="unverified", annual_100_achieved=None, actual_purchases=False,
        annual_roi=None, adoption_validated=False, reasons=[], summary=None)
    try:
        first, last, expected = _inventory(inventory)
        report.update(start=first.isoformat(), end=last.isoformat(), expected_races=len(expected),
                      inventory_sha256=inventory["inventory_sha256"])
        validate_calibrator(calibrator,manifest,manifest_sha256)
        _require((calibrator["evaluation_start"],calibrator["evaluation_end"]) == (inventory["start"],inventory["end"]),
                 "evaluation_period_mismatch", "校正と年間原本の評価期間が違います")
        saved, results = {}, {}
        for item in assessments:
            _require(item.get("race_id") not in saved, "duplicate_assessment", "同じレースの購入判断が複数あります")
            _validate_assessment(item,calibrator)
            saved[item["race_id"]] = item
        for item in outcomes:
            _require(item.get("race_id") not in results, "duplicate_outcome", "同じレースの確定結果が複数あります")
            results[item["race_id"]] = item
        missing, extra = sorted(set(expected)-set(saved)), sorted(set(saved)-set(expected))
        result_missing, result_extra = sorted(set(expected)-set(results)), sorted(set(results)-set(expected))
        report.update(missing_race_ids=missing, extra_race_ids=extra,
                      missing_outcome_race_ids=result_missing, extra_outcome_race_ids=result_extra)
        _require(not (missing or extra or result_missing or result_extra),
                 "incomplete_annual_cohort", "全レースの判断と結果が揃うまで年間成績は未評価です")
        bymode = {name:[] for name in MODES}
        skipped = {name:[] for name in MODES}
        unavailable = {name:[] for name in MODES}
        baseline, race_records = [], []
        for rid in sorted(expected):
            item = saved[rid]
            _require(item.get("date") == expected[rid], "race_date_mismatch", "年間原本と判断のレース日付が違います")
            row = dict(race_id=rid,date=item["date"],close_at=item["close_at"],riders=item["riders"])
            payouts, refunded, refund_all = _outcome(results[rid],row)
            tickets = item["forecast"].get("tickets")
            if tickets:
                baseline.append((rid,settle(dict.fromkeys(tickets,100),payouts,refunded,refund_all)))
            settlements = {}
            for name in MODES:
                mode = item["flat"] if name == "flat" else item["allocated"][name]
                if mode["status"] == "unavailable":
                    unavailable[name].append(rid)
                elif mode["status"] == "skip":
                    skipped[name].append(rid)
                else:
                    value = settle(mode["stakes"],payouts,refunded,refund_all)
                    bymode[name].append((rid,value))
                    settlements[name] = value
            race_records.append(dict(race_id=rid,date=item["date"],
                decision_receipt_sha256=item["receipt_sha256"],outcome=copy.deepcopy(results[rid]),settlements=settlements))
        summary = {}
        for name in MODES:
            measured = _metrics(bymode[name])
            complete = not unavailable[name] and len(baseline) == len(expected)
            summary[name] = dict(status="complete_hypothetical" if complete else "incomplete",
                annual_roi=measured["roi"] if complete else None,
                historical_roi_exceeds_100=(measured["roi"] > 1) if complete and measured["roi"] is not None else None,
                annual_100_achieved=None, observed=measured, purchase_race_ids=[rid for rid,_ in bymode[name]],
                purchase_coverage=len(bymode[name])/len(expected) if expected else 0,
                skip_race_ids=skipped[name], unavailable_race_ids=unavailable[name],
                monthly=[dict(start=_add_months(first,n).isoformat(),
                    end=(_add_months(first,n+1)-timedelta(days=1)).isoformat(),
                    **_metrics([(rid,s) for rid,s in bymode[name] if
                        _add_months(first,n).isoformat() <= expected[rid] < _add_months(first,n+1).isoformat()])) for n in range(12)])
            summary[name]["daily"] = [dict(date=day.isoformat(), **_metrics([
                (rid,s) for rid,s in bymode[name] if expected[rid] == day.isoformat()]))
                for day in (first+timedelta(days=n) for n in range((last-first).days+1))]
            biggest = max((pair for pair in bymode[name] if pair[1]["winnings"] > 0),
                          key=lambda pair:(pair[1]["winnings"],pair[0]), default=None)
            summary[name]["without_largest_payout_race"] = dict(
                excluded_race_id=biggest[0] if biggest else None, purpose="robustness_only_not_selection",
                **_metrics([pair for pair in bymode[name] if biggest is None or pair[0] != biggest[0]]))
        complete = all(s["status"] == "complete_hypothetical" for s in summary.values())
        report.update(status="complete_hypothetical" if complete else "incomplete",
            summary=dict(flat=summary.pop("flat"),allocated=summary), race_records=race_records,
            all_race_forecast_metrics=_metrics(baseline),
            all_race_forecast_coverage=len(baseline)/len(expected) if expected else 0)
        if not complete:
            report["reasons"].append("購入判断が未評価のレースを年間母数から除外できません")
    except (EvidenceError,KeyError,TypeError,ValueError,OverflowError) as exc:
        report["reasons"].append(str(exc))
        report["reason_code"] = getattr(exc,"code","invalid_annual_evidence")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit","assess","annual-audit"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT/"models/nexus53/manifest.json")
    args = parser.parse_args(argv)
    # Research commands cannot replace the input, model, or live/public feeds.
    _require(args.output.resolve() not in (args.input.resolve(),args.manifest.resolve())
             and args.output.resolve() not in {(ROOT/"outputs"/name).resolve() for name in
                ("nexus_live.json","nexus_results.json","nexus_forward.json","nexus_roi_research.json")},
             "protected_output", "入力原本・モデル・公開予想は研究CLIの出力先にできません")
    _require(not args.output.exists(), "output_exists", "保存済み研究原本は上書きできません。新しい出力先を指定してください")
    raw = args.manifest.read_bytes()
    manifest, model_hash = json.loads(raw), hashlib.sha256(raw).hexdigest()
    bundle = read_json(args.input)
    _require(isinstance(bundle,dict) and bundle.get("schema") == 1 and bundle.get("policy_id") == POLICY_ID,
             "invalid_bundle", "固定方式の研究入力bundleが必要です")
    if args.command == "fit":
        try:
            output = fit_calibrator(bundle["records"],manifest,model_hash,
                **{k:bundle[k] for k in ("calibration_start","calibration_end","evaluation_start","evaluation_end")})
        except (EvidenceError,KeyError,TypeError,ValueError,OverflowError) as exc:
            output = dict(schema=1,policy_id=POLICY_ID,status="unavailable",reason_code=getattr(exc,"code","invalid_calibration"),
                          reason=str(exc),fitted_at=None,annual_100_achieved=None)
    elif args.command == "assess":
        output = assess(bundle["frame"],bundle.get("calibrator"),manifest,model_hash)
    else:
        output = annual_audit(bundle["inventory"],bundle["assessments"],bundle["outcomes"],bundle.get("calibrator"),manifest,model_hash)
    # No implicit overwrites of frozen calibration or assessment evidence.
    _require(not args.output.exists(), "output_exists", "保存済み研究原本は上書きできません。新しい出力先を指定してください")
    save_json(args.output,output)
    print(canonical(dict(output=str(args.output),status=output.get("status", "fitted"),policy_id=POLICY_ID,
                         annual_100_achieved=None)))


if __name__ == "__main__":
    main()
