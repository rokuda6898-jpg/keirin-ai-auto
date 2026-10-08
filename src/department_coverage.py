"""Company-wide race forecast coverage, separate from eligibility to buy tickets.

The seven forecasting functions publish one entry per *known* race. Closed
races never get a prediction fabricated after their closing time.
"""
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from itertools import permutations
from zoneinfo import ZoneInfo

import pandas as pd

from common import OUTPUT_DIR

DEPARTMENTS = (
    "data_department", "pace_department", "line_department", "risk_department",
    "prediction_department", "strategist_department", "high_payout_department",
)
LABELS = {
    "data_department": "データ部", "pace_department": "展開部",
    "line_department": "ライン部", "risk_department": "リスク部",
    "prediction_department": "予想部", "strategist_department": "軍師",
    "high_payout_department": "高配当戦略部",
}
SPECIAL = DEPARTMENTS[4:]


def position_scenario(race, alternative=False):
    """Distinct 1/2/3 cars from available relative model weights."""
    if len(race) < 3 or race["car_no"].nunique() != len(race):
        return []
    options = sorted(int(x) for x in race["car_no"])
    scores = {}
    for column in ("p_win", "p_second", "p_third"):
        frame = pd.to_numeric(
            race[column] if column in race else pd.Series(0, index=race.index),
            errors="coerce",
        ).fillna(0).clip(lower=0)
        scores[column] = {
            int(car): float(value) for car, value in zip(race["car_no"], frame)
        }
    if max(scores["p_win"].values()) <= 0:
        return []
    if alternative:
        # This is an explicitly UNPRICED upset scenario, not a 100x pick.
        standard = position_scenario(race)
        if len(standard) != 3:
            return []
        competitors = sorted(options, key=lambda x: (-scores["p_win"][x], x))
        first = next((x for x in competitors if x != standard[0]), None)
        if first is None:
            return []
        others = [x for x in options if x != first]
        second = max(others, key=lambda x: (scores["p_second"][x], -x))
        third = max((x for x in others if x != second),
                    key=lambda x: (scores["p_third"][x], -x))
        return [first, second, third]
    return list(max(
        permutations(options, 3),
        key=lambda tup: (
            sum(math.log(max(scores[column][car], 1e-12))
                for column, car in zip(("p_win", "p_second", "p_third"), tup)),
            tuple(-car for car in tup),
        ),
    ))


def strategist_scenario(race, specialist_rows):
    usable = [r["top3_cars"] for r in specialist_rows
              if r.get("forecast_available") and len(r.get("top3_cars", [])) == 3]
    if not usable:
        return position_scenario(race), "model_fallback_no_specialist_vote"
    valid = sorted(int(x) for x in race["car_no"])
    votes = [Counter(int(p[pos]) for p in usable) for pos in range(3)]
    model = position_scenario(race)
    weights = {car: 0 for car in valid}
    if model:
        weights = {car: 3 - i for i, car in enumerate(model)}
    best = max(
        permutations(valid, 3),
        key=lambda pick: (
            sum(votes[i][car] * (3 - i) for i, car in enumerate(pick)),
            sum(weights.get(car, 0) * (3 - i) for i, car in enumerate(pick)),
            tuple(-car for car in pick),
        ),
    )
    return list(best), "independent_specialist_vote_consensus"


def _load_preserved(folder):
    path = folder / "all_department_prediction_ledger.jsonl"
    result = {}
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
                t = datetime.fromisoformat(entry["snapshot_at"])
                close = float(entry["close_at"])
                key = (str(entry["race_id"]), entry["department"])
                if (entry.get("forecast_available") and t.tzinfo is not None and
                        math.isfinite(close) and t.timestamp() < close and
                        (key not in result or entry["snapshot_at"] > result[key]["snapshot_at"])):
                    result[key] = entry
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
    return result


def build_all_department_coverage(pred, plans, specialist_rows, now, output_dir=OUTPUT_DIR):
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    plan_map = {str(p["race_id"]): p for p in plans}
    four_map = {(str(p["race_id"]), p["department"]): p for p in specialist_rows
                if p.get("department") in DEPARTMENTS[:4]}
    preserved = _load_preserved(folder)
    records = []
    fresh = []
    for race_id, race in pred.groupby("race_id", sort=False):
        rid = str(race_id)
        base = race.iloc[0]
        close = pd.to_numeric(base.get("close_at"), errors="coerce")
        valid_time = pd.notna(close) and math.isfinite(float(close))
        in_time = bool(valid_time and float(close) > now.timestamp())
        valid_entries = len(race) >= 3 and race["car_no"].nunique() == len(race)
        prefix = {
            "race_id": rid, "date": str(base.get("date", "")),
            "venue": str(base.get("venue", "")),
            "race_no": int(base.get("race_no", 0)),
            "close_at": float(close) if valid_time else None,
            "snapshot_at": now.isoformat(timespec="seconds"),
            "purchase_authorized": False,
        }
        existing_four = [four_map.get((rid, d)) for d in DEPARTMENTS[:4]]
        forecast_eligible = in_time and valid_entries
        for d, entry in zip(DEPARTMENTS[:4], existing_four):
            if entry is not None:
                normalized = {**prefix, **entry}
                records.append(normalized)
                if (forecast_eligible and normalized.get("forecast_available")
                        and normalized.get("close_at") is not None
                        and now.timestamp() < float(normalized["close_at"])):
                    fresh.append(normalized)
            else:
                old = preserved.get((rid, d))
                records.append(old if not forecast_eligible and old else {
                    **prefix, "department": d, "forecast_available": False,
                    "winner_car": None, "top3_cars": [], "tickets": [],
                    "display_status": "missing_specialist_forecast" if forecast_eligible
                                      else "closed_or_invalid_without_preclose",
                })

        specialty = [r for r in records[-4:] if r.get("race_id") == rid]
        main_scenario = position_scenario(race) if forecast_eligible else []
        alternatives = position_scenario(race, alternative=True) if forecast_eligible else []
        strategist, strategist_source = strategist_scenario(race, specialty) if forecast_eligible else ([], "closed")
        plan = plan_map.get(rid, {})
        high = plan.get("high_payout_department", {})
        # Use a genuinely quoted 100x proposal if supplied, otherwise a clearly
        # unpriced scenario. Do not classify the scenario as a valid bet.
        verified_hole = None
        for ticket in high.get("tickets", []):
            try:
                buy = str(ticket["buy"])
                car_list = [int(x) for x in buy.split("-")]
                price = float(ticket.get("odds_used", ticket.get("odds", 0)))
                if (len(car_list) == 3 and len(set(car_list)) == 3
                        and all(car in set(int(c) for c in race["car_no"]) for car in car_list)
                        and math.isfinite(price) and price >= 100):
                    verified_hole = car_list
                    break
            except (KeyError, TypeError, ValueError):
                continue
        if verified_hole and forecast_eligible:
            alternatives = verified_hole
            high_source = "quoted_100plus_reference"
        else:
            high_source = "unpriced_upset_scenario_not_100plus_bet"
        for d, top3, status in (
            ("prediction_department", main_scenario, "model_position_ranking"),
            ("strategist_department", strategist, strategist_source),
            ("high_payout_department", alternatives, high_source),
        ):
            old = preserved.get((rid, d))
            if not forecast_eligible and old:
                records.append({**old, "display_status": "preclose_forecast_preserved"})
                continue
            if not forecast_eligible:
                top3 = []
                status = "closed_or_invalid_without_preclose"
            entry = {
                **prefix, "department": d, "forecast_available": len(top3) == 3,
                "winner_car": top3[0] if top3 else None,
                "top3_cars": top3,
                "display_status": status,
                "tickets": [], "ticket_decision": "not_authorized_no_auto_purchase",
                "information_limitations": (
                    "No validated 100x quote; scenario only." if d == "high_payout_department"
                    and verified_hole is None else ""
                ),
            }
            records.append(entry)
            if entry["forecast_available"] and in_time:
                fresh.append(entry)

    # Report coverage as a matrix, never treat a missing forecast as success.
    by_race = defaultdict(dict)
    for row in records:
        key = (str(row["race_id"]), row["department"])
        if key[1] in by_race[key[0]]:
            raise RuntimeError(f"Duplicate department coverage entry: {key}")
        by_race[key[0]][key[1]] = row
    for rid, group in by_race.items():
        if set(group) != set(DEPARTMENTS):
            raise RuntimeError(f"Missing department response for {rid}: {set(DEPARTMENTS)-set(group)}")
    upcoming = [r for r in records if r["close_at"] is not None
                and r["close_at"] > now.timestamp()]
    uncovered = [r for r in upcoming if not r.get("forecast_available")]
    report = {
        "updated_at_jst": now.isoformat(timespec="seconds"),
        "race_count": len(by_race),
        "required_departments": list(DEPARTMENTS),
        "expected_responses": len(by_race) * len(DEPARTMENTS),
        "responses": len(records),
        "upcoming_responses": len(upcoming),
        "upcoming_missing_forecasts": len(uncovered),
        "coverage_status": "complete" if not uncovered else "incomplete",
        "scope": "known races with full entry data; never backfill expired race picks",
        "notes": "A forecast is mandatory; buying tickets is conditional on price, EV and safety checks.",
        "predictions": records,
    }
    (folder / "all_department_predictions.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if fresh:
        with (folder / "all_department_prediction_ledger.jsonl").open("a", encoding="utf-8") as handle:
            for row in fresh:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    render_all_department_coverage(report, folder)
    if uncovered:
        raise RuntimeError(f"Mandatory specialist forecasts missing for {len(uncovered)} upcoming race/department entries")
    return report


def render_all_department_coverage(report, folder):
    rows = defaultdict(list)
    for r in report["predictions"]:
        rows[str(r["race_id"])].append(r)
    body = (
        '<h1>全レース・全予想部署の提出状況</h1>'
        '<p>取得済み全レースについて7部署の予想を提出。締切後は締切前の記録のみ再掲し、後付け予想はしません。</p>'
        '<p>1着→2着→3着は仮説的な着順シナリオで、3連単の期待値や的中保証ではありません。'
        '買い目の購入可否は別判定です。高配当部の暫定シナリオは100倍以上の確認済み車券とは限りません。</p>'
        f'<p>対象{report["race_count"]}レース／提出{report["responses"]}件'
        f'／発走前の予想未成立{report["upcoming_missing_forecasts"]}件</p>'
    )
    for rid, group in rows.items():
        first = group[0]
        body += f'<section><h2>{html.escape(first["venue"])} {first["race_no"]}R</h2>'
        body += '<table><tr><th>部署</th><th>着順予想</th><th>情報状態</th><th>買い目</th></tr>'
        for r in sorted(group, key=lambda x: DEPARTMENTS.index(x["department"])):
            top3 = "-".join(map(str, r.get("top3_cars", []))) if r.get("forecast_available") else "予想未成立"
            tickets = str(len(r.get("tickets") or [])) + "点" if r.get("tickets") else "購入候補なし"
            body += (
                f'<tr><td>{html.escape(LABELS[r["department"]])}</td>'
                f'<td>{html.escape(top3)}</td>'
                f'<td>{html.escape(r.get("display_status", "未判定"))}</td>'
                f'<td>{html.escape(tickets)}</td></tr>'
            )
        body += '</table></section>'
    page = (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>全レース部署別予想｜NEXUS</title>'
        '<style>body{margin:0;padding:18px;background:#f4f7fb;color:#162c43;font-family:system-ui}'
        'main{max-width:1000px;margin:auto}section{background:white;border-radius:14px;padding:18px;margin:16px 0}'
        'table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid #ddd}'
        'p{line-height:1.7}a{color:#0965c7}@media(max-width:600px){table{font-size:12px}}</style>'
        '</head><body><main><p><a href="../index.html">今日の予想へ戻る</a></p>'
        + body + '</main></body></html>'
    )
    (folder / "all_department_predictions.html").write_text(page, encoding="utf-8")


def build_department_scoreboard(output_dir=OUTPUT_DIR):
    """Compare only pre-close frozen top-three forecasts with verified results.

    Maintain settled evidence across daily overwrites of latest_results.json.
    All seven departments are scored on the same available set of races.
    """
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    frozen = _load_preserved(folder)
    settled_path = folder / "all_department_settled.json"
    try:
        old = json.loads(settled_path.read_text(encoding="utf-8")) if settled_path.exists() else []
    except (ValueError, OSError, TypeError):
        old = []
    settled = {
        (str(row["race_id"]), row["department"]): row for row in old
        if row.get("department") in DEPARTMENTS and "race_id" in row
    }
    results_path = output_dir / "latest_results.json"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    except (ValueError, OSError, TypeError):
        results = []
    if not isinstance(results, list):
        results = []
    for outcome in results:
        if not isinstance(outcome, dict) or str(outcome.get("official_result_available", "")).lower() not in ("true", "1"):
            continue
        rid = str(outcome.get("race_id", ""))
        actual = str(outcome.get("actual_trifecta", "")).split("-")
        if len(actual) != 3 or len(set(actual)) != 3 or not all(x.isdigit() and 1 <= int(x) <= 9 for x in actual):
            continue
        for department in DEPARTMENTS:
            prediction = frozen.get((rid, department))
            if not prediction or len(prediction.get("top3_cars", [])) != 3:
                continue
            predicted = [str(car) for car in prediction["top3_cars"]]
            settled[(rid, department)] = {
                "race_id": rid, "department": department,
                "venue": prediction.get("venue", ""),
                "race_no": prediction.get("race_no", 0),
                "snapshot_at": prediction["snapshot_at"],
                "predicted": predicted,
                "actual": actual,
                "first_correct": predicted[0] == actual[0],
                "second_correct": predicted[1] == actual[1],
                "third_correct": predicted[2] == actual[2],
                "exact_trifecta": predicted == actual,
                "purchase_authorized": False,
            }
    rows = sorted(settled.values(), key=lambda row: (row["race_id"], row["department"]))
    settled_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    by_department = {}
    for department in DEPARTMENTS:
        samples = [row for row in rows if row["department"] == department]
        n = len(samples)
        by_department[department] = {
            "races": n,
            "first_correct": sum(bool(x["first_correct"]) for x in samples),
            "second_correct": sum(bool(x["second_correct"]) for x in samples),
            "third_correct": sum(bool(x["third_correct"]) for x in samples),
            "exact_trifecta": sum(bool(x["exact_trifecta"]) for x in samples),
            "first_hit_rate": (sum(bool(x["first_correct"]) for x in samples) / n if n else None),
            "exact_trifecta_rate": (sum(bool(x["exact_trifecta"]) for x in samples) / n if n else None),
        }
    now = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
    report = {
        "updated_at_jst": now,
        "note": "Independent advisory 1-2-3 scenario accuracy only. Not purchased tickets or real ROI.",
        "departments": by_department,
        "settled_predictions": len(rows),
    }
    (folder / "all_department_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    def percent(rate):
        return "未集計" if rate is None else f"{rate * 100:.1f}%"
    lines = []
    for department in DEPARTMENTS:
        d = by_department[department]
        lines.append(
            "<tr><td>" + html.escape(LABELS[department]) + "</td>"
            + f'<td>{d["races"]}</td><td>{d["first_correct"]}件 ({percent(d["first_hit_rate"])})</td>'
            + f'<td>{d["second_correct"]}件</td><td>{d["third_correct"]}件</td>'
            + f'<td>{d["exact_trifecta"]}件 ({percent(d["exact_trifecta_rate"])})</td></tr>'
        )
    page = (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>7部署の予想検証成績</title>'
        '<style>body{font-family:system-ui;background:#f4f7fb;padding:18px;color:#172b45}'
        'main{max-width:1000px;margin:auto}section{background:white;border-radius:14px;padding:20px}'
        'table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}'
        '.scroll{overflow:auto}p{line-height:1.8}a{color:#0965c7}</style></head><body><main>'
        '<p><a href="all_department_predictions.html">全レース7部署の予想へ戻る</a></p>'
        '<section><h1>7部署の着順予想・事後検証</h1>'
        '<p>各部署が締切前に提出した1着・2着・3着の並びを公式結果で検証。'
        '未提出や未確定のレースは成績に含めません。実購入した車券の的中率や回収率ではありません。</p>'
        '<div class="scroll"><table><tr><th>部署</th><th>検証レース</th>'
        '<th>1着的中</th><th>2着的中</th><th>3着的中</th><th>3連単の並び一致</th></tr>'
        + "".join(lines) + '</table></div>'
        '<p>各レースの予想提出状況は別画面で確認できます。'
        'サンプル数が少ない成績で自動的に買い目を変更しません。</p></section></main></body></html>'
    )
    (folder / "all_department_results.html").write_text(page, encoding="utf-8")
    return report
