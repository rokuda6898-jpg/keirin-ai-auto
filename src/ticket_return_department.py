"""Prospective, immutable full-portfolio audit; never grants purchase authority."""
import json
import html
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from betting_logic import STRATEGY_VERSION
from common import OUTPUT_DIR


def read_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def truth(value):
    return str(value).lower() in {"true", "1"}


def save_snapshots(plans, shadow_bets, now_jst, output_dir=OUTPUT_DIR, position_rows=None, candidate_rows=None):
    """Freeze even eligible zero-ticket decisions, with their exact ticket set."""
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "ticket_return_snapshots.jsonl"
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            seen.add((row["strategy_version"], row["race_id"], row["snapshot_at"]))
    rows = []
    for plan in plans:
        if not plan.get("timing_eligible"):
            continue
        close_at = plan.get("close_at")
        if close_at is None or not now_jst.timestamp() < float(close_at):
            continue
        race_id = str(plan["race_id"])
        key = (plan.get("strategy_version", STRATEGY_VERSION), race_id, now_jst.isoformat(timespec="seconds"))
        if key in seen:
            continue
        tickets = []
        if not shadow_bets.empty:
            matching = shadow_bets[
                shadow_bets.race_id.astype(str).eq(race_id)
                & shadow_bets.bet_type.eq("trifecta")
            ]
            for _, ticket in matching.drop_duplicates("buy").iterrows():
                tickets.append({
                    "buy": str(ticket["buy"]),
                    "group": str(ticket["ticket_group"]),
                    "stake_yen": int(ticket["stake_yen"]),
                    "prob": float(ticket["prob"]) if pd.notna(ticket.get("prob")) else None,
                    "ev": float(ticket["ev"]) if pd.notna(ticket.get("ev")) else None,
                    "purchase_authorized": truth(ticket.get("purchase_authorized", False)),
                    "odds_sources": ticket.get("odds_sources") if isinstance(ticket.get("odds_sources"),str) else None,
                    "odds_captured_at_jst": ticket.get("odds_captured_at_jst") if isinstance(ticket.get("odds_captured_at_jst"),str) else None,
                })
        positions, core_winner = [], None
        if position_rows is not None and not position_rows.empty:
            race = position_rows[position_rows.race_id.astype(str).eq(race_id)]
            for position, column in enumerate(["p_win", "p_second", "p_third"], 1):
                if column in race:
                    values = pd.to_numeric(race[column], errors="coerce")
                    if values.notna().all() and values.map(math.isfinite).all() and values.ge(0).all() and values.sum() > 0:
                        positions.append({"position": position,
                            "probabilities": {str(int(car)): float(prob) for car, prob in zip(race.car_no, values / values.sum())}})
            if "p_core" in race and pd.to_numeric(race.p_core, errors="coerce").notna().any():
                core_winner = int(race.loc[pd.to_numeric(race.p_core, errors="coerce").idxmax(), "car_no"])
        exact_challenger = None
        if position_rows is not None and not position_rows.empty:
            race = position_rows[position_rows.race_id.astype(str).eq(race_id)]
            exact_columns = ["p_second_exact_challenger", "p_third_exact_challenger"]
            if all(c in race for c in exact_columns) and not race.empty:
                values = race[exact_columns].apply(pd.to_numeric, errors="coerce")
                if values.notna().all().all() and values.apply(lambda col: col.map(math.isfinite).all()).all():
                    exact_challenger = {"version": "cumulative_difference_v1", "snapshot_at": key[2],
                        "probabilities": {str(position): {str(int(car)): float(prob) for car, prob in zip(race.car_no, values[column])}
                                          for position, column in zip([2, 3], exact_columns)},
                        "auto_promotion": False}
        candidate_evidence = None
        challenger = None
        lower_challenger = None
        axis_challenger = None
        if candidate_rows is not None and not candidate_rows.empty:
            matching = candidate_rows[candidate_rows.race_id.astype(str).eq(race_id) & candidate_rows.bet_type.eq("trifecta")]
            from selection_research import probability_first, lower_position_coverage
            if {"buy", "prob", "ev", "main_formation", "hole_formation"}.issubset(matching.columns):
                challenger = {"version": "probability_first_v1", "snapshot_at": key[2],
                              "tickets": probability_first(matching, plan,
                                  sum(t["group"] == "本線" for t in tickets),
                                  sum(t["group"] == "穴" for t in tickets))}
                lower_challenger = {"version":"lower_coverage_v1","snapshot_at":key[2],
                                    "tickets":lower_position_coverage(matching,plan,tickets)}
                if plan.get("first_fixed") and position_rows is not None and not position_rows.empty:
                    from betting_logic import score_riders, select_race
                    race=position_rows[position_rows.race_id.astype(str).eq(race_id)]
                    quotes=matching[['buy','bet_type','odds_used']].drop_duplicates('buy')
                    spread,_=select_race(score_riders(race,quotes),quotes,
                                        plan.get('main_ev',1.10),plan.get('hole_ev',1.25),force_no_fixed=True)
                    ranked=spread.sort_values(['ev','prob','buy'],ascending=[False,False,True])
                    main=ranked[ranked.main_formation & ranked.ev.ge(plan.get('main_ev',1.10))].head(sum(t['group']=='本線' for t in tickets))
                    holes=ranked[ranked.hole_formation & ranked.ev.ge(plan.get('hole_ev',1.25)) & ~ranked.buy.isin(main.buy)].head(sum(t['group']=='穴' for t in tickets))
                    axis_challenger={'version':'axis_spread_v1','snapshot_at':key[2],
                                     'tickets':[{'buy':t.buy,'group':group,'prob':float(t.prob),'ev':float(t.ev),'purchase_authorized':False}
                                                for group,part in [('本線',main),('穴',holes)] for t in part.itertuples()]}
            def finite(value):
                number = pd.to_numeric(value, errors="coerce")
                return float(number) if pd.notna(number) and math.isfinite(float(number)) else None
            candidate_evidence = {"schema_version": 2,
                "first_fixed": bool(plan.get("first_fixed", False)), "fixed_car": plan.get("fixed_car"),
                "formation": plan.get("formation", {}), "main_ev": finite(plan.get("main_ev", 1.10)),
                "hole_ev": finite(plan.get("hole_ev", 1.25)),
                "reason_buys": {"ev_compression_miss": [], "ev_filter_skip": [],
                                "odds_missing": [], "formation_omission": []}}
            for _, candidate in matching.drop_duplicates("buy").iterrows():
                ev = finite(candidate.get("ev"))
                main = truth(candidate.get("main_formation", False))
                hole = truth(candidate.get("hole_formation", False))
                if ev is None:
                    reason = "odds_missing"
                elif (main and ev >= candidate_evidence["main_ev"]) or (hole and ev >= candidate_evidence["hole_ev"]):
                    reason = "ev_compression_miss"
                else:
                    reason = "ev_filter_skip" if main or hole else "formation_omission"
                candidate_evidence["reason_buys"][reason].append(str(candidate["buy"]))

        rows.append({
            "strategy_version": key[0], "race_id": race_id, "snapshot_at": key[2],
            "date": plan.get("date", now_jst.strftime("%Y-%m-%d")), "close_at": float(close_at),
            "venue": plan["venue"], "race_no": plan["race_no"],
            "skip_reason": plan.get("skip_reason"), "tickets": tickets,
            "position_probabilities": positions, "core_winner": core_winner,
            "exact_position_challenger": exact_challenger,
            "candidate_evidence": candidate_evidence,
            "selection_challenger": challenger,
            "lower_challenger": lower_challenger,
            "axis_challenger": axis_challenger,
            "high_payout_department": plan.get('high_payout_department'),
        })
    if rows:
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(rows, group=None, authorized_only=False):
    eligible = len(rows)
    invested = hits = skipped = flat_stake = flat_return = stake = returned = 0
    for row in rows:
        tickets = [t for t in row["tickets"] if (group is None or t["group"] == group)
                   and (not authorized_only or t["purchase_authorized"])]
        if not tickets:
            skipped += 1
            continue
        invested += 1
        winning = [t for t in tickets if t["buy"] == row["actual_trifecta"]]
        hits += bool(winning)
        flat_stake += len(tickets) * 100
        flat_return += len(winning) * row["payout_per_100yen"]
        stake += sum(t["stake_yen"] for t in tickets)
        returned += sum(t["stake_yen"] / 100 * row["payout_per_100yen"] for t in winning)
    return {
        "eligible_settled_races": eligible, "bet_races": invested,
        "skip_races": skipped, "skip_rate": skipped / eligible if eligible else None,
        "hits": hits, "hit_rate": hits / invested if invested else None,
        "flat_100yen": {"stake_yen": flat_stake, "return_yen": flat_return,
                        "return_rate": flat_return / flat_stake if flat_stake else None},
        "allocated": {"stake_yen": stake, "return_yen": returned,
                      "profit_yen": returned - stake,
                      "return_rate": returned / stake if stake else None},
    }


def calibration_audit(rows):
    positions = []
    for position in [1, 2, 3]:
        samples = []
        for row in rows:
            matches = [p for p in row.get("position_probabilities", []) if p["position"] == position]
            if not matches:
                continue
            probs = matches[0]["probabilities"]
            actual = row["actual_trifecta"].split("-")[position - 1]
            best = max(probs, key=probs.get)
            samples.append((probs[best], int(best == actual), sum((prob - int(car == actual))**2 for car, prob in probs.items())))
        positions.append({"position": position, "races": len(samples),
            "mean_top_probability": sum(x[0] for x in samples)/len(samples) if samples else None,
            "actual_top_hit_rate": sum(x[1] for x in samples)/len(samples) if samples else None,
            "multiclass_brier": sum(x[2] for x in samples)/len(samples) if samples else None})
    comparison = [r for r in rows if r.get("core_winner") is not None and r.get("position_probabilities")]
    core_hits = sum(str(r["core_winner"]) == r["actual_trifecta"].split("-")[0] for r in comparison)
    final_hits = sum(max(r["position_probabilities"][0]["probabilities"], key=r["position_probabilities"][0]["probabilities"].get)
                     == r["actual_trifecta"].split("-")[0] for r in comparison)
    return {"positions": positions, "same_race_core_vs_final": {"races": len(comparison),
            "core_hits": core_hits, "final_hits": final_hits, "net_final_hits": final_hits-core_hits},
            "automatically_recalibrated": False}


def build_ticket_return_department(output_dir=OUTPUT_DIR):
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    latest = {}
    snapshot_path = folder / "ticket_return_snapshots.jsonl"
    if snapshot_path.exists():
        for line in snapshot_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if datetime.fromisoformat(row["snapshot_at"]).timestamp() >= row["close_at"]:
                continue
            key = (row["strategy_version"], row["race_id"])
            if key not in latest or row["snapshot_at"] > latest[key]["snapshot_at"]:
                latest[key] = row
    settled_path = folder / "ticket_return_settled.json"
    settled = {(r["strategy_version"], r["race_id"]): r
               for r in read_json(settled_path, [])}
    results = read_json(output_dir / "latest_results.json", [])
    # Official payout odds are actual_trifecta_odds, not prediction-time odds.
    for result in results:
        if not truth(result.get("official_result_available", False)):
            continue
        buy = str(result.get("actual_trifecta") or "")
        try:
            odds = float(result["actual_trifecta_odds"])
        except (TypeError, ValueError, KeyError):
            continue
        if not buy or not 0 < odds < float("inf"):
            continue
        for key, snapshot in latest.items():
            if key[1] != str(result["race_id"]):
                continue
            settled[key] = {**snapshot, "actual_trifecta": buy,
                            "payout_per_100yen": round(odds * 100)}
    settled_path.write_text(json.dumps(list(settled.values()), ensure_ascii=False,
                                       indent=2, allow_nan=False), encoding="utf-8")
    rows = [r for r in settled.values() if r["strategy_version"] == STRATEGY_VERSION]
    ordered=sorted(rows,key=lambda r:(r.get("date",""),r.get("close_at",0),r.get("race_id","")))
    today=datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    windows={"today":[r for r in ordered if r.get("date")==today],"all":ordered,"last50":ordered[-50:],"last100":ordered[-100:]}
    continuous={key:{"settled_races":len(values),"total":summarize(values),"main":summarize(values,"本線"),"hole":summarize(values,"穴")} for key,values in windows.items()}
    report = {
        "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "role": "ticket_return_department", "strategy_version": STRATEGY_VERSION,
        "status": "collecting" if not rows else "prospective_audit",
        "frozen_races": sum(k[0] == STRATEGY_VERSION for k in latest),
        "settled_races": len(rows), "target_return_rate": 1.20,
        "target_validated": False, "production_auto_promotion": False,
        "scope": "latest eligible pre-close frozen portfolio; full main and hole tickets",
        "today_date": today, "continuous": continuous,
        "shadow": {"total": summarize(rows), "main": summarize(rows, "本線"),
                   "hole": summarize(rows, "穴")},
        "authorized_purchase": summarize(rows, authorized_only=True),
        "calibration_audit": calibration_audit(rows),
        "monthly": {month: summarize([r for r in rows if r["date"][:7] == month])
                    for month in sorted({r["date"][:7] for r in rows})},
        "limitations": ["旧戦略と混合しない。部署追加前の予想を事後生成しない。",
                         "回収率120%の単純超過だけで購入許可しない。外部検証が必要。",
                         "未確定・公式払戻未取得は成績から除外。"],
    }
    (folder / "ticket_return_department.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 買い目・回収率検証部", "", f"更新: {report['updated_at_jst']}",
             f"戦略: {STRATEGY_VERSION}",
             f"保存済み {report['frozen_races']}レース / 公式払戻で検証済み {len(rows)}レース", "",
             "発走前に固定した本線・穴の全点数を対象とする検証用成績。実購入とは分離。", "",
             "|対象|買い目がある確定レース|的中レース|買い目の的中率|回収率・各点100円の試算|回収率・配分額の試算|見送り率|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "未集計" if value is None else f"{value * 100:.1f}%"
    for label, values in [("合計", report["shadow"]["total"]),
                          ("本線", report["shadow"]["main"]), ("穴", report["shadow"]["hole"])]:
        lines.append(f"|{label}|{values['bet_races']}|{values['hits']}|{percent(values['hit_rate'])}|"
                     f"{percent(values['flat_100yen']['return_rate'])}|"
                     f"{percent(values['allocated']['return_rate'])}|{percent(values['skip_rate'])}|")
    lines += ["", "回収率120%は未検証。部署には購入・モデル変更の許可権限はない。"]
    (folder / "ticket_return_department.md").write_text("\n".join(lines), encoding="utf-8")
    table_rows = [line.strip("|").split("|") for line in lines if line.startswith("|")]
    table = "<table>" + "".join(
        "<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>"
        for index, cells in enumerate(table_rows) if index != 1
        for tag in ["th" if index == 0 else "td"]
    ) + "</table>"
    simple_rows = []
    for label, values in [("本線＋穴", report["shadow"]["total"]), ("本線", report["shadow"]["main"]), ("穴", report["shadow"]["hole"])]:
        simple_rows.append(f'<tr><td>{label}</td><td>{values["hits"]} / {values["bet_races"]}レース</td><td>{percent(values["hit_rate"])}</td><td>{percent(values["flat_100yen"]["return_rate"])}</td></tr>')
    simple_table = '<table><tr><th>対象</th><th>的中 / 検証レース</th><th>買い目の的中率</th><th>回収率・各点100円</th></tr>' + ''.join(simple_rows) + '</table>'
    page = ('<!doctype html><html lang="ja"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>買い目・回収率検証部</title><style>'
            'body{font-family:system-ui;background:#f4f7fb;color:#172b45;padding:20px}'
            'main{max-width:900px;margin:auto;background:white;padding:24px;border-radius:16px}'
            'table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd}'
            '.scroll{overflow:auto}p{line-height:1.7}</style><main>'
            '<a href="../index.html">レース一覧へ戻る</a> ／ <a href="../performance.html">成績と数字の見方</a><h1>買い目・回収率検証部</h1>'
            f'<p>更新 {html.escape(report["updated_at_jst"])}<br>'
            f'保存済み {report["frozen_races"]}レース ／ 公式払戻で検証済み {len(rows)}レース</p>'
            '<p>本線・穴の全買い目を対象とした検証用成績です。実購入とは分けて集計しています。'
            '100円均等と予想時の配分額で比較します。見送り率は対象となった確定レースに対する割合です。</p>'
            '<h2>3連単の買い目成績（各点100円の試算）</h2>'
            f'<div class="scroll">{simple_table}</div>'
            '<p>的中率は「1点でも当たったレース ÷ 買い目がある確定レース」。回収率は「全点の払戻 ÷ 全点の購入額」です。実購入の成績ではありません。</p>'
            '<details><summary>配分額・見送り率も見る</summary>'
            f'<div class="scroll">{table}</div></details>'
            '<p><a href="prediction_quality.html">外れ方とAI確率の検証を見る</a></p>'
            '<p>回収率120%は未検証です。部署追加後の発走前予想から蓄積します。'
            '未確定・払戻未取得は成績に含めません。</p></main></html>')
    (folder / "ticket_return_department.html").write_text(page, encoding="utf-8")
    from prediction_quality import build_prediction_quality
    from high_payout_department import build_high_payout_department
    build_high_payout_department(list(settled.values()), output_dir)
    build_prediction_quality(list(settled.values()), output_dir)
    from selection_research import build_selection_research
    build_selection_research(list(settled.values()), output_dir)
    from validation_coverage import build_validation_coverage
    build_validation_coverage(output_dir)
    from company_operations import build_company_operations
    build_company_operations(output_dir)
    from public_performance import build_performance_page
    build_performance_page(output_dir)
    return report
