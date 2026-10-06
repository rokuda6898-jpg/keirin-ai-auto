"""Diagnose only frozen pre-close forecasts; never change probabilities or purchase gates."""
import html
import json
import math
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from common import OUTPUT_DIR
from betting_logic import STRATEGY_VERSION

LABELS = {"hit": "3連単的中", "skip": "買い目なし・見送り", "fixed_axis_miss": "1着固定の軸が負けた",
          "first_not_covered": "1着を買い目でカバーできなかった", "second_not_covered": "2着を買い目でカバーできなかった",
          "third_not_covered": "3着を買い目でカバーできなかった", "ev_compression_miss": "期待値順で絞る際に正解を除外",
          "ev_filter_skip": "正解が期待値基準を満たさなかった", "odds_missing": "正解候補のオッズ未取得",
          "formation_omission": "正解を候補に含められなかった", "detail_unavailable": "候補の詳細未保存"}


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def valid_row(row):
    try:
        stamp = datetime.fromisoformat(row["snapshot_at"])
        close = finite(row["close_at"])
        if close is None or stamp.tzinfo is None or stamp.timestamp() >= close:
            return False
        result = row["actual_trifecta"].split("-")
        return len(result) == 3 and len(set(result)) == 3 and all(x.isdigit() and 1 <= int(x) <= 9 for x in result)
    except (KeyError, ValueError, TypeError):
        return False


def miss_reason(row):
    actual = row["actual_trifecta"]
    tickets = row.get("tickets", [])
    if any(t["buy"] == actual for t in tickets):
        return "hit"
    if not tickets:
        return "skip"
    evidence = row.get("candidate_evidence")
    if evidence:
        if evidence.get("first_fixed") and str(evidence.get("fixed_car")) != actual.split("-")[0]:
            return "fixed_axis_miss"
        if evidence.get("schema_version") == 2:
            return next((reason for reason, buys in evidence.get("reason_buys", {}).items() if actual in buys), "formation_omission")
        candidate = next((t for t in evidence.get("candidates", []) if t["buy"] == actual), None)
        if candidate:
            ev = finite(candidate.get("ev"))
            if ev is None:
                return "odds_missing"
            main = candidate.get("main_formation") and ev >= (finite(evidence.get("main_ev")) or 1.10)
            hole = candidate.get("hole_formation") and ev >= (finite(evidence.get("hole_ev")) or 1.25)
            if main or hole:
                return "ev_compression_miss"
            if candidate.get("main_formation") or candidate.get("hole_formation"):
                return "ev_filter_skip"
            return "formation_omission"
        return "formation_omission"
    # Legacy ticket coverage is observable, but the reason a candidate was dropped is unknown.
    parts = [t["buy"].split("-") for t in tickets if len(t["buy"].split("-")) == 3]
    result = actual.split("-")
    if parts and not any(p[0] == result[0] for p in parts):
        return "first_not_covered"
    if parts and not any(p[:2] == result[:2] for p in parts):
        return "second_not_covered"
    if parts:
        return "third_not_covered"
    return "detail_unavailable"


def reliability(samples, edges=None):
    """Sample is (race ID, probability, binary outcome); include every observed competitor."""
    edges = edges or [i / 10 for i in range(11)]
    bins = []
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        group = [s for s in samples if lower <= s[1] and (s[1] < upper or index == len(edges) - 2 and s[1] == upper)]
        n = len(group)
        predicted = sum(s[1] for s in group) / n if n else None
        observed = sum(s[2] for s in group) / n if n else None
        bins.append({"lower": lower, "upper": upper, "observations": n,
                     "races": len({s[0] for s in group}), "mean_predicted": predicted, "actual_rate": observed,
                     "gap": observed - predicted if n else None})
    n = len(samples)
    return {"observations": n, "races": len({s[0] for s in samples}), "bins": bins,
            "expected_calibration_error": sum(b["observations"] * abs(b["gap"]) for b in bins if b["observations"]) / n if n else None,
            "sample_status": "参考集計・100レース未満" if len({s[0] for s in samples}) < 100 else "検証継続中"}


def ev_audit(rows):
    bands = []
    for lower, upper in [(0, 1.1), (1.1, 1.25), (1.25, 1.5), (1.5, 2), (2, 3), (3, None)]:
        samples = []
        for row in rows:
            payout = finite(row.get("payout_per_100yen"))
            if payout is None or payout <= 0:
                continue
            for ticket in row.get("tickets", []):
                ev = finite(ticket.get("ev"))
                if ev is not None and lower <= ev and (upper is None or ev < upper):
                    samples.append((row["race_id"], ev, payout / 100 if ticket["buy"] == row["actual_trifecta"] else 0))
        n = len(samples)
        bands.append({"lower": lower, "upper": upper, "observations": n, "races": len({s[0] for s in samples}),
                      "mean_predicted_ev": sum(s[1] for s in samples) / n if n else None,
                      "actual_flat_return_rate": sum(s[2] for s in samples) / n if n else None})
    return bands


def fixed_axis_audit(rows):
    samples = []
    missing = 0
    for row in rows:
        evidence = row.get("candidate_evidence") or {}
        if not evidence.get("first_fixed"):
            continue
        car = str(evidence.get("fixed_car"))
        first = next((p for p in row.get("position_probabilities", []) if p.get("position") == 1), {})
        probabilities = first.get("probabilities", {})
        probability = finite(probabilities.get(car))
        if probability is None or not 0 <= probability <= 1:
            missing += 1
            continue
        samples.append((row["race_id"], probability, int(car == row["actual_trifecta"].split("-")[0])))
    count = len(samples)
    return {"races_with_probability": count, "missing_probability_races": missing,
            "axis_wins": sum(s[2] for s in samples),
            "mean_predicted": sum(s[1] for s in samples) / count if count else None,
            "actual_win_rate": sum(s[2] for s in samples) / count if count else None,
            "automatic_change": False}


def quality_audit(rows, strategy_version=STRATEGY_VERSION):
    latest = {}
    rejected = 0
    for row in rows:
        if row.get("strategy_version") != strategy_version or not valid_row(row):
            rejected += 1
            continue
        key = (row.get("strategy_version"), row["race_id"])
        if key not in latest or row["snapshot_at"] > latest[key]["snapshot_at"]:
            latest[key] = row
    rows = list(latest.values())
    cases = [{"race_id": r["race_id"], "venue": r.get("venue", ""), "race_no": r.get("race_no", ""),
              "snapshot_at": r["snapshot_at"], "actual_trifecta": r["actual_trifecta"],
              "reason": miss_reason(r), "candidate_detail_saved": bool(r.get("candidate_evidence"))} for r in rows]
    categories = Counter(c["reason"] for c in cases)
    positions = []
    for position in [1, 2, 3]:
        samples, briers, baselines = [], [], []
        for row in rows:
            match = next((p for p in row.get("position_probabilities", []) if p.get("position") == position), None)
            if not match:
                continue
            probs = {str(car): finite(p) for car, p in match.get("probabilities", {}).items()}
            actual = row["actual_trifecta"].split("-")[position - 1]
            if not probs or actual not in probs or any(p is None or not 0 <= p <= 1 for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-6:
                continue
            samples.extend((row["race_id"], p, int(car == actual)) for car, p in probs.items())
            briers.append(sum((p - int(car == actual)) ** 2 for car, p in probs.items()))
            baselines.append(1 - 1 / len(probs))
        positions.append({"position": position, **reliability(samples),
                          "multiclass_brier": sum(briers) / len(briers) if briers else None,
                          "uniform_brier": sum(baselines) / len(baselines) if baselines else None})
    ticket_samples = [(r["race_id"], p, int(t["buy"] == r["actual_trifecta"])) for r in rows for t in r.get("tickets", [])
                      if (p := finite(t.get("prob"))) is not None and 0 <= p <= 1]
    return {"updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "strategy_version": strategy_version, "races": len(rows), "rejected_rows": rejected,
            "candidate_detail_races": sum(c["candidate_detail_saved"] for c in cases),
            "reason_counts": dict(categories), "cases": cases, "positions": positions,
            "fixed_axis_audit": fixed_axis_audit(rows),
            "selected_ticket_calibration": reliability(ticket_samples, [0, .01, .02, .05, .1, .2, .4, .6, .8, 1]), "ev_bands": ev_audit(rows), "automatically_recalibrated": False,
            "limitations": ["公式結果と払戻が確認できた締切前の保存記録のみ。未保存候補を事後生成しない。",
                            "分類は保存された買い目・候補の比較であり、選手が負けた原因の断定ではない。",
                            "同一レースの選手・買い目は独立した標本ではない。件数とレース数を併記する。",
                            "100レース以上でも確率の妥当性や回収率120%の達成を自動承認しない。"]}


def build_prediction_quality(rows, output_dir=OUTPUT_DIR):
    rows = list(rows)
    report = quality_audit(rows)
    report["previous_strategies"] = [quality_audit(rows, version) for version in sorted({r.get("strategy_version") for r in rows if r.get("strategy_version") and r.get("strategy_version") != STRATEGY_VERSION})]
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prediction_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    percent = lambda value: "未集計" if value is None else f"{100 * value:.1f}%"
    axis = report['fixed_axis_audit']
    axis_html = f"<h2>1着固定の軸は実際に勝っているか</h2><p>確率を保存した固定軸 {axis['races_with_probability']}レース ／ 軸の1着 {axis['axis_wins']}件。AIの平均推定勝率 {percent(axis['mean_predicted'])} ／ 実際の軸の勝率 {percent(axis['actual_win_rate'])}。</p><p>軸が勝った割合は3連単の的中率と異なります。少数の参考集計のため、閾値の最適化や確率補正には使わず、固定解除案との締切前比較を蓄積します。確率未保存は {axis['missing_probability_races']}レース。</p>"
    reasons = ''.join(f'<tr><td>{LABELS[key]}</td><td>{count}レース</td></tr>' for key, count in report["reason_counts"].items())
    examples = ''.join(f'<tr><td>{html.escape(str(c["venue"]))} {html.escape(str(c["race_no"]))}R</td><td>{c["actual_trifecta"]}</td><td>{LABELS[c["reason"]]}</td></tr>' for c in report["cases"][-20:])
    tables = []
    for item in report["positions"] + [{"position": "3連単・選択済み買い目", **report["selected_ticket_calibration"]}]:
        label = f'{item["position"]}着の確率' if isinstance(item["position"], int) else item["position"]
        cells = ''.join(f'<tr><td>{int(b["lower"] * 100)}〜{int(b["upper"] * 100)}%</td><td>{percent(b["mean_predicted"])}</td><td>{percent(b["actual_rate"])}</td><td>{b["observations"]}件 / {b["races"]}レース</td></tr>' for b in item["bins"] if b["observations"])
        tables.append(f'<h3>{label}</h3><p>{item["races"]}レース ／ {item["observations"]}件・{item["sample_status"]}</p><div class="scroll"><table><tr><th>AIの確率帯</th><th>平均の推定確率</th><th>実際に当たった割合</th><th>検証件数 / レース数</th></tr>{cells or "<tr><td colspan=4>保存済みの確率データがありません</td></tr>"}</table></div>')
    ev_cells = "".join(f'<tr><td>{b["lower"]:.2f}〜{b["upper"] if b["upper"] is not None else "以上"}</td><td>{b["mean_predicted_ev"]:.2f}</td><td>{percent(b["actual_flat_return_rate"])}</td><td>{b["observations"]}点 / {b["races"]}レース</td></tr>' for b in report["ev_bands"] if b["observations"])
    previous = "".join(f'<p>変更前：{html.escape(r["strategy_version"])}・{r["races"]}レース（最新版と合算しません）</p>' for r in report["previous_strategies"])
    page = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>外れ方とAI確率の検証</title><style>body{{font-family:system-ui;background:#f4f7fb;color:#172b45;padding:20px}}main{{max-width:960px;margin:auto;background:white;padding:24px;border-radius:16px}}p{{line-height:1.8}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}a{{color:#0965c7}}</style><main><a href="../performance.html">成績と数字の見方</a> ／ <a href="../index.html">今日の予想</a><h1>外れ方とAI確率の検証</h1><p>対象 {report['races']}レース ／ 候補の詳細まで保存済み {report['candidate_detail_races']}レース<br>更新 {report['updated_at_jst']}</p><p>締切前に保存した予想と公式結果・払戻を使う検証です。まだ少数のため、改善効果や回収率120%の達成を示すものではありません。</p><p>最新版の1着固定は、推定勝率60%以上かつ1位と2位の差10ポイント以上が条件です（暫定）。変更前の記録は別集計で保持します。</p>{previous}{axis_html}<h2>買い目のどこで取りこぼしたか</h2><table><tr><th>分類</th><th>件数</th></tr>{reasons or '<tr><td colspan=2>未集計</td></tr>'}</table><p>候補が保存されている場合は、期待値の基準で除外したのか、期待値順で点数を絞る際に除外したのかを区別します。旧記録は買い目のカバー範囲まで確認し、候補を除外した理由は後付けで推測しません。</p><details><summary>直近のレース別内訳</summary><table><tr><th>レース</th><th>公式結果</th><th>分類</th></tr>{examples}</table></details><h2>AIの確率と実際の結果は合っているか</h2><p>例えば30〜40%と予測した選手について、推定確率の平均と実際にその着順になった割合を比較します。1・2・3着は全選手が対象。3連単は選択済み買い目が対象です。同じレース内の複数選手・買い目を含むため、件数とレース数を分けて表示します。</p>{''.join(tables)}<h2>期待値と実際の回収率</h2><p>締切前の期待値と、各点100円の公式払戻を比較します。少数レースでは参考集計です。</p><table><tr><th>期待値帯</th><th>平均の推定期待値</th><th>実際の回収率</th><th>点数 / レース数</th></tr>{ev_cells or '<tr><td colspan=4>期待値付きの確定記録は未集計</td></tr>'}</table><p>確率・期待値の自動補正や購入許可の変更はしていません。検証結果を蓄積してから改善を判断します。1着確率の一致だけでは、3連単の期待値が正しいとは断定できません。</p></main></html>"""
    page=page.replace('<h1>', '<p><a href="selection_research.html">確率補正と買い目圧縮の比較検証</a></p><h1>',1)
    (folder / "prediction_quality.html").write_text(page, encoding="utf-8")
    return report
