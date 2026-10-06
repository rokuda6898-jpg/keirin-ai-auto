"""Prospective department consensus experiment; no production or purchase authority."""
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from common import OUTPUT_DIR
from betting_logic import STRATEGY_VERSION

VERSION = "strategist_consensus_v1"
LABELS = {"data_department": "データ部", "pace_department": "展開部",
          "line_department": "ライン部", "risk_department": "リスク部"}


def load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def preclose(row, now=None):
    try:
        parsed = datetime.fromisoformat(row["snapshot_at"])
        if parsed.tzinfo is None:
            return False
        stamp = parsed.timestamp()
        close = float(row["close_at"])
        return math.isfinite(close) and stamp < close and (now is None or stamp <= now.timestamp() < close)
    except (KeyError, TypeError, ValueError):
        return False


def consensus(proposals):
    """Equal department votes, conservative EV and consistent quoted prices only."""
    pool = defaultdict(dict)
    for proposal in proposals:
        dept = proposal["department"]
        for ticket in proposal.get("tickets", []):
            buy = str(ticket.get("buy", ""))
            parts = buy.split("-")
            if len(parts) != 3 or len(set(parts)) != 3 or not all(p in "123456789" and len(p) == 1 for p in parts):
                continue
            try:
                ev, prob = float(ticket["ev"]), float(ticket["prob"])
                odds = ev / prob
                if not (math.isfinite(ev) and 0 < prob <= 1 and 1 < odds < 9999.9):
                    continue
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            pool[buy][dept] = {**ticket, "ev": ev, "quoted_odds": odds}
    candidates = []
    for buy, views in pool.items():
        if len(views) < 2:
            continue
        prices = [v["quoted_odds"] for v in views.values()]
        if max(prices) - min(prices) > max(.1, min(prices) * .05):
            continue
        group = "穴" if any(v.get("group") == "穴" for v in views.values()) else "本線"
        ev = min(v["ev"] for v in views.values())
        if ev < (1.25 if group == "穴" else 1.10):
            continue
        candidates.append({"buy": buy, "group": group, "ev": ev,
                           "votes": len(views), "departments": sorted(views)})
    # A small, fixed experimental portfolio. Never forces a minimum ticket count.
    return sorted(candidates, key=lambda t: (-t["votes"], -t["ev"], t["buy"]))[:12]


def build_strategist_validation(output_dir=OUTPUT_DIR, now=None):
    now = now or datetime.now(ZoneInfo("Asia/Tokyo"))
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    latest = load(folder / "annual_department_predictions.json", {}).get("proposals", [])
    grouped = defaultdict(dict)
    for proposal in latest:
        if proposal.get("department") in LABELS and preclose(proposal, now):
            grouped[str(proposal["race_id"])][proposal["department"]] = proposal
    ledger = folder / "annual_strategist_ledger.jsonl"
    saved = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line] if ledger.exists() else []
    seen = {(r["race_id"], r["source_snapshot_at"], r.get("version")) for r in saved if "source_snapshot_at" in r}
    fresh = []
    for rid, views in grouped.items():
        if len(views) < 2:
            continue
        proposals = list(views.values())
        stamp = max(p["snapshot_at"] for p in proposals)
        close = min(float(p["close_at"]) for p in proposals)
        # Reject mixed or stale department batches rather than creating retrospective evidence.
        if len({p["snapshot_at"] for p in proposals}) != 1 or not preclose({"snapshot_at": stamp, "close_at": close}, now):
            continue
        first = proposals[0]
        row = {"version": VERSION, "race_id": rid, "snapshot_at": now.isoformat(timespec="seconds"),
               "source_snapshot_at": stamp, "close_at": close,
               "venue": first.get("venue", ""), "race_no": first.get("race_no", 0),
               "views": [{"department": p["department"], "winner_car": p["winner_car"],
                          "tickets": [t["buy"] for t in p.get("tickets", [])],
                          "minimum_reference_races": min((v.get("effective_races", 0) for v in p.get("rider_reference", {}).values()), default=0)} for p in proposals],
               "winner_votes": dict(Counter(str(p["winner_car"]) for p in proposals)),
               "tickets": consensus(proposals), "purchase_authorized": False}
        key = (rid, stamp, VERSION)
        if key not in seen:
            fresh.append(row); seen.add(key)
    if fresh:
        with ledger.open("a", encoding="utf-8") as handle:
            for row in fresh:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    selected = {}
    for row in saved + fresh:
        if row.get("version") == VERSION and "source_snapshot_at" in row and preclose(row):
            rid = row["race_id"]
            if rid not in selected or row["snapshot_at"] > selected[rid]["snapshot_at"]:
                selected[rid] = row
    results = {str(r["race_id"]): r for r in load(output_dir / "latest_results.json", [])
               if str(r.get("official_result_available", "")).lower() in {"true", "1"}}
    settled = {r["race_id"]: r for r in load(folder / "annual_strategist_settled.json", []) if r.get("version") == VERSION and "source_snapshot_at" in r and preclose(r)}
    for rid, row in selected.items():
        result = results.get(rid)
        if not result or now.timestamp() < row["close_at"]:
            continue
        try:
            odds = float(result["actual_trifecta_odds"])
            actual = str(result["actual_trifecta"])
            parts = actual.split("-")
            if not 0 < odds < float("inf") or len(parts) != 3 or len(set(parts)) != 3 or not all(len(p) == 1 and p in "123456789" for p in parts):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        settled[rid] = {**row, "actual_trifecta": actual, "payout_per_100yen": round(100 * odds)}
    production = {r["race_id"]: r for r in load(folder / "ticket_return_settled.json", [])
                  if r.get("strategy_version") == STRATEGY_VERSION and preclose(r)}
    matched = [(row, production[rid]) for rid, row in settled.items() if rid in production
               and production[rid].get("actual_trifecta") == row["actual_trifecta"]
               and production[rid].get("payout_per_100yen") == row["payout_per_100yen"]]

    def performance(rows):
        bet = [r for r in rows if r.get("tickets")]
        hits = sum(any(t["buy"] == r["actual_trifecta"] for t in r["tickets"]) for r in bet)
        stake = sum(100 * len(r["tickets"]) for r in bet)
        returned = sum(r["payout_per_100yen"] for r in bet if any(t["buy"] == r["actual_trifecta"] for t in r["tickets"]))
        return {"settled_races": len(rows), "bet_races": len(bet), "hits": hits,
                "hit_rate": hits / len(bet) if bet else None, "stake_yen": stake,
                "return_yen": returned, "return_rate": returned / stake if stake else None}

    reasons = load(folder / "prediction_quality.json", {}).get("reason_counts", {})
    report = {"updated_at_jst": now.isoformat(timespec="seconds"), "version": VERSION,
              "status": "collecting", "production_effect": "advisory_and_shadow_only",
              "auto_promotion": False, "minimum_paired_races_for_review": 300,
              "latest_plans": list(selected.values()), "settled": performance(list(settled.values())),
              "paired_races": len(matched), "paired_strategist": performance([a for a, _ in matched]),
              "paired_production": performance([b for _, b in matched]), "production_miss_reasons": reasons}
    for name, data in [("annual_strategist_report.json", report), ("annual_strategist_settled.json", list(settled.values()))]:
        (folder / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    render_report(report, folder)
    return report


def render_report(report, folder):
    escape = lambda value: html.escape(str(value))
    pct = lambda value: "未集計" if value is None else f"{value * 100:.1f}%"
    labels = {"ev_filter_skip": "期待値条件で除外", "fixed_axis_miss": "1着固定の軸違い",
              "ev_compression_miss": "期待値順の圧縮で取りこぼし", "formation_miss": "フォーメーション外",
              "odds_missing": "オッズ未取得"}
    body = '<h1>軍師の分析と比較検証</h1><p>軍師案は検証用です。現行買い目の自動変更・購入は行いません。</p>'
    body += '<h2>現行案との同一レース比較</h2><p>比較できる確定レース：'+str(report['paired_races'])+'件。両方が締切前保存・同じ公式払戻のレースだけを比較します。各点100円、点数の違いも回収率に反映します。</p>'
    body += '<table><tr><th>案</th><th>的中／購入対象</th><th>回収率</th></tr>'
    for key, name in [('paired_strategist','軍師案'),('paired_production','現行案')]:
        row = report[key]
        body += f'<tr><td>{name}</td><td>{row["hits"]}／{row["bet_races"]}R</td><td>{pct(row["return_rate"])}</td></tr>'
    body += '</table><p>改善の判断は最低300件の同一レース比較と期間を分けた検証後。多数決は正解や独立した証拠を意味しません。</p><h2>現行買い目の外れ・除外原因</h2><ul>'
    for reason, count in report['production_miss_reasons'].items():
        body += f'<li>{escape(labels.get(reason,reason))}：{count}件</li>'
    body += '</ul><h2>部署の見立てと軍師案</h2><p>展開部・ライン部・データ部・リスク部の見立てを比較します。未観測の展開を事実として扱いません。最低2部署の支持、価格の整合、各部署のうち最も低い期待値で選定し、最大12点まで。期待値は暫定推定です。</p>'
    for row in sorted(report['latest_plans'],key=lambda r:r['close_at']):
        body += f'<section><h3>{escape(row["venue"])} {escape(row["race_no"])}R</h3><p>締切前保存：{escape(row["snapshot_at"])} ／ '+('締切済み・結果は比較欄で確認' if row['close_at'] <= datetime.fromisoformat(report['updated_at_jst']).timestamp() else '締切前')+'</p>'
        for view in row['views']:
            body += f'<p>{LABELS[view["department"]]}：1着 {escape(view["winner_car"])}番・{len(view["tickets"])}点 ／ 選手の最小参考量 {view["minimum_reference_races"]:.1f}走相当</p>'
        body += '<p>軍師案：'+(' ／ '.join(f'{escape(t["buy"])}（{t["votes"]}部署支持）' for t in row['tickets']) or '条件を満たす候補なし')+'</p></section>'
    body += f'<p>更新：{escape(report["updated_at_jst"])}</p><a href="annual_department_report.html">各部署へ</a> ／ <a href="../index.html">今日の予想へ</a>'
    page = '<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>軍師の分析と比較検証</title><link rel="stylesheet" href="../site-ui.css"></head><body><main>'+body+'</main></body></html>'
    (folder / 'annual_strategist_report.html').write_text(page,encoding='utf-8')


if __name__ == "__main__":
    build_strategist_validation()
