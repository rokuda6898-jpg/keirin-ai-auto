"""Prospective, immutable full-portfolio audit; never grants purchase authority."""
import json
import html
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


def save_snapshots(plans, shadow_bets, now_jst, output_dir=OUTPUT_DIR):
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
                    "purchase_authorized": truth(ticket.get("purchase_authorized", False)),
                })
        rows.append({
            "strategy_version": key[0], "race_id": race_id, "snapshot_at": key[2],
            "date": plan.get("date", now_jst.strftime("%Y-%m-%d")), "close_at": float(close_at),
            "venue": plan["venue"], "race_no": plan["race_no"],
            "skip_reason": plan.get("skip_reason"), "tickets": tickets,
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
    report = {
        "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "role": "ticket_return_department", "strategy_version": STRATEGY_VERSION,
        "status": "collecting" if not rows else "prospective_audit",
        "frozen_races": sum(k[0] == STRATEGY_VERSION for k in latest),
        "settled_races": len(rows), "target_return_rate": 1.20,
        "target_validated": False, "production_auto_promotion": False,
        "scope": "latest eligible pre-close frozen portfolio; full main and hole tickets",
        "shadow": {"total": summarize(rows), "main": summarize(rows, "本線"),
                   "hole": summarize(rows, "穴")},
        "authorized_purchase": summarize(rows, authorized_only=True),
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
             "|対象|購入対象R|的中R|的中率|100円均等回収率|配分額回収率|見送り率|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "未算出" if value is None else f"{value * 100:.1f}%"
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
    page = ('<!doctype html><html lang="ja"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>買い目・回収率検証部</title><style>'
            'body{font-family:system-ui;background:#f4f7fb;color:#172b45;padding:20px}'
            'main{max-width:900px;margin:auto;background:white;padding:24px;border-radius:16px}'
            'table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd}'
            '.scroll{overflow:auto}p{line-height:1.7}</style><main>'
            '<a href="../index.html">レース一覧へ戻る</a><h1>買い目・回収率検証部</h1>'
            f'<p>更新 {html.escape(report["updated_at_jst"])}<br>'
            f'保存済み {report["frozen_races"]}レース ／ 公式払戻で検証済み {len(rows)}レース</p>'
            '<p>本線・穴の全買い目を対象とした検証用成績です。実購入とは分けて集計しています。'
            '100円均等と予想時の配分額で比較します。見送り率は対象となった確定レースに対する割合です。</p>'
            f'<div class="scroll">{table}</div>'
            '<p>回収率120%は未検証です。部署追加後の発走前予想から蓄積します。'
            '未確定・払戻未取得は成績に含めません。</p></main></html>')
    (folder / "ticket_return_department.html").write_text(page, encoding="utf-8")
    return report
