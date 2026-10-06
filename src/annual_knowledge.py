"""Rolling-year observed rider knowledge and independent department shadow forecasts."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import HISTORY_CSV, OUTPUT_DIR
from betting_logic import score_riders, select_race


def frame_read(path):
    try:
        return pd.read_csv(path, dtype={"race_id": str, "player_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def collect_observations(race_data, url, output_dir=OUTPUT_DIR):
    """Retain real official result factors/events; never infer missing events."""
    race = race_data.get("race", {})
    race_id = str(race.get("id", ""))
    if int(race.get("status", 0) or 0) < 3 or not race_data.get("results"):
        return
    from race_features import build_entry_rows
    date = datetime.strptime(race_id[-8:], "%Y%m%d").strftime("%Y-%m-%d")
    rows = build_entry_rows(race_data, date, "", race.get("number", 0), race_id, url, True)
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "rider_official_observations.csv"
    combined = pd.concat([frame_read(path), pd.DataFrame(rows)], ignore_index=True, sort=False)
    combined.drop_duplicates(["race_id", "player_id"], keep="last").to_csv(path, index=False)


def build_annual_profiles(asof=None, history_path=HISTORY_CSV, output_dir=OUTPUT_DIR):
    asof = pd.Timestamp(asof or datetime.now(ZoneInfo("Asia/Tokyo")).date()).normalize().tz_localize(None)
    start = asof - pd.DateOffset(years=1)
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    observations = folder / "rider_official_observations.csv"
    saved = folder / "annual_rider_knowledge.json"
    if not history_path.exists() and saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        previous["history_refresh_status"] = "preserved_until_full_history_is_mounted"
        return previous
    digest = hashlib.sha256(str(asof.date()).encode())
    for path in [history_path, observations, Path(__file__)]:
        if path.exists():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    fingerprint = digest.hexdigest()
    if saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == fingerprint:
            return previous
    history = frame_read(history_path)
    extra = frame_read(observations)
    if not history.empty and not extra.empty:
        # Missing fields from a supplemental observation cannot erase old values.
        history = history.set_index(["race_id", "player_id"])
        extra = extra.replace("", np.nan).set_index(["race_id", "player_id"])
        history = extra.combine_first(history).reset_index()
    elif not extra.empty:
        history = extra
    total_races = int(history.race_id.nunique()) if "race_id" in history else 0
    profiles = {}
    annual_races = 0
    coverage = {}
    if not history.empty and {"date", "player_id", "race_id", "finish_pos"}.issubset(history):
        dates = pd.to_datetime(history.date, errors="coerce")
        annual = history[dates.ge(start) & dates.lt(asof)].copy()
        annual = annual.drop_duplicates(["race_id", "player_id"], keep="last")
        annual["_date"] = pd.to_datetime(annual.date, errors="coerce")
        finish = pd.to_numeric(annual.finish_pos, errors="coerce")
        field = pd.to_numeric(annual.get("entries_number", pd.Series(9, index=annual.index)), errors="coerce").fillna(9)
        annual["_valid"] = finish.between(1, field)
        for position in [1, 2, 3]:
            annual[f"_p{position}"] = finish.eq(position).astype(int)
        annual_races = int(annual.race_id.nunique())
        coverage = {name: int(annual[name].notna().sum()) for name in annual.columns
                    if name.startswith("result_event_") or name == "result_factor"}
        for player_id, group in annual.groupby("player_id", sort=False):
            n = int(group._valid.sum())
            if not n:
                continue
            def stats(rows):
                count = int(rows._valid.sum())
                return {"races": count, "rates": [float(rows[f"_p{p}"].sum() / count)
                        if count else None for p in [1, 2, 3]]}
            profile = {**stats(group), "entries": len(group), "unplaced_rows": int((~group._valid).sum()),
                       "recent90": stats(group[group._date.ge(asof - pd.Timedelta(days=90))]),
                       "line_positions": {}, "tactics": {}, "events": {},
                       "style_counts": group.get("style", pd.Series(dtype=str)).dropna().astype(str).value_counts().to_dict()}
            if "line_position" in group:
                for position, rows in group.groupby("line_position"):
                    if pd.notna(position):
                        profile["line_positions"][str(int(float(position)))] = stats(rows)
            if "result_factor" in group:
                profile["tactics"] = group.result_factor.dropna().astype(str).loc[lambda s: s.ne("")].value_counts().to_dict()
                profile["winning_tactics"] = group.loc[group._p1.eq(1), "result_factor"].dropna().astype(str).loc[lambda s: s.ne("")].value_counts().to_dict()
            for column in group:
                if column.startswith("result_event_"):
                    observed = group[column].dropna().astype(str).str.lower()
                    profile["events"][column.removeprefix("result_event_")] = {
                        "observed": len(observed), "true": int(observed.isin(["true", "1", "1.0"]).sum())}
            profiles[str(player_id)] = profile
    report = {"updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
              "fingerprint": fingerprint, "asof_date": str(asof.date()), "window_start": str(start.date()),
              "window_end_exclusive": str(asof.date()), "total_archive_races": total_races,
              "annual_races": annual_races, "players": len(profiles), "profiles": profiles,
              "observation_coverage": coverage, "status": "ready" if profiles else "history_unavailable",
              "policy": "full archive retained; previous calendar year used for specialist proposals; same-day/future results excluded",
              "limitations": ["脚質と決まり手を区別。決まり手・行動の未取得分は推測しない。",
                               "部署予想は暫定の影予想。実戦検証前に本番モデルを置換しない。"]}
    saved.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def forecast_departments(pred, odds, report, now, output_dir=OUTPUT_DIR):
    """Each specialist creates its own 1/2/3-place proposal from the shared year."""
    proposals = []
    for race_id, race in pred.groupby("race_id", sort=False):
        close = pd.to_numeric(race.iloc[0].get("close_at"), errors="coerce")
        if pd.isna(close) or close <= now.timestamp() + 300:
            continue
        profiles = [report["profiles"].get(str(pid)) for pid in race.player_id]
        if not any(profiles):
            continue
        market = odds[odds.race_id.astype(str).eq(str(race_id))] if not odds.empty else odds
        for department in ["data_department", "pace_department", "line_department", "risk_department"]:
            values = []
            for (_, rider), profile in zip(race.iterrows(), profiles):
                baseline = np.repeat(1 / len(race), 3)
                if not profile:
                    values.append(baseline)
                    continue
                n = profile["races"]
                rates = (np.array(profile["rates"]) * n + baseline * 20) / (n + 20)
                context = None
                if department == "pace_department":
                    context = profile["recent90"]
                if department == "line_department":
                    position = pd.to_numeric(rider.get("line_position"), errors="coerce")
                    context = profile["line_positions"].get(str(int(position))) if pd.notna(position) else None
                if context and context["races"]:
                    weight = context["races"] / (context["races"] + 20)
                    rates = rates * (1 - weight) + np.array(context["rates"]) * weight
                if department == "risk_department":
                    reliability = n / (n + 40) * (1 - profile["unplaced_rows"] / profile["entries"])
                    rates = rates * reliability + baseline * (1 - reliability)
                values.append(rates)
            matrix = np.array(values)
            matrix /= matrix.sum(axis=0)
            temp = race.copy()
            temp["p_win"] = matrix[:, 0]
            riders = score_riders(temp, market)
            for index, name in enumerate(["first", "second", "third"]):
                riders[f"score_{name}"] = matrix[:, index] * 100
                riders[f"rank_{name}"] = riders[f"score_{name}"].rank(ascending=False, method="first").astype(int)
            riders["position_score_source"] = "annual_empirical_shadow"
            candidates, plan = select_race(riders, market)
            selected = candidates[candidates.is_selected]
            proposals.append({"department": department, "race_id": str(race_id),
                "venue": str(race.iloc[0].get("venue", "")), "race_no": int(race.iloc[0].get("race_no", 0)),
                "close_at": float(close), "snapshot_at": now.isoformat(timespec="seconds"),
                "winner_car": int(riders.loc[riders.rank_first.eq(1), "car_no"].iloc[0]),
                "rider_year_races": {str(int(car)): p["races"] if p else 0 for car, p in zip(race.car_no, profiles)},
                "tickets": [{"buy": str(t.buy), "group": str(t.ticket_group), "prob": float(t.prob), "ev": float(t.ev)}
                            for t in selected.itertuples()],
                "main_count": plan["main_count"], "hole_count": plan["hole_count"],
                "probability_status": "provisional_annual_shadow", "purchase_authorized": False})
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "annual_department_predictions.json").write_text(
        json.dumps({"updated_at_jst": now.isoformat(timespec="seconds"),
                    "annual_races": report["annual_races"], "proposals": proposals}, ensure_ascii=False,
                   indent=2, allow_nan=False), encoding="utf-8")
    if proposals:
        with (folder / "annual_department_prediction_ledger.jsonl").open("a", encoding="utf-8") as handle:
            for proposal in proposals:
                handle.write(json.dumps(proposal, ensure_ascii=False, allow_nan=False) + "\n")
    return proposals


def audit_department_predictions(output_dir=OUTPUT_DIR):
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    ledger = folder / "annual_department_prediction_ledger.jsonl"
    latest = {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if datetime.fromisoformat(row["snapshot_at"]).timestamp() >= row["close_at"]:
                continue
            key = (row["department"], row["race_id"])
            if key not in latest or row["snapshot_at"] > latest[key]["snapshot_at"]:
                latest[key] = row
    saved = folder / "annual_department_settled.json"
    previous = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else []
    settled = {(r["department"], r["race_id"]): r for r in previous}
    path = output_dir / "latest_results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    for result in results:
        if str(result.get("official_result_available")).lower() not in {"true", "1"}:
            continue
        actual = str(result.get("actual_trifecta") or "")
        if len(actual.split("-")) != 3:
            continue
        for key, forecast in latest.items():
            if key[1] == str(result["race_id"]):
                odds = pd.to_numeric(result.get("actual_trifecta_odds"), errors="coerce")
                settled[key] = {**forecast, "actual": actual,
                    "actual_odds": float(odds) if pd.notna(odds) and 0 < odds < float("inf") else None}
    saved.write_text(json.dumps(list(settled.values()), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    comparisons = []
    for department in ["data_department", "pace_department", "line_department", "risk_department"]:
        rows = [r for r in settled.values() if r["department"] == department]
        top_hits = sum(str(r["winner_car"]) == r["actual"].split("-")[0] for r in rows)
        valid = [r for r in rows if r["actual_odds"] is not None and r["tickets"]]
        stake = sum(len(r["tickets"]) * 100 for r in valid)
        returned = sum(r["actual_odds"] * 100 for r in valid if any(t["buy"] == r["actual"] for t in r["tickets"]))
        comparisons.append({"department": department, "races": len(rows), "top1_hits": top_hits,
            "top1_hit_rate": top_hits / len(rows) if rows else None,
            "portfolio_payout_races": len(valid), "flat_stake_yen": stake,
            "flat_return_yen": returned, "flat_return_rate": returned / stake if stake else None,
            "auto_promotion": False})
    report = {"updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
              "departments": comparisons, "scope": "frozen-before-close independent annual department proposals"}
    (folder / "annual_department_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    knowledge_path = folder / "annual_rider_knowledge.json"
    knowledge = json.loads(knowledge_path.read_text(encoding="utf-8")) if knowledge_path.exists() else {}
    labels = {"data_department": "データ部", "pace_department": "展開部", "line_department": "ライン部", "risk_department": "リスク部"}
    def percent(value):
        return "未算出" if value is None else f"{value * 100:.1f}%"
    cells = "".join(f'<tr><td>{labels[r["department"]]}</td><td>{r["races"]}</td>'
                    f'<td>{percent(r["top1_hit_rate"])}</td><td>{percent(r["flat_return_rate"])}</td></tr>' for r in comparisons)
    page = ('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>直近1年の部署予想</title><style>body{font-family:system-ui;background:#f4f7fb;padding:20px;color:#172b45}'
            'main{max-width:900px;margin:auto;background:white;padding:24px;border-radius:16px}'
            'table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}p{line-height:1.8}</style>'
            '<main><a href="../index.html">レース一覧へ戻る</a><h1>直近1年の部署予想</h1>'
            f'<p>保持している全履歴 {knowledge.get("total_archive_races", 0):,}レース<br>'
            f'直近1年の参考記録 {knowledge.get("annual_races", 0):,}レース ／ {knowledge.get("players", 0):,}選手<br>'
            f'集計期間 {knowledge.get("window_start", "未取得")} ～ {knowledge.get("window_end_exclusive", "未取得")}の前日</p>'
            '<p>各部署が年間成績・最近の調子・ライン位置・出走数を使って独立した検証用予想を作ります。'
            '全履歴は保持します。決まり手と行動記録は、取得できた実測分だけを集計します。</p>'
            '<table><tr><th>部署</th><th>実戦検証R</th><th>1着的中率</th><th>100円均等回収率</th></tr>' + cells + '</table>'
            '<p>発走前に固定した予想のみ検証します。本番モデルの自動置換・購入許可は行いません。'
            '回収率120%は未検証です。</p><a href="annual_department_predictions.json">最新の部署別予想</a>'
            ' ／ <a href="annual_rider_knowledge.json">選手の年間成績と取得状況</a></main></html>')
    (folder / "annual_department_report.html").write_text(page, encoding="utf-8")
    return report
