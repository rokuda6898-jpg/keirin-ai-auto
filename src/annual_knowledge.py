"""Rolling-year observed rider knowledge and independent department shadow forecasts."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def collect_prior_record_events(race_data, url, output_dir=OUTPUT_DIR):
    """Observed past-result records embedded in a current official race card."""
    from race_features import prior_results, _race_date_from_id
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    start = (pd.Timestamp(today) - pd.DateOffset(years=1)).date()
    rows = []
    names = {str(player.get("id")): str(player.get("name", "")) for player in race_data.get("players", [])}
    for record in race_data.get("records", []):
        for result in prior_results(record, today.isoformat(), limit=100):
            date = _race_date_from_id(result.get("raceId"))
            if date is None or date < start:
                continue
            row = {"race_id": str(result["raceId"]), "player_id": str(result.get("playerId", record.get("playerId"))),
                   "date": date.isoformat(), "finish_pos": result.get("order"), "result_factor": result.get("factor"),
                   "observation_source_url": url, "observation_kind": "official_prior_result_record"}
            row["player_name"] = names.get(row["player_id"], "")
            for key in ["back", "spurtSucceeded", "thrustSucceeded", "leftBehind", "splitLine", "snatchSucceeded", "competeSucceeded", "hasAccident"]:
                row[f"result_event_{key}"] = result.get(key)
            rows.append(row)
    if rows:
        folder = output_dir / "company"; folder.mkdir(parents=True, exist_ok=True)
        path = folder / "rider_official_observations.csv"
        previous = frame_read(path)
        fresh = pd.DataFrame(rows).drop_duplicates(["race_id", "player_id"], keep="last")
        if not previous.empty:
            fresh = fresh.set_index(["race_id", "player_id"]).combine_first(previous.set_index(["race_id", "player_id"])).reset_index()
        fresh.to_csv(path, index=False)
    return len(rows)


def backfill_observations(entries, limit=40, history_path=HISTORY_CSV, output_dir=OUTPUT_DIR):
    """Bounded official-result enrichment, including current-card past records."""
    history = frame_read(history_path)
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Tokyo")).date())
    urls = []
    if not history.empty and {"date", "source_url", "race_id", "player_id"}.issubset(history):
        dates = pd.to_datetime(history.date, errors="coerce")
        candidates = history[dates.ge(today - pd.DateOffset(years=1)) & dates.lt(today)].copy()
        existing = frame_read(output_dir / "company/rider_official_observations.csv")
        known = set(existing.race_id.astype(str)) if "race_id" in existing else set()
        candidates = candidates[~candidates.race_id.astype(str).isin(known)]
        active = set(entries.player_id.astype(str)) if "player_id" in entries else set()
        candidates["_active"] = candidates.player_id.astype(str).isin(active)
        races = candidates.sort_values(["_active", "date"], ascending=False).drop_duplicates("race_id")
        urls = races.source_url.dropna().astype(str).tolist()
    # Current race cards carry actual past factor/event records even when the
    # legacy archive omitted source URLs; these are never fabricated by style.
    current = entries.source_url.dropna().astype(str).unique().tolist() if "source_url" in entries else []
    urls = list(dict.fromkeys(current + urls))
    urls = [url for url in urls if url.startswith("https://www.winticket.jp/")][:limit]
    from fetch_today_entries import http_get, extract_preloaded_state, find_query_data
    def fetch(url):
        return url, find_query_data(extract_preloaded_state(http_get(url)), "FETCH_KEIRIN_RACE")
    fetched = failures = observed_rows = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, url) for url in urls]
        for future in as_completed(futures):
            try:
                url, data = future.result()
                if data:
                    collect_observations(data, url, output_dir)
                    observed_rows += collect_prior_record_events(data, url, output_dir)
                    fetched += 1
            except Exception as exc:
                failures += 1
                print(f"annual official-event backfill unavailable: {exc}")
    report = {"requested": len(futures), "fetched": fetched, "failures": failures, "prior_result_rows_seen": observed_rows,
              "limit_per_run": limit, "scope": "actual past-year events; current riders prioritized"}
    folder = output_dir / "company"; folder.mkdir(parents=True, exist_ok=True)
    (folder / "annual_event_backfill.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return report


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
    total_races = int(history.race_id.nunique()) if "race_id" in history else 0
    primary_ids = set(history.race_id.astype(str)) if "race_id" in history else set()
    primary_dates = pd.to_datetime(history.get("date", pd.Series(dtype=str)), errors="coerce")
    annual_archive_races = int(history.loc[primary_dates.ge(start) & primary_dates.lt(asof), "race_id"].nunique()) if "race_id" in history else 0
    extra = frame_read(observations)
    supplemental_races = len(set(extra.race_id.astype(str)) - primary_ids) if "race_id" in extra else 0
    if not history.empty and not extra.empty:
        # Missing fields from a supplemental observation cannot erase old values.
        history = history.set_index(["race_id", "player_id"])
        extra = extra.replace("", np.nan).set_index(["race_id", "player_id"])
        history = extra.combine_first(history).reset_index()
    elif not extra.empty:
        history = extra
    reference_races = int(history.race_id.nunique()) if "race_id" in history else 0
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
                        "observed": len(observed), "true": int(observed.isin(["true", "1", "1.0"]).sum()),
                        "true_results": stats(group.loc[group[column].astype(str).str.lower().isin(["true", "1", "1.0"])])}
            names = group.get("player_name", pd.Series(dtype=str)).dropna().astype(str).loc[lambda value: value.ne("")]
            profile["name"] = str(names.iloc[-1]) if len(names) else ""
            profiles[str(player_id)] = profile
    report = {"updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
              "fingerprint": fingerprint, "asof_date": str(asof.date()), "window_start": str(start.date()),
              "window_end_exclusive": str(asof.date()), "total_archive_races": total_races,
              "annual_races": annual_races, "annual_archive_races": annual_archive_races,
              "supplemental_result_races": supplemental_races, "reference_races_including_supplemental": reference_races,
              "players": len(profiles), "profiles": profiles,
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
                if department == "pace_department":
                    # Historical behavior associations are provisional shadow features.
                    # Require observed samples; absent flags never mean false.
                    associations = []
                    weights = []
                    for event in profile.get("events", {}).values():
                        result = event.get("true_results", {})
                        count = result.get("races", 0)
                        if event.get("observed", 0) >= 10 and count >= 5:
                            associations.append((np.array(result["rates"]) * count + rates * 20) / (count + 20))
                            weights.append(event["true"] / event["observed"])
                    if weights and sum(weights) > 0:
                        behavior = np.average(associations, axis=0, weights=weights)
                        strength = min(.25, sum(weights) / len(weights) * n / (n + 40))
                        rates = rates * (1 - strength) + behavior * strength
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
                "probability_status": "provisional_annual_shadow",
                "features_used": {"data_department": ["annual_place_rates"],
                                  "pace_department": ["annual_place_rates", "recent90", "observed_behavior_result_associations"],
                                  "line_department": ["annual_place_rates", "line_position_rates"],
                                  "risk_department": ["annual_place_rates", "sample_reliability", "unplaced_rate"]}[department],
                "purchase_authorized": False})
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
    from verified_live_audit import build_verified_live_audit
    verified_live = build_verified_live_audit(output_dir)
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
            f'選手別の補足結果 {knowledge.get("supplemental_result_races", 0):,}レース（一部選手の記録）<br>'
            f'直近1年の全体履歴 {knowledge.get("annual_archive_races", 0):,}レース<br>'
            f'補足込みの直近1年の参考記録 {knowledge.get("annual_races", 0):,}レース ／ {knowledge.get("players", 0):,}選手<br>'
            f'集計期間 {knowledge.get("window_start", "未取得")} ～ {knowledge.get("window_end_exclusive", "未取得")}の前日</p>'
            f'<p>従来集計 {verified_live["legacy_settled_rows"]}件のうち、締切前時刻を確認できる記録は {verified_live["timestamp_verified_races"]}件です。未確認分を新しい実戦検証に混ぜません。</p>'
            '<p>各部署が年間成績・最近の調子・ライン位置・出走数を使って独立した検証用予想を作ります。'
            '全履歴は保持します。決まり手と行動記録は、取得できた実測分だけを集計します。</p>'
            '<table><tr><th>部署</th><th>実戦検証R</th><th>1着的中率</th><th>100円均等回収率</th></tr>' + cells + '</table>'
            '<p>発走前に固定した予想のみ検証します。本番モデルの自動置換・購入許可は行いません。'
            '回収率120%は未検証です。</p><a href="annual_department_predictions.json">最新の部署別予想</a>'
            ' ／ <a href="annual_rider_knowledge.json">選手の年間成績と取得状況</a></main></html>')
    player_view = """<h2>選手の直近1年</h2><label for="player-search">選手名・選手IDで検索</label>
<input id="player-search" placeholder="選手名またはID" style="width:90%;padding:12px;margin:12px 0">
<p>出走数が少ない選手の率は参考値です。脚質から決まり手を推測せず、実際に取得できた記録だけを表示します。</p>
<div id="player-list">成績を読み込み中</div><script>
let riderKnowledge = {};
const percent = x => x == null ? '未取得' : (100*x).toFixed(1)+'%';
function renderPlayers() {
 const query = document.getElementById('player-search').value.trim().toLowerCase();
 const target = document.getElementById('player-list'); target.replaceChildren();
 const matched = Object.entries(riderKnowledge).filter(([id,p]) => (id+' '+(p.name||'')).toLowerCase().includes(query));
 for (const [id,p] of matched.slice(0,20)) {
  const article = document.createElement('article'); article.style.borderBottom='1px solid #ddd';
  const title = document.createElement('h3'); title.textContent=(p.name||'名前未取得')+' / ID '+id;
  const rates = document.createElement('p'); rates.textContent=p.races+'レース：1着 '+percent(p.rates[0])+' / 2着 '+percent(p.rates[1])+' / 3着 '+percent(p.rates[2]);
  const line = document.createElement('p'); line.textContent='ライン位置別：'+Object.entries(p.line_positions||{}).map(([pos,v])=>pos+'番手 '+v.races+'R・1着 '+percent(v.rates[0])).join(' ／ ');
  const tactics = document.createElement('p'); const records=Object.entries(p.winning_tactics||{}); tactics.textContent='取得済みの勝利時の決まり手：'+(records.length?records.map(([name,count])=>name+' '+count+'回').join(' ／ '):'未取得');
  const eventLabels = {back:'バック獲得',spurtSucceeded:'先行成功',thrustSucceeded:'突っ張り成功',leftBehind:'離れ',splitLine:'ライン分断',snatchSucceeded:'捲り成功',competeSucceeded:'競り成功',hasAccident:'事故あり'};
  const events = document.createElement('p'); events.textContent='取得済みの動き：'+Object.entries(p.events||{}).filter(([key,v])=>v.observed>0).map(([key,v])=>(eventLabels[key]||key)+' '+v.true+'/'+v.observed+'記録').join(' ／ ');
  article.append(title,rates,line,tactics,events); target.append(article);
 }
 const count = document.createElement('p'); count.textContent=matched.length+'人中、最大20人を表示'; target.append(count);
}
document.getElementById('player-search').addEventListener('input',renderPlayers);
fetch('annual_rider_knowledge.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('load');return r.json()}).then(d=>{riderKnowledge=d.profiles||{};renderPlayers()}).catch(()=>{document.getElementById('player-list').textContent='成績を読み込めませんでした。ページを更新してください。'});
</script>"""
    forecast_view = """<h2>部署ごとの最新予想（検証用）</h2><p>取得時点の予想です。締切後も記録を残します。買い目が空の場合は見送りです。</p><div id="department-forecasts">読み込み中</div><script>
fetch('annual_department_predictions.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('load');return r.json()}).then(d=>{
 const target=document.getElementById('department-forecasts');target.replaceChildren();
 const labels={data_department:'データ部署',pace_department:'展開部署',line_department:'ライン部署',risk_department:'リスク部署'};
 for(const p of d.proposals||[]){
  const article=document.createElement('article');article.style.borderBottom='1px solid #ddd';
  const title=document.createElement('h3');title.textContent=p.venue+' '+p.race_no+'R ／ '+labels[p.department];
  const content=document.createElement('p');content.textContent='1着候補 '+p.winner_car+'番 ／ 本線 '+p.main_count+'点・穴 '+p.hole_count+'点';
  const tickets=document.createElement('p');tickets.textContent=(p.tickets||[]).map(t=>((t.group==='main'||t.group==='本線')?'本線':'穴')+' '+t.buy+'（期待値 '+t.ev.toFixed(2)+'）').join(' ／ ')||'期待値条件を満たす買い目なし・見送り';
  article.append(title,content,tickets);target.append(article);
 }
 if(!(d.proposals||[]).length)target.textContent='対象の発走前レースがありません。';
}).catch(()=>document.getElementById('department-forecasts').textContent='予想を読み込めませんでした。');
</script>"""
    page = page.replace('</main></html>', forecast_view + player_view + '</main></html>')
    (folder / "annual_department_report.html").write_text(page, encoding="utf-8")
    return report


if __name__ == "__main__":
    from common import TODAY_CSV, TODAY_ODDS_CSV
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    entries = frame_read(TODAY_CSV)
    backfill_observations(entries)
    knowledge = build_annual_profiles(now.date())
    market = frame_read(TODAY_ODDS_CSV)
    if not entries.empty:
        entries["p_win"] = 1 / entries.groupby("race_id")["car_no"].transform("size")
        forecast_departments(entries, market, knowledge, now)
    audit_department_predictions()
    print(json.dumps({"archive_races": knowledge["total_archive_races"], "annual_races": knowledge["annual_races"],
                      "players": knowledge["players"]}, ensure_ascii=False))
