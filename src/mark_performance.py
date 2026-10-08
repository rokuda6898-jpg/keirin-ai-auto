"""Prospective rider-mark audit. No historical marks are reconstructed after results."""
import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common import OUTPUT_DIR

MARKS = ("◎", "○", "▲", "△", "☆")
MARK_ORDER = MARKS + ("印なし",)
SNAPSHOT_CSV = "mark_prediction_ledger.csv"
RESULTS_CSV = "mark_verified_results.csv"
JST = ZoneInfo("Asia/Tokyo")


def read_csv(path, dtypes=None):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtypes)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return pd.DataFrame()


def race_prediction_marks(group):
    """Five ranked marks; rank by normalized first/second/third-place scores."""
    scored = group.copy()
    if "car_no" not in scored or scored["car_no"].isna().any():
        return {}
    for field in ("p_win", "p_second", "p_third"):
        values = pd.to_numeric(
            scored[field] if field in scored else pd.Series(0, index=scored.index),
            errors="coerce",
        ).fillna(0).clip(lower=0)
        scored[field] = values
        maximum = values.max()
        scored[field + "_relative"] = values / maximum if maximum > 0 else 0.0
    if scored["p_win"].max() <= 0:
        return {}
    scored["_mark_score"] = (
        0.60 * scored["p_win_relative"]
        + 0.25 * scored["p_second_relative"]
        + 0.15 * scored["p_third_relative"]
    )
    scored = scored.sort_values(
        ["_mark_score", "p_win", "p_second", "p_third", "car_no"],
        ascending=[False, False, False, False, True], kind="mergesort",
    )
    return {int(row.car_no): mark for row, mark in zip(
        scored.itertuples(), MARKS
    )}


def record_preclose_marks(predictions, now_jst, output_dir=OUTPUT_DIR):
    """Freeze the first valid snapshot 40–5 minutes before close, one race only."""
    path = output_dir / SNAPSHOT_CSV
    old = read_csv(path, {"race_id": str, "car_no": int})
    recorded = set(old["race_id"].astype(str)) if "race_id" in old else set()
    additions = []
    epoch = now_jst.timestamp()
    for race_id, group in predictions.groupby("race_id", sort=False):
        race_id = str(race_id)
        if race_id in recorded or len(group) < 5 or group["car_no"].nunique() != len(group):
            continue
        close_values = pd.to_numeric(group.get("close_at"), errors="coerce")
        if close_values is None or close_values.isna().any() or close_values.nunique() != 1:
            continue
        seconds = float(close_values.iloc[0]) - epoch
        if not (300 < seconds <= 2400):
            continue
        marks = race_prediction_marks(group)
        if len(marks) != 5:
            continue
        recorded.add(race_id)
        for _, rider in group.iterrows():
            car = int(rider["car_no"])
            additions.append({
                "race_id": race_id,
                "date": str(rider.get("date", "")),
                "venue": str(rider.get("venue", "")),
                "race_no": int(rider.get("race_no", 0)),
                "car_no": car,
                "mark": marks.get(car, "印なし"),
                "snapshot_created_at_jst": now_jst.isoformat(timespec="seconds"),
                "close_at": int(close_values.iloc[0]),
                "p_win": float(rider.get("p_win", 0)),
                "p_second": float(rider.get("p_second", 0)),
                "p_third": float(rider.get("p_third", 0)),
            })
    if additions:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([old, pd.DataFrame(additions)], ignore_index=True).to_csv(path, index=False)
    return len(set(row["race_id"] for row in additions))


def retain_official_results(output_dir=OUTPUT_DIR):
    """Append verified top-3 finishes, never erase previous known results."""
    path = output_dir / RESULTS_CSV
    previous = read_csv(path, {"race_id": str})
    seen = set(previous["race_id"].astype(str)) if "race_id" in previous else set()
    snapshots = read_csv(output_dir / SNAPSHOT_CSV, {"race_id": str})
    frozen = set(snapshots["race_id"].astype(str)) if "race_id" in snapshots else set()
    source = output_dir / "latest_results.json"
    if not source.exists():
        return previous
    try:
        entries = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return previous
    if not isinstance(entries, list):
        return previous
    additions = []
    for result in entries:
        if not isinstance(result, dict):
            continue
        rid = str(result.get("race_id", ""))
        raw = str(result.get("actual_trifecta", "")).strip()
        flag = result.get("official_result_available") is True or str(result.get("official_result_available")).lower() == "true"
        if rid not in frozen or rid in seen or not flag or not re.fullmatch(r"[1-9]-[1-9]-[1-9]", raw):
            continue
        cars = [int(x) for x in raw.split("-")]
        if len(set(cars)) != 3:
            continue
        seen.add(rid)
        additions.append({"race_id": rid, "actual_trifecta": raw, "result_source": str(result.get("result_source", ""))})
    if additions:
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = pd.concat([previous, pd.DataFrame(additions)], ignore_index=True)
        previous.to_csv(path, index=False)
    return previous


def build_mark_report(output_dir=OUTPUT_DIR):
    """Display only results matched to marks captured before closing time."""
    output_dir = Path(output_dir)
    snapshot = read_csv(output_dir / SNAPSHOT_CSV, {"race_id": str})
    results = retain_official_results(output_dir)
    counts = {}
    evaluated = pd.DataFrame()
    if not snapshot.empty and not results.empty and "race_id" in snapshot and "race_id" in results:
        results = results.drop_duplicates("race_id", keep="first")
        evaluated = snapshot.merge(results[["race_id", "actual_trifecta"]], on="race_id", how="inner")
        if not evaluated.empty:
            # Check each recorded race has exactly one of each mark and no duplicate car.
            valid = []
            for rid, race in evaluated.groupby("race_id"):
                stamped = pd.to_datetime(race["snapshot_created_at_jst"], errors="coerce", utc=True)
                closing = pd.to_numeric(race["close_at"], errors="coerce")
                # Pandas 3 uses microsecond-backed datetimes in some paths,
                # while Pandas 2 may use nanoseconds. Do not divide raw int64
                # datetime storage by a hard-coded nanosecond conversion.
                snap_seconds = stamped.map(
                    lambda value: value.timestamp() if pd.notna(value) else float("nan")
                )
                distinct = set(race["mark"].astype(str))
                if (len(race) >= 5 and race["car_no"].nunique() == len(race)
                        and distinct.issuperset(MARKS)
                        and stamped.notna().all() and closing.notna().all()
                        and ((closing - snap_seconds) > 300).all()
                        and ((closing - snap_seconds) <= 2400).all()):
                    valid.append(rid)
            evaluated = evaluated[evaluated["race_id"].isin(valid)].copy()
            if not evaluated.empty:
                positions = evaluated["actual_trifecta"].str.split("-", expand=True)
                evaluated["first"] = evaluated["car_no"].astype(str).eq(positions[0])
                evaluated["top2"] = evaluated["first"] | evaluated["car_no"].astype(str).eq(positions[1])
                evaluated["top3"] = evaluated["top2"] | evaluated["car_no"].astype(str).eq(positions[2])
    for mark in MARK_ORDER:
        rows = evaluated[evaluated["mark"].eq(mark)] if not evaluated.empty else evaluated
        n = len(rows)
        counts[mark] = {
            "riders": n,
            "first": int(rows["first"].sum()) if n else 0,
            "top2": int(rows["top2"].sum()) if n else 0,
            "top3": int(rows["top3"].sum()) if n else 0,
            "first_rate": float(rows["first"].mean()) if n else None,
            "top2_rate": float(rows["top2"].mean()) if n else None,
            "top3_rate": float(rows["top3"].mean()) if n else None,
        }
    frozen_races = int(snapshot["race_id"].nunique()) if "race_id" in snapshot else 0
    settled_races = int(evaluated["race_id"].nunique()) if not evaluated.empty else 0
    report = {
        "updated_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "method": "first_valid_40_to_5_minutes_before_close; official_top3_only",
        "frozen_races": frozen_races,
        "settled_races": settled_races,
        "marks": counts,
    }
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "mark_performance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    def pct(x):
        return "未集計" if x is None else f"{x * 100:.1f}%"
    rows_html = "".join(
        "<tr><th>" + html.escape(mark) + "</th><td>" + str(counts[mark]["riders"]) + "</td>"
        + "".join("<td>" + str(counts[mark][key]) + " / " + str(counts[mark]["riders"])
                  + "（" + pct(counts[mark][key + "_rate"]) + "）</td>"
                  for key in ("first", "top2", "top3"))
        + "</tr>" for mark in MARK_ORDER
    )
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>予想印の的中成績 | NEXUS</title><style>
body{{font-family:system-ui,sans-serif;background:#f4f7fb;color:#172b45;margin:0;padding:20px}}
main{{max-width:950px;margin:auto}}section{{background:#fff;border-radius:14px;padding:22px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:11px;text-align:left;border-bottom:1px solid #d9e1ec;white-space:nowrap}}
.scroll{{overflow-x:auto}}p{{line-height:1.8}}a{{color:#0965c7}}small{{color:#52657e}}
</style></head><body><main><a href="../index.html">今日の予想</a> ／ <a href="../performance.html">成績と数字の見方</a>
<h1>予想印の的中成績</h1><section><p>発走前に固定した印の保存：{frozen_races}レース ／ 公式着順で照合済み：{settled_races}レース</p>
<div class="scroll"><table><thead><tr><th>予想印</th><th>対象選手数</th><th>1着</th><th>2着以内</th><th>3着以内</th></tr></thead>
<tbody>{rows_html}</tbody></table></div>
<p>◎本命・○対抗・▲単穴・△連下・☆注意・印なし。印は各レースの1着60%、2着25%、3着15%の相対評価です。</p>
<p>締切40～5分前に保存した最初の予想印のみ使用。公式結果が取得済みのレースに限り照合し、過去レースの印は後付けしません。
同じレースの選手ごとの割合であり、3連単の的中率・回収率とは異なります。印なしは複数選手になる場合があります。</p>
<small>更新 {html.escape(report["updated_at_jst"])}</small></section></main></body></html>"""
    (folder / "mark_performance.html").write_text(page, encoding="utf-8")
    return report
