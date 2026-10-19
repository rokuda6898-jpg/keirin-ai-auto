"""Compare saved pre-close forecasts through a fixed date; never rerun a model."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-10-18"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def eligible(feed, field):
    rows = {}
    for race in feed["races"]:
        forecast = race.get(field)
        if race["date"] > CUTOFF or not forecast or not race.get("actual"):
            continue
        tickets = list(dict.fromkeys(forecast.get("tickets", [])))
        snapshot = datetime.fromisoformat(forecast["snapshot_at"])
        if (len(tickets) != 12 or snapshot.tzinfo is None
                or snapshot.timestamp() >= float(race["close_at"])):
            continue
        if any(len(t.split("-")) != 3 or len(set(t.split("-"))) != 3
               or any(c not in "123456789" or len(c) != 1 for c in t.split("-"))
               for t in tickets):
            raise ValueError("Invalid ticket: " + race["id"])
        payouts = {t: float(race.get("payouts", {}).get(t, 0)) for t in race["actual"]}
        if not all(v > 0 and v.is_integer() for v in payouts.values()):
            continue
        cancelled = set(map(str, race.get("cancelled_cars", [])))
        tickets = [t for t in tickets if not cancelled.intersection(t.split("-"))]
        if not tickets:
            continue
        if race["id"] in rows:
            raise ValueError("Duplicate race: " + race["id"])
        rows[race["id"]] = {"date": race["date"], "tickets": tickets,
                            "payouts": {t: int(v) for t, v in payouts.items()}}
    return rows


def summarize(rows):
    hits = sum(bool(set(r["tickets"]).intersection(r["payouts"])) for r in rows)
    stake = sum(len(r["tickets"]) * 100 for r in rows)
    payout = sum(sum(r["payouts"].get(t, 0) for t in r["tickets"]) for r in rows)
    dates = sorted({r["date"] for r in rows})
    return {"races": len(rows), "hits": hits, "hit_rate": hits / len(rows) if rows else None,
            "stake_yen": stake, "return_yen": payout, "profit_yen": payout - stake,
            "return_rate": payout / stake if stake else None,
            "first_date": dates[0] if dates else None, "last_date": dates[-1] if dates else None}


def main():
    conditional_path, graded_path = "outputs/conditional_live.json", "outputs/graded_live.json"
    conditional, graded = read(conditional_path), read(graded_path)
    metadata = read("models/conditional_trifecta_v1.json")
    model_path = "models/conditional_trifecta_v1.joblib"
    model_sha = hashlib.sha256((ROOT / model_path).read_bytes()).hexdigest()
    if model_sha != metadata["model_sha256"]:
        raise ValueError("Selected model does not match its manifest")
    methods = {"conditional": eligible(conditional, "shadow"),
               "graded_normal": eligible(graded, "grade"),
               "graded_previous_best": eligible(graded, "best_equation")}
    common = sorted(set.intersection(*(set(rows) for rows in methods.values())))
    if not common:
        raise ValueError("No comparable saved races")
    for rid in common:
        if any(rows[rid]["date"] != methods["conditional"][rid]["date"]
               or rows[rid]["payouts"] != methods["conditional"][rid]["payouts"]
               for rows in methods.values()):
            raise ValueError("Conflicting outcome: " + rid)
    scores = {name: summarize([rows[rid] for rid in common]) for name, rows in methods.items()}
    both_best = all(scores["conditional"][metric] >= score[metric]
                    for score in scores.values() for metric in ("hit_rate", "return_rate"))
    report = {"schema": 1, "selection_cutoff_date": CUTOFF,
              "selected_model": {"id": metadata["version"], "name": "条件付き3連単・上位12点",
                                 "path": model_path, "sha256": model_sha,
                                 "entrypoint": "src/conditional_live.py", "feed": conditional_path},
              "selection_basis": "同じ締切前保存レース・各12点・1点100円で的中率と回収率を比較",
              "best_on_both_metrics_in_common_cohort": both_best,
              "common_cohort": {"race_ids": common,
                                "dates": sorted({methods["conditional"][r]["date"] for r in common}),
                                "methods": scores},
              "conditional_all_settled": summarize(list(methods["conditional"].values())),
              "historical_reference": {"races": metadata["historical_races"],
                                       "hits": metadata["historical_hits"],
                                       "hit_rate": metadata["historical_rate"],
                                       "annual_return_rate": None, "annual_return_rate_verified": False,
                                       "source": "outputs/conditional_research/results.json"},
              "limitations": ["保存済み予想の比較で、再学習・再予想はしていません。",
                              "同一48レースでの優位は、全方式・全期間での最良や将来成績を保証しません。",
                              "回収率は100円均等。5,000〜10,000円の新しい配分の実績ではありません。",
                              "公式払戻未確認のレースは除外。返還対象の買い目は投資額から除外。",
                              "年間回収率は未検証。過去1年の的中率は過去ページからの再検証です。"],
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in (conditional_path, graded_path, "models/conditional_trifecta_v1.json")}}
    output = ROOT / "outputs/model_selection.json"
    output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "common": scores,
                      "conditional_all": report["conditional_all_settled"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
