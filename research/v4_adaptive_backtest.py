"""Read-only chronological validation of V4 ticket budget adaptation.

Target: 2025-10-13..2026-10-12 JST. A V4-style *new* frozen model is fit
strictly before the target, never using its finishing orders for fit or selection.
This does not assert reproducibility of the older 52.93% walk-forward report.
Only official G1/G2/G3 cup-grade matches may access the 18-ticket ceiling.
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from validate_v4 import (
    prepare_history, fit_win_model, score_win_model,
    attach_position_scores, actual_trifecta, top10_tickets,
    ordered_ticket_probability,
)

START = pd.Timestamp("2025-10-13")
END = pd.Timestamp("2026-10-13")
REPORT = ROOT / "reports" / "v4_adaptive_20261012.json"
ROWS_CSV = ROOT / "reports" / "v4_adaptive_20261012.csv"


def known_grades(target):
    """Use provider schedule grade; unknown cups never promoted to graded."""
    grade_by_cup = {}
    failures = []
    try:
        from build_history import extract_month_cups
        from graded_department import GRADES
        for month in pd.period_range("2025-10", "2026-10", freq="M"):
            try:
                cups = extract_month_cups(month.strftime("%Y%m"))
                for cup in cups:
                    grade = GRADES.get(cup.get("grade"))
                    if grade:
                        grade_by_cup[str(cup["id"])] = grade
            except Exception as exc:
                failures.append({"month": str(month), "type": type(exc).__name__, "error": str(exc)[:200]})
    except Exception as exc:
        failures.append({"type":type(exc).__name__,"error":str(exc)[:200]})
    return grade_by_cup, failures


def rank_v4(race):
    """Identical candidate universe/order to original src/validate_v4.py for top10."""
    top = race.sort_values("p_win", ascending=False, kind="mergesort")
    probabilities = pd.to_numeric(top["p_win"], errors="coerce").fillna(0).tolist()
    margin = float(probabilities[0] - probabilities[1]) if len(probabilities) > 1 else 1.0
    head_count = 1 if margin >= .40 else (2 if margin >= .03 else 3)
    heads = set(top.head(head_count)["car_no"].astype(int))
    seconds = set(race.sort_values("p_second", ascending=False, kind="mergesort").head(min(4,len(race)))["car_no"].astype(int))
    thirds = set(race.sort_values("p_third", ascending=False, kind="mergesort").head(min(6,len(race)))["car_no"].astype(int))
    rows = race[["car_no","p_win","p_second","p_third"]].to_dict("records")
    candidates = []
    for a in rows:
        ca = int(a["car_no"])
        if ca not in heads: continue
        for b in rows:
            cb = int(b["car_no"])
            if cb == ca or cb not in seconds: continue
            for c in rows:
                cc = int(c["car_no"])
                if cc in (ca,cb) or cc not in thirds: continue
                prob=ordered_ticket_probability(a,b,c,"p_second","p_third")
                candidates.append((prob,f"{ca}-{cb}-{cc}"))
    candidates.sort(key=lambda x:x[0],reverse=True)
    return [x[1] for x in candidates], margin


def compute():
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    frame = prepare_history()
    columns = list(frame.columns)
    if "date" not in frame: raise ValueError("history date missing")
    train = frame.loc[frame["date"] < START].copy()
    test = frame.loc[(frame["date"] >= START)&(frame["date"] < END)].copy()
    if train.race_id.nunique() < 8000 or test.race_id.nunique() < 200:
        raise ValueError(f"Insufficient data train={train.race_id.nunique()} test={test.race_id.nunique()}")
    print("V4_DATA",json.dumps({
        "total":int(frame.race_id.nunique()),"train_races":int(train.race_id.nunique()),
        "target_races":int(test.race_id.nunique()),"date_first":str(frame.date.min().date()),
        "date_last":str(frame.date.max().date()),"columns":columns},ensure_ascii=False),flush=True)
    del frame

    winner, fills = fit_win_model(train)
    print("V4_TRAINED_WIN_MODEL",flush=True)
    scored = score_win_model(winner, fills, test)
    del winner, fills
    scored = attach_position_scores(scored, train, "cumulative_place")
    print("V4_TRAINED_POSITION_MODELS",flush=True)
    del train, test

    grades, grade_failures = known_grades(scored)
    print("V4_GRADE_CUPS",len(grades),"FAILURES",len(grade_failures),flush=True)
    all_rows = []
    rejected=defaultdict(int)
    for rid, race in scored.groupby("race_id",sort=False):
        finish = pd.to_numeric(race["finish_pos"],errors="coerce")
        if not all(int(finish.eq(i).sum()) == 1 for i in (1,2,3)):
            rejected["ambiguous_or_missing_podium"]+=1
            continue
        cars = pd.to_numeric(race["car_no"],errors="coerce")
        if cars.isna().any() or len(race)!=cars.nunique():
            rejected["invalid_entries"]+=1;continue
        actual=actual_trifecta(race)
        if actual is None:
            rejected["missing_actual"]+=1;continue

        tickets, margin=rank_v4(race)
        expected=top10_tickets(race,"p_second","p_third",adaptive=True)
        if tickets[:10]!=expected:
            raise RuntimeError(f"V4 top10 mismatch {rid}")
        meta=race.iloc[0]
        cup_id=None
        if "source_url" in race:
            import re
            mo=re.search(r"/racecard/(\d{10})/",str(meta["source_url"]))
            cup_id=mo.group(1) if mo else None
        grade=grades.get(cup_id) if cup_id else None
        grade_flag=str(meta.get("is_grade_race","")).lower() in ("1","1.0","true")
        verified_grade=grade if grade_flag else None
        # Cutoffs are held fixed before opening the test year.
        tier=3 if margin>=.40 else (8 if margin>=.03 else 12)
        if verified_grade and tier==12:
            tier=18
        buys=tickets[:tier]
        all_rows.append({
            "race_id":str(rid),"date":str(meta["date"].date()),"venue":str(meta.get("venue","")),
            "race_no":int(meta.get("race_no",0)),"race_class":str(meta.get("race_class","")),
            "race_type":str(meta.get("race_type","")),"grade":verified_grade or "",
            "grade_flag":grade_flag,"cup_resolved":bool(grade),"margin":round(margin,6),
            "tier":tier,"actual":actual,"hit":int(actual in buys),"n_bets":len(buys),
            "hit_3":int(actual in tickets[:3]),"hit_8":int(actual in tickets[:8]),
            "hit_10":int(actual in tickets[:10]),"hit_12":int(actual in tickets[:12]),
            "hit_18":int(actual in tickets[:18]),"first_hit":int(
                int(race.sort_values("p_win",ascending=False).iloc[0]["car_no"])==int(actual.split("-")[0])),
            "baseline_n":len(tickets[:10]),
        })
    if not all_rows: raise RuntimeError("No evaluable races")
    data=pd.DataFrame(all_rows)
    data.to_csv(ROWS_CSV,index=False,encoding="utf-8")
    def summarize(d):
        races=int(len(d))
        return {"races":races,"hits":int(d.hit.sum()),"hit_rate":float(d.hit.mean()) if races else None,
            "avg_tickets":float(d.n_bets.mean()) if races else None,"total_tickets":int(d.n_bets.sum()),
            "v4_fixed_10_hit_rate":float(d.hit_10.mean()) if races else None,
            "v4_fixed_10_hits":int(d.hit_10.sum()),"max_18_hit_rate":float(d.hit_18.mean()) if races else None}
    comparison={str(n):{"races":len(data),"hits":int(data[f"hit_{n}"].sum()),
        "rate":float(data[f"hit_{n}"].mean()),"total_tickets":int(np.minimum(n,data.baseline_n*0+999).sum())}
        for n in (3,8,10,12,18)}
    baseline=defaultdict(int)
    for row in all_rows:
        hit=bool(row["hit"]);old=bool(row["hit_10"])
        if hit and not old:baseline["rescued"]+=1
        if old and not hit:baseline["lost"]+=1
    report={
        "version":"v4_adaptive_one_year_frozen_model_v1",
        "data_period":{"start_inclusive":"2025-10-13","end_exclusive":"2026-10-13",
            "actual_last_day":max(x["date"] for x in all_rows)},
        "training_cutoff_exclusive":"2025-10-13",
        "train_target_outcomes_used":False,
        "design":"single frozen HistGradientBoosting V4 win plus cumulative_place second/third, V4 candidate universe identical",
        "ticket_policy":"3 if win-pwin2 >= 0.40; 8 if >=0.03; otherwise 12; verified G1/G2/G3 with 12-tier becomes 18",
        "all":summarize(data),"tiers":{str(t):summarize(sub) for t,sub in data.groupby("tier")},
        "graded":summarize(data.loc[data.grade.ne("")]),
        "nongraded":summarize(data.loc[data.grade.eq("")]),
        "months":{str(t):summarize(sub) for t,sub in data.groupby(data.date.str.slice(0,7))},
        "fixed_k_comparison":comparison,"vs_fixed_10":dict(baseline),
        "invalid_races":dict(rejected),
        "grade_coverage":{"cups_resolved":len(grades),
            "rows_with_grade_flag":int(data.grade_flag.sum()),"graded_exact":int(data.grade.ne("").sum()),
            "flagged_unresolved":int((data.grade_flag & ~data.cup_resolved).sum()),
            "months_failed":grade_failures},
        "no_preclose_odds_roi":True,
        "proof_of_52_93_reproduction":False,
        "production_unchanged":True,
    }
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V4_ADAPTIVE_RESULTS",json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    compute()
