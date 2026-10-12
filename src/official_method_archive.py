"""Verified WINTICKET first/second-place decision-method archive.

Uses one published cup results page per day, not rider style or inferred methods.
The feed's factor codes are exact race results: 1st 逃/捲/差, 2nd 逃/捲/差/マ.
All output is separate from production models and purchase recommendations.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fetch_today_entries import http_get, extract_preloaded_state, find_query_data
from venue_time_learning import EXPECTED_VENUES, JST, classify, jst_hour, time_band, venue_name, winner_method

VERSION="official_winticket_methods_v1"
CUP_LINK=re.compile(r"^https://www\.winticket\.jp/keirin/([a-z0-9-]+)/racecard/(\d{10})/(\d+)/(\d+)(?:[/?#].*)?$")
FIRST=("逃","捲","差")
SECOND=("逃","捲","差","マ")
OUTPUT_COLUMNS=("race_id","date","venue","race_class","race_type","tier","division",
    "field","hour_jst","time_band","grade_race","winner_method","second_method",
    "winner_player_id","second_player_id","result_source","source_url","method_status")


def second_method(raw):
    name=str(raw or "").strip()
    if name in ("マ","マーク","ﾏ"):return "マ"
    return winner_method(name)


def targets(history, asof=None, min_date=None):
    """Select only real completed races with a source URL and independent identities."""
    needed={"race_id","date","venue","source_url","finish_pos","player_id"}
    absent=needed-set(history.columns)
    if absent:raise ValueError(f"History missing: {sorted(absent)}")
    df=history.copy()
    df["race_id"]=df.race_id.astype(str)
    df["_date"]=pd.to_datetime(df.date,errors="coerce")
    if asof:
        df=df[df._date.lt(pd.Timestamp(asof))]
    if min_date:
        df=df[df._date.ge(pd.Timestamp(min_date))]
    df=df.dropna(subset=["_date","race_id","source_url"])
    decoded=df.source_url.astype(str).str.extract(CUP_LINK)
    df["slug"]=decoded[0]
    df["cup_id"]=decoded[1]
    df["day_index"]=pd.to_numeric(decoded[2],errors="coerce")
    df=df.dropna(subset=["cup_id","slug","day_index"])
    df["day_index"]=df.day_index.astype(int)
    df["venue"]=df.venue.map(venue_name)
    df=df[df.venue.isin(EXPECTED_VENUES)]
    df["finish_pos"]=pd.to_numeric(df.finish_pos,errors="coerce")
    df=df.drop_duplicates(["race_id","player_id"],keep="last")
    meta={}
    for rid,g in df.groupby("race_id",sort=False):
        if len(g)<3 or (g.finish_pos.eq(1).sum()!=1 or g.finish_pos.eq(2).sum()!=1):
            continue
        if g.venue.nunique()!=1 or g.slug.nunique()!=1 or g.cup_id.nunique()!=1 or g.day_index.nunique()!=1:
            continue
        winner=g.loc[g.finish_pos.eq(1)].iloc[0]
        second=g.loc[g.finish_pos.eq(2)].iloc[0]
        cup_url=f"https://www.winticket.jp/keirin/{winner.slug}/racecard/{winner.cup_id}"
        meta[str(rid)]={
            "race_id":str(rid),"venue":str(winner.venue),
            "date":str(pd.Timestamp(winner["_date"]).date()),
            "cup_url":cup_url,"index":int(winner.day_index),
            "winner_id":str(winner.player_id),"second_id":str(second.player_id),
        }
    days=defaultdict(list)
    for info in meta.values():
        days[(info["cup_url"],info["index"])].append(info)
    return days


def distribute_across_venues(days, already, max_pages):
    """Rotate venues before revisiting one to surface all available tracks."""
    by_venue=defaultdict(list)
    for key, races in days.items():
        if all(r["race_id"] in already for r in races):continue
        current=max(x["date"] for x in races)
        venue=races[0]["venue"]
        by_venue[venue].append((current,key))
    # Do not let a new collector see only the most recent 2026 races.
    # Cycle years within each venue to build genuine train-before-test
    # evidence while preserving equal geographic coverage.
    for venue,items in list(by_venue.items()):
        year_buckets=defaultdict(deque)
        for entry in sorted(items,key=lambda x:(x[0],x[1]),reverse=True):
            year_buckets[str(entry[0])[:4]].append(entry)
        balanced=[]
        while year_buckets:
            for year in sorted(list(year_buckets),reverse=True):
                balanced.append(year_buckets[year].popleft())
                if not year_buckets[year]:del year_buckets[year]
        by_venue[venue]=balanced
    queues={k:deque(v) for k,v in sorted(by_venue.items())}
    answer=[]
    while queues and (not max_pages or len(answer)<max_pages):
        for venue in list(queues):
            if max_pages and len(answer)>=max_pages:break
            queue=queues[venue]
            _,key=queue.popleft()
            answer.append((key,days[key]))
            if not queue:del queues[venue]
    return answer


def decode_cup_results(state,index,expected):
    """Require the exact requested day's result query and correct finish/player IDs."""
    root=state.get("tanStackQuery",{}).get("queries",[])
    actual=None
    for item in root:
        key=item.get("queryKey",[])
        if "FETCH_KEIRIN_CUP_RACE_RESULT" not in str(key):continue
        args=key[-1] if isinstance(key,list) and key and isinstance(key[-1],dict) else {}
        if int(args.get("index",-1))==int(index):
            actual=item.get("state",{}).get("data",{})
            break
    if not isinstance(actual,dict) or not isinstance(actual.get("raceResults"),list):
        raise ValueError("No race results for exact requested meeting day")
    cup=find_query_data(state,"FETCH_KEIRIN_CUP")
    if not isinstance(cup,dict) or not cup.get("races"):
        raise ValueError("Missing cup race metadata")
    cup_venue=venue_name((cup.get("venue") or {}).get("name",""))
    races={str(r.get("id")):r for r in cup.get("races",[])}
    schedules={str(s.get("id")):s for s in cup.get("schedules",[])}
    expectations={x["race_id"]:x for x in expected}
    saved=[];errors=Counter()
    for result in actual.get("raceResults",[]):
        rid=str(result.get("raceId",""))
        if rid not in expectations:continue
        ref=expectations[rid]
        race=races.get(rid,{})
        if not race or venue_name(ref["venue"])!=cup_venue:
            errors["venue_or_race_mismatch"]+=1;continue
        schedule=schedules.get(str(race.get("scheduleId")),{})
        if int(schedule.get("index",-1))!=int(index) or str(schedule.get("date"))!=str(ref["date"]).replace("-",""):
            errors["schedule_mismatch"]+=1;continue
        results=result.get("results",[])
        winners=[r for r in results if r.get("order")==1]
        seconds=[r for r in results if r.get("order")==2]
        if len(winners)!=1 or len(seconds)!=1:
            errors["nonunique_podium"]+=1;continue
        first,second=winners[0],seconds[0]
        if str(first.get("playerId"))!=ref["winner_id"] or str(second.get("playerId"))!=ref["second_id"]:
            errors["archive_result_mismatch"]+=1;continue
        m1=winner_method(first.get("factor"))
        m2=second_method(second.get("factor"))
        if not m1 or not m2:
            errors["missing_official_factor"]+=1;continue
        start=jst_hour(race.get("startAt"))
        rc=str(race.get("class") or "")
        tier,division=classify({"race_class":rc,"race_type":race.get("raceType"),
                                "is_grade_race":race.get("isGradeRace")})
        if tier=="unknown":
            errors["unknown_grade"]+=1;continue
        saved.append({
            "race_id":rid,"date":ref["date"],"venue":ref["venue"],"race_class":rc,
            "race_type":str(race.get("raceType") or ""), "tier":tier,"division":division,
            "field":int(race.get("entriesNumber") or 0),"hour_jst":start,
            "time_band":time_band(start),"grade_race":bool(race.get("isGradeRace")),
            "winner_method":m1,"second_method":m2,
            "winner_player_id":ref["winner_id"],"second_player_id":ref["second_id"],
            "result_source":"WINTICKET:FETCH_KEIRIN_CUP_RACE_RESULT",
            "source_url":ref["cup_url"]+f"/results?index={index}",
            "method_status":"confirmed_by_podium_ids",
        })
    return saved,errors


def merge_verified(path, new_rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        old=pd.read_csv(path,dtype={"race_id":str,"winner_player_id":str,"second_player_id":str},low_memory=False)
    else:
        old=pd.DataFrame(columns=OUTPUT_COLUMNS)
    fresh=pd.DataFrame(new_rows,columns=OUTPUT_COLUMNS)
    merged=pd.concat([old,fresh],ignore_index=True,sort=False)
    if len(merged):
        merged=merged.drop_duplicates("race_id",keep="last").sort_values(["date","venue","race_id"]).reset_index(drop=True)
    tmp=path.with_suffix(".tmp")
    merged.to_csv(tmp,index=False)
    os.replace(tmp,path)
    return merged


def method_stats(frame,destination):
    """Raw, verifiable 1st and 2nd methods. Small samples explicitly flagged."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    base=frame.copy()
    if base.empty:
        base=pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    for c in ("field","hour_jst"):
        if c in base:base[c]=pd.to_numeric(base[c],errors="coerce")
    base["eligible_1st"]=base.winner_method.isin(FIRST)
    base["eligible_2nd"]=base.second_method.isin(SECOND)
    def grouped(columns):
        records=[]
        for key,g in base.groupby(columns,dropna=False):
            if not isinstance(key,tuple):key=(key,)
            seen=g[g.eligible_1st]
            seen2=g[g.eligible_2nd]
            record={c:None if pd.isna(value) else value for c,value in zip(columns,key)}
            record.update({"races":len(g),"observed_first":len(seen),"observed_second":len(seen2),
                           "sample_status":"usable" if len(seen)>=30 else "low_sample"})
            for x in FIRST:
                record[f"first_{x}_n"]=int(seen.winner_method.eq(x).sum())
                record[f"first_{x}_pct"]=round(100*seen.winner_method.eq(x).mean(),2) if len(seen) else None
            for x in SECOND:
                record[f"second_{x}_n"]=int(seen2.second_method.eq(x).sum())
                record[f"second_{x}_pct"]=round(100*seen2.second_method.eq(x).mean(),2) if len(seen2) else None
            paired=g[g.eligible_1st & g.eligible_2nd]
            pair_labels=paired.winner_method.astype(str)+paired.second_method.astype(str)
            record["observed_pairs"]=len(paired)
            for a in FIRST:
                for b in SECOND:
                    pattern=a+b
                    record[f"pair_{pattern}_n"]=int(pair_labels.eq(pattern).sum())
                    record[f"pair_{pattern}_pct"]=round(100*pair_labels.eq(pattern).mean(),2) if len(paired) else None
            records.append(record)
        return pd.DataFrame(records)
    overall=grouped(["venue"])
    expected=pd.DataFrame({"venue":list(EXPECTED_VENUES)})
    overall=expected.merge(overall,on="venue",how="left")
    overall.to_csv(destination/"venue_methods.csv",index=False)
    grouped(["venue","tier","division","field","time_band"]).to_csv(destination/"venue_grade_field_band.csv",index=False)
    grouped(["venue","tier","division","field","hour_jst"]).to_csv(destination/"venue_grade_field_hour.csv",index=False)
    grouped(["tier","division","field","time_band"]).to_csv(destination/"national_class_time.csv",index=False)
    return {"observed_races":int(len(base)),
            "complete_first":int(base.eligible_1st.sum()),
            "complete_second":int(base.eligible_2nd.sum()),
            "distinct_venues":int(base.venue.nunique()),
            "division_counts":base.tier.value_counts().to_dict(),
            "method_counts_first":base.winner_method.value_counts().to_dict(),
            "method_counts_second":base.second_method.value_counts().to_dict()}


def run(history_path,output_path,report_dir,asof,minimum_date,max_pages=80):
    columns={"race_id","date","venue","source_url","finish_pos","player_id"}
    history=pd.read_csv(history_path,usecols=lambda c:c in columns,
        dtype={"race_id":str,"player_id":str},low_memory=False)
    days=targets(history,asof=asof,min_date=minimum_date)
    output_path=Path(output_path)
    if output_path.exists():
        old=pd.read_csv(output_path,dtype={"race_id":str})
        already=set(old.race_id.astype(str))
    else:already=set()
    todo=distribute_across_venues(days,already,max_pages)
    failures=[];counts=Counter();new_rows=[]
    for i,((cup_url,index),expected) in enumerate(todo,1):
        url=cup_url+f"/results?index={index}"
        try:
            state=extract_preloaded_state(http_get(url,attempts=3))
            rows,err=decode_cup_results(state,index,expected)
            new_rows.extend(rows)
            counts.update(err)
            counts["pages_success"]+=1
            counts["verified_new_races"]+=len(rows)
            counts["eligible_races_on_pages"]+=len(expected)
        except Exception as exc:
            failures.append({"url":url,"reason":str(exc)[:180],"expected_races":len(expected)})
            counts["pages_failed"]+=1
        if i%10==0 or i==len(todo):
            print("PROGRESS",json.dumps({"pages":i,"of":len(todo),
                    "verified":len(new_rows),"errors":dict(counts),
                    "failures":len(failures)},ensure_ascii=False),flush=True)
        # Checkpoint even if process fails later: no loss of already validated rows.
        if i%25==0:
            merge_verified(output_path,new_rows)
            new_rows=[]
    combined=merge_verified(output_path,new_rows)
    aggregate=method_stats(combined,report_dir)
    result={"version":VERSION,"asof_exclusive":asof,"minimum_date_inclusive":minimum_date,
        "history_race_days_with_source":int(len(days)),"pages_this_run":len(todo),
        "days_remaining_with_uncaptured_races":int(sum(
            any(x["race_id"] not in set(combined.race_id) for x in expected) for expected in days.values())),
        "new_method_records_this_run":int(counts["verified_new_races"]),
        "total_method_records":int(len(combined)),
        "counts":dict(counts),"samples":aggregate,
        "failures":failures[:40],
        "production_changed":False,"purchase_authorized":False,
        "data_origin":"published WINTICKET confirmed results / not style inference",
        "commercial_use_status":"review_required"}
    Path(report_dir,"method_acquisition_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("METHOD_BACKFILL",json.dumps({k:v for k,v in result.items() if k!="failures"},ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--history",default="data/raw/history.csv")
    p.add_argument("--output",default="data/raw/official_methods.csv")
    p.add_argument("--report-dir",default="research/official_method_analysis")
    p.add_argument("--asof",default=datetime.now(JST).strftime("%Y-%m-%d"))
    p.add_argument("--since",default="2021-01-01")
    p.add_argument("--max-pages",type=int,default=80)
    a=p.parse_args()
    run(a.history,a.output,a.report_dir,a.asof,a.since,a.max_pages)


if __name__=="__main__":
    main()
