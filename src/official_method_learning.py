"""Chronological all-venue, class, field and start-time model of real race methods.

Training labels are explicitly observed first (逃/捲/差) and second
(逃/捲/差/マ), not rider profiles or reconstructed in-race paths.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from official_method_archive import FIRST,SECOND,OUTPUT_COLUMNS,method_stats
from venue_time_learning import EXPECTED_VENUES,JST,classify,jst_hour,time_band,venue_name

VERSION="venue_time_actual_decisive_methods_v2"
PAIRS=tuple(a+b for a in FIRST for b in SECOND)
LEVELS=("national","venue","venue_band","venue_hour")


class Distribution:
    def __init__(self, records, label, methods):
        self.label=label
        self.methods=tuple(methods)
        self.counts=defaultdict(Counter)
        for r in records.itertuples(index=False):
            actual=getattr(r,label)
            if actual not in self.methods:continue
            for tier,key in self.keys(r).items():
                self.counts[(tier,key)][actual]+=1
        self.n=sum(self.counts[('all',())].values())

    @staticmethod
    def keys(row):
        d=(str(row.division),int(row.field))
        out={'all':(), 'national':d,'venue':(str(row.venue),*d)}
        band=str(row.time_band)
        if band and band!="unknown":
            out['venue_band']=(str(row.venue),*d,band)
        hour=pd.to_numeric(getattr(row,"hour_jst",None),errors="coerce")
        if pd.notna(hour) and 0<=float(hour)<=23:
            out['venue_hour']=(str(row.venue),*d,int(hour))
        return out

    def predict(self,row):
        keys=self.keys(row)
        global_counts=self.counts[('all',())]
        total=sum(global_counts.values())
        p={k:(global_counts[k]+1)/(total+len(self.methods)) for k in self.methods}
        stages={}
        for level,k in zip(LEVELS,(100.,90.,140.,175.)):
            c=self.counts[(level,keys[level])] if level in keys else Counter()
            n=sum(c.values())
            p={m:(c[m]+k*p[m])/(n+k) for m in self.methods}
            stages[level]=p.copy()
        return stages

    def export(self):
        return {"label":self.label,"methods":list(self.methods),"observations":self.n,
                "counts":[{"level":t,"key":list(k),"n":sum(c.values()),
                           "method_counts":{m:c[m] for m in self.methods}}
                          for (t,k),c in sorted(self.counts.items(),key=lambda a:(a[0][0],str(a[0][1])))]}


def load_records(path,asof):
    p=Path(path)
    if not p.exists():return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df=pd.read_csv(p,dtype={"race_id":str},low_memory=False)
    if not set(OUTPUT_COLUMNS).issubset(df):
        raise ValueError("Observed methods archive schema incomplete")
    df["date"]=pd.to_datetime(df.date,errors="coerce")
    df=df[df.date.lt(pd.Timestamp(asof)) & df.date.notna()].copy()
    df=df[df.method_status.eq("confirmed_by_podium_ids")].drop_duplicates("race_id",keep="last")
    df["field"]=pd.to_numeric(df.field,errors="coerce")
    df["hour_jst"]=pd.to_numeric(df.hour_jst,errors="coerce")
    df=df[df.field.between(3,9) & df.venue.isin(EXPECTED_VENUES)]
    df=df[df.winner_method.isin(FIRST)&df.second_method.isin(SECOND)]
    df["method_pair"]=df.winner_method.astype(str)+df.second_method.astype(str)
    return df


def train_validate(df,asof):
    # A joint label is deterministic from the two verified result factors.
    # Accept either an already-enriched archive or raw confirmed observations.
    df=df.copy()
    df["method_pair"]=df.winner_method.astype(str)+df.second_method.astype(str)
    cutoff=pd.Timestamp(asof)-pd.DateOffset(months=3)
    train=df[df.date.lt(cutoff)]
    test=df[df.date.ge(cutoff)]
    data={"test_period_start":str(cutoff.date()),
          "train_races":len(train),"test_races":len(test),
          "outcome_leakage_guard":"only strictly earlier races fit",
          "status":"not_evaluated_insufficient_history","metrics":{}}
    if len(train)<400 or len(test)<100:
        return data
    models={"first":Distribution(train,"winner_method",FIRST),
            "second":Distribution(train,"second_method",SECOND),
            "pair":Distribution(train,"method_pair",PAIRS)}
    scores=defaultdict(list)
    correct=defaultdict(list)
    for row in test.itertuples(index=False):
        for key,model in models.items():
            label=getattr(row,model.label)
            stages=model.predict(row)
            for stage, probs in stages.items():
                scores[(key,stage)].append(-math.log(max(1e-10,probs[label])))
                correct[(key,stage)].append(int(max(probs,key=probs.get)==label))
    for key,model in models.items():
        data["metrics"][key]={}
        for stage in LEVELS:
            data["metrics"][key][stage]={
                "mean_logloss":float(np.mean(scores[(key,stage)])),
                "accuracy":float(np.mean(correct[(key,stage)]))}
    data["status"]="evaluated_shadow_only"
    return data


def infer_today(today_path, first_model, second_model, pair_model):
    if not today_path or not Path(today_path).is_file():return pd.DataFrame()
    frame=pd.read_csv(today_path,dtype={"race_id":str},low_memory=False)
    required={'race_id','car_no','venue','date'}
    if not required.issubset(frame):return pd.DataFrame()
    rows=[]
    for rid,g in frame.groupby("race_id",sort=False):
        if g.car_no.nunique()!=len(g) or not 3<=len(g)<=9:continue
        row=g.iloc[0]
        tier,division=classify(row)
        v=venue_name(row.venue)
        if tier=="unknown" or v not in EXPECTED_VENUES:continue
        hour=jst_hour(row.get("start_at"))
        context=type("PreRace",(),{"venue":v,"division":division,
             "field":len(g),"time_band":time_band(hour),"hour_jst":hour})()
        first=first_model.predict(context)
        second=second_model.predict(context)
        pairs=pair_model.predict(context)
        result={"race_id":str(rid),"date":str(row.date),"venue":v,"tier":tier,
                "division":division,"field":len(g),"hour_jst":hour,
                "time_band":time_band(hour),"is_shadow":True,
                "purchase_authorized":False}
        for level in LEVELS:
            for m in FIRST: result[f"p_first_{m}_{level}"]=round(first[level][m],6)
            for m in SECOND: result[f"p_second_{m}_{level}"]=round(second[level][m],6)
            for m in PAIRS: result[f"p_pair_{m}_{level}"]=round(pairs[level][m],6)
        rows.append(result)
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",default="data/raw/official_methods.csv")
    p.add_argument("--asof",default=datetime.now(JST).strftime("%Y-%m-%d"))
    p.add_argument("--today",default="data/raw/today_entries.csv")
    p.add_argument("--out-dir",default="research/official_method_analysis")
    args=p.parse_args()
    dest=Path(args.out_dir);dest.mkdir(parents=True,exist_ok=True)
    df=load_records(args.archive,args.asof)
    if df.empty:
        summary={"version":VERSION,"status":"no_verified_first_second_methods","races":0,
            "production_modified":False}
        (dest/"method_training_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
        print(json.dumps(summary,ensure_ascii=False));return
    models={"first":Distribution(df,"winner_method",FIRST),
            "second":Distribution(df,"second_method",SECOND),
            "pair":Distribution(df,"method_pair",PAIRS)}
    for which,model in models.items():
        (dest/f"{which}_smoothed_profile.json").write_text(
            json.dumps(model.export(),ensure_ascii=False,indent=2),encoding="utf-8")
    preview=infer_today(args.today,models["first"],models["second"],models["pair"])
    preview.to_csv(dest/"today_method_shadow.csv",index=False)
    holdout=train_validate(df,args.asof)
    summary={"version":VERSION,"trained_actual_races":int(len(df)),
             "venue_count":int(df.venue.nunique()),"tier_counts":df.tier.value_counts().to_dict(),
             "first_methods":df.winner_method.value_counts().to_dict(),
             "second_methods":df.second_method.value_counts().to_dict(),
             "pair_methods":df.method_pair.value_counts().to_dict(),
             "holdout":holdout,
             "status":"shadow_only_waiting_validation" if holdout["status"]!="evaluated_shadow_only"
                else "shadow_only_evaluated",
             "production_modified":False,"purchase_authorized":False,
             "commercial_use_status":"review_required",
             "limitation":"Only captured official race methods; incomplete historical coverage. Time effects shrink to venue/national priors."}
    (dest/"method_training_summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("OFFICIAL_METHOD_MODEL",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
