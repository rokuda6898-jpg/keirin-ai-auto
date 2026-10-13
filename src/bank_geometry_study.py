"""All-track physical bank geometry × officially observed race-method study.

No track-specific conclusion is hard-coded; outcome labels come only from the
stored original confirmed-race feed. Chronological holdout is never refitted
when calculating its own performance.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_geometry import add_bank_geometry, read_bank_geometry
from official_method_archive import FIRST, SECOND
from venue_time_learning import JST

VERSION = "venue_geometry_actual_method_v1"
TARGETS = {
    "first":"winner_method",
    "second":"second_method",
    "pair":"first_second_method_pair",
}
CATEGORICAL = ("venue", "tier", "division", "meeting_grade", "time_band")
NUMERIC_BASE = ("field", "hour_jst")
NUMERIC_GEOMETRY = ("bank_lap_m","bank_straight_m","bank_cant_deg",
    "bank_home_width_m","bank_back_width_m","bank_center_width_m",
    "bank_straight_fraction","bank_width_drop_m")


def read_verified(path, asof):
    frame = pd.read_csv(path, dtype={"race_id":str},low_memory=False)
    required = {"race_id","date","venue","winner_method","second_method","method_status"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing actual-method fields: {sorted(required-set(frame.columns))}")
    frame = frame[frame.method_status.eq("confirmed_by_podium_ids")].copy()
    frame["date"] = pd.to_datetime(frame["date"],errors="coerce").dt.tz_localize(None)
    frame = frame[frame.date.lt(pd.Timestamp(asof)) & frame.date.notna()]
    frame = frame.drop_duplicates("race_id",keep="last")
    frame = add_bank_geometry(frame)
    frame = frame[frame.bank_geometry_usable & frame.bank_race_system.eq("KEIRIN")].copy()
    frame["first_second_method_pair"] = frame.winner_method.astype(str) + frame.second_method.astype(str)
    for col in NUMERIC_BASE + NUMERIC_GEOMETRY:
        frame[col] = pd.to_numeric(frame.get(col,pd.Series(np.nan,index=frame.index)),errors="coerce")
    for col in CATEGORICAL:
        frame[col] = frame.get(col,pd.Series("unknown",index=frame.index)).fillna("unknown").astype(str)
    return frame


def tabulate(frame, output):
    # Shape groups only. No guesswork or copying third-party rate estimates.
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = frame.copy()
    rows["straight_band"] = pd.cut(rows.bank_straight_m, bins=[0,45,55,65,100],
         labels=["<45m","45-55m","55-65m","65m+"])
    rows["cant_band"] = pd.cut(rows.bank_cant_deg,bins=[0,30,33,36,90],
         labels=["<30deg","30-33deg","33-36deg","36deg+"])
    patterns = {
        "bank_shape_method_rates.csv":["bank_lap_m","straight_band","cant_band"],
        "bank_class_time_methods.csv":["bank_lap_m","division","field","time_band"],
        "bank_venue_time_methods.csv":["venue","division","time_band"],
        "bank_venue_year_methods.csv":["venue","division","year"],
    }
    rows["year"] = rows.date.dt.year
    for filename,cols in patterns.items():
        data=[]
        for keys,g in rows.groupby(cols,observed=True,dropna=False):
            if not isinstance(keys,tuple):keys=(keys,)
            sample = {k: None if pd.isna(v) else str(v) if k.endswith("_band") else v
                      for k,v in zip(cols,keys)}
            sample["races"] = int(len(g))
            sample["small_sample"] = bool(len(g)<30)
            for column,available in (("winner_method", FIRST),("second_method",SECOND)):
                known = g[g[column].isin(available)]
                sample[f"{column}_n"] = int(len(known))
                for tactic in available:
                    sample[f"{column}_{tactic}_count"] = int(known[column].eq(tactic).sum())
                    sample[f"{column}_{tactic}_rate"] = round(float(known[column].eq(tactic).mean()),5) if len(known) else None
            pairs = g[g.winner_method.isin(FIRST) & g.second_method.isin(SECOND)]
            labels = pairs.winner_method.astype(str) + pairs.second_method.astype(str)
            sample["joint_methods_n"] = int(len(labels))
            for first in FIRST:
                for second in SECOND:
                    key=first+second
                    sample[f"pair_{key}_count"] = int(labels.eq(key).sum())
                    sample[f"pair_{key}_rate"] = round(float(labels.eq(key).mean()),5) if len(labels) else None
            data.append(sample)
        pd.DataFrame(data).to_csv(output/filename,index=False)
    return {f: int(len(pd.read_csv(output/f))) for f in patterns}


def make_features(frame, with_geometry):
    cols = list(CATEGORICAL) + list(NUMERIC_BASE)
    if with_geometry:
        cols += list(NUMERIC_GEOMETRY)
    return frame.reindex(columns=cols)


def make_model(with_geometry):
    numeric = list(NUMERIC_BASE) + (list(NUMERIC_GEOMETRY) if with_geometry else [])
    processing = ColumnTransformer([
        ("categorical",Pipeline([
            ("fill",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore"))
        ]),list(CATEGORICAL)),
        ("numeric",Pipeline([
            ("fill",SimpleImputer(strategy="median")),
            ("scale",StandardScaler())
        ]),numeric),
    ],remainder="drop")
    return Pipeline([
        ("before_race_features",processing),
        ("classifier",LogisticRegression(C=0.4,max_iter=500,random_state=20261013))
    ])


def score(test, model, with_geometry, target):
    x = make_features(test,with_geometry)
    probs=model.predict_proba(x)
    classes=list(model.named_steps["classifier"].classes_)
    idx={name:i for i,name in enumerate(classes)}
    actual=test[target].astype(str).tolist()
    logloss=[]
    hits=[]
    missing=0
    for y,p in zip(actual,probs):
        if y not in idx:
            missing+=1
            # Unsupported future method counts as a miss, not an invented label.
            logloss.append(-math.log(1e-12));hits.append(0)
            continue
        logloss.append(-math.log(max(float(p[idx[y]]),1e-12)))
        hits.append(int(classes[int(np.argmax(p))]==y))
    return {"races":len(test),"unseen_label_races":missing,
            "mean_logloss":float(np.mean(logloss)) if logloss else None,
            "accuracy":float(np.mean(hits)) if hits else None}


def train_and_validate(frame, asof, destination, min_train=1500, min_holdout=300):
    cutoff = pd.Timestamp(asof) - pd.Timedelta(days=365)
    older=frame[frame.date.lt(cutoff)]
    newer=frame[frame.date.ge(cutoff)]
    report={"holdout_start":cutoff.strftime("%Y-%m-%d"),
        "training_races_available":int(len(older)),
        "holdout_races_available":int(len(newer)),
        "same_day_or_future_data_trained":False,
        "methods":{},
        "status":"insufficient_historical_coverage"}
    if len(older)<min_train or len(newer)<min_holdout:
        return report
    report["status"]="evaluated_shadow_only"
    for name,target in TARGETS.items():
        valid=lambda data: data[data[target].isin(
            FIRST if name=="first" else SECOND if name=="second"
            else [a+b for a in FIRST for b in SECOND])]
        train=valid(older)
        test=valid(newer)
        if len(train)<min_train or len(test)<min_holdout or train[target].nunique()<2:
            report["methods"][name]={"status":"insufficient_eligible_races",
                  "train":int(len(train)),"test":int(len(test))}
            continue
        result={"train":len(train),"test":len(test),"scenarios":{}}
        for variant in ("baseline","bank_geometry"):
            with_shape=variant=="bank_geometry"
            model=make_model(with_shape)
            model.fit(make_features(train,with_shape),train[target])
            result["scenarios"][variant]=score(test,model,with_shape,target)
            # This is a frozen fit, preserved only as historical evidence.
            joblib.dump(model,Path(destination)/f"holdout_frozen_{name}_{variant}.joblib")
        b=result["scenarios"]["baseline"]["mean_logloss"]
        g=result["scenarios"]["bank_geometry"]["mean_logloss"]
        result["geometry_minus_baseline_logloss"]=g-b
        result["improved_logloss"]=bool(g < b)
        report["methods"][name]=result
    return report


def run(archive, output_dir, asof=None):
    asof=asof or datetime.now(JST).strftime("%Y-%m-%d")
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    shape=read_bank_geometry()
    if not Path(archive).exists():
        report={"version":VERSION,"status":"missing_verified_method_archive",
                "venues":43,"no_production_changes":True}
        (out/"geometry_study_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
        return report
    frame=read_verified(archive,asof)
    counts=tabulate(frame,out)
    train_report=train_and_validate(frame,asof,out)
    report={"version":VERSION,"asof_exclusive":asof,
            "reference_velodromes":len(shape),
            "current_comparable_banks":int(shape.race_system.eq("KEIRIN").sum()),
            "verified_observed_method_races_with_valid_layout":int(len(frame)),
            "venue_coverage":int(frame.venue.nunique()),
            "bank_laps":{str(k):int(v) for k,v in shape.lap_m.value_counts().to_dict().items()},
            "method_type_counts":{
                "first":frame.winner_method.value_counts().to_dict(),
                "second":frame.second_method.value_counts().to_dict()},
            "output_cells":counts,"holdout":train_report,
            "data_sources":[
                "https://www.sog-rc27.org/field/",
                "https://www.toyama-keirin.com/guide/bank",
                "https://www.keirin.jp/pc/dfw/portal/guest/column/tenbo/2018/20180406_derby.html",
                "WINTICKET verified official race factors in original historical archive"
            ],
            "warnings":["Physical geometry does not establish tactics without real race outcomes.",
                        "Kumamoto pre-2024 layout is excluded; 250m PIST6 is a separate competition.",
                        "The source index had incorrect 333/400 labels for Toyama/Hiratsuka, corrected.",
                        "No post-race factors enter pre-race feature matrices."],
            "production_modified":False,"automatic_bet_change":False}
    (out/"geometry_study_summary.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print("BANK_GEOMETRY_STUDY "+json.dumps({k:v for k,v in report.items()
          if k not in ("data_sources","warnings")},ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",default="data/raw/official_methods.csv")
    parser.add_argument("--out-dir",default="research/bank_geometry_study")
    parser.add_argument("--asof",default=datetime.now(JST).strftime("%Y-%m-%d"))
    args=parser.parse_args()
    run(args.archive,args.out_dir,args.asof)


if __name__=="__main__":
    main()
