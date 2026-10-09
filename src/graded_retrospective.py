"""Archive-only graded study; never modifies any live model or forecast ledger."""
import hashlib
import json
import math
import re
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from graded_department import rank_tickets, GRADES

START = '2025-10-10'
END = '2026-10-10'
TRAIN_START = '2022-10-10'
METHODS = ('current_v1', 'learned_positions', 'stage_positions', 'stage_conditional')
FIELDS = ('score','win_rate','place2_rate','place3_rate','back_count','front_runner_count',
          'stalker_count','deep_closer_count','marker_count','age','recent_avg_finish',
          'recent_races_count','days_since_last_race','track_win_rate','track_place2_rate',
          'track_place3_rate','track_races')


class ArchiveEstimator:
    """Train-only imputation plus explicit missing flags, including empty columns."""
    def fit(self, x, y, sample_weight):
        self.fills=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
        self.model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,
            l2_regularization=2.,learning_rate=.05,early_stopping=False,random_state=20261010)
        self.model.fit(self.transform(x),y,sample_weight=sample_weight)
        return self

    def transform(self,x):
        missing=~np.isfinite(x)
        return np.column_stack([np.where(missing,self.fills,x),missing.astype(float)])

    def predict_proba(self,x):
        return self.model.predict_proba(self.transform(x))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def podium(race):
    finish = pd.to_numeric(race.finish_pos, errors='coerce')
    cars = [race.loc[finish.eq(p), 'car_no'].tolist() for p in (1,2,3)]
    return tuple(int(c[0]) for c in cars) if all(len(c)==1 for c in cars) else None


def valid(race):
    cars = pd.to_numeric(race.car_no, errors='coerce')
    size = len(race)
    return (3 <= size <= 9 and cars.between(1,9).all() and cars.mod(1).eq(0).all()
            and cars.nunique()==size and race.player_id.nunique()==size
            and pd.to_numeric(race.entries_number,errors='coerce').eq(size).all())


def classify(frame, folder):
    """Resolve the exact meeting grade; do not call all is_grade_race rows G1-3."""
    from fetch_today_entries import http_get, extract_preloaded_state, find_query_data
    frame=frame.copy()
    frame['cup_id']=frame.source_url.astype(str).str.extract(r'/racecard/(\d{10})/')[0]
    cache=folder/'cup_grades.json'
    mapping=json.loads(cache.read_text()) if cache.exists() else {}
    failures=[]
    candidates=frame.sort_values('date').groupby('venue',sort=True).tail(1)
    for r in candidates.itertuples():
        cups=set(frame.loc[frame.venue.eq(r.venue),'cup_id'].dropna())
        if cups.issubset(mapping):continue
        try:
            state=extract_preloaded_state(http_get(r.source_url,attempts=2))
            data=find_query_data(state,'FETCH_KEIRIN_RACE')
            for c in data.get('cups',[]):mapping[str(c['id'])]=c.get('grade')
            c=find_query_data(state,'FETCH_KEIRIN_CUP_RACES').get('cup',{})
            if c.get('id'):mapping[str(c['id'])]=c.get('grade')
        except Exception as exc:failures.append({'venue':r.venue,'reason':str(exc)})
        write(cache,mapping)
    # The venue's latest page does not necessarily list every historical cup.
    # Resolve remaining meetings from their own page, once per cup, not per race.
    missing=frame.loc[frame.cup_id.notna()&~frame.cup_id.isin(mapping)].drop_duplicates('cup_id')
    print(f'RESOLVE_MISSING_CUPS {len(missing)}',flush=True)
    for r in missing.itertuples():
        try:
            state=extract_preloaded_state(http_get(r.source_url,attempts=2))
            c=find_query_data(state,'FETCH_KEIRIN_CUP_RACES').get('cup',{})
            if str(c.get('id'))==r.cup_id:mapping[r.cup_id]=c.get('grade')
            else:
                for c in find_query_data(state,'FETCH_KEIRIN_RACE').get('cups',[]):
                    if str(c.get('id'))==r.cup_id:mapping[r.cup_id]=c.get('grade')
        except Exception as exc:failures.append({'cup_id':r.cup_id,'reason':str(exc)})
        write(cache,mapping)
    frame['grade']=frame.cup_id.map(mapping).map(GRADES)
    write(folder/'grade_coverage.json',{'flagged_races':int(frame.race_id.nunique()),
          'exact_target_races':int(frame.loc[frame.grade.notna(),'race_id'].nunique()),
          'unresolved_races':int(frame.loc[frame.cup_id.map(mapping).isna(),'race_id'].nunique()),
          'non_target_races':int(frame.loc[frame.cup_id.map(mapping).notna()&frame.grade.isna(),'race_id'].nunique()),
          'failures':failures})
    return frame[frame.grade.notna()].copy()


def vector(row):
    def value(k):
        try:
            v=float(row.get(k,np.nan))
            return v if math.isfinite(v) else np.nan
        except (ValueError,TypeError):return np.nan
    return [value(k) for k in FIELDS]


def features(records, candidate, prefix, stages, contextual):
    rows={int(r['car_no']):r for r in records}
    r=rows[candidate]
    x=vector(r)
    scores=[float(z.get('score',0)) for z in records]
    x += [len(rows), x[0]-np.nanmean(scores)]
    if contextual:
        # One-hot vocabulary fitted only on old training data; unknown is all-zero.
        stage=str(r.get('race_type',''))
        x += [float(stage==s) for s in stages]
        x += [float(r.get('grade')==g) for g in ('G1','G2','G3')]
    for car in prefix:
        other=rows[car]
        x += vector(other)
        x += [x[0]-vector(other)[0]]
        verified=all(z.get('line_verification_status')=='verified' and pd.notna(z.get('line_id')) for z in (r,other))
        x += [float(r['line_id']==other['line_id']) if verified else np.nan]
    return x


def fit(groups, method, stages):
    models=[]
    for position in (1,2,3):
        xx=[]; yy=[]; ww=[]
        for _,race in groups:
            actual=podium(race)
            if actual is None:continue
            records=race.to_dict('records')
            prefix=actual[:position-1] if method=='stage_conditional' else ()
            candidates=[int(r['car_no']) for r in records if int(r['car_no']) not in prefix]
            for c in candidates:
                xx.append(features(records,c,prefix,stages,method!='learned_positions'))
                yy.append(int(c==actual[position-1]))
                ww.append((2. if str(race.iloc[0].date)>='2024-10-10' else 1.)/len(candidates))
        if not xx or len(set(yy))<2:raise ValueError('Insufficient labels')
        model=ArchiveEstimator()
        model.fit(np.asarray(xx,dtype=float),yy,sample_weight=ww)
        models.append(model)
        print(f'FIT {method} position={position} rows={len(xx)}',flush=True)
    return models


def distribution(records, models, method, stages):
    cars=sorted(int(r['car_no']) for r in records)
    triples=list(permutations(cars,3))
    p1=models[0].predict_proba(np.asarray([features(records,c,(),stages,method!='learned_positions') for c in cars]))[:,1]
    p1=np.maximum(p1,1e-9); first=dict(zip(cars,p1/p1.sum()))
    if method!='stage_conditional':
        rest=[dict(zip(cars,np.maximum(m.predict_proba(np.asarray([features(records,c,(),stages,method!='learned_positions') for c in cars]))[:,1],1e-9))) for m in models[1:]]
        return {k:first[k[0]]*rest[0][k[1]]/sum(v for c,v in rest[0].items() if c!=k[0])*
                rest[1][k[2]]/sum(v for c,v in rest[1].items() if c not in k[:2]) for k in triples}
    pairs=list(permutations(cars,2))
    p2=np.maximum(models[1].predict_proba(np.asarray([features(records,b,(a,),stages,True) for a,b in pairs]))[:,1],1e-9)
    second=dict(zip(pairs,p2)); masses={a:sum(v for (x,b),v in second.items() if x==a) for a in cars}
    p3=np.maximum(models[2].predict_proba(np.asarray([features(records,c,(a,b),stages,True) for a,b,c in triples]))[:,1],1e-9)
    third=dict(zip(triples,p3)); denominators={pair:sum(third[*pair,c] for c in cars if c not in pair) for pair in pairs}
    return {k:first[k[0]]*second[k[:2]]/masses[k[0]]*third[k]/denominators[k[:2]] for k in triples}


def assess(predictions, target, folder):
    from official_outcomes import winning_orders
    def actual_orders(race):
        rows=[]
        for r in race.itertuples():
            try:
                pos=float(r.finish_pos)
                if math.isfinite(pos) and pos.is_integer() and pos>0:rows.append((int(pos),int(r.car_no)))
            except (TypeError,ValueError):pass
        return winning_orders(rows)
    actuals={str(rid):actual_orders(r) for rid,r in target.groupby('race_id',sort=False)}
    totals={m:{'races':0,'hits':0,'stake_yen':0,'log_loss_sum':0.} for m in METHODS}
    sub={}; paired=[]
    for row in predictions:
        actual=actuals[row['race_id']]
        if not actual:continue
        hits={}
        for m in METHODS:
            hit=int(bool(set(actual)&set(row['tickets'][m]))); hits[m]=hit
            t=totals[m];t['races']+=1;t['hits']+=hit;t['stake_yen']+=len(row['tickets'][m])*100
            if m!='current_v1':t['log_loss_sum']-=math.log(max(sum(row['probabilities'][m].get(key,0) for key in actual),1e-12))
            for dimension in ('grade','stage','field_size'):
                bucket=f'{dimension}:{row[dimension]}'
                z=sub.setdefault(bucket,{}).setdefault(m,{'races':0,'hits':0})
                z['races']+=1;z['hits']+=hit
        paired.append({'date':row['date'],**hits})
    for m,t in totals.items():
        t['hit_rate']=t['hits']/t['races'] if t['races'] else None
        t['roi']=None
        if m=='current_v1':t['log_loss_sum']=None
    # Day-block bootstrap avoids treating same-day races as independent.
    paired=pd.DataFrame(paired); differences={}
    rng=np.random.default_rng(20261010)
    for m in METHODS[1:]:
        daily=paired.assign(delta=paired[m]-paired.current_v1).groupby('date').agg(delta=('delta','sum'),n=('delta','size'))
        arr=daily.to_numpy(); idx=rng.integers(0,len(arr),size=(4000,len(arr)))
        selected=arr[idx].sum(axis=1); boot=selected[:,0]/selected[:,1]
        # 98.33% marginal intervals: Bonferroni across 3 predeclared comparisons.
        differences[m]={'hit_rate_difference':float(daily.delta.sum()/daily.n.sum()),
            'familywise_95_interval':[float(v) for v in np.quantile(boot,[.05/6,1-.05/6])],
            'unit':'fraction; multiply by 100 for percentage points'}
    report={'methods':totals,'subgroups':sub,'paired_differences':differences,
            'predicted_races':len(predictions),'unscorable_races':sum(not x for x in actuals.values()),
            'tied_podium_races':sum(len(x)>1 for x in actuals.values()),
            'roi_status':'not_computable_without_verified_payouts',
            'hole_status':'not_tested_without_preclose_odds',
            'scope':'retrospective_archive_features_not_proven_point_in_time; no_live_promotion'}
    write(folder/'results.json',report)
    print('GRADED_RESEARCH_RESULTS '+json.dumps({k:v for k,v in report.items() if k!='subgroups'},ensure_ascii=False),flush=True)


def main():
    folder=Path('research/graded_retrospective');folder.mkdir(parents=True,exist_ok=True)
    path=Path('data/raw/history.csv')
    frame=pd.read_csv(path,dtype={'race_id':str,'player_id':str},low_memory=False)
    frame=frame[frame.date.ge(TRAIN_START)&frame.date.lt(END)].copy()
    all_races=int(frame.race_id.nunique())
    frame=frame[pd.to_numeric(frame.is_grade_race,errors='coerce').eq(1)].copy()
    frame=classify(frame,folder)
    exclusions=Counter(); accepted=[]
    for rid,race in frame.groupby('race_id',sort=False):
        if not valid(race):exclusions['invalid_or_incomplete']+=1
        else:accepted.append(rid)
    frame=frame[frame.race_id.isin(accepted)]
    train=frame[frame.date.lt(START)];target=frame[frame.date.ge(START)]
    groups=list(train.groupby('race_id',sort=False))
    if len(groups)<100 or target.race_id.nunique()<50:raise ValueError('Insufficient exact graded history')
    stages=sorted(train.race_type.fillna('').astype(str).unique())
    manifest={'asof_exclusive':END,'training_from':TRAIN_START,'training_before':START,
        'training_races':len(groups),'test_races':int(target.race_id.nunique()),
        'training_labeled_races':sum(podium(r) is not None for _,r in groups),
        'training_ambiguous_podium_excluded':sum(podium(r) is None for _,r in groups),
        'actual_test_first':min(target.date),'actual_test_last':max(target.date),'archive_races_in_period':all_races,
        'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'exclusions':dict(exclusions),
        'methods':METHODS,'tickets':12,'stake_per_ticket':100,'preclose_odds_available':False,
        'feature_capture_time_available':'feature_captured_at_jst' in frame,
        'verified_line_rows':int(frame.get('line_verification_status',pd.Series('',index=frame.index)).eq('verified').sum()),
        'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(folder/'manifest.json',manifest);print('GRADED_MANIFEST '+json.dumps(manifest),flush=True)
    fitted={m:fit(groups,m,stages) for m in METHODS[1:]}
    import joblib
    joblib.dump({'models':fitted,'stages':stages,'manifest':manifest},folder/'frozen_models.joblib')
    predictions=[]
    for i,(rid,race) in enumerate(target.groupby('race_id',sort=False),1):
        # Result fields never enter the allowed feature vectors or current-v1 logic.
        records=race.drop(columns=['finish_pos','official_finish_pos','result_available'],errors='ignore').to_dict('records')
        row={'race_id':rid,'date':str(race.iloc[0].date),'grade':str(race.iloc[0].grade),
             'stage':str(race.iloc[0].race_type),'field_size':len(race),'tickets':{},'probabilities':{}}
        row['tickets']['current_v1']=[r['buy'] for r in rank_tickets(records,{})['main']]
        for m in METHODS[1:]:
            dist=distribution(records,fitted[m],m,stages)
            row['probabilities'][m]={'-'.join(map(str,k)):v for k,v in dist.items()}
            row['tickets'][m]=['-'.join(map(str,k)) for k in sorted(dist,key=lambda k:(-dist[k],k))[:12]]
        predictions.append(row)
        if i%100==0:print(f'PREDICT {i}',flush=True)
    import gzip
    with gzip.open(folder/'predictions.jsonl.gz','wt',encoding='utf-8') as f:
        for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    assess(predictions,target,folder)


if __name__=='__main__':main()
