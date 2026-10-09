"""Predeclared archive challengers. No live writes; inspected year is exploratory."""
import gzip
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import graded_retrospective as base

ORIGINAL_FEATURES = base.features
METHODS = ('current_v1', 'champion', 'interaction', 'favorite_collapse', 'combined')


def summary(values):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    return [float(a.mean()),float(a.std()),float(a.max()),float(a.min())] if len(a) else [np.nan]*4


def interaction_features(records,candidate,prefix,stages,contextual):
    x=ORIGINAL_FEATURES(records,candidate,prefix,stages,contextual)
    own=base.vector(next(r for r in records if int(r['car_no'])==candidate))
    others=[base.vector(r) for r in records if int(r['car_no']) not in (*prefix,candidate)]
    # Remaining opposition, not labels or hypothetical post-race narratives.
    for index in (0,1,2,3,4,5,6,7,8):
        stats=summary([v[index] for v in others])
        x+=stats+[own[index]-stats[0],own[index]-stats[2]]
    return x


@contextmanager
def feature_mode(interaction):
    old=base.features
    base.features=interaction_features if interaction else ORIGINAL_FEATURES
    try: yield
    finally: base.features=old


def favorite(records):
    # Highest archived competition score, NOT market favorite (odds absent).
    return max(records,key=lambda r:(np.nan_to_num(base.vector(r)[0],nan=-1e9),-int(r['car_no'])))


def scenario_features(records,stages):
    lead=favorite(records); x=ORIGINAL_FEATURES(records,int(lead['car_no']),(),stages,True)
    vectors=[base.vector(r) for r in records]
    for index in range(len(base.FIELDS)):x+=summary([r[index] for r in vectors])
    return x


def category(order,car):
    return order.index(car) if car in order else 3


def reweight(dist,car,probabilities):
    masses=np.zeros(4)
    for order,p in dist.items():masses[category(order,car)]+=p
    p=np.asarray(probabilities)* (masses>0)
    p=p/p.sum()
    result={order:value*p[category(order,car)]/masses[category(order,car)] for order,value in dist.items()}
    return result


def run(frame,folder,start,end):
    folder.mkdir(parents=True,exist_ok=True)
    train=frame[frame.date.lt(start)]; target=frame[frame.date.ge(start)&frame.date.lt(end)]
    groups=list(train.groupby('race_id',sort=False))
    stages=sorted(train.race_type.fillna('').astype(str).unique())
    fitted={}
    # Champion is reproduced with its original, fixed weighting rule.
    for mode in (False,True):
        with feature_mode(mode):fitted[mode]=base.fit(groups,'stage_conditional',stages)
    xx=[];yy=[];ww=[]
    for _,race in groups:
        actual=base.podium(race)
        if actual is None:continue
        records=race.to_dict('records');xx.append(scenario_features(records,stages))
        yy.append(category(actual,int(favorite(records)['car_no'])))
        ww.append(2. if str(race.iloc[0].date)>='2024-10-10' else 1.)
    gate=base.ArchiveEstimator().fit(np.asarray(xx),yy,np.asarray(ww))
    if list(gate.model.classes_)!=[0,1,2,3]:raise ValueError('Missing scenario labels')
    predictions=[]
    for i,(rid,race) in enumerate(target.groupby('race_id',sort=False),1):
        records=race.drop(columns=['finish_pos','official_finish_pos','result_available'],errors='ignore').to_dict('records')
        ds={}
        for mode,name in ((False,'champion'),(True,'interaction')):
            with feature_mode(mode):ds[name]=base.distribution(records,fitted[mode],'stage_conditional',stages)
        probs=gate.predict_proba(np.asarray([scenario_features(records,stages)]))[0]
        car=int(favorite(records)['car_no'])
        ds['favorite_collapse']=reweight(ds['champion'],car,probs)
        ds['combined']=reweight(ds['interaction'],car,probs)
        row={'race_id':str(rid),'date':str(race.iloc[0].date),'grade':str(race.iloc[0].grade),'stage':str(race.iloc[0].race_type),'field_size':len(race),
             'tickets':{'current_v1':[r['buy'] for r in base.rank_tickets(records,{})['main']]},'probabilities':{}}
        for name,dist in ds.items():
            assert abs(sum(dist.values())-1)<1e-8
            row['tickets'][name]=['-'.join(map(str,k)) for k in sorted(dist,key=lambda k:(-dist[k],k))[:12]]
            row['probabilities'][name]={'-'.join(map(str,k)):v for k,v in dist.items()}
        predictions.append(row)
        if i%100==0:print(f'CHALLENGER {start} {i}',flush=True)
    old=base.METHODS
    try:
        base.METHODS=METHODS
        base.assess(predictions,target,folder)
    finally:base.METHODS=old
    # Paired differences against the retained champion, not just weak v1.
    report=json.loads((folder/'results.json').read_text())
    # Base assessor's intervals assume three comparisons; only the three
    # challenger-versus-champion intervals below apply to this experiment.
    report.pop('paired_differences',None)
    from official_outcomes import winning_orders
    actual={str(rid):set(winning_orders([(int(r.finish_pos),int(r.car_no)) for r in race.itertuples() if pd.notna(r.finish_pos)])) for rid,race in target.groupby('race_id')}
    diffs={}
    for method in METHODS[2:]:
        ds=[]
        for row in predictions:
            truth=actual[row['race_id']]
            if truth:ds.append({'date':row['date'],'delta':int(bool(truth&set(row['tickets'][method])))-int(bool(truth&set(row['tickets']['champion'])))})
        daily=pd.DataFrame(ds).groupby('date').delta.agg(['sum','count']).to_numpy()
        rng=np.random.default_rng(20261010);sample=daily[rng.integers(0,len(daily),(4000,len(daily)))].sum(axis=1)
        diffs[method]={'delta':float(daily[:,0].sum()/daily[:,1].sum()),'familywise_95_day_block_interval':np.quantile(sample[:,0]/sample[:,1],[.05/6,1-.05/6]).tolist()}
    report['against_champion']=diffs
    report['test_role']='older_development' if start<'2025-10-10' else 'already_inspected_exploratory_year'
    base.write(folder/'results.json',report)
    base.write(folder/'manifest.json',{'train_before':start,'test_end_exclusive':end,'training_races':len(groups),'test_races':len(predictions),'methods':METHODS,'favorite_definition':'highest competition score; market popularity unavailable'})
    with gzip.open(folder/'predictions.jsonl.gz','wt',encoding='utf-8') as f:
        for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('CHALLENGER_RESULTS '+json.dumps({k:v for k,v in report.items() if k!='subgroups'}),flush=True)


def main():
    folder=Path('research/graded_challengers');folder.mkdir(parents=True,exist_ok=True)
    path=Path('data/raw/history.csv')
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='9678dad40ece12b216cd5e998aae90d20f14243e8467a32db7d45cd18c906569'
    frame=pd.read_csv(path,dtype={'race_id':str,'player_id':str},low_memory=False)
    frame=frame[frame.date.ge(base.TRAIN_START)&frame.date.lt(base.END)]
    frame=base.classify(frame[pd.to_numeric(frame.is_grade_race,errors='coerce').eq(1)],folder)
    accepted=[rid for rid,r in frame.groupby('race_id') if base.valid(r)]
    frame=frame[frame.race_id.isin(accepted)]
    base.write(folder/'scope.json',{'blocked':{'tactical_third_place':'Verified pre-race line and tactical scenario records insufficient; do not substitute invented scenarios.','department_council':'No historical point-in-time independent department forecasts in history.csv; retrospective regeneration cannot validate original council.'},'implemented':['remaining opposition interaction','score-leader collapse (market favorite unavailable)','their combination'],'not_implemented':'race-wise softmax loss and partial pooling remain separate future experiments','timestamp_limitation':'Archive feature capture times unverified; no prospective performance claim','promotion':False})
    for start,end in [('2024-10-10','2025-10-10'),('2025-10-10','2026-10-10')]:run(frame,folder/start,start,end)


if __name__=='__main__':main()
