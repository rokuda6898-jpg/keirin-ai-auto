"""Re-fetch archived provider formations; evaluate all eligible test races."""
import gzip
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
import graded_retrospective as base
from graded_tactics import tactical_inputs, conditional_tactical_features, tactical_interaction_features
from fetch_today_entries import http_get, extract_preloaded_state, find_query_data

ORIGINAL=base.features
TACTICS={}


def enhanced(records,candidate,prefix,stages,contextual):
    x=ORIGINAL(records,candidate,prefix,stages,contextual)
    info=TACTICS.get(str(records[0]['race_id']))
    return x+(conditional_tactical_features(info,candidate,prefix) if info else [np.nan]*8)


def enhanced_v2(records,candidate,prefix,stages,contextual):
    x=enhanced(records,candidate,prefix,stages,contextual)
    info=TACTICS.get(str(records[0]['race_id']))
    return x+(tactical_interaction_features(info,records,candidate,prefix) if info else [np.nan]*10)


def fetch_one(rid,url,folder):
    file=folder/(rid+'.json')
    if file.exists():return rid,json.loads(file.read_text(encoding='utf-8'))
    try:
        data=find_query_data(extract_preloaded_state(http_get(url,attempts=2)),'FETCH_KEIRIN_RACE')
        if str(data.get('race',{}).get('id'))!=rid:raise ValueError('race identity mismatch')
        obj=tactical_inputs(data)
        # Do not import newly fetched scores, results or odds into model inputs.
        obj['source_url']=url
    except Exception as exc:obj={'race_id':rid,'line_status':'fetch_failed','error':str(exc),'source_url':url}
    base.write(file,obj)
    return rid,obj


def main():
    folder=Path('research/graded_tactics');folder.mkdir(parents=True,exist_ok=True)
    sources=folder/'source_inputs';sources.mkdir(exist_ok=True)
    seed=Path('research/tactical_seed')
    for file in seed.glob('*.json'):
        shutil.copy2(file,(folder if file.name=='cup_grades.json' else sources)/file.name)
    seed_formations=seed/'historical_formations.jsonl.gz'
    if seed_formations.exists():
        with gzip.open(seed_formations,'rt',encoding='utf-8') as f:
            for line in f:
                obj=json.loads(line)
                base.write(sources/(str(obj['race_id'])+'.json'),obj)
    path=Path('data/raw/history.csv')
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='9678dad40ece12b216cd5e998aae90d20f14243e8467a32db7d45cd18c906569'
    frame=pd.read_csv(path,dtype={'race_id':str,'player_id':str},low_memory=False)
    frame=frame[frame.date.ge(base.TRAIN_START)&frame.date.lt(base.END)&pd.to_numeric(frame.is_grade_race,errors='coerce').eq(1)]
    frame=base.classify(frame,folder)
    valid=[rid for rid,r in frame.groupby('race_id') if base.valid(r)]
    frame=frame[frame.race_id.isin(valid)]
    # Use every archived outcome in the already-inspected exploratory year.
    test_ids=frame[frame.date.ge(base.START)].sort_values(['date','race_id']).drop_duplicates('race_id').race_id.tolist()
    assert len(test_ids)==1315, 'Full-year coverage changed'
    train_ids=[p.stem for p in sources.glob('*.json')]
    frame=frame[frame.date.lt(base.START)|frame.race_id.isin(test_ids)]
    races=frame[frame.race_id.isin(train_ids+test_ids)].drop_duplicates('race_id')
    missing=[r for r in races.itertuples() if not (sources/(str(r.race_id)+'.json')).exists()]
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures=[pool.submit(fetch_one,str(r.race_id),r.source_url,sources) for r in missing]
        for i,future in enumerate(as_completed(futures),1):
            rid,obj=future.result();TACTICS[rid]=obj
            if i%100==0:print(f'TACTICAL_SOURCES {i}/{len(futures)}',flush=True)
    # A cached formation is usable only if its cars match this archive race.
    for rid,race in frame.groupby('race_id'):
        obj=TACTICS.setdefault(rid,{'race_id':rid,'line_status':'not_retrieved'})
        if obj['line_status']=='verified' and set(obj['riders'])!=set(str(int(c)) for c in race.car_no):obj['line_status']='archive_roster_mismatch'
    train=frame[frame.date.lt(base.START)];target=frame[frame.date.ge(base.START)]
    coverage={}
    for name,data in [('train',train),('test',target)]:
        ids=data.race_id.unique();counts={}
        for rid in ids:
            status=TACTICS[rid]['line_status'];counts[status]=counts.get(status,0)+1
        coverage[name]={'races':len(ids),'lines':counts,'advancement_available':sum(bool(TACTICS[r].get('advancement_text')) for r in ids)}
    base.write(folder/'coverage.json',coverage);print('TACTICAL_COVERAGE '+json.dumps(coverage),flush=True)
    import joblib
    model_path=seed/'models.joblib'
    model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model_hash=='2d2fc06e43dc7ea54369dab71865fda36de1f93901686cb425b4b840df045657'
    frozen=joblib.load(model_path)
    baseline,learned,stages=frozen['champion'],frozen['tactical'],frozen['stages']
    groups=list(train.groupby('race_id',sort=False))
    try:
        base.features=enhanced_v2
        learned_v2=base.fit(groups,'stage_conditional',stages)
    finally:base.features=ORIGINAL
    joblib.dump({'champion':baseline,'tactical':learned,'tactical_v2':learned_v2,'stages':stages},folder/'models.joblib')
    predictions=[]
    for i,(rid,race) in enumerate(target.groupby('race_id',sort=False),1):
        records=race.drop(columns=['finish_pos','official_finish_pos','result_available'],errors='ignore').to_dict('records')
        dist=base.distribution(records,baseline,'stage_conditional',stages)
        try:
            base.features=enhanced
            tactical=base.distribution(records,learned,'stage_conditional',stages)
        finally:base.features=ORIGINAL
        try:
            base.features=enhanced_v2
            tactical_v2=base.distribution(records,learned_v2,'stage_conditional',stages)
        finally:base.features=ORIGINAL
        row={'race_id':rid,'date':str(race.iloc[0].date),'grade':str(race.iloc[0].grade),'stage':str(race.iloc[0].race_type),'field_size':len(race),
             'tickets':{'current_v1':[r['buy'] for r in base.rank_tickets(records,{})['main']]},'probabilities':{},'line_status':TACTICS[rid]['line_status']}
        for name,d in [('champion',dist),('tactical',tactical),('tactical_v2',tactical_v2)]:
            assert abs(sum(d.values())-1)<1e-8
            row['tickets'][name]=['-'.join(map(str,k)) for k in sorted(d,key=lambda k:(-d[k],k))[:12]]
            row['probabilities'][name]={'-'.join(map(str,k)):v for k,v in d.items()}
        predictions.append(row)
        if i%100==0:print(f'TACTICAL_PREDICT {i}/{target.race_id.nunique()}',flush=True)
    with gzip.open(folder/'predictions.jsonl.gz','wt',encoding='utf-8') as f:
        for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    base.METHODS=('current_v1','champion','tactical','tactical_v2');base.assess(predictions,target,folder)
    report=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    report.pop('paired_differences',None)
    report['coverage']=coverage
    report['model_sha256']=model_hash
    report['sample_selection']='All 1315 eligible races; exploratory comparison after the target year had already been inspected'
    report['limitations']=['Historical provider formations retrieved now; original preclose capture unverified','Advancement text not encoded without verified historical availability','No learned action timing or response scenarios; conditional formation features only','Previously inspected test year; exploratory comparison','No ROI or 100x eligibility claims']
    report['delta_vs_champion']={m:report['methods'][m]['hit_rate']-report['methods']['champion']['hit_rate'] for m in ('tactical','tactical_v2')}
    base.write(folder/'results.json',report)
    print('TACTICAL_RESULT '+json.dumps({k:v for k,v in report.items() if k!='subgroups'}),flush=True)


if __name__=='__main__':main()
