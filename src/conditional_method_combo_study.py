"""Chronology-guarded retrospective combination study for frozen conditional 3-place AI.

Predict stage never loads outcomes; assess stage never creates/refits predictions.
Existing production, commercial feeds and purchase logic are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from conditional_method_combo import K_VALUES, VARIANTS, rank, rerank_all

FIRST='2025-10-09'
END='2026-10-09'
REQUIRED_FULL_ARCHIVE='7c7493e6eb6b1dfbf085afa8fade8de73c2bb2c458b135af7fb8c0009c154333'
BASELINE_HITS_12=6242
BASELINE_RACES=13612
VERSION='conditional_scenario_comparisons_v1'
ENTRY_COLUMNS={'race_id','date','venue','race_class','race_type','is_grade_race','car_no',
               'player_id','entries_number','line_id','line_position','line_verification_status','source_url'}
RESULT_COLUMNS={'race_id','date','car_no','finish_pos'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def js(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def source_guard(args):
    main_sha=digest(args.history)
    if main_sha != REQUIRED_FULL_ARCHIVE:
        raise ValueError('history SHA differs from frozen conditional study; refusing performance claim')
    metadata=json.loads(args.model_manifest.read_text(encoding='utf-8'))
    if digest(args.model)!=metadata['model_sha256'] or metadata.get('training_cutoff_exclusive')!=FIRST:
        raise ValueError('frozen conditional model identity/cutoff mismatch')
    return main_sha,metadata


def get_entries(args):
    data=pd.read_csv(args.history,dtype={'race_id':str,'player_id':str},
                     low_memory=False,usecols=lambda c:c not in ('finish_pos','is_dnf','odds_win','payout')
                     and not c.startswith(('result_','official_','actual_')))
    if ENTRY_COLUMNS-set(data.columns):
        raise ValueError('missing required historical pre-race columns: '+str(ENTRY_COLUMNS-set(data.columns)))
    data=data[data.date.ge(FIRST)&data.date.lt(END)].copy()
    data=data.sort_values(['date','race_id','car_no'],kind='stable').reset_index(drop=True)
    return data


def predict(args):
    from method_scenario_fusion import MethodScenarioFusion
    from trifecta_chain_research import predict as chain_predict
    from fixed_year_study import eligible
    sha,manifest=source_guard(args)
    data=get_entries(args)
    # Labels are NEVER requested from CSV in this function or in chain_predict.
    clean,excluded=eligible(data)
    if excluded:
        raise ValueError('historic entrants incomplete; change cohort only with explicit audit: '+str(excluded))
    if clean.race_id.nunique()!=13742:
        raise ValueError('not exactly the frozen 13742-race comparison cohort')
    entrants={str(rid):group for rid,group in clean.groupby('race_id',sort=False)}
    scenario=MethodScenarioFusion.from_csv(args.archive,FIRST)
    if scenario.training_races<45000:
        raise ValueError('incomplete official method archive before cutoff')
    bundle=joblib.load(args.model)
    if str(bundle.get('cutoff'))!=FIRST or bundle.get('fit_races')!=30232:
        raise ValueError('conditional frozen model training manifest does not match')
    if bundle.get('leaves')!=31:
        raise ValueError('conditional model leaf configuration differs')
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    result=out/'predictions.jsonl.gz'
    seen=set();line_counts=Counter();support_counts=Counter()
    with gzip.open(result,'wt',encoding='utf-8',compresslevel=5) as dst:
        for row in chain_predict(clean,bundle,batch_size=128):
            rid=row['race_id']
            if rid in seen or rid not in entrants:
                raise ValueError('duplicate or unknown race in conditional inference')
            seen.add(rid)
            group=entrants[rid]
            if len(group)!=row['field'] or str(group.date.iloc[0])!=row['date']:
                raise ValueError('entry identity mismatch')
            meta={k:group.iloc[0][k] for k in ('venue','race_class','race_type','is_grade_race')}
            meta['date']=row['date']
            riders=[{'car_no':int(t['car_no']),'player_id':str(t['player_id']),
                     'line_id':t['line_id'],'line_position':t['line_position'],
                     'line_verification_status':t['line_verification_status'],'source_url':t['source_url']}
                    for t in group.to_dict('records')]
            pure,diag=scenario.pair_factors(riders,meta)
            verified,_=scenario.pair_factors(riders,meta,line_weight=.30)
            geometric10,_=scenario.pair_factors(riders,meta,line_weight=.30,unverified_line_weight=.10)
            geometric30,_=scenario.pair_factors(riders,meta,line_weight=.30,unverified_line_weight=.30)
            keys=row['keys']
            arms,context=rerank_all(row['chain'],keys,pure,verified,geometric10,geometric30)
            for k in ('line_verified','line_structured_unverified'):
                line_counts[k]+=int(diag[k])
            support_counts['first_riders_with_history']+=int(diag['first_riders_with_history'])
            support_counts['second_riders_with_history']+=int(diag['second_riders_with_history'])
            dst.write(json.dumps({'race_id':rid,'date':row['date'],'venue':str(meta['venue']),
                'race_class':str(meta['race_class']),'field':row['field'],
                'line_verified':diag['line_verified'],
                'line_structured_unverified':diag['line_structured_unverified'],
                'adaptive_alpha':context['adaptive_alpha'],'top1_first_mass':context['top1_first_mass'],
                'keys':keys, 'probabilities':{name:arr.tolist() for name,arr in arms.items()}},
                ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
            if len(seen)%1000==0:print('CONDITIONAL_METHOD_PREDICT',len(seen),flush=True)
    if len(seen)!=13742:raise ValueError('conditional predictions did not reproduce full cohort')
    m={'version':VERSION,'status':'frozen_unassessed','history_sha256':sha,
       'model_sha256':digest(args.model),'method_archive_sha256':digest(args.archive),
       'code_sha256':digest(Path(__file__)),'transform_sha256':digest(Path(__file__).with_name('conditional_method_combo.py')),
       'predictions_sha256':digest(result),'races':len(seen),
       'variants':list(VARIANTS),'ticket_counts':list(K_VALUES),
       'official_method_training_races':scenario.training_races,'paired_method_training_races':scenario.pair_races,
       'line_support':dict(line_counts),'rider_support':dict(support_counts),
       'prior_selection_experiment_seen_holdout':True,
       'historical_feature_capture_before_close_verified':False,
       'line_from_archive_unverified_is_research_only':True,
       'results_loaded_during_prediction':False,'commercial_use_review_required':True,
       'production_modified':False,'betting_authorized':False}
    js(out/'prediction_manifest.json',m)
    print('CONDITIONAL_METHOD_FROZEN '+json.dumps({k:m[k] for k in ('races','line_support','official_method_training_races')},ensure_ascii=False),flush=True)


def heldout_actuals(args):
    # Only opened AFTER all predictions were independently frozen.
    history=pd.read_csv(args.history,usecols=lambda c:c in RESULT_COLUMNS,dtype={'race_id':str},low_memory=False)
    history=history[history.date.ge(FIRST)&history.date.lt(END)]
    actual={}; ambiguous=0
    for rid,g in history.groupby('race_id',sort=False):
        p=pd.to_numeric(g.finish_pos,errors='coerce')
        podium=[g.loc[p.eq(pos),'car_no'].tolist() for pos in (1,2,3)]
        if all(len(x)==1 for x in podium):
            actual[str(rid)]='-'.join(str(int(x[0])) for x in podium)
        else:ambiguous+=1
    return actual,ambiguous


def score_records(records, actuals, *, bootstrap=1250):
    counters={m:Counter() for m in VARIANTS}
    scored=[];exclusions=Counter();bucket=defaultdict(lambda:defaultdict(lambda:Counter()))
    observed_ids=set()
    for row in records:
        rid=row['race_id']
        if rid in observed_ids:raise ValueError('duplicate frozen prediction')
        observed_ids.add(rid)
        actual=actuals.get(rid)
        if actual not in row['keys']:
            exclusions['missing_or_ambiguous_podium']+=1
            continue
        ix=row['keys'].index(actual)
        outcome=tuple(map(int,actual.split('-')))
        hits={};arrays={}
        for name in VARIANTS:
            p=np.asarray(row['probabilities'][name],dtype=float)
            if len(p)!=len(row['keys']) or (p<=0).any() or not np.isfinite(p).all() or not np.isclose(p.sum(),1,atol=1e-8):
                raise ValueError('invalid frozen candidate distribution '+name)
            pred=rank(row['keys'],p,max(K_VALUES));hits[name]={k:int(actual in pred[:k]) for k in K_VALUES}
            count=counters[name]; count['races']+=1;count['nll']-=math.log(p[ix])
            predicted_top=tuple(map(int,pred[0].split('-')))
            count['top1_first']+=int(predicted_top[0]==outcome[0])
            count['top1_pair']+=int(predicted_top[:2]==outcome[:2])
            for k in K_VALUES:count['hit_'+str(k)]+=hits[name][k]
            for segment in ('class:'+row['race_class'][:1],'field:'+str(row['field']),
                            'line_verified:'+str(bool(row['line_verified'])),
                            'archive_structure:'+str(bool(row['line_structured_unverified']))):
                bucket[segment][name]['races']+=1
                bucket[segment][name]['hits_12']+=hits[name][12]
        scored.append({'race_id':rid,'date':row['date'],'hits':hits})
    n=counters['conditional']['races']
    if n!=BASELINE_RACES or counters['conditional']['hit_12']!=BASELINE_HITS_12:
        raise ValueError(f'FROZEN CONDITIONAL BASELINE NOT REPRODUCED: {n} races, {counters["conditional"]["hit_12"]} / expected {BASELINE_RACES}, {BASELINE_HITS_12}')
    methods={};pair_deltas={}
    for name,c in counters.items():
        methods[name]={'races':c['races'], 'hits':{str(k):c['hit_'+str(k)] for k in K_VALUES},
           'hit_rate':{str(k):c['hit_'+str(k)]/n for k in K_VALUES},
           'mean_nll':c['nll']/n,'top1_first_rate':c['top1_first']/n,
           'top1_pair_rate':c['top1_pair']/n}
        if name!='conditional':
            delta={}
            for k in K_VALUES:
                rescue=sum(int(not r['hits']['conditional'][k] and r['hits'][name][k]) for r in scored)
                lost=sum(int(r['hits']['conditional'][k] and not r['hits'][name][k]) for r in scored)
                delta[str(k)]={'rescued':rescue,'lost':lost,'net':rescue-lost}
            pair_deltas[name]=delta
    # All variants reported; no picking winning parameter on the same period.
    by_day=defaultdict(list)
    for row in scored:by_day[row['date']].append(row)
    days=sorted(by_day)
    daily_n=np.asarray([len(by_day[d]) for d in days],dtype=float)
    rng=np.random.default_rng(20261013)
    sample=rng.integers(0,len(days),size=(bootstrap,len(days)))
    intervals={}
    for name in VARIANTS:
        if name=='conditional':continue
        diffs=np.array([sum(r['hits'][name][12]-r['hits']['conditional'][12] for r in by_day[d]) for d in days],dtype=float)
        est=100*diffs.sum()/n
        draw=100*diffs[sample].sum(axis=1)/daily_n[sample].sum(axis=1)
        intervals[name]={'delta_pp':float(est),'nominal_95_ci_pp':list(map(float,np.quantile(draw,[.025,.975])))}
    segments={name:{method:{'races':v['races'],'top12_hit_rate':v['hits_12']/v['races']}
                      for method,v in methods.items()} for name,methods in bucket.items()}
    return {'version':VERSION,'status':'retrospective_only_frozen_scored','races':n,
        'excluded':dict(exclusions),'days':len(days),'methods':methods,'paired_vs_conditional':pair_deltas,
        'bootstrap_nominal':intervals,'segments':segments,
        'baseline_reproduced':True,'basis':'Frozen conditional chain 45.856597% / 13612, no recomputation of old result labels',
        'return_rate':None,'over_50x_hit_rate':None,'over_100x_hit_rate':None,
        'no_historical_preclose_odds':True,'commercial_use_review_required':True,
        'limitations':['Previously inspected retrospective year (not new independent holdout)',
          'Historical rider/line fields not independently shown to be captured before close',
          'Line geometry without verification is exploratory and potentially contaminated',
          'Multiple comparisons not corrected; unselected coefficients reported as-is',
          'No full pre-close 50x/100x odds/payout evidence; ROI not estimated',
          'No automatic promotion or paid publication']}


def assess(args):
    manifest=json.loads((args.out_dir/'prediction_manifest.json').read_text(encoding='utf-8'))
    prediction_file=args.out_dir/'predictions.jsonl.gz'
    if manifest['status']!='frozen_unassessed' or manifest['results_loaded_during_prediction'] is not False:
        raise ValueError('wrong prediction manifest provenance')
    for p,k in ((prediction_file,'predictions_sha256'),(args.history,'history_sha256'),
                (args.model,'model_sha256'),(args.archive,'method_archive_sha256'),
                (Path(__file__),'code_sha256'),(Path(__file__).with_name('conditional_method_combo.py'),'transform_sha256')):
        if digest(p)!=manifest[k]:raise ValueError('frozen file changed: '+str(p))
    actual,ambiguous=heldout_actuals(args)
    with gzip.open(prediction_file,'rt',encoding='utf-8') as f:
        result=score_records((json.loads(line) for line in f),actual)
    result['input_races']=manifest['races'];result['unscorable_races']=ambiguous
    result['model_sha256']=manifest['model_sha256'];result['history_sha256']=manifest['history_sha256']
    js(args.out_dir/'comparison_results.json',result)
    print('CONDITIONAL_METHOD_RESULTS '+json.dumps({
         'baseline_reproduced':result['baseline_reproduced'],'races':result['races'],
         'top12':{k:round(v['hit_rate']['12'],6) for k,v in result['methods'].items()},
         'recovery':{k:v['12'] for k,v in result['paired_vs_conditional'].items()}},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['predict','assess'],required=True)
    p.add_argument('--history',type=Path,default=Path('data/raw/history.csv'))
    p.add_argument('--archive',type=Path,default=Path('data/raw/official_methods.csv'))
    p.add_argument('--model',type=Path,default=Path('models/conditional_trifecta_v1.joblib'))
    p.add_argument('--model-manifest',type=Path,default=Path('models/conditional_trifecta_v1.json'))
    p.add_argument('--out-dir',type=Path,default=Path('research/conditional_method_combo'))
    args=p.parse_args()
    if args.stage=='predict':predict(args)
    else:assess(args)

if __name__=='__main__':main()
