"""Frozen-source, two-phase research replay against the existing full-trifecta fusion.

Run `predict` first WITHOUT loading podium results, then `assess` later.
Only an already-frozen historical baseline and official results archive are used;
this is retrospective research, not saved-before-close evidence or ROI evidence.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from method_scenario_fusion import MethodScenarioFusion, VERSION

VARIANTS = {
    'baseline': (0.,0.),
    'rider_035': (.35,0.),
    'rider_line_035': (.35,.30),
    'rider_line_070': (.70,.30),
}
TOP_K = (3,8,12,18)
ENTRY_COLUMNS = {'race_id','date','venue','race_class','race_type','is_grade_race',
                 'car_no','player_id','entries_number','line_id','line_position',
                 'line_verification_status'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def inputs(path, cutoff, *, labels=False):
    columns = ({'race_id','date','car_no','finish_pos'} if labels else ENTRY_COLUMNS)
    d = pd.read_csv(path, usecols=lambda name: name in columns,
                    dtype={'race_id':str,'player_id':str},low_memory=False)
    if not columns.intersection(d.columns).issuperset({'race_id','date','car_no'}):
        raise ValueError('incomplete saved history schema')
    if labels and 'finish_pos' not in d:
        raise ValueError('missing official podium field')
    if not labels and not ENTRY_COLUMNS.issubset(d.columns):
        raise ValueError('missing pre-race entrant or line fields')
    return d[d.date.astype(str).ge(str(cutoff))].copy()


def verify_sources(args):
    for file in (args.archive,args.history,args.predictions,args.baseline_manifest):
        if not Path(file).is_file():
            raise FileNotFoundError(f'required actual source missing: {file}')
    m=read(args.baseline_manifest)
    if m['predictions_sha256'] != sha(args.predictions) or m['outcomes_accessed'] is not False:
        raise ValueError('original frozen predictions do not match outcome-free manifest')
    if str(m['first']) != args.cutoff:
        raise ValueError('scenario cutoff differs from frozen baseline start')
    if not m.get('retrospective_only',False):
        raise ValueError('baseline provenance is not the expected retrospective study')
    return m


def predict(args):
    source = verify_sources(args)
    model = MethodScenarioFusion.from_csv(args.archive,args.cutoff)
    # Read only an explicit pre-race whitelist. No finish/result/payout field is loaded.
    entries = inputs(args.history,args.cutoff,labels=False)
    groups = {str(rid):group for rid,group in entries.groupby('race_id',sort=False)}
    out_dir = Path(args.out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    output = out_dir/'scenario_predictions.jsonl.gz'
    rejected=Counter();count=0;lines=0;rider_support=Counter()
    with gzip.open(args.predictions,'rt',encoding='utf-8') as prev, gzip.open(output,'wt',encoding='utf-8') as dst:
        for line in prev:
            row=json.loads(line)
            rid=str(row['race_id']);group=groups.get(rid)
            if group is None:
                rejected['history_race_missing']+=1;continue
            if len(group)!=int(row['field']) or group.car_no.nunique()!=len(group):
                rejected['different_or_invalid_field']+=1;continue
            if group.date.astype(str).nunique()!=1 or str(group.iloc[0].date)!=str(row['date']):
                rejected['race_date_mismatch']+=1;continue
            meta={name:group.iloc[0][name] for name in ('venue','race_class','race_type','is_grade_race')}
            meta['date']=str(row['date'])
            riders=[{'car_no':int(g['car_no']),'player_id':str(g['player_id']),
                     'line_id':g['line_id'],'line_position':g['line_position'],
                     'line_verification_status':g['line_verification_status']}
                    for g in group.to_dict('records')]
            # No normalizations or substitutions for missing frozen baseline odds.
            keys=list(row['keys']); probs=list(row['distributions']['fusion'])
            if len(keys)!=len(probs) or set(keys)!={'-'.join(map(str,(a,b,c)))
                for a in group.car_no for b in group.car_no for c in group.car_no if a!=b and a!=c and b!=c}:
                rejected['baseline_ticket_set_mismatch']+=1;continue
            baseline=dict(zip(keys,probs))
            alternatives={}
            diagnostics=None
            for name,(strength,line_weight) in VARIANTS.items():
                reranked,diag=model.rerank(baseline,riders,meta,strength=strength,line_weight=line_weight)
                alternatives[name]=[reranked[k] for k in keys]
                if diagnostics is None:
                    diagnostics=diag
            dst.write(json.dumps({'race_id':rid,'date':meta['date'],'venue':str(meta['venue']),
                'race_class':str(meta['race_class']),'field':len(riders),
                'line_verified':diagnostics['line_verified'],
                'keys':keys,'probabilities':alternatives},ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
            count+=1;lines+=int(diagnostics['line_verified'])
            for k in ('first_riders_with_history','second_riders_with_history'):
                rider_support[k]+=diagnostics[k]
            if count%1000==0:
                print(f'METHOD_SCENARIO_PREDICT {count}',flush=True)
    manifest = {
        'version':VERSION,'status':'predictions_frozen_not_assessed',
        'cutoff_exclusive_for_training':args.cutoff,
        'trained_confirmed_races':model.training_races,
        'trained_paired_method_races':model.pair_races,
        'baseline_predictions_sha256':sha(args.predictions),
        'original_prediction_manifest_sha256':sha(args.baseline_manifest),
        'method_archive_sha256':sha(args.archive),
        'history_archive_sha256':sha(args.history),
        'scenario_predictions_sha256':sha(output),
        'predicted_races':count,'excluded_prediction_races':dict(rejected),
        'fully_verified_forecast_line_races':lines,
        'rider_profile_support_count':dict(rider_support),
        'variants':{k:list(v) for k,v in VARIANTS.items()},
        'used_columns':sorted(ENTRY_COLUMNS),
        'results_loaded_during_prediction':False,
        'evaluation_scope':'exploratory_reused_year_not_fresh_unseen_data',
        'point_in_time_line_capture_verified':False,
        'retrospective_only':True,'production_modified':False,'purchase_authorized':False,
    }
    save(out_dir/'scenario_prediction_manifest.json',manifest)
    if count<100:
        raise ValueError(f'insufficient paired real baseline predictions: {count}')
    print('METHOD_SCENARIO_FROZEN '+json.dumps({k:manifest[k] for k in
          ('trained_confirmed_races','predicted_races','fully_verified_forecast_line_races','excluded_prediction_races')},ensure_ascii=False),flush=True)


def assess(args):
    out_dir=Path(args.out_dir);manifest=read(out_dir/'scenario_prediction_manifest.json')
    pred=out_dir/'scenario_predictions.jsonl.gz'
    if sha(pred)!=manifest['scenario_predictions_sha256']:
        raise ValueError('new prediction snapshot changed after freezing')
    if manifest['cutoff_exclusive_for_training']!=args.cutoff:
        raise ValueError('cutoff changed after predictions were frozen')
    for file,key in ((args.archive,'method_archive_sha256'),(args.history,'history_archive_sha256'),
                     (args.predictions,'baseline_predictions_sha256'),(args.baseline_manifest,'original_prediction_manifest_sha256')):
        if sha(file)!=manifest[key]:
            raise ValueError('source changed between prediction and scoring: '+file)
    # Separate loading of podium results occurs ONLY in this assess stage.
    labels=inputs(args.history,args.cutoff,labels=True)
    actual={}
    for rid,race in labels.groupby('race_id',sort=False):
        places=pd.to_numeric(race.finish_pos,errors='coerce')
        choices=[race.loc[places.eq(p),'car_no'].tolist() for p in (1,2,3)]
        if all(len(x)==1 for x in choices):
            actual[str(rid)]='-'.join(str(int(c[0])) for c in choices)
    totals=defaultdict(lambda:defaultdict(float));sub=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    paired=defaultdict(dict);excluded=Counter();n=0
    with gzip.open(pred,'rt',encoding='utf-8') as src:
        for line in src:
            race=json.loads(line);rid=str(race['race_id']);winner=actual.get(rid)
            if winner not in race['keys']:
                excluded['missing_or_ambiguous_podium']+=1;continue
            n+=1;ix=race['keys'].index(winner);ranks={}
            for name,values in race['probabilities'].items():
                order=sorted(range(len(values)),key=lambda j:(-values[j],race['keys'][j]))
                hit={k:int(ix in order[:k]) for k in TOP_K}
                ranks[name]=hit
                t=totals[name]
                t['races']+=1;t['nll']-=math.log(max(1e-15,values[ix]))
                t['first_hit']+=int(race['keys'][order[0]].split('-')[0]==winner.split('-')[0])
                t['first_pair_hit']+=int(race['keys'][order[0]].split('-')[:2]==winner.split('-')[:2])
                for k in TOP_K:t[f'top{k}_hits']+=hit[k]
                for key in (f'field:{race["field"]}',f'class:{race["race_class"][:1]}',
                            f'line_verified:{bool(race["line_verified"])}'):
                    s=sub[key][name];s['races']+=1;s['top12_hits']+=hit[12]
            for name in VARIANTS:
                paired[race['date']][name]=paired[race['date']].get(name,[]) + [ranks[name][12]]
                if name != 'baseline':
                    totals[name]['rescued_12']+=int(bool(ranks[name][12]) and not ranks['baseline'][12])
                    totals[name]['lost_12']+=int(bool(ranks['baseline'][12]) and not ranks[name][12])
            if n%1000==0:print(f'METHOD_SCENARIO_ASSESS {n}',flush=True)
    if n<100:raise ValueError('insufficient assessable results')
    results={}
    for name,t in totals.items():
        d={'races':int(t['races']), 'nll':t['nll']/t['races'],
           'first_hit_rate':t['first_hit']/t['races'],
           'first_pair_hit_rate':t['first_pair_hit']/t['races']}
        for k in TOP_K:
            d[f'top{k}_hit_rate']=t[f'top{k}_hits']/t['races']
            d[f'top{k}_hits']=int(t[f'top{k}_hits'])
        if name!='baseline':
            d['rescued_12']=int(t['rescued_12']);d['lost_12']=int(t['lost_12'])
            d['top12_diff_pp']=(t['top12_hits']-totals['baseline']['top12_hits'])/t['races']*100
        results[name]=d
    rng=np.random.default_rng(20261013)
    daily=list(paired.values()); intervals={}
    for name in VARIANTS:
        if name=='baseline':continue
        per_day=np.asarray([(sum(x[name])-sum(x['baseline']),len(x['baseline'])) for x in daily],dtype=float)
        selected=rng.integers(0,len(daily),size=(1000,len(daily)))
        sums=per_day[selected].sum(axis=1)
        boot=100*sums[:,0]/sums[:,1]
        intervals[name]=[float(x) for x in np.quantile(boot,[.025,.975])]
    details={bucket:{name:{'races':int(v['races']),'top12_hit_rate':v['top12_hits']/v['races']}
         for name,v in choices.items()} for bucket,choices in sub.items()}
    report={'version':VERSION,'status':'exploratory_reused_holdout_scored',
            'assessment_races':n,'outcome_exclusions':dict(excluded),
            'methods':results,'paired_day_bootstrap_95_interval_pp':intervals,
            'segments':details,'method_archive_train_cutoff':args.cutoff,
            'original_baseline_frozen':True,
            'independently_observed_1st_2nd_methods_only':True,
            'third_place_method_improvement_assumption':False,
            'preclose_odds_available':False,'return_rate':None,
            'line_history_point_in_time_verified':False,
            'fresh_untouched_test':False,
            'production_modified':False,'purchase_authorized':False}
    save(out_dir/'scenario_results.json',report)
    print('METHOD_SCENARIO_RESULTS '+json.dumps({'assessment_races':n,'methods':results,'outcome_exclusions':dict(excluded)},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('predict','assess'),required=True)
    p.add_argument('--archive',default='data/raw/official_methods.csv')
    p.add_argument('--history',default='data/raw/history.csv')
    p.add_argument('--predictions',default='research/frozen_original/holdout_predictions.jsonl.gz')
    p.add_argument('--baseline-manifest',default='research/frozen_original/prediction_manifest.json')
    p.add_argument('--cutoff',default='2025-10-09')
    p.add_argument('--out-dir',default='research/method_scenario')
    args=p.parse_args()
    if args.stage=='predict':predict(args)
    else:assess(args)

if __name__=='__main__':main()
