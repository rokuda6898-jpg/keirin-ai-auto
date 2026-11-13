"""Research-only cross-combination of independently frozen 2025-26 trifecta forecasts.

Do NOT use this as pre-close production performance. Archived lineup geometry is
unverified. Source prediction and method studies both saw this historical period.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION = 'frozen_anchor_method_compatibility_v1'
POINTS = (3, 8, 12, 18)
BASE = 'refit_first_anchor'
OPTIONS = ('rider_035', 'archive_geometry_010', 'archive_geometry_030')
VARIANTS = (('old_fusion',None), ('refit_fusion',None), ('refit_base',None),
            (BASE,None), *[(f'anchor_x_{x}', x) for x in OPTIONS],
            *[(f'first_fixed_x_{x}',x) for x in OPTIONS])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):
            h.update(c)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def verify(repair, scenario, *, result_check=True):
    r = load(repair/'refit/prediction_manifest.json')
    s = load(scenario/'scenario_prediction_manifest.json')
    if r.get('outcomes_accessed') is not False or s.get('results_loaded_during_prediction') is not False:
        raise ValueError('outcome-free prediction provenance not verified')
    if r.get('source_predictions_sha256') != s.get('baseline_predictions_sha256'):
        raise ValueError('predictions use different baseline cohort')
    if s.get('cutoff_exclusive_for_training')!='2025-10-09':
        raise ValueError('unexpected method training cut-off')
    files=((repair/'refit/predictions.jsonl.gz',r['predictions_sha256']),
           (scenario/'scenario_predictions.jsonl.gz',s['scenario_predictions_sha256']))
    for path,expected in files:
        if not path.is_file() or digest(path)!=expected:
            raise ValueError('source prediction artifact changed or missing: '+str(path))
    if result_check and not (repair/'assessed_races.jsonl.gz').is_file():
        raise FileNotFoundError('required labeled evidence missing; never substitute outcomes')
    return r,s


def combine(keys, reference, scenario_baseline, scenario_alternatives):
    """All candidates; without labels. Keep conditional third probabilities unchanged.

    New ratio is the scenario's calibrated *relative* first-two weight. Applying
    to a different full distribution is exploratory, not a fitted ensemble.
    """
    if len(keys)!=len(reference) or keys != sorted(keys) or len(keys)!=len(set(keys)):
        raise ValueError('invalid or noncanonical ordered trifecta keys')
    triples=np.asarray([list(map(int,k.split('-'))) for k in keys],dtype=np.int16)
    if triples.shape!=(len(keys),3):raise ValueError('invalid ordered trifecta')
    first, second=triples[:,0],triples[:,1]
    n=len(set(first))
    expected_cars=sorted(set(first))
    if (len(keys)!=n*(n-1)*(n-2) or set(triples.flatten())!=set(expected_cars)
            or (triples[:,0]==triples[:,1]).any()
            or (triples[:,0]==triples[:,2]).any()
            or (triples[:,1]==triples[:,2]).any()):
        raise ValueError('incomplete or invalid ordered trifecta space')
    p=np.asarray(reference,dtype=np.float64)
    prior=np.asarray(scenario_baseline,dtype=np.float64)
    if len(prior)!=len(keys) or not np.isfinite(p).all() or not np.isfinite(prior).all():
        raise ValueError('missing/invalid probability')
    if np.any(p<=0) or np.any(prior<=0) or not np.isclose(p.sum(),1,atol=1e-8) or not np.isclose(prior.sum(),1,atol=1e-8):
        raise ValueError('unnormalized or nonpositive probability')
    first_mass=np.bincount(first,weights=p,minlength=10)
    pair_ids=first*10+second
    pair_counts=np.bincount(pair_ids,minlength=100)
    combined={}
    for name in OPTIONS:
        candidate=np.asarray(scenario_alternatives[name],dtype=np.float64)
        if len(candidate)!=len(keys) or np.any(~np.isfinite(candidate)) or np.any(candidate<=0) or not np.isclose(candidate.sum(),1.,atol=1e-8):
            raise ValueError('invalid scenario candidate '+name)
        ratio=candidate/prior
        # Vectorized test: every ticket with the same first-two vehicles
        # must have the same weight multiplier, independent of third.
        pair_mean=np.bincount(pair_ids,weights=ratio,minlength=100)/np.maximum(1,pair_counts)
        if not np.allclose(ratio,pair_mean[pair_ids],rtol=1e-10,atol=1e-10):
            raise ValueError('third-specific scenario ratio is not supported')
        weighted=p*ratio
        weighted/=weighted.sum()
        combined['anchor_x_'+name]=weighted
        new_mass=np.bincount(first,weights=weighted,minlength=10)
        if np.any(new_mass[first]<=0):
            raise ValueError('zero first probability')
        fixed=weighted*(first_mass[first]/new_mass[first])
        fixed/=fixed.sum()
        if not np.allclose(np.bincount(first,weights=fixed,minlength=10),first_mass,atol=1e-10):
            raise ValueError('first-stage probability was not preserved')
        # Pair adjustment: third | first,second remains unchanged.
        ratios=fixed/p
        pair_ratio_mean=np.bincount(pair_ids,weights=ratios,minlength=100)/np.maximum(1,pair_counts)
        if not np.allclose(ratios,pair_ratio_mean[pair_ids],rtol=1e-10,atol=1e-10):
            raise ValueError('third-stage ordering was modified')
        combined['first_fixed_x_'+name]=fixed
    return combined


def predict(repair,scenario,out):
    manifest_repair, manifest_scenario=verify(repair,scenario,result_check=False)
    out.mkdir(parents=True,exist_ok=True)
    names=[name for name,_ in VARIANTS]
    limit=manifest_repair['predicted_races']
    tensor_path=out/'combined_probabilities.npy'
    meta_path=out/'prediction_metadata.jsonl.gz'
    # Binary output avoids serializing large all-trifecta arrays through slow gzip.
    tensor=np.lib.format.open_memmap(tensor_path,mode='w+',dtype='float64',shape=(limit,len(names),504))
    tensor[:]=0.
    n=0;line_verified=0;line_archive=0
    with gzip.open(repair/'refit/predictions.jsonl.gz','rt',encoding='utf8') as f1, \
         gzip.open(scenario/'scenario_predictions.jsonl.gz','rt',encoding='utf8') as f2, \
         gzip.open(meta_path,'wt',encoding='utf8',compresslevel=1) as fout:
        for l1,l2 in zip(f1,f2):
            r=json.loads(l1);s=json.loads(l2)
            if n>=limit or not (r['race_id']==s['race_id'] and r['date']==s['date']
                    and r['keys']==s['keys'] and r['field']==s['field']):
                raise ValueError('misaligned independent archived race predictions')
            keys=r['keys']
            b={'old_fusion':np.asarray(s['probabilities']['baseline']),
               'refit_fusion':np.asarray(r['distributions']['fusion']),
               'refit_base':np.asarray(r['distributions']['base']),
               BASE:np.asarray(r['distributions']['first_anchor'])}
            b.update(combine(keys,b[BASE],b['old_fusion'],s['probabilities']))
            if set(b)!=set(names):raise ValueError('unexpected variant set')
            for j,name in enumerate(names):
                prob=b[name]
                if (len(prob)!=len(keys) or not np.isfinite(prob).all()
                    or np.any(prob<=0) or not np.isclose(prob.sum(),1,atol=1e-8)):
                    raise ValueError('invalid output distribution')
                tensor[n,j,:len(keys)]=prob
            row={'race_id':r['race_id'],'date':r['date'],'race_class':r['race_class'],
                 'field':r['field'],'line_verified':s['line_verified'],
                 'archived_line_unverified':s['line_structured_unverified'],'keys':keys}
            fout.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
            n+=1;line_verified+=bool(s['line_verified']);line_archive+=bool(s['line_structured_unverified'])
            if n%2000==0:print(f'COMPATIBILITY_PREDICT {n}',flush=True)
    del tensor
    if n!=limit or n!=manifest_scenario['predicted_races']:
        raise ValueError('truncated forecast archive')
    m={'version':VERSION,'predicted_races':n,'outcomes_loaded':False,
       'original_predictions_sha256':manifest_repair['source_predictions_sha256'],
       'repair_predictions_sha256':manifest_repair['predictions_sha256'],
       'scenario_predictions_sha256':manifest_scenario['scenario_predictions_sha256'],
       'combined_probabilities_sha256':digest(tensor_path),
       'prediction_metadata_sha256':digest(meta_path),
       'variants':names,'tensor_width':504,'line_verified':line_verified,
       'archived_structured_unverified':line_archive,
       'source_snapshot_historical_not_preclose_verified':True,
       'untouched_future_holdout':False,'production_changed':False,
       'purchase_authorized':False}
    save(out/'prediction_manifest.json',m)
    print('COMPATIBILITY_FROZEN '+json.dumps(m,ensure_ascii=False),flush=True)


def assess(repair,scenario,out):
    r,s=verify(repair,scenario,result_check=True)
    m=load(out/'prediction_manifest.json')
    p=out/'combined_probabilities.npy'
    meta=out/'prediction_metadata.jsonl.gz'
    if (m['combined_probabilities_sha256']!=digest(p)
        or m['prediction_metadata_sha256']!=digest(meta)
        or m['outcomes_loaded'] is not False or
        m['repair_predictions_sha256']!=r['predictions_sha256'] or
        m['scenario_predictions_sha256']!=s['scenario_predictions_sha256']):
        raise ValueError('prediction manifest not frozen from supplied artifacts')
    tensor=np.load(p,mmap_mode='r')
    names=m['variants']
    if tensor.shape!=(m['predicted_races'],len(names),m['tensor_width']):
        raise ValueError('frozen tensor shape changed')
    outcomes={}
    excluded=Counter()
    with gzip.open(repair/'assessed_races.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if 'actual' in row:outcomes[row['race_id']]=row['actual']
            else:excluded[row.get('exclusion','unknown')]+=1
    results=defaultdict(lambda:{'count':0,'nll':0.,'hits':Counter()})
    byday=defaultdict(lambda:defaultdict(lambda:Counter()))
    segments=defaultdict(lambda:defaultdict(lambda:Counter()))
    rescued=defaultdict(Counter)
    scored=0
    with gzip.open(meta,'rt',encoding='utf8') as f:
        for i,line in enumerate(f):
            race=json.loads(line)
            actual=outcomes.get(race['race_id'])
            if actual is None:continue
            keys=race['keys']
            if actual not in keys:raise ValueError('non-matching official finish')
            wanted=keys.index(actual)
            hits={}
            for j,name in enumerate(names):
                probs=tensor[i,j,:len(keys)]
                rank=np.argsort(-probs,kind='stable')
                hit={k:int(wanted in rank[:k]) for k in POINTS}
                hits[name]=hit
                a=results[name];a['count']+=1;a['nll']-=math.log(float(probs[wanted]))
                for k,v in hit.items():a['hits'][k]+=v
                for segment in (f'class:{race["race_class"]}',f'field:{race["field"]}',f'archived_line:{race["archived_line_unverified"]}'):
                    segments[segment][name]['count']+=1
                    segments[segment][name]['hits12']+=hit[12]
            for name in hits:
                if name==BASE:continue
                for k in POINTS:
                    if hits[name][k] and not hits[BASE][k]:rescued[name][f'rescued_{k}']+=1
                    elif hits[BASE][k] and not hits[name][k]:rescued[name][f'lost_{k}']+=1
                byday[race['date']][name]['delta12']+=hits[name][12]-hits[BASE][12]
            byday[race['date']][BASE]['n']+=1
            scored+=1
    if scored!=len(outcomes) or scored!=13612:
        raise ValueError('unexpected number of official, comparable outcomes')
    # Reproduce both independently published baselines to protect evidence.
    if results['old_fusion']['hits'][12]!=4893 or results[BASE]['hits'][12]!=5278:
        raise ValueError('original results did not reproduce')
    dates=sorted(byday)
    rng=np.random.default_rng(224467)
    draws=rng.integers(0,len(dates),size=(2000,len(dates)),dtype=np.int32)
    nday=np.array([byday[d][BASE]['n'] for d in dates])
    ci={}
    for name in results:
        if name==BASE:continue
        diffs=np.array([byday[d][name]['delta12'] for d in dates])
        boot=diffs[draws].sum(axis=1)/nday[draws].sum(axis=1)
        ci[name]={'nominal_95_pp':(100*np.quantile(boot,[.025,.975])).tolist(),
                  'delta_pp':100*diffs.sum()/scored}
    report={'version':VERSION,'status':'historical_exploratory_only','races':scored,'days':len(dates),
            'not_new_untouched_holdout':True,'point_in_time_line_verified':False,
            'preclose_odds_available':False,'return_rate':None,
            'results':{name:{'hits':{str(k):r['hits'][k] for k in POINTS},
                        'hit_rates':{str(k):r['hits'][k]/scored for k in POINTS},
                        'nll':r['nll']/scored} for name,r in results.items()},
            'rescued_lost_vs_anchor':dict(rescued),
            'nominal_unadjusted_day_bootstrap_95':ci,
            'segments':{seg:{arm:{'races':c['count'],'hit12':c['hits12']/c['count']} for arm,c in arms.items()} for seg,arms in segments.items()},
            'official_result_exclusions':dict(excluded),'prediction_manifest_sha256':digest(out/'prediction_manifest.json'),
            'publication_authorized':False,'production_changed':False,
            'limitations':['Previously inspected historical year 2025-10-09 to 2026-10-08',
                           'Unverified historic line source and timing, no claim of preclose reproducibility',
                           'Multiple exploratory variants without family-wise correction',
                           'Cannot assess >50x/>100x or ROI without frozen preclose odds']}
    save(out/'results.json',report)
    print('COMPATIBILITY_RESULTS '+json.dumps({k:v['hit_rates'] for k,v in report['results'].items()},ensure_ascii=False),flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--stage',choices=('predict','assess'),required=True)
    a.add_argument('--repair-dir',type=Path,default=Path('research/logic_compatibility/repair'))
    a.add_argument('--scenario-dir',type=Path,default=Path('research/logic_compatibility/scenario'))
    a.add_argument('--out-dir',type=Path,default=Path('research/logic_compatibility/output'))
    args=a.parse_args()
    globals()[args.stage](args.repair_dir,args.scenario_dir,args.out_dir)


if __name__=='__main__':main()
