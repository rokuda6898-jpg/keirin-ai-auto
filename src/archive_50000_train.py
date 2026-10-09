"""Fit two distinct standalone models on 50,000 REAL prior race outcomes.

The central cache is a retrospective archive, not a 3-way odds or
intermediate-path database. Neither market-odds residuals nor observed
race-stage transitions can be learned from it. Nothing here places bets.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from archive_50000 import (FEATURE_NAMES, NAMES, VERSION, body_digest,
                           probabilities, rider_vectors)

TARGET = 50000
MIN_HOLDOUT = 1000

def eligible_races(frame, asof):
    """Complete real fields, one known finish each at positions 1, 2 and 3."""
    dates = pd.to_datetime(frame['date'],errors='coerce')
    before = dates.lt(pd.Timestamp(asof))
    frame = frame.loc[before].copy()
    cars = pd.to_numeric(frame['car_no'],errors='coerce')
    frame['car_no'] = cars
    positions = pd.to_numeric(frame['finish_pos'],errors='coerce')
    frame['_p'] = positions
    frame['_one'] = positions.eq(1).astype(int)
    frame['_two'] = positions.eq(2).astype(int)
    frame['_three'] = positions.eq(3).astype(int)
    frame['_valid_car'] = cars.between(1,9) & cars.mod(1).eq(0)
    group = frame.groupby('race_id',sort=False)
    summary = group.agg(
        count=('car_no','size'),distinct_cars=('car_no','nunique'),
        distinct_players=('player_id','nunique'),days=('date','nunique'),
        good_cars=('_valid_car','sum'),ones=('_one','sum'),
        twos=('_two','sum'),threes=('_three','sum'),
        day=('date','first'))
    good = (summary['count'].between(3,9)
            & summary['count'].eq(summary['distinct_cars'])
            & summary['count'].eq(summary['distinct_players'])
            & summary['count'].eq(summary['good_cars'])
            & summary['days'].eq(1)
            & summary['ones'].eq(1)&summary['twos'].eq(1)&summary['threes'].eq(1))
    if 'entries_number' in frame:
        expected = pd.to_numeric(frame['entries_number'],errors='coerce')
        size = group['race_id'].transform('size')
        mismatch = expected.notna() & expected.ne(size)
        bad_ids = set(frame.loc[mismatch,'race_id'].astype(str))
        good.loc[good.index.isin(bad_ids)] = False
    valid = summary.loc[good,['day']].copy()
    valid['race_id'] = valid.index.astype(str)
    valid.sort_values(['day','race_id'],inplace=True)
    valid_ids = set(valid['race_id'])
    return frame[frame.race_id.isin(valid_ids)].copy(),valid

def chronological_split(valid, target=TARGET):
    if len(valid) < target+MIN_HOLDOUT:
        raise ValueError(f'insufficient real races: {len(valid)}; need at least {target+MIN_HOLDOUT}')
    train = valid.iloc[:target]
    last_day = train['day'].iloc[-1]
    heldout = valid.iloc[target:]
    heldout = heldout.loc[heldout['day'] > last_day]
    if len(heldout) < MIN_HOLDOUT:
        raise ValueError('not enough later distinct-day evaluation races')
    if set(train.race_id) & set(heldout.race_id):
        raise ValueError('race overlap')
    return list(train.race_id),list(heldout.race_id),str(last_day),str(heldout['day'].iloc[0])

def design(frame, train_ids):
    """Actual place ranks only; unplaced rider order is deliberately unknown."""
    X, weight, labels = [], [], [[],[],[]]
    px, py, pw = [], [], []
    selected = frame[frame.race_id.isin(set(train_ids))]
    for rid,race in selected.groupby('race_id',sort=False):
        records = race.to_dict('records')
        fs = rider_vectors(records)
        ranks = {int(row['car_no']):int(row['_p']) for row in records
                 if int(row['_p']) in (1,2,3)}
        if len(ranks) != 3:
            raise ValueError('incomplete podium after validation')
        for row in records:
            car = int(row['car_no'])
            X.append(fs[car])
            for p in (1,2,3):
                labels[p-1].append(int(ranks.get(car)!=None and ranks[car]==p))
            weight.append(1/len(records))
        pairs = [(a,b) for a,b in combinations(sorted(fs),2) if a in ranks or b in ranks]
        for a,b in pairs:
            px.append([u-v for u,v in zip(fs[a],fs[b])])
            py.append(int(ranks.get(a,99) < ranks.get(b,99)))
            pw.append(1/len(pairs))
    return (np.asarray(X,dtype=np.float32),
            [np.asarray(v,dtype=np.int8) for v in labels],
            np.asarray(weight,dtype=np.float32),
            np.asarray(px,dtype=np.float32),
            np.asarray(py,dtype=np.int8),np.asarray(pw,dtype=np.float32))

def train_models(frame, train_ids):
    X,labels,weights,px,py,pw = design(frame,train_ids)
    if len(X) == 0 or len(px) == 0 or len(np.unique(py))<2:
        raise ValueError('training lacks sufficient actual placements')
    betas,intercepts = [],[]
    for idx,y in enumerate(labels):
        classifier = LogisticRegression(C=3,max_iter=120,solver='lbfgs',random_state=2026+idx)
        classifier.fit(X,y,sample_weight=weights)
        betas.append([float(c) for c in classifier.coef_[0]])
        intercepts.append(float(classifier.intercept_[0]))
        print(f'ARCHIVE_50K fit-position-{idx+1} examples={len(X)}',flush=True)
    pairs = LogisticRegression(C=3,max_iter=120,solver='lbfgs',
                               fit_intercept=False,random_state=2026)
    pairs.fit(px,py,sample_weight=pw)
    print(f'ARCHIVE_50K fit-pairwise examples={len(px)}',flush=True)
    return {'position_betas':betas,'position_intercepts':intercepts,
            'pairwise_beta':[float(x) for x in pairs.coef_[0]]}

def evaluate(frame, holdout_ids, model, limit=3000):
    chosen = holdout_ids[-limit:]
    evaluated = {m:{'races':0,'top1_hits':0,'top6_hits':0,'top12_hits':0,
                    'first_hits':0,'first_pair_hits':0,'nll_total':0.0}
                 for m in NAMES}
    for rid,race in frame[frame.race_id.isin(set(chosen))].groupby('race_id',sort=False):
        actual=sorted(((int(row['car_no']),int(row['_p'])) for row in race.to_dict('records')
                       if int(row['_p']) in (1,2,3)),key=lambda x:x[1])
        if len(actual)!=3:continue
        answer='-'.join(str(c) for c,_ in actual)
        records=race.to_dict('records')
        for method in NAMES:
            dist=probabilities(method,model,records)
            order=sorted(dist,key=lambda k:(-dist[k],k))
            metrics=evaluated[method];metrics['races']+=1
            metrics['top1_hits']+=int(answer==order[0])
            metrics['top6_hits']+=int(answer in order[:6])
            metrics['top12_hits']+=int(answer in order[:12])
            metrics['first_hits']+=int(answer.split('-')[0]==order[0].split('-')[0])
            metrics['first_pair_hits']+=int(answer.split('-')[:2]==order[0].split('-')[:2])
            metrics['nll_total']+=-math.log(max(1e-15,dist[answer]))
    for metrics in evaluated.values():
        n=metrics['races']
        for k in ('top1_hits','top6_hits','top12_hits','first_hits','first_pair_hits'):
            metrics[k.replace('_hits','_rate')]=metrics[k]/n if n else None
        metrics['mean_nll']=metrics.pop('nll_total')/n if n else None
    return evaluated

def run(history_path, output_model, output_report, asof='2026-10-09'):
    if not history_path.is_file():
        raise FileNotFoundError('central REAL historical race archive is required; no synthetic fallback')
    frame=pd.read_csv(history_path,dtype={'race_id':str,'player_id':str},low_memory=False)
    frame,valid=eligible_races(frame,asof)
    train_ids,holdout_ids,end_day,start_holdout=chronological_split(valid)
    prior=frame[frame.race_id.isin(set(train_ids))]
    if pd.to_datetime(prior['date']).max()>=pd.Timestamp(asof):
        raise ValueError('future training label')
    params=train_models(frame,train_ids)
    checksum=hashlib.sha256(history_path.read_bytes()).hexdigest()
    frozen={'version':VERSION,'trained_races':len(train_ids),'feature_names':list(FEATURE_NAMES),
            'training_start':str(valid['day'].iloc[0]),'training_end':end_day,
            'holdout_start':start_holdout,'asof_exclusive':asof,
            'train_holdout_overlap':False,'archive_sha256':checksum,
            **params,'purchase_authorized':False,'ceo_integration':False,
            'training_source':'real_retrospective_central_archive_not_frozen_preclose_quotes',
            'history_odds_training':False,'observed_path_training':False}
    frozen['model_sha256']=body_digest(frozen)
    holdout=evaluate(frame,holdout_ids,frozen)
    report={'version':VERSION,'trained_races':len(train_ids),'heldout_available_races':len(holdout_ids),
            'heldout_evaluated_races':next(iter(holdout.values()))['races'],
            'train_end':end_day,'holdout_start':start_holdout,
            'historical_valid_races':len(valid),'excluded_races':int(frame.race_id.nunique()-len(valid)),
            'metrics':holdout,'model_sha256':frozen['model_sha256'],
            'archive_sha256':checksum,'evaluation':'chronological_retrospective_holdout',
            'historical_3way_odds_available':False,'observed_start_bell_back_paths_available':False,
            'auto_ceo_promotion':False,'purchase_authorized':False}
    output_model.parent.mkdir(parents=True,exist_ok=True)
    output_report.parent.mkdir(parents=True,exist_ok=True)
    output_model.write_text(json.dumps(frozen,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    output_report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('ARCHIVE_50K_COMPLETE '+json.dumps({k:v for k,v in report.items() if k not in ('metrics','archive_sha256')},ensure_ascii=False),flush=True)
    return report

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--history',type=Path,default=Path('data/raw/history.csv'))
    ap.add_argument('--model',type=Path,default=Path('models/archive_50000.json'))
    ap.add_argument('--report',type=Path,default=Path('outputs/company/archive_50000_report.json'))
    ap.add_argument('--asof',default='2026-10-09')
    args=ap.parse_args()
    run(args.history,args.model,args.report,args.asof)
