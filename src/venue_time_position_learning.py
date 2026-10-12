"""Venue/class/field/hour win and podium models, trained with a frozen holdout.

Actual finish places are the labels. Missing official winner tactics are never
inferred from racer style. Research models never overwrite production or bets.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from venue_time_learning import (EXPECTED_VENUES, JST, classify, jst_hour,
                                 USED, time_band, to_races, venue_name)

VERSION = 'venue_time_positions_v1'
DIVISIONS = ('A_regular','A_challenge','S_regular','S_graded','L')
BANDS = ('morning','day','evening','late','unknown')
NUMERIC = ('score','score_centered','win_rate','place2_rate','place3_rate',
           'back_count','front_runner_count','stalker_count','deep_closer_count',
           'marker_count','recent_avg_finish','age','line_position','line_size',
           'number_of_lines','track_win_rate','track_place2_rate',
           'days_since_last_race','field')
FEATURE_SETS = {
    'national': (*NUMERIC,'division_code'),
    'venue': (*NUMERIC,'division_code','venue_code'),
    'venue_time': (*NUMERIC,'division_code','venue_code','band_code','hour_code'),
}
CATEGORICAL = {'division_code','venue_code','band_code','hour_code'}


def load_position_history(path):
    columns=set(USED)|set(NUMERIC)|{'date','race_id','venue','car_no','finish_pos',
        'line_verification_status','official_finish_pos'}
    return pd.read_csv(path,usecols=lambda name:name in columns,
                       dtype={'race_id':str,'player_id':str},low_memory=False)


def prepare(entries, asof):
    """Attach only known-before-the-race covariates to validated outcomes."""
    races=to_races(entries, asof)
    if races.empty:
        return pd.DataFrame(),races
    valid=set(races.race_id)
    frame=entries[entries.race_id.astype(str).isin(valid)].copy()
    frame['race_id']=frame.race_id.astype(str)
    frame['car_no']=pd.to_numeric(frame.car_no,errors='coerce')
    frame=frame.merge(races[['race_id','date','venue','division','tier','field','hour','band']],
                      on='race_id',how='inner',suffixes=('_original',''),validate='many_to_one')
    frame['finish_pos']=pd.to_numeric(frame.finish_pos,errors='coerce')
    frame['target_win']=frame.finish_pos.eq(1).astype(int)
    frame['target_top3']=frame.finish_pos.isin((1,2,3)).astype(int)
    for col in NUMERIC:
        if col in ('score_centered','field'):continue
        if col not in frame:frame[col]=np.nan
        frame[col]=pd.to_numeric(frame[col],errors='coerce')
    if 'line_verification_status' in frame:
        verified=frame.line_verification_status.astype(str).eq('verified')
        for col in ('line_position','line_size','number_of_lines'):
            frame[col]=frame[col].where(verified)
    frame['score_centered']=frame.score-frame.groupby('race_id').score.transform('mean')
    frame['division_code']=frame.division.map({d:i for i,d in enumerate(DIVISIONS)}).fillna(-1).astype(int)
    frame['venue_code']=frame.venue.map({v:i for i,v in enumerate(EXPECTED_VENUES)}).fillna(-1).astype(int)
    frame['band_code']=frame.band.map({b:i for i,b in enumerate(BANDS)}).fillna(len(BANDS)-1).astype(int)
    frame['hour_code']=pd.to_numeric(frame.hour,errors='coerce').fillna(24).astype(int)
    frame['race_weight']=1.0/frame.field
    return frame,races


def features(frame, name):
    result=frame.reindex(columns=list(FEATURE_SETS[name])).apply(pd.to_numeric,errors='coerce')
    return result.replace([np.inf,-np.inf],np.nan)


def fit_model(train, variant, target, max_iter=85, min_leaf=55):
    xx=features(train,variant)
    y=train[target].astype(int)
    if len(train)<100 or y.nunique()<2:
        raise ValueError(f'Insufficient training samples or labels: {len(train)}')
    # Select usable inputs on training rows only. Sklearn histograms can fail
    # with all-null/constant columns under different NumPy builds.
    retained=[c for c in xx if xx[c].nunique(dropna=True)>1]
    if not retained:
        raise ValueError('No nonconstant pre-race features available for fitting')
    estimator=HistGradientBoostingClassifier(max_iter=max_iter,
        max_leaf_nodes=15,min_samples_leaf=min_leaf,l2_regularization=2.0,
        learning_rate=.06,early_stopping=False,random_state=20261013,
        categorical_features=[i for i,c in enumerate(retained) if c in CATEGORICAL])
    model=Pipeline([
        ('retain_train_only_features', ColumnTransformer(
            [('used','passthrough',retained)],remainder='drop',sparse_threshold=0)),
        ('estimator',estimator),
    ])
    model.fit(xx,y,estimator__sample_weight=train.race_weight.astype(float))
    return model


def score_holdout(test, predicted):
    """Race-level evaluation, keeping one real winner and three placed finishers."""
    frame=test[['race_id','date','venue','tier','division','field','band',
                'car_no','target_win','target_top3']].copy()
    for name, scores in predicted.items():
        frame[f'win_{name}']=np.clip(scores['win'],1e-8,1)
        frame[f'top3_{name}']=np.clip(scores['top3'],1e-8,1)
    out=[]
    for rid,group in frame.groupby('race_id',sort=False):
        if group.target_win.sum()!=1 or group.target_top3.sum()!=3:continue
        record={key:group.iloc[0][key] for key in ('race_id','date','venue','tier','division','field','band')}
        for name in FEATURE_SETS:
            chances=group[f'win_{name}'].to_numpy(dtype=float)
            odds=chances/sum(chances)
            actual=group.target_win.to_numpy(dtype=bool)
            record[f'win_logloss_{name}']=-math.log(max(1e-12,float(odds[actual][0])))
            record[f'top1_hit_{name}']=int(group.iloc[int(np.argmax(chances))].target_win==1)
            podium=group[f'top3_{name}'].to_numpy(dtype=float)
            picks=np.argsort(-podium,kind='stable')[:3]
            record[f'top3_recall_{name}']=float(group.iloc[picks].target_top3.sum()/3)
        record['date']=str(pd.Timestamp(record['date']).date())
        out.append(record)
    return pd.DataFrame(out)


def summarise_eval(results, train_races, test_races):
    metrics={}
    for name in FEATURE_SETS:
        metrics[name]={
            'top1_hit_rate':float(results[f'top1_hit_{name}'].mean()) if len(results) else None,
            'winner_logloss':float(results[f'win_logloss_{name}'].mean()) if len(results) else None,
            'top3_candidate_recall':float(results[f'top3_recall_{name}'].mean()) if len(results) else None}
    ci={}
    if len(results):
        daily=results.assign(
            win_improvement=results.win_logloss_national-results.win_logloss_venue_time,
            top1_improvement=results.top1_hit_venue_time-results.top1_hit_national,
            top3_improvement=results.top3_recall_venue_time-results.top3_recall_national,
        ).groupby('date')[['win_improvement','top1_improvement','top3_improvement']].mean()
        if len(daily)>=15:
            rng=np.random.default_rng(20261013)
            for col in daily:
                array=daily[col].to_numpy(dtype=float)
                bootstrap=[float(rng.choice(array,len(array),replace=True).mean()) for _ in range(500)]
                ci[col]=[float(z) for z in np.quantile(bootstrap,[.025,.975])]
    return {'trained_races':train_races, 'holdout_races':test_races,
            'evaluated_races':int(len(results)), 'metrics':metrics,
            'venue_time_minus_national_daily_bootstrap_ci95':ci,
            'notes':'1着的中率と3着内候補の再現率。3連単の的中率・回収率ではない。'}


def today_shadow(today, models):
    if not {'race_id','date','venue','car_no'}.issubset(today):return pd.DataFrame()
    data=today.copy()
    data['race_id']=data.race_id.astype(str)
    data['car_no']=pd.to_numeric(data.car_no,errors='coerce')
    data['field']=data.groupby('race_id').race_id.transform('size')
    data['venue']=data.venue.map(venue_name)
    classifications=data.apply(classify,axis=1)
    data['tier']=[x[0] for x in classifications]
    data['division']=[x[1] for x in classifications]
    data['hour']=data.get('start_at',pd.Series(np.nan,index=data.index)).map(jst_hour)
    data['band']=data.hour.map(time_band)
    for col in NUMERIC:
        if col in ('score_centered','field'):continue
        data[col]=pd.to_numeric(data.get(col,pd.Series(np.nan,index=data.index)),errors='coerce')
    verified=data.get('line_verification_status',pd.Series('',index=data.index)).eq('verified')
    for col in ('line_position','line_size','number_of_lines'):
        data[col]=data[col].where(verified)
    data['score_centered']=data.score-data.groupby('race_id').score.transform('mean')
    data['division_code']=data.division.map({d:i for i,d in enumerate(DIVISIONS)}).fillna(-1).astype(int)
    data['venue_code']=data.venue.map({v:i for i,v in enumerate(EXPECTED_VENUES)}).fillna(-1).astype(int)
    data['band_code']=data.band.map({b:i for i,b in enumerate(BANDS)}).fillna(len(BANDS)-1).astype(int)
    data['hour_code']=data.hour.fillna(24).astype(int)
    chosen=models['venue_time']
    xx=features(data,'venue_time')
    data['p_first_raw']=chosen['win'].predict_proba(xx)[:,1]
    data['p_top3_raw']=chosen['top3'].predict_proba(xx)[:,1]
    norm=data.groupby('race_id').p_first_raw.transform('sum')
    data['p_first']=data.p_first_raw/norm.replace(0,np.nan)
    data['first_rank']=data.groupby('race_id').p_first.rank(ascending=False,method='first').astype(int)
    data['top3_rank']=data.groupby('race_id').p_top3_raw.rank(ascending=False,method='first').astype(int)
    data['is_shadow']=True
    return data[['race_id','date','venue','tier','division','field','band','hour','car_no',
                 'p_first','first_rank','p_top3_raw','top3_rank','is_shadow']]


def run(history, out_dir, asof, today_path=None, max_iter=85, min_leaf=55, min_history_races=30000):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    entries=load_position_history(history)
    prepared,races=prepare(entries,asof)
    cutoff=pd.Timestamp(asof)-pd.Timedelta(days=365)
    train_ids=set(races.loc[races.date.lt(cutoff),'race_id'])
    test_ids=set(races.loc[races.date.ge(cutoff),'race_id'])
    if train_ids & test_ids:raise ValueError('Train/holdout race overlap')
    if len(train_ids)<min_history_races or len(test_ids)<200:
        raise ValueError(f'Insufficient strictly independent race split: {len(train_ids)} train / {len(test_ids)} test')
    train=prepared[prepared.race_id.isin(train_ids)].copy()
    test=prepared[prepared.race_id.isin(test_ids)].copy()
    fitted={}; predicted={}
    for variant in FEATURE_SETS:
        fitted[variant]={};predicted[variant]={}
        for label in ('win','top3'):
            target='target_win' if label=='win' else 'target_top3'
            model=fit_model(train,variant,target,max_iter=max_iter,min_leaf=min_leaf)
            fitted[variant][label]=model
            predicted[variant][label]=model.predict_proba(features(test,variant))[:,1]
            joblib.dump(model,out/f'{variant}_{label}.joblib')
            print(f'FITTED {variant} {label} train_entries={len(train)} test_entries={len(test)}',flush=True)
    scored=score_holdout(test,predicted)
    summary=summarise_eval(scored,len(train_ids),len(test_ids))
    summary.update({'version':VERSION,'asof_exclusive':str(pd.Timestamp(asof).date()),
        'holdout_start':str(cutoff.date()),'venues_in_train':int(races[races.race_id.isin(train_ids)].venue.nunique()),
        'venues_in_holdout':int(races[races.race_id.isin(test_ids)].venue.nunique()),
        'feature_sets':{k:list(v) for k,v in FEATURE_SETS.items()},
        'training_policy':'single frozen cutoff, no test labels or test fit; equal weight per race',
        'source_limitation':'historical source may not be a saved true pre-close snapshot; study only',
        'official_tactics_coverage':'not used; winner tactic was not inferred from style',
        'commercial_use_status':'review_required',
        'production_modified':False,'purchase_authorized':False,'return_rate_evaluated':False})
    by_venue=scored.groupby(['venue','division'],dropna=False).agg(
        races=('race_id','size'),national_top1=('top1_hit_national','mean'),
        venue_top1=('top1_hit_venue','mean'),context_top1=('top1_hit_venue_time','mean'),
        national_top3_recall=('top3_recall_national','mean'),
        context_top3_recall=('top3_recall_venue_time','mean')).reset_index()
    by_venue['context_top1_lift']=by_venue.context_top1-by_venue.national_top1
    by_venue['adequate_sample']=by_venue.races.ge(100)
    by_venue.to_csv(out/'venue_class_holdout.csv',index=False)
    by_time=scored.groupby(['venue','division','band'],dropna=False).agg(
        races=('race_id','size'),national_top1=('top1_hit_national','mean'),
        context_top1=('top1_hit_venue_time','mean')).reset_index()
    by_time['context_top1_lift']=by_time.context_top1-by_time.national_top1
    by_time['adequate_sample']=by_time.races.ge(100)
    by_time.to_csv(out/'venue_class_time_holdout.csv',index=False)
    scored.to_csv(out/'race_holdout_evidence.csv',index=False)
    # Refit today's shadow on ALL prior races, after frozen-holdout scoring.
    # Never evaluate the past holdout using these fully refitted models.
    full_shadow={'venue_time':{}}
    for label in ('win','top3'):
        target='target_win' if label=='win' else 'target_top3'
        fresh=fit_model(prepared,'venue_time',target,max_iter=max_iter,min_leaf=min_leaf)
        full_shadow['venue_time'][label]=fresh
        joblib.dump(fresh,out/f'venue_time_full_{label}.joblib')
    summary['full_history_shadow_trained_races']=int(len(races))
    if today_path and Path(today_path).is_file():
        preview=today_shadow(pd.read_csv(today_path,dtype={'race_id':str}),full_shadow)
    else:preview=pd.DataFrame()
    preview.to_csv(out/'today_position_shadow.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='feature_sets'},ensure_ascii=False),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--history',default='data/raw/history.csv')
    parser.add_argument('--today',default='data/raw/today_entries.csv')
    parser.add_argument('--asof',default=datetime.now(JST).strftime('%Y-%m-%d'))
    parser.add_argument('--out-dir',default='research/venue_time_positions')
    parser.add_argument('--max-iter',type=int,default=85)
    parser.add_argument('--min-leaf',type=int,default=55)
    parser.add_argument('--min-races',type=int,default=30000)
    args=parser.parse_args()
    run(args.history,args.out_dir,args.asof,args.today,args.max_iter,args.min_leaf,args.min_races)


if __name__=='__main__':main()
