"""Feature preparation with a frozen, pre-holdout result reference."""
import numpy as np
import pandas as pd
from common import FEATURE_COLS, add_categorical_codes, add_strength_features

PRIOR_COLUMNS=('player_prior_races','player_prior_win_rate','player_prior_place2_rate',
               'player_prior_place3_rate','player_prior_avg_finish','player_prior_days_since_last_race')
FEATURES=[c for c in FEATURE_COLS if not c.startswith(('meeting_','h2h_','line_pair_',
    'player_elo','player_recent','player_form','player_prior_strength')) and c!='odds_win']+['line_information_missing']


def mask_outcomes(frame):
    forbidden=[c for c in frame if c.startswith(('result_','actual_','official_')) or
               c in ('finish_pos','is_dnf','payout','payout_per_100yen','odds_win')]
    out=frame.drop(columns=forbidden,errors='ignore').copy()
    out['finish_pos']=np.nan
    return out


def daily_priors(frame):
    """Training covariates never use the same day's or a later day's results."""
    valid=pd.to_numeric(frame['finish_pos'],errors='coerce')
    field=frame.groupby('race_id').race_id.transform('size')
    observed=valid.between(1,field)&valid.mod(1).eq(0)
    valid=valid.where(observed)
    work=pd.DataFrame({'player_id':frame.player_id.astype(str),'date':frame.date.astype(str),
        'n':observed.astype(float),'win':valid.eq(1).astype(float),'p2':valid.le(2).astype(float),
        'p3':valid.le(3).astype(float),'finish':valid.fillna(0)})
    daily=work.groupby(['player_id','date'],as_index=False).sum().sort_values(['player_id','date'])
    columns=['n','win','p2','p3','finish']
    sums=daily.groupby('player_id')[columns].cumsum()
    prior=sums-daily[columns]
    derived=daily[['player_id','date']].copy()
    derived['player_prior_races']=prior.n
    for output,col in (('win_rate','win'),('place2_rate','p2'),('place3_rate','p3'),('avg_finish','finish')):
        derived['player_prior_'+output]=prior[col]/prior.n.replace(0,np.nan)
    previous=daily.groupby('player_id').date.shift()
    derived['player_prior_days_since_last_race']=(pd.to_datetime(daily.date)-pd.to_datetime(previous)).dt.days
    enriched=mask_outcomes(frame).drop(columns=list(PRIOR_COLUMNS),errors='ignore').merge(derived,on=['player_id','date'],how='left',validate='many_to_one')
    frozen=daily[['player_id','date']].copy();frozen[columns]=sums
    frozen=frozen.groupby('player_id',sort=False).tail(1).set_index('player_id')
    reference={str(pid):{'last_date':r.date,'n':float(r.n),'win':float(r.win),'p2':float(r.p2),
        'p3':float(r.p3),'finish':float(r.finish)} for pid,r in frozen.iterrows()}
    return enriched,reference


def frozen_priors(frame,reference):
    out=mask_outcomes(frame)
    rows=[reference.get(str(pid),{}) for pid in out.player_id]
    n=np.array([r.get('n',0) for r in rows],dtype=float)
    out['player_prior_races']=n
    for output,col in (('win_rate','win'),('place2_rate','p2'),('place3_rate','p3'),('avg_finish','finish')):
        values=np.array([r.get(col,np.nan) for r in rows],dtype=float)
        out['player_prior_'+output]=np.divide(values,n,out=np.full_like(n,np.nan),where=n>0)
    dates=pd.to_datetime([r.get('last_date') for r in rows])
    out['player_prior_days_since_last_race']=(pd.to_datetime(out.date).to_numpy()-dates.to_numpy())/np.timedelta64(1,'D')
    return out


def matrix(frame,config=None):
    # Impute these two derived-strength inputs using training values only.
    out=mask_outcomes(frame)
    if config is None:
        medians={c:float(pd.to_numeric(out.get(c,pd.Series(np.nan,index=out.index)),errors='coerce').median()) for c in ('score','recent_avg_finish')}
        medians={k:v if np.isfinite(v) else 0. for k,v in medians.items()}
    else:medians=config['strength_medians']
    for c,v in medians.items():out[c]=pd.to_numeric(out.get(c,pd.Series(np.nan,index=out.index)),errors='coerce').fillna(v)
    verified=out.get('line_verification_status',pd.Series('',index=out.index)).eq('verified')
    for c in ('line_id','line_position','line_size','is_line_leader','number_of_lines'):
        out[c]=pd.to_numeric(out.get(c,pd.Series(np.nan,index=out.index)),errors='coerce').where(verified)
    out=add_strength_features(add_categorical_codes(out))
    out['line_information_missing']=(~verified).astype(float)
    for c in FEATURES:
        if c not in out:out[c]=np.nan
        if c!='line_information_missing' and c.startswith(('line_','strength_vs_line','is_line_leader','number_of_lines','lines_per_field','other_line_')):
            out[c]=out[c].where(verified)
    X=out[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    if config is None:
        fills=X.median().fillna(0).to_dict();config={'features':FEATURES,'fills':fills,'strength_medians':medians}
    if config['features']!=FEATURES:raise ValueError('feature schema mismatch')
    return X.fillna(config['fills']),config
