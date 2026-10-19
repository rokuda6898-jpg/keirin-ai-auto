"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
from . import team as m,role as retained
FLAGS=['spurtSucceeded', 'thrustSucceeded', 'leftBehind', 'splitLine', 'snatchSucceeded', 'competeSucceeded']
MET=['win', 'podium', 'finish_percentile', 'accident', 'invalid_finish', 'back', 'positive_spurtSucceeded', 'positive_thrustSucceeded', 'positive_leftBehind', 'positive_splitLine', 'positive_snatchSucceeded', 'positive_competeSucceeded']
HN=['pooled_d14_log_records', 'pooled_d14_win', 'pooled_d14_podium', 'pooled_d14_finish_percentile', 'pooled_d14_accident', 'pooled_d14_invalid_finish', 'pooled_d14_back', 'pooled_d14_positive_spurtSucceeded', 'pooled_d14_positive_thrustSucceeded', 'pooled_d14_positive_leftBehind', 'pooled_d14_positive_splitLine', 'pooled_d14_positive_snatchSucceeded', 'pooled_d14_positive_competeSucceeded', 'pooled_d30_log_records', 'pooled_d30_win', 'pooled_d30_podium', 'pooled_d30_finish_percentile', 'pooled_d30_accident', 'pooled_d30_invalid_finish', 'pooled_d30_back', 'pooled_d30_positive_spurtSucceeded', 'pooled_d30_positive_thrustSucceeded', 'pooled_d30_positive_leftBehind', 'pooled_d30_positive_splitLine', 'pooled_d30_positive_snatchSucceeded', 'pooled_d30_positive_competeSucceeded', 'pooled_d90_log_records', 'pooled_d90_win', 'pooled_d90_podium', 'pooled_d90_finish_percentile', 'pooled_d90_accident', 'pooled_d90_invalid_finish', 'pooled_d90_back', 'pooled_d90_positive_spurtSucceeded', 'pooled_d90_positive_thrustSucceeded', 'pooled_d90_positive_leftBehind', 'pooled_d90_positive_splitLine', 'pooled_d90_positive_snatchSucceeded', 'pooled_d90_positive_competeSucceeded', 'pooled_d365_log_records', 'pooled_d365_win', 'pooled_d365_podium', 'pooled_d365_finish_percentile', 'pooled_d365_accident', 'pooled_d365_invalid_finish', 'pooled_d365_back', 'pooled_d365_positive_spurtSucceeded', 'pooled_d365_positive_thrustSucceeded', 'pooled_d365_positive_leftBehind', 'pooled_d365_positive_splitLine', 'pooled_d365_positive_snatchSucceeded', 'pooled_d365_positive_competeSucceeded', 'current_role_d90_log_records', 'current_role_d90_win', 'current_role_d90_podium', 'current_role_d90_cross_line_second', 'current_role_d90_positive_spurtSucceeded', 'current_role_d90_positive_thrustSucceeded', 'current_role_d90_positive_leftBehind', 'current_role_d90_positive_splitLine', 'current_role_d90_positive_snatchSucceeded', 'current_role_d90_positive_competeSucceeded', 'current_role_d90_shrunken_win', 'current_role_d90_shrunken_podium', 'current_role_d365_log_records', 'current_role_d365_win', 'current_role_d365_podium', 'current_role_d365_cross_line_second', 'current_role_d365_positive_spurtSucceeded', 'current_role_d365_positive_thrustSucceeded', 'current_role_d365_positive_leftBehind', 'current_role_d365_positive_splitLine', 'current_role_d365_positive_snatchSucceeded', 'current_role_d365_positive_competeSucceeded', 'current_role_d365_shrunken_win', 'current_role_d365_shrunken_podium', 'days_since_observed_start', 'days_since_same_role', 'verified_role_category', 'known_role_support', 'line_front_score_gap', 'line_front_form14_gap', 'other_line_best_score_gap', 'other_line_best_form14_gap', 'retained_self_front_d90_records', 'retained_self_front_d90_spurtSucceeded', 'retained_self_front_d90_thrustSucceeded', 'retained_self_front_d90_leftBehind', 'retained_self_front_d90_splitLine', 'retained_self_front_d90_snatchSucceeded', 'retained_self_front_d90_competeSucceeded', 'retained_self_front_d365_records', 'retained_self_front_d365_spurtSucceeded', 'retained_self_front_d365_thrustSucceeded', 'retained_self_front_d365_leftBehind', 'retained_self_front_d365_splitLine', 'retained_self_front_d365_snatchSucceeded', 'retained_self_front_d365_competeSucceeded', 'retained_self_follower_d90_records', 'retained_self_follower_d90_spurtSucceeded', 'retained_self_follower_d90_thrustSucceeded', 'retained_self_follower_d90_leftBehind', 'retained_self_follower_d90_splitLine', 'retained_self_follower_d90_snatchSucceeded', 'retained_self_follower_d90_competeSucceeded', 'retained_self_follower_d365_records', 'retained_self_follower_d365_spurtSucceeded', 'retained_self_follower_d365_thrustSucceeded', 'retained_self_follower_d365_leftBehind', 'retained_self_follower_d365_splitLine', 'retained_self_follower_d365_snatchSucceeded', 'retained_self_follower_d365_competeSucceeded', 'retained_self_front_days_since', 'retained_self_follower_days_since', 'retained_self_verified_front', 'retained_self_verified_follower', 'retained_partner_front_d90_records', 'retained_partner_front_d90_spurtSucceeded', 'retained_partner_front_d90_thrustSucceeded', 'retained_partner_front_d90_leftBehind', 'retained_partner_front_d90_splitLine', 'retained_partner_front_d90_snatchSucceeded', 'retained_partner_front_d90_competeSucceeded', 'retained_partner_front_d365_records', 'retained_partner_front_d365_spurtSucceeded', 'retained_partner_front_d365_thrustSucceeded', 'retained_partner_front_d365_leftBehind', 'retained_partner_front_d365_splitLine', 'retained_partner_front_d365_snatchSucceeded', 'retained_partner_front_d365_competeSucceeded', 'retained_partner_follower_d90_records', 'retained_partner_follower_d90_spurtSucceeded', 'retained_partner_follower_d90_thrustSucceeded', 'retained_partner_follower_d90_leftBehind', 'retained_partner_follower_d90_splitLine', 'retained_partner_follower_d90_snatchSucceeded', 'retained_partner_follower_d90_competeSucceeded', 'retained_partner_follower_d365_records', 'retained_partner_follower_d365_spurtSucceeded', 'retained_partner_follower_d365_thrustSucceeded', 'retained_partner_follower_d365_leftBehind', 'retained_partner_follower_d365_splitLine', 'retained_partner_follower_d365_snatchSucceeded', 'retained_partner_follower_d365_competeSucceeded', 'retained_partner_front_days_since', 'retained_partner_follower_days_since', 'retained_partner_verified_front', 'retained_partner_verified_follower']
BASE_HN_COUNT=84
STAGES=['予選', '一般', '選抜', '準決', '決勝', '特選', 'チャレンジ', '新人']
REL=['known_same_line', 'known_cross_line', 'role_pair_category', 'self_vs_first_role_win', 'self_vs_first_role_podium', 'self_vs_first_pooled14_win', 'self_vs_first_pooled30_win', 'first_front_self_third', 'self_behind_first', 'other_line_relative_strength', 'both_short_history_sparse', 'self_long_rest']

def stage_type(name):
    if '準々' in name:return 9
    if '準決' in name:return 3
    if '決勝' in name or name in ['グランプリ','ガールズグランプリ','ヤンググランプリ']:return 4
    if '特秀' in name:return 10
    if '優秀' in name:return 11
    for k,word in enumerate(STAGES):
        if word in name:return k
    return 12


def roles(d,i):
    r=d['rows'][i];n=r['active_cars'];li=d['line_ids'][i,:n];po=d['line_positions'][i,:n];out=[]
    for j in range(n):
        if not r['lines_applicable']:v=6
        elif li[j]<0:v=5
        elif sum(li==li[j])==1:v=3
        elif sum((li==li[j])&(po==po[j]))>1:v=4
        else:v=0 if po[j]==0 else 1 if po[j]==1 else 2
        out.append(v)
    return out


def avg(records,k):
    a=[r[k] for r in records if r[k] is not None]
    return sum(a)/len(a) if a else np.nan


def profile(hist,rhist,day,role,last,lastrole):
    out=[];means={}
    for window in [14,30,90,365]:
        rr=[v for dt,v in hist if day-window<=dt<day]
        vals=[avg(rr,k) for k in range(len(MET))];means[window]=vals
        out += [math.log1p(len(rr))]+vals
    for window in [90,365]:
        rr=[v for dt,v in rhist if day-window<=dt<day]
        count=len(rr);rw=avg(rr,0);rp=avg(rr,1);rc=avg(rr,len(MET))
        shr=[]
        for val,prior in [(rw,means[window][0]),(rp,means[window][1])]:
            shr.append((count*val+20*prior)/(count+20) if count and math.isfinite(prior) else prior)
        out += [math.log1p(count),rw,rp,rc]+[avg(rr,k) for k in range(6,12)]+shr
    out += [day-last if last is not None else np.nan,day-lastrole if lastrole is not None else np.nan,role,float(role<5)]
    return out


def correct_context(a,names,stage):
    for k,n in enumerate(names):
        if n.startswith('context_stage_') or n.startswith('stage_'):
            word=n.split('stage_',1)[1]
            if word in STAGES:a[...,k]=float(stage==STAGES.index(word))
    return a


def names(k):
    return m.names(k)+['race_stage_category','specialist_query_category']+[f'{p}_{h}' for p in ['new_self']+[f'new_prefix{j+1}' for j in range(k)] for h in HN]+(REL if k else [])


def categorical(k):
    return [j for j,n in enumerate(names(k)) if n in ['context_venue','context_class','context_cup_grade','race_stage_category','specialist_query_category','role_pair_category'] or n.endswith('_style') or n.endswith('_verified_role_category')]


def features(d,i,cc,first=None,second=None,route=0):
    cc=np.asarray(cc,int);n=len(cc);stage=int(first is not None)+int(second is not None)
    a=correct_context(m.features(d,i,cc,first,second),m.names(stage),d['new_stage'][i])
    parts=[a,np.tile([d['new_stage'][i],route],(n,1)),d['new_extra'][i,cc]]
    for pre in [first,second]:
        if pre is not None:parts.append(d['new_extra'][i,np.asarray(pre,int)])
    if first is not None:
        first=np.asarray(first,int);li=d['line_ids'][i];pos=d['line_positions'][i];x=d['new_extra'][i,cc];y=d['new_extra'][i,first];hx={v:j for j,v in enumerate(HN)}
        known=(li[cc]>=0)&(li[first]>=0)&bool(d['rows'][i]['lines_applicable'])
        role=x[:,hx['verified_role_category']];fr=y[:,hx['verified_role_category']]
        vals=[np.where(known,li[cc]==li[first],np.nan),np.where(known,li[cc]!=li[first],np.nan),role*7+fr]
        for h in ['current_role_d365_shrunken_win','current_role_d365_shrunken_podium','pooled_d14_win','pooled_d30_win']:vals.append(x[:,hx[h]]-y[:,hx[h]])
        vals += [(fr==0)&(role==2),np.where(known,(li[cc]==li[first])&(pos[cc]>pos[first]),np.nan),x[:,hx['other_line_best_score_gap']],(x[:,0]<math.log(4))&(y[:,0]<math.log(4)),x[:,hx['days_since_observed_start']]>30]
        parts.append(np.column_stack(vals))
    out=np.column_stack(parts).astype(np.float32);assert out.shape==(n,len(names(stage)));return out


def judge_names():return m.class_names()+['race_stage_category']+[f'new_r{j}_{h}' for j in range(9) for h in HN]


def judge_row(d,i):return np.r_[correct_context(m.class_row(d,i),m.class_names(),d['new_stage'][i]),d['new_stage'][i],d['new_extra'][i].ravel()].astype(np.float32)

