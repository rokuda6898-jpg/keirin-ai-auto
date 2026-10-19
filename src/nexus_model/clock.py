"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
JST=timezone(timedelta(hours=9))
METRICS=['log_starts', 'log_valid', 'win_rate', 'podium_rate', 'mean_finish', 'score_relative_form', 'accident_rate', 'invalid_finish_rate']

def number(x):
    try:return float(x) if x is not None and math.isfinite(float(x)) else None
    except (TypeError,ValueError):return None


def time_vector(card):
    names=['start_hour_jst','start_hour_sin','start_hour_cos','morning_before12','evening17to20','late20plus','day_of_week','schedule_day','race_number']
    start=card['startAt'];close=card['closeAt'];valid=False
    if isinstance(start,(int,float)) and isinstance(close,(int,float)):
        moment=datetime.fromtimestamp(start,JST);valid=moment.date().isoformat()==card['date'] and 0<start-close<=3600
    hour=(moment.hour+moment.minute/60) if valid else np.nan
    return [hour,math.sin(hour*math.pi/12),math.cos(hour*math.pi/12),float(hour<12) if valid else np.nan,
        float(17<=hour<20) if valid else np.nan,float(hour>=20) if valid else np.nan,
        date.fromisoformat(card['date']).weekday(),card['schedule_day'],card['race_number']],names,valid


def profile(events,day):
    events=[(d,v) for d,v in events if d<day]
    vals=[];names=[];means={}
    for w in [14,30,90]:
        e=[v for d,v in events if day-w<=d<day];valid=[v for v in e if v['finish'] is not None]
        def avg(k):
            z=[v[k] for v in e if v[k] is not None];return sum(z)/len(z) if z else np.nan
        z=[math.log1p(len(e)),math.log1p(len(valid)),sum(v['rank']==1 for v in e)/len(e) if e else np.nan,
            sum(1<=v['rank']<=3 for v in e)/len(e) if e else np.nan,avg('finish'),avg('residual'),avg('accident'),
            sum(v['finish'] is None for v in e)/len(e) if e else np.nan]
        means[w]=z;vals+=z;names += [f'd{w}_{k}' for k in METRICS]
    vals += [float(day-events[-1][0]) if events else np.nan,
        means[14][5]-means[90][5],means[30][5]-means[90][5],means[14][3]-means[90][3],means[30][3]-means[90][3]]
    names += ['days_since_last_observed_start','d14_vs90_score_form','d30_vs90_score_form','d14_vs90_podium','d30_vs90_podium']
    return vals,names

