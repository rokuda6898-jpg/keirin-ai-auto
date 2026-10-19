"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
from .lineup import normalize
RECORD_FIELDS=['racePoint', 'firstRate', 'secondRate', 'thirdRate', 'standing', 'back', 'frontRunner', 'stalker', 'deepCloser', 'marker', 'first', 'second', 'third', 'others', 'gearRatio']

def number(v):
    try:
        n=float(v);return n if math.isfinite(n) and not isinstance(v,bool) else np.nan
    except (ValueError,TypeError):return np.nan


def class_code(m):
    s=m['race_class']
    return 3 if 'SA' in s else 2 if s.startswith('L') else 1 if s.startswith('S') else 0 if s.startswith('A') else 4


def age_on(birthday,day):
    try:
        b=date.fromisoformat(str(birthday)[:4]+'-'+str(birthday)[4:6]+'-'+str(birthday)[6:8]);d=date.fromisoformat(day)
        return d.year-b.year-((d.month,d.day)<(b.month,b.day))
    except (ValueError,TypeError):return np.nan


def extract(raw,m):
    z=normalize(raw,m)
    fatal={'race_id_mismatch','date_mismatch','meeting_mismatch','invalid_entry','duplicate_entry','unknown_absence_status','missing_player','duplicate_player'}
    if fatal.intersection(z['errors']) or len(z['roster'])<3:raise ValueError('entry_roster_invalid:'+','.join(z['errors']))
    players={str(p['id']):p for p in raw.get('players',[])};records={str(x['playerId']):x for x in raw.get('records',[])}
    if len(players)!=len(raw.get('players',[])) or len(records)!=len(raw.get('records',[])):raise ValueError('duplicate_player_or_record_identity')
    riders={}
    for cs,r in z['roster'].items():
        p=players[r['player_id']];rec=records.get(r['player_id'],{});v={k:number(rec.get(k)) for k in RECORD_FIELDS}
        v.update(player_id=r['player_id'],age=age_on(p.get('birthday'),m['date']),term=number(p.get('term')),
            style={'逃':1,'追':2,'両':3}.get(rec.get('style'),0),female=int(p.get('gender')=='女'))
        riders[int(cs)]=v
    return dict(race_id=m['race_id'],date=m['date'],meta=m,riders=riders,line=z,distance=number(raw['race'].get('distance')))

