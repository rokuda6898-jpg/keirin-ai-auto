"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
from . import tactical,venue,confidence,base as t
ROLES=['general', 'MAIN', 'MEDIUM', 'HIGH']

def names(stage):
 z=tactical.names('after','role',stage)
 z += ['bank_'+n for n in venue.GEO]+['venue_context_'+n for n in venue.RN]+['venue_'+n for n in venue.HN]
 for role in ['candidate']+[f'prefix{k+1}' for k in range(stage)]:z += [role+'_venue_'+n for n in venue.CN]
 z += [f'bank_{g}_x_race_{n}' for g in venue.GX for n in venue.RN]
 z += [f'bank_{g}_x_candidate_{n}' for g in venue.GX for n in venue.RI]
 return z


def features(d,i,cc,first=None,second=None):
 cc=np.asarray(cc,int);stage=int(first is not None)+int(second is not None);n=len(cc)
 a=tactical.features(d,i,'after','role',cc,first,second)
 g=d['venue_geometry'][i];rn=d['venue_race_extra'][i];h=d['venue_history'][i]
 parts=[a,np.repeat(np.r_[g,rn,h][None],n,axis=0),d['venue_candidate_extra'][i,cc]]
 for pre in [first,second]:
  if pre is not None:parts.append(d['venue_candidate_extra'][i,np.asarray(pre,int)])
 gi=[venue.GEO.index(v) for v in venue.GX];ri=[t.U_NAMES.index(v) for v in venue.RI]
 parts += [np.repeat((g[gi,None]*rn[None,:]).reshape(1,-1),n,axis=0),
           (g[None,gi,None]*d['riders'][i,cc][:,None,ri]).reshape(n,-1)]
 x=np.column_stack(parts).astype(np.float32);assert x.shape==(n,len(names(stage)))
 return x


def class_names():
 z=confidence.names('all_data')
 for j in range(9):z += [f'r{j}_'+n for n in tactical.EXTRA+['venue_'+v for v in venue.CN]]
 return z+['bank_'+n for n in venue.GEO]+['venue_context_'+n for n in venue.RN]+['venue_'+n for n in venue.HN]


def class_row(d,i):
 n=d['rows'][i]['active_cars'];parts=[confidence.row(d,i,'all_data')]
 for j in range(9):parts.append(np.r_[d['tactics'][i,j],d['venue_candidate_extra'][i,j]] if j<n else np.full(len(tactical.EXTRA)+len(venue.CN),np.nan))
 return np.concatenate(parts+[d['venue_geometry'][i],d['venue_race_extra'][i],d['venue_history'][i]]).astype(np.float32)


def calibrate(p,temp):
 p=np.maximum(np.asarray(p,float),1e-15);z=np.log(p)/temp;z-=z.max(-1,keepdims=True);v=np.exp(z);return v/v.sum(-1,keepdims=True)


def three(p):return np.array([p[0]+p[1],p[2],p[3]])


def cap(p):
 b=int(three(p).argmax());return (6 if p[0]>=.60 else 12) if b==0 else 15 if b==1 else 24


def predict_matrices(class_x,mats,ns,mods):
 bands=mods['judge',-1].predict_proba(class_x);out=[]
 spans=[]
 for stage in range(3):spans.append(np.r_[0,np.cumsum([len(t.templates(n)[stage]) for n in ns])])
 vv={(role,k):np.maximum(mods[role,k].predict_proba(mats[k])[:,1],1e-15) for role in ROLES for k in range(3)}
 for j,n in enumerate(ns):
  z={}
  for role in ROLES:
   parts=[]
   for k in range(3):
    a,b=spans[k][j:j+2];v=vv[role,k][a:b].reshape(-1,n-k);parts.append((v/v.sum(1)[:,None]).ravel())
   q=np.repeat(parts[0],(n-1)*(n-2))*np.repeat(parts[1],n-2)*parts[2];assert abs(q.sum()-1)<1e-10;z[role]=q
  out.append(dict(raw=bands[j],scores=z))
 return out

