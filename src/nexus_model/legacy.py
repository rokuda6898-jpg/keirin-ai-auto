"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
from . import base as old
BASE_CTX=['venue', 'class', 'field', 'graded', 'cup_grade', 'race_number', 'distance', 'absence_count', 'girls', 'month_sin', 'month_cos', 'stage_予選', 'stage_一般', 'stage_選抜', 'stage_準決', 'stage_決勝', 'stage_特選', 'stage_チャレンジ', 'stage_新人', 'official_pre_event_advance_format', 'independent_lines_applicable', 'score_std_all', 'score_range_all', 'top_score_tied']
STATIC=['racePoint', 'firstRate', 'secondRate', 'thirdRate', 'back', 'frontRunner', 'stalker', 'deepCloser', 'marker', 'age', 'term', 'style', 'score_rank_fraction', 'score_minus_field_mean', 'car_number_fraction']
LINE=['known', 'position', 'size', 'contested', 'score_mean', 'score_max', 'peer_score_max', 'front_score_max']
RIVAL=['other_line_max', 'other_line_mean', 'own_mean_minus_other_max', 'known_line_count', 'singletons', 'line_fronts', 'own_back_sum', 'other_back_sum', 'contested_cars', 'unassigned_cars']
HISTORY=['prior_n', 'prior_exact_first', 'prior_exact_second', 'prior_exact_third', 'role_n', 'role_exact_first', 'role_exact_second', 'role_exact_third']
REL=['same_line', 'same_slot', 'candidate_ahead']

def line_values(d,lid,pos):
 out=np.full((len(d['rows']),9,len(LINE+RIVAL)),np.nan,np.float32)
 for i,r in enumerate(d['rows']):
  n=r['active_cars'];ids=lid[i,:n];pp=pos[i,:n];u=d['riders'][i,:n];known=ids>=0;groups=set(ids[known].tolist());singleton=sum((ids==k).sum()==1 for k in groups);contested=sum(known[j] and ((ids==ids[j])&(pp==pp[j])).sum()>1 for j in range(n))
  for j in range(n):
   out[i,j,0]=float(known[j]) if r['lines_applicable'] else np.nan
   if not known[j]:continue
   own=(ids==ids[j]);other=known&~own;peer=own.copy();peer[j]=False;front=own&(pp==0);sc=u[:,0];back=u[:,old.U_NAMES.index('back')]
   mx=lambda a:float(np.max(a)) if len(a) and np.isfinite(a).all() else np.nan
   mean=lambda a:float(np.mean(a)) if len(a) and np.isfinite(a).all() else np.nan
   sm=lambda a:float(np.sum(a)) if len(a) and np.isfinite(a).all() else np.nan
   om=mx(sc[other]);ownmean=mean(sc[own]);out[i,j]=[1,pp[j],int(own.sum()),int(((ids==ids[j])&(pp==pp[j])).sum()>1),ownmean,mx(sc[own]),mx(sc[peer]),mx(sc[front]),om,mean(sc[other]),ownmean-om,len(groups),singleton,int((known&(pp==0)).sum()),sm(back[own]),sm(back[other]),contested,int((~known).sum())]
 return out


def context_names(v):return BASE_CTX if v!='all_data' else old.CTX+old.EXTRA_CTX


def rider_names(v,kind):return STATIC+(['form_'+n for n in old.FORM] if v=='all_data' else [])+['line_'+n for n in (LINE if v=='legacy_structure' else LINE+RIVAL)]+(HISTORY if kind=='role' else [])


def rider_values(d,i,v,kind):
 u=d['riders'][i];base=u[:,[old.U_NAMES.index(n) for n in STATIC]];mode='legacy' if v=='legacy_structure' else 'verified';parts=[base]
 if v=='all_data':parts.append(u[:,[old.U_NAMES.index(n) for n in old.FORM]])
 parts.append(d[mode+'_line'][i,:,:len(LINE) if v=='legacy_structure' else len(LINE+RIVAL)])
 if kind=='role':parts.append(d[mode+'_history'][i])
 return np.concatenate(parts,axis=1)


def feature_names(v,kind,stage):
 names=['context_'+n for n in context_names(v)];rn=rider_names(v,kind if kind!='joint' else 'graph');names += ['candidate_'+n for n in rn]
 for k in range(stage):names += [f'prefix{k+1}_{n}' for n in rn]+[f'delta{k+1}_{n}' for n in rn]+[f'relation{k+1}_{n}' for n in REL]
 if stage==2:names+=['first_second_same_line']
 return names


def features(d,i,v,kind,candidate,first=None,second=None):
 cc=np.asarray(candidate,int);n=len(cc);u=rider_values(d,i,v,kind if kind!='joint' else 'graph');ctx=d['context'][i,[ (old.CTX+old.EXTRA_CTX).index(x) for x in context_names(v)]];parts=[np.repeat(ctx[None,:],n,axis=0),u[cc]];mode='legacy' if v=='legacy_structure' else 'verified';li=d[mode+'_ids'][i];po=d[mode+'_positions'][i]
 for prefix in [first,second]:
  if prefix is None:continue
  prefix=np.asarray(prefix,int);known=(li[cc]>=0)&(li[prefix]>=0);rel=np.full((n,3),np.nan,np.float32);same=li[cc]==li[prefix];rel[known]=np.column_stack([same,same&(po[cc]==po[prefix]),same&(po[cc]<po[prefix])])[known];parts.extend([u[prefix],u[cc]-u[prefix],rel])
 if second is not None:
  rel=np.full(n,np.nan,np.float32);known=(li[first]>=0)&(li[second]>=0);rel[known]=(li[first][known]==li[second][known]);parts.append(rel[:,None])
 return np.column_stack(parts).astype(np.float32)

