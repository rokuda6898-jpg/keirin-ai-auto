"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
from . import legacy as parent

def names(v):
 z=['context_'+x for x in parent.context_names(v)];rn=parent.rider_names(v,'role')
 for j in range(9):z += [f'r{j}_present']+[f'r{j}_{x}' for x in rn]
 for a,b in itertools.combinations(range(9),2):z += [f'r{a}_r{b}_same_line',f'r{a}_r{b}_same_slot']
 return z


def row(d,i,v):
 r=d['rows'][i];n=r['active_cars'];u=parent.rider_values(d,i,v,'role');out=d['context'][i,[(parent.old.CTX+parent.old.EXTRA_CTX).index(k) for k in parent.context_names(v)]].tolist()
 for j in range(9):out += [int(j<n)]+(u[j].tolist() if j<n else [np.nan]*u.shape[1])
 mode='legacy' if v=='legacy_structure' else 'verified';li=d[mode+'_ids'][i];po=d[mode+'_positions'][i]
 for a,b in itertools.combinations(range(9),2):
  out += [int(li[a]==li[b]),int(li[a]==li[b] and po[a]==po[b])] if b<n and li[a]>=0 and li[b]>=0 else [np.nan,np.nan]
 return np.array(out,np.float32)

