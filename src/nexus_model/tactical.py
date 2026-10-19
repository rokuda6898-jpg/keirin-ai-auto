"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
from . import legacy
ORIGINAL_NAMES=legacy.feature_names
ORIGINAL_FEATURES=legacy.features
FIELDS=['spurtSucceeded', 'thrustSucceeded', 'leftBehind', 'splitLine', 'snatchSucceeded', 'competeSucceeded']
PN=['front_spurt', 'front_thrust', 'front_leftBehind', 'follower_splitLine', 'follower_snatch', 'follower_compete', 'front_left_and_follow_split', 'front_spurt_and_follow_split']
EXTRA=['tactic_d90_records', 'tactic_d90_spurtSucceeded', 'tactic_d90_thrustSucceeded', 'tactic_d90_leftBehind', 'tactic_d90_splitLine', 'tactic_d90_snatchSucceeded', 'tactic_d90_competeSucceeded', 'tactic_d365_records', 'tactic_d365_spurtSucceeded', 'tactic_d365_thrustSucceeded', 'tactic_d365_leftBehind', 'tactic_d365_splitLine', 'tactic_d365_snatchSucceeded', 'tactic_d365_competeSucceeded', 'tactic_days_since_record', 'unique_verified_pair_available', 'pair_role_front0_follow1', 'pair_d365_records', 'pair_days_since_record', 'partner_d90_records', 'partner_d90_spurtSucceeded', 'partner_d90_thrustSucceeded', 'partner_d90_leftBehind', 'partner_d90_splitLine', 'partner_d90_snatchSucceeded', 'partner_d90_competeSucceeded', 'pair_d365_front_spurt', 'pair_d365_front_thrust', 'pair_d365_front_leftBehind', 'pair_d365_follower_splitLine', 'pair_d365_follower_snatch', 'pair_d365_follower_compete', 'pair_d365_front_left_and_follow_split', 'pair_d365_front_spurt_and_follow_split']
REL=['candidate_front_of_prefix', 'candidate_follower_of_prefix', 'same_unique_verified_pair']

def pair_indices(d,i):
 r=d['rows'][i];n=r['active_cars'];out=[]
 if not r['lines_applicable']:return out
 li=d['line_ids'][i,:n];po=d['line_positions'][i,:n]
 for group in sorted(set(int(z) for z in li if z>=0)):
  a=np.flatnonzero((li==group)&(po==0));b=np.flatnonzero((li==group)&(po==1))
  if len(a)==len(b)==1:out.append((int(a[0]),int(b[0])))
 return out


def eligible(d,i):
 # Applicability is a pre-race class field, never current outcome or event presence.
 return d['context'][i,(legacy.old.CTX+legacy.old.EXTRA_CTX).index('girls')]==0


def names(variant,kind,stage):
 out=ORIGINAL_NAMES('all_data',kind,stage)
 if variant=='before':return out
 return out+['candidate_'+n for n in EXTRA]+[f'prefix{k+1}_{n}' for k in range(stage) for n in EXTRA+REL]


def features(d,i,variant,kind,candidate,first=None,second=None):
 base=ORIGINAL_FEATURES(d,i,'all_data',kind,candidate,first,second)
 if variant=='before':return base
 cc=np.asarray(candidate,int);parts=[base,d['tactics'][i,cc]]
 for pre in [first,second]:
  if pre is None:continue
  pre=np.asarray(pre,int);relation=np.full((len(cc),3),np.nan,np.float32)
  if d['rows'][i]['lines_applicable']:
   partner=d['partners'][i,cc];roles=d['pair_roles'][i,cc];known=(d['line_ids'][i,cc]>=0)&(d['line_ids'][i,pre]>=0)
   relation[known]=np.column_stack([(partner==pre)&(roles==0),(partner==pre)&(roles==1),partner==pre])[known]
  parts += [d['tactics'][i,pre],relation]
 return np.column_stack(parts).astype(np.float32)

