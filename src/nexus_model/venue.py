"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
from . import base as t
GEO=['circumference', 'straight', 'cant_degrees', 'home_width', 'back_width', 'center_width']
RN=['score_gap12', 'verified_line_count', 'longest_known_line', 'singletons', 'leader_form14_minus90', 'leader_form30_minus90', 'class_A', 'class_S', 'girls', 'advance', 'graded']
HSTAT=['MAIN', 'MID', 'HIGH', 'win_escape', 'win_dash', 'win_chase', 'cross_line_top2', 'leader_win', 'leader_podium']
HN=['prior_family_log_count', 'prior_family_MAIN', 'prior_family_MID', 'prior_family_HIGH', 'prior_family_win_escape', 'prior_family_win_dash', 'prior_family_win_chase', 'prior_family_cross_line_top2', 'prior_family_leader_win', 'prior_family_leader_podium', 'prior_venue_log_count', 'prior_venue_MAIN', 'prior_venue_MID', 'prior_venue_HIGH', 'prior_venue_win_escape', 'prior_venue_win_dash', 'prior_venue_win_chase', 'prior_venue_cross_line_top2', 'prior_venue_leader_win', 'prior_venue_leader_podium', 'prior_venue_family_lines_log_count', 'prior_venue_family_lines_MAIN', 'prior_venue_family_lines_MID', 'prior_venue_family_lines_HIGH', 'prior_venue_family_lines_win_escape', 'prior_venue_family_lines_win_dash', 'prior_venue_family_lines_win_chase', 'prior_venue_family_lines_cross_line_top2', 'prior_venue_family_lines_leader_win', 'prior_venue_family_lines_leader_podium']
CN=['score_minus_other_line_best', 'form14_minus_other_line_best', 'own_line_form14_mean', 'own_line_form30_mean', 'score_minus_line_leader', 'form14_minus_line_leader', 'form14_minus90', 'form30_minus90']
GX=['straight', 'cant_degrees', 'home_width']
RI=['racePoint', 'score_minus_field_mean', 'back', 'frontRunner', 'deepCloser', 'line_position', 'line_size', 'd14_score_relative_form', 'd30_score_relative_form', 'd90_score_relative_form']

def row_context(d,i):
 r=d['rows'][i];n=r['active_cars'];c=dict(zip(t.CTX+t.EXTRA_CTX,d['context'][i]));a=d['riders'][i];li=d['line_ids'][i,:n];sz=Counter(int(v) for v in li if v>=0);known=bool(r['lines_applicable']);form=lambda w:float(a[0,t.U_NAMES.index(f'd{w}_score_relative_form')]);cl=float(c['class'])
 # Actual labels come from the pre-event class code, not from winning payouts.
 family=(cl,int(c['graded']),int(c['official_pre_event_advance_format']),int(c['girls']),n)
 line_bucket=min(len(sz),4) if known and np.all(li>=0) else -1
 rn=[float(a[0,0]-a[1,0]),len(sz) if known else np.nan,max(sz.values()) if sz else np.nan,sum(v==1 for v in sz.values()) if known else np.nan,form(14)-form(90),form(30)-form(90),float(cl==0),float(cl==1),float(c['girls']),float(c['official_pre_event_advance_format']),float(c['graded'])]
 return rn,[family,(r['race_id'][2:4],),(r['race_id'][2:4],)+family+(line_bucket,)]

