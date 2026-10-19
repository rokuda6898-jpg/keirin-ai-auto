"""Prospective adapter. Only explicitly dated prior-day history is queried."""
import gzip,json,math
from collections import defaultdict
from datetime import date
from pathlib import Path
import numpy as np
from . import base,raw,clock,legacy,tactical,role,v2,venue

def key(*parts):return '|'.join(map(str,parts))
def mean(a,k):
    x=[v[k] for _,v in a if v[k] is not None]
    return float(np.mean(x)) if x else np.nan
def recent(a,day,w):return [(dt,v) for dt,v in a if day-w<=dt<day]
def stat(a,day,w,n):
    q=recent(a,day,w)
    return [len(q)]+[mean(q,k) for k in range(n)]
def since(a,day):
    q=[dt for dt,_ in a if dt<day]
    return day-max(q) if q else np.nan
def emit(q):return [q[0]]+[v/q[0] for v in q[1:]] if q[0] else [0,np.nan,np.nan,np.nan]

class History:
    def __init__(self,obj=None):
        self.data=obj or dict(through=None,overall={},positions={},form={},tactics={},tactic_roles={},pairs={},pooled={},roles={},venue=[{},{},{}],global_stats=[[0.]*9,[0.]*9],geometry={})
    @classmethod
    def load(cls,path):
        with gzip.open(path,'rt',encoding='utf8') as f:return cls(json.load(f))
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        # Last observed dates are retained even for infrequently active riders.
        day=date.fromisoformat(self.data['through']).toordinal()
        for name in ['form','tactics','tactic_roles','pairs','pooled','roles']:
            for k,a in self.data[name].items():
                self.data[name][k]=[z for z in a if z[0]>=day-365] or a[-1:]
        with gzip.open(path,'wt',encoding='utf8') as f:json.dump(self.data,f,separators=(',',':'),allow_nan=False)
    def add(self,event):
        """Append a completed race after all predictions on its date. No model fitting."""
        h=self.data;day=date.fromisoformat(event['date']).toordinal()
        for p in event['people']:
            pid=str(p['pid'])
            for name,field,k in [('form','form',pid),('pooled','pooled',pid),('roles','role_event',key(pid,p.get('role'))),('tactics','flags',pid),('tactic_roles','role_flags',key(pid,p.get('pair_role')))]:
                if p.get(field) is not None:h[name].setdefault(k,[]).append([day,p[field]])
            if p.get('valid_legacy'):
                delta=[1,int(p['rank']==1),int(p['rank']==2),int(p['rank']==3)]
                for table,k in [('overall',pid)]+([('positions',key(pid,p['position']))] if p.get('position',-1)>=0 else []):
                    q=h[table].setdefault(k,[0]*4);h[table][k]=[a+b for a,b in zip(q,delta)]
        for a,b,values in event.get('pairs',[]):h['pairs'].setdefault(key(a,b),[]).append([day,values])
        if event.get('venue_values') is not None:
            stats=[h['global_stats']]+[table.setdefault(key(*k),[[0.]*9,[0.]*9]) for table,k in zip(h['venue'],event['venue_keys'])]
            for st in stats:
                for j,x in enumerate(event['venue_values']):
                    if x is not None:st[0][j]+=1;st[1][j]+=x
        h['through']=max(h.get('through') or event['date'],event['date'])

def build(history,source,meta):
    h=history.data;day=date.fromisoformat(meta['date']).toordinal()
    if h['through'] and h['through']>=meta['date']:raise ValueError('history must end before race day')
    entry=raw.extract(source,meta);rr=entry['riders'];z=entry['line']
    advance=('アドバンス' in meta['race_type'] or 'アドバンス' in meta['race_class'] or (meta['date']>='2025-04-01' and '新人' in meta['race_type'] and not any(r['female'] for r in rr.values())))
    girls=any(r['female'] for r in rr.values());applicable=not girls and not advance
    cars=sorted(rr,key=lambda c:(-rr[c]['racePoint'] if math.isfinite(rr[c]['racePoint']) else 1e9,c));n=len(cars);ix={c:j for j,c in enumerate(cars)}
    row=dict(race_id=meta['race_id'],date=meta['date'],active_cars=n,cars=cars,rider_ids=[rr[c]['player_id'] for c in cars],lines_applicable=applicable)
    d=dict(rows=[row],context=np.full((1,len(base.CTX+base.EXTRA_CTX)),np.nan,np.float32),riders=np.full((1,9,len(base.U_NAMES)),np.nan,np.float32),line_ids=np.full((1,9),-1,np.int16),line_positions=np.full((1,9),-1,np.int16),cars=np.zeros((1,9),np.int16))
    scores=[rr[c]['racePoint'] for c in cars];finite=[x for x in scores if math.isfinite(x)];avg=float(np.mean(finite)) if finite else np.nan
    for j,c in enumerate(cars):
        r=rr[c];vals,ns=clock.profile(h['form'].get(r['player_id'],[]),day);fp=dict(zip(ns,vals))
        d['cars'][0,j]=c;d['riders'][0,j]=[r[k] for k in base.STATIC]+[fp[k] for k in base.FORM]+[j/max(1,n-1),r['racePoint']-avg,0 if applicable else np.nan,np.nan,np.nan,np.nan,np.nan,c/9]
    groups=z['known_unchanged_groups'] if applicable else []
    for k,slots in enumerate(groups):
        cc=[c for slot in slots for c in slot];sc=[rr[c]['racePoint'] for c in cc];pair=float(np.mean(sorted(sc,reverse=True)[:2])) if all(map(math.isfinite,sc)) else np.nan
        for pos,slot in enumerate(slots):
            for c in slot:
                j=ix[c];d['line_ids'][0,j]=k;d['line_positions'][0,j]=pos;d['riders'][0,j,-6:-1]=[1,pos,len(cc),int(len(slot)>1),pair-avg]
    card=dict(date=meta['date'],startAt=source['race'].get('startAt'),closeAt=source['race'].get('closeAt'),schedule_day=source['schedule'].get('day',np.nan),race_number=meta['race_number'])
    tv,_,_=clock.time_vector(card);month=int(meta['date'][5:7]);st=meta['race_type']
    context=[int(meta['venue_id']),raw.class_code(meta),n,meta['graded'],meta['cup_grade'],meta['race_number'],entry['distance'],len(z['absent_cars']),girls,math.sin(2*math.pi*month/12),math.cos(2*math.pi*month/12)]+[float(s in st) for s in v2.STAGES]+tv[:8]
    context += [float(advance),float(applicable),len(groups) if applicable else np.nan,sum(d['line_ids'][0,:n]<0) if applicable else np.nan,float(np.std(finite)) if finite else np.nan,max(finite)-min(finite) if finite else np.nan,float(len(finite)>1 and finite.count(max(finite))>1)]
    d['context'][0]=context
    d['verified_ids']=d['line_ids'];d['verified_positions']=d['line_positions'];d['verified_line']=legacy.line_values(d,d['line_ids'],d['line_positions']);d['verified_history']=np.full((1,9,8),np.nan,np.float32)
    d['partners']=np.full((1,9),-1,np.int16);d['pair_roles']=np.full((1,9),-1,np.int16);d['tactics']=np.full((1,9,len(tactical.EXTRA)),np.nan,np.float32);d['role_history']=np.full((1,9,len(role.RN)),np.nan,np.float32)
    pids=row['rider_ids'];li=d['line_ids'][0,:n];po=d['line_positions'][0,:n]
    for j,pid in enumerate(pids):
        d['verified_history'][0,j,:4]=emit(h['overall'].get(pid,[0]*4))
        if li[j]>=0:d['verified_history'][0,j,4:]=emit(h['positions'].get(key(pid,int(po[j])),[0]*4))
    if not girls:
        for j,pid in enumerate(pids):
            a=h['tactics'].get(pid,[]);d['tactics'][0,j,:16]=stat(a,day,90,6)+stat(a,day,365,6)+[since(a,day),0]
        for a,b in tactical.pair_indices(d,0):
            q=h['pairs'].get(key(pids[a],pids[b]),[]);qs=stat(q,day,365,8)
            for j,k,r in [(a,b,0),(b,a,1)]:
                d['partners'][0,j]=k;d['pair_roles'][0,j]=r;d['tactics'][0,j,15:]=[1,r,qs[0],since(q,day)]+d['tactics'][0,k,:7].tolist()+qs[1:]
        for j,pid in enumerate(pids):
            a=[h['tactic_roles'].get(key(pid,r),[]) for r in [0,1]];r=int(d['pair_roles'][0,j]);d['role_history'][0,j]=sum((stat(a[r],day,w,6) for r in [0,1] for w in [90,365]),[])+[since(v,day) for v in a]+([float(r==0),float(r==1)] if r>=0 else [np.nan,np.nan])
    rn,keys=venue.row_context(d,0);d['venue_race_extra']=np.array([rn],np.float32);g=h['geometry'].get(meta['venue_id'],{});d['venue_geometry']=np.array([[g.get(k,np.nan) if meta['venue_id'] not in ['51','54','71'] else np.nan for k in venue.GEO]],np.float32)
    gs=np.asarray(h['global_stats']);prior=np.divide(gs[1],gs[0],out=np.full(9,np.nan),where=gs[0]>0);vh=[]
    for tab,k in zip(h['venue'],keys):
        a=np.asarray(tab.get(key(*k),[[0.]*9,[0.]*9]));vh += [math.log1p(a[0,0])]+((a[1]+30*prior)/(a[0]+30)).tolist()
    d['venue_history']=np.array([vh],np.float32);cx=np.full((1,9,8),np.nan,np.float32);u=d['riders'][0];ui={k:j for j,k in enumerate(base.U_NAMES)};f14=u[:n,ui['d14_score_relative_form']];f30=u[:n,ui['d30_score_relative_form']];f90=u[:n,ui['d90_score_relative_form']];cx[0,:n,6]=f14-f90;cx[0,:n,7]=f30-f90
    for j in range(n):
        if li[j]<0:continue
        same=li==li[j];other=(li>=0)&~same;front=same&(po==0)
        if other.any():cx[0,j,:2]=[u[j,0]-np.max(u[:n,0][other]),f14[j]-np.max(f14[other])]
        cx[0,j,2:4]=[np.mean(f14[same]),np.mean(f30[same])]
        if front.sum()==1:k=int(np.flatnonzero(front)[0]);cx[0,j,4:6]=[u[j,0]-u[k,0],f14[j]-f14[k]]
    d['venue_candidate_extra']=cx;d['new_stage']=np.array([v2.stage_type(st)],np.int16);d['new_extra']=np.full((1,9,len(v2.HN)),np.nan,np.float32);roles=v2.roles(d,0)
    for j,pid in enumerate(pids):
        a=h['pooled'].get(pid,[]);b=h['roles'].get(key(pid,roles[j]),[]);last=day-since(a,day) if a else None;lastrole=day-since(b,day) if b else None
        vals=v2.profile(a,b,day,roles[j],last,lastrole)
        same=[k for k in range(n) if li[j]>=0 and li[k]==li[j] and po[k]==0];other=[k for k in range(n) if li[j]>=0 and li[k]>=0 and li[k]!=li[j]]
        for field,items,one in [('racePoint',same,True),('d14_score_relative_form',same,True),('racePoint',other,False),('d14_score_relative_form',other,False)]:
            vv=[float(u[k,ui[field]]) for k in items if math.isfinite(u[k,ui[field]])];vals.append(float(u[j,ui[field]])-(vv[0] if one else max(vv)) if vv and (not one or len(items)==1) else np.nan)
        partner=int(d['partners'][0,j]);vals+=d['role_history'][0,j].tolist()+(d['role_history'][0,partner].tolist() if partner>=0 else [np.nan]*len(role.RN));d['new_extra'][0,j]=vals
    return d,z
