"""History updates are applied only after the whole previous date is available."""
import math
from collections import defaultdict
from datetime import date,timedelta
from concurrent.futures import ThreadPoolExecutor
from . import inputs,tactical,v2,venue,source

def event(history,d,m):
    if d['race'].get('cancel'):return dict(date=m['date'],people=[],pairs=[])
    dd,_=inputs.build(history,d,m);r=dd['rows'][0];pids=r['rider_ids'];n=len(pids);idx={p:j for j,p in enumerate(pids)};rr={x['playerId']:x for x in d.get('results',[])};roles=v2.roles(dd,0);pairs=tactical.pair_indices(dd,0) if tactical.eligible(dd,0) else [];pr={j:k for a,b in pairs for j,k in [(a,0),(b,1)]};out=dict(date=m['date'],people=[],pairs=[])
    if d['race'].get('cancel'):return out
    if not d['race'].get('decidedAt') or not rr:raise ValueError('historical result not finalized: '+m['race_id'])
    payouts,_,_,complete=source.outcomes(d);winners=[p for p,z in rr.items() if z.get('order')==1 and p in idx];scores=dd['riders'][0,:n,0].astype(float);allsc=all(map(math.isfinite,scores));girls=not tactical.eligible(dd,0)
    for j,pid in enumerate(pids):
        z=rr.get(pid)
        if z is None:raise ValueError('missing individual result: '+m['race_id'])
        rank=z.get('order');valid=isinstance(rank,int) and 1<=rank<=n;finish=(rank-1)/(n-1) if valid else None;expected=(sum(x>scores[j] for x in scores)+(sum(x==scores[j] for x in scores)-1)/2)/(n-1) if allsc else None;li=dd['line_ids'][0];po=dd['line_positions'][0];cross=None
        if rank==2 and len(winners)==1 and r['lines_applicable'] and min(li[j],li[idx[winners[0]]])>=0:cross=float(li[j]!=li[idx[winners[0]]])
        flags=[z.get(k) if type(z.get(k)) is bool else None for k in v2.FLAGS]
        pooled=[float(rank==1),float(valid and rank<=3),finish,z.get('hasAccident'),float(not valid),z.get('back')]+[float(x) if x is not None and not girls else None for x in flags]
        pooled=[float(x) if isinstance(x,(bool,int,float)) else None for x in pooled]
        p=dict(pid=pid,rank=rank,position=int(po[j]) if li[j]>=0 else -1,valid_legacy=complete and valid,form=dict(rank=rank or 0,field=n,finish=finish,residual=expected-finish if expected is not None and finish is not None else None,accident=z.get('hasAccident') if type(z.get('hasAccident')) is bool else None),pooled=pooled,role=roles[j],role_event=pooled+[cross])
        if not girls and all(type(x) is bool for x in flags):
            p['flags']=flags
            if j in pr:p.update(pair_role=pr[j],role_flags=flags)
        out['people'].append(p)
    people={p['pid']:p for p in out['people']}
    for a,b in pairs:
        aa=people[pids[a]].get('flags');bb=people[pids[b]].get('flags')
        if aa is not None and bb is not None:out['pairs'].append([pids[a],pids[b],[aa[0],aa[1],aa[2],bb[3],bb[4],bb[5],aa[2] and bb[3],aa[0] and bb[3]]])
    if complete:
        bands={0 if x<4000 else 1 if x<10000 else 2 for x in payouts.values()};methods={rr[p].get('factor') for p in winners};methods.discard(None)
        cross=[]
        for ticket in payouts:
            a,b,_=map(int,ticket.split('-'));j,k=r['cars'].index(a),r['cars'].index(b);li=dd['line_ids'][0]
            if min(li[j],li[k])>=0 and r['lines_applicable']:cross.append(float(li[j]!=li[k]))
        leader=rr.get(pids[0],{}).get('order');out['venue_keys']=venue.row_context(dd,0)[1];out['venue_values']=[int(k in bands) for k in range(3)]+[int(k in methods) if methods else None for k in ['逃','捲','差']]+[max(cross) if cross else None,float(leader==1),float(isinstance(leader,int) and 1<=leader<=3)]
    return out

def update(history,metas,cache,path,today):
    byday=defaultdict(list)
    for m in metas:
        if (history.data['through'] or '')<m['date']<today:byday[m['date']].append(m)
    ds=date.fromisoformat(history.data['through'])+timedelta(days=1)
    while ds<date.fromisoformat(today):
        day=ds.isoformat();rows=byday[day]
        with ThreadPoolExecutor(max_workers=4) as ex:sources=list(ex.map(lambda m:source.race(m,cache,past=True),rows))
        pending=[event(history,z['data'],m) for z,m in zip(sources,rows)]
        for e in pending:history.add(e)
        history.data['through']=day;history.save(path);print('History through',day,'races',len(rows),flush=True);ds+=timedelta(days=1)
