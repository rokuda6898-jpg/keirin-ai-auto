"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
from . import forms as old,exact

@functools.lru_cache(None)
def masks(n):
    return ((np.arange(1,1<<n)[:,None]>>np.arange(n))&1).astype(bool)


def profile(v,firsts,cap):
    n=len(v);fs=list(firsts);available=np.array([j not in fs for j in range(n)])
    # Keeping firsts out of second/third is a declared restriction for this
    # family. The old optimizer remains a candidate, so it can never lower mass.
    mask=masks(n);bb=mask[(mask&~available).sum(1)==0];sizes=bb.sum(1)
    if len(bb)==0:return {}
    a=v[fs].sum(0);col=bb.astype(float)@a
    inside=bb;outside=available[None,:]&~bb
    orderi=np.argsort(-np.where(inside,col,-np.inf),axis=1,kind='stable')
    ordero=np.argsort(-np.where(outside,col,-np.inf),axis=1,kind='stable')
    ci=np.column_stack([np.zeros(len(bb)),np.cumsum(np.take_along_axis(np.where(inside,col,0),orderi,1),axis=1)])
    co=np.column_stack([np.zeros(len(bb)),np.cumsum(np.take_along_axis(np.where(outside,col,0),ordero,1),axis=1)])
    u=np.arange(n+1)[None,:,None];w=np.arange(n+1)[None,None,:]
    costs=len(fs)*(u*(sizes[:,None,None]-1)+w*sizes[:,None,None])
    values=ci[:,:,None]+co[:,None,:]
    valid=(u<=sizes[:,None,None])&(w<=(available.sum()-sizes)[:,None,None])&(costs>0)&(costs<=cap)
    flatv=values.ravel();flatc=np.broadcast_to(costs,values.shape).ravel();idx=np.flatnonzero(valid)
    best=np.full(cap+1,-np.inf);np.maximum.at(best,flatc[idx],flatv[idx])
    chosen=idx[flatv[idx]==best[flatc[idx]]];by={}
    for ix in chosen:
        cost=int(flatc[ix])
        if cost in by:continue
        b,ui,wi=np.unravel_index(ix,values.shape)
        second=np.flatnonzero(bb[b]).tolist();third=sorted(set(orderi[b,:ui].tolist()+ordero[b,:wi].tolist()))
        block=dict(first=fs,second=second,third=third,count=cost)
        by[cost]=(float(flatv[ix]),block)
    return by


def solve_single(profiles,cap,maxblocks=4,exclude=()):
    states={(0,0):(0.,[])}
    for j,p in enumerate(profiles):
        if j in exclude:continue
        nxt=dict(states)
        for (cost,k),(value,blocks) in states.items():
            if k>=maxblocks:continue
            for c,(val,bl) in p.items():
                key=(cost+c,k+1)
                if cost+c>cap:continue
                new=value+val
                if new>nxt.get(key,(-np.inf,None))[0]+1e-15:nxt[key]=(new,blocks+[bl])
        states=nxt
    return states


def select(cars,q,cap):
    baseline=old.select(cars,q,cap,'axis2' if cap==6 else 'four')
    tri=list(itertools.permutations(cars,3));scores=dict(zip(tri,map(float,q)))
    value=lambda ts:sum(scores[tuple(t)] for t in ts)
    best=value(baseline['tickets']);answer=baseline;kind='legacy_shape'
    # Lossless packing of the unconstrained top set, when it exists, reaches
    # the absolute top-cap score bound and needs no further search.
    top=tuple(sorted(sorted(tri,key=lambda t:(-scores[t],t))[:cap]));packed=exact.pack(top)
    if packed is not None:return dict(blocks=packed,tickets=[list(t) for t in top],search='exact_top_set')
    n=len(cars);v=np.zeros((n,n,n));ix=np.array(list(itertools.permutations(range(n),3)));v[tuple(ix.T)]=q
    ps=[profile(v,[j],cap) for j in range(n)];states=solve_single(ps,cap)
    candidates=[(val,bl,'free_single_first') for (cost,k),(val,bl) in states.items() if cost==cap]
    order=sorted(range(n),key=lambda j:(-v[j].sum(),j))
    firstsets=list(itertools.combinations(range(n),2))+[tuple(sorted(order[:k])) for k in [3,4] if n-k>=2]
    # Multi-first candidates can cover lower-marginal winners without spending
    # a separate block. Other blocks have disjoint firsts, so no double stake.
    for fs in firstsets:
        p=profile(v,fs,cap)
        if not p:continue
        rest=solve_single(ps,cap,3,fs)
        for c,(vv,bl) in p.items():
            for k in range(4):
                z=rest.get((cap-c,k))
                if z is not None:candidates.append((vv+z[0],z[1]+[bl],'multi_first'))
    for vv,bl,tag in candidates:
        if vv<=best+1e-15:continue
        converted=[dict(first=[cars[j] for j in b['first']],second=[cars[j] for j in b['second']],third=[cars[j] for j in b['third']],count=b['count']) for b in bl]
        tickets=old.expand(converted);assert len(tickets)==cap and len(converted)<=4
        assert abs(value(tickets)-vv)<1e-10
        best=vv;answer=dict(blocks=converted,tickets=[list(t) for t in tickets]);kind=tag
    return answer|dict(search=kind)

