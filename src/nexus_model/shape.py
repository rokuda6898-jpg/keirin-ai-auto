"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta


def expand(blocks):
    return sorted({t for f,b,c in blocks for t in itertools.product([f],b,c) if len(set(t))==3})


def cube(cars,q):
    n=len(cars);v=np.zeros((n,n,n));ix=np.array(list(itertools.permutations(range(n),3)));v[tuple(ix.T)]=q;return v


def flatten_third(v):
    pair=v.sum(axis=2,keepdims=True);out=np.sqrt(v);den=out.sum(axis=2,keepdims=True)
    return np.divide(out,den,out=np.zeros_like(out),where=den>0)*pair


def profiles(v,first,cap,need_two):
    """Each block: fixed first, contiguous rank group of seconds, third-pool prefix.
    Two blocks split second riders into disjoint groups, hence no duplicate spend.
    """
    n=len(v);others=[i for i in range(n) if i!=first]
    order=sorted(others,key=lambda b:(-float(v[first,b].sum()),b))
    def options(bb):
        col=v[first,bb,:].sum(axis=0);cc=sorted(others,key=lambda c:(-float(col[c]),c));out=[(0,0.,[])]
        cost=0;value=0.;bs=set(bb)
        for j,c in enumerate(cc):
            cost+=len(bb)-int(c in bs);value+=float(col[c])
            if 0<cost<=cap:out.append((cost,value,[[first,list(bb),cc[:j+1]]]))
        return out
    single={0:(0.,[])}
    def put(dest,cost,score,blocks):
        if cost>cap:return
        old=dest.get(cost)
        if old is None or score>old[0]+1e-15:dest[cost]=(score,blocks)
    prefixes={k:options(order[:k]) for k in range(1,n)}
    for opts in prefixes.values():
        for cost,score,blocks in opts:put(single,cost,score,blocks)
    if not need_two:return single,None
    double=dict(single)
    for k in range(1,n-1):
        suffix=options(order[k:])
        for ca,va,ba in prefixes[k]:
            for cb,vb,bb in suffix:put(double,ca+cb,va+vb,ba+bb)
    return single,double


def combine(profile_list,cap):
    states={0:(0.,[])}
    for prof in profile_list:
        nxt={}
        for used,(val,blocks) in states.items():
            for cost,(vv,bb) in prof.items():
                total=used+cost
                if total>cap:continue
                old=nxt.get(total);score=val+vv
                if old is None or score>old[0]+1e-15:nxt[total]=(score,blocks+bb)
        states=nxt
    assert cap in states,(cap,[sorted(p) for p in profile_list])
    return states[cap][1]


def formation_set(v,cap,anchors,which):
    ps=[profiles(v,int(f),cap,j<2) for j,f in enumerate(anchors[:4 if 'four' in which else 2])]
    ans={}
    if 'axis2' in which:ans['axis2']=combine([p[1] for p in ps[:2]],cap)
    if 'axis1' in which:
        if cap<=(len(v)-1)*(len(v)-2):assert cap in ps[0][1]
        ans['axis1']=ps[0][1][cap][1] if cap in ps[0][1] else combine([p[1] for p in ps[:2]],cap)
    if 'four' in which:ans['four']=combine([p[0] for p in ps[:4]],cap)
    for bb in ans.values():assert len(bb)<=4 and len(expand(bb))==cap
    return ans

