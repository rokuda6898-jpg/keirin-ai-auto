"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
from . import optimizer as old

def profiles(v,first,cap):
    n=len(v);available=np.arange(n)!=first;bb=old.masks(n);bb=bb[(bb&~available).sum(1)==0];sizes=bb.sum(1)
    col=bb.astype(float)@v[first];inside=bb;outside=available[None,:]&~bb
    oi=np.argsort(-np.where(inside,col,-np.inf),axis=1,kind='stable');oo=np.argsort(-np.where(outside,col,-np.inf),axis=1,kind='stable')
    ci=np.column_stack([np.zeros(len(bb)),np.cumsum(np.take_along_axis(np.where(inside,col,0),oi,1),axis=1)])
    co=np.column_stack([np.zeros(len(bb)),np.cumsum(np.take_along_axis(np.where(outside,col,0),oo,1),axis=1)])
    u=np.arange(n+1)[None,:,None];w=np.arange(n+1)[None,None,:]
    cost=np.broadcast_to(u*(sizes[:,None,None]-1)+w*sizes[:,None,None],(len(bb),n+1,n+1));value=ci[:,:,None]+co[:,None,:]
    good=(u<=sizes[:,None,None])&(w<=available.sum()-sizes[:,None,None])&(cost>0)&(cost<=cap)
    rows=np.broadcast_to(np.arange(len(bb))[:,None,None],cost.shape);idx=np.flatnonzero(good)
    best=np.full((len(bb),cap+1),-np.inf);np.maximum.at(best,(rows.ravel()[idx],cost.ravel()[idx]),value.ravel()[idx])
    codes=np.full((len(bb),cap+1),-1,int)
    for ix in idx[value.ravel()[idx]==best[rows.ravel()[idx],cost.ravel()[idx]]]:
        row,c=rows.ravel()[ix],cost.ravel()[ix]
        if codes[row,c]<0:codes[row,c]=ix
    def block(row,c):
        _,ui,wi=np.unravel_index(codes[row,c],cost.shape)
        return dict(first=[first],second=np.flatnonzero(bb[row]).tolist(),third=sorted(set(oi[row,:ui].tolist()+oo[row,:wi].tolist())),count=int(c))
    result={}
    for c in range(1,cap+1):
        row=int(np.argmax(best[:,c]))
        if np.isfinite(best[row,c]):result[c,1]=(float(best[row,c]),[block(row,c)])
    # One isolated second can get a different third pool from the remaining
    # second subset. The sets never overlap, hence all stakes remain unique.
    aa,zz=np.where((sizes==1)[:,None]&~np.any(bb[:,None,:]&bb[None,:,:],axis=2))
    for c1 in range(1,min(n-2,cap-1)+1):
        vals=best[aa,c1,None]+best[zz,:cap+1-c1]
        for c2 in range(1,cap+1-c1):
            ix=int(np.argmax(vals[:,c2]));val=float(vals[ix,c2]);key=(c1+c2,2)
            if np.isfinite(val) and val>result.get(key,(-np.inf,None))[0]+1e-15:
                result[key]=(val,[block(aa[ix],c1),block(zz[ix],c2)])
    return result


def select(cars,q,cap,fallback):
    tri=list(itertools.permutations(cars,3));scores=dict(zip(tri,q));ix=np.asarray(list(itertools.permutations(range(len(cars)),3)));v=np.zeros((len(cars),)*3);v[tuple(ix.T)]=q
    states={(0,0):(0.,[])}
    for first in range(len(cars)):
        ps=profiles(v,first,cap);nxt=dict(states)
        for (cost,k),(val,blocks) in states.items():
            for (c,nb),(vv,bl) in ps.items():
                key=(cost+c,k+nb)
                if key[0]<=cap and key[1]<=4 and val+vv>nxt.get(key,(-np.inf,None))[0]+1e-15:nxt[key]=(val+vv,blocks+bl)
        states=nxt
    baseline=sum(scores[tuple(t)] for t in fallback['tickets']);answer=fallback
    for (cost,k),(val,blocks) in states.items():
        if cost!=cap or val<=baseline+1e-15:continue
        bl=[dict(first=[cars[j] for j in z['first']],second=[cars[j] for j in z['second']],third=[cars[j] for j in z['third']],count=z['count']) for z in blocks]
        ts=old.old.expand(bl);assert len(ts)==cap
        assert abs(sum(scores[t] for t in ts)-val)<1e-10
        answer=dict(blocks=bl,tickets=[list(t) for t in ts],search='pair_split_four_blocks');baseline=val
    return answer

