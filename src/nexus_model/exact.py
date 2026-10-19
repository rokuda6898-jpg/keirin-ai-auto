"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta


def subsets(bits):
    s=bits
    while s:
        yield s;s=(s-1)&bits


def rectangles(first,pairs):
    # Enumerate every second/third subset supported by the selected tickets.
    bs=sorted({b for b,c in pairs});cs=sorted({c for b,c in pairs})
    cm={c:1<<i for i,c in enumerate(cs)};idx={bc:i for i,bc in enumerate(sorted(pairs))}
    row={b:sum(cm[c] for bb,c in pairs if bb==b) for b in bs};full=(1<<len(cs))-1
    out={}
    for bm in range(1,1<<len(bs)):
        bb=[b for i,b in enumerate(bs) if bm>>i&1];allow=full
        for b in bb:allow &= row[b]|cm.get(b,0) # same-car repetitions do not generate a ticket
        for cc_mask in subsets(allow):
            cc=[c for i,c in enumerate(cs) if cc_mask>>i&1]
            pp=[(b,c) for b in bb for c in cc if b!=c]
            if not pp:continue
            assert all(p in idx for p in pp)
            mask=sum(1<<idx[p] for p in pp)
            if mask not in out:out[mask]=dict(first=[first],second=bb,third=cc,count=len(pp))
    return out


def cover(rects,n,limit):
    full=(1<<n)-1;by=[[] for _ in range(n)]
    for mask in rects:
        for i in range(n):
            if mask>>i&1:by[i].append(mask)
    for opts in by:opts.sort(key=lambda m:(-m.bit_count(),m))
    @functools.lru_cache(None)
    def solve(rem,k):
        if not rem:return ()
        if k==0:return None
        if rem in rects:return (rem,)
        if k==1:return None
        opts=min(([m for m in by[i] if m&rem==m] for i in range(n) if rem>>i&1),key=len)
        for mask in opts:
            tail=solve(rem^mask,k-1)
            if tail is not None:return (mask,)+tail
        return None
    for k in range(1,limit+1):
        found=solve(full,k)
        if found is not None:return [rects[m] for m in found]
    return None


@functools.lru_cache(maxsize=60000)
def pack(tickets,maxblocks=4):
    first=sorted({t[0] for t in tickets})
    if len(first)>maxblocks:return None
    groups=[]
    for f in first:
        pairs={(b,c) for a,b,c in tickets if a==f}
        rects=rectangles(f,pairs);block=cover(rects,len(pairs),maxblocks-len(first)+1)
        if block is None:return None
        groups.extend(block)
    if len(groups)>maxblocks:return None
    assert expand(groups)==set(tickets)
    return groups


def expand(blocks):
    ts=[t for b in blocks for t in itertools.product(b['first'],b['second'],b['third']) if len(set(t))==3]
    assert len(ts)==len(set(ts)) and sum(b['count'] for b in blocks)==len(ts)
    return set(ts)

