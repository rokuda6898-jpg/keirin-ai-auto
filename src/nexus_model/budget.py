"""Discrete low-odds-heavy maximin allocation; preclose quotes only."""
from fractions import Fraction
from math import ceil

def allocate(tickets,odds,budget):
    if budget not in range(5000,10001,1000):raise ValueError('budget must be 5000..10000 in 1000 yen steps')
    tickets=list(tickets)
    if len(set(tickets))!=len(tickets) or not tickets or len(tickets)*100>budget:raise ValueError('invalid ticket set')
    try:
        prices=[Fraction(str(odds[t])) for t in tickets]
        if any(p<=0 for p in prices):raise ValueError()
    except (KeyError,ValueError,ZeroDivisionError,TypeError):return dict(available=False,reason='購入前オッズ待ち')
    units=budget//100;levels=sorted({p*j for p in prices for j in range(1,units-len(tickets)+2)});lo,hi=0,len(levels)
    while lo<hi:
        mid=(lo+hi)//2
        if sum(max(1,ceil(levels[mid]/p)) for p in prices)<=units:lo=mid+1
        else:hi=mid
    target=levels[lo-1];amounts=[max(1,ceil(target/p)) for p in prices]
    for _ in range(units-sum(amounts)):
        j=min(range(len(tickets)),key=lambda j:(prices[j]*amounts[j],prices[j],tickets[j]));amounts[j]+=1
    minimum=min(p*a*100 for p,a in zip(prices,amounts))
    return dict(available=True,budget=budget,stakes=dict(zip(tickets,[a*100 for a in amounts])),minimum_return=float(minimum),quote_break_even=minimum>=budget)

def settle(stakes,payouts,refunded=(),refund_all=False):
    if refund_all:return dict(stake=sum(stakes.values()),returned=sum(stakes.values()),hit=False,void=True)
    refunded=set(refunded);stake=sum(stakes.values());refund=sum(v for t,v in stakes.items() if t in refunded)
    won=sum(float(payouts[t])*v/100 for t,v in stakes.items() if t in payouts and t not in refunded)
    return dict(stake=stake,returned=won+refund,refund=refund,hit=any(t in payouts for t in stakes if t not in refunded),void=False)
