"""Publish explicitly provisional trio/exacta bets from preclose racecard snapshots.

This is NOT the archived 2026-11-03 research model. Never backfill forecasts after
close or rewrite tickets that have already been publicly frozen.
"""
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
JST = ZoneInfo('Asia/Tokyo')
MODELS = {'trio': 'trio_snapshot_fallback_v1_20261104', 'exacta': 'exacta_snapshot_fallback_v1_20261104'}
LEGACY = {'trio': 'trio_selected9891_454_20261103', 'exacta': 'exacta_native_highroi_345_20261103'}
COUNTS = {'trio': {'MAIN': 4, 'MEDIUM': 5, 'HIGH': 4}, 'exacta': {'MAIN': 3, 'MEDIUM': 4, 'HIGH': 5}}


def num(v, default=0.0):
    try:
        n = float(v)
        return n if math.isfinite(n) else default
    except (ValueError, TypeError):
        return default


def snapshot_time(v):
    try:
        stamp = datetime.fromisoformat(str(v).replace('Z', '+00:00'))
        return stamp.astimezone(JST) if stamp.tzinfo else stamp.replace(tzinfo=JST)
    except (TypeError, ValueError):
        return None


def score(row, local_mean):
    # Relative, hand-weighted performance proxy. Not a calibrated probability.
    score_delta = max(-15, min(15, num(row.get('score'))-local_mean))
    value = (0.06 + 0.43*num(row.get('win_rate')) + 0.29*num(row.get('place2_rate'))
             + 0.24*num(row.get('place3_rate')) + .007*score_delta
             + .02/max(1, num(row.get('recent_avg_finish'), 5)))
    return max(.025, value)


def form_tickets(rows, pool, band, company_top):
    cars = [int(r['car_no']) for r in rows]
    mean = sum(num(r.get('score')) for r in rows)/len(rows)
    weights = {int(r['car_no']): score(r, mean) for r in rows}
    ordered = sorted(cars, key=lambda car: (-weights[car], car))
    marks = [dict(car=car, mark=['◎','○','▲','△','☆'][i] if i<5 else '印なし') for i, car in enumerate(ordered)]
    unmarked = ordered[5:]
    ranked = []
    if pool == 'trio':
        for triple in itertools.combinations(sorted(cars), 3):
            s = math.prod(weights[c] for c in triple)
            # Only the independently saved company candidate is a bonus.
            support = max((len(company_top)-i for i, t in enumerate(company_top)
                           if len(t)==3 and tuple(sorted(t))==triple), default=0)
            ranked.append((s * (1+0.06*support), tuple(triple)))
    else:
        for a in cars:
            for b in cars:
                if a == b:
                    continue
                support = max((len(company_top)-i for i, t in enumerate(company_top)
                               if len(t)==3 and t[:2] == (a, b)), default=0)
                ranked.append((weights[a]**1.35 * weights[b]**.85 * (1+.05*support), (a,b)))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    amount = COUNTS[pool][band]
    selected = [t for _,t in ranked[:amount]]
    # Fulfill exactly the published ticket-level policies, without doubling a ticket.
    required = [lambda t: bool(set(t).intersection(unmarked))] if unmarked else []
    if pool=='trio':
        third=ordered[2]
        required.append(lambda t: third in t)
    if required and not all(any(f(t) for t in selected) for f in required):
        for predicate in required:
            if any(predicate(t) for t in selected):
                continue
            current=selected[-1]
            remaining=[f for f in required if f is not predicate]
            alternative=next((t for _,t in ranked if t not in selected and predicate(t) and
                             all(any(f(x) for x in selected[:-1]) or f(t) for f in remaining)),None)
            if alternative is not None:
                selected[-1]=alternative
            else:
                selected[-1]=current
    if pool=='trio':
        blocks=trio_blocks(selected, ranked, amount)
    else:
        blocks=[dict(a=[a],b=[b]) for a,b in selected]
    tickets=expand(blocks,pool)
    if len(tickets)!=amount:
        raise ValueError('formation ticket count mismatch')
    if unmarked and not any(c in t for t in tickets for c in unmarked):
        raise ValueError('missing unmarked rider')
    if pool=='trio' and not any(ordered[2] in t for t in tickets):
        raise ValueError('missing ▲')
    return marks, unmarked, blocks, tickets


def trio_blocks(selected, ranked, amount):
    def pack(tickets):
        pair_hits=defaultdict(list)
        for t in tickets:
            for pair in itertools.combinations(t,2):
                pair_hits[pair].append(t)
        best=max(pair_hits.items(),key=lambda kv: (len(kv[1]),kv[0]))
        pair, group=best
        if len(tickets)>4 and len(group)<2:
            return None
        first=dict(a=[pair[0]],b=[pair[1]],c=sorted(t[0] if t[1:]==pair else next(c for c in t if c not in pair) for t in group))
        remaining=[dict(a=[t[0]],b=[t[1]],c=[t[2]]) for t in tickets if t not in group]
        return [first]+remaining
    blocks=pack(selected)
    if blocks is not None and len(blocks)<=4:
        return blocks
    # Rare disjoint-pair top five: swap the weakest choice to make a pair group.
    for _, candidate in ranked:
        if candidate in selected:
            continue
        alternative=selected[:-1]+[candidate]
        b=pack(alternative)
        if b is not None and len(b)<=4:
            return b
    raise ValueError('cannot form <=4 formation blocks')


def expand(blocks, pool):
    tickets=set()
    for b in blocks:
        for a in b['a']:
            for c in b['b']:
                for d in b['c'] if pool=='trio' else [None]:
                    t=[a,c,d] if pool=='trio' else [a,c]
                    if len(set(t)) != len(t):
                        continue
                    if pool=='trio':t.sort()
                    tickets.add(tuple(t))
    return sorted(tickets,key=lambda t:'-'.join(map(str,t)))


def tier(rows):
    mean=sum(num(r.get('score')) for r in rows)/len(rows)
    strengths=sorted((score(r, mean) for r in rows),reverse=True)
    lead = strengths[0]/sum(strengths)
    return 'MAIN' if lead >= .28 else 'MEDIUM' if lead >= .23 else 'HIGH'


def build_row(race_id, rows, decision, now):
    rows=sorted(rows,key=lambda r:int(r['car_no']))
    date=rows[0]['date']
    close=int(num(rows[0].get('close_at')))
    start=int(num(rows[0].get('start_at')))
    if close <= 0 or not all(int(num(r.get('close_at'))) == close for r in rows):
        raise ValueError(f'inconsistent close time for {race_id}')
    if len(rows)<5 or len(set(int(r['car_no']) for r in rows))!=len(rows):
        raise ValueError(f'invalid active lineup for {race_id}')
    original_url=rows[0].get('source_url') or ''
    p=urlparse(original_url)
    tail=p.path.strip('/').split('/')
    if p.hostname!='www.winticket.jp' or len(tail)<6 or tail[-3].isdigit() is False:
        raise ValueError(f'unknown racecard origin for {race_id}')
    close_jst=datetime.fromtimestamp(close,JST)
    hour=datetime.fromtimestamp(start,JST).hour
    session='morning' if hour<11 else 'day' if hour<16 else 'night' if hour<20 else 'midnight'
    groups=defaultdict(list)
    for rider in rows:
        try:
            line=float(rider.get('line_id') or 'nan')
            position=float(rider.get('line_position') or 'nan')
            if math.isfinite(line) and math.isfinite(position):
                groups[line].append((position,int(rider['car_no'])))
        except ValueError:
            pass
    lineup=[[[c] for _,c in sorted(group)] for _,group in sorted(groups.items())] if len(groups)>1 else []
    snap=snapshot_time(decision.get('snapshot_at_jst')) if decision else None
    if snap is None or snap > now or snap.date().isoformat() != date:
        snap=None
    top=[]
    for ticket in (decision.get('top12') or []) if decision else []:
        try:
            cars=tuple(map(int,ticket.split('-')))
            if len(cars)==3 and set(cars).issubset({int(r['car_no']) for r in rows}):top.append(cars)
        except (ValueError,TypeError):
            continue
    encoded=json.dumps({'entries':rows,'top12':top},sort_keys=True,ensure_ascii=False)
    digest=hashlib.sha256(encoded.encode('utf-8')).hexdigest()
    out={'race_id':race_id,'date':date,'cup_id':tail[-3],
         'index':int(tail[-2]),'venue_id':race_id[2:4],
         'venue':rows[0]['venue'],'race_class':rows[0].get('race_class') or '',
         'race_type':rows[0].get('race_type') or '',
         'graded':str(rows[0].get('is_grade_race')) in ('1','True','true'),
         'race_number':int(rows[0]['race_no']),'start_at':start,'close_at':close,
         'session':session,'source_url':original_url,'status':'forecast',
         'riders':[dict(car=int(r['car_no']),name=r.get('player_name') or '',
                        prefecture=r.get('prefecture') or '',score=num(r.get('score')),
                        style=r.get('style') or '') for r in rows],
         'lineup':lineup,'lineup_unknown':[], 'lineup_errors':[],
         'captured_at':snap.isoformat() if snap else None,
         'source_state':{'active':[[int(r['car_no']),str(r.get('player_id') or '')] for r in rows],
                         'groups':lineup,'unknown':[],'errors':[],'cancel':False,'close_at':close},
         'forecast':None,'settlement':None,'performance_eligible':False}
    if not snap:
        out['note']='有効な締切前入力スナップショットがありません。'
    elif close_jst - now <= timedelta(minutes=5):
        out['note']='締切直前または締切済みのため、新しい予想を遡って掲載しません。'
    else:
        band=tier(rows)
        out['_scores']=(band,top,digest,snap)
    return out


def observed(rows, model):
    stats={'races':0,'hits':0,'HIGH_hits':0,'stake':0,'returned':0,
           'hit_pct':None,'roi_pct':None,'void':0,'pending':0,'model_id':model}
    for r in rows:
        if not r.get('performance_eligible') or not r.get('forecast'):
            continue
        settled=r.get('settlement')
        if not isinstance(settled,dict):
            stats['pending']+=1
            continue
        if settled.get('status')=='void':
            stats['void']+=1
            continue
        if settled.get('status') not in ('hit','miss'):
            stats['pending']+=1
            continue
        stats['races']+=1
        stats['hits']+=int(settled['status']=='hit')
        stats['HIGH_hits']+=int(bool(settled.get('HIGH_hit')))
        stats['stake']+=int(settled.get('stake') or 0)
        stats['returned']+=int(settled.get('returned') or 0)
    if stats['races']:
        stats['hit_pct']=100*stats['hits']/stats['races']
    if stats['stake']:
        stats['roi_pct']=100*stats['returned']/stats['stake']
    return stats


def run(root=ROOT, now=None):
    root=Path(root)
    now=now or datetime.now(JST)
    today=now.date().isoformat()
    entry=root/'data/raw/today_entries.csv'
    company=root/'outputs/company/company_decision_predictions.json'
    if not entry.is_file() or not company.is_file():
        raise FileNotFoundError('daily entry data and frozen company input are required')
    with entry.open(encoding='utf-8-sig',newline='') as f:
        entries=list(csv.DictReader(f))
    dec=json.loads(company.read_text(encoding='utf-8'))
    if dec.get('coverage',{}).get('date')!=today:
        raise ValueError('company snapshot is not for today')
    decisions={r['race_id']:r for r in dec.get('predictions',[]) if r.get('date')==today}
    grouped=defaultdict(list)
    for row in entries:
        if row.get('date')==today:
            grouped[row['race_id']].append(row)
    if not grouped or not decisions:
        raise ValueError('no independent pre-close source races for today')
    for pool in MODELS:
        path=root/f'outputs/{pool}_live.json'
        previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if pool == 'exacta' and previous.get('publisher') == 'owner_exacta_hit_v1':
            print('POOL_FORECASTS exacta: owner fixed-model publisher active; no fallback replacement', flush=True)
            continue
        by_id={r['race_id']:r for r in previous.get('races',[]) if r.get('race_id') and
               r.get('date','')>= (now.date()-timedelta(days=3)).isoformat()}
        forecasted=0
        skipped=0
        for rid, group in sorted(grouped.items()):
            prior=by_id.get(rid)
            if prior and prior.get('forecast'):
                continue  # Immutable once published, including after close.
            try:
                row=build_row(rid,group,decisions.get(rid),now)
                inputs=row.pop('_scores',None)
                if inputs:
                    band, top, digest, snap=inputs
                    marks,unmarked,blocks, tickets=form_tickets(group,pool,band,top)
                    row['forecast']={
                        'model_id':MODELS[pool],'pool':pool,'category':band,
                        'count':len(tickets),'unit_yen':100,'total_yen':100*len(tickets),
                        'blocks':blocks,'tickets':['-'.join(map(str,t)) for t in tickets],
                        'marks':marks,'unmarked':unmarked,'required_triangle':pool=='trio',
                        'EX_overlay':False,'captured_at':snap.isoformat(),
                        'saved_at':now.isoformat(),'history_through':(now.date()-timedelta(days=1)).isoformat(),
                        'source_sha256':digest,'source_url':group[0].get('source_url'),
                        'method_note':'未検証の暫定算出方式。旧モデルとは異なります。'}
                    row['first_published_at']=now.isoformat()
                    row['performance_eligible']=True
                    forecasted+=1
                else:
                    skipped+=1
            except Exception as exc:
                print(f'{pool} {rid} skipped: {type(exc).__name__}: {exc}',flush=True)
                skipped+=1
                continue
            by_id[rid]=row
        rows=sorted(by_id.values(),key=lambda r:(r.get('date',''),r.get('close_at',0),r.get('race_id','')))
        research=previous.get('research') or {}
        health={'stage':'ready' if any(r.get('date')==today and r.get('forecast') for r in rows) else 'warning',
                'message':'暫定モデルによる予想（旧方式の年間研究値は適用できません）。',
                'errors':skipped,'forecast_new':forecasted}
        feed={'schema':1,'model_id':MODELS[pool],'pool':pool,'generated_at':now.isoformat(),
              'today':today,'history_through':(now.date()-timedelta(days=1)).isoformat(),
              'races':rows,'observed':observed(rows,MODELS[pool]),'research':research,
              'reference_model_id':LEGACY[pool],'health':health,
              'results_updated_at':previous.get('results_updated_at')}
        path.write_text(json.dumps(feed,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
        print(f'POOL_FORECASTS {pool} date={today} new={forecasted} skipped={skipped} rows={len(rows)}',flush=True)
    return True


if __name__=='__main__':
    run()

