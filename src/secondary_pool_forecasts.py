"""Publish explicitly provisional trio/exacta bets from preclose racecard snapshots.

This is NOT the archived 2026-11-03 research model. Never backfill forecasts after
close or rewrite tickets that have already been publicly frozen.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import time
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
INPUT_PATH = Path('data/raw/secondary_pool_inputs.json')
SNAPSHOT_PATH = Path('data/raw/nexus_page_snapshots')


def now_jst():
    return datetime.now(JST)


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else default


def rows_digest(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_rows(data, meta):
    """Use the existing feature builder; never invent a company-model result."""
    from fetch_today_entries import _entry_rows_complete
    from race_features import build_entry_rows
    rows = build_entry_rows(data, race_date=meta['date'], venue=meta['venue'],
                            race_no=meta['race_number'], race_id=meta['race_id'],
                            source_url=meta['source_url'], include_results=False)
    if not _entry_rows_complete(rows)[0]:
        raise ValueError('incomplete independent racecard')
    # Preserve CSV-compatible units and make missing source values valid JSON.
    return [{k: '' if v is None or (isinstance(v, float) and not math.isfinite(v))
             else str(v) for k, v in row.items()} for row in rows]


def receipt_from_snapshot(snapshot, now, data=None):
    """Accept only intact, same-day, pre-close page captures already obtained."""
    from nexus_input_snapshots import canonical, timestamp
    meta, saved = snapshot['meta'], snapshot['data']
    captured = timestamp(snapshot['captured_at']).astimezone(JST)
    if (snapshot.get('source_kind') != 'existing_public_page_response'
            or meta['date'] != now.date().isoformat()
            or captured.date() != now.date() or captured > now
            or (not meta.get('cancel') and (captured.timestamp() >= float(meta['close_at'])
                or saved['race'].get('decidedAt') or saved.get('results') or saved.get('finalOdds')))):
        raise ValueError('not a same-day pre-close racecard')
    model_data = {k: saved[k] for k in ('players', 'entries', 'records', 'linePrediction') if k in saved}
    model_data['race'] = {k: saved['race'].get(k) for k in ('id', 'distance', 'startAt', 'closeAt')}
    model_data['schedule'] = {k: saved['schedule'].get(k) for k in ('date', 'cupId', 'index', 'day')}
    model_meta = {k: v for k, v in meta.items() if k not in ('source_url', 'session')}
    digest = hashlib.sha256(canonical([model_meta, model_data]).encode()).hexdigest()
    if digest != snapshot.get('input_sha256'):
        raise ValueError('independent input hash mismatch')
    rows = [] if meta.get('cancel') else source_rows(data or saved, meta)
    return {'race_id': meta['race_id'], 'date': meta['date'], 'source_url': meta['source_url'],
            'captured_at': captured.isoformat(), 'source_sha256': snapshot['sha256'],
            'rows_sha256': rows_digest(rows), 'rows': rows,
            'cancelled': bool(meta.get('cancel')), 'meta': meta}


def fetch_racecard_receipt(url, clock=now_jst):
    """Fetch the established public page only, without odds or model jobs."""
    from fetch_today_entries import extract_preloaded_state, http_get
    from nexus_input_snapshots import query_data, snapshot_from_state
    html = http_get(url, attempts=2)
    captured = clock()
    state = extract_preloaded_state(html)
    snapshot = snapshot_from_state(state, url, html, captured.isoformat())
    original = next(d for marker, d in query_data(state) if 'FETCH_KEIRIN_RACE' in marker
                    and isinstance(d, dict) and isinstance(d.get('race'), dict)
                    and isinstance(d.get('schedule'), dict) and 'entries' in d)
    return receipt_from_snapshot(snapshot, captured, data=original)


def valid_receipt(receipt, now):
    try:
        captured = datetime.fromisoformat(receipt['captured_at'].replace('Z', '+00:00'))
        rows = receipt['rows']
        return bool(captured.tzinfo is not None and captured.astimezone(JST).date() == now.date()
                    and captured <= now and receipt['date'] == now.date().isoformat()
                    and re.fullmatch(r'[a-f0-9]{64}', receipt['source_sha256'])
                    and rows_digest(rows) == receipt['rows_sha256']
                    and (rows or (receipt.get('cancelled') is True and receipt.get('meta',{}).get('cancel') is True
                                 and receipt['meta'].get('race_id')==receipt['race_id']
                                 and receipt['meta'].get('date')==receipt['date']))
                    and all(r.get('race_id') == receipt['race_id'] and r.get('date') == receipt['date']
                            and r.get('source_url') == receipt['source_url']
                            and num(r.get('close_at')) > captured.timestamp() for r in rows))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def load_legacy_inputs(root, today):
    path = root/'data/raw/today_entries.csv'
    entries = []
    if path.is_file():
        with path.open(encoding='utf-8-sig', newline='') as f:
            entries = list(csv.DictReader(f))
    company = read_json(root/'outputs/company/company_decision_predictions.json', {})
    decisions = {r['race_id']: r for r in company.get('predictions', [])
                 if r.get('date') == today} if company.get('coverage', {}).get('date') == today else {}
    grouped = defaultdict(list)
    for row in entries:
        if row.get('date') == today:
            grouped[row['race_id']].append(row)
    return grouped, decisions


def refresh_inputs(root=ROOT, clock=now_jst, index_loader=None,
                   fetcher=fetch_racecard_receipt, max_wall_seconds=600):
    """Recover missing inputs independently; a denied source has no alternate route."""
    root, now = Path(root), clock()
    if now.tzinfo is None:
        raise ValueError('timezone-aware clock required')
    today = now.astimezone(JST).date().isoformat()
    cache = read_json(root/INPUT_PATH, {})
    cache = cache if cache.get('date') == today else {'date': today, 'races': {}}
    receipts = {rid: r for rid, r in cache.get('races', {}).items() if valid_receipt(r, now)}
    grouped, decisions = load_legacy_inputs(root, today)
    feed = read_json(root/'outputs/trio_live.json', {})
    frozen_urls = {r.get('source_url') for r in feed.get('races', [])
                   if r.get('date') == today and r.get('forecast')}
    usable_urls = set(frozen_urls)
    for rid, rows in grouped.items():
        try:
            if build_row(rid, rows, decisions.get(rid), now).get('_scores'):
                usable_urls.add(rows[0]['source_url'])
        except (KeyError, TypeError, ValueError):
            pass
    errors = []
    saved_dir = root/SNAPSHOT_PATH/today
    for path in sorted((saved_dir/'preclose').glob('*.json')):
        try:
            saved = read_json(path)
            if saved['meta']['source_url'] not in usable_urls and saved['meta']['race_id'] not in receipts:
                receipt = receipt_from_snapshot(saved, now)
                receipts[receipt['race_id']] = receipt
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({'source': str(path.relative_to(root)), 'reason': str(exc)})
    usable_urls.update(r['source_url'] for r in receipts.values())
    saved_index = read_json(saved_dir/'index.json', {})
    expected_urls = set(cache.get('expected_urls', [])) | usable_urls
    if saved_index.get('date') == today:
        expected_urls.update(saved_index.get('urls', []))
    expected_urls.update(rows[0]['source_url'] for rows in grouped.values())
    saved_index_time=snapshot_time(saved_index.get('captured_at'))
    index_ok=bool(saved_index.get('date')==today and saved_index.get('urls') and saved_index_time
                  and saved_index_time.date()==now.date() and saved_index_time<=now)
    blocked=False
    try:
        if index_ok:
            links=saved_index['urls']
        elif index_loader is None:
            from fetch_today_entries import RACECARD_URL, collect_race_links, http_get
            links = collect_race_links(http_get(RACECARD_URL, attempts=2), today)
        else:
            links = index_loader(today)
        if not links:
            raise ValueError('empty racecard index')
        expected_urls.update(links)
        index_ok = True
    except Exception as exc:
        errors.append({'source': 'racecard_index', 'reason': f'{type(exc).__name__}: {exc}'})
        blocked=getattr(getattr(exc,'response',None),'status_code',None) in (401,403,477)
    closes={group[0]['source_url']:num(group[0].get('close_at')) for group in grouped.values()}
    saved_schedule=read_json(saved_dir/'schedule.json',{})
    if saved_schedule.get('date')==today:
        closes.update({r.get('source_url'):num(r.get('close_at')) for r in saved_schedule.get('races',[])})
    started = time.monotonic()
    for url in sorted(expected_urls-usable_urls):
        if blocked or time.monotonic()-started >= max_wall_seconds:
            break
        if closes.get(url) and closes[url]<=clock().timestamp()+300:
            continue
        try:
            receipt = fetcher(url, clock=clock)
            if not valid_receipt(receipt, clock()):
                raise ValueError('invalid independent receipt')
            receipts[receipt['race_id']] = receipt
            usable_urls.add(url)
        except Exception as exc:
            errors.append({'source': url, 'reason': f'{type(exc).__name__}: {exc}'[:300]})
            blocked=getattr(getattr(exc,'response',None),'status_code',None) in (401,403,477)
    result = {'schema': 1, 'date': today, 'updated_at': clock().isoformat(),
              'expected_urls': sorted(expected_urls), 'index_verified': index_ok,
              'races': receipts, 'errors': errors}
    path = root/INPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)
    print(f'POOL_INPUTS date={today} saved={len(receipts)} errors={len(errors)}', flush=True)
    return result


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


def build_row(race_id, rows, decision, now, independent=None):
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
    company_snap=snapshot_time(decision.get('snapshot_at_jst')) if decision else None
    if (company_snap is None or company_snap > now or company_snap.date().isoformat() != date
            or company_snap >= close_jst):
        company_snap=None
        decision=None
    snap=company_snap
    if independent and valid_receipt(independent, now):
        snap=snapshot_time(independent['captured_at'])
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
         'input_source':'independent_public_racecard' if independent else 'company_snapshot_and_entries',
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


def normalize_sessions(rows):
    """Keep a meeting together when its later races cross the hourly boundary."""
    meetings=defaultdict(list)
    for row in rows:
        meetings[(row.get('date'), row.get('venue_id') or str(row.get('race_id',''))[2:4])].append(row)
    for group in meetings.values():
        starts=[datetime.fromtimestamp(num(r.get('start_at')),JST) for r in group if num(r.get('start_at'))>0]
        if not starts:
            continue
        first=min(starts)
        hour=first.hour+first.minute/60
        session='midnight' if hour>=20 else 'night' if hour>=14 else 'morning' if hour<10.5 else 'day'
        for row in group:
            row['session']=session  # Published forecast objects are never altered.


def coverage_report(root, today, rows, grouped, cache, now):
    current={r['race_id']:r for r in rows if r.get('date')==today}
    expected=set(grouped)|set(current)
    known_urls={r.get('source_url') for r in current.values()}
    for group in grouped.values():
        known_urls.add(group[0].get('source_url'))
    schedule=read_json(root/SNAPSHOT_PATH/today/'schedule.json', {})
    schedule_rows={}
    company=read_json(root/'outputs/company/company_decision_predictions.json',{}).get('coverage',{})
    sources=[]
    if schedule.get('date')==today:
        sources.extend(schedule.get('races',[]))
    if company.get('date')==today:
        sources.extend(company.get('races',[]))
    for row in sources:
        if row.get('date')==today and row.get('race_id'):
            expected.add(str(row['race_id']))
            schedule_rows[str(row['race_id'])]=dict(row,venue_id=str(row['race_id'])[2:4])
            if row.get('source_url'):
                known_urls.add(row.get('source_url'))
    missing=sorted(rid for rid in expected if not current.get(rid,{}).get('forecast')
                   and current.get(rid,{}).get('status')!='cancelled')
    unresolved=sorted(set(cache.get('expected_urls', []))-known_urls)
    all_scheduled={**schedule_rows,**current}
    normalize_sessions(list(all_scheduled.values()))
    morning=[r for r in all_scheduled.values() if r.get('session')=='morning' and r.get('status')!='cancelled']
    schedule_time=snapshot_time(schedule.get('captured_at'))
    verified=bool(cache.get('index_verified') or (schedule_rows and schedule_time
                  and schedule_time.date()==now.date() and schedule_time<=now)
                  or (company.get('date')==today and company.get('races') and company.get('schedule_verified') is True))
    return {'date':today,'expected_races':len(expected)+len(unresolved),
            'forecasted_races':sum(bool(r.get('forecast')) for r in current.values()),
            'cancelled_races':sum(r.get('status')=='cancelled' for r in current.values()),
            'schedule_verified':verified,
            'missing_race_ids':missing,'unresolved_source_urls':unresolved,
            'morning_expected':len(morning),
            'morning_forecasted':sum(bool(r.get('forecast')) for r in morning),
            'morning_missing_race_ids':[r['race_id'] for r in morning if not r.get('forecast')]}


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
    now=now or now_jst()
    if now.tzinfo is None:
        raise ValueError('timezone-aware clock required')
    now=now.astimezone(JST)
    today=now.date().isoformat()
    grouped,decisions=load_legacy_inputs(root,today)
    cache=read_json(root/INPUT_PATH,{})
    cache=cache if cache.get('date')==today else {}
    independent={}
    cancelled={}
    for rid,receipt in cache.get('races',{}).items():
        if not valid_receipt(receipt,now):
            continue
        if receipt.get('cancelled'):
            cancelled[rid]=receipt
            grouped.pop(rid,None)
            continue
        company_snap=snapshot_time(decisions.get(rid,{}).get('snapshot_at_jst'))
        if (rid not in grouped or company_snap is None or company_snap>now or company_snap.date()!=now.date()
                or company_snap.timestamp()>=num(receipt['rows'][0].get('close_at'))):
            grouped[rid]=receipt['rows']
            independent[rid]=receipt
    if not grouped and not cancelled:
        raise ValueError('no verified source races for today; retry independent input refresh')
    complete=True
    for pool in MODELS:
        path=root/f'outputs/{pool}_live.json'
        previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if pool == 'exacta' and previous.get('publisher') == 'owner_exacta_hit_v1':
            print('POOL_FORECASTS exacta: owner fixed-model publisher active; no fallback replacement', flush=True)
            continue
        by_id={r['race_id']:r for r in previous.get('races',[]) if r.get('race_id') and
               r.get('date','')>= (now.date()-timedelta(days=3)).isoformat()}
        for rid,receipt in cancelled.items():
            meta=receipt['meta']
            row=by_id.get(rid) or {**meta,'riders':[],'lineup':[],'forecast':None,
                                  'settlement':None,'performance_eligible':False}
            row['status']='cancelled'
            row['note']='出走表で開催中止を確認しました。'
            by_id[rid]=row
        forecasted=0
        skipped=0
        for rid, group in sorted(grouped.items()):
            prior=by_id.get(rid)
            if prior and prior.get('forecast'):
                continue  # Immutable once published, including after close.
            try:
                row=build_row(rid,group,decisions.get(rid),now,independent.get(rid))
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
        normalize_sessions(rows)
        research=previous.get('research') or {}
        coverage=coverage_report(root,today,rows,grouped,cache,now)
        ready=bool(coverage['expected_races'] and coverage['schedule_verified']
                   and not coverage['missing_race_ids'] and not coverage['unresolved_source_urls'])
        complete=complete and ready
        message='暫定モデルによる予想（旧方式の年間研究値は適用できません）。'
        if not ready:
            message+=f" 本日{coverage['forecasted_races']}/{coverage['expected_races']}レース掲載。未掲載レースがあります。"
        health={'stage':'ready' if ready else 'warning',
                'message':message,
                'errors':skipped,'forecast_new':forecasted}
        feed={'schema':1,'model_id':MODELS[pool],'pool':pool,'generated_at':now.isoformat(),
              'today':today,'history_through':(now.date()-timedelta(days=1)).isoformat(),
              'races':rows,'observed':observed(rows,MODELS[pool]),'research':research,
              'reference_model_id':LEGACY[pool],'health':health,'coverage':coverage,
              'results_updated_at':previous.get('results_updated_at')}
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(feed,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
        print(f'POOL_FORECASTS {pool} date={today} new={forecasted} skipped={skipped} rows={len(rows)}',flush=True)
    return complete


def check_coverage(root=ROOT, now=None):
    now=now or now_jst()
    feed=read_json(Path(root)/'outputs/trio_live.json',{})
    coverage=feed.get('coverage',{})
    if (feed.get('today')!=now.astimezone(JST).date().isoformat()
            or feed.get('health',{}).get('stage')!='ready'
            or not coverage.get('expected_races')
            or not coverage.get('schedule_verified')
            or coverage.get('missing_race_ids') or coverage.get('unresolved_source_urls')):
        raise ValueError('incomplete daily trio coverage: '+json.dumps(coverage,ensure_ascii=False))
    print('POOL_COVERAGE '+json.dumps(coverage,ensure_ascii=False),flush=True)
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-inputs',action='store_true',help='Recover missing racecards independently before publishing.')
    parser.add_argument('--check-coverage',action='store_true',help='Check saved coverage after committing valid forecasts; do not regenerate.')
    args=parser.parse_args()
    if args.check_coverage:
        check_coverage()
    else:
        if args.refresh_inputs:
            refresh_inputs()
        run()  # Always write valid partial output; the post-push gate checks completeness.

