"""Collect EX from historical racecards; never claim pre-race availability."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time, urllib.error
from datetime import datetime, timezone
from pathlib import Path
from winticket_ex import fetch_public, snapshot

# These groups were extracted from the supplied 60,092-race history manifest.
GROUPS = [
 ('2026-10-06','iwakidaira','2026100413',3,'1-11'),
 ('2026-10-06','maebashi','2026100622',1,'1-9'),
 ('2026-10-06','omiya','2026100625',1,'1-12'),
 ('2026-10-06','ogaki','2026100644',1,'1-9'),
 ('2026-10-06','nara','2026100553',2,'1-12'),
 ('2026-10-06','hofu','2026100463',3,'1-7'),
 ('2026-10-06','matsuyama','2026100475',3,'1-12'),
 ('2026-10-07','maebashi','2026100622',2,'1-9'),
 ('2026-10-07','omiya','2026100625',2,'1-12'),
 ('2026-10-07','shizuoka','2026100738',1,'1-8,10-11'),
 ('2026-10-07','ogaki','2026100644',2,'1-9'),
 ('2026-10-07','yokkaichi','2026100748',1,'1-12'),
 ('2026-10-07','nara','2026100553',3,'1-12'),
 ('2026-10-07','kochi','2026100774',1,'1-7'),
 ('2026-10-08','maebashi','2026100622',3,'1-9'),
 ('2026-10-08','omiya','2026100625',3,'1-12'),
 ('2026-10-08','hiratsuka','2026100835',1,'1-9,11-12'),
 ('2026-10-08','shizuoka','2026100738',2,'1-11'),
 ('2026-10-08','ogaki','2026100644',3,'1-9'),
 ('2026-10-08','yokkaichi','2026100748',2,'1-12'),
 ('2026-10-08','kochi','2026100774',2,'1-7'),
]

def manifest():
    rows=[]
    for day, venue, cup, index, spec in GROUPS:
        for part in spec.split(','):
            lo, hi = map(int, part.split('-')) if '-' in part else (int(part), int(part))
            for number in range(lo, hi+1):
                rows.append({'date':day, 'race_id':f'{number:02d}{cup[-2:]}{day.replace("-", "")}',
                    'source_url':f'https://www.winticket.jp/keirin/{venue}/racecard/{cup}/{index}/{number}'})
    assert len(rows)==214 and len({x['race_id'] for x in rows})==214
    return sorted(rows,key=lambda x:(x['date'],x['race_id']),reverse=True)

def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()
    tmp=path.with_suffix('.tmp'); tmp.write_bytes(raw); os.replace(tmp,path)
    return hashlib.sha256(raw).hexdigest()

def wrap_observation(snap,day):
    if snap['preclose_eligible']:
        raise ValueError('historical collection unexpectedly predates close')
    return {'schema':'historical_ex_candidate_v1','historical_date':day,
        'temporal_status':'retrospective_asof_unverified','eligible_for_backtest':False,
        'observation':snap}

def live_inventory(root):
    by_date={}
    for p in (root/'outputs/ex_data/snapshots').glob('*/*.json'):
        s=json.loads(p.read_text())
        if not s.get('preclose_eligible'): continue
        day=p.parent.name
        item=by_date.setdefault(day,{'unique_race_ids':set(),'snapshots':0,'latest':{}})
        item['unique_race_ids'].add(s['race_id']); item['snapshots']+=1
        old=item['latest'].get(s['race_id'])
        if old is None or old['captured_at']<s['captured_at']: item['latest'][s['race_id']]=s
    result={}
    for day,item in by_date.items():
        snaps=list(item['latest'].values())
        result[day]={'unique_races':len(item['unique_race_ids']),'snapshots':item['snapshots'],
            'active_rider_rows':sum(x['active_entries'] for x in snaps),
            'complete_races':sum(x['coverage']=='complete' for x in snaps),
            'available_metrics':sum(x['available_metrics'] for x in snaps),
            'expected_metrics':sum(x['expected_metrics'] for x in snaps)}
    return result

def collect(root,limit=214,interval=1.0):
    if not 1<=limit<=214 or interval<1: raise ValueError('invalid bounds')
    root=Path(root); out=root/'outputs/ex_data/historical/20261006_20261008'
    out.mkdir(parents=True,exist_ok=True)
    rows=manifest()[:limit]
    with (out/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['date','race_id','source_url']);w.writeheader();w.writerows(rows)
    report={'requested':len(rows),'date_start':min(x['date'] for x in rows),
        'date_end':max(x['date'] for x in rows),'started_at':datetime.now(timezone.utc).isoformat(),
        'retrieved':0,'cached':0,'complete_races':0,'active_rider_rows':0,
        'available_metrics':0,'expected_metrics':0,'observed_metrics':0,
        'failures':[],'races':[],'eligible_for_backtest':0,'forecast_model_changed':False,
        'temporal_status':'retrospective_asof_unverified','live_archive_inventory':live_inventory(root)}
    consecutive=0
    for i,row in enumerate(rows):
        try:
            folder=out/'observations';folder.mkdir(exist_ok=True)
            cached=[]
            for p in folder.glob(row['race_id']+'_*.json'):
                v=json.loads(p.read_text());s=v['observation']
                if v.get('eligible_for_backtest') is False and s['coverage']=='complete' and s['race_id']==row['race_id']:
                    cached.append((p,v))
            if cached:
                path,payload=cached[-1]; s=payload['observation'];report['cached']+=1
            else:
                for attempt in range(2):
                    try: data,ts=fetch_public(row['source_url']);break
                    except urllib.error.HTTPError as e:
                        if e.code not in (429,500,502,503,504) or attempt: raise
                        wait=float(e.headers.get('Retry-After','5'))
                        if wait>60: raise RuntimeError('server asked to defer; stop this attempt') from e
                        time.sleep(max(5,wait))
                s=snapshot(data,row['source_url'],ts,row['race_id'])
                payload=wrap_observation(s,row['date'])
                digest=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
                path=folder/f'{s["race_id"]}_{digest[:20]}.json'
                atomic_json(path,payload);report['retrieved']+=1
            report['complete_races']+=int(s['coverage']=='complete')
            for dest,source in [('active_rider_rows','active_entries'),('available_metrics','available_metrics'),
                                ('expected_metrics','expected_metrics'),('observed_metrics','observed_metrics')]:
                report[dest]+=s[source]
            report['races'].append({'race_id':s['race_id'],'coverage':s['coverage'],
                'active_entries':s['active_entries'],'captured_at':s['captured_at'],
                'path':str(path.relative_to(root))})
            if s['coverage']!='complete':report['failures'].append({'race_id':row['race_id'],'reason':'partial_ex'})
            consecutive=0
        except Exception as e:
            consecutive+=1
            report['failures'].append({'race_id':row['race_id'],'reason':f'{type(e).__name__}: {e}'[:240]})
        report['attempted']=i+1
        atomic_json(out/'summary.json',report)
        if (i+1)%20==0 or consecutive:print('PROGRESS',i+1,'retrieved',report['retrieved'],'failures',len(report['failures']),flush=True)
        if consecutive>=3:
            report['stopped_reason']='three_consecutive_errors'; break
        time.sleep(interval)
    report['finished_at']=datetime.now(timezone.utc).isoformat()
    atomic_json(out/'summary.json',report)
    flat=[]
    for item in report['races']:
        v=json.loads((root/item['path']).read_text());s=v['observation']
        for rider in s['riders']:
            f={'race_id':s['race_id'],'date':v['historical_date'],'car_no':rider['car_no'],
                'player_id':rider['player_id'],'captured_at':s['captured_at'],
                'temporal_status':v['temporal_status'],'eligible_for_backtest':False}
            for name,m in rider['metrics'].items():
                for key in ('total','succeeded','rate','status'):f[f'ex_{name}_{key}']=m[key]
            flat.append(f)
    if flat:
        with (out/'ex_rider_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps({k:v for k,v in report.items() if k!='races'},ensure_ascii=False),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--limit',type=int,default=214);p.add_argument('--interval',type=float,default=1.0)
    a=p.parse_args();r=collect(a.root,a.limit,a.interval)
    raise SystemExit(1 if r['failures'] or r.get('attempted',0)<r['requested'] else 0)
