"""Public race source with bounded retry, original response cache and no auth bypass."""
import gzip,json,time,hashlib,threading
from datetime import datetime,timezone,timedelta,date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
JST=timezone(timedelta(hours=9));_lock=threading.Lock();_next=0.
def now():return datetime.now(JST)
def fetch(url):
    global _next
    for attempt in range(4):
        with _lock:
            wait=max(0,_next-time.monotonic());_next=max(time.monotonic(),_next)+.3
        if wait:time.sleep(wait)
        try:
            r=requests.get(url,headers={'User-Agent':'Mozilla/5.0 (compatible; keirin-ai-auto/1.0)'},timeout=25)
            if r.status_code in (401,403):r.raise_for_status()
            if r.status_code==429 or r.status_code>=500:
                if attempt<3:time.sleep(min(30,2**attempt+1));continue
            r.raise_for_status();return r.content
        except requests.HTTPError as e:
            if e.response.status_code in (401,403) or attempt==3:raise
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(2**attempt)
    raise RuntimeError('unavailable source')
def decode(b):
    x=json.loads(b);return x.get('data',x)
def calendar(start,end):
    out={};month=date.fromisoformat(start).replace(day=1);last=date.fromisoformat(end)
    while month<=last:
        d=decode(fetch('https://api.winticket.jp/v1/keirin/cups/-/schedules?year_month='+month.strftime('%Y%m')))
        sd={str(x['id']):x for x in d['schedules']};cp={str(x['id']):x for x in d['cups']};vn={str(x['id']):x for x in d['venues']}
        for r in d['races']:
            s=sd[str(r['scheduleId'])];c=cp[str(s['cupId'])];v=vn[str(c['venueId'])];ds=str(s['date']);ds=ds[:4]+'-'+ds[4:6]+'-'+ds[6:]
            if not start<=ds<=end:continue
            out[r['id']]=dict(race_id=r['id'],date=ds,cup_id=c['id'],index=s['index'],venue_id=str(c['venueId']).zfill(2),venue=v['name'],race_class=r['class'],race_type=r['raceType'],graded=r['isGradeRace'],cup_grade=c['grade'],cancel=r['cancel'],race_number=r['number'],start_at=r.get('startAt'),close_at=r.get('closeAt'),source_url=f"https://www.winticket.jp/keirin/{v['slug']}/racecard/{c['id']}/{s['index']}/{r['number']}")
        month=(month.replace(day=28)+timedelta(days=4)).replace(day=1)
    # Group the whole meeting session; a morning meeting does not change category at noon.
    groups={}
    for m in out.values():groups.setdefault((m['date'],m['cup_id']),[]).append(m)
    for rows in groups.values():
        hours=[datetime.fromtimestamp(m['start_at'],JST).hour+datetime.fromtimestamp(m['start_at'],JST).minute/60 for m in rows if m['start_at']]
        first=min(hours) if hours else None;session=None if first is None else 'midnight' if first>=20 else 'night' if first>=14 else 'morning' if first<10.5 else 'day'
        for m in rows:m['session']=session
    return sorted(out.values(),key=lambda x:(x['date'],x['race_id']))
def race(meta,cache,past=False):
    path=Path(cache)/meta['date']/(meta['race_id']+'.json.gz');path.parent.mkdir(parents=True,exist_ok=True)
    if past and path.exists():
        with gzip.open(path,'rt',encoding='utf8') as f:z=json.load(f)
        return z
    url=f"https://api.winticket.jp/v1/keirin/cups/{meta['cup_id']}/schedules/{meta['index']}/races/{meta['race_number']}"
    content=fetch(url);d=decode(content)
    if d.get('race',{}).get('id')!=meta['race_id'] or str(d['schedule']['date'])!=meta['date'].replace('-',''):raise ValueError('source identity mismatch')
    z=dict(captured_at=now().isoformat(),url=url,sha256=hashlib.sha256(content).hexdigest(),data=d)
    with gzip.open(path,'wt',encoding='utf8') as f:json.dump(z,f,ensure_ascii=False,separators=(',',':'))
    return z

def outcomes(d):
    quotes=d.get('trifecta',[]);wins=set(d.get('trifectaWinningOddsIds') or [])
    payouts={'-'.join(map(str,q['key'])):q['payoffUnitPrice'] for q in quotes if q.get('id') in wins and q.get('payoffUnitPrice',0)>0}
    refund_all=bool(d['race'].get('cancel')) or (d.get('trifectaPayoffStatus')==3 and bool(d['race'].get('decidedAt')) and not wins)
    refunds=['-'.join(map(str,q['key'])) for q in quotes if q.get('absent') is True]
    decided=bool(d['race'].get('decidedAt'))
    # No invented result: require the whole winning payoff set.
    complete=decided and bool(wins) and len(payouts)==len(wins)
    return payouts,refunds,refund_all,complete
