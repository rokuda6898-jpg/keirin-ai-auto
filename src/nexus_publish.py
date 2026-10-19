"""Publish saved model forecasts, whole formations, timestamped stakes and outcomes."""
import argparse,json,gzip,hashlib,os,traceback
from pathlib import Path
from datetime import date,datetime,timedelta
from concurrent.futures import ThreadPoolExecutor
from nexus_model import source,inputs,update_history,budget
from nexus_model.engine import Engine
from nexus_model.publication import is_open_prediction, allocation_snapshot, settle_prediction
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';MODEL=ROOT/'models/nexus53';STATE=ROOT/'data/nexus';CACHE=STATE/'sources'
def read(path,default=None):return json.loads(path.read_text('utf-8-sig')) if path.exists() else default
def save(path,v):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf8');os.replace(temp,path)
def iso(x):return datetime.fromtimestamp(x,source.JST).isoformat() if x else None
def run(history_only=False):
    today=source.now().date().isoformat();STATE.mkdir(parents=True,exist_ok=True)
    hp=STATE/'history.json.gz';H=inputs.History.load(hp if hp.exists() else MODEL/'history.json.gz');start=(date.fromisoformat(H.data['through'])+timedelta(days=1)).isoformat()
    metas=source.calendar(min(start,today),today)
    update_history.update(H,metas,CACHE,hp,today)
    if history_only:return
    manifest=read(MODEL/'manifest.json');model=Engine(MODEL);todaymetas=[m for m in metas if m['date']==today]
    ledger=OUT/'nexus_forecasts';ledger.mkdir(exist_ok=True);existing={}
    for path in sorted(ledger.glob('*.json')):
        for r in read(path,[]):existing[r['race_id']]=r
    rows=[];errors=[]
    def get(m):
        try:return m,source.race(m,CACHE)
        except Exception as e:return m,dict(error=str(e))
    with ThreadPoolExecutor(max_workers=4) as ex:fetched=list(ex.map(get,todaymetas))
    for m,z in fetched:
        rid=m['race_id'];old=existing.get(rid,{});r=dict(old);r.update(race_id=rid,date=today,venue=m['venue'],race_no=m['race_number'],class_name=m['race_class'],stage=m['race_type'],session=m['session'],source_url=m['source_url'],start_at=iso(m['start_at']),close_at=iso(m['close_at']))
        r['closed']=not m['close_at'] or source.now().timestamp()>=m['close_at'];r['cancelled']=m['cancel']
        if 'error' in z:
            errors.append(dict(race_id=rid,error=z['error']));r['message']='データ取得待ち';rows.append(r);continue
        d=z['data'];race=d['race'];captured=datetime.fromisoformat(z['captured_at']);r.update(start_at=iso(race.get('startAt')),close_at=iso(race.get('closeAt')),closed=not race.get('closeAt') or source.now().timestamp()>=race['closeAt'],cancelled=bool(race.get('cancel')))
        players={x['id']:x for x in d['players']};r['riders']=[dict(car=x['number'],name=players.get(x['playerId'],{}).get('name',''),pid=x['playerId']) for x in d['entries'] if not x['absent']]
        active={x['car'] for x in r['riders']};p=r.get('prediction')
        # Re-evaluate all positions for a withdrawal or a previously fixed second place.
        if p and not r['closed'] and (not is_open_prediction(p) or any(int(c) not in active for t in p['tickets'] for c in t.split('-'))):
            r.setdefault('previous_predictions', []).append(p);p=None;r.pop('prediction',None)
        if not p and not r['closed'] and not r['cancelled'] and not race.get('decidedAt'):
            try:
                dd,line=inputs.build(H,d,m);p=model.predict(dd);p.pop('q',None);p.update(captured_at=source.now().isoformat(),input_captured_at=z['captured_at'],source_sha256=z['sha256'],history_through=H.data['through'],allocations={},odds={});r['lines']=line['known_unchanged_groups']
                # Inference can take time; never publish a newly generated post-close forecast.
                if source.now().timestamp()>=race['closeAt']:p=None;r['closed']=True
                else:r['prediction']=p;print('Forecast',rid,p['count'],'points',flush=True)
            except Exception as e:errors.append(dict(race_id=rid,error=str(e)));r['message']='予想入力を確認中（欠測を推定で補完しません）';p=None
        if p and not r['closed'] and not d.get('finalOdds') and captured.timestamp()<race['closeAt']:
            quotes={'-'.join(map(str,q['key'])):q['odds'] for q in d.get('trifecta',[]) if q.get('unitPrice')==100 and not q.get('absent') and q.get('odds',0)>0 and len(q.get('key',[]))==3}
            selected={t:quotes[t] for t in p['tickets'] if t in quotes}
            if len(selected)==p['count'] and source.now().timestamp()<race['closeAt']:
                allocation_time=source.now().isoformat()
                try:
                    prepared=allocation_snapshot(p, selected, z['captured_at'], allocation_time, r['close_at'])
                except (ValueError, TypeError) as e:
                    prepared=None
                    if source.now().timestamp()<race['closeAt']:
                        errors.append(dict(race_id=rid,error='配分の入力確認: '+str(e)))
                # Allocation can also cross the cutoff while computing.
                if prepared and source.now().timestamp()<race['closeAt']:
                    p.update(prepared);p['odds_source_sha256']=z['sha256'];p['policy_id']='flexible-first-v1'
        payouts,refunded,void,complete=source.outcomes(d)
        if complete or void:
            r.update(winners=sorted(payouts),payouts=payouts,refunded=refunded,refund_all=void,
                     result_complete=True,result_source_sha256=z['sha256'],result_captured_at=z['captured_at'])
            if p:
                assert datetime.fromisoformat(p['captured_at']).timestamp()<race['closeAt']
                r['settlement']=settle_prediction(p,payouts,refunded,void)
        if not p and r['closed']:
            r['message']='締切前予想の保存なし。公式結果のみ表示し、成績には含めません。'
        rows.append(r)
        # Checkpoint every race, so interrupted publication cannot lose valid preclose snapshots.
        seen={x['race_id'] for x in rows}
        save(ledger/(today+'.json'),rows+[x for x in existing.values() if x['date']==today and x['race_id'] not in seen])
    # Recheck unresolved older published forecasts (also after a day boundary).
    older=[r for r in existing.values() if r['date']<today]
    pending=[r for r in older if not r.get('result_complete') and not r.get('refund_all')]
    for ds in sorted({r['date'] for r in pending}):
        mm={m['race_id']:m for m in source.calendar(ds,ds)}
        for r in pending:
            if r['date']!=ds or r['race_id'] not in mm:continue
            try:
                z=source.race(mm[r['race_id']],CACHE);pay,ref,void,complete=source.outcomes(z['data'])
                if complete or void:
                    p=r.get('prediction')
                    if p:r['settlement']=settle_prediction(p,pay,ref,void)
                    r.update(winners=sorted(pay),payouts=pay,refunded=ref,refund_all=void,result_complete=True,closed=True,result_captured_at=z['captured_at'],result_source_sha256=z['sha256'])
            except Exception as e:errors.append(dict(race_id=r['race_id'],error=str(e)))
        save(ledger/(ds+'.json'),[r for r in older if r['date']==ds])
    save(ledger/(today+'.json'),rows)
    save(OUT/'nexus_live.json',dict(schema=1,date=today,updated_at=source.now().isoformat(),model={k:manifest[k] for k in ['id','name','historical','train','counts','live_policy']},history_through=H.data['through'],races=older+rows,notice=('一部レースの取得・予想を確認中です。確認できたレースから掲載しています。' if errors else ''),coverage=dict(scheduled=len(todaymetas),predicted=sum(bool(r.get('prediction')) for r in rows))))
    save(STATE/'run_status.json',dict(completed=True,updated_at=source.now().isoformat(),errors=errors,history_through=H.data['through']))
    print('Published',len(rows),'races; errors',len(errors),flush=True)
    if errors:print(json.dumps(errors,ensure_ascii=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--history-only',action='store_true');args=parser.parse_args();run(args.history_only)

