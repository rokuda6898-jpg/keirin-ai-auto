"""Apply the new preclose stake policy to already-published source snapshots.

Does not fetch source data, re-train, infer missing probabilities, or overwrite
closed forecasts. A fixed-second forecast is held until Engine re-evaluates it.
"""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from nexus_model.publication import allocation_snapshot, is_open_prediction

ROOT=Path(__file__).resolve().parents[1]
JST=timezone(timedelta(hours=9))


def refresh(feed, now=None):
    now=now or datetime.now(JST)
    counts={'updated':0,'held':0,'unchanged_closed':0,'near_close':0,'auto_available':0,'auto_unavailable':0}
    for race in feed['races']:
        prediction=race.get('prediction')
        if not prediction:continue
        close=datetime.fromisoformat(race['close_at']) if race.get('close_at') else None
        if close is None or now>=close or race.get('closed') or race.get('settlement') or race.get('cancelled'):
            counts['unchanged_closed']+=1;continue
        if not is_open_prediction(prediction):
            race['prediction_hold']='fixed_second';counts['held']+=1;continue
        # Allow time for this saved-feed migration to reach the public branch.
        if now+timedelta(minutes=5)>=close:
            counts['near_close']+=1;continue
        quotes=prediction.get('odds',{});odds_at=prediction.get('odds_at')
        if not odds_at or any(t not in quotes for t in prediction['tickets']):continue
        snapshot=allocation_snapshot(prediction,quotes,odds_at,now.isoformat(),race['close_at'])
        if datetime.now(JST)+timedelta(minutes=5)>=close:continue
        prediction.update(snapshot);prediction['policy_id']='flexible-first-v1'
        counts['updated']+=1;counts['auto_available' if snapshot['auto_budget'] else 'auto_unavailable']+=1
    feed['policy_updated_at']=now.isoformat()
    return counts


if __name__=='__main__':
    path=ROOT/'outputs/nexus_live.json'
    feed=json.loads(path.read_text('utf-8'))
    counts=refresh(feed)
    manifest=json.loads((ROOT/'models/nexus53/manifest.json').read_text('utf-8'))
    feed['model']['live_policy']=manifest['live_policy']
    path.write_text(json.dumps(feed,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    # Persist the exact same preclose snapshots for later official settlement.
    for day in sorted({r['date'] for r in feed['races']}):
        ledger=ROOT/'outputs/nexus_forecasts'/(day+'.json')
        ledger.parent.mkdir(parents=True,exist_ok=True)
        ledger.write_text(json.dumps([r for r in feed['races'] if r['date']==day],ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(counts,ensure_ascii=False))
