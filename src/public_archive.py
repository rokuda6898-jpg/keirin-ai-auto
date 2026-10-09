"""Browsable evidence archive; does not reconstruct forecasts after results."""
import json
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from public_live import ROOT, read_csv, read_json


def build(root=ROOT):
    out = root / 'outputs'
    races = {}
    history = read_csv(out / 'company/historical_race_archive.csv')
    for row in history + read_csv(out / 'company/settled_race_archive.csv'):
        rid = row['race_id']
        actual = [row.get(key, '') for key in ('winner_car_no','second_car_no','third_car_no')]
        try:
            actual = '-'.join(str(int(float(v))) for v in actual)
        except (ValueError, TypeError):
            actual = ''
        races[rid] = dict(id=rid, date=row['date'], venue=row['venue'], number=row['race_no'], actual=actual, forecasts=[])
    # Keep each recorded strategy/snapshot separately; no mixing initial/latest.
    for row in read_json(out / 'company/ticket_return_settled.json', []):
        rid = row['race_id']
        race = races.setdefault(rid, dict(id=rid,date=row.get('date',''),venue=row.get('venue',''),number=row.get('race_no',''),actual='',forecasts=[]))
        race['forecasts'].append({key:row.get(key) for key in ('snapshot_at','strategy_version','tickets','actual_trifecta','actual_trifecta_odds','is_hit')})
    knowledge = read_json(out / 'company/annual_rider_knowledge.json', {})
    manifest = read_json(out / 'company/central_history_manifest.json', {})
    learning = {key:knowledge.get(key) for key in ('updated_at_jst','as_of','status','total_archive_races','annual_races','policy')}
    learning['history_report'] = {key:manifest.get(key) for key in ('updated_at_jst','historical_races','settled_race_archive_races','history_cache_loaded_this_run')}
    learning['candidate'] = read_json(out / 'company/archive_learning_run.json', {})
    payload = dict(schema=1, generated_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds'),
                   races=sorted(races.values(),key=lambda r:(r['date'],r['venue'],float(r['number'] or 0)),reverse=True), learning=learning)
    (out / 'public_archive.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(f'Public archive: {len(races)} races')
    return payload


if __name__ == '__main__':
    build()
