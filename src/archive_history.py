"""Export known historical results, without inventing historical predictions."""
import csv
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def export(root=ROOT):
    races = defaultdict(dict)
    today = datetime.now(ZoneInfo('Asia/Tokyo')).date().isoformat()
    with (root/'data/raw/history.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            # Same-day results remain the responsibility of live settlement.
            if not row.get('date') or row['date'][:10] >= today:
                continue
            try:
                pos = int(float(row.get('finish_pos','')))
                car = int(float(row.get('car_no','')))
            except (ValueError, TypeError):
                continue
            if pos in (1,2,3) and 1 <= car <= 9:
                races[row['race_id']].setdefault(pos, set()).add(car)
                races[row['race_id']]['meta'] = row
    target=root/'outputs/company/historical_race_archive.csv'
    fields=['race_id','date','venue','race_no','winner_car_no','second_car_no','third_car_no']
    retained = {}
    if target.exists():
        with target.open(encoding='utf-8-sig',newline='') as stream:
            retained = {r['race_id']: {k:r.get(k,'') for k in fields} for r in csv.DictReader(stream)}
    for rid, race in sorted(races.items()):
        if not all(len(race.get(p, set())) == 1 for p in (1,2,3)):
            continue
        cars = [next(iter(race[p])) for p in (1,2,3)]
        if len(set(cars)) != 3:
            continue
        row=race['meta'];record={k:row.get(k,'') for k in fields[:4]}
        record.update(zip(fields[4:], cars))
        retained[rid] = record
    with target.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows(retained[rid] for rid in sorted(retained))
    count = len(retained)
    print(f'Historical result archive: {count} races')
    return count


if __name__=='__main__':
    export()
