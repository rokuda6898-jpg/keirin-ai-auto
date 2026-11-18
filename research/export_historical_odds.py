"""Read-only inventory and whitelist export of archived final trifecta quotes.

These quotes have no certified pre-close capture time. They are unsuitable for
the same race's pre-race forecast. Any future research must explicitly lag them
to strictly later races and separately audit source dates and coverage.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path

COLUMNS=['race_id','date','buy','trifecta_odds','source_url']


def quote(row):
    if row.get('bet_type') not in (None,'','trifecta'):raise ValueError('not_trifecta')
    rid=str(row.get('race_id','')).strip();day=str(row.get('date','')).strip();buy=str(row.get('buy','')).strip()
    if not rid or not re.fullmatch(r'\d+',rid):raise ValueError('invalid_race_id')
    try:date.fromisoformat(day)
    except ValueError:raise ValueError('invalid_date') from None
    if not re.fullmatch(r'[1-9]-[1-9]-[1-9]',buy) or len(set(buy.split('-')))!=3:raise ValueError('invalid_ticket')
    raw=row.get('trifecta_odds',row.get('odds_used',row.get('odds')))
    try:value=float(raw)
    except (TypeError,ValueError):raise ValueError('invalid_odds') from None
    if not math.isfinite(value) or value<1.:raise ValueError('invalid_odds')
    url=str(row.get('source_url',''))
    if not url.startswith('https://www.winticket.jp/keirin/'):raise ValueError('unrecognized_source')
    return {'race_id':rid,'date':day,'buy':buy,'trifecta_odds':value,'source_url':url}


def records(root,inventory):
    trifecta=root/'history_trifecta_odds.csv';all_odds=root/'history_odds.csv'
    for path in (trifecta,all_odds):
        if path.is_file() and path.stat().st_size:
            inventory['selected_source']=path.name
            seen=False
            with path.open(encoding='utf-8-sig',newline='') as src:
                for row in csv.DictReader(src):seen=True;yield row
            if seen:return
    cache=root/'race_cache';inventory['selected_source']='race_cache'
    for path in sorted(cache.glob('*.json')) if cache.is_dir() else []:
        inventory['cache_files_seen']+=1
        try:data=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,OSError):inventory['bad_cache_files']+=1;continue
        if not isinstance(data,dict):inventory['bad_cache_files']+=1;continue
        rows=data.get('odds_rows',[])
        if not isinstance(rows,list):inventory['bad_cache_files']+=1;continue
        for row in rows:
            if isinstance(row,dict):yield row


def export(root,output,cache_key=''):
    output.mkdir(parents=True,exist_ok=True)
    path=output/'historical_final_trifecta.csv.gz'
    if path.exists():raise ValueError('refusing to overwrite evidence')
    inventory={'cache_files_seen':0,'bad_cache_files':0,'selected_source':None}
    skipped=Counter();races=set();dates=set();total=0
    with gzip.open(path,'wt',encoding='utf-8',newline='') as dest:
        writer=csv.DictWriter(dest,fieldnames=COLUMNS);writer.writeheader()
        for raw in records(root,inventory):
            try:row=quote(raw)
            except ValueError as e:skipped[str(e)]+=1;continue
            writer.writerow(row);races.add(row['race_id']);dates.add(row['date']);total+=1
    with path.open('rb') as src:digest=hashlib.file_digest(src,'sha256').hexdigest()
    inventory.update({'schema_version':1,'columns':COLUMNS,'matched_cache_key':cache_key,
        'source_file_sizes':{p.name:p.stat().st_size for p in root.glob('history*.csv') if p.is_file()},
        'quotes':total,'races':len(races),'first_date':min(dates) if dates else None,'last_date':max(dates) if dates else None,
        'skipped_rows':dict(skipped),'sha256':digest,'data_available':bool(total),'outcome_columns_exported':False,
        'preclose_timestamp_certified':False,'usable_for_same_race_forecast':False,'production_changed':False,
        'research_boundary':'Historical final quotes only. Audit dates and completeness, then consider strictly lagged features for later races. Never use for the quoted race forecast.'})
    (output/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    return inventory


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache-key',default='');a=p.parse_args()
    print(json.dumps(export(a.root,a.output,a.cache_key),ensure_ascii=False))
