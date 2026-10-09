"""Prospective three-year equations and independently frozen 6000-yen plans."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import archive_50000 as archive
import equation_budget as budget
import fixed_year_features as features
import fixed_year_study as study
import individual_equations_live as common
import individual_three_year as training
from department_experiment_v2 import quote_view
from official_outcomes import normalize_outcome

ROOT = common.ROOT
FOLDER = ROOT / 'outputs/company/three_year_equations'
NAMES = {k: v for k,v in common.LABELS.items() if not k.startswith('market_')}
NAMES.update(archive_positions='順位別ロジット式（3年学習）',
             archive_pairwise='先着関係ロジット式（3年学習）',
             first_anchor='1着補正付き統合式（3年学習）')
VERSION = 'three_year_equations_6000_v1'


def prediction_marks(method):
    """Display-only ranking from the immutable equation's saved top-12 picks.

    Compare summed first-place support, then second/third support for ties.
    These truncated sums are not full-field winning probabilities.
    """
    support = {}
    for pick in method['tickets']:
        for place, car in enumerate(map(int, pick['buy'].split('-'))):
            support.setdefault(car, [0., 0., 0.])[place] += float(pick['probability'])
    cars = sorted(support, key=lambda car: (*(-p for p in support[car]), car))
    return [{'mark':mark, 'car_no':car} for mark,car in zip(('◎','○','▲','△','☆'),cars)]


def allocations(folder):
    rows = common.read_json(folder/'allocations.json', [])
    seen = set()
    for row in rows:
        key = (row['race_id'], row['points'], row['equation'])
        if key in seen or common.digest({k:v for k,v in row.items() if k != 'sha256'}) != row['sha256']:
            raise ValueError('corrupt or duplicate allocation evidence')
        seen.add(key)
    return rows


def freeze_allocations(saved, forecast, race, market, now):
    if now.timestamp() >= forecast['close_at'] or forecast['date'] != now.date().isoformat():
        return 0
    prices, stamps, _, _ = quote_view(race, market, now)
    count = 0
    for points in budget.POINTS:
        for name,method in forecast['methods'].items():
            if any(r['race_id']==forecast['race_id'] and r['points']==points and r['equation']==name for r in saved):
                continue
            plan=budget.allocate(method['tickets'],prices,points)
            if plan['status']!='allocated':
                continue
            quote_times={stamps[t['buy']] for t in plan['tickets']}
            if len(quote_times)!=1:
                continue
            completed=common.clock()
            quote_time=datetime.fromisoformat(next(iter(quote_times)))
            if completed.timestamp()>=forecast['close_at'] or not 0<=(completed-quote_time).total_seconds()<=300:
                continue
            row={'version':VERSION,'race_id':forecast['race_id'],'points':points,'equation':name,
                'date':forecast['date'],'close_at':forecast['close_at'],
                'forecast_sha256':forecast['sha256'],'allocated_at_jst':completed.isoformat(),
                'quote_at_jst':quote_time.isoformat(),'methods':{name:plan},
                'allocation_code_sha256':hashlib.sha256(Path(budget.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                'purchase_authorized':False}
            row['sha256']=common.digest(row); saved.append(row); count+=1
    return count


def field_signature(race):
    return common.digest(sorted((int(r.car_no),str(r.player_id).lstrip('0') or '0') for r in race.itertuples()))


def fresh_market(race, report_dir=None):
    from fetch_today_entries import parse_race_page
    entries, odds = parse_race_page(str(race.iloc[0].source_url), completeness_attempts=1,
        market_report_dir=report_dir if report_dir is not None else FOLDER/'market_evidence')
    fresh = pd.DataFrame(entries)
    if (fresh.empty or set(fresh.race_id.astype(str)) != {str(race.iloc[0].race_id)}
            or field_signature(fresh) != field_signature(race)
            or set(pd.to_numeric(fresh.close_at)) != {float(race.iloc[0].close_at)}):
        raise ValueError('出走選手または締切が変更されたため配分停止')
    return pd.DataFrame(odds)


def allocate_pending(folder=FOLDER, refresh_odds=False):
    records, saved = common.ledger(folder), allocations(folder)
    now = common.clock()
    entries = pd.read_csv(ROOT/'data/raw/today_entries.csv',dtype={'race_id':str,'player_id':str})
    market = pd.read_csv(ROOT/'data/raw/today_odds.csv',dtype={'race_id':str})
    by_id = {str(rid):r for rid,r in entries.groupby('race_id',sort=False)}
    quotes = {str(rid):r for rid,r in market.groupby('race_id',sort=False)}
    decided = {str(r.get('race_id')) for r in common.read_json(ROOT/'outputs/latest_results.json',[]) if normalize_outcome(r)}
    diagnostics = []
    for row in sorted(records,key=lambda r:r['close_at']):
        rid=row['race_id']; now=common.clock()
        if rid not in by_id or rid in decided or row['date']!=now.date().isoformat() or row['close_at']<=now.timestamp():
            continue
        if sum(r['race_id']==rid for r in saved)==len(NAMES)*len(budget.POINTS):
            continue
        race=by_id[rid]
        if row.get('field_sha256') != field_signature(race):
            diagnostics.append({'race_id':rid,'reason':'出走選手の固定記録と不一致'})
            continue
        quote=quotes.get(rid,pd.DataFrame())
        if refresh_odds and row['close_at']-now.timestamp()<=3600:
            try:
                quote=fresh_market(race,folder/'market_evidence')
            except Exception as exc:
                diagnostics.append({'race_id':rid,'reason':str(exc)})
                continue
        added=freeze_allocations(saved,row,race,quote,common.clock())
        diagnostics.append({'race_id':rid,'new_plans':added})
        # Persist immediately after each capture, before fetching another race.
        common.save(folder/'allocations.json',saved)
    common.save(folder/'allocation_status.json',{'updated_at':common.clock().isoformat(),'races':diagnostics})
    return report(folder)


def forecast(folder=FOLDER, refresh_odds=False):
    records, saved = common.ledger(folder), allocations(folder)
    now = common.clock(); day = now.date().isoformat()
    entries = pd.read_csv(ROOT/'data/raw/today_entries.csv', dtype={'race_id':str, 'player_id':str})
    entries['date'] = pd.to_datetime(entries.date, errors='coerce').dt.strftime('%Y-%m-%d')
    entries = entries[entries.date.eq(day)].copy()
    if entries.empty:
        raise ValueError('current race card is missing or stale')
    official = common.read_json(ROOT/'outputs/latest_results.json', [])
    decided = {str(r.get('race_id')) for r in official if normalize_outcome(r)}
    known = {r['race_id'] for r in records}
    coverage, ready = [], []
    for rid,race in entries.groupby('race_id', sort=False):
        stamps = pd.to_numeric(race.close_at, errors='coerce')
        reason = ('保存済み' if rid in known else '予想前に結果確定' if rid in decided else
                  '締切時刻不明' if stamps.isna().any() or stamps.nunique()!=1 else
                  '初回確認時点で締切後' if stamps.iloc[0]<=now.timestamp() else study.input_reason(race) or '')
        coverage.append({'race_id':rid, 'venue':str(race.iloc[0].get('venue','')), 'race_no':int(race.iloc[0].race_no), 'reason':reason})
        if not reason:
            ready.append(race)
    if ready:
        manifest, bundle = training.load(day)
        clean = features.mask_outcomes(pd.concat(ready, ignore_index=True))
        predicted = common.shadow.restore_race_metadata(study.base_predict(clean, bundle), clean)
        for rid,race in predicted.groupby('race_id', sort=False):
            methods = common.race_distributions(race, bundle, race, bundle, bundle['stage'])
            raw = clean[clean.race_id.eq(rid)].to_dict('records')
            for name in archive.NAMES:
                methods[name] = {'basis':'trained', 'tickets':common.ranked(archive.probabilities(name,bundle['archive'],raw))}
            if set(methods) != set(NAMES):
                raise ValueError('incomplete independent equation set')
            item = race.iloc[0]
            meta = {'race_id':str(rid), 'date':day, 'venue':str(item.get('venue','')),
                'race_no':int(item.race_no), 'close_at':float(item.close_at),
                'start_at':float(item.start_at) if pd.notna(item.get('start_at')) else None,
                'field_sha256':field_signature(clean[clean.race_id.eq(rid)]),
                'producer_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                'input_sha256':hashlib.sha256(clean[clean.race_id.eq(rid)].to_json(orient='records').encode()).hexdigest()}
            frozen = common.freeze(records, 'three_year_421', meta, methods,
                {'three_year':{'training_cutoff_exclusive':manifest['training_cutoff_exclusive'],
                              'model_sha256':manifest['model_sha256']}}, common.clock())
            for row in coverage:
                if row['race_id']==rid:
                    row['reason'] = '保存済み' if frozen else '計算完了時点で締切後'
    common.save(folder/'forecasts.json',records)
    common.save(folder/'allocations.json',saved)
    common.save(folder/'coverage.json',{'date':day,'races':coverage,'updated_at':common.clock().isoformat()})
    return allocate_pending(folder,refresh_odds)


def report(folder=FOLDER):
    records, saved = common.ledger(folder), allocations(folder)
    source = {r['race_id']:r for r in records}
    if any(r['race_id'] not in source or r['forecast_sha256'] != source[r['race_id']]['sha256'] for r in saved):
        raise ValueError('allocation does not match immutable forecast')
    outcomes = common.update_results(folder,records,common.read_json(ROOT/'outputs/latest_results.json',[]))
    manifest = common.read_json(training.FOLDER/'manifest.json', {})
    stats = {name:{str(n):common.metrics([r for r in saved if r['points']==n and r['equation']==name],outcomes,name,n)
                  for n in budget.POINTS} for name in NAMES}
    groups={}
    for row in saved:
        key=(row['race_id'],row['points'],row['quote_at_jst'],row['forecast_sha256'])
        groups.setdefault(key,set()).add(row['equation'])
    paired=[r for r in saved if groups[(r['race_id'],r['points'],r['quote_at_jst'],r['forecast_sha256'])]==set(NAMES)]
    common_stats={name:{str(n):common.metrics([r for r in paired if r['points']==n and r['equation']==name],outcomes,name,n)
                       for n in budget.POINTS} for name in NAMES}
    value = {'version':VERSION,'updated_at_jst':common.clock().isoformat(),'training':manifest,
        'statistics':stats,'common_comparison_statistics':common_stats,
        'forecast_races':len(records),'allocated_race_plans':len(saved),
        'budget_yen_per_equation_per_race_per_selected_plan':6000,
        'points_are_alternative_plans_not_combined':True,
        'market_models_without_three_year_data':['market_residual','pairwise_order','state_paths'],
        'coverage':common.read_json(folder/'coverage.json',{}),'purchase_authorized':False}
    common.save(folder/'report.json',value)
    render(folder,records,saved,outcomes,value)
    return value


def render(folder,records,saved,outcomes,value):
    from three_year_equations_ui import document
    page=document(records,saved,outcomes,value,NAMES,prediction_marks,
                  ROOT/'outputs/latest_race_schedule.csv')
    (folder/'index.html').write_text(page,encoding='utf-8')


def merge(remote,folder=FOLDER):
    other=common.ledger(remote); keys={(r['family'],r['race_id']) for r in other}
    common.save(folder/'forecasts.json',other+[r for r in common.ledger(folder) if (r['family'],r['race_id']) not in keys])
    old=allocations(remote); keys={(r['race_id'],r['points'],r['equation']) for r in old}
    combined=old+[r for r in allocations(folder) if (r['race_id'],r['points'],r['equation']) not in keys]
    forecasts={r['race_id']:r['sha256'] for r in common.ledger(folder)}
    # If a concurrent earlier forecast won, discard only its unpublished local
    # allocation, never attach an allocation to a different probability snapshot.
    common.save(folder/'allocations.json',[r for r in combined if forecasts.get(r['race_id'])==r['forecast_sha256']])
    common.merge(remote,folder,render_report=False)
    return report(folder)


def command(name,points,race_id,folder=FOLDER):
    if name not in NAMES or points not in budget.POINTS:
        raise ValueError('unknown equation or points command')
    found=[r for r in allocations(folder) if r['race_id']==race_id and r['points']==points and r['equation']==name]
    if not found:
        return {'status':'no_frozen_preclose_allocation','equation':name,'points':points,'race_id':race_id,'purchase_authorized':False}
    return {'equation':name,'points':points,'race_id':race_id,'allocated_at_jst':found[0]['allocated_at_jst'],
            'purchase_authorized':False,**found[0]['methods'][name]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('forecast','allocate','report','merge','list','command'))
    parser.add_argument('--refresh-odds',action='store_true')
    parser.add_argument('--remote',type=Path)
    parser.add_argument('--equation',choices=list(NAMES))
    parser.add_argument('--points',type=int,choices=budget.POINTS,default=3)
    parser.add_argument('--race-id')
    args=parser.parse_args()
    if args.command=='list':
        print(json.dumps(NAMES,ensure_ascii=False))
    elif args.command=='command':
        print(json.dumps(command(args.equation,args.points,args.race_id),ensure_ascii=False))
    else:
        value=forecast(refresh_odds=args.refresh_odds) if args.command=='forecast' else allocate_pending(refresh_odds=args.refresh_odds) if args.command=='allocate' else merge(args.remote) if args.command=='merge' else report()
        print(json.dumps({k:value[k] for k in ('forecast_races','allocated_race_plans')},ensure_ascii=False))
