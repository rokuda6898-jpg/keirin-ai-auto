"""Audit evidence coverage against recorded schedules, never infer missing history."""
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from common import OUTPUT_DIR
from betting_logic import STRATEGY_VERSION
from selection_research import eligible


def load(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def timestamp(value):
    try:
        parsed=datetime.fromisoformat(value)
        return parsed.timestamp() if parsed.tzinfo else None
    except (TypeError,ValueError):return None


def preclose(row, now):
    stamp=timestamp(row.get('snapshot_at'))
    close=row.get('close_at')
    return stamp is not None and isinstance(close,(int,float)) and math.isfinite(close) and stamp < close and stamp <= now.timestamp()


def metrics(rows):
    bet=[r for r in rows if r.get('tickets')]
    hits=sum(any(t['buy']==r['actual_trifecta'] for t in r['tickets']) for r in bet)
    stake=sum(len(r['tickets'])*100 for r in bet)
    returned=sum(r['payout_per_100yen'] for r in bet if any(t['buy']==r['actual_trifecta'] for t in r['tickets']))
    return {'settled_races':len(rows),'bet_races':len(bet),'skip_races':len(rows)-len(bet),
            'hits':hits,'hit_rate':hits/len(bet) if bet else None,'stake_yen':stake,
            'return_yen':returned,'return_rate':returned/stake if stake else None,
            'sample_status':'少数・参考' if len(bet)<30 else '検証継続'}


def build_validation_coverage(output_dir=OUTPUT_DIR,now=None):
    now=now or datetime.now(ZoneInfo('Asia/Tokyo'))
    today=now.astimezone(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d')
    folder=output_dir/'company';folder.mkdir(parents=True,exist_ok=True)
    try:schedule=pd.read_csv(output_dir/'latest_race_schedule.csv',dtype={'race_id':str})
    except (OSError,pd.errors.EmptyDataError):schedule=pd.DataFrame()
    schedule_rows=schedule.drop_duplicates('race_id').to_dict('records') if 'race_id' in schedule else []
    schedule_map={str(r['race_id']):r for r in schedule_rows}
    target={rid:r for rid,r in schedule_map.items() if str(r.get('date'))==today}
    ledger=folder/'ticket_return_snapshots.jsonl'
    snapshots=[json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines() if line] if ledger.exists() else []
    latest={};invalid=0;quote_unknown=0;quote_late=0
    for row in snapshots:
        if row.get('strategy_version')!=STRATEGY_VERSION:continue
        if not preclose(row,now):invalid+=1;continue
        rid=str(row['race_id'])
        if rid not in latest or row['snapshot_at']>latest[rid]['snapshot_at']:latest[rid]=row
    for row in latest.values():
        for ticket in row.get('tickets',[]):
            capture=timestamp(ticket.get('odds_captured_at_jst'))
            if capture is None:quote_unknown+=1
            elif capture>timestamp(row['snapshot_at']) or capture>=row['close_at']:quote_late+=1
    official={str(r['race_id']):r for r in load(output_dir/'latest_results.json',[])
              if str(r.get('official_result_available','')).lower() in {'true','1'}}
    settled_rows=eligible(load(folder/'ticket_return_settled.json',[]))
    settled={str(r['race_id']):r for r in settled_rows}
    stages=Counter();gaps=[]
    for rid,race in target.items():
        snapshot=latest.get(rid)
        if snapshot:stages['saved_preclose']+=1
        if snapshot and snapshot.get('tickets'):stages['saved_with_tickets']+=1
        if snapshot and not snapshot.get('tickets'):stages['saved_skip']+=1
        market=load(output_dir/f'market_odds_{rid}.json',{})
        capture=timestamp(market.get('captured_at_jst'))
        close=pd.to_numeric(race.get('close_at'),errors='coerce')
        quote_verified=str(market.get('race_id'))==rid and capture is not None and pd.notna(close) and capture<close and market.get('usable_count',0)>0
        if quote_verified:stages['verified_odds_record']+=1
        if rid in official:stages['official_result']+=1
        if rid in settled:stages['settled_current_strategy']+=1
        missing=[]
        if not snapshot:missing.append('締切前予想の保存なし')
        if not quote_verified:missing.append('使用可能オッズの時刻付き確認記録なし')
        if rid not in official:missing.append('公式結果の確認待ち')
        if rid not in settled:missing.append('現行戦略の買い目成績未照合')
        if missing:gaps.append({'race_id':rid,'venue':race.get('venue',''),'race_no':race.get('race_no',''),'missing':missing})
    # Conditions use recorded pre-race information; unavailable schedule fields remain unknown.
    groups={key:defaultdict(list) for key in ['venue','time','riders','payout_band']}
    for row in settled_rows:
        rid=str(row['race_id']);race=schedule_map.get(rid,{})
        start=race.get('start_at')
        if isinstance(start,(int,float)) and math.isfinite(start):
            hour=datetime.fromtimestamp(start,ZoneInfo('Asia/Tokyo')).hour
            time='朝（12時未満）' if hour<12 else '昼（12〜17時）' if hour<17 else '夜（17〜21時）' if hour<21 else '深夜（21時以降）'
        else:time='発走時刻未保存'
        counts=[len(p.get('probabilities',{})) for p in row.get('position_probabilities',[]) if p.get('position')==1]
        riders=str(counts[0])+'車' if counts and counts[0] else '選手数未保存'
        odds=row['payout_per_100yen']/100
        band='10倍未満' if odds<10 else '10〜100倍未満' if odds<100 else '100〜1000倍未満' if odds<1000 else '1000倍以上'
        for key,value in [('venue',row.get('venue') or '場未保存'),('time',time),('riders',riders),('payout_band',band)]:groups[key][str(value)].append(row)
    conditions={key:[{'condition':value,**metrics(rows)} for value,rows in sorted(values.items())] for key,values in groups.items()}
    knowledge=load(folder/'annual_rider_knowledge.json',{})
    cutoff=knowledge.get('window_end_exclusive')
    cutoff_ok=isinstance(cutoff,str) and cutoff<=today and knowledge.get('asof_date')==cutoff
    profiles=knowledge.get('profiles',{})
    sparse=Counter()
    for profile in profiles.values():
        evaluation=profile.get('evaluation',profile)
        n=evaluation.get('effective_races',evaluation.get('races',0))
        sparse['参考なし' if n<=0 else '15走相当未満' if n<15 else '15〜30走相当未満' if n<30 else '30走相当以上']+=1
    report={'updated_at_jst':now.isoformat(timespec='seconds'),'today_date':today,'strategy_version':STRATEGY_VERSION,
            'coverage':{'scheduled_races':len(target),**{key:stages[key] for key in ['saved_preclose','saved_with_tickets','saved_skip','verified_odds_record','official_result','settled_current_strategy']}},
            'gaps':gaps,'conditions':conditions,'time_checks':{'invalid_snapshot_rows':invalid,
            'selected_tickets_without_quote_time':quote_unknown,'quote_after_snapshot_or_close':quote_late,
            'annual_cutoff':cutoff,'annual_cutoff_confirmed':cutoff_ok,'model_training_and_full_feature_cutoffs':'未確認・この検査だけでは未来情報の混入なしと断定できない'},
            'sparse_rider_reference':dict(sparse),'actual_purchase':False,'auto_promotion':False}
    (folder/'validation_coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    render(report,folder)
    return report


def render(report,folder):
    esc=lambda value:html.escape(str(value))
    pct=lambda v:'未集計' if v is None else f'{v*100:.1f}%'
    body='<h1>検証の抜けと条件別成績</h1><h2>今日の検証範囲</h2><p>対象日 '+report['today_date']+'。取得済み開催予定が分母です。全国の全開催を取得できた保証ではありません。</p><table>'
    labels={'scheduled_races':'取得済み開催予定','saved_preclose':'締切前予想を保存','saved_with_tickets':'買い目ありで保存','saved_skip':'見送りを保存','verified_odds_record':'使用可能オッズの時刻付き記録','official_result':'公式結果を確認','settled_current_strategy':'現行戦略の成績を照合'}
    for key,label in labels.items():body+=f'<tr><th>{label}</th><td>{report["coverage"][key]}レース</td></tr>'
    body+='</table><p>対象は現行買い目ロジックのみ。変更前の記録はこの成績に合算しません。各段階は独立した件数です。オッズの記録なしは取得失敗と断定せず、古い予想には時刻の保存がない場合があります。締切後の予想を事前予想へ数え直しません。</p><details><summary>レース別の未確認項目</summary><ul>'
    for row in report['gaps']:body+=f'<li>{esc(row["venue"])} {esc(row["race_no"])}R：{esc(" ／ ".join(row["missing"]))}</li>'
    body+='</ul></details><h2>条件別の成績</h2><p>現行戦略の締切前保存・公式払戻確定分。各点100円の検証で、実購入成績ではありません。30レース未満は少数参考です。</p>'
    for key,label in [('venue','競輪場'),('time','発走時間帯'),('riders','選手数'),('payout_band','結果の3連単払戻倍率帯')]:
        body+='<h3>'+label+'</h3>'
        if key=='payout_band':body+='<p>結果確定後の分類です。予想時の人気・オッズ帯ではなく、購入判断の条件には使えません。</p>'
        body+='<table><tr><th>条件</th><th>的中／購入対象</th><th>的中率</th><th>回収率</th><th>評価</th></tr>'
        for r in report['conditions'][key]:body+=f'<tr><td>{esc(r["condition"])}</td><td>{r["hits"]}／{r["bet_races"]}R</td><td>{pct(r["hit_rate"])}</td><td>{pct(r["return_rate"])}</td><td>{r["sample_status"]}</td></tr>'
        body+='</table>'
    checks=report['time_checks']
    body+='<h2>未来情報・時刻の検査</h2><ul>'
    for label,value in [('締切前時刻を確認できず除外した行',checks['invalid_snapshot_rows']),('価格取得時刻が未保存の選択買い目',checks['selected_tickets_without_quote_time']),('価格取得時刻が予想後または締切後',checks['quote_after_snapshot_or_close'])]:body+=f'<li>{label}：{value}件</li>'
    body+='</ul><p>選手成績の集計終端（この日を含めない）：'+esc(checks['annual_cutoff'])+' ／ '+('日付整合を確認' if checks['annual_cutoff_confirmed'] else '日付整合は未確認')+'</p><p>'+esc(checks['model_training_and_full_feature_cutoffs'])+'</p><h2>出走数の少ない選手</h2><p>既存処理は直近1〜3年を参照し、重み付き成績を20走相当の基準値へ寄せて少数実績の影響を抑えます。未知の選手を低確率と決めつけません。</p><ul>'
    for label,count in report['sparse_rider_reference'].items():body+=f'<li>{esc(label)}：{count}選手</li>'
    body+='</ul><p>蓄積レース数と現行ロジックの実戦検証数は別です。</p><p>更新 '+esc(report['updated_at_jst'])+'</p><a href="selection_research.html">買い目の比較検証</a> ／ <a href="../index.html">今日の予想</a>'
    page='<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>検証の抜けと条件別成績</title><link rel="stylesheet" href="../site-ui.css"><style>table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid #dde6ef;text-align:left}</style></head><body><main>'+body+'</main></body></html>'
    (folder/'validation_coverage.html').write_text(page,encoding='utf-8')
