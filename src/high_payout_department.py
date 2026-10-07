"""Independent results from frozen-before-close longshot decisions only."""
import json
import html
import math
from datetime import datetime
from zoneinfo import ZoneInfo
from high_payout_strategy import VERSION


def summarize_high_payout(rows):
    eligible=[]
    for row in rows:
        department=row.get('high_payout_department') or {}
        try:
            valid=datetime.fromisoformat(row['snapshot_at']).timestamp()<float(row['close_at'])
            payout=float(row['payout_per_100yen'])
        except (KeyError,TypeError,ValueError):continue
        if department.get('version')!=VERSION or not valid or not math.isfinite(payout) or payout<=0:continue
        eligible.append(row)
    unique={}
    for row in eligible:
        if row['race_id'] not in unique or row['snapshot_at']>unique[row['race_id']]['snapshot_at']:unique[row['race_id']]=row
    proposed=hits=stake=returned=0;payouts=[];bands={str(n):0 for n in [100,300,500,1000]}
    patterns={p:{'races':0,'hits':0,'hit_rate':None} for p in ['1着荒れ','2着荒れ','3着荒れ','ライン崩壊','人気過剰']}
    for row in unique.values():
        department=row['high_payout_department']
        proposed_tickets={t['buy']:t for t in department.get('tickets',[]) if t.get('odds',0)>=100}
        tickets={t['buy'] for t in row.get('tickets',[]) if t.get('group')=='穴' and t['buy'] in proposed_tickets}
        if not tickets:continue
        proposed+=1;stake+=100*len(tickets)
        hit=row['actual_trifecta'] in tickets
        if hit:
            hits+=1;returned+=row['payout_per_100yen'];payouts.append(row['payout_per_100yen'])
            for n in bands:bands[n]+=int(row['payout_per_100yen']>=int(n)*100)
        for pattern,counts in patterns.items():
            subset={buy for buy in tickets if pattern in proposed_tickets[buy].get('patterns',[])}
            if subset:counts['races']+=1;counts['hits']+=int(row['actual_trifecta'] in subset)
    for counts in patterns.values():counts['hit_rate']=counts['hits']/counts['races'] if counts['races'] else None
    return {'target_races':len(unique),'predicted_races':proposed,'skipped_races':len(unique)-proposed,
            'hits':hits,'hit_rate':hits/proposed if proposed else None,'stake_yen':stake,'return_yen':returned,
            'return_rate':returned/stake if stake else None,'average_payout_per_100yen':sum(payouts)/len(payouts) if payouts else None,
            'payout_hit_counts':bands,'pattern_results':patterns}


def build_high_payout_department(rows,output_dir):
    folder=output_dir/'company';folder.mkdir(parents=True,exist_ok=True)
    plans_path=output_dir/'latest_race_strategy.json'
    plans=json.loads(plans_path.read_text(encoding='utf-8')) if plans_path.exists() else []
    now=datetime.now(ZoneInfo('Asia/Tokyo'))
    report={'version':VERSION,'updated_at_jst':now.isoformat(timespec='seconds'),
            'total':summarize_high_payout(rows),'today':summarize_high_payout([r for r in rows if r.get('date')==now.strftime('%Y-%m-%d')]),
            'scope':'新部署の締切前保存・公式払戻済みのみ。各点100円の試算。本線と旧穴予想は含めない。',
            'live_decisions':[{'race_id':p['race_id'],'venue':p.get('venue'),'race_no':p.get('race_no'),
                              **p['high_payout_department']} for p in plans if p.get('high_payout_department')],
            'automatic_purchase':False,'performance_validated':False}
    (folder/'high_payout_department.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    esc=lambda v:html.escape(str(v))
    pct=lambda v:'未集計' if v is None else f'{100*v:.1f}%'
    body='<h1>高配当戦略部｜穴予想特化</h1><p>'+esc(report['scope'])+'</p><p>100倍以上のみ、最大12点。根拠・オッズ・推定期待値が不足なら見送りします。期待度は成立確率ではありません。確率・期待値・配分基準は暫定で、回収率改善は未確認です。</p>'
    for key,label in [('today','今日'),('total','累計')]:
        s=report[key]
        body+=f'<h2>{label}</h2><p>確定対象{s["target_races"]}R／穴を出した{s["predicted_races"]}R／見送り{s["skipped_races"]}R／的中{s["hits"]}R</p><p>的中率 {pct(s["hit_rate"])}｜回収率 {pct(s["return_rate"])}｜平均配当 '+esc(s['average_payout_per_100yen'] if s['average_payout_per_100yen'] is not None else '未集計')+'円（100円あたり）</p>'
        body+='<p>'+esc('／'.join(f'{n}倍以上的中：{count}本' for n,count in s['payout_hit_counts'].items()))+'</p>'
        body+='<ul>'+''.join('<li>'+esc(p)+'：'+str(v['hits'])+'/'+str(v['races'])+'R（'+pct(v['hit_rate'])+'）</li>' for p,v in s['pattern_results'].items())+'</ul>'
    body+='<p><a href="high_payout_history.html">他部署と共通の過去レース検証を見る</a></p><h2>各レースの統括判定</h2><p>以下は保存時の判定・オッズです。現在価格や実購入の承認ではありません。未確定レースは成績に含めません。</p>'
    for d in report['live_decisions']:
        body+='<details><summary>'+esc(d['venue'])+' '+str(d['race_no'])+'R｜'+esc(d['rating'])+'｜'+str(d['recommended_count'])+'点</summary><p>穴期待度 '+str(d['expectation_score'])+'/100｜波乱箇所 '+esc('・'.join(d['patterns']) or 'なし')+'</p>'
        if d['skip_reason']:body+='<p>'+esc(d['skip_reason'])+'</p>'
        for t in d['tickets']:body+='<p>'+esc(t['buy'])+'｜'+f'{t["odds"]:.1f}倍｜推定EV {t["ev"]:.2f}'+'</p>'
        body+='<p>'+esc('／'.join(d['unknowns']))+'</p></details>'
    body+='<h2>担当と確認範囲</h2>'
    if report['live_decisions']:
        for a in report['live_decisions'][0].get('analysts',[]):body+='<p><b>'+esc(a['role'])+'</b>：'+esc(a['evidence_scope'])+'</p>'
    body+='<p>1着波乱・2着突っ込み・3着紛れ・ライン崩壊・人気過剰・期待値審査の担当は、取得項目に基づくルール分析です。未取得の競りやコース取り、ライン崩壊確率は作りません。</p><a href="operations.html">会社の担当報告</a> ／ <a href="../index.html#picks">買い目へ</a>'
    (folder/'high_payout_department.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>高配当戦略部</title><link rel="stylesheet" href="../site-ui.css"><main>'+body+'</main></html>',encoding='utf-8')
    return report
