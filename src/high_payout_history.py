"""Chronological shared-history research, isolated from prospective records."""
import json
import html
from itertools import permutations
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from betting_logic import score_riders,race_plan,clean_odds
from high_payout_strategy import select_high_payout,VERSION


def replay_high_payout(race, quotes, train_end):
    date=pd.to_datetime(race.date).min()
    if pd.Timestamp(train_end)>=date:
        raise ValueError('historical model must be trained strictly before the test date')
    # Finish labels are used only after selection, never as department input.
    inputs=race.drop(columns=['finish_pos','target_win','target_top2','target_top3'],errors='ignore').copy()
    inputs['position_model_source']='position_specialists'
    riders=score_riders(inputs)
    market=clean_odds(quotes)
    candidates,decision=select_high_payout(riders,market,race_plan(riders))
    selected=candidates[candidates.high_payout_selected] if not candidates.empty else candidates
    keys={'-'.join(map(str,p)) for p in permutations(riders.car_no.astype(int),3)}
    full=set(market.loc[market.odds_used.lt(9999.9),'buy'])==keys
    scope='unverified_odds_simulation' if full else 'missing_market'
    close=pd.to_numeric(race.get('close_at',pd.Series(index=race.index,dtype=float)),errors='coerce').dropna()
    if full and quotes is not None and not quotes.empty and {'odds_captured_at_jst','odds_verification_status'}.issubset(quotes.columns) and len(close):
        q=quotes[quotes.bet_type.eq('trifecta')]
        stamps=pd.to_datetime(q.odds_captured_at_jst,utc=True,errors='coerce')
        if len(q) and stamps.notna().all() and q.odds_verification_status.eq('verified').all() and stamps.map(lambda s:s.timestamp()<float(close.min())).all():
            scope='verified_preclose_odds_replay'
    actual=race[pd.to_numeric(race.finish_pos,errors='coerce').isin([1,2,3])].sort_values('finish_pos')
    buy='-'.join(str(int(c)) for c in actual.car_no) if len(actual)==3 else None
    hit=bool(buy and selected.buy.eq(buy).any()) if not selected.empty else False
    winning=market[market.buy.eq(buy)]
    simulated_return=float(winning.odds_used.iloc[0])*100 if hit and not winning.empty else 0
    return {'date':str(date.date()),'train_end':str(pd.Timestamp(train_end).date()),'scope':scope,
            'count':len(selected),'hit':hit,'stake_yen':100*len(selected),
            'simulated_return_yen':simulated_return,'rating':decision['rating'],
            'patterns':decision['patterns'],'unknowns':decision['unknowns']}


def build_high_payout_history(records, output_dir, total_history_races=None):
    groups={}
    for scope in ['verified_preclose_odds_replay','unverified_odds_simulation','missing_market']:
        part=[r for r in records if r['scope']==scope]
        bets=[r for r in part if r['count']]
        stake=sum(r['stake_yen'] for r in bets);returned=sum(r['simulated_return_yen'] for r in bets)
        groups[scope]={'races':len(part),'proposed_races':len(bets),'skipped_races':len(part)-len(bets),
                       'hits':sum(r['hit'] for r in bets),'hit_rate':sum(r['hit'] for r in bets)/len(bets) if bets else None,
                       'stake_yen':stake,'simulated_return_yen':returned,
                       'simulated_return_rate':returned/stake if stake else None}
    report={'version':VERSION,'updated_at_jst':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds'),
            'shared_history_races':total_history_races,'evaluated_races':len(records),'groups':groups,
            'scheme':'他部署と同じ履歴・同じ時系列分割・同じ学習済み着順スコアで、新部署の独立ルールを再現。検証日の前までで学習。',
            'scope':'過去再現検証。実戦の締切前保存成績とは別。払戻は保存オッズによる試算で、公式払戻と同一とは未確認。',
            'limitations':['モデルの学習期間は検証日より前。全特徴量の当時の利用可能時刻と、既存のモデル形式選択までの完全な時点一致は未確認。'],
            'status':'completed_research' if records else 'awaiting_shared_history_replay',
            'auto_promotion':False,'prospective_performance_included':False}
    folder=output_dir/'company';folder.mkdir(parents=True,exist_ok=True)
    (folder/'high_payout_history.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    esc=lambda v:html.escape(str(v));pct=lambda v:'未集計' if v is None else f'{100*v:.1f}%'
    body='<h1>高配当戦略部・過去レース検証</h1><p>'+esc(report['scheme'])+'</p><p>'+esc(report['scope'])+'</p><p>共有履歴 '+esc(total_history_races if total_history_races is not None else '実行待ち')+'レース／今回検証 '+str(len(records))+'レース。</p>'
    for scope,label in [('verified_preclose_odds_replay','発走前オッズの時刻・確認状態あり'),('unverified_odds_simulation','オッズ取得時刻未確認・参考シミュレーション'),('missing_market','三連単市場不足')]:
        g=groups[scope];body+='<h2>'+label+'</h2><p>対象'+str(g['races'])+'R／穴あり'+str(g['proposed_races'])+'R／的中'+str(g['hits'])+'R｜的中率'+pct(g['hit_rate'])+'｜保存オッズでの試算回収率'+pct(g['simulated_return_rate'])+'</p>'
    body+='<p>'+esc('／'.join(report['limitations']))+'</p><p>全組み合わせの有効オッズが不足するレースは穴を作りません。未確認のオッズでの好成績を本番採用の根拠にしません。過去の保存予想・実績を書き換えず、週次の共有履歴検証で更新します。</p><a href="high_payout_department.html">新部署の実戦検証へ</a>'
    (folder/'high_payout_history.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>高配当・過去検証</title><link rel="stylesheet" href="../site-ui.css"><main>'+body+'</main></html>',encoding='utf-8')
    return report
