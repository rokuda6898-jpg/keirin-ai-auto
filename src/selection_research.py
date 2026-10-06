"""Frozen probability-first challenger and race-separated calibration diagnostics."""
import html
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

from common import OUTPUT_DIR
from betting_logic import STRATEGY_VERSION
from prediction_quality import valid_row


def probability_first(candidates, plan, main_count, hole_count):
    """Preserve original EV gates, formation and ticket budget; change ranking only."""
    if candidates is None or candidates.empty or main_count + hole_count == 0:
        return []
    frame = candidates.drop_duplicates('buy').copy()
    frame = frame[frame.prob.map(lambda p: isinstance(p,(int,float)) and math.isfinite(p) and 0 < p <= 1)]
    ranked = frame.sort_values(['prob','ev','buy'],ascending=[False,False,True])
    main = ranked[ranked.main_formation & ranked.ev.ge(plan.get('main_ev',1.10))].head(main_count)
    holes = ranked[ranked.hole_formation & ranked.ev.ge(plan.get('hole_ev',1.25)) & ~ranked.buy.isin(main.buy)].head(hole_count)
    return [{'buy':str(row.buy),'group':group,'prob':float(row.prob),'ev':float(row.ev),'stake_yen':100,'purchase_authorized':False}
            for group,part in [('本線',main),('穴',holes)] for row in part.itertuples()]


def eligible(rows):
    latest = {}
    for row in rows:
        payout=row.get('payout_per_100yen')
        if row.get('strategy_version') != STRATEGY_VERSION or not valid_row(row) or not isinstance(payout,(int,float)) or not math.isfinite(payout) or payout <= 0:
            continue
        rid = row['race_id']
        if rid not in latest or row['snapshot_at'] > latest[rid]['snapshot_at']:
            latest[rid] = row
    return sorted(latest.values(),key=lambda r:(r['close_at'],r['race_id']))


def samples(rows):
    return [(float(t['prob']),int(t['buy']==r['actual_trifecta'])) for r in rows for t in r.get('tickets',[])
            if isinstance(t.get('prob'),(int,float)) and math.isfinite(t['prob']) and 0<=t['prob']<=1]


def chronological_calibration(rows, minimum_train=100, minimum_test=50):
    rows = [row for row in eligible(rows) if samples([row])]
    split = max(minimum_train,int(len(rows)*.7))
    train,test = rows[:split],rows[split:]
    summary = {'status':'collecting','train_races':len(train),'test_races':len(test),
               'minimum_train_races':minimum_train,'minimum_test_races':minimum_test,'auto_apply':False,
               'scope':'時系列で分けた保存済み選択買い目の検証。全候補や着順確率へは適用しない。'}
    if len(train)<minimum_train or len(test)<minimum_test or train[-1]['close_at']>=test[0]['close_at']:
        return summary
    training = samples(train)
    testing = samples(test)
    if not training or not testing:
        return summary
    edges = [0,.01,.02,.05,.1,.2,.4,.6,.8,1.01]
    bins=[]
    for lower,upper in zip(edges,edges[1:]):
        values=[(p,y) for p,y in training if lower<=p<upper]
        expected=sum(p for p,_ in values)
        # Shrink toward the existing estimate rather than fitting zero from rare losses.
        factor=(sum(y for _,y in values)+20)/(expected+20)
        bins.append({'lower':lower,'upper':upper,'factor':factor,'training_tickets':len(values)})
    corrected=[]
    for p,y in testing:
        factor=next(b['factor'] for b in bins if b['lower']<=p<b['upper'])
        corrected.append((min(1,max(0,p*factor)),y))
    brier=lambda values:sum((p-y)**2 for p,y in values)/len(values)
    summary.update(status='offline_holdout',training_bins=bins,test_tickets=len(testing),
                   raw_brier=brier(testing),corrected_brier=brier(corrected),
                   train_last_close=train[-1]['close_at'],test_first_close=test[0]['close_at'])
    return summary


def build_selection_research(rows, output_dir=OUTPUT_DIR):
    rows = eligible(rows)
    paired = [r for r in rows if (r.get('selection_challenger') or {}).get('version')=='probability_first_v1'
              and r['selection_challenger'].get('snapshot_at') == r['snapshot_at']]
    def totals(challenger):
        stake=returned=hits=bet=0
        for row in paired:
            tickets=row['selection_challenger']['tickets'] if challenger else row.get('tickets',[])
            if not tickets:continue
            bet+=1;stake+=100*len(tickets)
            if any(t['buy']==row['actual_trifecta'] for t in tickets):
                hits+=1;returned+=row['payout_per_100yen']
        return {'bet_races':bet,'hits':hits,'hit_rate':hits/bet if bet else None,
                'stake_yen':stake,'return_yen':returned,'return_rate':returned/stake if stake else None}
    calibration=chronological_calibration(rows)
    report={'updated_at_jst':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds'),
            'strategy_version':STRATEGY_VERSION,'paired_races':len(paired),'probability_first':totals(True),
            'current_ev_first':totals(False),'calibration':calibration,'auto_promotion':False}
    folder=output_dir/'company';folder.mkdir(parents=True,exist_ok=True)
    (folder/'selection_research.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pct=lambda v:'未集計' if v is None else f'{v*100:.1f}%'
    table=''
    for key,label in [('current_ev_first','現行・期待値順'),('probability_first','比較案・的中確率順')]:
        r=report[key];table+=f'<tr><td>{label}</td><td>{r["hits"]}／{r["bet_races"]}R</td><td>{pct(r["hit_rate"])}</td><td>{pct(r["return_rate"])}</td><td>{r["stake_yen"]:,}円</td></tr>'
    page='<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>確率補正と買い目の比較</title><link rel="stylesheet" href="../site-ui.css"><main><h1>確率補正と買い目の比較</h1><h2>買い目を絞る方法の比較</h2><p>現行と同じフォーメーション・期待値基準・本線と穴の点数上限を使い、的中確率順で選ぶ比較案を締切前に保存します。各点100円の試算です。候補不足なら埋めません。</p><p>比較可能な確定レース：'+str(len(paired))+'件。過去の候補を後付けで作り直しません。</p><table><tr><th>案</th><th>的中／購入対象</th><th>的中率</th><th>回収率</th><th>投資額</th></tr>'+table+'</table><h2>確率補正の検証</h2><p>古いレースだけで補正係数を作り、新しいレースで検証します。同じレースを学習と検証に分けません。最低100レースで作成、別の50レースで検証します。</p><p>現在：作成側 '+str(calibration['train_races'])+'レース／検証側 '+str(calibration['test_races'])+'レース。</p>'
    if calibration['status']=='offline_holdout':
        page+=f'<p>確率の誤差（小さいほど良い）：補正前 {calibration["raw_brier"]:.6f}／補正後 {calibration["corrected_brier"]:.6f}</p>'
    else:page+='<p>必要な保存済みレースが不足しています。補正の有効性は未確認です。</p>'
    page+='<p>'+html.escape(calibration['scope'])+'</p><p>現行予想の確率・買い目を自動変更する段階ではありません。時系列の再計算は実戦検証と区別します。</p><a href="prediction_quality.html">着順別の確率と外れ原因</a> ／ <a href="../index.html">今日の予想</a></main></html>'
    (folder/'selection_research.html').write_text(page,encoding='utf-8')
    return report
