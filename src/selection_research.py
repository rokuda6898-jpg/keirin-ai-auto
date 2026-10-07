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


def lower_position_coverage(candidates, plan, tickets):
    """Spread supported second/third riders under the original heads and budget."""
    if candidates is None or candidates.empty or not tickets:
        return []
    frame=candidates.drop_duplicates('buy').copy()
    frame=frame[frame.prob.map(lambda p:isinstance(p,(int,float)) and math.isfinite(p) and 0<p<=1)]
    heads={t['buy'].split('-')[0] for t in tickets}
    frame=frame[frame.buy.map(lambda b:b.split('-')[0] in heads)]
    selected=[];covered_second=set();covered_third=set();used=set()
    for group,flag,threshold in [('本線','main_formation',plan.get('main_ev',1.10)),('穴','hole_formation',plan.get('hole_ev',1.25))]:
        pool=frame[frame[flag] & frame.ev.ge(threshold) & ~frame.buy.isin(used)]
        rows=pool.to_dict('records')
        second_mass={};third_mass={}
        for row in rows:
            a,b,c=row['buy'].split('-')
            second_mass[(a,b)]=second_mass.get((a,b),0)+row['prob']
            third_mass[(a,c)]=third_mass.get((a,c),0)+row['prob']
        for _ in range(sum(t['group']==group for t in tickets)):
            if not rows:break
            def rank(row):
                a,b,c=row['buy'].split('-')
                gain=.5*(second_mass[(a,b)] if (a,b) not in covered_second else 0)+.5*(third_mass[(a,c)] if (a,c) not in covered_third else 0)+.01*row['prob']
                return (-gain,-row['prob'],-row['ev'],row['buy'])
            row=min(rows,key=rank);rows.remove(row)
            a,b,c=row['buy'].split('-');covered_second.add((a,b));covered_third.add((a,c));used.add(row['buy'])
            selected.append({'buy':row['buy'],'group':group,'prob':float(row['prob']),'ev':float(row['ev']),'stake_yen':100,'purchase_authorized':False})
    return selected


def lower_miss_audit(rows):
    counts={'hit':0,'first_missing':0,'second_missing_under_correct_first':0,
            'third_missing_under_correct_first_second':0,'skip':0}
    for row in eligible(rows):
        buys=[t['buy'].split('-') for t in row.get('tickets',[])]
        actual=row['actual_trifecta'].split('-')
        if not buys:reason='skip'
        elif actual in buys:reason='hit'
        elif not any(b[0]==actual[0] for b in buys):reason='first_missing'
        elif not any(b[:2]==actual[:2] for b in buys):reason='second_missing_under_correct_first'
        else:reason='third_missing_under_correct_first_second'
        counts[reason]+=1
    return counts


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


def exact_position_comparison(rows):
    """Use only independently saved, pre-close proposals on paired races."""
    pairs = []
    for row in rows:
        challenger = row.get("exact_position_challenger") or {}
        if challenger.get("version") != "cumulative_difference_v1" or challenger.get("snapshot_at") != row.get("snapshot_at"):
            continue
        actual = str(row.get("actual_trifecta", "")).split("-")
        baseline = {str(p["position"]): p["probabilities"] for p in row.get("position_probabilities", [])}
        proposed = challenger.get("probabilities", {})
        if len(actual) != 3 or any(str(p) not in baseline or str(p) not in proposed for p in [2, 3]):
            continue
        valid = True
        for p in [2, 3]:
            old, new = baseline[str(p)], proposed[str(p)]
            valid &= set(old) == set(new) and actual[p-1] in old
            valid &= all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in list(old.values()) + list(new.values()))
            valid &= abs(sum(new.values()) - 1) < 1e-6
        if valid:
            pairs.append((row, baseline, proposed, actual))
    pairs.sort(key=lambda item: (item[0]["close_at"], item[0]["race_id"]))
    # Separate calendar dates; do not split one day's meeting across periods.
    dates = sorted({item[0]["date"] for item in pairs})
    boundary = dates[max(1, len(dates) // 2)] if len(dates) >= 2 else None
    report = {"paired_races": len(pairs), "auto_promotion": False,
              "status": "chronological_comparison" if boundary else "collecting", "test_start_date": boundary}
    for label, part in [("earlier", [v for v in pairs if boundary and v[0]["date"] < boundary]),
                        ("later", [v for v in pairs if boundary and v[0]["date"] >= boundary])]:
        scores = []
        for p in [2, 3]:
            def brier(column):
                return sum(sum((prob - (car == actual[p-1])) ** 2 for car, prob in item[str(p)].items())
                           for _, old, new, actual in part for item in [old if column == "baseline" else new]) / len(part) if part else None
            scores.append({"position": p, "races": len(part), "baseline_brier": brier("baseline"), "challenger_brier": brier("challenger")})
        report[label] = scores
    return report


def build_selection_research(rows, output_dir=OUTPUT_DIR):
    rows = eligible(rows)
    paired = [r for r in rows if (r.get('selection_challenger') or {}).get('version')=='probability_first_v1'
              and r['selection_challenger'].get('snapshot_at') == r['snapshot_at']]
    def totals(challenger, cohort=paired, field='selection_challenger'):
        stake=returned=hits=bet=0
        for row in cohort:
            tickets=row[field]['tickets'] if challenger else row.get('tickets',[])
            if not tickets:continue
            bet+=1;stake+=100*len(tickets)
            if any(t['buy']==row['actual_trifecta'] for t in tickets):
                hits+=1;returned+=row['payout_per_100yen']
        return {'bet_races':bet,'hits':hits,'hit_rate':hits/bet if bet else None,
                'stake_yen':stake,'return_yen':returned,'return_rate':returned/stake if stake else None}
    calibration=chronological_calibration(rows)
    lower_paired=[r for r in rows if (r.get('lower_challenger') or {}).get('version')=='lower_coverage_v1'
                  and r['lower_challenger'].get('snapshot_at')==r['snapshot_at']]
    axis_paired=[r for r in rows if (r.get('axis_challenger') or {}).get('version')=='axis_spread_v1'
                 and r['axis_challenger'].get('snapshot_at')==r['snapshot_at']]
    report={'exact_position_comparison': exact_position_comparison(rows), 'updated_at_jst':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds'),
            'strategy_version':STRATEGY_VERSION,'paired_races':len(paired),'probability_first':totals(True),
            'current_ev_first':totals(False),'calibration':calibration,'auto_promotion':False,
            'lower_miss_audit':lower_miss_audit(rows),'lower_comparison':{'paired_races':len(lower_paired),
            'coverage':totals(True,lower_paired,'lower_challenger'),'baseline':totals(False,lower_paired)},
            'axis_comparison':{'paired_races':len(axis_paired),'spread':totals(True,axis_paired,'axis_challenger'),
                               'baseline':totals(False,axis_paired)}}
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
    page+='<h2>2・3着の取りこぼし</h2><p>実際の1着を含む買い目があったか、1・2着の組み合わせまで含まれていたかを分けます。選手の敗因を断定する分析ではありません。</p><ul>'
    labels={'hit':'的中','first_missing':'1着を買い目に含めていなかった','second_missing_under_correct_first':'1着は含むが正しい2着との組み合わせなし','third_missing_under_correct_first_second':'1・2着の組み合わせは含むが3着で外れ','skip':'見送り'}
    for key,label in labels.items():page+=f'<li>{label}：{report["lower_miss_audit"][key]}件</li>'
    page+='</ul><h2>2・3着を広げる比較案</h2><p>現行と同じ1着候補・期待値基準・点数予算で、2・3着の推定確率を使い、同じ下位選手への偏りを抑えて選びます。締切前に保存した比較案だけを検証します。</p><p>比較可能な確定レース：'+str(len(lower_paired))+'件。</p><table><tr><th>案</th><th>的中／購入対象</th><th>回収率</th><th>投資額</th></tr>'
    for key,label in [('baseline','現行案'),('coverage','2・3着分散案')]:
        r=report['lower_comparison'][key];page+=f'<tr><td>{label}</td><td>{r["hits"]}／{r["bet_races"]}R</td><td>{pct(r["return_rate"])}</td><td>{r["stake_yen"]:,}円</td></tr>'
    page+='</table><h2>1着固定を外す比較案</h2><p>現行が1着固定するレースだけを対象に、同じ点数予算・期待値基準で1着候補を広げます。2・3着の候補範囲は狭めません。検証用で、現行へ自動採用しません。</p><p>比較可能な確定レース：'+str(len(axis_paired))+'件。</p><table><tr><th>案</th><th>的中／購入対象</th><th>回収率</th></tr>'
    for key,label in [('baseline','現行・1着固定'),('spread','1着を広げる案')]:
        r=report['axis_comparison'][key];page+=f'<tr><td>{label}</td><td>{r["hits"]}／{r["bet_races"]}R</td><td>{pct(r["return_rate"])}</td></tr>'
    exact = report['exact_position_comparison']
    page+='</table><h2>ちょうど2着・3着の比較</h2><p>2着以内・3着以内の推定値から差分を取り、ちょうど2着・3着を評価する比較案を締切前に保存します。推定値の大小関係を整えてから差分を計算し、計算できない場合は欠測とします。確率の校正や改善を保証する方法ではありません。</p><p>保存済みの比較可能レース：'+str(exact['paired_races'])+'件。異なる開催日を前半・後半に分け、確率誤差を確認します。自動採用しません。</p><table><tr><th>期間</th><th>着順</th><th>レース数</th><th>現行の誤差</th><th>比較案の誤差</th></tr>'
    for period,label in [('earlier','前半'),('later','後半')]:
        for entry in exact[period]:
            fmt=lambda value: '未集計' if value is None else f'{value:.4f}'
            page+=f'<tr><td>{label}</td><td>{entry["position"]}着</td><td>{entry["races"]}</td><td>{fmt(entry["baseline_brier"])}</td><td>{fmt(entry["challenger_brier"])}</td></tr>'
    page+='</table><p>'+html.escape(calibration['scope'])+'</p><p>比較案は自動採用しません。時系列の再計算は実戦検証と区別します。</p><a href="validation_coverage.html">検証の抜け・条件別成績・データ時刻の確認</a><br><a href="prediction_quality.html">着順別の確率と外れ原因</a> ／ <a href="../index.html">今日の予想</a></main></html>'
    (folder/'selection_research.html').write_text(page,encoding='utf-8')
    return report
