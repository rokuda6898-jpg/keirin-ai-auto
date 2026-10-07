"""Independent longshot department. Provisional estimates, no purchase authority."""
from itertools import permutations
import math
import pandas as pd

VERSION = 'high_payout_v1'
MIN_PROBABILITY = .001  # provisional: reject below 0.1%, even with huge odds


def select_high_payout(riders, market, risk, main_buys=(), minimum_ev=1.25):
    minimum_ev=max(1.25,float(minimum_ev))
    columns=['buy','bet_type','prob','odds_used','ev','head','second','third','scenario','hole_pattern','high_payout_selected']
    rows=[]
    report={'version':VERSION,'rating':'見送り','expectation_score':0,'recommended_count':0,
            'patterns':[],'tickets':[],'skip_reason':'このレースは穴予想見送り',
            'unknowns':['番手競り・分断・コース取り・当日状態・展開成立確率・類似パターン成績は未検証。'],
            'probability_method':'geometric_model_market_estimate_uncalibrated',
            'minimum_probability':MIN_PROBABILITY,'minimum_ev':minimum_ev,'automatic_purchase':False}
    report['analysts']=[{'role':role,'status':'検証中','evidence_scope':scope} for role,scope in [
        ('高配当専用軍師','各担当の根拠・市場乖離・リスクと候補数から出す／見送る・点数を決定。'),
        ('1着波乱担当','市場1着人気とAI1着順位、荒れ指数。競り・仕掛け不発・当日状態は未確認。'),
        ('2着突っ込み担当','市場人気薄と2着スコアの一致。番手差し・外伸び・コース取りは未確認。'),
        ('3着紛れ担当','人気薄と3着スコアの一致。内残り・落車回避・流れ込みは未確認。'),
        ('ライン崩壊担当','並び確認済みの別線組み合わせを仮説評価。分断・援護失敗・崩壊確率は未確認。'),
        ('細切れ戦担当','確認済み並びが4ライン以上で、別線の上位着順スコアを評価。先行争い・仕掛け時刻は未確認。'),
        ('単騎担当','確認済みの1人ラインと着順別スコアを照合。位置取り・当日の追走先は未確認。'),
        ('人気過剰担当','全三連単市場の1着人気順位とAI評価順位の乖離。地元・知名度の影響は未確認。'),
        ('高配当期待値担当','全市場とモデルを合わせた暫定確率・EV・最低確率を審査。過去類似パターンは未確認。')]]
    report['analysts'] += [{'role':'データ確認担当','status':'検証中','evidence_scope':'全組み合わせオッズを確認。欠場・並び・取得時刻は取得情報の範囲だけ確認。'}, {'role':'比較検証担当','status':'検証中','evidence_scope':'締切前に現在方式と3着重点方式を同時保存。確定結果で別集計し、自動採用しない。'}]
    report['comparison']={'version':'third_focus_v1','automatic_promotion':False,'variants':{'current':[], 'third_focus':[]}}
    report['risk_coordination']={'chaos_index':risk['chaos_index'],'risk_level':risk.get('risk_level','未取得'),
                                 'line_collapse_probability':None,'scenario_probability':None,
                                 'skip_recommendation':'順当寄りなら見送り。その他は各候補を審査。'}
    cars=riders.car_no.astype(int).tolist()
    keys={'-'.join(map(str,p)) for p in permutations(cars,3)}
    quotes={str(r.buy):float(r.odds_used) for r in market.itertuples()
            if math.isfinite(float(r.odds_used)) and 1 <= float(r.odds_used) < 9999.9 and str(r.buy) in keys}
    if set(quotes)!=keys:
        report['unknowns'].append('全組み合わせの有効オッズが不足。市場確率との比較不可。')
        return pd.DataFrame(rows,columns=columns),report
    if risk['chaos_index']<30 and riders.p_win.max()>=.6 and risk['first_gap']>=20:
        report['skip_reason']+='（順当決着寄り）'
        return pd.DataFrame(rows,columns=columns),report
    inverse_sum=sum(1/o for o in quotes.values())
    by=riders.set_index('car_no')
    # Full-market first-place marginals give a transparent popularity proxy.
    marginal={car:sum(1/o for buy,o in quotes.items() if int(buy.split('-')[0])==car) for car in cars}
    popularity={car:i+1 for i,car in enumerate(sorted(cars,key=lambda c:(-marginal[c],c)))}
    line_verified=('line_verification_status' in riders and riders.line_verification_status.eq('verified').all()
                   and 'line_id' in riders and riders.line_id.notna().all()
                   and riders.line_id.astype(str).str.strip().ne('').all())
    line_sizes=riders.groupby('line_id').size().to_dict() if line_verified else {}
    report['specialist_analysis']={'細切れ戦':{'status':'分析可能' if line_verified else '並び未確認・判定保留','line_count':len(line_sizes) if line_verified else None},
                                   '単騎':{'status':'分析可能' if line_verified else '並び未確認・判定保留','cars':[int(c) for c in cars if line_sizes.get(by.at[c,'line_id'])==1] if line_verified else []}}
    if not line_verified: report['unknowns'].append('並びの確認状態が不足。ライン崩壊は判定保留。')
    for a,b,c in permutations(cars,3):
        buy=f'{a}-{b}-{c}'; odds=quotes[buy]
        if odds<100 or buy in main_buys:continue
        patterns=[]
        if popularity[a]>=4 and by.at[a,'rank_first']<=3 and risk['chaos_index']>=45:patterns.append('1着荒れ')
        if popularity[a]<=3 and popularity[b]>=4 and by.at[b,'rank_second']<=4:patterns.append('2着荒れ')
        if popularity[a]<=3 and popularity[b]<=3 and popularity[c]>=4 and by.at[c,'rank_third']<=4:patterns.append('3着荒れ')
        if line_verified and 'line_id' in by and popularity[a]<=3 and by.at[a,'line_id']!=by.at[b,'line_id'] and risk['chaos_index']>=60:patterns.append('ライン崩壊')
        overpopular=[car for car in cars if popularity[car]<=2 and by.at[car,'rank_first']-popularity[car]>=2]
        if overpopular and a not in overpopular and by.at[a,'rank_first']<=3:patterns.append('人気過剰')
        if not patterns:continue
        second_mass=sum(by.at[x,'score_second'] for x in cars if x!=a)
        third_mass=sum(by.at[x,'score_third'] for x in cars if x not in (a,b))
        model=float(by.at[a,'p_win']*by.at[b,'score_second']/second_mass*by.at[c,'score_third']/third_mass)
        market_prob=(1/odds)/inverse_sum
        # Shrink extreme model disagreement toward the complete market; never
        # normalize over the longshot pool or claim this is calibrated EV.
        probability=math.sqrt(model*market_prob)
        ev=probability*odds
        if probability<MIN_PROBABILITY or ev<minimum_ev:continue
        # Evidence tags do not inflate chaos, expectation score or ticket limits.
        if line_verified:
            if len(line_sizes)>=4 and by.at[a,'line_id']!=by.at[b,'line_id'] and by.at[a,'rank_first']<=3 and by.at[b,'rank_second']<=4:
                patterns.append('細切れ戦')
            if any(line_sizes.get(by.at[car,'line_id'])==1 and popularity[car]>=4 and by.at[car,rank]<=4 for car,rank in [(a,'rank_first'),(b,'rank_second'),(c,'rank_third')]):
                patterns.append('単騎')
        rows.append(dict(zip(columns,[buy,'trifecta',probability,odds,ev,a,b,c,'高配当独立分析',patterns,False])))
    frame=pd.DataFrame(rows,columns=columns)
    if frame.empty:return frame,report
    patterns=sorted({p for row in rows for p in row['hole_pattern']})
    score=min(100,round(.5*risk['chaos_index']+10*len([p for p in patterns if p not in ('細切れ戦','単騎')])+min(20,5*len(frame))))
    rating='S' if score>=85 else 'A' if score>=70 else 'B' if score>=55 else 'C'
    # Twelve points additionally need twelve candidates with stronger provisional
    # EV, not twelve barely qualifying prices. These are design rules, not fits.
    strong_count=int(frame.ev.ge(1.5).sum())
    limit=12 if score>=85 and risk['chaos_index']>=60 and strong_count>=12 else 10 if score>=70 else 8 if score>=55 else 6 if score>=40 else 0
    if limit==0:return frame,report
    chosen=frame.sort_values(['ev','prob','buy'],ascending=[False,False,True]).head(limit)
    frame.loc[chosen.index,'high_payout_selected']=True
    third_pool=frame[frame.hole_pattern.map(lambda patterns:'3着荒れ' in patterns)]
    third_chosen=third_pool.sort_values(['prob','ev','buy'],ascending=[False,False,True]).head(len(chosen))
    pack=lambda f:[{'buy':r.buy,'odds':r.odds_used,'probability':r.prob,'ev':r.ev} for r in f.itertuples()]
    report['comparison']['variants']={'current':pack(chosen),'third_focus':pack(third_chosen)}
    report['comparison']['decision']='本番は現在方式。3着重点方式は同じ上限点数・同じ最低EVで成立確率順に比較。'
    report.update(rating=rating,expectation_score=score,recommended_count=len(chosen),patterns=patterns,
                  strong_ev_candidate_count=strong_count, twelve_point_min_ev=1.5,
                  tickets=[{'buy':r.buy,'odds':r.odds_used,'probability':r.prob,'ev':r.ev,'patterns':r.hole_pattern} for r in chosen.itertuples()],skip_reason='')
    return frame,report
