"""One immutable model; an entire recent year is held out until scoring.

This is a retrospective archive study, not a backdated prediction ledger.
The four chronological fitting phases all precede the held-out year.
"""
import gc
import gzip
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import fixed_year_features as features
import fixed_year_models as models
import fusion_equations as fusion
import equation_models as legacy
from annual_knowledge import build_annual_profiles

VERSION='fixed_older_training_recent_year_v1'
NAMES={'original':'順位別モデル','risk':'リスク部の現行評価式','data':'データ部・順位保持',
       'pace':'展開部・順位保持','line':'ライン部・順位保持','pairwise':'先着関係式',
       'data_legacy':'データ部・２／３着を従来式で上書き','pace_legacy':'展開部・２／３着を従来式で上書き',
       'line_legacy':'ライン部・２／３着を従来式で上書き',
       'joint':'１・２着に応じた組み合わせ式','fusion':'統合式',
       'without_departments':'統合式から部署の分布を除外','without_pairwise':'統合式から先着関係式を除外',
       'without_joint':'統合式から組み合わせ式を除外'}


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def phase_split(training):
    days=sorted(training.date.astype(str).unique())
    if len(days)<40:raise ValueError('at least 40 older days required for four fitting periods')
    cuts=[0,int(len(days)*.6),int(len(days)*.8),int(len(days)*.9),len(days)]
    return [training[training.date.isin(days[a:b])].reset_index(drop=True) for a,b in zip(cuts,cuts[1:])]


def input_reason(race):
    cars=pd.to_numeric(race.car_no,errors='coerce')
    if not 3<=len(race)<=9 or not cars.between(1,9).all() or not cars.mod(1).eq(0).all() or cars.nunique()!=len(race):
        return 'invalid_or_duplicate_car'
    if race.player_id.isna().any() or race.player_id.nunique()!=len(race):return 'invalid_or_duplicate_player'
    if race.date.nunique()!=1:return 'inconsistent_race_date'
    if 'entries_number' in race:
        expected=pd.to_numeric(race.entries_number,errors='coerce')
        if expected.notna().any() and not expected.dropna().eq(len(race)).all():return 'incomplete_field'
    return None


def outcome(race):
    """Tied or incomplete podiums are accounted for, not silently resolved."""
    pos=pd.to_numeric(race.finish_pos,errors='coerce')
    cars=[]
    for p in (1,2,3):
        found=race.loc[pos.eq(p),'car_no']
        if len(found)!=1:return None
        cars.append(int(found.iloc[0]))
    return '-'.join(map(str,cars))


def eligible(frame,labels=False):
    selected=[];excluded=Counter()
    for rid,race in frame.groupby('race_id',sort=False):
        reason=input_reason(race)
        if not reason and labels and outcome(race) is None:reason='ambiguous_or_missing_podium'
        if reason:excluded[reason]+=1
        else:selected.append(str(rid))
    return frame[frame.race_id.isin(selected)].reset_index(drop=True),dict(excluded)


def base_predict(frame,bundle):
    out=features.frozen_priors(frame,bundle['reference'])
    X,_=features.matrix(out,bundle['feature_config'])
    for p,name in enumerate(('p_win','p_second','p_third'),1):
        values=bundle['classifiers'][p].predict_proba(X)[:,1]
        out[name]=np.maximum(values,1e-8)
        out[name]/=out.groupby('race_id')[name].transform('sum')
    out['position_model_source']='fixed_year_exact_positions'
    # Do not carry forbidden result fields or unused archive columns into equations.
    keep={'race_id','date','venue','race_class','car_no','player_id','finish_pos','entries_number',
          'p_win','p_second','p_third','position_model_source','score','back_count','recent_avg_finish',
          'place2_rate','place3_rate','line_id','line_position','line_verification_status','wind_speed',
          *features.PRIOR_COLUMNS}
    keep.update(f'{prefix}_place{p}_rate' for prefix in ('track','weather','race_type','line_role') for p in (2,3))
    return out[[c for c in out if c in keep]]


def source_hashes():
    root=Path(__file__).parent
    names=('fixed_year_research.py','fixed_year_study.py','fixed_year_features.py','fixed_year_models.py',
           'fusion_equations.py','equation_models.py','annual_knowledge.py','common.py','betting_logic.py',
           'department_ticket_v2.py','department_experiment_v2.py','department_context.py')
    return {n:hashlib.sha256((root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in names}


def train(training,folder,cutoff,fast=False):
    if not pd.to_datetime(training.date).lt(pd.Timestamp(cutoff)).all():
        raise ValueError('training contains held-out dates')
    training,excluded=eligible(training,labels=True)
    phases=phase_split(training)
    info=[]
    for name,frame in zip(('base','equations','mixture','temperature'),phases):
        info.append({'name':name,'races':int(frame.race_id.nunique()),'rows':len(frame),
                     'first':min(frame.date),'last':max(frame.date),'days':int(frame.date.nunique())})
    print('FIXED_YEAR_PHASES '+json.dumps(info),flush=True)
    base=phases[0]
    enriched,reference=features.daily_priors(base)
    X,feature_config=features.matrix(enriched)
    target=pd.to_numeric(base.finish_pos,errors='coerce')
    classifiers={}
    for p in (1,2,3):
        estimator=HistGradientBoostingClassifier(max_iter=3 if fast else 180,learning_rate=.05,
            max_leaf_nodes=31,l2_regularization=.03,early_stopping=False,random_state=41+p)
        estimator.fit(X,target.eq(p).astype(int));classifiers[p]=estimator
        print(f'FIXED_YEAR_PROGRESS base_position_{p}_fitted',flush=True)
    bundle={'version':VERSION,'reference':reference,'feature_config':feature_config,'classifiers':classifiers}
    del X,enriched;gc.collect()
    profile_input=folder/'profile_training.csv'
    base.to_csv(profile_input,index=False)
    profile_cutoff=str((pd.Timestamp(max(base.date))+pd.Timedelta(days=1)).date())
    knowledge=build_annual_profiles(profile_cutoff,profile_input,folder/'profiles')
    profiles=knowledge['profiles'];bundle['profiles']=profiles
    print(f'FIXED_YEAR_PROGRESS older_profiles_fitted {len(profiles)}',flush=True)
    # The equations see base-model predictions from a later, disjoint OLD period.
    eq=phases[1];actuals={str(rid):outcome(r) for rid,r in eq.groupby('race_id',sort=False)}
    predicted=base_predict(eq,bundle)
    lengths=[len(r)*(len(r)-1)*(len(r)-2) for _,r in predicted.groupby('race_id',sort=False)]
    scratch=folder/'joint_design.f32'
    X=np.memmap(scratch,dtype=np.float32,mode='w+',shape=(sum(lengths),122))
    offset=0;targets=[];px=[];py=[];pw=[]
    for count,(rid,race) in enumerate(predicted.groupby('race_id',sort=False),1):
        packet=models.make_packet(race,profiles);ts=fusion.triples(packet);ctx=fusion.feature_context(packet)
        designs=np.asarray([fusion.joint_features(packet,t,ctx) for t in ts],dtype=np.float32)
        if designs.shape[1]!=122:raise ValueError('joint schema changed')
        X[offset:offset+len(ts)]=designs;offset+=len(ts)
        actual=actuals[str(rid)];targets.append(fusion.keys(packet).index(actual))
        fs=legacy.rider_features(packet['source']['evidence']['inputs']['risk_department'])
        ranks={car:i for i,car in enumerate(map(int,actual.split('-')))}
        pairs=[(a,b) for a,b in combinations(sorted(fs),2) if a in ranks or b in ranks]
        for a,b in pairs:
            px.append([u-v for u,v in zip(fs[a],fs[b])]);py.append(float(ranks.get(a,3)<ranks.get(b,3)));pw.append(1/len(pairs))
        if count%1000==0:print(f'FIXED_YEAR_PROGRESS equation_designs {count}/{len(lengths)}',flush=True)
    model={'joint_beta':models.fit_joint_matrix(X,lengths,targets,iterations=2 if fast else None),
           'pair_beta':models.fit_pairwise(px,py,pw)}
    del X,predicted,px,py,pw;gc.collect();scratch.unlink()
    print('FIXED_YEAR_PROGRESS equations_fitted',flush=True)
    records=[]
    actuals={str(rid):outcome(r) for rid,r in phases[2].groupby('race_id',sort=False)}
    for rid,race in base_predict(phases[2],bundle).groupby('race_id',sort=False):
        records.append((actuals[str(rid)],models.components(model,models.make_packet(race,profiles))))
    model['pool']=fusion.fit_pool(records);del records;gc.collect()
    print('FIXED_YEAR_PROGRESS mixture_fitted',flush=True)
    losses={tau:0. for tau in fusion.CONFIG['temperatures']};n=0
    actuals={str(rid):outcome(r) for rid,r in phases[3].groupby('race_id',sort=False)}
    for rid,race in base_predict(phases[3],bundle).groupby('race_id',sort=False):
        distribution=fusion.mix(model['pool'],models.components(model,models.make_packet(race,profiles)))
        for tau in losses:losses[tau]-=math.log(fusion.temperature(distribution,tau)[actuals[str(rid)]])
        n+=1
    losses={tau:loss/n for tau,loss in losses.items()}
    model['temperature']=min(losses,key=lambda tau:(losses[tau],abs(tau-1)))
    model['calibration_fit_losses']=losses;bundle['equations']=model
    bundle['training_cutoff_exclusive']=str(cutoff)
    joblib.dump(bundle,folder/'frozen_model.joblib',compress=3)
    manifest={'version':VERSION,'training_cutoff_exclusive':str(cutoff),'training_exclusions':excluded,
              'phases':info,'model_sha256':digest_file(folder/'frozen_model.joblib'),
              'source_hashes':source_hashes(),'pool':model['pool'],'temperature':model['temperature'],
              'calibration_fit_losses':losses,'features':feature_config['features'],
              'model_update_during_holdout':False,'profile_cutoff_exclusive':profile_cutoff,
              'retrospective_only':True,'ceo_integration':False,'purchase_authorized':False,
              'missing_families':['market_residual','rank_specific_market','observed_state_paths']}
    write_json(folder/'frozen_manifest.json',manifest)
    print('FIXED_YEAR_FROZEN '+json.dumps({k:v for k,v in manifest.items() if k in ('model_sha256','phases','temperature','training_exclusions')}),flush=True)
    return bundle


def predict(target,folder):
    manifest=json.loads((folder/'frozen_manifest.json').read_text(encoding='utf-8'))
    model_path=folder/'frozen_model.joblib'
    if digest_file(model_path)!=manifest['model_sha256']:raise ValueError('frozen model was modified')
    if source_hashes()!=manifest['source_hashes']:raise ValueError('source changed after fit')
    bundle=joblib.load(model_path)
    # Labels are removed BEFORE input validation, feature generation or prediction.
    clean=features.mask_outcomes(target)
    clean,excluded=eligible(clean)
    rows=base_predict(clean,bundle)
    path=folder/'holdout_predictions.jsonl.gz'
    count=0
    with gzip.open(path,'wt',encoding='utf-8') as handle:
        for rid,race in rows.groupby('race_id',sort=False):
            packet=models.make_packet(race,bundle['profiles'])
            distributions=models.prediction(bundle['equations'],packet)
            key=fusion.keys(packet)
            if any(not fusion.valid_distribution(p,key) for p in distributions.values()):
                raise ValueError('invalid full-field probability distribution')
            row={'race_id':str(rid),'date':str(race.iloc[0].date),'field':len(race),
                 'race_class':str(race.iloc[0].get('race_class','missing')),'keys':key,
                 'distributions':{name:[p[k] for k in key] for name,p in distributions.items()},
                 'scenarios':packet['auxiliary']['departments']}
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
            count+=1
            if count%1000==0:print(f'FIXED_YEAR_PROGRESS holdout_predicted {count}',flush=True)
    if digest_file(model_path)!=manifest['model_sha256']:raise ValueError('model changed during holdout')
    result={'predicted_races':count,'input_exclusions':excluded,'model_sha256':manifest['model_sha256'],
            'predictions_sha256':digest_file(path),'outcomes_accessed':False,'model_updated':False,
            'first':min(rows.date),'last':max(rows.date),'retrospective_only':True}
    write_json(folder/'prediction_manifest.json',result)
    print('FIXED_YEAR_PREDICTED '+json.dumps(result),flush=True)


def metrics(distribution,actual):
    result=fusion.probability_scores(distribution,actual)
    ranked=sorted(distribution,key=lambda k:(-distribution[k],tuple(map(int,k.split('-')))))
    for k in (1,6,12):result[f'top{k}_hit']=int(actual in ranked[:k])
    actual_parts=actual.split('-');picked=ranked[0].split('-')
    result['first_hit']=int(picked[0]==actual_parts[0])
    result['first_pair_hit']=int(picked[:2]==actual_parts[:2])
    first_failure=next((i+1 for i in range(3) if picked[i]!=actual_parts[i]),0)
    for i in range(4):result[f'first_error_at_{i}']=int(first_failure==i)
    # Oracle prefix diagnostics isolate the third-place decision from prior errors.
    prefix='-'.join(actual_parts[:2])+'-'
    conditional=max((k for k in ranked if k.startswith(prefix)),key=lambda k:distribution[k])
    result['third_hit_given_true_pair']=int(conditional==actual)
    return result


def assess(target,folder):
    manifest=json.loads((folder/'prediction_manifest.json').read_text(encoding='utf-8'))
    predictions=folder/'holdout_predictions.jsonl.gz'
    if digest_file(predictions)!=manifest['predictions_sha256']:raise ValueError('prediction evidence changed')
    answers={str(rid):outcome(r) for rid,r in target.groupby('race_id',sort=False)}
    grouped=defaultdict(lambda:defaultdict(Counter));counts=Counter();exclusions=Counter();daily=defaultdict(Counter)
    scenarios=defaultdict(Counter);scenarios_same=Counter();seen=set()
    with gzip.open(predictions,'rt',encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line);rid=row['race_id'];actual=answers.get(rid)
            if rid in seen:raise ValueError('duplicate race prediction')
            seen.add(rid)
            if not actual:exclusions['ambiguous_or_missing_podium']+=1;continue
            groups=('all','month:'+row['date'][:7],'field:'+str(row['field']),'class:'+row['race_class'])
            ms={name:metrics(dict(zip(row['keys'],values)),actual) for name,values in row['distributions'].items()}
            for group in groups:
                counts[group]+=1
                for name,m in ms.items():grouped[group][name].update(m)
            for name,m in ms.items():
                daily[row['date']][name]+=m['top12_hit']
                for k in (1,6,12):
                    if m[f'top{k}_hit'] and not ms['risk'][f'top{k}_hit']:grouped['all'][name][f'rescued_top{k}']+=1
                    if ms['risk'][f'top{k}_hit'] and not m[f'top{k}_hit']:grouped['all'][name][f'lost_top{k}']+=1
            orders={name:'-'.join(map(str,v['top3_cars'])) for name,v in row['scenarios'].items()}
            for name,pick in orders.items():scenarios[name]['races']+=1;scenarios[name]['hits']+=int(pick==actual)
            for a,b in combinations(orders,2):scenarios_same[a+'|'+b]+=int(orders[a]==orders[b])
    total=counts['all']
    if not total:raise ValueError('no evaluable held-out races')
    summary={g:{'races':counts[g],'methods':{name:{k:float(v/counts[g]) for k,v in sums.items()}
             for name,sums in methods.items()}} for g,methods in grouped.items()}
    # One common day-resampling schedule for all methods; descriptive, not promotion tests.
    day_rows=[daily[d] for d in sorted(daily)];rng=np.random.default_rng(43019)
    day_n=Counter(str(r.date) for r in target[['race_id','date']].drop_duplicates().itertuples() if str(r.race_id) in seen and answers.get(str(r.race_id)))
    sizes=np.asarray([day_n[d] for d in sorted(daily)]);ix=rng.integers(0,len(day_rows),size=(2000,len(day_rows)))
    comparisons={};preservation={}
    for name in NAMES:
        delta=np.asarray([d[name]-d['risk'] for d in day_rows]);draw=delta[ix].sum(axis=1)/sizes[ix].sum(axis=1)
        comparisons[name]={'top12_hit_difference':float(delta.sum()/total),
            'day_bootstrap_95_percent_interval':[float(x) for x in np.quantile(draw,[.025,.975])],
            'interval_scope':'descriptive; multiple formulas inspected; not a deployment decision'}
    for name in ('data','pace','line'):
        delta=np.asarray([d[name]-d[name+'_legacy'] for d in day_rows]);draw=delta[ix].sum(axis=1)/sizes[ix].sum(axis=1)
        preservation[name]={'top12_hit_difference':float(delta.sum()/total),
            'day_bootstrap_95_percent_interval':[float(x) for x in np.quantile(draw,[.025,.975])],
            'comparison':'same department first-place score; preserve vs overwritten 2nd/3rd scores; no prices'}
    result={'version':VERSION,'status':'retrospective_holdout_scored','predicted_races':manifest['predicted_races'],
            'evaluated_races':total,'outcome_exclusions':dict(exclusions),'input_exclusions':manifest['input_exclusions'],
            'summary':summary,'comparisons_to_risk_formula':comparisons,'position_preservation_comparison':preservation,
            'department_scenarios':dict(scenarios),
            'department_scenario_agreement':{k:v/total for k,v in scenarios_same.items()},
            'model_sha256':manifest['model_sha256'],'predictions_sha256':manifest['predictions_sha256'],
            'roi':None,'high_payout_dependency':None,'main_hole_comparison_available':False,
            'limitations':['保存済み標本の過去検証。全国全レースではない。',
                '結果列は予想から除外したが、元の出走表の取得時点が締切前だったことは全件証明できない。',
                '当時の締切前オッズがないため、上位６・12候補は無価格の予想診断。本線・穴の買い目や回収率ではない。',
                'リスク部の現行評価式を再利用した再計算であり、当時の実際の買い目・実績を再現したものではない。',
                '市場補正式・順位別市場支持・途中隊列式は必要データ不足で未検証。',
                '部署除外は分布の合成対象からの除外。組み合わせ式に残る部署特徴の完全な除去ではない。',
                'この１年の結果を使って次の式を修正した場合、同じ１年を再び未使用の検証期間とは呼べない。'],
            'model_update_during_holdout':False,'ceo_integration':False,'purchase_authorized':False}
    write_json(folder/'results.json',result)
    render_report(result,folder)
    compact={k:v for k,v in result.items() if k not in ('summary','department_scenario_agreement')}
    compact['overall']=summary['all'];compact['months']={g:v for g,v in summary.items() if g.startswith('month:')}
    print('FIXED_YEAR_RESULTS '+json.dumps(compact,ensure_ascii=False),flush=True)
    return result


def render_report(result,folder):
    inventory=json.loads((folder/'inventory.json').read_text(encoding='utf-8'))
    frozen=json.loads((folder/'frozen_manifest.json').read_text(encoding='utf-8'))
    overall=result['summary']['all']['methods']
    rows=''.join(f'<tr><th>{html.escape(NAMES[name])}</th><td>{v["top1_hit"]:.2%}</td><td>{v["top6_hit"]:.2%}</td><td>{v["top12_hit"]:.2%}</td><td>{v["nll"]:.4f}</td><td>{v["third_given_pair_nll"]:.4f}</td></tr>' for name,v in overall.items())
    phases=''.join(f'<li>{html.escape(p["name"])}：{p["first"]}〜{p["last"]}／{p["races"]:,}レース</li>' for p in frozen['phases'])
    limitations=''.join(f'<li>{html.escape(x)}</li>' for x in result['limitations'])
    months=''.join(f'<tr><th>{g[6:]}</th><td>{v["races"]:,}</td><td>{v["methods"]["risk"]["top12_hit"]:.2%}</td><td>{v["methods"]["fusion"]["top12_hit"]:.2%}</td></tr>' for g,v in sorted(result['summary'].items()) if g.startswith('month:'))
    doc=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>１年前より古いデータで学習・直近１年で検証</title>
<style>body{{max-width:1100px;margin:auto;padding:30px;font:16px/1.7 system-ui;background:#f5f7fa;color:#182235}}h1{{line-height:1.3}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border-bottom:1px solid #ddd;padding:9px;text-align:right}}th:first-child{{text-align:left}}section{{margin:30px 0}}.note{{background:#fff4d7;padding:18px;border-radius:10px}}.scroll{{overflow:auto}}small{{overflow-wrap:anywhere}}</style>
<h1>１年前より古いデータで学習<br>直近１年を固定モデルで予想</h1>
<p>学習に使えるのは {inventory['training_cutoff_exclusive']} より前のレース。検証対象は {inventory['holdout_start_inclusive']} 〜 {str((pd.Timestamp(inventory['asof_exclusive'])-pd.Timedelta(days=1)).date())}。検証中の追加学習は０回です。</p>
<p>予想 {result['predicted_races']:,} レース／答え合わせ {result['evaluated_races']:,} レース。７部署を残し、社長・本番予想への組み込みは行っていません。</p>
<p class="note">以下は同じレース・同じ候補数で比べた<b>無価格の予想診断</b>です。本線・穴の買い目比較や回収率ではありません。当時の事前予想の実績にも加算しません。</p>
<section><h2>同じ検証レースでの比較</h2><div class="scroll"><table><tr><th>予想方式</th><th>１候補的中率</th><th>６候補的中率</th><th>12候補的中率</th><th>全体の予測損失↓</th><th>３着の予測損失↓</th></tr>{rows}</table></div><p>損失は小さいほど良好。３着の損失は正しい１・２着が与えられたときの評価です。</p></section>
<section><h2>月ごとの確認（12候補）</h2><table><tr><th>月</th><th>レース数</th><th>リスク部の評価式</th><th>統合式</th></tr>{months}</table></section>
<section><h2>古い学習期間の役割分担</h2><ol>{phases}</ol><p>すべて検証開始より前です。基礎予想、その出力から学ぶ組み合わせ式、各式の配分、確率の補正を時系列で分けています。基礎モデルと部署の知識は最初の期間で固定され、後の期間での出力を使って後段を学習しています。</p></section>
<section><h2>解釈上の制限</h2><ul>{limitations}</ul><p>回収率・100倍以上の穴・高配当的中への依存度：必要な時点のオッズ不足により未算出。</p></section>
<small>モデル確認値：{result['model_sha256']}<br>予想保存物の確認値：{result['predictions_sha256']}</small></html>'''
    (folder/'fixed_year_report.html').write_text(doc,encoding='utf-8')
