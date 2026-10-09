"""Prospective three-year equations and independently frozen 6000-yen plans."""
import argparse
import hashlib
import html
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


def forecast(folder=FOLDER):
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
                'producer_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                'input_sha256':hashlib.sha256(clean[clean.race_id.eq(rid)].to_json(orient='records').encode()).hexdigest()}
            frozen = common.freeze(records, 'three_year_421', meta, methods,
                {'three_year':{'training_cutoff_exclusive':manifest['training_cutoff_exclusive'],
                              'model_sha256':manifest['model_sha256']}}, common.clock())
            for row in coverage:
                if row['race_id']==rid:
                    row['reason'] = '保存済み' if frozen else '計算完了時点で締切後'
    market = pd.read_csv(ROOT/'data/raw/today_odds.csv', dtype={'race_id':str})
    by_id = {str(rid):r for rid,r in entries.groupby('race_id',sort=False)}
    for row in records:
        if row['race_id'] in by_id and row['race_id'] not in decided:
            freeze_allocations(saved,row,by_id[row['race_id']],market,common.clock())
    common.save(folder/'forecasts.json',records)
    common.save(folder/'allocations.json',saved)
    common.save(folder/'coverage.json',{'date':day,'races':coverage,'updated_at':common.clock().isoformat()})
    return report(folder)


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
    esc=lambda x:html.escape(str(x))
    pct=lambda p:'未集計' if p is None else f'{p:.1%}'
    lookup={(r['race_id'],r['points'],r['equation']):r for r in saved}
    visible_dates=set(sorted({r['date'] for r in records})[-2:])
    visible_records=[r for r in records if r['date'] in visible_dates]
    m=value['training']
    page=['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>3年学習・個別方程式予想｜KEIRIN NEXUS</title><style>body{margin:0;background:#0b1322;color:#e8edf7;font:16px/1.7 system-ui}main{max-width:1150px;margin:auto;padding:24px 16px}a{color:#9bd8ff}section{background:#15243a;border-radius:12px;padding:18px;margin:20px 0}button,select{font:inherit;padding:12px;border-radius:8px;border:1px solid #7197bb;background:#213653;color:white;margin:5px 5px 5px 0}button[aria-pressed=true]{background:#235e67;border-color:#8be4d0}select{max-width:100%;width:640px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #405068;vertical-align:top}td{min-width:85px}.picks{min-width:300px}.muted{color:#bbcadc}.warn{color:#f8d998}</style><main>',
        '<p><a href="../../index.html">今日の予想</a> ／ <a href="../individual_equations/index.html">導入前の独立式</a> ／ <a href="../fusion_shadow_live_report.html">既存統合式</a></p><h1>3年学習・個別方程式予想</h1>',
        '<p>各式が単独で三連単を選び、選択した1プランに6,000円を配分します。本線・穴とは別の仮想購入です。</p>',
        f'<p>学習：{esc(m.get("training_first","確認中"))}〜{esc(m.get("training_last","確認中"))} ／ {esc(m.get("training_races","—"))}レース。重みは直近1年：その前1年：さらに前1年＝<b>4：2：1</b>。</p>',
        '<p class="muted">買い目は各式の確率順位。金額は当たる確率と締切前オッズを使って100円単位で配分します。表示する回収見込みは予測値で、実際の回収率と区別しています。過去3年の締切前三連単オッズ・途中隊列が不足するため、市場残差式・市場条件付き先着式・途中隊列式は3年学習済みとは表示しません。</p>',
        '<label for="equation">個別の方程式</label><br><select id="equation" onchange="choose()">']
    page.extend(f'<option value="{esc(n)}"'+(' selected' if n=='first_anchor' else '')+f'>{esc(label)}</option>' for n,label in NAMES.items())
    page.append('</select><div id="commands">')
    page.extend(f'<button type="button" data-choice="{n}" aria-pressed="'+('true' if n==3 else 'false')+f'" onclick="points={n};choose()">{n}点方程式予想・6,000円</button>' for n in budget.POINTS)
    page.append('</div><p>3点・6点・12点は切替用です。合計18,000円の購入を意味しません。各プランは締切前に固定し、後から当たったプランへ変更しません。金額未配分のレースは6,000円プランの成績集計に含めません。</p><p>買い目一覧は直近2開催日の全レース、成績は保存開始からの累計です。過去の固定記録はページ末尾から確認できます。</p>')
    for name,label in NAMES.items():
        page.append(f'<section data-equation="{esc(name)}"'+(' hidden' if name!='first_anchor' else '')+f'><h2>{esc(label)}</h2>')
        for points in budget.POINTS:
            s=value['statistics'][name][str(points)]
            page.append(f'<div data-points="{points}"'+(' hidden' if points!=3 else '')+f'><h3>{points}点・6,000円プラン</h3><p>配分保存 {s["forecast_races"]}R ／ 的中 {s["hits"]} / 確定 {s["settled_races"]}R ／ 的中率 <b>{pct(s["hit_rate"])}</b> ／ 実績回収率 <b>{pct(s["roi"])}</b><br>仮想投資 {s["stake_yen"]:,}円 ／ 払戻し {s["return_yen"]:,.0f}円 ／ 払戻し未取得 {s["payout_pending_races"]}R<br>50倍超 {s["hits_over_50x"]}R ／ 100倍以上 {s["hits_at_least_100x"]}R</p><div class="scroll"><table><tr><th>開催・R</th><th>買い目・金額</th><th>回収見込み</th><th>公式結果</th><th>記録時刻</th></tr>')
            for row in sorted(visible_records,key=lambda r:(r['date'],r['close_at']),reverse=True):
                frozen=lookup.get((row['race_id'],points,name))
                plan=frozen['methods'][name] if frozen else None
                picks=plan['tickets'] if plan else row['methods'][name]['tickets'][:points]
                descriptions=[f'{t["buy"]}：{t["stake_yen"]:,}円（{t["odds_at_capture"]:g}倍）' if plan else t['buy'] for t in picks]
                if not plan:
                    descriptions.append('金額未配分：必要な締切前オッズ待ち' if row['close_at']>common.clock().timestamp() else '金額未配分：締切前オッズ不足')
                out=outcomes.get(row['race_id']); status='結果待ち'
                if out:
                    status='公式結果不一致・集計除外' if out.get('conflict') else ' ／ '.join(out['winning_buys'])
                    if not out.get('conflict'):
                        status+=' ／ '+('的中' if {t['buy'] for t in picks}&set(out['winning_buys']) else '不的中')
                        if plan and out['payout_complete']:
                            cash=sum(out['payouts'].get(t['buy'],0)*t['stake_yen']/100 for t in picks)
                            status+=f' ／ {cash:,.0f}円'
                expected=f'{plan["estimated_return_yen"]:,.0f}円（{plan["estimated_roi"]:.1%}）' if plan else 'オッズ未確定'
                stamp=frozen['allocated_at_jst'] if frozen else row['snapshot_at_jst']
                page.append(f'<tr><td>{esc(row["date"])}<br>{esc(row["venue"])} {row["race_no"]}R</td><td class="picks">'+ '<br>'.join(esc(t) for t in descriptions)+f'</td><td>{esc(expected)}</td><td>{esc(status)}</td><td>{esc(stamp)}</td></tr>')
            shared=value['common_comparison_statistics'][name][str(points)]
            page.append(f'</table></div><p>20式の共通条件だけで比較：的中 {shared["hits"]} / 確定 {shared["settled_races"]}R ／ 的中率 {pct(shared["hit_rate"])} ／ 回収率 {pct(shared["roi"])}。この式の最大払戻への依存：{pct(s["largest_return_share"])}。</p></div>')
        page.append('</section>')
    page.append('<section><h2>全レースの確認状況</h2><p>締切後の予想や配分は作りません。個別成績と、20式すべての配分が同じレース・同じオッズ時点で揃った共通比較を分けて表示します。</p><div class="scroll"><table>')
    for r in value['coverage'].get('races',[]):
        page.append(f'<tr><td>{esc(r["venue"])} {r["race_no"]}R</td><td>{esc(r["reason"])}</td></tr>')
    page.append('</table></div></section><p><a href="report.json">学習情報・集計</a> ／ <a href="forecasts.json">固定予想</a> ／ <a href="allocations.json">固定配分</a> ／ <a href="results.json">公式結果</a></p><script>let points=3;function choose(){const name=document.getElementById("equation").value;document.querySelectorAll("[data-equation]").forEach(e=>e.hidden=e.dataset.equation!==name);document.querySelectorAll("[data-points]").forEach(e=>e.hidden=Number(e.dataset.points)!==points);document.querySelectorAll("[data-choice]").forEach(e=>e.setAttribute("aria-pressed",String(Number(e.dataset.choice)===points)));}</script></main></html>')
    (folder/'index.html').write_text(''.join(page),encoding='utf-8')


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
    parser.add_argument('command',choices=('forecast','report','merge','list','command'))
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
        value=forecast() if args.command=='forecast' else merge(args.remote) if args.command=='merge' else report()
        print(json.dumps({k:value[k] for k in ('forecast_races','allocated_race_plans')},ensure_ascii=False))
