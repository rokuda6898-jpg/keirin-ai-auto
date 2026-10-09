"""Read-only venue/time navigation over frozen independent predictions."""
import csv
import html
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

JST = ZoneInfo('Asia/Tokyo')


def race_times(row, schedule):
    source = schedule.get(row['race_id'], {})
    # Schedule supplements display metadata only; it never changes a forecast.
    if source.get('date') != row['date'] or source.get('venue') != row['venue']:
        source = {}
    def stamp(value):
        try:
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                return None
            dt = datetime.fromtimestamp(value, JST)
            return dt if dt.date().isoformat() == row['date'] else None
        except (ValueError, TypeError, OverflowError, OSError):
            return None
    start = stamp(row.get('start_at')) or stamp(source.get('start_at'))
    close = stamp(row.get('close_at'))
    band = ('朝' if start.hour < 12 else '昼' if start.hour < 17 else
            '夜' if start.hour < 21 else '深夜') if start else '時刻未確認'
    return {'start_label':start.strftime('%H:%M') if start else '未確認',
            'close_label':close.strftime('%H:%M') if close else '未確認', 'band':band}


def document(records, saved, outcomes, value, names, marks, schedule_path):
    schedule = {}
    if schedule_path.exists():
        with schedule_path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                schedule[str(row['race_id'])] = row
    dates = sorted({r['date'] for r in records}, reverse=True)[:2]
    races = []
    for row in sorted(records, key=lambda r:r['close_at']):
        if row['date'] not in dates:
            continue
        races.append({**{k:row[k] for k in ('race_id','date','venue','race_no','close_at','snapshot_at_jst')},
            **race_times(row, schedule), 'methods':row['methods'],
            'marks':{name:marks(method) for name,method in row['methods'].items()},
            'outcome':outcomes.get(row['race_id'])})
    plans = {}
    for row in saved:
        plans.setdefault(row['race_id'],{}).setdefault(row['equation'],{})[str(row['points'])] = {
            **row['methods'][row['equation']], 'saved_at':row['allocated_at_jst']}
    payload = {'races':races,'plans':plans,'names':names,'dates':dates,
               'stats':value['statistics'],'shared':value['common_comparison_statistics']}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    m = value['training']; esc = lambda v:html.escape(str(v))
    training = f'{esc(m.get("training_first","未確認"))}〜{esc(m.get("training_last","未確認"))}・{esc(m.get("training_races","—"))}レース。直近1年から重み4：2：1。'
    coverage = ''.join(f'<li>{esc(r["venue"])} {r["race_no"]}R：{esc(r["reason"])}</li>' for r in value['coverage'].get('races',[]))
    return HTML.replace('@@STYLE@@', CSS).replace('@@TRAINING@@',training).replace('@@COVERAGE@@',coverage).replace('@@UPDATED@@',esc(value['updated_at_jst'])).replace('@@DATA@@',encoded).replace('@@SCRIPT@@',JS)


HTML = '''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>競輪場別・個別方程式予想｜KEIRIN NEXUS</title><style>@@STYLE@@</style></head><body><main>
<nav class="top"><a href="../../index.html">← トップ・今日の予想</a><a href="#performance">成績を見る</a></nav>
<header><span class="eyebrow">KEIRIN NEXUS / 個別方程式</span><h1>競輪場から予想を探す</h1><p>予想印と三連単。選んだ1プランに<strong>6,000円</strong>を配分。</p></header>
<section class="controls" aria-label="予想を選ぶ"><div class="selectors"><label>開催日<select id="day"></select></label><label>個別の方程式<select id="equation"></select></label></div>
<div id="plans" class="pills" aria-label="買い目点数"><button data-points="3" aria-pressed="true">3点方程式予想</button><button data-points="6" aria-pressed="false">6点方程式予想</button><button data-points="12" aria-pressed="false">12点方程式予想</button></div><p class="muted compact">各プラン6,000円。3・6・12点は切替用です。</p>
<h2 class="label">レース場</h2><div id="venues" class="pills" aria-label="レース場"></div>
<h2 class="label">発走時間帯 <small>日本時間</small></h2><div id="times" class="pills" aria-label="発走時間帯"><button data-band="all" aria-pressed="true">すべて</button><button data-band="朝" aria-pressed="false">朝 <small>12時前</small></button><button data-band="昼" aria-pressed="false">昼 <small>12〜17時</small></button><button data-band="夜" aria-pressed="false">夜 <small>17〜21時</small></button><button data-band="深夜" aria-pressed="false">深夜 <small>21時以降</small></button><button data-band="時刻未確認" aria-pressed="false">時刻未確認</button></div>
<label class="check"><input id="upcoming" type="checkbox">締切前だけ表示</label></section>
<div class="list-heading"><h2 id="selection-title">レース一覧</h2><span id="count" role="status" aria-live="polite"></span></div><p class="muted compact">レースを押すと買い目が開きます。各場の次のレースを先頭に表示します。</p>
<div id="races"></div><noscript><p>レースの切替にはJavaScriptが必要です。<a href="forecasts.json">固定予想の記録を見る</a></p></noscript>
<section id="performance" class="panel"><h2>この式の成績</h2><p class="muted">全レース場・保存開始からの累計。上の表示条件による絞り込みは含みません。</p><div id="metrics"></div></section>
<details class="panel"><summary>予想印・金額配分・学習について</summary><p>◎本命・○対抗・▲単穴・△連下・☆注目。各式の保存済み上位12買い目から1着支持を合算した順です。同点は2着、3着の支持順。印は3・6・12点で共通です。</p><p>学習：@@TRAINING@@</p><p>各式が独立して予想します。本線・穴とは別の仮想購入です。金額は予測確率と締切前オッズを使って100円単位で配分します。「回収見込み」は予測値で、実績回収率とは別です。</p><p>締切まで60分以内は更新時にオッズを取得。価格不足の場合は未配分とし、6,000円プランの成績には含めません。過去の予想・配分は書き換えません。</p><p>過去の締切前三連単オッズ・途中隊列が不足する市場残差式・市場条件付き先着式・途中隊列式は、3年学習済みには含めません。</p></details>
<details class="panel"><summary>全レースの確認状況</summary><p>初回確認が締切後のレースなどは後から予想を作りません。</p><ul>@@COVERAGE@@</ul></details>
<footer><p>更新：@@UPDATED@@ ／ 買い目は直近2開催日を表示</p><a href="../individual_equations/index.html">導入前の独立式</a> ／ <a href="../fusion_shadow_live_report.html">既存統合式</a><p><a href="report.json">集計</a> ／ <a href="forecasts.json">固定予想</a> ／ <a href="allocations.json">固定配分</a> ／ <a href="results.json">公式結果</a></p></footer>
<script id="race-data" type="application/json">@@DATA@@</script><script>@@SCRIPT@@</script></main></body></html>'''

CSS = '''*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#18324f;font:16px/1.65 system-ui,sans-serif}main{max-width:1060px;margin:auto;padding:20px 18px 50px}a{color:#1764bf;text-underline-offset:3px}.top{display:flex;justify-content:space-between;gap:16px;font-size:14px}header{padding:26px 0 14px}.eyebrow{color:#1764bf;font-size:12px;letter-spacing:.08em;font-weight:800}h1{font-size:clamp(24px,5vw,34px);margin:8px 0}h2{font-size:21px;margin:0 0 10px}h3{margin:0;font-size:19px}p{margin:10px 0}.muted,small,footer{color:#526880}.compact{font-size:13px}.controls,.panel{padding:20px;background:white;border:1px solid #dbe5f0;border-radius:18px;margin:14px 0}.selectors{display:grid;grid-template-columns:180px 1fr;gap:14px}label,.label{font-size:14px;font-weight:700}.label{margin:18px 0 8px}select{display:block;width:100%;margin-top:5px;min-height:46px;padding:10px;border:1px solid #9caec1;border-radius:10px;background:white;color:#18324f;font:inherit}button{cursor:pointer;font:inherit;min-height:44px;border:1px solid #c5d5e7;background:#f4f8fe;color:#204266;border-radius:10px;padding:10px 15px}button[aria-pressed=true]{background:#1764bf;color:white;border-color:#1764bf}button[aria-pressed=true] small{color:white}button small{display:block;font-size:11px}.pills{display:flex;gap:8px;flex-wrap:wrap}#plans{margin-top:16px}#plans button{font-weight:700}.check{display:flex;align-items:center;gap:8px;margin-top:18px}.check input{width:20px;height:20px;accent-color:#1764bf}.list-heading{display:flex;align-items:center;justify-content:space-between;margin-top:26px;gap:10px}#count{font-size:13px;color:#526880}.venue{margin:24px 0 30px}.venue>h2{border-left:5px solid #1764bf;padding-left:12px}.time-title{font-size:14px;margin:18px 0 8px;color:#526880}.race-card{background:#fff;border:1px solid #d1dfed;border-radius:14px;margin:10px 0;overflow:hidden}.race-card[open]{border-color:#6d9dd4;box-shadow:0 3px 14px #234d7410}summary{cursor:pointer;min-height:48px}.race-card>summary{padding:15px;display:flex;justify-content:space-between;gap:12px;align-items:center;list-style:none}.race-card>summary::-webkit-details-marker{display:none}.race-card>summary:after{content:'＋';font-size:22px;color:#1764bf}.race-card[open]>summary:after{content:'−'}.race-heading{flex:1}.race-heading b{font-size:20px}.race-heading small{display:block}.badge{border-radius:20px;padding:4px 10px;background:#edf2f7;color:#415970;font-size:12px;white-space:nowrap}.badge.open{background:#e7f4ed;color:#17613c}.badge.hit{background:#fff1cb;color:#775000}.race-body{border-top:1px solid #dbe5f0;padding:16px}.marks{display:flex;flex-wrap:wrap;gap:9px;margin:0 0 16px;font-weight:800}.mark{padding:6px 10px;border-radius:8px;background:#eaf3ff;color:#174c88}.tickets{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.ticket{border:1px solid #dce5ef;padding:12px;border-radius:10px;display:flex;align-items:center;justify-content:space-between;gap:8px}.ticket>div{display:flex;gap:5px;align-items:center}.car{display:inline-grid;place-items:center;width:32px;height:36px;font-weight:800;border-radius:6px;border:1px solid #acb8c5;background:white;color:#172b42}.car.n2{background:#242a32;color:white}.car.n3{background:#c9283c;color:white}.car.n4{background:#2163bf;color:white}.car.n5{background:#f2d538;color:#172b42}.car.n6{background:#1c764c;color:white}.car.n7{background:#f58b30;color:#172b42}.car.n8{background:#f29bc5;color:#172b42}.car.n9{background:#754ba9;color:white}.ticket-amount{text-align:right;white-space:nowrap;font-weight:700}.ticket-amount small{display:block;font-size:11px;font-weight:400}.money-line{background:#f1f6fd;padding:12px;border-radius:9px;margin:12px 0;font-size:14px}.warn{color:#805700;background:#fff8e5;padding:10px;border-radius:9px;font-size:13px}.result{font-weight:700;margin-top:14px}.stamp{font-size:12px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.stats div{padding:12px;background:#f1f6fd;border-radius:10px}.stats small{display:block;font-size:12px}.stats b{font-size:23px}footer{font-size:12px;margin-top:25px}.empty{background:white;padding:30px;border:1px dashed #afc1d5;border-radius:14px;text-align:center}:focus-visible{outline:3px solid #e89412;outline-offset:3px}[hidden]{display:none!important}@media(max-width:600px){main{padding:16px 12px 40px}.controls,.panel{padding:15px}.selectors{grid-template-columns:1fr}.tickets{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}#plans{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}#plans button{padding:10px 5px;font-size:12px}.race-heading b{font-size:18px}.race-card>summary{padding:12px;gap:8px}.pills button{padding:9px 12px}.race-body{padding:12px}}'''

JS = r'''
const D=JSON.parse(document.getElementById('race-data').textContent),$=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const yen=v=>Number(v).toLocaleString('ja-JP',{maximumFractionDigits:0})+'円',pct=v=>v==null?'未集計':(v*100).toFixed(1)+'%';
const stamp=v=>v?new Date(v).toLocaleString('ja-JP',{timeZone:'Asia/Tokyo',hour12:false}):'未確認';
let points=3,venue='all',band='all';
$('day').innerHTML=D.dates.map(d=>`<option>${esc(d)}</option>`).join('');
$('equation').innerHTML=Object.entries(D.names).map(([n,l])=>`<option value="${esc(n)}" ${n==='first_anchor'?'selected':''}>${esc(l)}</option>`).join('');
function dayRows(){return D.races.filter(r=>r.date===$('day').value)}
function venues(){const vs=[...new Set(dayRows().map(r=>r.venue))];if(!vs.includes(venue))venue='all';$('venues').innerHTML=[['all','全レース場'],...vs.map(v=>[v,v])].map(([v,l])=>`<button data-venue="${esc(v)}" aria-pressed="${venue===v}">${esc(l)}${v==='all'?'':` <small>${dayRows().filter(r=>r.venue===v).length}レース</small>`}</button>`).join('')}
function ticket(t,allocated){return `<div class="ticket"><div aria-label="買い目 ${esc(t.buy)}">${t.buy.split('-').map(n=>`<span class="car n${esc(n)}">${esc(n)}</span>`).join('<span>−</span>')}</div><span class="ticket-amount">${allocated?yen(t.stake_yen):'配分待ち'}${allocated?`<small>${Number(t.odds_at_capture).toLocaleString('ja-JP')}倍</small>`:''}</span></div>`}
function card(r,name,next){const plan=D.plans[r.race_id]?.[name]?.[String(points)],picks=plan?.tickets||r.methods[name].tickets.slice(0,points),out=r.outcome,closed=r.close_at<=Date.now()/1000,hit=out&&!out.conflict&&picks.some(t=>out.winning_buys.includes(t.buy));
const state=out?(out.conflict?'結果確認中':hit?'的中':'不的中'):closed?'締切済み':'締切前';
let result=out?(out.conflict?'公式結果不一致・集計除外':`結果 ${out.winning_buys.map(esc).join(' ／ ')} ・ ${hit?'的中':'不的中'}${!plan?'（金額未配分）':''}`):'公式結果待ち';
if(plan&&out&&!out.conflict){result+=out.payout_complete?' ／ 払戻し '+yen(picks.reduce((s,t)=>s+(out.payouts[t.buy]||0)*t.stake_yen/100,0)):' ／ 払戻し未取得'}
return `<details class="race-card" id="race-${esc(r.race_id)}" ${r.race_id===next?'open':''}><summary><div class="race-heading"><b>${esc(r.venue)} ${r.race_no}R</b><small>発走 ${esc(r.start_label)} ／ 締切 ${esc(r.close_label)}</small></div><span class="badge ${hit?'hit':!closed?'open':''}">${state}</span></summary><div class="race-body"><div class="marks" aria-label="予想印">${r.marks[name].map(m=>`<span class="mark">${m.mark}${m.car_no}番</span>`).join('')}</div><div class="tickets">${picks.map(t=>ticket(t,!!plan)).join('')}</div>${plan?`<div class="money-line"><strong>合計 ${yen(plan.spent_yen)}</strong> ／ ${points}点<br>回収見込み ${yen(plan.estimated_return_yen)}（${pct(plan.estimated_roi)}・予測値）</div>`:`<p class="warn">${closed?'金額未配分：締切前オッズ不足':'金額配分待ち：必要な締切前オッズを確認中'}</p>`}<p class="result">${result}</p><p class="stamp muted">予想保存 ${stamp(r.snapshot_at_jst)}${plan?'<br>配分保存 '+stamp(plan.saved_at):''}</p></div></details>`}
function render(){const name=$('equation').value,now=Date.now()/1000;document.querySelectorAll('[data-points]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.points)===points)));document.querySelectorAll('[data-band]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.band===band)));document.querySelectorAll('[data-venue]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.venue===venue)));
const rows=dayRows().filter(r=>(venue==='all'||r.venue===venue)&&(band==='all'||r.band===band)&&(!$('upcoming').checked||r.close_at>now));
$('selection-title').textContent=(venue==='all'?'レース場別の予想':venue+'の予想');$('count').textContent=rows.length+'レース ／ '+points+'点';
const vs=[...new Set(rows.map(r=>r.venue))];vs.sort((a,b)=>{const next=v=>Math.min(...rows.filter(r=>r.venue===v&&r.close_at>now).map(r=>r.close_at),Infinity);return next(a)-next(b)||a.localeCompare(b,'ja')});
$('races').innerHTML=vs.length?vs.map(v=>{const list=rows.filter(r=>r.venue===v).sort((a,b)=>(a.close_at<=now)-(b.close_at<=now)||a.close_at-b.close_at),next=list.find(r=>r.close_at>now)?.race_id;const bands=[...new Set(list.map(r=>r.band))];return `<section class="venue"><h2>${esc(v)} <small>${list.length}レース</small></h2>${bands.map(t=>`<h3 class="time-title">${esc(t)}のレース</h3>${list.filter(r=>r.band===t).map(r=>card(r,name,next)).join('')}`).join('')}</section>`}).join(''):'<div class="empty">条件に合うレースがありません。<br><button id="reset">絞り込みを解除</button></div>';
if($('reset'))$('reset').onclick=()=>{venue='all';band='all';$('upcoming').checked=false;render()};
const s=D.stats[name][String(points)],c=D.shared[name][String(points)];$('metrics').innerHTML=`<h3>${esc(D.names[name])}・${points}点</h3><div class="stats"><div><small>的中率</small><b>${pct(s.hit_rate)}</b></div><div><small>実績回収率</small><b>${pct(s.roi)}</b></div><div><small>的中 / 確定</small><b>${s.hits} / ${s.settled_races}R</b></div><div><small>配分保存</small><b>${s.forecast_races}R</b></div></div><p>確定分の仮想投資 ${yen(s.stake_yen)} ／ 払戻し ${yen(s.return_yen)}<br>払戻し未取得 ${s.payout_pending_races}R ／ 50倍超 ${s.hits_over_50x}R ／ 100倍以上 ${s.hits_at_least_100x}R</p><details><summary>同じ条件での比較・高配当への依存</summary><p>20式共通：的中 ${c.hits} / 確定 ${c.settled_races}R ／ 的中率 ${pct(c.hit_rate)} ／ 回収率 ${pct(c.roi)}<br>最大払戻への依存 ${pct(s.largest_return_share)}</p></details>`}
$('day').onchange=()=>{venues();render()};$('equation').onchange=render;$('upcoming').onchange=render;
$('plans').onclick=e=>{const b=e.target.closest('[data-points]');if(b){points=Number(b.dataset.points);render()}};
$('venues').onclick=e=>{const b=e.target.closest('[data-venue]');if(b){venue=b.dataset.venue;render()}};
$('times').onclick=e=>{const b=e.target.closest('[data-band]');if(b){band=b.dataset.band;render()}};
venues();render();
'''
