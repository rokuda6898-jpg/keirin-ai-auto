"""Display saved original shadow forecasts on the homepage; never rescore them."""
from bs4 import BeautifulSoup

SECTION = r'''
<section id="shadow-original-picks" style="margin:20px 0;padding:20px;background:#fff;border:2px solid #247be6;border-radius:18px">
<h2>38.77％案・独立予想</h2>
<p>過去一年の検証で12点的中率38.77％だった方式の運用予想。会社の会議予想とは別の買い目です。38.77％は今後の的中率ではありません。</p>
<label>開催日 <select data-shadow-day aria-label="38.77％案の開催日"></select></label>
<label>レース場 <select data-shadow-venue aria-label="38.77％案のレース場"></select></label>
<p data-shadow-status role="status">保存済みの予想を取得中です。</p>
<div data-shadow-races></div>
<a href="company/fusion_shadow_live_report.html">この式の成績・全レースの確認状況</a>
<script>
(()=>{
const box=document.getElementById('shadow-original-picks');
const day=box.querySelector('[data-shadow-day]'),venue=box.querySelector('[data-shadow-venue]'),status=box.querySelector('[data-shadow-status]'),list=box.querySelector('[data-shadow-races]');
const node=(tag,text)=>{const e=document.createElement(tag);e.textContent=text;return e;};
const today=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Tokyo'}).format(new Date());
const time=seconds=>new Intl.DateTimeFormat('ja-JP',{timeZone:'Asia/Tokyo',hour:'2-digit',minute:'2-digit'}).format(new Date(seconds*1000));
let rows=[],coverage=[];
function marks(r){
const scores=new Map();
r.top12.forEach((buy,i)=>buy.split('-').forEach((car,pos)=>{
const score=scores.get(car)||[0,0,0];
const p=Number(r.probabilities&&r.probabilities[i]);
score[pos]+=Number.isFinite(p)&&p>0?p:1/(i+1);scores.set(car,score);
}));
return [...scores].sort((a,b)=>b[1][0]-a[1][0]||b[1][1]-a[1][1]||b[1][2]-a[1][2]||Number(a[0])-Number(b[0])).slice(0,5).map(([car],i)=>['◎','○','▲','△','☆'][i]+car+'番').join('　');
}
function allRows(){const map=new Map(coverage.map(r=>[String(r.race_id),r]));rows.forEach(r=>map.set(String(r.race_id),r));return [...map.values()];}
function options(select,values){select.replaceChildren();values.forEach(([value,label])=>{const o=node('option',label);o.value=value;select.append(o);});}
function render(){
const shown=allRows().filter(r=>r.date===day.value&&(!venue.value||r.venue===venue.value)).sort((a,b)=>Number(a.close_at)-Number(b.close_at)||Number(a.race_no)-Number(b.race_no));
const saved=shown.filter(r=>Array.isArray(r.top12)&&r.top12.length);
list.replaceChildren();status.textContent=day.value+' ／ 対象 '+shown.length+'レース ／ 予想済み '+saved.length+' ／ 未作成 '+(shown.length-saved.length)+' ／ 各レース最大12点';
if(!shown.length){list.append(node('p','この条件の締切前予想はまだ保存されていません。'));return;}
const next=shown.find(r=>Number(r.close_at)*1000>Date.now());
shown.forEach(r=>{
const card=document.createElement('details');card.style.cssText='margin:12px 0;padding:12px;border:1px solid #dfe8f2;border-radius:12px';
card.open=r===next||(!next&&shown.length===1);
if(!Array.isArray(r.top12)||!r.top12.length){card.append(node('summary',r.venue+' '+r.race_no+'R ／ 予想未作成'));card.append(node('p',r.reason||'予想データを確認中です。'));list.append(card);return;}
const actual=typeof r.actual==='string'?r.actual:null;
const state=actual?(r.top12.includes(actual)?'的中':'不的中'):(Number(r.close_at)*1000>Date.now()?'締切前':'結果待ち');
card.append(node('summary',r.venue+' '+r.race_no+'R ／ 締切 '+time(Number(r.close_at))+' ／ '+state));
card.append(node('p',marks(r)));
card.append(node('small','この独立予想の保存済み買い目から付けた印（1着支持、同点時は2着・3着支持の順）。'));
const tickets=node('div','');tickets.style.cssText='display:flex;flex-wrap:wrap;gap:8px;margin:12px 0';
r.top12.forEach((buy,i)=>{const t=node('span',(i+1)+'. '+buy);t.style.cssText='padding:8px 12px;background:#edf5ff;border-radius:8px;font-weight:700';tickets.append(t);});card.append(tickets);
if(actual)card.append(node('p','公式結果 '+actual));
card.append(node('small','予想保存 '+r.snapshot_at_jst+' ／ 保存済み順位のまま表示。金額配分なし。'));
list.append(card);
});
}
function changeDay(){const venues=[...new Set(allRows().filter(r=>r.date===day.value).map(r=>r.venue))].sort();options(venue,[['','全レース場'],...venues.map(v=>[v,v])]);render();}
Promise.all(['company/fusion_shadow_live_ledger.jsonl','company/fusion_shadow_live_predictions.json'].map(url=>fetch(url+'?t='+Date.now(),{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('fetch');return r.text();}))).then(([text,latest])=>{
coverage=JSON.parse(latest).coverage.races;
rows=text.split(/\r?\n/).filter(s=>s.trim()).map(s=>JSON.parse(s));
if(rows.some(r=>!Array.isArray(r.top12)||!r.top12.every(t=>typeof t==='string'&&/^[1-9]-[1-9]-[1-9]$/.test(t))||!Number.isFinite(Number(r.close_at))))throw Error('invalid ledger');
const dates=[...new Set([today,...allRows().map(r=>r.date)])].sort().reverse();options(day,dates.map(d=>[d,d]));day.value=today;changeDay();
}).catch(()=>{status.textContent='保存済み予想を取得できません。再読み込みするか、この式の詳細ページを開いてください。';});
day.addEventListener('change',changeDay);venue.addEventListener('change',render);
})();
</script></section>
'''


def add_shadow_home(document):
    soup = BeautifulSoup(document, 'html.parser')
    if soup.main is None:
        return document
    old = soup.select_one('#shadow-original-picks')
    if old:
        old.decompose()
    section = BeautifulSoup(SECTION, 'html.parser').section
    hero = soup.main.select_one('.hero')
    if hero:
        hero.insert_after(section)
    else:
        soup.main.insert(0, section)
    return str(soup)
