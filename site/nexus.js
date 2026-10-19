'use strict';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)], C=NexusCore;
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const yen=x=>x==null?'—':Math.round(x).toLocaleString('ja-JP')+'円',pct=x=>x==null?'—':(x*100).toFixed(2)+'％';
const cats={FAVORITE:'本命',MAIN:'本線',MEDIUM:'中穴',HIGH:'大穴'},sessions={morning:'モーニング',day:'デイ',night:'ナイト',midnight:'ミッドナイト'};
let data=null,tab='forecast',session='all',budget='auto',loading=false;
const car=n=>`<span class="car c${Number(n)}">${Number(n)}</span>`;
const clock=t=>t?new Date(t).toLocaleTimeString('ja-JP',{timeZone:'Asia/Tokyo',hour:'2-digit',minute:'2-digit'}):'—';
const stamp=t=>t?new Date(t).toLocaleString('ja-JP',{timeZone:'Asia/Tokyo'}):'未保存';
const moneyTitle=()=>budget==='auto'?'5,000〜10,000円 自動配分':yen(Number(budget))+' 配分';
const profit=x=>`${x>=0?'+':''}${yen(x)}`;
function effectivePrediction(r){return r.prediction&&(C.closed(r)||C.openPodium(r.prediction))?r.prediction:null;}
function formHTML(blocks){return `<div class="formations"><div class="form-head"><span>組</span><span>1着</span><span>2着</span><span>3着</span></div>${blocks.map((b,i)=>`<div class="formation"><small>${i+1}</small>${['first','second','third'].map(k=>`<div class="car-set">${b[k].map(car).join('')}</div>`).join('')}</div>`).join('')}</div>`;}
function outcomeHTML(r,p){
  const flat=r.settlement?.flat, known=r.result_complete||r.cancelled||flat||r.winners?.length;
  if(!known)return `<div class="result-bar"><span>結果待ち</span><span>公式結果の取得後に反映</span></div>`;
  const voided=r.refund_all||r.cancelled||flat?.void;
  const label=voided?'中止・返還':flat?flat.hit?'的中':'不的中':'公式結果';
  return `<div class="result-bar ${flat?.hit?'hit':''}"><b>${label}</b><div class="winning-sets">${(r.winners||[]).map(t=>`<span class="winning-ticket">${t.split('-').map(car).join('<i>→</i>')}<small>${yen(r.payouts?.[t])}</small></span>`).join('')}</div></div>${!p&&!voided?'<p class="result-excluded">締切前予想なし・的中率と回収率には含めません</p>':''}`;
}
function renderStats(rows){
  const clean=rows.map(r=>effectivePrediction(r)?r:{...r,prediction:null});
  $('#stats').innerHTML=['flat',budget].map((mode,i)=>{
    const a=C.totals(clean,mode);
    return `<div class="stat-card"><div class="stat-label">${i?moneyTitle():'1点100円'} · 確定成績</div><div class="stat-row"><div class="metric"><b>${pct(a.hitRate)}</b><span>的中率 · ${a.hits}/${a.settled}レース</span></div><div class="metric"><b>${pct(a.roi)}</b><span>回収率</span></div></div><div class="stat-profit ${a.profit>=0?'positive':'negative'}">収支 ${profit(a.profit)}</div><p>購入 ${yen(a.stake)} ／ 払戻 ${yen(a.returned)}${a.refund?' ／ 返還 '+yen(a.refund):''}<br>結果待ち ${a.pending} · 配分なし ${a.unavailable} · 予想なし ${a.unpredicted} · 全返還 ${a.void}</p></div>`;
  }).join('');
}
function allocationNote(p,a){
  if(a)return `<p class="ok">最低払戻見込み <b>${yen(a.minimum_return)}</b><br>的中時の最低収支 ${profit(a.minimum_return-a.budget)}</p>`;
  const raw=budget==='auto'?null:p.allocations?.[budget], records=Object.values(p.allocations||{});
  const complete=records.length>=51 || p.allocation_policy==='break-even-v1';
  const reason=raw?.reason||(raw?.quote_break_even===false?'この予算ではトリガミ回避不可':null);
  if(reason)return `<p class="warning">${esc(reason)}</p>`;
  if(records.length&&complete)return '<p class="warning">5,000〜10,000円で全点のトリガミ回避はできません。配分購入は見送り。</p>';
  return '<p class="warning">締切前の配分更新待ち</p>';
}
function moneyHTML(r,p){
  const a=C.allocation(p,budget), flat=r.settlement?.flat, weighted=r.settlement?.[budget];
  const line=x=>x?`<p class="${x.returned>=x.stake?'positive':'negative'}">払戻 ${yen(x.returned)}<br>収支 ${profit(x.returned-x.stake)}${x.refund?'（返還 '+yen(x.refund)+'）':''}</p>`:'';
  return `<div class="money"><div><small>1点100円 · 全${p.count}点</small><strong>${yen(p.count*100)}</strong>${line(flat)}</div><div><small>${budget==='auto'?'自動配分 · 成立する最小予算':'トリガミ回避配分'}</small><strong>${a?yen(a.budget):weighted?yen(weighted.stake):'配分なし'}</strong>${weighted?line(weighted):allocationNote(p,a)}</div></div>`;
}
function forecastHTML(r,p){
  const riders=(r.riders||[]).slice().sort((a,b)=>a.car-b.car), a=C.allocation(p,budget);
  const lineup=(r.lines||[]).map(line=>line.map(slot=>slot.join('・')).join('－')).join(' ／ ');
  const groups=a?C.amountGroups(a.stakes):[];
  const source=/^https:\/\/www\.winticket\.jp\//.test(r.source_url||'')?`<a class="source" href="${esc(r.source_url)}" target="_blank" rel="noopener noreferrer">出走表・最新オッズ ↗</a>`:'';
  const legacy=C.closed(r)&&!C.openPodium(p)?'<p class="warning">2着候補の方針変更前に保存された予想記録です。</p>':'';
  return `<div class="route-line"><span class="pill ${p.category==='HIGH'?'high':p.category==='MEDIUM'?'medium':''}">${esc(cats[p.category]||p.category)}</span><span class="count">3連単 ${p.count}点 · ${p.blocks.length}フォーメーション</span></div><div class="riders">${riders.map(x=>`<span class="rider"><span class="mark ${p.marks?.[x.car]?'':'unmarked'}">${esc(p.marks?.[x.car]||'・')}</span>${car(x.car)}${esc(x.name)}</span>`).join('')}</div><p class="lineup">予想隊列：${esc(lineup||'未確認')}</p>${legacy}${formHTML(p.blocks)}<p class="form-note">各組の全組み合わせ · 同一車と重複買い目を除き${p.count}点</p>${moneyHTML(r,p)}${groups.length?`<details class="amount-details"><summary>配分をフォーメーションで見る <span>${groups.length}種類の金額</span></summary><p class="form-note">同じ金額の買い目をまとめました。下の組をそれぞれ各指定金額で追加します。</p>${groups.map(g=>`<div class="amount-group"><h4>各${yen(g.amount)} <span>${g.count}点 · 計${yen(g.amount*g.count)}</span></h4>${formHTML(g.blocks)}</div>`).join('')}</details>`:''}<details><summary>取得オッズと保存時刻</summary><div class="quote-list">${p.blocks.map((b,i)=>`<div><b>組${i+1}</b>${C.expand([b]).map(t=>`<span>${esc(t)} <em>${p.odds?.[t]!=null?Number(p.odds[t]).toFixed(1)+'倍':'未取得'}</em></span>`).join('')}</div>`).join('')}</div><p class="form-note">オッズ取得 ${esc(stamp(p.odds_at))}<br>予想保存 ${esc(stamp(p.captured_at))}${p.allocation_saved_at?'<br>配分保存 '+esc(stamp(p.allocation_saved_at)):''}</p></details><div class="actions"><button class="copy" data-copy="${esc(r.race_id)}">フォーメーションをコピー</button>${a?`<button class="copy" data-copy="${esc(r.race_id)}" data-amounts="true">配分もコピー</button>`:''}${source}</div>`;
}
function raceHTML(r){
  const p=effectivePrediction(r),ss=r.settlement,done=r.result_complete||r.winners?.length||ss,status=r.cancelled?'中止・返還':done?'結果確定':C.closed(r)?'締切済み':'受付中';
  const head=`<div class="race-head"><div><div class="venue-title">${esc(r.venue)}<span class="race-number">${Number(r.race_no)}R</span></div><div class="race-meta">${esc(r.class_name)} ${esc(r.stage)} · ${esc(sessions[r.session]||'時間帯未確認')}</div></div><div class="time"><b>${clock(r.start_at)}</b><span>${status} · 締切 ${clock(r.close_at)}</span></div></div>`;
  if(!p){const message=r.prediction&&!C.openPodium(r.prediction)?'2着に複数候補を残す予想へ更新待ちです。':r.message||'予想の準備中です。';return `<article class="race">${head}${C.closed(r)||done?outcomeHTML(r,null):''}<div class="race-body"><p class="muted">${esc(message)}</p></div></article>`;}
  const body=forecastHTML(r,p);
  return `<article class="race" id="race-${esc(r.race_id)}">${head}${tab==='results'?outcomeHTML(r,p):''}<div class="race-body">${tab==='results'?`${moneyHTML(r,p)}<details><summary>予想印・フォーメーションを確認</summary>${body}</details>`:body}</div></article>`;
}
function modelHTML(){
  const h=data.model?.historical||{};
  return `<div class="research"><div class="eyebrow">採用モデル</div><h2>隊列・役割フォーメーション</h2><p>保存済みの1〜3着モデルで全着順を評価します。1着はモデル評価に応じて固定・複数候補を使い分け、2着は複数候補を残します。選んだ全買い目を、最大4本の完全なフォーメーションで表示します。</p><h3>採用元モデルの過去検証</h3><div class="stat-row"><div class="metric"><b>${pct(h.hit_rate)}</b><span>的中率</span></div><div class="metric"><b>${pct(h.roi)}</b><span>各100円の回収率</span></div><div class="metric"><b>${Number(h.races||0).toLocaleString()}</b><span>レース</span></div></div><p>学習：2023/10/9〜2025/10/8<br>検証：2025/10/9〜2026/10/8（反復研究した期間）</p><p>共有された直近の同条件8,205レース比較では、採用元53.69％・76.56％、全着順の再学習案53.31％・75.24％でした。両指標が下がった再学習案への置き換えは行っていません。</p><ul><li>本命6点・本線15点・中穴15点・大穴24点。</li><li>開催場、隊列と役割、強さ、戦法履歴、時間帯、14・30・90日の調子を評価。</li><li>◎○▲△☆は選手評価です。印だけで1着・2着へ固定しません。</li></ul><h3>今回の指定を反映した配分</h3><p>全点100円の購入方法と、5,000〜10,000円を100円刻みで計算した配分を併記。自動配分は、取得オッズで全点が購入額以上になる最小予算を選び、その予算内で最低払戻見込みを高めます。成立しなければ配分なしと表示します。</p><p>上の過去成績は、今回追加した2着候補の条件・配分見送りを含む新仕様の実績ではありません。公開後の確定成績は「結果・成績」で別に集計します。短期141レースの143.06％を年間成績として扱いません。</p><details><summary>結果と回収率の計算方法</summary><p>結果は公式の確定払戻で計算し、同着はすべての的中券を合算します。全額返還は的中率の対象外。部分返還は購入額から返還額を差し引いた金額を回収率の分母にします。予想がないレースは公式結果のみ表示します。</p><p>金額配分の記録は締切前に保存したものだけを集計します。オッズが下がった場合や同着の場合は、結果的にトリガミになることがあります。配分版は見送りがあるため、100円均等版と対象レース数が異なる場合があります。</p></details></div>`;
}
function render(){
  if(!data)return;
  $('#race-view').classList.toggle('hidden',tab==='model');$('#model-view').classList.toggle('hidden',tab!=='model');
  const day=$('#date').value,venue=$('#venue').value;
  const dayRows=(data.races||[]).filter(r=>r.date===day);
  $$('[data-session]').forEach(b=>{const n=dayRows.filter(r=>(b.dataset.session==='all'||r.session===b.dataset.session)&&(tab==='results'?C.closed(r)||r.settlement:day!==data.date||!C.closed(r)&&!r.cancelled)).length;b.innerHTML=`${esc(b.dataset.session==='all'?'すべて':sessions[b.dataset.session])}<span class="session-count">${n}</span>`;});
  let rows=dayRows.filter(r=>(session==='all'||r.session===session)&&(!venue||r.venue===venue));renderStats(rows);
  if(tab==='results')rows=rows.filter(r=>C.closed(r)||r.settlement||r.cancelled);
  else if(day===data.date)rows=rows.filter(r=>!C.closed(r)&&!r.cancelled);
  rows.sort((a,b)=>(a.start_at||'').localeCompare(b.start_at||'')||a.race_id.localeCompare(b.race_id));
  const groups=session==='all'?[...Object.keys(sessions),'unknown'].map(k=>[k,rows.filter(r=>(r.session||'unknown')===k)]).filter(([,rr])=>rr.length):[[session,rows]];
  $('#races').innerHTML=rows.length?groups.map(([k,rr])=>`${session==='all'?`<h3 class="session-heading">${esc(sessions[k]||'時間帯未確認')} <small>${rr.length}レース</small></h3>`:''}${rr.map(raceHTML).join('')}`).join(''):'<div class="empty">この条件のレースはありません。</div>';
  $('#list-title').textContent=tab==='results'?'結果一覧':'これからのレース';$('#race-count').textContent=rows.length+'レース';
  const stale=Date.now()-Date.parse(data.updated_at)>30*60000;
  const note=[data.notice,stale?'取得元の更新から30分以上経過しています。最新オッズ・欠場・結果をご確認ください。':''].filter(Boolean).join(' ');$('#status-note').classList.toggle('hidden',!note);$('#status-note').textContent=note;
  $('#model-view').innerHTML=modelHTML();
  $('#prev-day').disabled=$('#date').selectedIndex<=0;$('#next-day').disabled=$('#date').selectedIndex>=$('#date').options.length-1;
}
function toast(t){$('#toast').textContent=t;$('#toast').classList.remove('hidden');setTimeout(()=>$('#toast').classList.add('hidden'),2500);}
$('#races').addEventListener('click',async e=>{
  const button=e.target.closest('[data-copy]');if(!button)return;const r=data.races.find(x=>x.race_id===button.dataset.copy),p=effectivePrediction(r);if(!p)return;
  let text=`${r.date} ${r.venue} ${r.race_no}R 3連単 ${p.count}点\n`+p.blocks.map(x=>`${x.first.join('・')} → ${x.second.join('・')} → ${x.third.join('・')}`).join('\n')+`\n各100円・計${yen(p.count*100)}`;
  if(button.dataset.amounts){const a=C.allocation(p,budget);if(!a)return;text+=`\n\n金額配分・計${yen(a.budget)}\n`+C.amountGroups(a.stakes).map(g=>`各${yen(g.amount)}（${g.count}点）\n`+g.blocks.map(x=>`${x.first.join('・')} → ${x.second.join('・')} → ${x.third.join('・')}`).join('\n')).join('\n')+`\nオッズ取得 ${stamp(p.odds_at)}\n最低払戻見込み ${yen(a.minimum_return)}`;}
  try{await navigator.clipboard.writeText(text);toast('フォーメーションをコピーしました');}catch{toast('コピーできませんでした。表示を選択してコピーしてください。');}
});
$$('[data-tab]').forEach(b=>b.addEventListener('click',()=>{tab=b.dataset.tab;$$('[data-tab]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});render();}));
$$('[data-session]').forEach(b=>b.addEventListener('click',()=>{session=b.dataset.session;$$('[data-session]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});render();}));
$('#budget').addEventListener('change',()=>{budget=$('#budget').value;render();});$('#date').addEventListener('change',render);$('#venue').addEventListener('change',render);
for(const [id,direction]of[['prev-day',-1],['next-day',1]])$('#'+id).addEventListener('click',()=>{const s=$('#date');s.selectedIndex=Math.max(0,Math.min(s.options.length-1,s.selectedIndex+direction));render();});
async function load(quiet=false){
  if(loading)return;loading=true;$('#refresh').disabled=true;
  const oldDate=$('#date').value,oldVenue=$('#venue').value;let fresh=null;
  for(const url of ['https://raw.githubusercontent.com/rokuda6898-jpg/keirin-ai-auto/main/outputs/nexus_live.json','original/nexus_live.json','../outputs/nexus_live.json']){
    try{const response=await fetch(url+'?t='+Math.floor(Date.now()/60000),{cache:'no-store',signal:AbortSignal.timeout(12000)});if(!response.ok)continue;const candidate=await response.json();if(candidate.schema===1&&Array.isArray(candidate.races)){fresh=candidate;break;}}catch{}
  }
  loading=false;$('#refresh').disabled=false;$('#loading').classList.add('hidden');
  if(!fresh){if(data){render();if(!quiet)toast('通信できませんでした。保存済みの表示を維持しています。');return;}$('#races').innerHTML='<div class="empty">予想データを読み込めませんでした。更新ボタンで再読み込みしてください。</div>';$('#updated').textContent='通信を確認してください';return;}
  data=fresh;const dates=[...new Set([...data.races.map(r=>r.date),data.date].filter(Boolean))].sort();$('#date').innerHTML=dates.map(d=>`<option value="${esc(d)}">${esc(d.replaceAll('-','/'))}</option>`).join('');$('#date').value=dates.includes(oldDate)?oldDate:data.date||dates.at(-1);
  const venues=[...new Set(data.races.map(r=>r.venue).filter(Boolean))].sort();$('#venue').innerHTML='<option value="">全開催場</option>'+venues.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('');$('#venue').value=venues.includes(oldVenue)?oldVenue:'';
  $('#updated').textContent='取得 '+new Date(data.updated_at).toLocaleString('ja-JP',{timeZone:'Asia/Tokyo',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'});render();
}
$('#refresh').addEventListener('click',()=>load());
setInterval(()=>{if(!document.hidden){if(data)render();load(true);}},60000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden){if(data)render();load(true);}});
load();
