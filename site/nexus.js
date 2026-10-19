'use strict';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const yen=x=>x==null?'—':Math.round(x).toLocaleString('ja-JP')+'円',pct=x=>x==null?'—':(x*100).toFixed(2)+'％';
const cats={FAVORITE:'本命',MAIN:'本線',MEDIUM:'中穴',HIGH:'大穴'},sessions={morning:'モーニング',day:'デイ',night:'ナイト',midnight:'ミッドナイト'};
let data=null,tab='forecast',session='all',budget=5000;
const car=n=>`<span class="car c${Number(n)}">${Number(n)}</span>`;
function renderStats(rows){
  $('#stats').innerHTML=['flat',String(budget)].map((mode,index)=>{
    const a=rows.filter(r=>r.settlement?.[mode]&&!r.settlement[mode].void).map(r=>r.settlement[mode]);
    const hits=a.filter(x=>x.hit).length,stake=a.reduce((s,x)=>s+x.stake,0),ret=a.reduce((s,x)=>s+x.returned,0);
    return `<div class="stat-card"><div class="stat-label">${index?yen(budget)+' 配分':'1点100円'} · 確定成績</div><div class="stat-row"><div class="metric"><b>${pct(a.length?hits/a.length:null)}</b><span>的中率 · ${hits}/${a.length}レース</span></div><div class="metric"><b>${pct(stake?ret/stake:null)}</b><span>回収率</span></div></div><p>購入 ${yen(stake)} ／ 払戻 ${yen(ret)}<br>${index?'締切前に配分を保存できたレースのみ':'締切前に予想を保存できたレースのみ'}</p></div>`;
  }).join('');
}
function raceHTML(r){
  const p=r.prediction,al=p?.allocations?.[budget],ss=r.settlement,status=r.cancelled?'中止・返還':ss?'結果確定':r.closed?'締切済み':'受付中';
  const time=r.start_at?new Date(r.start_at).toLocaleTimeString('ja-JP',{timeZone:'Asia/Tokyo',hour:'2-digit',minute:'2-digit'}):'時刻未確認';
  const close=r.close_at?new Date(r.close_at).toLocaleTimeString('ja-JP',{timeZone:'Asia/Tokyo',hour:'2-digit',minute:'2-digit'}):'—';
  const head=`<div class="race-head"><div><div class="venue-title">${esc(r.venue)}<span class="race-number">${Number(r.race_no)}R</span></div><div class="race-meta">${esc(r.class_name)} ${esc(r.stage)} · ${esc(sessions[r.session]||'時間帯未確認')}</div></div><div class="time"><b>${time}</b><span>${status} · 締切 ${close}</span></div></div>`;
  if(!p)return `<article class="race">${head}<div class="race-body"><div class="notice">${esc(r.message||'予想の準備中です。')}</div></div></article>`;
  const marks=(r.riders||[]).slice().sort((a,b)=>(p.first_probabilities?.[b.car]||0)-(p.first_probabilities?.[a.car]||0)).filter(x=>p.marks?.[x.car]);
  const forms=p.blocks.map((b,i)=>`<div class="formation"><small>${i+1}</small>${['first','second','third'].map(k=>`<div class="car-set">${b[k].map(car).join('')}</div>`).join('')}</div>`).join('');
  const flat=ss?.flat,weighted=ss?.[budget];
  const moneyNote=al?.available?`<p class="${al.quote_break_even?'ok':'warning'}">最低払戻見込み ${yen(al.minimum_return)}<br>${al.quote_break_even?'取得時点では全点で購入額以上':'この点数・予算ではトリガミ回避不可'}</p>`:'<p class="warning">購入前オッズ待ち・配分未確定</p>';
  const profits=x=>x?`<p class="${x.returned>=x.stake?'positive':'negative'}">払戻 ${yen(x.returned)} ／ ${x.returned>=x.stake?'+':''}${yen(x.returned-x.stake)}</p>`:'';
  const stakeRows=p.tickets.slice().sort((a,b)=>(p.odds?.[a]??Infinity)-(p.odds?.[b]??Infinity)).map(t=>`<tr><td>${esc(t)}</td><td>${p.odds?.[t]!=null?Number(p.odds[t]).toFixed(1)+'倍':'—'}</td><td>100円</td><td>${al?.available?yen(al.stakes[t]):'—'}</td></tr>`).join('');
  const lineup=(r.lines||[]).map(line=>line.map(slot=>slot.join('・')).join('－')).join(' ／ ');
  const source=/^https:\/\/www\.winticket\.jp\//.test(r.source_url||'')?`<a class="source" href="${esc(r.source_url)}" target="_blank" rel="noopener noreferrer">出走表・最新オッズ ↗</a>`:'';
  return `<article class="race" id="race-${esc(r.race_id)}">${head}<div class="race-body"><div class="route-line"><span class="pill ${p.category==='HIGH'?'high':p.category==='MEDIUM'?'medium':''}">${esc(cats[p.category]||p.category)}</span><span class="count">3連単 ${p.count}点 · ${p.blocks.length}フォーメーション</span></div><div class="riders">${marks.map(x=>`<span class="rider"><span class="mark">${esc(p.marks[x.car])}</span>${car(x.car)}${esc(x.name)}</span>`).join('')}</div><div class="formations"><div class="form-head"><span></span><span>1着</span><span>2着</span><span>3着</span></div>${forms}</div><p class="form-note">各段の全組み合わせを購入 · 同一車と重複買い目を除き${p.count}点</p><div class="money"><div><small>1点100円</small><strong>${yen(p.count*100)}</strong>${profits(flat)}</div><div><small>オッズに応じた強弱配分</small><strong>${yen(budget)}</strong>${weighted?profits(weighted):moneyNote}</div></div><details><summary>金額の内訳・隊列を見る</summary><p class="lineup">予想隊列：${esc(lineup||'未確認（補完なし）')}</p><table class="stakes"><thead><tr><th>買い目</th><th>取得オッズ</th><th>均等</th><th>配分</th></tr></thead><tbody>${stakeRows}</tbody></table><p class="form-note">${esc(p.odds_at?'オッズ取得 '+new Date(p.odds_at).toLocaleString('ja-JP',{timeZone:'Asia/Tokyo'}):'締切前オッズの保存なし')}<br>${esc(p.captured_at?'予想保存 '+new Date(p.captured_at).toLocaleString('ja-JP',{timeZone:'Asia/Tokyo'}):'')}</p></details><div class="actions"><button class="copy" data-copy="${esc(r.race_id)}">フォーメーションをコピー</button>${source}</div></div>${ss?`<div class="result-bar ${flat?.hit?'hit':''}"><span>${flat?.void?'中止・返還':flat?.hit?'的中':'不的中'}</span><span>${esc((r.winners||[]).join(' / '))}</span></div>`:''}</article>`;
}
function render(){
  if(!data)return;$('#race-view').classList.toggle('hidden',tab==='model');$('#model-view').classList.toggle('hidden',tab!=='model');
  const day=$('#date').value;let rows=(data.races||[]).filter(r=>r.date===day&&(session==='all'||r.session===session));renderStats(rows);
  if(tab==='results')rows=rows.filter(r=>r.closed||r.settlement||r.cancelled);
  else if(day===data.date)rows=rows.filter(r=>!r.closed&&!r.cancelled);
  rows.sort((a,b)=>(a.start_at||'').localeCompare(b.start_at||'')||a.race_id.localeCompare(b.race_id));
  $('#races').innerHTML=rows.length?rows.map(raceHTML).join(''):'<div class="empty">この条件のレースはありません。</div>';
  $('#list-title').textContent=tab==='results'?'結果一覧':'レース一覧';$('#race-count').textContent=rows.length+'レース';
  $('#status-note').classList.toggle('hidden',!data.notice);$('#status-note').textContent=data.notice||'';
  const h=data.model?.historical||{};
  $('#model-view').innerHTML=`<div class="research"><div class="eyebrow">採用済み / COMPLETE FORMATION</div><h2>隊列・役割フォーメーション</h2><p>レースごとに1〜3着の候補を評価して、最大4本のフォーメーションを組みます。1・2着を固定する前提は置いていません。</p><div class="stat-row"><div class="metric"><b>${pct(h.hit_rate)}</b><span>過去検証の的中率</span></div><div class="metric"><b>${pct(h.roi)}</b><span>各100円の回収率</span></div><div class="metric"><b>${Number(h.races||0).toLocaleString()}</b><span>検証レース数</span></div></div><p>学習：2023/10/9〜2025/10/8<br>検証：2025/10/9〜2026/10/8（繰り返し研究に使用した期間）</p><ul><li>本命6点・本線15点・中穴15点・大穴24点。全組み合わせを維持。</li><li>開催場、隊列、先頭・番手などの役割、強さ、過去の戦法、時間帯、14・30・90日の調子を評価。</li><li>予想印は選手評価の目安です。実際の買い目はフォーメーションを確認してください。</li></ul><p>上の数字は過去検証です。公開後の成績は「結果・成績」に分けて表示します。回収率は100％を下回っており、利益や的中を保証するモデルではありません。</p><details><summary>データと集計の注意点</summary><p>当日の結果・確定オッズは予想入力に使いません。欠測は埋めず、過去の出走表や開催場形状には取得時点の限界があります。返還は的中率から除外。配分成績は締切前にオッズと金額を保存できたレースのみ集計します。</p></details></div>`;
  $$('.copy').forEach(b=>b.addEventListener('click',async()=>{const r=data.races.find(x=>x.race_id===b.dataset.copy),p=r.prediction;const text=`${r.date} ${r.venue} ${r.race_no}R 3連単 ${p.count}点\n`+p.blocks.map(x=>`${x.first.join('・')} → ${x.second.join('・')} → ${x.third.join('・')}`).join('\n');try{await navigator.clipboard.writeText(text);toast('フォーメーションをコピーしました');}catch{toast('コピーできませんでした。買い目を選択してコピーしてください。');}}));
}
function toast(t){$('#toast').textContent=t;$('#toast').classList.remove('hidden');setTimeout(()=>$('#toast').classList.add('hidden'),2500);}
$$('[data-tab]').forEach(b=>b.addEventListener('click',()=>{tab=b.dataset.tab;$$('[data-tab]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});render();}));
$$('[data-session]').forEach(b=>b.addEventListener('click',()=>{session=b.dataset.session;$$('[data-session]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});render();}));
$('#budget').addEventListener('change',()=>{budget=Number($('#budget').value);render();});$('#date').addEventListener('change',render);
for(const [id,direction]of[['prev-day',-1],['next-day',1]])$('#'+id).addEventListener('click',()=>{const s=$('#date');s.selectedIndex=Math.max(0,Math.min(s.options.length-1,s.selectedIndex+direction));render();});
async function load(){
  for(const url of ['https://raw.githubusercontent.com/rokuda6898-jpg/keirin-ai-auto/main/outputs/nexus_live.json','original/nexus_live.json','../outputs/nexus_live.json']){
    try{const response=await fetch(url+'?t='+Math.floor(Date.now()/60000),{cache:'no-store'});if(!response.ok)throw Error(response.status);data=await response.json();if(data.schema!==1)throw Error('invalid feed');break;}catch{data=null;}
  }
  $('#loading').classList.add('hidden');
  if(!data){$('#races').innerHTML='<div class="empty">予想データを読み込めませんでした。しばらくしてから再読み込みしてください。</div>';$('#updated').textContent='通信を確認してください';return;}
  const dates=[...new Set([...(data.races||[]).map(r=>r.date),data.date].filter(Boolean))].sort();$('#date').innerHTML=dates.map(d=>`<option value="${esc(d)}">${esc(d.replaceAll('-','/'))}</option>`).join('');$('#date').value=data.date||dates.at(-1);$('#updated').textContent='更新 '+new Date(data.updated_at).toLocaleString('ja-JP',{timeZone:'Asia/Tokyo',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'});render();
}
load();
