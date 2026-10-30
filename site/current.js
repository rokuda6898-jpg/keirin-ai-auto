'use strict';
(() => {
 const MODEL='current459_expanded_history_MAIN4_20261030';
 const labels={MAIN:'本線',MEDIUM:'中穴',HIGH:'大穴'};
 const sessions={morning:'モーニング',day:'デイ',night:'ナイト',midnight:'ミッドナイト'};
 const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const num=x=>Number(x).toLocaleString('ja-JP');
 const pct=x=>Number.isFinite(x)?x.toFixed(2):'—';
 function expand(blocks){const ts=new Set();for(const f of blocks)for(const a of f.first)for(const b of f.second)for(const c of f.third)if(new Set([a,b,c]).size===3)ts.add([a,b,c].join('-'));return [...ts].sort();}
 function validate(r){
  const p=r.forecast;if(!p)return true;
  const cap=r.graded?18:p.category==='HIGH'?10:8;
  if(p.model_id!==MODEL||!labels[p.category]||p.unit_yen!==100||p.count!==cap||p.total_yen!==cap*100||p.blocks.length<1||p.blocks.length>4)throw Error('予想の方式・購入条件を確認できません');
  const ts=expand(p.blocks);if(ts.length!==cap||JSON.stringify(ts)!==JSON.stringify([...new Set(p.tickets)].sort()))throw Error('フォーメーションと買い目が一致しません');
  if(!p.band_probabilities||p.band_probabilities.length!==3||p.band_probabilities.some(x=>!Number.isFinite(x)||x<0||x>1)||Math.abs(p.band_probabilities.reduce((a,b)=>a+b,0)-1)>1e-5)throw Error('判定の数値を確認できません');
  const saved=Date.parse(p.saved_at),captured=Date.parse(p.captured_at);if(!Number.isFinite(saved)||!Number.isFinite(captured)||saved>=r.close_at*1000||captured>saved||p.history_through>=r.date)throw Error('締切前の保存を確認できません');
  const cars=new Set(r.riders.map(x=>x.car));if(ts.some(t=>t.split('-').some(c=>!cars.has(Number(c)))))throw Error('買い目と出走車が一致しません');
  return true;
 }
 function status(r,now=Date.now()){
  if(r.settlement)return {void:'返還',hit:'的中',miss:'結果確定'}[r.settlement.status]||'結果確認中';
  if(r.status==='cancelled')return '開催中止';
  if(!r.forecast)return r.status==='unavailable'?'予想確認中':r.close_at*1000<=now?'予想なし':'準備中';
  if(r.close_at*1000<=now)return '締切・結果待ち';
  return '予想公開中';
 }
 function copyText(r){const p=r.forecast;return `${r.date} ${r.venue} ${r.race_number}R｜${labels[p.category]}判定\n`+p.blocks.map((b,i)=>`${i+1}. ${b.first.join('・')} → ${b.second.join('・')} → ${b.third.join('・')}`).join('\n')+`\n同一車・重複を除く${p.count}点 × 100円 = ${num(p.total_yen)}円\n保存 ${stamp(p.saved_at)}／締切 ${time(r.close_at)}\nNEXUS 現行方式`;
 }
 const time=x=>x?new Date(x*1000).toLocaleTimeString('ja-JP',{timeZone:'Asia/Tokyo',hour:'2-digit',minute:'2-digit'}):'未定';
 const stamp=x=>x?new Date(x).toLocaleString('ja-JP',{timeZone:'Asia/Tokyo',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}):'—';
 const car=c=>`<span class="car c${Number(c)}">${Number(c)}</span>`;
 function stat(label,value,unit,detail){return `<div class="stat"><div class="stat-label">${label}</div><div class="stat-value">${esc(value)}<small>${unit}</small></div><div class="stat-detail">${detail}</div></div>`;}
 if(typeof module!=='undefined'&&module.exports){module.exports={expand,validate,status,copyText,MODEL};return;}
 let data=null,day='',session='all',venue='',selected='',copyTimer;
 const $=id=>document.getElementById(id);
 function performance(){const s=data.observed||{};$('observed').innerHTML=stat('的中率',pct(s.hit_pct),'%',`${num(s.hits||0)} / ${num(s.races||0)}レース`)+stat('回収率',pct(s.roi_pct),'%',`払戻 ${num(s.returned||0)}円 / 購入 ${num(s.stake||0)}円`)+stat('100倍以上の的中',s.HIGH_hits||0,'件','実際の払戻が100倍以上')+stat('結果待ち',s.pending||0,'レース',`全額返還 ${s.void||0}件は的中率から除外`);$('performance-note').textContent=s.races?'公開後に保存した同じ方式の予想だけを集計しています。':'公開後の結果が揃うと、自動で成績を表示します。終了済みレースを後から予想して加えることはありません。';
 const v=data.research;$('research').innerHTML=v?`<div class="stats">${stat('全国・的中率',pct(v.hit_pct),'%',`${num(v.hits)} / ${num(v.races)}レース`)}${stat('全国・回収率',pct(v.roi_pct),'%','全成立レースの集計')}${stat('全国・100倍以上の的中',v.HIGH_hits,'件','各100円・現行の点数')}${stat('大穴判定内・回収率',pct(v.HIGH_call_roi_pct),'%','大穴判定の6,385レースに限定')}</div>`:'';
 }
 function visible(){return data.races.filter(r=>r.date===day&&(session==='all'||session==='graded'?session==='all'||r.graded:!r.graded&&r.session===session));}
 function render(){if(!data)return;
 const rs=visible();const vs=[...new Map(rs.map(r=>[r.venue_id,r])).values()].sort((a,b)=>a.venue_id.localeCompare(b.venue_id));
 if(!vs.some(x=>x.venue_id===venue)){const next=rs.find(r=>r.forecast&&r.close_at*1000>Date.now());venue=next?.venue_id||vs[0]?.venue_id||'';}
 $('venues').innerHTML=vs.length?vs.map(r=>`<button data-venue="${esc(r.venue_id)}" aria-pressed="${r.venue_id===venue}">${esc(r.venue)}<small>${r.graded?'重賞':sessions[r.session]||''}</small></button>`).join(''):'<span class="muted">この条件の開催はありません。</span>';
 const races=rs.filter(r=>r.venue_id===venue).sort((a,b)=>a.race_number-b.race_number);
 if(!races.some(r=>r.race_id===selected))selected=(races.find(r=>r.forecast&&r.close_at*1000>Date.now())||races.find(r=>r.forecast)||races[0])?.race_id||'';
 $('race-count').textContent=races.length?`${races.length}レース`:'';
 $('race-list').innerHTML=races.length?races.map(r=>{const p=r.forecast;return `<button class="race-item" data-race="${esc(r.race_id)}" aria-current="${r.race_id===selected}" aria-label="${esc(r.venue)} ${r.race_number}レース ${p?labels[p.category]+'判定':''} ${esc(status(r))}"><div><div class="race-no">${r.race_number}<small>R</small></div><div class="race-time">${time(r.close_at)}</div></div><div><div class="race-name">${esc(r.race_type)}</div><div class="race-state">${esc(status(r))}</div></div><span class="tag ${p?p.category:'none'}">${p?labels[p.category]:r.status==='cancelled'?'中止':'未作成'}</span></button>`;}).join(''):'<p class="empty">条件を変更してご確認ください。</p>';
 detail(races.find(r=>r.race_id===selected));
 }
 function detail(r){
 if(!r){$('race-detail').innerHTML='<div class="empty detail-empty"><span class="empty-icon">◎</span><h2>該当するレースがありません</h2><p>日付や時間帯を変更してください。</p></div>';return;}
 const p=r.forecast;let body=`<div class="detail-header"><div><h2>${esc(r.venue)} <span>${r.race_number}R</span></h2><div class="detail-meta">${esc(r.race_class)} ${esc(r.race_type)} · ${r.graded?'重賞 · ':''}${esc(sessions[r.session]||'')} · ${esc(status(r))}</div></div><div class="deadline"><span>締切</span><b>${time(r.close_at)}</b>${p?`<div class="tag ${p.category}">${labels[p.category]}判定</div>`:''}</div></div>`;
 if(r.withdrawal_warning)body+=`<p class="notice warn">${esc(r.withdrawal_warning)}</p>`;
 if(r.settlement){const s=r.settlement;body+=`<div class="result ${s.status==='hit'?'hit':''}"><div class="result-flex"><strong>${esc(status(r))}</strong><span>払戻・返還 ${num(s.returned)}円 ／ 購入 ${num(s.stake)}円</span></div><div>${s.winning_tickets.map(t=>`${esc(t)}（${num(s.payoffs[t])}円 / 100円）`).join(' ・ ')||'対象車券は全額返還'}</div>${r.performance_eligible?'':'<p class="tiny">締切前の公開を確認できないため、公開後の成績には含めません。</p>'}</div>`;}
 if(!p){body+=`<div class="empty detail-empty"><span class="empty-icon">—</span><h2>${r.status==='unavailable'?'予想を確認中です':r.status==='cancelled'?'開催中止です':'このレースの予想は未作成です'}</h2><p>${esc(r.note||'締切前の情報が揃い次第、予想を表示します。')}</p></div>`;}
 else{
 body+=`<div class="pick-header"><h3>3連単フォーメーション<small>最大4本</small></h3><button class="copy" id="copy">買い目をコピー</button></div><div class="formation-box"><div class="formation-labels"><span></span><span>1着</span><span></span><span>2着</span><span></span><span>3着</span></div>${p.blocks.map((f,i)=>`<div class="formation-row"><small>0${i+1}</small><div class="car-group">${f.first.map(car).join('')}</div><span class="arrow">→</span><div class="car-group">${f.second.map(car).join('')}</div><span class="arrow">→</span><div class="car-group">${f.third.map(car).join('')}</div></div>`).join('')}<div class="cost"><span>重複を除く <b>${p.count}点</b></span><span>各100円 · 合計 <b>${num(p.total_yen)}円</b></span></div></div><p class="formation-note">上記の全組み合わせを購入。同一車の重複・フォーメーション間の重複は除きます。</p><details class="ticket-list"><summary>展開した${p.count}点を確認</summary><div class="tickets">${p.tickets.map(t=>`<span>${esc(t)}　100円</span>`).join('')}</div></details>`;
 body+=`<h3 class="minor-title">配当帯の見立て</h3><div class="probabilities">${['MAIN','MEDIUM','HIGH'].map((k,i)=>`<div class="prob ${k}"><div class="prob-label"><span>${labels[k]}</span><b>${(100*p.band_probabilities[i]).toFixed(1)}<small>%</small></b></div><div class="bar"><i style="width:${100*p.band_probabilities[i]}%"></i></div></div>`).join('')}</div><p class="tiny">本線40倍未満 ／ 中穴40〜100倍未満 ／ 大穴100倍以上。車券の的中率とは異なります。</p>`;
 const li=r.lineup||[];body+=`<h3 class="minor-title">予想隊列</h3><div class="lineup">${li.length?li.map(g=>`<span class="lineup-group">${g.map(slot=>slot.length>1?`<span>(${slot.map(car).join('／')})</span>`:slot.map(car).join('')).join('<span class="lineup-separator">−</span>')}</span>`).join('<span class="lineup-separator">／</span>'):'<span class="muted">隊列不明、またはライン適用外</span>'}</div>${r.lineup_errors?.length?'<p class="tiny">未確認の並びは補完せず、不明として予想しています。</p>':''}`;
 const marks=new Map(p.marks.map(m=>[m.car,m.mark]));body+=`<h3 class="minor-title">出走選手と予想印</h3><div class="riders-wrap"><table class="riders"><thead><tr><th>車番</th><th>印</th><th>選手</th><th>府県</th><th>競走得点</th><th>脚質</th></tr></thead><tbody>${r.riders.map(v=>`<tr><td>${car(v.car)}</td><td class="${marks.get(v.car)==='印なし'?'unmarked':'mark'}">${esc(marks.get(v.car))}</td><td>${esc(v.name)}</td><td>${esc(v.prefecture)}</td><td>${Number.isFinite(v.score)?v.score.toFixed(2):'—'}</td><td>${esc(v.style||'—')}</td></tr>`).join('')}</tbody></table></div><p class="tiny">印順だけでは選ばず、隊列・役割・別線・過去の調子を含めて全着順を評価します。</p><p class="tiny">予想保存 ${stamp(p.saved_at)} ／ 履歴 ${esc(p.history_through)}まで</p>`;
 }
 if(/^https:\/\/www\.winticket\.jp\//.test(r.source_url||''))body+=`<a class="source-link" href="${esc(r.source_url)}" target="_blank" rel="noopener noreferrer">出走表・最新情報を確認 ↗</a>`;
 $('race-detail').innerHTML=body;const button=$('copy');if(button)button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(copyText(r));$('toast').textContent='フォーメーションをコピーしました';$('toast').hidden=false;clearTimeout(copyTimer);copyTimer=setTimeout(()=>$('toast').hidden=true,2400);}catch{button.textContent='コピーできませんでした';}});
 }
 async function refresh(){
 $('reload').disabled=true;
 const paths=['original/current_live.json',...(location.hostname.endsWith('github.io')?['https://raw.githubusercontent.com/rokuda6898-jpg/keirin-ai-auto/main/outputs/current_live.json']:[])];
 try{const rs=await Promise.allSettled(paths.map(async url=>{const res=await fetch(url+'?t='+Date.now(),{cache:'no-store',signal:AbortSignal.timeout(18000)});if(!res.ok)throw Error(res.status);const d=await res.json();if(d.schema!==1||d.model_id!==MODEL||!Array.isArray(d.races)||!Number.isFinite(Date.parse(d.generated_at)))throw Error('現行データではありません');d.races.forEach(validate);return d;}));const ok=rs.filter(r=>r.status==='fulfilled').map(r=>r.value).sort((a,b)=>Date.parse(b.generated_at)-Date.parse(a.generated_at));if(!ok.length)throw Error('現在の予想データを読み込めません。');data=ok[0];
 const days=[...new Set([data.today,...data.races.map(r=>r.date)])].sort().reverse();if(!days.includes(day))day=data.today;
 $('date').innerHTML=days.map(d=>`<option value="${esc(d)}" ${d===day?'selected':''}>${esc(d.replace(/-/g,'/'))}${d===data.today?'（今日）':''}</option>`).join('');
 const age=(Date.now()-Date.parse(data.generated_at))/60000;const stale=age>45;const forecasts=data.races.filter(r=>r.date===data.today&&r.forecast).length;
 $('notice').classList.toggle('warn',stale||data.health.stage!=='ready');$('notice').textContent=stale?`最終更新は${stamp(data.generated_at)}です。更新が遅れています。締切・欠場情報は公式でご確認ください。`:`${data.health.message} 本日${forecasts}レースの予想を保存済みです。`;
 $('updated').textContent=`最終更新 ${stamp(data.generated_at)}（日本時間）／ 現行方式 2026.10.30 ／ 約20分ごとに更新`;
 render();performance();
 }catch(e){$('notice').classList.add('warn');$('notice').textContent='予想を読み込めませんでした。「更新」で再試行してください。';if(!data){$('race-list').innerHTML='<p class="empty">予想データを確認中です。</p>';$('race-detail').innerHTML='<div class="empty detail-empty"><h2>データを確認中です</h2><p>古い方式の予想への置き換えは行いません。</p></div>';}}finally{$('reload').disabled=false;}
 }
 $('date').addEventListener('change',e=>{day=e.target.value;venue='';selected='';render();});$('sessions').addEventListener('click',e=>{const q=e.target.closest('[data-session]');if(!q)return;session=q.dataset.session;venue='';selected='';for(const b of $('sessions').querySelectorAll('button'))b.setAttribute('aria-pressed',String(b===q));render();});$('venues').addEventListener('click',e=>{const q=e.target.closest('[data-venue]');if(q){venue=q.dataset.venue;selected='';render();}});$('race-list').addEventListener('click',e=>{const q=e.target.closest('[data-race]');if(q){selected=q.dataset.race;render();}});$('reload').addEventListener('click',refresh);refresh();setInterval(()=>{if(!document.hidden)refresh();},120000);setInterval(()=>{if(!document.hidden&&data)render();},60000);
})();
