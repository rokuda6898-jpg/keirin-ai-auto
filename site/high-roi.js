'use strict';
(() => {
 const MODEL='ROI9891_fixed_EX_20261102';
 const CLOUD_MODEL='high_roi_cloud_20261107_v1';
 const SELECTION_RULE='SC10955_native_20261102';
 const isSelected=r=>r.forecast?.model_id===MODEL&&r.selection?.rule_id===SELECTION_RULE&&r.selection?.selected===true;
 const selectionBadge=r=>isSelected(r)?'<span class="selection-badge" title="過去検証と同じ厳選条件を満たしたレース">厳選</span>':'';
 const MODEL_LABELS={[MODEL]:'旧高回収率モデル',[CLOUD_MODEL]:'高配当・再構築版（検証中）'};
 const FILES={[MODEL]:'high_roi_live.json',[CLOUD_MODEL]:'high_roi_cloud_live.json'};
 const labels={MAIN:'本線',MEDIUM:'中穴',HIGH:'大穴'};
 const sessions={morning:'モーニング',day:'デイ',night:'ナイト',midnight:'ミッドナイト'};
 const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const num=x=>Number(x).toLocaleString('ja-JP');
 const pct=x=>Number.isFinite(x)?x.toFixed(2):'—';
 function expand(blocks){const ts=new Set();for(const f of blocks)for(const a of f.first)for(const b of f.second)for(const c of f.third)if(new Set([a,b,c]).size===3)ts.add([a,b,c].join('-'));return [...ts].sort();}
 function validate(r,expectedModel){
  const p=r.forecast;if(!p)return true;
  if(typeof expectedModel==='string'&&p.model_id!==expectedModel)throw Error('異なる方式の予想が混在しています');
  if(!Array.isArray(r.riders)||!Array.isArray(p.blocks)||!Array.isArray(p.tickets)||!Array.isArray(p.marks))throw Error('予想の構造を確認できません');
  if(p.model_id===CLOUD_MODEL&&r.selection)throw Error('再構築版に旧方式の厳選条件は適用できません');
  if(r.selection){const s=r.selection;if(s.rule_id!==SELECTION_RULE||![true,false,null].includes(s.selected))throw Error('厳選の判定を確認できません');if(s.selected!==null&&(!Number.isFinite(Date.parse(s.created_at))||s.basis!=='before_EX'||!['S','A','B','C'].includes(s.confidence_grade)||s.history_through!==p.history_through||s.source_sha256!==p.source_sha256))throw Error('厳選の保存情報を確認できません');if(s.selected&& !['S','C'].includes(s.confidence_grade))throw Error('厳選条件が一致しません');}
  const cap=r.graded?18:p.category==='HIGH'?10:8;
  if(!Object.hasOwn(MODEL_LABELS,p.model_id)||!labels[p.category]||p.unit_yen!==100||p.count!==cap||p.total_yen!==cap*100||p.blocks.length<1||p.blocks.length>4)throw Error('予想の方式・購入条件を確認できません');
  for(const b of p.blocks)for(const key of ['first','second','third'])if(!Array.isArray(b[key])||!b[key].length||b[key].some(c=>!Number.isInteger(c)||c<1||c>9)||new Set(b[key]).size!==b[key].length)throw Error('フォーメーションの車番を確認できません');
  const ts=expand(p.blocks);if(ts.length!==cap||p.tickets.length!==cap||JSON.stringify(ts)!==JSON.stringify([...new Set(p.tickets)].sort()))throw Error('フォーメーションと買い目が一致しません');
  if(!p.band_probabilities||p.band_probabilities.length!==3||p.band_probabilities.some(x=>!Number.isFinite(x)||x<0||x>1)||Math.abs(p.band_probabilities.reduce((a,b)=>a+b,0)-1)>1e-5)throw Error('判定の数値を確認できません');
  const saved=Date.parse(p.saved_at),captured=Date.parse(p.captured_at);if(!Number.isFinite(saved)||!Number.isFinite(captured)||!Number.isFinite(r.close_at)||saved>=r.close_at*1000||captured>saved||!/^\d{4}-\d{2}-\d{2}$/.test(p.history_through)||p.history_through>=r.date)throw Error('締切前の保存を確認できません');
  const cars=new Set(r.riders.map(x=>x.car));if(ts.some(t=>t.split('-').some(c=>!cars.has(Number(c)))))throw Error('買い目と出走車が一致しません');
  if(p.model_id===CLOUD_MODEL){
   const c=p.confidence;
   if(!c||!['S','A','B'].includes(c.grade)||c.status!=='assessed'||!Number.isFinite(c.model_mass)||c.model_mass<0||c.model_mass>1)throw Error('再構築版の自信度を確認できません');
   if(p.marks.length!==cars.size||new Set(p.marks.map(m=>m.car)).size!==cars.size||p.marks.some(m=>!cars.has(m.car)))throw Error('再構築版の予想印が出走車と一致しません');
   if(p.odds&&Object.keys(p.odds).length){
    const quoted=Date.parse(p.odds_at);
    if(!Number.isFinite(quoted)||quoted>saved||quoted>=r.close_at*1000||Object.entries(p.odds).some(([t,o])=>!p.tickets.includes(t)||!Number.isFinite(o)||o<1))throw Error('予想作成時のオッズを確認できません');
   }
  }
  return true;
 }
 function validateFeed(d,expectedModel){
  if(d?.schema!==1||d.model_id!==expectedModel||!Object.hasOwn(MODEL_LABELS,d.model_id)||!Array.isArray(d.races)||!Number.isFinite(Date.parse(d.generated_at))||!/^\d{4}-\d{2}-\d{2}$/.test(d.today))throw Error('指定した方式の予想データではありません');
  if(d.observed?.model_id!==expectedModel)throw Error('成績の方式が一致しません');
  const ids=new Set();for(const r of d.races){if(!r||typeof r.race_id!=='string'||ids.has(r.race_id))throw Error('レースの識別情報が不正です');ids.add(r.race_id);validate(r,expectedModel);}
  return d;
 }
 function todayCount(d,today){return d?.races?.filter(r=>r.date===today&&r.forecast?.model_id===d.model_id).length||0;}
 function defaultModel(feeds,today){
  if(todayCount(feeds[MODEL],today))return MODEL;
  if(feeds[CLOUD_MODEL]?.today===today||todayCount(feeds[CLOUD_MODEL],today))return CLOUD_MODEL;
  return Object.keys(feeds).filter(id=>Object.hasOwn(MODEL_LABELS,id)).sort((a,b)=>Date.parse(feeds[b].generated_at)-Date.parse(feeds[a].generated_at))[0]||'';
 }
 function newestFeed(feeds){const checked=d=>Date.parse(d.model_id===CLOUD_MODEL?(d.updated_at||d.generated_at):d.generated_at);return [...feeds].sort((a,b)=>(checked(b)-checked(a))||((Date.parse(b.results_updated_at||'')||0)-(Date.parse(a.results_updated_at||'')||0)))[0];}
 function status(r,now=Date.now()){
  if(r.settlement)return {void:'返還',hit:'的中',miss:'結果確定'}[r.settlement.status]||'結果確認中';
  if(r.status==='cancelled')return '開催中止';
  if(!r.forecast)return r.status==='unavailable'?'予想確認中':r.close_at*1000<=now?'予想なし':'準備中';
  if(r.close_at*1000<=now)return '締切・結果待ち';
  return '予想公開中';
 }
 function copyText(r){const p=r.forecast;return `${r.date} ${r.venue} ${r.race_number}R｜${labels[p.category]}判定${isSelected(r)?'｜厳選':''}${p.model_id===CLOUD_MODEL?'｜自信度'+p.confidence.grade:''}\n`+p.blocks.map((b,i)=>`${i+1}. ${b.first.join('・')} → ${b.second.join('・')} → ${b.third.join('・')}`).join('\n')+`\n同一車・重複を除く${p.count}点 × 100円 = ${num(p.total_yen)}円\n保存 ${stamp(p.saved_at)}／締切 ${time(r.close_at)}\nNEXUS ${MODEL_LABELS[p.model_id]}`;
 }
 const time=x=>x?new Date(x*1000).toLocaleTimeString('ja-JP',{timeZone:'Asia/Tokyo',hour:'2-digit',minute:'2-digit'}):'未定';
 const stamp=x=>x?new Date(x).toLocaleString('ja-JP',{timeZone:'Asia/Tokyo',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}):'—';
 const jstToday=()=>new Date(Date.now()+9*3600000).toISOString().slice(0,10);
 const car=c=>`<span class="car c${Number(c)}">${Number(c)}</span>`;
 function stat(label,value,unit,detail){return `<div class="stat"><div class="stat-label">${label}</div><div class="stat-value">${esc(value)}<small>${unit}</small></div><div class="stat-detail">${detail}</div></div>`;}
 if(typeof module!=='undefined'&&module.exports){module.exports={expand,validate,validateFeed,status,copyText,MODEL,CLOUD_MODEL,MODEL_LABELS,isSelected,selectionBadge,SELECTION_RULE,todayCount,defaultModel,newestFeed};return;}
 let data=null,day='',session='all',venue='',selected='',copyTimer,chosenModel='',userChoseModel=false,feeds={},refreshing=false;
 const $=id=>document.getElementById(id);
 function performance(){const cloud=data.model_id===CLOUD_MODEL,s=data.observed||{};$('performance-title').textContent=(cloud?'再構築版':'旧方式')+'の公開後成績';$('observed').innerHTML=stat('的中率',pct(s.hit_pct),'%',`${num(s.hits||0)} / ${num(s.races||0)}レース`)+stat('回収率',pct(s.roi_pct),'%',`払戻 ${num(s.returned||0)}円 / 購入 ${num(s.stake||0)}円`)+stat('100倍以上の的中',s.HIGH_hits||0,'件','実際の払戻が100倍以上')+stat('結果待ち',s.pending||0,'レース',`全額返還 ${s.void||0}件は的中率から除外`);$('performance-note').textContent=(cloud?'再構築版':'旧高回収率モデル')+'として締切前に公開した予想だけの集計です。'+(cloud?'旧方式の成績・過去研究値は引き継ぎません。':'再構築版の結果は含みません。')+(s.races?'':'確定結果が揃ってから的中率・回収率を表示します。');
 const v=!cloud&&data.research;$('research').innerHTML=v?`<div class="stats">${stat('EX追加前・的中率',pct(v.hit_pct),'%',`${num(v.hits)} / ${num(v.races)}レース`)}${stat('EX追加前・回収率',pct(v.roi_pct),'%','旧方式の研究期間の集計')}${stat('EX追加前・100倍以上',v.HIGH_hits,'件','各100円・旧方式の点数')}</div>`:'';
 $('legacy-research').hidden=cloud;$('cloud-method').hidden=!cloud;$('legacy-intro').hidden=cloud;
 }
 function visible(){return data.races.filter(r=>r.date===day&&(session==='all'||session==='graded'?session==='all'||r.graded:!r.graded&&r.session===session));}
 function render(){if(!data)return;
 const rs=visible();const vs=[...new Map(rs.map(r=>[r.venue_id,r])).values()].sort((a,b)=>String(a.venue_id).localeCompare(String(b.venue_id)));
 if(!vs.some(x=>x.venue_id===venue)){const next=rs.find(r=>r.forecast&&r.close_at*1000>Date.now());venue=next?.venue_id||vs[0]?.venue_id||'';}
 $('venues').innerHTML=vs.length?vs.map(r=>`<button data-venue="${esc(r.venue_id)}" aria-pressed="${r.venue_id===venue}">${esc(r.venue)}<small>${r.graded?'重賞':sessions[r.session]||''}</small></button>`).join(''):'<span class="muted">この条件の開催はありません。</span>';
 const races=rs.filter(r=>r.venue_id===venue).sort((a,b)=>a.race_number-b.race_number);
 if(!races.some(r=>r.race_id===selected))selected=(races.find(r=>r.forecast&&r.close_at*1000>Date.now())||races.find(r=>r.forecast)||races[0])?.race_id||'';
 $('race-count').textContent=races.length?`${races.length}レース`:'';
 $('race-list').innerHTML=races.length?races.map(r=>{const p=r.forecast;return `<button class="race-item" data-race="${esc(r.race_id)}" aria-current="${r.race_id===selected}" aria-label="${esc(r.venue)} ${r.race_number}レース ${p?labels[p.category]+'判定':''} ${isSelected(r)?'厳選 ':''}${esc(status(r))}"><div><div class="race-no">${r.race_number}<small>R</small></div><div class="race-time">${time(r.close_at)}</div></div><div><div class="race-name">${esc(r.race_type)}</div><div class="race-state">${esc(status(r))}</div></div><span class="race-tags"><span class="tag ${p?p.category:'none'}">${p?labels[p.category]:r.status==='cancelled'?'中止':'未作成'}</span>${selectionBadge(r)}</span></button>`;}).join(''):'<p class="empty">条件を変更してご確認ください。</p>';
 detail(races.find(r=>r.race_id===selected));
 }
 function detail(r){
 if(!r){$('race-detail').innerHTML='<div class="empty detail-empty"><span class="empty-icon">◎</span><h2>該当するレースがありません</h2><p>日付や時間帯を変更してください。</p></div>';return;}
 const p=r.forecast;let body=`<div class="detail-header"><div><h2>${esc(r.venue)} <span>${r.race_number}R</span></h2>${selectionBadge(r)}<div class="detail-meta">${esc(r.race_class)} ${esc(r.race_type)} · ${r.graded?'重賞 · ':''}${esc(sessions[r.session]||'')} · ${esc(status(r))}</div></div><div class="deadline"><span>締切</span><b>${time(r.close_at)}</b>${p?`<div class="tag ${p.category}">${labels[p.category]}判定</div>`:''}</div></div>`;
 if(p)body+=`<p class="notice ${p.model_id===CLOUD_MODEL?'warn':''}">${esc(MODEL_LABELS[p.model_id])}の保存予想です。${p.model_id===CLOUD_MODEL?'旧方式と別に成績を集計しています。':''}</p>`;
 if(r.withdrawal_warning)body+=`<p class="notice warn">${esc(r.withdrawal_warning)}</p>`;
 if(r.settlement){const s=r.settlement;body+=`<div class="result ${s.status==='hit'?'hit':''}"><div class="result-flex"><strong>${esc(status(r))}</strong><span>払戻・返還 ${num(s.returned)}円 ／ 購入 ${num(s.stake)}円</span></div><div>${s.winning_tickets.map(t=>`${esc(t)}（${num(s.payoffs[t])}円 / 100円）`).join(' ・ ')||'対象車券は全額返還'}</div>${r.performance_eligible?'':'<p class="tiny">締切前の公開を確認できないため、公開後の成績には含めません。</p>'}</div>`;}
 if(!p){body+=`<div class="empty detail-empty"><span class="empty-icon">—</span><h2>${r.status==='unavailable'?'予想を確認中です':r.status==='cancelled'?'開催中止です':'このレースの予想は未作成です'}</h2><p>${esc(r.note||'締切前の情報が揃い次第、予想を表示します。')}</p></div>`;}
 else{
 if(p.model_id===CLOUD_MODEL){
  const quoted=Object.keys(p.odds||{}).length>0;
  body+=`<div class="cloud-confidence"><strong>自信度 ${esc(p.confidence.grade)}</strong><span>${esc(p.confidence.reason||'選択した買い目へのモデル支持で判定')}</span></div><p class="tiny">自信度はモデル評価に基づく区分です。実測の的中率を示すものではありません。</p><p class="cloud-basis">${quoted?'締切前のモデル評価と、保存したオッズを考慮した買い目です。':'有効なオッズが揃っていないため、保存したモデル評価による暫定の買い目です。'}</p>`;
 }
 body+=`<div class="pick-header"><h3>3連単フォーメーション<small>最大4本</small></h3><button class="copy" id="copy">買い目をコピー</button></div><div class="formation-box"><div class="formation-labels"><span></span><span>1着</span><span></span><span>2着</span><span></span><span>3着</span></div>${p.blocks.map((f,i)=>`<div class="formation-row"><small>0${i+1}</small><div class="car-group">${f.first.map(car).join('')}</div><span class="arrow">→</span><div class="car-group">${f.second.map(car).join('')}</div><span class="arrow">→</span><div class="car-group">${f.third.map(car).join('')}</div></div>`).join('')}<div class="cost"><span>重複を除く <b>${p.count}点</b></span><span>各100円 · 合計 <b>${num(p.total_yen)}円</b></span></div></div><p class="formation-note">上記の全組み合わせを購入。同一車の重複・フォーメーション間の重複は除きます。</p><details class="ticket-list"><summary>展開した${p.count}点を確認</summary><div class="tickets">${p.tickets.map(t=>`<span>${esc(t)}　100円</span>`).join('')}</div></details>`;
 if(p.model_id===CLOUD_MODEL&&Object.keys(p.odds||{}).length){body+=`<details class="ticket-list"><summary>予想作成時のオッズを確認</summary><p class="tiny">取得 ${stamp(p.odds_at)}（日本時間）。オッズは変動します。</p><div class="tickets">${p.tickets.map(t=>`<span>${esc(t)}　${Number.isFinite(p.odds[t])?p.odds[t].toFixed(1)+'倍':'未取得'}</span>`).join('')}</div></details>`;}
 body+=`<h3 class="minor-title">配当帯の見立て</h3><div class="probabilities">${['MAIN','MEDIUM','HIGH'].map((k,i)=>`<div class="prob ${k}"><div class="prob-label"><span>${labels[k]}</span><b>${(100*p.band_probabilities[i]).toFixed(1)}<small>%</small></b></div><div class="bar"><i style="width:${100*p.band_probabilities[i]}%"></i></div></div>`).join('')}</div><p class="tiny">${p.model_id===CLOUD_MODEL?'本線・中穴・大穴のモデル評価です。買い目の的中率や実際の配当とは異なります。':'本線40倍未満 ／ 中穴40〜100倍未満 ／ 大穴100倍以上。車券の的中率とは異なります。'}</p>`;
 const li=r.lineup||[];body+=`<h3 class="minor-title">予想隊列</h3><div class="lineup">${li.length?li.map(g=>`<span class="lineup-group">${g.map(slot=>slot.length>1?`<span>(${slot.map(car).join('／')})</span>`:slot.map(car).join('')).join('<span class="lineup-separator">−</span>')}</span>`).join('<span class="lineup-separator">／</span>'):'<span class="muted">隊列不明、またはライン適用外</span>'}</div>${r.lineup_errors?.length?'<p class="tiny">未確認の並びは補完せず、不明として予想しています。</p>':''}`;
 const marks=new Map(p.marks.map(m=>[m.car,m.mark]));body+=`<h3 class="minor-title">出走選手と予想印</h3><div class="riders-wrap"><table class="riders"><thead><tr><th>車番</th><th>印</th><th>選手</th><th>府県</th><th>競走得点</th><th>脚質</th></tr></thead><tbody>${r.riders.map(v=>`<tr><td>${car(v.car)}</td><td class="${marks.get(v.car)==='印なし'?'unmarked':'mark'}">${esc(marks.get(v.car))}</td><td>${esc(v.name)}</td><td>${esc(v.prefecture)}</td><td>${Number.isFinite(v.score)?v.score.toFixed(2):'—'}</td><td>${esc(v.style||'—')}</td></tr>`).join('')}</tbody></table></div><p class="tiny">印順だけでは選ばず、隊列・役割・別線・過去の調子を含めて全着順を評価します。</p><p class="tiny">予想保存 ${stamp(p.saved_at)} ／ 履歴 ${esc(p.history_through)}まで</p>`;
 }
 if(/^https:\/\/www\.winticket\.jp\//.test(r.source_url||''))body+=`<a class="source-link" href="${esc(r.source_url)}" target="_blank" rel="noopener noreferrer">出走表・最新情報を確認 ↗</a>`;
 $('race-detail').innerHTML=body;const button=$('copy');if(button)button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(copyText(r));$('toast').textContent='フォーメーションをコピーしました';$('toast').hidden=false;clearTimeout(copyTimer);copyTimer=setTimeout(()=>$('toast').hidden=true,2400);}catch{button.textContent='コピーできませんでした';}});
 }
 function showFeed(){
  data=feeds[chosenModel];if(!data)return;
  const cloud=chosenModel===CLOUD_MODEL,today=jstToday();
  $('forecast-model').innerHTML=[CLOUD_MODEL,MODEL].map(id=>`<option value="${esc(id)}" ${id===chosenModel?'selected':''} ${feeds[id]?'':'disabled'}>${esc(MODEL_LABELS[id])}${feeds[id]?'':'・取得待ち'}</option>`).join('');
  $('model-note').textContent=cloud?'再構築版は検証中です。保存した全着順評価と取得できた締切前オッズから買い目を選び、公開後の成績を独立して記録します。':'旧高回収率モデルの保存記録です。再構築版の予想・成績とは別に表示しています。';
  const days=[...new Set([data.today,...data.races.map(r=>r.date)])].sort().reverse();if(!days.includes(day))day=data.today;
  $('date').innerHTML=days.map(d=>`<option value="${esc(d)}" ${d===day?'selected':''}>${esc(d.replace(/-/g,'/'))}${d===today?'（今日）':''}</option>`).join('');
  const checkedAt=cloud?(data.updated_at||data.generated_at):data.generated_at;
  const age=(Date.now()-Date.parse(checkedAt))/60000,stale=age>45,previousDay=data.today!==today;
  const todays=data.races.filter(r=>r.date===today),forecasts=todayCount(data,today),upcoming=todays.filter(r=>r.forecast&&r.close_at*1000>Date.now()).length;
  const missingOpen=todays.filter(r=>!r.forecast&&r.status!=='cancelled'&&r.close_at*1000>Date.now()).length;
  $('notice').classList.toggle('warn',previousDay||stale||missingOpen>0||data.health?.stage!=='ready');
  const name=cloud?'再構築版':'旧高回収率モデル';
  const scheduled=cloud&&Number.isInteger(data.coverage?.source_races)?data.coverage.source_races:todays.length;
  let message=previousDay?`${name}の本日分は未掲載です。${data.today.replace(/-/g,'/')}までの保存記録を表示しています。`:`${name}：本日${forecasts} / ${scheduled}レースを保存済み。締切前の予想は${upcoming}レースです。`;
  if(!previousDay&&missingOpen)message+=` 未作成の締切前レース：${missingOpen}件。`;
  if(!previousDay&&!cloud)message+=` 厳選${todays.filter(isSelected).length}レース。`;
  if(!previousDay&&stale)message+=` 最終確認は${stamp(checkedAt)}です。`;
  if(!previousDay&&data.health?.message&&(!forecasts||missingOpen))message+=' '+data.health.message;
  $('notice').textContent=message;
  $('updated').textContent=`${cloud?'データ確認':'予想更新'} ${stamp(checkedAt)} ／ 結果更新 ${stamp(data.results_updated_at)}（日本時間）／ ${MODEL_LABELS[chosenModel]}`;
  render();performance();
 }
 async function fetchFeed(id){
  const filename=FILES[id],paths=['original/'+filename,'https://raw.githubusercontent.com/rokuda6898-jpg/keirin-ai-auto/main/outputs/'+filename];
  const rs=await Promise.allSettled(paths.map(async url=>{const res=await fetch(url+'?t='+Date.now(),{cache:'no-store',signal:AbortSignal.timeout(18000)});if(!res.ok)throw Error('HTTP '+res.status);return validateFeed(await res.json(),id);}));
  const ok=rs.filter(r=>r.status==='fulfilled').map(r=>r.value);
  if(!ok.length)throw Error('この方式のデータを取得できません');
  return newestFeed(ok);
 }
 async function refresh(){
  if(refreshing)return;refreshing=true;$('reload').disabled=true;
  try{
   const ids=[MODEL,CLOUD_MODEL],rs=await Promise.allSettled(ids.map(fetchFeed));
   const received=new Set();rs.forEach((result,i)=>{if(result.status==='fulfilled'){feeds[ids[i]]=newestFeed([result.value,...(feeds[ids[i]]?[feeds[ids[i]]]:[])]);received.add(ids[i]);}});
   if(!Object.keys(feeds).length)throw Error('予想データを取得できません');
   const next=userChoseModel&&feeds[chosenModel]?chosenModel:defaultModel(feeds,jstToday());
   if(chosenModel!==next){day='';venue='';selected='';}chosenModel=next;
   showFeed();
   if(!received.has(chosenModel)){$('notice').classList.add('warn');$('notice').textContent='最新データを取得できませんでした。前回読み込んだ'+MODEL_LABELS[chosenModel]+'の保存記録を表示しています。「更新」で再試行してください。';}
  }catch(e){$('notice').classList.add('warn');$('notice').textContent='予想を読み込めませんでした。「更新」で再試行してください。';if(!data){$('race-list').innerHTML='<p class="empty">予想データを確認中です。</p>';$('race-detail').innerHTML='<div class="empty detail-empty"><h2>データを確認中です</h2><p>この方式の保存済み予想を取得でき次第、表示します。</p></div>';}}
  finally{refreshing=false;$('reload').disabled=false;}
 }
 $('forecast-model').addEventListener('change',e=>{if(!feeds[e.target.value])return;chosenModel=e.target.value;userChoseModel=true;day='';venue='';selected='';showFeed();});
 $('date').addEventListener('change',e=>{day=e.target.value;venue='';selected='';render();});$('sessions').addEventListener('click',e=>{const q=e.target.closest('[data-session]');if(!q)return;session=q.dataset.session;venue='';selected='';for(const b of $('sessions').querySelectorAll('button'))b.setAttribute('aria-pressed',String(b===q));render();});$('venues').addEventListener('click',e=>{const q=e.target.closest('[data-venue]');if(q){venue=q.dataset.venue;selected='';render();}});$('race-list').addEventListener('click',e=>{const q=e.target.closest('[data-race]');if(q){selected=q.dataset.race;render();}});$('reload').addEventListener('click',refresh);refresh();setInterval(()=>{if(!document.hidden)refresh();},120000);setInterval(()=>{if(!document.hidden&&data)showFeed();},60000);
})();
