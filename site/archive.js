'use strict';
const $=s=>document.querySelector(s);
const node=(tag,text,className)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;};
const raw=['localhost','127.0.0.1'].includes(location.hostname)?'../outputs/':'https://raw.githubusercontent.com/rokuda6898-jpg/keirin-ai-auto/main/outputs/';
let archive,dates=[],page=0;
const pageSize=30;
const hasTickets=r=>r.forecasts.some(p=>(p.tickets||[]).length>0);
const number=n=>Number(n).toLocaleString('ja-JP');
function numbers(buy){const box=node('span',undefined,'ticket-numbers');String(buy||'').split('-').forEach((value,i)=>{if(i)box.append(node('span','−','ticket-separator'));box.append(node('span',value,'car c'+value));});return box;}
function labelDate(date){return new Intl.DateTimeFormat('ja-JP',{year:'numeric',month:'long',day:'numeric',weekday:'short',timeZone:'Asia/Tokyo'}).format(new Date(date+'T12:00:00+09:00'));}
function button(text,action){const b=node('button',text,'quiet-button');b.type='button';b.addEventListener('click',action);return b;}
function raceCard(r){
 const card=node('details',undefined,'race archive-race');const summary=node('summary');
 summary.append(node('strong',r.number+'R','race-number'),node('span',r.venue,'archive-race-venue'));
 const result=node('span',undefined,'archive-result');result.append(node('small','公式着順'),r.actual?numbers(r.actual):node('span','照合中'));summary.append(result,node('span',hasTickets(r)?'買い目あり':'結果のみ','badge '+(hasTickets(r)?'recorded':'')));card.append(summary);
 const body=node('div',undefined,'race-body');
 if(!r.forecasts.length)body.append(node('p','当時の買い目記録はありません。公式着順のみ保存しています。','scope'));
 r.forecasts.forEach((p,index)=>{
  const section=node('section',undefined,'forecast-record');section.append(node('h3','保存予想 '+(index+1)),node('p','保存 '+String(p.snapshot_at||'未確認').replace('T',' ').replace('+09:00',''),'scope'));
  const groups=new Map();for(const t of p.tickets||[]){const group=t.group||'買い目';if(!groups.has(group))groups.set(group,[]);groups.get(group).push(t);}
  for(const [group,tickets] of groups){section.append(node('h4',group+' '+tickets.length+'点'));const grid=node('div',undefined,'archive-tickets');tickets.forEach(t=>{const hit=p.actual_trifecta===t.buy;const item=node('div',undefined,'archive-ticket'+(hit?' hit':''));item.append(numbers(t.buy),node('span',t.stake_yen==null?'配分未確認':number(t.stake_yen)+'円','ticket-stake'),node('span',p.actual_trifecta?(hit?'的中':'不的中'):'結果待ち','ticket-status'));grid.append(item);});section.append(grid);}
  if(p.actual_trifecta)section.append(node('p','100円当たり払戻 '+(p.payout_per_100yen==null?'未確認':number(p.payout_per_100yen)+'円'),'result'));
  const meta=node('details',undefined,'opinion');meta.append(node('summary','予想版・記録の詳細'),node('p',p.strategy_version||'未確認'),node('p','金額は保存時点の参考配分です。'));section.append(meta);body.append(section);
 });card.append(body);return card;
}
function renderArchive(){
 const day=$('#archive-day').value,venue=$('#archive-venue').value,saved=$('#saved-only').checked;
 const rows=archive.races.filter(r=>r.date===day&&(!venue||r.venue===venue)&&(!saved||hasTickets(r))).sort((a,b)=>a.venue.localeCompare(b.venue,'ja')||Number(a.number)-Number(b.number));
 const pages=Math.max(1,Math.ceil(rows.length/pageSize));page=Math.min(page,pages-1);
 $('#archive-title').textContent=day?labelDate(day):'日付を選んでください';$('#archive-status').textContent=rows.length?`${number(rows.length)}レース ／ ${page*pageSize+1}〜${Math.min((page+1)*pageSize,rows.length)}件を表示`:'この条件のレースはありません';
 const target=$('#archive-races');target.replaceChildren();let venueHeading='';for(const r of rows.slice(page*pageSize,(page+1)*pageSize)){if(r.venue!==venueHeading){target.append(node('h3',r.venue,'venue-title'));venueHeading=r.venue;}target.append(raceCard(r));}
 if(!rows.length)target.append(node('p','前後の開催日を選ぶか、絞り込みを解除してください。','empty'));
 const pager=$('#archive-pagination');pager.replaceChildren();if(pages>1){const prev=button('← 前のページ',()=>{page--;renderArchive();$('#archive-title').scrollIntoView({block:'start'});});prev.disabled=page===0;const next=button('次のページ →',()=>{page++;renderArchive();$('#archive-title').scrollIntoView({block:'start'});});next.disabled=page===pages-1;pager.append(prev,node('span',`${page+1} / ${pages}`),next);}
 $('#previous-day').disabled=!dates.some(d=>d<day);$('#next-day').disabled=!dates.some(d=>d>day);
}
function changeDay(){page=0;const selected=$('#archive-venue').value;const venues=[...new Set(archive.races.filter(r=>r.date===$('#archive-day').value).map(r=>r.venue))].sort((a,b)=>a.localeCompare(b,'ja'));const select=$('#archive-venue');select.replaceChildren();for(const value of ['',...venues]){const option=node('option',value||'全レース場');option.value=value;select.append(option);}if(venues.includes(selected))select.value=selected;renderArchive();}
function setDay(day){$('#archive-day').value=day;changeDay();}
async function load(){try{const response=await fetch(raw+'public_archive.json?t='+Math.floor(Date.now()/60000),{cache:'no-store'});if(!response.ok)throw Error(response.status);archive=await response.json();dates=[...new Set(archive.races.map(r=>r.date))].filter(d=>/^\d{4}-\d{2}-\d{2}$/.test(d)).sort();if(!dates.length)throw Error('empty');const input=$('#archive-day');input.min=dates[0];input.max=dates.at(-1);
 const report=archive.learning.history_report||{},candidate=archive.learning.candidate||{};
 for(const [label,value] of [['保管レース',number(archive.races.length)+' R'],['保管期間',dates[0].replaceAll('-','/')+' 〜 '+dates.at(-1).replaceAll('-','/')],['追加学習',candidate.completed_at?'完了・候補を保存':'完了記録なし']]){const box=node('div');box.append(node('span',label),node('strong',value));$('#archive-overview').append(box);}
 $('#learning').append(node('p','中央履歴 '+number(report.historical_races||0)+'レース。選手・部署の知識は過去1〜3年を参照し、直近1年を重視。'),node('p','知識の更新 '+(report.updated_at_jst||'未確認')),node('p',candidate.completed_at?`追加学習 ${candidate.completed_at} ／ ${number(candidate.training_races)}レース。候補モデルとして保存済み。本番採用は別途検証。`:'追加モデル学習の完了記録はまだありません。'));
 input.addEventListener('change',changeDay);$('#archive-venue').addEventListener('change',()=>{page=0;renderArchive();});$('#saved-only').addEventListener('change',()=>{page=0;renderArchive();});$('#previous-day').addEventListener('click',()=>setDay(dates.filter(d=>d<input.value).at(-1)));$('#next-day').addEventListener('click',()=>setDay(dates.find(d=>d>input.value)));$('#latest-day').disabled=false;$('#latest-day').addEventListener('click',()=>setDay(dates.at(-1)));$('#reset-filters').disabled=false;$('#reset-filters').addEventListener('click',()=>{$('#archive-venue').value='';$('#saved-only').checked=false;page=0;renderArchive();});setDay(dates.at(-1));
 }catch(error){$('#archive-status').textContent='保管庫を読み込めませんでした。';$('#archive-races').replaceChildren(button('もう一度読み込む',()=>location.reload()));}}
load();
