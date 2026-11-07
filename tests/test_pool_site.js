const assert=require('node:assert/strict');
const fs=require('node:fs');
const {policies,expand,validate,state,copyText,rosterInfo,rosterText,withMeetingSessions,visibleRaces,selectDisplayDay}=require('../site/pools.js');
assert.deepEqual(expand([{a:[1,2],b:[1,2],c:[3,4]},{a:[3],b:[1],c:[2]}],'trio'),['1-2-3','1-2-4']);
assert.deepEqual(expand([{a:[1,2],b:[1,2]}],'exacta'),['1-2','2-1']);
{
 const race=(id,venue,cup,date,time,session,graded=false)=>({race_id:id,venue_id:venue,cup_id:cup,date,start_at:Date.parse(`${date}T${time}:00+09:00`)/1000,session,graded,forecast:{tickets:['1-2-3'],saved_at:`${date}T06:00:00+09:00`}});
 const rows=[
  race('morning-7','12','a','2026-11-08','11:45','day'),
  race('morning-1','12','a','2026-11-08','08:45','morning'),
  race('day-1','21','b','2026-11-08','10:30','morning'),
  race('day-12','21','b','2026-11-08','16:30','night'),
  race('night-1','48','c','2026-11-08','14:00','day'),
  race('night-12','48','c','2026-11-08','20:30','midnight'),
  race('midnight-1','53','d','2026-11-08','20:00','midnight'),
  race('graded-1','87','e','2026-11-08','08:30','morning',true),
  race('tomorrow-1','12','a','2026-11-09','14:30','day'),
  race('different-cup','12','f','2026-11-08','20:30','midnight'),
  {race_id:'unknown',venue_id:'99',date:'2026-11-08',session:'day'}
 ];
 const original=JSON.stringify(rows),display=withMeetingSessions(rows);
 assert.deepEqual(visibleRaces(display,'2026-11-08','morning').map(r=>r.race_id),['morning-7','morning-1']);
 assert.deepEqual(visibleRaces(display,'2026-11-08','day').map(r=>r.race_id),['day-1','day-12','unknown']);
 assert.deepEqual(visibleRaces(display,'2026-11-08','night').map(r=>r.race_id),['night-1','night-12']);
 assert.deepEqual(visibleRaces(display,'2026-11-08','graded').map(r=>r.race_id),['graded-1']);
 assert.deepEqual(visibleRaces(display,'2026-11-09','night').map(r=>r.race_id),['tomorrow-1']);
 assert.deepEqual(visibleRaces(display,'2026-11-08','midnight').map(r=>r.race_id),['midnight-1','different-cup']);
 assert.equal(visibleRaces(display,'2026-11-08','all').length,10);
 assert.equal(JSON.stringify(rows),original,'Display grouping must not rewrite source forecasts.');
 display.forEach((r,i)=>assert.equal(r.forecast,rows[i].forecast));
}
{
 const days=['2026-11-08','2026-11-07','2026-11-06'];
 assert.equal(selectDisplayDay('',undefined,'2026-11-08',days),'2026-11-08');
 assert.equal(selectDisplayDay('2026-11-07','2026-11-07','2026-11-08',days),'2026-11-08','An open current-day view follows newly published morning forecasts.');
 assert.equal(selectDisplayDay('2026-11-06','2026-11-07','2026-11-08',days),'2026-11-06','An explicitly selected historical day stays selected.');
 assert.equal(selectDisplayDay('2026-11-07','2026-11-08','2026-11-08',days),'2026-11-07');
 assert.equal(selectDisplayDay('2026-11-04','2026-11-07','2026-11-08',days),'2026-11-08');
}
for(const pool of ['trio','exacta']){
 const d=JSON.parse(fs.readFileSync(`outputs/${pool}_live.json`,'utf8'));
 assert.equal(d.model_id,policies[pool].model);assert.equal(d.pool,pool);assert.equal(d.observed.model_id,d.model_id);
 const ids=new Set();for(const r of d.races){assert.ok(!ids.has(r.race_id));ids.add(r.race_id);validate(r,pool);}
 const sample=d.races.find(r=>r.forecast);
 if(sample){
  assert.ok(copyText(sample,pool).includes(pool==='trio'?'着順不問':' → '));assert.ok(copyText(sample,pool).includes('100円'));
  for(const mutation of [r=>r.forecast.count++,r=>r.forecast.total_yen++,r=>r.forecast.tickets.pop(),r=>r.forecast.model_id='other',r=>r.forecast.pool='trifecta',r=>r.forecast.saved_at=new Date(r.close_at*1000+1).toISOString(),r=>r.forecast.history_through=r.date,r=>r.riders=[]]){
   const r=structuredClone(sample);mutation(r);assert.throws(()=>validate(r,pool));
  }
  assert.equal(state({...sample,settlement:{status:'void'}}),'返還');
 }
 const seven=d.races.find(r=>r.forecast&&r.riders.length===7&&new Set(r.forecast.tickets.flatMap(t=>t.split('-').map(Number))).size<7);
 if(seven){const r=structuredClone(seven),used=new Set(r.forecast.tickets.flatMap(t=>t.split('-').map(Number))),missing=r.riders.find(v=>!used.has(v.car)).car;r.riders=r.riders.filter(v=>v.car!==missing);assert.throws(()=>validate(r,pool),/選手/);}
 const withdrawal=structuredClone(d.races.find(r=>r.race_id==='054520261103'));
 if(withdrawal?.forecast){
  const before=JSON.stringify(withdrawal.forecast);
  withdrawal.roster_context={registered_cars:[1,2,3,4,5,6,7],active_cars:[1,3,4,5,6,7],withdrawn:[{car:2,name:'中石昌芳',prefecture:'広島'}]};
  validate(withdrawal,pool);assert.equal(rosterInfo(withdrawal).riders.length,7);assert.equal(rosterInfo(withdrawal).activeCount,6);assert.ok(rosterText(withdrawal).includes('2番車 中石昌芳：欠場'));assert.ok(rosterText(withdrawal).includes('出走6車'));
  assert.equal(JSON.stringify(withdrawal.forecast),before);
  const bad=structuredClone(withdrawal);bad.roster_context.active_cars.push(2);assert.throws(()=>validate(bad,pool),/欠場/);
  const later=structuredClone(withdrawal);later.roster_context.active_cars=later.roster_context.active_cars.filter(c=>c!==1);later.roster_context.withdrawn.push({car:1,name:'後から欠場'});validate(later,pool);assert.ok(rosterText(later).includes('予想作成時は6車'));assert.equal(rosterInfo(later).riders.filter(v=>!v.absent).length,5);assert.equal(JSON.stringify(later.forecast),before);
 }
 const page=fs.readFileSync(`site/${pool}.html`,'utf8');assert.ok(page.includes(`data-pool="${pool}"`));assert.ok(page.includes('各100円')||page.includes('1点100円'));assert.ok(page.includes('EXの補助評価は加えていません'));assert.ok(!page.includes('selection-intro'));assert.ok(!page.includes('109.55'));
 console.log(`${pool}: ${d.races.length} races, ${d.races.filter(r=>r.forecast).length} forecasts; validation passed.`);
}
assert.equal(policies.exacta.triangle,false);assert.equal(policies.trio.triangle,true);
const nav=fs.readFileSync('src/site_model_navigation.py','utf8');for(const url of ['index.html','previous.html','high-roi.html','ex.html','nexus-forward.html','graded.html','original/index.html','company.html','trio.html','exacta.html'])assert.ok(nav.includes(url));
