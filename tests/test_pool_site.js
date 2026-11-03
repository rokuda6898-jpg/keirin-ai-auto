const assert=require('node:assert/strict');
const fs=require('node:fs');
const {policies,expand,validate,state,copyText}=require('../site/pools.js');
assert.deepEqual(expand([{a:[1,2],b:[1,2],c:[3,4]},{a:[3],b:[1],c:[2]}],'trio'),['1-2-3','1-2-4']);
assert.deepEqual(expand([{a:[1,2],b:[1,2]}],'exacta'),['1-2','2-1']);
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
 const page=fs.readFileSync(`site/${pool}.html`,'utf8');assert.ok(page.includes(`data-pool="${pool}"`));assert.ok(page.includes('各100円')||page.includes('1点100円'));assert.ok(page.includes('EXの補助評価は加えていません'));assert.ok(!page.includes('selection-intro'));assert.ok(!page.includes('109.55'));
 console.log(`${pool}: ${d.races.length} races, ${d.races.filter(r=>r.forecast).length} forecasts; validation passed.`);
}
assert.equal(policies.exacta.triangle,false);assert.equal(policies.trio.triangle,true);
const nav=fs.readFileSync('src/site_model_navigation.py','utf8');for(const url of ['index.html','previous.html','high-roi.html','ex.html','nexus-forward.html','graded.html','original/index.html','company.html','trio.html','exacta.html'])assert.ok(nav.includes(url));
