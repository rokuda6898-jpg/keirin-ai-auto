const assert=require('node:assert/strict');
const fs=require('node:fs');
const {expand,validate,status,copyText,MODEL}=require('../site/current.js');
assert.deepEqual(expand([{first:[1],second:[1,2],third:[2,3,4]},{first:[1],second:[2],third:[3]}]),['1-2-3','1-2-4']);
const data=JSON.parse(fs.readFileSync('outputs/current_live.json','utf8'));
assert.equal(data.model_id,MODEL);data.races.forEach(validate);
const sample=data.races.find(r=>r.forecast);
assert.ok(sample,'At least one actual prospective forecast must be connected');
assert.ok(copyText(sample).includes('100円'));
for(const change of [r=>r.forecast.total_yen++,r=>r.forecast.tickets.pop(),r=>r.forecast.model_id='legacy',r=>r.forecast.saved_at=new Date(r.close_at*1000+1).toISOString(),r=>r.forecast.history_through=r.date]){
 const r=structuredClone(sample);change(r);assert.throws(()=>validate(r));
}
assert.equal(status({...sample,settlement:{status:'void'}}),'返還');
assert.equal(status(sample,sample.close_at*1000+1),'締切・結果待ち');
console.log(`Current site: ${data.races.length} race rows verified; formation, deadline, policy and model guards passed.`);
