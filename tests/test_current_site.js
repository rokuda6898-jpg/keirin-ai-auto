const assert=require('node:assert/strict');
const fs=require('node:fs');
const {expand,validate,status,copyText,officialRecord,archivedOfficialRows,attachOfficialResults,MODEL,MODEL_LABELS}=require('../site/current.js');
assert.deepEqual(expand([{first:[1],second:[1,2],third:[2,3,4]},{first:[1],second:[2],third:[3]}]),['1-2-3','1-2-4']);
const data=JSON.parse(fs.readFileSync('outputs/current_live.json','utf8'));
assert.equal(data.model_id,MODEL);data.races.forEach(validate);
assert.equal(data.observed.model_id,MODEL);
const sample=data.races.find(r=>r.forecast);
// A fully cancelled day or a first launch after closing may have no forecast.
if(sample){
assert.ok(copyText(sample).includes('100円'));
for(const change of [r=>r.forecast.total_yen++,r=>r.forecast.tickets.pop(),r=>r.forecast.model_id='legacy',r=>r.forecast.saved_at=new Date(r.close_at*1000+1).toISOString(),r=>r.forecast.history_through=r.date]){
 const r=structuredClone(sample);change(r);assert.throws(()=>validate(r));
}
assert.equal(status({...sample,settlement:{status:'void'}}),'返還');
assert.equal(status({...sample,settlement:null},sample.close_at*1000+1),'締切・結果待ち');
}
console.log(`Current site: ${data.races.length} race rows verified; formation, deadline, policy and model guards passed.`);

if(sample){for(const id of Object.keys(MODEL_LABELS)){const r=structuredClone(sample);r.forecast.model_id=id;validate(r);assert.ok(copyText(r).includes(MODEL_LABELS[id]));}}

const liveFixture={observed:{model_id:MODEL,races:2,hits:1,stake:1600,returned:8000,HIGH_hits:0,pending:1},races:[
 {race_id:'010120261103',performance_eligible:true,forecast:{model_id:MODEL,tickets:['6-1-5','1-2-3'],total_yen:200}},
 {race_id:'020120261103',forecast:null}
]};
const sourceFixture=[{race_id:'010120261103',official_result_available:true,actual_trifecta_buys:['6-1-5'],payouts_trifecta_json:'{"6-1-5":11480}',result_observed_at_jst:'2026-11-03T14:40:00+09:00'},
 {race_id:'020120261103',official_result_available:true,actual_trifecta_buys:['2-4-7'],payouts_trifecta_json:'{"2-4-7":4670}',result_observed_at_jst:'2026-11-03T14:41:00+09:00'}];
assert.equal(attachOfficialResults(liveFixture,sourceFixture),2);
assert.equal(liveFixture.races[0].settlement.status,'hit');
assert.equal(liveFixture.races[0].settlement.returned,11480);
assert.equal(liveFixture.races[0].settlement.stake,200);
assert.equal(liveFixture.races[0].forecast.total_yen,200,'frozen forecast never changes');
assert.equal(liveFixture.observed.races,3);
assert.equal(liveFixture.observed.hits,2);
assert.equal(liveFixture.observed.HIGH_hits,1);
assert.equal(liveFixture.observed.pending,0);
assert.equal(status(liveFixture.races[1]),'結果確定','official finish visible without forecast');
assert.equal(liveFixture.races[1].settlement,undefined,'no fabricated bet');
assert.equal(attachOfficialResults(liveFixture,sourceFixture),0,'same results must not double-count');
assert.equal(liveFixture.observed.races,3);
assert.equal(officialRecord({official_result_available:true,actual_trifecta_buys:['1-2-3','2-1-3'],payouts_trifecta_json:'{"1-2-3":980}'}),null,'tied race requires every official payout');
assert.equal(officialRecord({official_result_available:false,actual_trifecta:'1-2-3',payouts_trifecta_json:'{"1-2-3":980}'}),null);
console.log('Official result overlay: payouts, settlement, homepage totals, missing predictions and idempotency passed');

const archive=archivedOfficialRows([
 {race_id:'010120261102',result_complete:true,winners:['2-1-3'],payouts:{'2-1-3':20980},results_updated_at:'2026-11-02T11:01:00+09:00'},
 {race_id:'020120261102',result_complete:false,winners:['3-2-1'],payouts:{'3-2-1':980}},
 {race_id:'030120261102',result_complete:true,winners:['4-5-6','5-4-6'],payouts:{'4-5-6':890}}]);
assert.equal(archive.length,1,'only identity-complete historical payouts may be shown');
assert.deepEqual(officialRecord(archive[0]).winning_tickets,['2-1-3']);
const previousDay={observed:{model_id:MODEL,races:0},races:[{race_id:'010120261102',forecast:null}]};
assert.equal(attachOfficialResults(previousDay,archive),1);
assert.equal(status(previousDay.races[0]),'結果確定');
assert.equal(attachOfficialResults(previousDay,archive),0,'reopening yesterday does not count results twice');
