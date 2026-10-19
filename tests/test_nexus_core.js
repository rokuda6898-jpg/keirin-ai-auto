const assert=require('node:assert/strict');
const C=require('../site/nexus-core.js');
const tickets=['1-2-3','2-1-3','1-2-4','2-1-4','3-4-1','4-3-1'];
const stakes=Object.fromEntries(tickets.map((t,i)=>[t,i<4?300:500]));
const recovered={};
for(const g of C.amountGroups(stakes)){
 const ts=C.expand(g.blocks);assert.equal(ts.length,g.count);
 for(const b of g.blocks)for(const t of C.expand([b])){assert.equal(recovered[t],undefined,'money groups overlap');recovered[t]=g.amount;}
}
assert.deepEqual(recovered,stakes);
assert(C.openPodium({tickets}));assert(C.openPodium({tickets:['1-2-3','1-3-2']}));assert(!C.openPodium({tickets:['1-2-3','3-2-1']}));
const rows=[{prediction:{},settlement:{flat:{hit:true,stake:1000,returned:2100,refund:100}}},{prediction:{},settlement:{flat:{hit:false,stake:1000,returned:0}}},{prediction:{},settlement:{flat:{void:true,stake:1000,returned:1000}}},{prediction:{}},{}];
const total=C.totals(rows,'flat');assert.equal(total.settled,2);assert.equal(total.hits,1);assert.equal(total.stake,1900);assert.equal(total.returned,2000);assert.equal(total.profit,100);assert.equal(total.void,1);assert.equal(total.pending,1);assert.equal(total.unpredicted,1);assert.equal(total.roi,2000/1900);
const p={tickets:['1-2-3','2-1-3'],auto_budget:5000,allocations:{5000:{available:true,quote_break_even:true,budget:5000,minimum_return:7500,stakes:{'1-2-3':2500,'2-1-3':2500}}}};
assert.equal(C.allocation(p,'auto').key,'5000');p.allocations[5000].quote_break_even=false;assert.equal(C.allocation(p,'auto'),null);
const oldFeed={schema:1,races:[],updated_at:'2026-10-19T13:56:00+09:00'};
const allocatedFeed={...oldFeed,policy_updated_at:'2026-10-19T14:11:00+09:00'};
const newerFeed={...oldFeed,updated_at:'2026-10-19T14:16:00+09:00'};
assert.equal(C.latestFeed([oldFeed,allocatedFeed]),allocatedFeed);
assert.equal(C.latestFeed([allocatedFeed,null,newerFeed]),newerFeed);
assert.equal(C.latestFeed([null,{schema:2,races:[]}]),null);
const resultsFeed={...allocatedFeed,results_updated_at:'2026-10-19T16:04:00+09:00'};
assert.equal(C.latestFeed([allocatedFeed,resultsFeed]),resultsFeed,'new results win for the same prediction snapshot');
assert.equal(C.latestFeed([resultsFeed,newerFeed]),newerFeed,'result-only updates do not disguise an older prediction snapshot');
assert.equal(C.resultState({winners:['1-2-3'],payouts:{'1-2-3':9880}}),'confirmed');
assert.equal(C.resultState({winners:['1-2-3']}),'payout_pending');
assert.equal(C.resultState({winners:['1-2-3','2-1-3'],payouts:{'1-2-3':980}}),'payout_pending','all dead-heat payouts must be known');
assert.equal(C.resultState({result_complete:false,winners:['1-2-3'],payouts:{'1-2-3':980}}),'payout_pending','an explicit incomplete result remains pending');
assert.equal(C.resultState({cancelled:true}),'void');
assert.equal(C.resultState({}),'pending');
const incomplete={prediction:{tickets:['1-2-3']},result_complete:false,winners:['1-2-3','2-1-3'],payouts:{'1-2-3':980},settlement:{flat:{hit:true,stake:100,returned:980}}};
assert.equal(C.settlement(incomplete,'flat'),null,'incomplete payouts cannot expose a premature settlement');
assert.equal(C.totals([incomplete],'flat').settled,0);
assert.equal(C.totals([incomplete],'flat').pending,1);
// Every real forecast gets a grade; absent forecasts never get a fabricated one.
assert.equal(C.confidence(null),null);
assert.equal(C.confidence({tickets:[]}),null);
assert.equal(C.confidence({tickets:['1-2-3']}).assessed,false);
for(const [mass,grade]of [[0,'B'],[0.44999,'B'],[0.45,'A'],[0.65,'S'],[1,'S']]){
  assert.equal(C.confidence({tickets:['1-2-3'],confidence:{status:'assessed',policy_id:'selected-mass-v1',model_mass:mass}}).grade,grade);
}
assert.equal(C.confidence({tickets:['1-2-3'],confidence:{status:'assessed',policy_id:'selected-mass-v1',model_mass:NaN,grade:'S'}}).assessed,false);
const conditional={id:'r2',date:'2026-10-19',venue:'青森',number:2,start_at:1792366800,close_at:1792366500,shadow:{tickets:['1-2-3','2-1-3'],snapshot_at:'2026-10-19T07:00:00+09:00',marks:[{mark:'◎',car:'1'}],confidence:{grade:'S',status:'assessed',policy_id:'selected-mass-v1',model_mass:0.7}},budget:{status:'allocated',total_stake_yen:5000,minimum_payout_yen:6500,odds_captured_at_jst:'2026-10-19T07:01:00+09:00',allocation_frozen_at_jst:'2026-10-19T07:02:00+09:00',allocations:[{buy:'1-2-3',stake_yen:2500},{buy:'2-1-3',stake_yen:2500}]},actual:['1-2-3'],payouts:{'1-2-3':980}};
const cp=C.conditionalPrediction(conditional);
assert.deepEqual(C.expand(cp.blocks),[...conditional.shadow.tickets].sort());
assert.deepEqual(C.allocation(cp,'auto').stakes,{'1-2-3':2500,'2-1-3':2500});
assert.equal(C.confidence(cp).grade,'S');
assert.equal(C.conditionalPrediction({...conditional,shadow:{...conditional.shadow,snapshot_at:'2026-10-20T07:00:00+09:00'}}),null,'post-close forecasts cannot become a fallback');
assert.equal(C.allocation(C.conditionalPrediction({...conditional,budget:{...conditional.budget,allocation_frozen_at_jst:'2026-10-20T07:00:00+09:00'}}),'auto'),null,'post-close allocation is never displayed');
assert.equal(C.allocation(C.conditionalPrediction({...conditional,budget:{...conditional.budget,allocations:[{buy:'1-2-3',stake_yen:2500},{buy:'1-2-3',stake_yen:2500}]}}),'auto'),null,'duplicate stakes must not get collapsed and accepted');
const malformedBudget=C.conditionalPrediction({...conditional,budget:{...conditional.budget,allocations:[null,null]}});
assert.deepEqual(malformedBudget.tickets,conditional.shadow.tickets,'malformed allocation must not remove the forecast');
assert.equal(C.allocation(malformedBudget,'auto'),null);
const mainRow={race_id:'r1',date:'2026-10-19',venue:'青森',race_no:1,session:'morning',prediction:{tickets:['3-2-1']},settlement:{flat:{hit:false,stake:100,returned:0}}};
const merged=C.mergeForecastRows({date:'2026-10-19',races:[mainRow]},{schedule_date:'2026-10-19',races:[conditional]});
assert.equal(merged.length,2);assert.equal(merged[1].session,'morning');
assert.equal(merged[1].prediction,undefined,'conditional forecasts must stay out of primary predictions');
assert.equal(C.totals(merged,'flat').settled,1);assert.equal(C.totals(merged,'flat').hits,0);assert.equal(C.totals(merged,'flat').stake,100);
assert.equal(C.resultState(merged[1]),'confirmed');
assert.equal(mainRow.conditional_prediction,undefined,'raw feed is not mutated');
assert.equal(C.mergeForecastRows({date:'2026-10-19',races:[{...mainRow,race_id:'r2'}]},{races:[conditional]})[0].conditional_prediction,undefined,'identity mismatch cannot join a forecast');
const annotatedFeed={...resultsFeed,confidence_updated_at:'2026-10-19T18:30:00+09:00'};
assert.equal(C.latestFeed([resultsFeed,annotatedFeed]),annotatedFeed);
console.log('Nexus UI: amount formations, flexible first place, settlement, allocation and feed freshness passed');
