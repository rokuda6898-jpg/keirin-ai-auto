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
console.log('Nexus UI: amount formations, flexible first place, settlement, allocation and feed freshness passed');
