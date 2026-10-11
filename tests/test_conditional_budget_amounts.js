'use strict';
const assert = require('node:assert/strict');
const {conditionalTicketAmounts: amounts, conditionalAmountStatus: status} = require('../site/conditional-budget.js');
const r={shadow:{tickets:['1-2-3','2-1-3']},budget:{status:'allocated',budget_yen:10000,total_stake_yen:10000,
 allocations:[{buy:'2-1-3',stake_yen:3000,odds:'20'},{buy:'1-2-3',stake_yen:7000,odds:'10'}]}};
assert.equal(amounts(r).get('1-2-3').stake_yen,7000); // Exact buy, never array index.
assert.equal(amounts(r).get('2-1-3').stake_yen,3000);
assert.equal(amounts({...r,budget:null}),null);
assert.equal(amounts({...r,budget:{...r.budget,allocations:[r.budget.allocations[0]]}}),null);
assert.equal(amounts({...r,budget:{...r.budget,allocations:[r.budget.allocations[0],r.budget.allocations[0]]}}),null);
assert.equal(amounts({...r,budget:{...r.budget,allocations:[{buy:'1-2-3',stake_yen:7000},{buy:'2-1-4',stake_yen:3000}]}}),null);
for(const n of [0,50,7001,NaN,Infinity,'7000']) {
 assert.equal(amounts({...r,budget:{...r.budget,allocations:[{buy:'1-2-3',stake_yen:n},{buy:'2-1-3',stake_yen:3000}]}}),null);
}
assert.equal(status({...r,budget:{status:'infeasible'}}),'見送り・購入なし');
assert.equal(status({...r,budget:{status:'unavailable'}}),'金額未算出');
assert.equal(status({...r,budget:null,close_at:200},100000),'初回配分の作成待ち');
assert.equal(status({...r,budget:null,close_at:100},100000),'未配分・購入記録なし');
assert.equal(status({...r,budget:null,close_at:200,actual:['1-2-3']},100000),'未配分・購入記録なし');
console.log('Per-ticket amount mapping, validation and missing-state assertions passed');
