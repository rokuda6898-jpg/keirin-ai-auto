'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const ui = require('../site/nexus-forward.js');

function plan(second) {
  const blocks = [{first: [1], second, third: [2, 3, 4, 5, 6, 7]}];
  const tickets = [];
  for (const b of second) for (const c of blocks[0].third) if (b !== c) tickets.push(`1-${b}-${c}`);
  blocks[0].count = tickets.length;
  return {count: tickets.length, blocks, tickets, captured_at: '2026-10-20T09:00:00+09:00', allocations: {}, auto_budget: null};
}
function addAllocation(p, amount = 5000) {
  const stakes = Object.fromEntries(p.tickets.map(t => [t, 100]));
  stakes[p.tickets[0]] += amount - p.tickets.length * 100;
  p.allocations[amount] = {available: true, budget: amount, stakes, minimum_return: amount, quote_break_even: true};
  p.auto_budget = amount;
}
function cash(p, payouts, key) {
  const stakes = key === 'flat' ? Object.fromEntries(p.tickets.map(t => [t, 100])) : p.allocations[key].stakes;
  const stake = Object.values(stakes).reduce((a, b) => a + b, 0);
  const returned = Object.entries(stakes).reduce((a, [t, n]) => a + n / 100 * (payouts[t] || 0), 0);
  return {stake, returned, hit: returned > 0, void: false, refund: 0};
}
function row() {
  const baseline = plan([2, 3, 4]), candidate = plan([2, 3]);
  candidate.captured_at = '2026-10-20T10:00:00+09:00';
  const value = {race_id: '017120261020', date: '2026-10-20', close_at: '2026-10-20T12:00:00+09:00', saved_at: '2026-10-20T10:00:00+09:00', status: 'settled', baseline, candidate, result: {result_complete: true, winners: ['1-2-3'], payouts: {'1-2-3': 1200}, refunded: [], refund_all: false, cancelled: false}, settlement: {}};
  for (const side of ['baseline', 'candidate']) {addAllocation(value[side]); value.settlement[side] = {flat: cash(value[side], value.result.payouts, 'flat'), auto: cash(value[side], value.result.payouts, 5000), '5000': cash(value[side], value.result.payouts, 5000)};}
  return value;
}

test('formation expansion preserves overlap once and rejects a displayed count mismatch', () => {
  const p = plan([2, 3]); p.blocks.push({first: [1], second: [2], third: [3], count: 1});
  assert.equal(ui.planTickets(p).length, 10);
  p.blocks[1].count = 2;
  assert.deepEqual(ui.planTickets(p), []);
});
test('after-close or unbound forecast is not eligible for displayed performance', () => {
  const r = row(); r.saved_at = r.close_at;
  assert.equal(ui.hasSnapshot(r), false); assert.equal(ui.comparison([r]).races, 0);
  delete r.saved_at; assert.equal(ui.hasSnapshot(r), false);
});
test('zero completed common races render null rates, never a invented 0% result', () => {
  const r = row(); r.status = 'result_pending';
  const result = ui.comparison([r]);
  assert.equal(result.races, 0); assert.equal(result.candidate.roi, null); assert.equal(result.baseline.hit_rate, null); assert.equal(result.pending, 1);
});
test('flat comparison uses the same race and each published ticket count', () => {
  const result = ui.comparison([row()]);
  assert.equal(result.races, 1); assert.equal(result.candidate.stake_yen, 1000); assert.equal(result.baseline.stake_yen, 1500);
  assert.equal(result.candidate.roi, 1.2); assert.equal(result.baseline.roi, .8);
});
test('a missing or malformed stake excludes both sides of the allocation comparison', () => {
  const r = row(); delete r.baseline.allocations[5000].stakes['1-2-3'];
  assert.equal(ui.allocation(r.baseline), null); assert.equal(ui.comparison([r], 'allocated').races, 0);
  assert.equal(ui.comparison([r], 'flat').races, 1);
});
test('different auto budgets are not represented as a same-budget comparison', () => {
  const r = row(); addAllocation(r.baseline, 6000); r.settlement.baseline.auto = cash(r.baseline, r.result.payouts, 6000);
  assert.equal(ui.comparison([r], 'allocated').races, 0);
  assert.equal(ui.comparison([r], 'allocated', '5000').races, 1);
});
test('manual 5100 yen remains selectable even when no common auto allocation exists', () => {
  const r = row(); addAllocation(r.candidate, 5100); r.candidate.auto_budget = null;
  assert.equal(ui.allocation(r.candidate), null); assert.equal(ui.allocation(r.candidate, '5100').budget, 5100);
  assert.equal(ui.comparison([r], 'allocated', '5100').races, 0);
});
test('dead heat adds official payouts and inconsistent recorded returns are withheld', () => {
  const r = row(); r.result.winners.push('1-3-2'); r.result.payouts['1-3-2'] = 900;
  for (const side of ['baseline', 'candidate']) r.settlement[side].flat = cash(r[side], r.result.payouts, 'flat');
  assert.equal(ui.comparison([r]).candidate.payout_yen, 2100);
  r.settlement.candidate.flat.returned = 1200;
  assert.equal(ui.comparison([r]).races, 0);
});
test('refund and cancellation records never enter hits or ROI', () => {
  const r = row(); r.result.refunded = ['1-2-3']; r.status = 'excluded';
  const result = ui.comparison([r]); assert.equal(result.races, 0); assert.equal(result.excluded, 1);
});
test('duplicate feed identity and a different policy are rejected', () => {
  const r = row();
  assert.throws(() => ui.validateFeed({schema: 1, policy: {id: 'nexus-forward-ten-v1'}, races: [r, r]}));
  assert.throws(() => ui.validateFeed({schema: 1, policy: {id: 'different-policy'}, races: [r]}));
});
test('automatic selection advances after close and prioritizes a saved upcoming plan', () => {
  const earlier = row(), waiting = row(), later = row();
  waiting.race_id = '027120261020'; waiting.close_at = '2026-10-20T13:00:00+09:00'; delete waiting.candidate; delete waiting.baseline;
  later.race_id = '037120261020'; later.close_at = '2026-10-20T14:00:00+09:00';
  const rows = [earlier, waiting, later];
  assert.equal(ui.meetingSelection(rows, undefined, Date.parse('2026-10-20T11:00:00+09:00')), earlier);
  assert.equal(ui.meetingSelection(rows, undefined, Date.parse('2026-10-20T12:01:00+09:00')), later);
  assert.equal(ui.meetingSelection(rows, earlier.race_id, Date.parse('2026-10-20T12:01:00+09:00')), earlier);
});
test('selection falls back to an open missing record or the last race without hiding races', () => {
  const r = row(), missing = row(); missing.race_id = '027120261020'; missing.close_at = '2026-10-20T13:00:00+09:00'; delete missing.candidate;
  assert.equal(ui.meetingSelection([r, missing], undefined, Date.parse('2026-10-20T12:01:00+09:00')), missing);
  assert.equal(ui.meetingSelection([r, missing], undefined, Date.parse('2026-10-20T14:00:00+09:00')), missing);
  assert.equal(ui.meetingSelection([], undefined), undefined);
});
test('replay provenance keeps original input, prediction, publication and current save distinct', () => {
  const r = row(); r.baseline.input_captured_at = '2026-10-20T08:55:00+09:00';
  const normal = ui.provenance(r);
  assert.equal(normal.replay, null); assert.equal(normal.records.length, 3);
  r.source_replay = {evidence_kind: 'preclose_replay_of_published_forecast', input_captured_at: '2026-10-20T08:55:00+09:00', original_prediction_at: '2026-10-20T09:00:00+09:00', original_published_at: '2026-10-20T09:02:00+09:00', replayed_at: '2026-10-20T09:58:00+09:00', original_full_q_observed: false};
  const values = Object.fromEntries(ui.provenance(r).records.map(v => [v.key, v.at]));
  assert.deepEqual(values, {input: r.source_replay.input_captured_at, prediction: r.source_replay.original_prediction_at, publication: r.source_replay.original_published_at, replay: r.source_replay.replayed_at, ten: r.saved_at});
});
