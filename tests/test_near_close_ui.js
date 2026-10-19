'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {nearCloseIndex, nearCloseRecord, nearCloseComparison} = require('../site/forecast.js');
const math = require('../site/forecast-math.js');

const close = Date.parse('2026-10-19T20:00:00+09:00');
const now = close + 3600000;
const stamp = offset => new Date(close + offset * 60000).toISOString();
function fixture(id, hit = true) {
  const selected = ['1-2-3', '2-1-3'];
  const initial = {status: 'allocated', total_stake_yen: 5000, allocations: [
    {buy: selected[0], stake_yen: 4000}, {buy: selected[1], stake_yen: 1000}
  ], odds_captured_at_jst: stamp(-120), allocation_frozen_at_jst: stamp(-119)};
  const near = {status: 'allocated', total_stake_yen: 5000, allocations: [
    {buy: selected[0], stake_yen: 3000}, {buy: selected[1], stake_yen: 2000}
  ], odds_captured_at_jst: stamp(-20), allocation_frozen_at_jst: stamp(-19)};
  const settle = paid => ({status: 'settled', invested_yen: 5000, payout_yen: hit ? paid : 0, hit});
  const race = {id, date: '2026-10-19', venue: '検証', number: 1, close_at: close / 1000,
    shadow: {tickets: selected, snapshot_at: stamp(-180)}, budget: initial, budget_settlement: settle(8000),
    actual: [hit ? selected[0] : '3-1-2'], payouts: {'1-2-3': 200, '3-1-2': 1000}};
  const receipt = {id, date: race.date, venue: race.venue, number: race.number, close_at: race.close_at, prediction_snapshot_at_jst: race.shadow.snapshot_at,
    tickets: [...selected], initial_budget: structuredClone(initial), initial_budget_settlement: settle(8000),
    budget: near, budget_settlement: settle(6000), status: 'allocated'};
  return {race, receipt};
}
const indexFor = rows => nearCloseIndex({schema: 1, policy: {id: 'conditional-near-close-v1'}, races: rows});

test('compares only the same races at the same saved budget using official payout', () => {
  const a = fixture('202610190001'), b = fixture('202610190002', false);
  const s = nearCloseComparison([a.race, b.race], indexFor([a.receipt, b.receipt]), now);
  assert.equal(s.settled, 2);
  assert.equal(s.initial.invested, 10000);
  assert.equal(s.near.invested, 10000);
  assert.equal(s.initial.roi, 0.8);
  assert.equal(s.near.roi, 0.6);
  assert.equal(s.initial.hitRate, 0.5);
  assert.equal(s.near.hitRate, 0.5);
  const reconstructed = math.stakeFormations(a.receipt.budget.allocations).flatMap(f =>
    math.expandFormation(f).map(buy => ({buy, stake_yen: f.stake_yen})));
  assert.deepEqual(reconstructed.sort((x, y) => x.buy.localeCompare(y.buy)), a.receipt.budget.allocations);
});

test('a changed snapshot, selection, budget or too-late quote cannot enter the comparison', () => {
  const {race, receipt} = fixture('202610190003');
  for (const mutate of [
    r => {r.venue = '別開催';},
    r => {r.number = 2;},
    r => {r.prediction_snapshot_at_jst = stamp(-179);},
    r => {r.tickets[1] = '1-3-2';},
    r => {r.initial_budget.allocations[0].stake_yen = 3900; r.initial_budget.allocations[1].stake_yen = 1100;},
    r => {r.budget.total_stake_yen = 5100; r.budget.allocations[0].stake_yen = 3100;},
    r => {r.budget.odds_captured_at_jst = stamp(-41);},
    r => {r.budget.allocation_frozen_at_jst = stamp(-4);},
    r => {r.budget.allocation_frozen_at_jst = stamp(1);}
  ]) {
    const changed = structuredClone(receipt); mutate(changed);
    assert.equal(nearCloseRecord(race, changed, now).status, 'invalid');
    assert.equal(nearCloseComparison([race], indexFor([changed]), now).settled, 0);
  }
  assert.equal(nearCloseRecord(race, indexFor([receipt, receipt]).get(race.id), now).status, 'invalid');
});

test('pending, infeasible and missing records stay out of both denominators', () => {
  const a = fixture('202610190004'), b = fixture('202610190005'), c = fixture('202610190006'), d = fixture('202610190008');
  a.receipt.budget_settlement = {status: 'pending'};
  b.receipt.status = 'infeasible'; b.receipt.budget = {status: 'infeasible'};
  d.receipt.status = 'missed'; d.receipt.budget = null;
  const s = nearCloseComparison([a.race, b.race, c.race, d.race], indexFor([a.receipt, b.receipt, d.receipt]), now);
  assert.equal(s.pending, 1); assert.equal(s.infeasible, 1); assert.equal(s.unavailable, 1);
  assert.equal(s.unregistered, 1); assert.equal(s.registered, 3);
  assert.equal(s.settled, 0); assert.equal(s.initial.roi, null); assert.equal(s.near.hitRate, null);
  assert.equal(s.initial.net, null);
  assert.equal(nearCloseComparison([], new Map(), now).near.roi, null);
  assert.equal(nearCloseIndex({schema: 1, policy: {id: 'unknown'}, races: [a.receipt]}).size, 0);
});

test('a saved settlement inconsistent with official payout is held from both sides', () => {
  const {race, receipt} = fixture('202610190007');
  receipt.budget_settlement.payout_yen = 9000;
  const s = nearCloseComparison([race], indexFor([receipt]), now);
  assert.equal(s.settled, 0); assert.equal(s.excluded, 1); assert.equal(s.initial.roi, null);
  const withdrawn = structuredClone(race); withdrawn.cancelled_cars = ['2'];
  assert.equal(nearCloseComparison([withdrawn], indexFor([receipt]), now).settled, 0);
});
