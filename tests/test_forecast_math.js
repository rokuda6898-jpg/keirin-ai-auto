const assert = require('node:assert/strict');
const test = require('node:test');
const math = require('../site/forecast-math.js');

function checkExact(tickets) {
  const formations = math.toFormations(tickets);
  const expanded = formations.flatMap(math.expandFormation);
  assert.equal(expanded.length, new Set(expanded).size, 'formation rows overlap');
  assert.deepEqual(expanded.sort(), [...new Set(tickets)].sort(), 'selected tickets changed');
  for (const form of formations) assert.equal(form.count, math.expandFormation(form).length);
  return formations;
}

test('a compact formation excludes repeated racers and preserves all selections', () => {
  const tickets = ['1-2-3', '1-2-4', '1-3-2', '1-3-4'];
  assert.equal(checkExact(tickets).length, 1);
  assert.equal(checkExact(['1-2-3', '3-2-1']).length, 1);
});

test('a missing corner never becomes an extra ticket', () => {
  assert.ok(checkExact(['1-2-3', '1-2-4', '1-5-3']).length > 1);
  checkExact(['9-1-2', '9-2-1', '8-1-9', '1-8-9', '7-9-8']);
  assert.deepEqual(checkExact([]), []);
});

test('irregular seven- and nine-racer selections remain an exact disjoint partition', () => {
  for (const racers of [7, 9]) {
    const universe = [];
    for (let a = 1; a <= racers; a++) for (let b = 1; b <= racers; b++) {
      for (let c = 1; c <= racers; c++) {
        if (a !== b && a !== c && b !== c) universe.push(`${a}-${b}-${c}`);
      }
    }
    for (let shift = 0; shift < 8; shift++) {
      checkExact(Array.from({ length: 12 }, (_, i) => universe[(i * 29 + shift * 13) % universe.length]));
    }
  }
});

test('stake groups preserve the exact per-ticket amount and total investment', () => {
  const allocations = [
    { buy: '1-2-3', stake_yen: 500 }, { buy: '1-2-4', stake_yen: 500 },
    { buy: '1-3-2', stake_yen: 100 }, { buy: '9-8-7', stake_yen: 1000 }
  ];
  const forms = math.stakeFormations(allocations);
  const rebuilt = forms.flatMap(f => math.expandFormation(f).map(buy => ({ buy, stake_yen: f.stake_yen })));
  const sorted = a => [...a].sort((x, y) => x.buy.localeCompare(y.buy));
  assert.deepEqual(sorted(rebuilt), sorted(allocations));
  assert.equal(forms.reduce((sum, f) => sum + f.total_stake_yen, 0), 2100);
});

test('invalid selections or amounts fail visibly', () => {
  assert.throws(() => math.toFormations(['1-1-2']), TypeError);
  assert.throws(() => math.toFormations(['10-2-3']), TypeError);
  assert.throws(() => math.stakeFormations([{ buy: '1-2-3', stake_yen: 150 }]), TypeError);
  assert.throws(() => math.stakeFormations([
    { buy: '1-2-3', stake_yen: 100 }, { buy: '1-2-3', stake_yen: 200 }
  ]), TypeError);
});
