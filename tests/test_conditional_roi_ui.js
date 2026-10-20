'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const roi = require('../site/conditional-roi-ui.js');
const ui = require('../site/forecast.js');
const math = require('../site/forecast-math.js');

const close = Date.parse('2026-10-20T20:00:00+09:00'), now = close + 60000;
const stamp = minutes => new Date(close + minutes * 60000).toISOString();
const oldTickets = ['1-2-3', '1-3-2', '2-1-3', '2-3-1', '1-2-4', '1-4-2', '2-1-4', '2-4-1', '3-1-2', '3-2-1', '4-1-2', '4-2-1'];
const probabilities = [.16, .15, .14, .13, .04, .035, .03, .025, .024, .023, .022, .021];
const settle = (invested, paid, hit) => ({status: 'settled', invested_yen: invested, payout_yen: paid, profit_yen: paid - invested, roi: paid / invested, hit});
function fixture(id = '202610200001') {
  const buys = roi.SELECTED_RANKS.map(rank => oldTickets[rank - 1]), probs = roi.SELECTED_RANKS.map(rank => probabilities[rank - 1]);
  const plan = (selected, amounts) => ({status: 'allocated', total_stake_yen: 5000, allocations: selected.map((buy, i) => ({buy, stake_yen: amounts[i]}))});
  const old = {id, date: '2026-10-20', venue: '検証', number: 1, start_at: stamp(5), close_at: stamp(0),
    shadow: {tickets: [...oldTickets], snapshot_at: stamp(-120), marks: [{car: '1', mark: '◎'}]},
    riders: [1, 2, 3, 4].map(car => ({car: String(car), name: '選手' + car})), actual: ['1-2-3'], payouts: {'1-2-3': 2000}, cancelled_cars: [],
    budget: plan(oldTickets, [600, ...Array(11).fill(400)]), budget_settlement: settle(5000, 12000, true)};
  const row = {...structuredClone(old), forecast_status: 'saved',
    original_tickets: [...oldTickets], original_probabilities: [...probabilities], original_snapshot_at_jst: old.shadow.snapshot_at,
    probabilities: probs, shadow: {tickets: buys, snapshot_at: stamp(-60), marks: [{car: '1', mark: '◎'}, {car: '2', mark: '○'}],
      confidence: {status: 'assessed', policy_id: 'selected-mass-v1', grade: 'S', model_mass: probs.reduce((a, b) => a + b, 0)}},
    budget: plan(buys, [800, 800, 800, 800, 500, 500, 400, 400]), budget_settlement: settle(5000, 16000, true), flat_settlement: settle(800, 2000, true)};
  const original = {updated_at: stamp(1), races: [old]}, feed = {schema: 1, updated_at: stamp(1),
    policy: {id: roi.POLICY, activated_at: stamp(-90), backfill: false, selected_ranks: [...roi.SELECTED_RANKS]},
    model: {id: 'conditional_trifecta_v1', policy_id: roi.POLICY, ticket_count: 8}, races: [row]};
  return {old, row, original, feed};
}

test('default is eight; unavailable feed keeps official results and never inherits twelve tickets or stakes', () => {
  const {old, original} = fixture();
  const display = roi.composeFeed(original, null, undefined, now);
  assert.equal(display.display_policy, 'roi8');
  assert.equal(display.races.length, 1);
  const r = display.races[0];
  assert.equal(r.shadow, null); assert.equal(r.budget, null); assert.equal(r.budget_settlement, null);
  assert.deepEqual(ui.officialResults(r), old.actual); assert.deepEqual(r.riders, old.riders);
  assert.equal(ui.aggregate(display.races, 'uniform').roi, null);
  assert.equal(ui.aggregate(display.races, 'allocated').settled, 0);
  assert.equal(roi.copyText(r, false, math), null); assert.equal(ui.confidence(r), null);
  const legacy = roi.composeFeed(original, null, 'original12', now).races[0];
  assert.deepEqual(ui.tickets(legacy), oldTickets); assert.equal(ui.aggregate([legacy], 'uniform').roi, 2000 / 1200);
});

test('a valid saved eight keeps the exact fixed ranks, all riders, marks, confidence and official settlement', () => {
  const {original, feed, row} = fixture(), shown = roi.composeFeed(original, feed, undefined, now).races[0];
  assert.equal(shown.forecast_status, 'saved');
  assert.deepEqual(ui.tickets(shown), row.shadow.tickets); assert.equal(ui.tickets(shown).length, 8);
  assert.deepEqual(shown.shadow.marks, row.shadow.marks); assert.equal(shown.riders.length, 4);
  assert.equal(ui.confidence(shown).grade, 'S'); assert.equal(ui.baseSettlement(shown, now).invested, 800);
  assert.equal(ui.allocatedSettlement(shown, now).paid, 16000);
  assert.equal(ui.aggregate([shown], 'uniform').roi, 2.5);
  assert.equal(ui.aggregate([shown], 'allocated').roi, 3.2);
});

test('wrong selection, probabilities, source time, duplicate receipt or late freeze cannot display as saved eight', () => {
  const {original, feed} = fixture();
  for (const mutate of [
    f => {f.races[0].shadow.tickets = [...oldTickets];},
    f => {f.races[0].shadow.tickets[4] = oldTickets[4];},
    f => {f.races[0].shadow.tickets[7] = f.races[0].shadow.tickets[0];},
    f => {f.races[0].probabilities[0] += .001;},
    f => {f.races[0].original_probabilities[5] = .8;},
    f => {f.races[0].original_snapshot_at_jst = stamp(-121);},
    f => {f.races[0].shadow.snapshot_at = stamp(-.5);},
    f => {f.races[0].shadow.snapshot_at = stamp(-91);},
    f => {f.races[0].shadow.snapshot_at = stamp(5);},
    f => {f.races[0].venue = '別の場';},
    f => {f.races[0].close_at = stamp(10);},
    f => {f.races.push(structuredClone(f.races[0]));}
  ]) {
    const changed = structuredClone(feed); mutate(changed);
    const r = roi.composeFeed(original, changed, 'roi8', now).races[0];
    assert.equal(r.shadow, null); assert.equal(r.budget, null); assert.equal(r.forecast_status, 'invalid');
    assert.equal(ui.aggregate([r], 'uniform').settled, 0); assert.equal(roi.copyText(r, true, math), null);
    assert.deepEqual(ui.officialResults(r), ['1-2-3']);
  }
  const changedOriginal = structuredClone(original); changedOriginal.races[0].shadow.tickets.reverse();
  assert.equal(roi.composeFeed(changedOriginal, feed, 'roi8', now).races[0].shadow, null);
});

test('past races are marked pre-policy, while all source and new-feed races remain available', () => {
  const {original, feed} = fixture();
  feed.policy.activated_at = stamp(5); feed.races = [];
  const past = roi.composeFeed(original, feed, undefined, now).races[0];
  assert.equal(past.forecast_status, 'pre_policy'); assert.equal(roi.missingMessage(past).title, '導入前・8点予想なし');
  assert.deepEqual(past.actual, ['1-2-3']);
  const a = fixture('first'), b = fixture('second'); a.feed.races.push(b.row);
  const all = roi.composeFeed(a.original, a.feed, 'roi8', now);
  assert.equal(all.races.length, 2); assert.equal(all.races.filter(r => ui.tickets(r).length === 8).length, 2);
  const originalOnly = roi.composeFeed(a.original, a.feed, 'original12', now).races.find(r => r.id === 'second');
  assert.equal(originalOnly.shadow, null); assert.equal(originalOnly.budget, null);
});

function copiedAllocations(value) {
  const allocations = [];
  for (const line of value.split('\n')) {
    const m = line.match(/^([1-9・]+) → ([1-9・]+) → ([1-9・]+) \/ (\d+)点 各([\d,]+)円$/);
    if (!m) continue;
    const buys = math.expandFormation({first: m[1].split('・').map(Number), second: m[2].split('・').map(Number), third: m[3].split('・').map(Number)});
    assert.equal(buys.length, Number(m[4]));
    for (const buy of buys) allocations.push({buy, stake_yen: Number(m[5].replaceAll(',', ''))});
  }
  return allocations.sort((a, b) => a.buy.localeCompare(b.buy));
}
test('uniform and weighted copied formations expand to exactly the saved eight without duplicates or invented stakes', () => {
  const {original, feed} = fixture(), r = roi.composeFeed(original, feed, 'roi8', now).races[0];
  for (const weighted of [false, true]) {
    const value = roi.copyText(r, weighted, math), expanded = copiedAllocations(value);
    assert.match(value, /回収重視8点/); assert.match(value, /合計 8点/);
    assert.equal(expanded.length, 8); assert.equal(new Set(expanded.map(a => a.buy)).size, 8);
    const expected = weighted ? r.budget.allocations : r.shadow.tickets.map(buy => ({buy, stake_yen: 100}));
    assert.deepEqual(expanded, [...expected].sort((a, b) => a.buy.localeCompare(b.buy)));
    assert.equal(expanded.reduce((n, a) => n + a.stake_yen, 0), weighted ? 5000 : 800);
  }
  for (const mutate of [b => {b.allocations[0].stake_yen += 100;}, b => {b.allocations[0].buy = oldTickets[4];}, b => {b.total_stake_yen = 20000;}, b => {b.status = 'infeasible';}]) {
    const changed = structuredClone(r); mutate(changed.budget); assert.equal(roi.copyText(changed, true, math), null);
  }
});

test('S A B grades describe eight selected probabilities; missing or inconsistent confidence is unassessed', () => {
  for (const [factor, grade] of [[1, 'S'], [.8, 'A'], [.5, 'B']]) {
    const {original, feed} = fixture(), row = feed.races[0];
    row.original_probabilities = probabilities.map(p => p * factor);
    row.probabilities = roi.SELECTED_RANKS.map(rank => row.original_probabilities[rank - 1]);
    row.shadow.confidence.model_mass = row.probabilities.reduce((n, p) => n + p, 0); row.shadow.confidence.grade = grade;
    const shown = roi.composeFeed(original, feed, 'roi8', now).races[0];
    assert.equal(ui.confidence(shown).grade, grade); assert.equal(ui.confidence(shown).status, 'assessed');
    shown.shadow.confidence.model_mass = .99;
    assert.equal(ui.confidence(shown).status, 'insufficient');
  }
  const {original, feed} = fixture(); const shown = roi.composeFeed(original, feed, 'roi8', now).races[0];
  delete shown.shadow.confidence; assert.equal(ui.confidence(shown).status, 'insufficient');
});

test('whole-race refund and incomplete payout evidence hold eight-pick returns even for unaffected tickets', () => {
  const {original, feed} = fixture();
  for (const status of ['refund_pending', 'payout_pending']) {
    const changed = structuredClone(feed); changed.races[0].flat_settlement.status = status;
    changed.races[0].budget_settlement.status = status;
    changed.races[0].cancelled_cars = ['7'];
    const shown = roi.composeFeed(original, changed, 'roi8', now).races[0];
    assert.equal(ui.baseSettlement(shown, now).status, 'held'); assert.equal(ui.allocatedSettlement(shown, now).status, 'held');
    assert.equal(ui.aggregate([shown], 'uniform').settled, 0); assert.equal(ui.aggregate([shown], 'allocated').roi, null);
  }
  const changed = structuredClone(feed); changed.races[0].refund_pending = true;
  const shown = roi.composeFeed(original, changed, 'roi8', now).races[0];
  assert.equal(ui.allocatedSettlement(shown, now).status, 'held');
  const wrong = structuredClone(feed); wrong.races[0].flat_settlement.payout_yen = 9999;
  assert.equal(ui.baseSettlement(roi.composeFeed(original, wrong, 'roi8', now).races[0], now).status, 'invalid');
});

const metric = (races, hits, stake, paid) => ({races, hits, invested_yen: stake, payout_yen: paid, profit_yen: paid - stake, roi: races ? paid / stake : null, hit_rate: races ? hits / races : null});
function paired(feed, n) {
  feed.comparison = {scope: 'prospectively_saved_policy_only', flat_100: {paired_races: n, initial: metric(n, n, n * 1200, n * 2000), roi_eight: metric(n, n, n * 800, n * 2000)},
    allocation: {paired_races: n, initial: metric(n, n, n * 5000, n * 12000), roi_eight: metric(n, n, n * 5000, n * 16000)}};
  return feed;
}
test('prospective comparison keeps zero settled races as null rates and requires same-budget pairing', () => {
  const {feed} = fixture(), zero = roi.prospectiveComparison(paired(structuredClone(feed), 0));
  assert.equal(zero.allocation.paired_races, 0); assert.equal(zero.allocation.roi_eight.roi, null); assert.equal(zero.flat_100.initial.hit_rate, null);
  assert.equal(roi.prospectiveComparison(paired(structuredClone(feed), 1)).allocation.roi_eight.roi, 3.2);
  const wrong = paired(structuredClone(feed), 1); wrong.comparison.allocation.roi_eight = metric(1, 1, 6000, 16000);
  assert.equal(roi.prospectiveComparison(wrong), null);
  const wrongRate = paired(structuredClone(feed), 1); wrongRate.comparison.flat_100.roi_eight.roi = 104.72;
  assert.equal(roi.prospectiveComparison(wrongRate), null); assert.equal(roi.prospectiveComparison(null), null);
});

test('historical research values cannot enter prospective stats; the page labels evidence and retains original comparison', () => {
  const {original, feed} = fixture();
  feed.races[0].shadow = null; feed.races[0].forecast_status = 'pre_policy';
  feed.research = {roi: 1.047196709, invested_yen: 1045200, payout_yen: 1094530};
  feed.comparison = {scope: 'retrospective-research', allocation: {roi_eight: feed.research}};
  const display = roi.composeFeed(original, feed, 'roi8', now);
  assert.equal(ui.aggregate(display.races, 'uniform').roi, null); assert.equal(roi.prospectiveComparison(feed), null);
  const html = fs.readFileSync(path.join(__dirname, '../site/conditional.html'), 'utf8');
  const research = html.match(/<section id="roi-research"[\s\S]*?<\/section>/)[0];
  assert.match(research, /104\.72%/); assert.match(research, /62\.11%/); assert.match(research, /91\.26%/);
  assert.match(research, /完全未見の試験ではありません/); assert.doesNotMatch(research, /105\.22%/);
  assert.match(html, /id="performance-note"[^>]*>[^<]*研究の回収率は含みません/);
  assert.match(html, /id="prospective-comparison"[^>]*data-evidence-scope="prospective-only"/);
  assert.match(html, /従来12点・配分だけを比較/);
});
