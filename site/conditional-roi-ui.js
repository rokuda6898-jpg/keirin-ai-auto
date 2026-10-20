/* Two independently frozen prediction feeds. Display only; never creates bets. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ConditionalROI = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const POLICY = 'conditional-roi-eight-v1';
  const SELECTED_RANKS = [1, 2, 3, 4, 9, 10, 11, 12];
  const DEFAULT_VIEW = 'roi8';
  const epoch = v => typeof v === 'number' ? v * (v < 1e12 ? 1000 : 1) : Date.parse(v);
  const validTicket = b => typeof b === 'string' && /^[1-9]-[1-9]-[1-9]$/.test(b) && new Set(b.split('-')).size === 3;
  const selected = r => Array.isArray(r?.shadow?.tickets) && r.shadow.tickets.every(validTicket)
    && new Set(r.shadow.tickets).size === r.shadow.tickets.length ? r.shadow.tickets : [];
  function validFeed(feed) {
    return feed?.schema === 1 && feed?.policy?.id === POLICY && feed?.model?.ticket_count === 8
      && feed?.model?.policy_id === POLICY && feed?.policy?.backfill === false
      && JSON.stringify(feed.policy.selected_ranks) === JSON.stringify(SELECTED_RANKS)
      && Number.isFinite(epoch(feed.policy.activated_at)) && Array.isArray(feed.races);
  }
  function identity(a, b) {
    return String(a.id) === String(b.id) && a.date === b.date && a.venue === b.venue
      && Number(a.number) === Number(b.number) && epoch(a.close_at) === epoch(b.close_at);
  }
  function validSavedEight(r, feed, now, original) {
    if (!r || r.forecast_status !== 'saved' || selected(r).length !== 8) return false;
    const source = r.original_tickets, probabilities = r.original_probabilities;
    if (!Array.isArray(source) || source.length !== 12 || source.some(b => !validTicket(b)) || new Set(source).size !== 12
      || !selected(r).every((b, i) => b === source[SELECTED_RANKS[i] - 1])
      || !Array.isArray(probabilities) || probabilities.length !== 12
      || probabilities.some((p, i) => typeof p !== 'number' || !Number.isFinite(p) || p < 0 || p > 1 || i > 0 && p > probabilities[i - 1])
      || probabilities.reduce((n, p) => n + p, 0) > 1 + 1e-12
      || !Array.isArray(r.probabilities) || r.probabilities.length !== 8
      || r.probabilities.some((p, i) => p !== probabilities[SELECTED_RANKS[i] - 1])) return false;
    if (original && (selected(original).length !== 12 || source.some((b, i) => b !== selected(original)[i])
      || epoch(r.original_snapshot_at_jst) !== epoch(original.shadow?.snapshot_at))) return false;
    const frozen = epoch(r.shadow.snapshot_at), close = epoch(r.close_at), activated = epoch(feed.policy.activated_at);
    const sourceFrozen = epoch(r.original_snapshot_at_jst);
    return Number.isFinite(frozen) && Number.isFinite(close) && Number.isFinite(sourceFrozen) && sourceFrozen <= frozen
      && frozen >= activated
      && frozen <= now && close - frozen >= 60000;
  }
  function missingStatus(row, feed, now) {
    if (['pre_policy', 'missing', 'invalid', 'expired'].includes(row?.forecast_status)) return row.forecast_status;
    if (feed && epoch(row.close_at) <= epoch(feed.policy.activated_at)) return 'pre_policy';
    return epoch(row.close_at) <= now ? 'expired' : 'missing';
  }
  function composeFeed(original, roi, view = DEFAULT_VIEW, now = Date.now()) {
    const originalRows = Array.isArray(original?.races) ? original.races : [];
    const valid = validFeed(roi), roiRows = valid ? roi.races : [];
    if (!original && !valid) return null;
    const originalIndex = new Map(), roiIndex = new Map(), keys = new Set();
    for (const r of originalRows) {if (!r?.id) continue; keys.add(String(r.id)); originalIndex.set(String(r.id), r);}
    for (const r of roiRows) {
      if (!r?.id) continue; const id = String(r.id); keys.add(id);
      roiIndex.set(id, roiIndex.has(id) ? null : r);
    }
    const races = [];
    for (const id of keys) {
      const old = originalIndex.get(id), current = roiIndex.get(id), base = old || current;
      if (!base) continue;
      if (view === 'original12') {
        races.push(old ? {...old, display_policy: 'original12'}
          : {...base, shadow: null, budget: null, budget_settlement: null,
            display_policy: 'original12', forecast_error: '従来12点のデータ取得待ち'});
        continue;
      }
      const compatible = current && (!old || identity(old, current));
      const official = compatible && Array.isArray(current.actual) && current.actual.length ? current : old || current;
      const metadata = {...base, ...(compatible ? current : {}), actual: official?.actual || [],
        payouts: official?.payouts || {}, cancelled_cars: official?.cancelled_cars || [],
        riders: current?.riders?.length && compatible ? current.riders : old?.riders || [], display_policy: 'roi8'};
      if (compatible && validSavedEight(current, roi, now, old)) {
        races.push({...metadata, shadow: current.shadow, budget: current.budget,
          budget_settlement: current.budget_settlement, forecast_status: 'saved'});
      } else {
        const status = roiIndex.has(id) && (!compatible || current?.shadow) ? 'invalid'
          : missingStatus({...metadata, forecast_status: current?.forecast_status}, valid ? roi : null, now);
        races.push({...metadata, shadow: null, budget: null, budget_settlement: null,
          forecast_status: status, forecast_error: current?.forecast_error || '',
          roi_feed_available: valid, original_available: Boolean(selected(old).length)});
      }
    }
    const active = view === 'original12' ? original : valid ? roi : original;
    return {...(active || original || roi), races, display_policy: view,
      roi_feed_available: valid, original_feed_available: Boolean(original),
      policy_activated_at: valid ? roi.policy.activated_at : null};
  }
  function missingMessage(r) {
    if (r.display_policy !== 'roi8') return {title: '予想の保存待ち', detail: r.forecast_error || '買い目が保存され次第、フォーメーションを表示します。'};
    if (r.forecast_status === 'pre_policy') return {title: '導入前・8点予想なし', detail: 'このレースの公式結果は確認できます。当時の買い目・成績は「従来12点」でご覧ください。'};
    if (r.forecast_status === 'invalid') return {title: '8点予想の記録を確認中', detail: '保存内容の一致を確認できません。「従来12点」から元の予想を確認できます。'};
    if (r.forecast_status === 'expired') return {title: '締切前の8点予想なし', detail: '締切前に保存された8点の予想記録がありません。公式結果と、従来12点の記録を確認できます。'};
    return {title: '8点予想の保存待ち', detail: r.roi_feed_available === false ? '8点のデータを取得できていません。「従来12点」は別に確認できます。' : '新しい8点予想が保存され次第、買い目と予想印を表示します。'};
  }
  function prospectiveComparison(feed) {
    if (!validFeed(feed) || feed.comparison?.scope !== 'prospectively_saved_policy_only') return null;
    const result = {};
    for (const key of ['flat_100', 'allocation']) {
      const pair = feed.comparison[key], n = pair?.paired_races;
      if (!Number.isSafeInteger(n) || n < 0) return null;
      for (const side of ['initial', 'roi_eight']) {
        const s = pair[side];
        if (!s || s.races !== n || !Number.isSafeInteger(s.hits) || s.hits < 0 || s.hits > n
          || !Number.isSafeInteger(s.invested_yen) || s.invested_yen < 0
          || !Number.isSafeInteger(s.payout_yen) || s.payout_yen < 0 || s.profit_yen !== s.payout_yen - s.invested_yen
          || n === 0 && (s.invested_yen !== 0 || s.payout_yen !== 0 || s.roi !== null || s.hit_rate !== null)
          || n > 0 && (s.invested_yen <= 0 || !Number.isFinite(s.roi) || !Number.isFinite(s.hit_rate)
            || Math.abs(s.roi - s.payout_yen / s.invested_yen) > 1e-12 || Math.abs(s.hit_rate - s.hits / n) > 1e-12)) return null;
        if (key === 'flat_100' && s.invested_yen !== n * (side === 'initial' ? 1200 : 800)) return null;
        if (key === 'allocation' && (s.invested_yen < n * 5000 || s.invested_yen > n * 10000)) return null;
      }
      if (key === 'allocation' && pair.initial.invested_yen !== pair.roi_eight.invested_yen) return null;
      result[key] = pair;
    }
    return result;
  }
  function copyText(r, weighted, math) {
    const buys = selected(r);
    if (!buys.length || r.display_policy === 'roi8' && buys.length !== 8) return null;
    let forms, total;
    if (weighted) {
      const budget = r.budget, list = budget?.allocations;
      if (budget?.status !== 'allocated' || !Array.isArray(list) || list.length !== buys.length
        || new Set(list.map(a => a?.buy)).size !== buys.length || !Number.isSafeInteger(budget.total_stake_yen)
        || budget.total_stake_yen < 5000 || budget.total_stake_yen > 10000
        || list.some(a => !a || !buys.includes(a.buy) || !Number.isSafeInteger(a.stake_yen) || a.stake_yen < 100 || a.stake_yen % 100)
        || list.reduce((n, a) => n + a.stake_yen, 0) !== budget.total_stake_yen) return null;
      forms = math.stakeFormations(list); total = budget.total_stake_yen;
    } else {forms = math.toFormations(buys).map(f => ({...f, stake_yen: 100})); total = buys.length * 100;}
    const title = r.display_policy === 'roi8' ? '回収重視8点' : '従来12点';
    const yen = n => n.toLocaleString('ja-JP') + '円';
    const lines = [r.date + ' ' + r.venue + ' ' + r.number + 'R', title + ' / ' + (weighted ? '保存配分' : '各点100円'),
      '合計 ' + buys.length + '点 / ' + yen(total)];
    for (const f of forms) lines.push(f.first.join('・') + ' → ' + f.second.join('・') + ' → ' + f.third.join('・') + ' / ' + f.count + '点 各' + yen(f.stake_yen));
    lines.push('各行を購入。同じ車番の重複を除く。');
    return lines.join('\n');
  }
  return {POLICY, SELECTED_RANKS, DEFAULT_VIEW, validFeed, composeFeed, missingMessage, prospectiveComparison, copyText};
});
