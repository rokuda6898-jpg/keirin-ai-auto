'use strict';
// Read-only allocation display. No simulated purchase is executed here.
function conditionalBudgetStats(races) {
  const allocated = races.filter(r => r.budget?.status === 'allocated');
  const done = allocated.filter(r => r.budget_settlement?.status === 'settled');
  const blocked = allocated.filter(r => ['payout_pending', 'refund_pending', 'invalid_snapshot'].includes(r.budget_settlement?.status));
  const pending = allocated.length - done.length - blocked.length;
  const stake = done.reduce((s, r) => s + r.budget_settlement.invested_yen, 0);
  const paid = done.reduce((s, r) => s + r.budget_settlement.payout_yen, 0);
  const hits = done.filter(r => r.budget_settlement.hit).length;
  return {allocated: allocated.length, settled: done.length, blocked: blocked.length,
    pending, stake, paid, hits, hitRate: done.length ? hits / done.length : null,
    roi: done.length && !blocked.length ? paid / stake : null,
    skipped: races.filter(r => r.budget?.status === 'infeasible').length,
    unavailable: races.filter(r => r.budget?.status === 'unavailable').length,
    legacy: races.filter(r => r.shadow && !r.budget).length};
}
function conditionalTicketAmounts(r) {
  const p = r.budget, tickets = r.shadow?.tickets || [];
  if (!p || p.status !== 'allocated') return null;
  const lines = p.allocations;
  // Never show partial, duplicated, positional or default-100-yen stakes.
  if (!Array.isArray(lines) || !tickets.length || new Set(tickets).size !== tickets.length
      || lines.length !== tickets.length || new Set(lines.map(a => a.buy)).size !== tickets.length
      || p.total_stake_yen !== 10000 || p.budget_yen !== 10000
      || lines.some(a => !tickets.includes(a.buy) || !Number.isSafeInteger(a.stake_yen)
          || a.stake_yen < 100 || a.stake_yen % 100 !== 0)
      || lines.reduce((sum, a) => sum + a.stake_yen, 0) !== 10000) return null;
  return new Map(lines.map(a => [a.buy, a]));
}
function conditionalAmountStatus(r, now = Date.now()) {
  if (r.budget?.status === 'allocated') return conditionalTicketAmounts(r) ? '' : '配分不整合・使用不可';
  if (r.budget?.status === 'infeasible') return '見送り・購入なし';
  if (r.budget?.status === 'unavailable') return '金額未算出';
  if (r.actual?.length || !r.close_at || Number(r.close_at) * 1000 <= now) return '未配分・購入記録なし';
  return '初回配分の作成待ち';
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {conditionalBudgetStats, conditionalTicketAmounts, conditionalAmountStatus};
}
if (typeof document !== 'undefined' && document.body.dataset.feed === 'conditional_live.json') {
  (() => {
    const money = n => Number(n).toLocaleString('ja-JP', {maximumFractionDigits: 2}) + '円';
    const pct = n => n === null ? '—' : (100 * n).toFixed(1) + '%';
    const node = (tag, text, cls) => {
      const n = document.createElement(tag);
      if (text !== undefined) n.textContent = text;
      if (cls) n.className = cls;
      return n;
    };
    const labels = {pending: '結果待ち', payout_pending: '公式払戻の確定待ち',
      refund_pending: '欠車・返還の公式精算確認待ち', invalid_snapshot: '保存配分の不整合・精算保留',
      not_purchased: '購入対象外', settled: '公式払戻で精算済み'};
    const isDeferred = p => p?.odds_basis === 'initial_allocation_after_forecast';
    const basisLabel = p => isDeferred(p) ? '初回配分時オッズ' : '予想時オッズ';
    function stamp(p) {
      if (!p?.odds_captured_at_jst) return '';
      const when = p.odds_captured_at_jst.replace('T', ' ').replace('+09:00', '');
      return basisLabel(p) + ' ' + when + ' ／ ' + p.provider;
    }
    function inlineAmounts(card, r) {
      const amounts = conditionalTicketAmounts(r), status = conditionalAmountStatus(r);
      const withdrawn = r.budget_settlement?.status === 'refund_pending';
      for (const ticket of card.querySelectorAll('.tickets .ticket')) {
        const label = ticket.getAttribute('aria-label') || '';
        const buy = label.startsWith('買い目 ') ? label.slice(4) : '';
        const amount = amounts?.get(buy);
        let box = ticket.querySelector('.ticket-budget-amount');
        if (!box) { box = node('div', undefined, 'ticket-budget-amount'); ticket.append(box); }
        box.replaceChildren();
        if (amount) {
          box.append(node('small', withdrawn ? '保存配分・投票しない' : '購入額（配分）'),
            node('strong', money(amount.stake_yen), 'ticket-stake-yen'));
          box.append(node('small', basisLabel(r.budget) + ' ' + Number(amount.odds).toFixed(1) + '倍'));
        } else {
          box.append(node('strong', status || '金額の照合不可', 'ticket-stake-unavailable'));
        }
      }
      const tickets = card.querySelector('.tickets');
      if (!tickets || card.querySelector('.budget-inline-summary')) return;
      const summary = node('div', undefined, 'budget-inline-summary');
      summary.append(node('strong', amounts ? '購入額合計 10,000円' : status));
      summary.append(node('span', ' ／ 仮想購入の配分・自動投票なし'));
      if (r.budget?.reason && !amounts) summary.append(node('p', r.budget.reason));
      if (stamp(r.budget)) summary.append(node('p', stamp(r.budget)));
      if (isDeferred(r.budget)) summary.append(node('p', '買い目は元の予想のまま。金額は上記の初回配分時に確定し、元の予想時オッズとは区別します。'));
      tickets.before(summary);
    }
    function panel(r) {
      const section = node('section', undefined, 'conditional-budget');
      section.append(node('h4', '配分明細・結果', 'pick-label'));
      const p = r.budget, amounts = conditionalTicketAmounts(r);
      if (!p) {
        section.append(node('p', conditionalAmountStatus(r) + '。締切後の後付け計算はしません。', 'saved'));
        return section;
      }
      if (!amounts) {
        section.append(node('p', conditionalAmountStatus(r) + '：' + (p.reason || '保存配分を確認してください。')));
        if (p.required_stake_yen) section.append(node('p', '必要最低額 ' + money(p.required_stake_yen) + ' ＞ 予算10,000円。買い目は削りません。', 'saved'));
      } else {
        section.append(node('p', '合計 ' + money(p.total_stake_yen) + ' ／ 保存オッズでの最低想定払戻 ' + money(p.minimum_payout_yen)
          + ' ／ 最低想定収支 ' + (p.minimum_profit_yen === 0 ? '±0円' : '+' + money(p.minimum_profit_yen))));
        const detail = node('details', undefined, 'budget-table-details');
        detail.append(node('summary', '買い目・金額・想定払戻の一覧'));
        const scroll = node('div', undefined, 'budget-table-scroll');
        const table = node('table');
        const head = node('tr');
        ['買い目', '購入額（配分）', basisLabel(p), '想定払戻'].forEach(t => head.append(node('th', t)));
        const thead = node('thead'); thead.append(head); table.append(thead);
        const tbody = node('tbody');
        for (const buy of r.shadow.tickets) {
          const a = amounts.get(buy), tr = node('tr');
          [a.buy, money(a.stake_yen), Number(a.odds).toFixed(1) + '倍', money(a.estimated_payout_yen)].forEach(t => tr.append(node('td', t)));
          tbody.append(tr);
        }
        table.append(tbody); scroll.append(table); detail.append(scroll); section.append(detail);
        const s = r.budget_settlement || {status: 'pending'};
        const text = s.status === 'settled'
          ? '最終払戻 ' + money(s.payout_yen) + ' ／ 最終収支 ' + (s.profit_yen > 0 ? '+' : '') + money(s.profit_yen) + ' ／ 回収率 ' + pct(s.roi)
          : (labels[s.status] || '精算確認待ち') + '：最終収支・回収率は未確定です。';
        section.append(node('p', text, 'result'));
      }
      section.append(node('p', '初回配分を保存した後は変更しません。オッズ下落で的中しても元本割れする場合があります。実投票ではなく仮想購入の検証です。', 'saved'));
      return section;
    }
    function update() {
      if (typeof state === 'undefined' || !state.data) return;
      const races = selected(), s = conditionalBudgetStats(races);
      const stats = document.querySelector('#budget-stats');
      if (stats) {
        stats.replaceChildren();
        const values = [
          ['配分成立', s.allocated + 'R', '1レース10,000円'],
          ['見送り', s.skipped + 'R', '全買い目での回避不成立'],
          ['算出不可', s.unavailable + 'R', 'オッズ不足など'],
          ['未配分', s.legacy + 'R', '締切前のみ初回配分の対象'],
          ['精算完了', s.settled + 'R', '公式結果・払戻確認済み'],
          ['的中率', pct(s.hitRate), s.hits + ' / ' + s.settled + 'R・精算完了分'],
          ['結果待ち', s.pending + 'R', '資金 ' + money(s.pending * 10000)],
          ['精算保留', s.blocked + 'R', '払戻不足・返還確認など'],
          ['投資額', money(s.stake), '精算完了分・不的中も含む'],
          ['払戻額', s.blocked ? '—' : s.settled ? money(s.paid) : '—', s.blocked ? '精算保留があるため未確定' : '公式払戻 × 保存した購入口数'],
          ['最終収支', s.settled && !s.blocked ? money(s.paid - s.stake) : '—', '精算完了分'],
          ['回収率', pct(s.roi), '公式払戻額 ÷ 投資額']];
        for (const [label, value, note] of values) {
          const box = node('div', undefined, 'stat');
          box.append(node('span', label), node('strong', value), node('small', note)); stats.append(box);
        }
      }
      const scope = document.querySelector('#budget-scope');
      if (scope) scope.textContent = '仮想購入による検証。選択中の開催日・開催場・時間帯を集計します。結果待ちは分母に入れず、待機資金を別表示。公式払戻不足・返還等の保留があれば回収率は未確定（—）です。導入前の予想は、締切前に取得した初回配分時オッズを別時刻で保存。元の予想時オッズや実投票とは区別します。';
      const title = document.querySelector('#performance-title');
      if (title) title.textContent = '参考比較｜元の予想・各点100円';
      const byId = new Map(races.map(r => [r.id, r]));
      for (const card of document.querySelectorAll('#races details.race')) {
        const r = byId.get(card.dataset.raceId);
        if (!r) continue;
        inlineAmounts(card, r);
        if (!card.querySelector('.conditional-budget')) card.querySelector('.race-body').append(panel(r));
      }
    }
    // Only observe the renderer's direct replacement boundary: no feedback loop.
    const target = document.querySelector('#races');
    if (target) new MutationObserver(update).observe(target, {childList: true});
    update();
  })();
}
