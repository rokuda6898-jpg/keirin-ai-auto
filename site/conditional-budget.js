'use strict';
// No simulated purchase is executed. Official settlement comes from Python.
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
if (typeof module !== 'undefined' && module.exports) module.exports = {conditionalBudgetStats};
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
    function panel(r) {
      const section = node('section', undefined, 'conditional-budget');
      section.append(node('h4', '1万円配分', 'pick-label'));
      const p = r.budget;
      if (!p) {
        section.append(node('p', r.shadow ? '導入前の予想：予想時オッズ未保存のため、後付け配分はしません。' : '予想作成後に配分を判定します。', 'saved'));
        return section;
      }
      if (p.status !== 'allocated') {
        section.append(node('p', (p.status === 'infeasible' ? '見送り：' : '算出不可：') + p.reason));
        if (p.required_stake_yen) section.append(node('p', '必要最低額 ' + money(p.required_stake_yen) + ' ＞ 予算10,000円。買い目は削りません。', 'saved'));
      } else {
        section.append(node('p', '合計 ' + money(p.total_stake_yen) + ' ／ 予想時の最低想定払戻 ' + money(p.minimum_payout_yen)
          + ' ／ 最低想定収支 ' + (p.minimum_profit_yen === 0 ? '±0円' : '+' + money(p.minimum_profit_yen))));
        const scroll = node('div'); scroll.style.overflowX = 'auto';
        const table = node('table'); table.style.width = '100%'; table.style.minWidth = '440px';
        const head = node('tr');
        ['買い目', '予想時オッズ', '配分額', '想定払戻'].forEach(t => head.append(node('th', t)));
        const thead = node('thead'); thead.append(head); table.append(thead);
        const tbody = node('tbody');
        for (const a of p.allocations) {
          const tr = node('tr');
          [a.buy, Number(a.odds).toFixed(1) + '倍', money(a.stake_yen), money(a.estimated_payout_yen)].forEach(t => tr.append(node('td', t)));
          tbody.append(tr);
        }
        table.append(tbody); scroll.append(table); section.append(scroll);
        const s = r.budget_settlement || {status: 'pending'};
        const text = s.status === 'settled'
          ? '最終払戻 ' + money(s.payout_yen) + ' ／ 最終収支 ' + (s.profit_yen > 0 ? '+' : '') + money(s.profit_yen) + ' ／ 回収率 ' + pct(s.roi)
          : (labels[s.status] || '精算確認待ち') + '：最終収支・回収率は未確定です。';
        section.append(node('p', text, 'result'));
      }
      if (p.odds_captured_at_jst) section.append(node('p', 'オッズ取得 ' + p.odds_captured_at_jst.replace('T', ' ') + ' ／ ' + p.provider, 'saved'));
      section.append(node('p', '配分は予想時点で固定。その後のオッズ下落により、的中しても元本割れする場合があります。実投票ではなく仮想購入の検証です。', 'saved'));
      return section;
    }
    function update() {
      if (typeof state === 'undefined' || !state.data) return;
      const races = selected();
      const s = conditionalBudgetStats(races);
      const stats = document.querySelector('#budget-stats');
      if (!stats) return;
      stats.replaceChildren();
      const values = [
        ['配分成立', s.allocated + 'R', '1レース10,000円'],
        ['見送り', s.skipped + 'R', '全買い目での回避不成立'],
        ['算出不可', s.unavailable + 'R', '予想時オッズ不足など'],
        ['導入前・未配分', s.legacy + 'R', '後付け計算なし'],
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
      document.querySelector('#budget-scope').textContent = '仮想購入による検証。選択中の開催日・開催場・時間帯を集計します。結果待ちは回収率の分母に入れず、待機資金を別表示。公式払戻不足・返還等の保留があれば回収率は未確定（—）です。予想時オッズは配分のみに使い、最終精算には使用しません。';
      const title = document.querySelector('#performance-title');
      if (title) title.textContent = '参考比較｜元の予想・各点100円';
      const byId = new Map(races.map(r => [r.id, r]));
      for (const card of document.querySelectorAll('#races details.race')) {
        const r = byId.get(card.dataset.raceId);
        if (r && !card.querySelector('.conditional-budget')) card.querySelector('.race-body').append(panel(r));
      }
    }
    // The existing renderer replaces only #races' direct children. Observe that
    // boundary without patching shared app.js or affecting other departments.
    new MutationObserver(update).observe(document.querySelector('#races'), {childList: true});
    update();
  })();
}
