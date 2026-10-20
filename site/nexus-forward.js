'use strict';

// Render frozen preclose plans. This page never chooses a ticket or an amount.
(() => {
  const POLICY = 'nexus-forward-ten-v1';
  const BANDS = [['all', 'すべて'], ['morning', 'モーニング'], ['day', 'デイ'], ['night', 'ナイト'], ['midnight', 'ミッドナイト']];
  const integer = n => Number.isSafeInteger(n) && n >= 0;
  const ticket = t => typeof t === 'string' && /^[1-9]-[1-9]-[1-9]$/.test(t) && new Set(t.split('-')).size === 3;
  const epoch = value => typeof value === 'string' && /(?:Z|[+-]\d\d:\d\d)$/.test(value) ? Date.parse(value) : NaN;
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const dateLabel = day => /^\d{4}-\d\d-\d\d$/.test(day || '') ? new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', month: 'numeric', day: 'numeric', weekday: 'short'}).format(new Date(day + 'T12:00:00+09:00')) : '日付未確認';
  const time = value => Number.isFinite(epoch(value)) ? new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', hour: '2-digit', minute: '2-digit', hour12: false}).format(epoch(value)) : '—';
  const stamp = value => Number.isFinite(epoch(value)) ? new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false}).format(epoch(value)) : '未確認';
  const yen = n => integer(n) ? n.toLocaleString('ja-JP') + '円' : '—';
  const profit = n => Number.isSafeInteger(n) ? (n > 0 ? '+' : n < 0 ? '−' : '±') + Math.abs(n).toLocaleString('ja-JP') + '円' : '—';
  const pct = n => typeof n === 'number' && Number.isFinite(n) ? (n * 100).toFixed(2) + '%' : '—';
  const bandLabel = key => BANDS.find(item => item[0] === key)?.[1] || '時間帯未確認';
  const car = n => /^[1-9]$/.test(String(n)) ? `<span class="car car-${n}">${n}</span>` : '';
  const cars = values => `<div class="car-set">${values.map(car).join('')}</div>`;

  function planTickets(plan) {
    const values = plan?.tickets;
    if (!Array.isArray(values) || !values.length || values.some(t => !ticket(t)) || new Set(values).size !== values.length || plan.count !== values.length) return [];
    if (!Array.isArray(plan.blocks) || !plan.blocks.length || plan.blocks.length > 4) return [];
    const expanded = new Set();
    for (const b of plan.blocks) {
      if (!b || ['first', 'second', 'third'].some(k => !Array.isArray(b[k]) || !b[k].length || new Set(b[k]).size !== b[k].length || b[k].some(n => !Number.isInteger(n) || n < 1 || n > 9))) return [];
      const part = new Set();
      for (const a of b.first) for (const c of b.second) for (const d of b.third) if (new Set([a, c, d]).size === 3) part.add([a, c, d].join('-'));
      if (!part.size || (b.count !== undefined && b.count !== part.size)) return [];
      part.forEach(t => expanded.add(t));
    }
    return expanded.size === values.length && values.every(t => expanded.has(t)) ? values : [];
  }

  function hasSnapshot(row) {
    const count = planTickets(row?.candidate).length;
    const saved = epoch(row?.saved_at), close = epoch(row?.close_at), created = epoch(row?.candidate?.captured_at), original = epoch(row?.baseline?.captured_at);
    return count === 10 && planTickets(row.baseline).length > 0 && Number.isFinite(saved) && Number.isFinite(close) && Number.isFinite(created) && Number.isFinite(original) && original <= saved && created <= saved && saved < close;
  }

  function allocation(plan, selection = 'auto') {
    const chosen = selection === 'auto' ? plan?.auto_budget : Number(selection);
    const amounts = plan?.allocations?.[String(chosen)], selected = planTickets(plan);
    if (!integer(chosen) || chosen < 5000 || chosen > 10000 || chosen % 100 || amounts?.available !== true || amounts.budget !== chosen || !amounts.stakes || typeof amounts.stakes !== 'object' || !selected.length) return null;
    const entries = Object.entries(amounts.stakes);
    if (entries.length !== selected.length || entries.some(([t, stake]) => !selected.includes(t) || !integer(stake) || stake < 100 || stake % 100) || entries.reduce((sum, [, stake]) => sum + stake, 0) !== chosen) return null;
    return amounts;
  }

  function official(row) {
    const result = row?.result, winners = result?.winners, payouts = result?.payouts;
    if (result?.result_complete !== true || !Array.isArray(winners) || !winners.length || !payouts || typeof payouts !== 'object' || winners.some(t => !ticket(t)) || new Set(winners).size !== winners.length || Object.keys(payouts).length !== winners.length || winners.some(t => !integer(payouts[t]) || payouts[t] <= 0)) return null;
    return result;
  }

  function settlement(row, side, method = 'flat', budget = 'auto') {
    if (row?.status !== 'settled' || !hasSnapshot(row)) return null;
    const result = official(row);
    if (!result || result.refund_all || result.cancelled || (Array.isArray(result.refunded) && result.refunded.length)) return null;
    const p = row[side], amounts = method === 'flat' ? null : allocation(p, budget);
    if (method !== 'flat' && !amounts) return null;
    const stakes = method === 'flat' ? Object.fromEntries(planTickets(p).map(t => [t, 100])) : amounts.stakes;
    const key = method === 'flat' ? 'flat' : budget === 'auto' ? 'auto' : String(budget);
    const saved = row.settlement?.[side]?.[key];
    const stake = Object.values(stakes).reduce((s, n) => s + n, 0);
    const returned = Object.entries(stakes).reduce((s, [t, n]) => s + n / 100 * (result.payouts[t] || 0), 0);
    if (!saved || !integer(saved.stake) || !integer(saved.returned) || saved.stake !== stake || saved.returned !== returned || saved.hit !== (returned > 0) || saved.void || (saved.refund || 0) !== 0) return null;
    return {stake, returned, profit: returned - stake, hit: returned > 0, budget: amounts?.budget || null};
  }

  function comparison(rows, method = 'flat', budget = 'auto') {
    const result = {race_ids: [], races: 0, baseline: {hits: 0, stake_yen: 0, payout_yen: 0}, candidate: {hits: 0, stake_yen: 0, payout_yen: 0}, saved: 0, pending: 0, excluded: 0, unavailable: 0};
    for (const row of rows) {
      if (hasSnapshot(row)) result.saved++;
      if (['excluded', 'invalid_result'].includes(row.status)) result.excluded++;
      if (hasSnapshot(row) && ['saved', 'result_pending'].includes(row.status)) result.pending++;
      const b = settlement(row, 'baseline', method, budget), c = settlement(row, 'candidate', method, budget);
      if (!b || !c || (method !== 'flat' && b.budget !== c.budget)) {if (row.status === 'settled' && hasSnapshot(row)) result.unavailable++; continue;}
      result.race_ids.push(row.race_id); result.races++;
      for (const [side, data] of [['baseline', b], ['candidate', c]]) {result[side].hits += Number(data.hit); result[side].stake_yen += data.stake; result[side].payout_yen += data.returned;}
    }
    for (const side of ['baseline', 'candidate']) Object.assign(result[side], {races: result.races, hit_rate: result.races ? result[side].hits / result.races : null, roi: result[side].stake_yen ? result[side].payout_yen / result[side].stake_yen : null, profit_yen: result[side].payout_yen - result[side].stake_yen});
    return result;
  }

  function validateFeed(feed) {
    if (feed?.schema !== 1 || feed?.policy?.id !== POLICY || !Array.isArray(feed.races)) throw new Error('保存データの形式を確認できません。');
    const ids = new Set();
    for (const row of feed.races) {
      if (!row || !/^\d{12}$/.test(row.race_id) || !/^\d{4}-\d\d-\d\d$/.test(row.date) || ids.has(row.race_id)) throw new Error('レースの識別情報を確認できません。');
      ids.add(row.race_id);
    }
    return feed;
  }

  const math = {planTickets, hasSnapshot, allocation, official, settlement, comparison, validateFeed};
  if (typeof module !== 'undefined' && module.exports) module.exports = math;
  if (typeof document === 'undefined') return;

  const byId = id => document.getElementById(id);
  const state = {feed: null, day: '', band: 'all', venue: 'all', budget: 'auto', view: location.hash === '#results' ? 'results' : 'forecast', method: 'flat', selected: new Map(), loading: false, error: ''};

  function scopeRows(ignoreVenue = false, ignoreBand = false) {
    return (state.feed?.races || []).filter(r => (state.day === 'all' || r.date === state.day) && (ignoreVenue || state.venue === 'all' || r.venue === state.venue) && (ignoreBand || state.band === 'all' || r.session === state.band)).sort((a, b) => a.date.localeCompare(b.date) || epoch(a.start_at) - epoch(b.start_at) || a.race_id.localeCompare(b.race_id));
  }

  function status(row) {
    if (row.status === 'excluded') return {label: '返還・中止', css: 'excluded'};
    if (row.status === 'invalid_result') return {label: '精算確認中', css: 'excluded'};
    if (!hasSnapshot(row)) return {label: epoch(row.close_at) > Date.now() ? '予想待ち' : '事前記録なし', css: ''};
    if (row.status === 'settled') {
      const value = settlement(row, 'candidate');
      return value ? {label: value.hit ? '的中' : '不的中', css: value.hit ? 'hit' : 'miss'} : {label: '精算確認中', css: 'excluded'};
    }
    return epoch(row.close_at) > Date.now() ? {label: '締切前', css: 'live'} : {label: '結果待ち', css: ''};
  }

  function grade(plan) {
    const c = plan?.confidence;
    if (!['S', 'A', 'B'].includes(c?.grade) || c.status !== 'assessed' || typeof c.model_mass !== 'number' || !Number.isFinite(c.model_mass)) return '<span class="grade">自信度 未評価</span>';
    return `<span class="grade grade-${c.grade}">自信度 <b>${c.grade}</b></span>`;
  }

  function formation(plan, label) {
    const tickets = planTickets(plan);
    if (!tickets.length) return '<p class="missing">保存フォーメーションの整合性を確認中です。</p>';
    let sum = 0;
    const rows = plan.blocks.map(b => {
      const expanded = new Set();
      for (const a of b.first) for (const c of b.second) for (const d of b.third) if (new Set([a, c, d]).size === 3) expanded.add([a, c, d].join('-'));
      sum += expanded.size;
      return `<tr><td>${cars(b.first)}</td><td>${cars(b.second)}</td><td>${cars(b.third)}</td><td>${expanded.size}点</td></tr>`;
    }).join('');
    const dedup = sum > tickets.length ? `行をまたぐ重複${sum - tickets.length}点も1回だけ購入。` : '';
    return `<div class="formation-box"><div class="formation-caption"><h5>${escape(label)}</h5><strong>${tickets.length}点</strong></div><table class="formation-table" aria-label="${escape(label)}"><thead><tr><th scope="col">1着</th><th scope="col">2着</th><th scope="col">3着</th><th scope="col">点数</th></tr></thead><tbody>${rows}</tbody></table><p class="formation-note">同じ車番の重複を除外。${dedup}合計${tickets.length}点・各100円なら${yen(tickets.length * 100)}。</p></div>`;
  }

  function resultCash(row, side, method, budget) {
    const value = settlement(row, side, method, budget);
    if (!value) return '';
    return `<div class="cash-result"><span>払戻 <b>${yen(value.returned)}</b></span><span>収支 <b class="${value.profit >= 0 ? 'positive' : 'negative'}">${profit(value.profit)}</b></span></div>`;
  }

  function allocatedCard(row, side) {
    const p = row[side], a = allocation(p, state.budget), amount = state.budget === 'auto' ? p?.auto_budget : Number(state.budget);
    if (!a) {
      const saved = p?.allocations?.[String(amount)];
      const reason = saved?.reason || row.allocation_reason || '締切前の配分を保存できていません。';
      const feasible = Object.keys(p?.allocations || {}).map(Number).filter(n => allocation(p, String(n))).sort((x, y) => x - y);
      const alternative = side === 'candidate' && feasible.length ? `<p>10点案は${yen(feasible[0])}で配分が成立しています。</p><button type="button" class="allocation-choice" data-budget-recommend="${feasible[0]}">10点案を${yen(feasible[0])}配分で表示 →</button>` : '';
      return `<article class="purchase-card"><h5>${state.budget === 'auto' ? '比較用の自動配分 · 共通予算' : '予算を配分'}</h5><div class="purchase-total"><strong>${state.budget === 'auto' ? '共通配分なし' : '配分なし'}</strong>${integer(amount) ? `<small>予算 ${yen(amount)}</small>` : ''}</div><p>${escape(reason)}</p>${integer(saved?.best_possible_return) ? `<p>この予算での最低想定払戻は${yen(saved.best_possible_return)}です。</p>` : ''}${alternative}<p>100円ずつの買い目は上に表示しています。</p></article>`;
    }
    const table = planTickets(p).map(t => `<tr>${t.split('-').map(n => `<td>${car(n)}</td>`).join('')}<td><b>${yen(a.stakes[t])}</b></td></tr>`).join('');
    return `<article class="purchase-card"><h5>${state.budget === 'auto' ? '比較用の自動配分 · 共通予算' : '予算を配分'}</h5><div class="purchase-total"><strong>${yen(a.budget)}</strong><small>100円単位</small></div><p class="minimum">保存時の最低想定払戻 <b>${yen(a.minimum_return)}</b></p><p>オッズ ${stamp(p.odds_at)} 保存。変更後の払戻は変わります。</p>${resultCash(row, side, 'allocated', state.budget)}<details class="allocation-detail"><summary>買い目ごとの配分（${p.count}点）</summary><table class="allocation-table"><thead><tr><th scope="col">1着</th><th scope="col">2着</th><th scope="col">3着</th><th scope="col">購入額</th></tr></thead><tbody>${table}</tbody></table></details></article>`;
  }

  function purchases(row, side) {
    const p = row[side];
    return `<div class="purchase-grid"><article class="purchase-card"><h5>100円ずつ購入</h5><div class="purchase-total"><strong>${yen(p.count * 100)}</strong><small>${p.count}点 × 100円</small></div><p>上のフォーメーションを各点100円で購入した場合。</p>${resultCash(row, side, 'flat', 'auto')}</article>${allocatedCard(row, side)}</div>`;
  }

  function raceDetail(row) {
    const s = status(row), p = row.candidate, ready = hasSnapshot(row);
    const heading = `<div class="race-heading"><div><h4>${escape(row.venue)} ${escape(row.race_no)}R</h4><p class="race-times">${time(row.start_at)} 発走 · ${time(row.close_at)} 締切</p></div><div class="race-flags"><span class="status ${s.css}">${s.label}</span>${ready ? grade(p) : ''}</div></div>`;
    if (!ready) return `${heading}<div class="missing"><strong>${epoch(row.close_at) > Date.now() ? '10点予想の保存待ちです' : '締切前の10点予想が保存されていません'}</strong><p>${escape(row.reason || '必要な予測データを確認できたレースから、締切前に保存します。')}</p><a href="index.html#forecast">NEXUSの元の予想・結果を見る →</a></div>`;
    const c = p.confidence;
    const riderList = (Array.isArray(row.riders) ? row.riders : []).filter(r => /^[1-9]$/.test(String(r.car))).map(r => `<li>${car(r.car)}<span class="mark">${escape(p.marks?.[String(r.car)] || '—')}</span><span class="rider-name">${escape(r.name)}</span></li>`).join('');
    return `${heading}<p class="confidence-note">${c?.status === 'assessed' ? `選んだ10点へのモデル支持度 ${pct(c.model_mass)}。実際の的中率とは異なります。` : '自信度の根拠を確認中です。'}</p><ul class="rider-list" aria-label="選手の予想印">${riderList}</ul>${formation(p, '10点フォーメーション')}${purchases(row, 'candidate')}<details class="baseline-details"><summary>元の予想と比較（${row.baseline.count}点）</summary><div class="baseline-content">${formation(row.baseline, '同時に保存した元のフォーメーション')}${purchases(row, 'baseline')}<p class="saved-note">元の予想 ${stamp(row.baseline.captured_at)} 保存。配分は比較用に同じオッズで保存しています。</p></div></details><p class="saved-note">10点予想 ${stamp(row.saved_at)} 保存 · 買い目は保存時のまま表示しています。</p>`;
  }

  function renderMeetings(rows) {
    const offsets = new Map([...document.querySelectorAll('.race-picker')].map(el => [el.dataset.pickerKey, el.scrollLeft]));
    byId('forecast-count').textContent = `${rows.length}R`;
    if (!rows.length) {byId('meetings').innerHTML = '<p class="empty-state">この条件に該当するレースはありません。開催日・時間帯・開催場を変更してください。</p>'; return;}
    const groups = new Map();
    for (const r of rows) {const key = `${r.date}|${r.venue}`; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(r);}
    byId('meetings').innerHTML = [...groups].map(([key, group]) => {
      const chosen = group.find(r => r.race_id === state.selected.get(key)) || group.find(r => epoch(r.close_at) > Date.now()) || group[group.length - 1];
      state.selected.set(key, chosen.race_id);
      const nav = group.map(r => {const s = status(r); return `<button type="button" data-race="${r.race_id}" data-group="${escape(key)}" aria-pressed="${r.race_id === chosen.race_id}" aria-label="${escape(r.venue)} ${r.race_no}レース ${time(r.start_at)}発走 ${s.label}"><strong>${r.race_no}R${hasSnapshot(r) && ['S', 'A', 'B'].includes(r.candidate.confidence?.grade) ? `<span class="tiny-grade"> ${r.candidate.confidence.grade}</span>` : ''}</strong><small>${time(r.start_at)}</small><small>${s.label}</small></button>`;}).join('');
      return `<section class="meeting" aria-label="${escape(group[0].venue)} ${dateLabel(group[0].date)}"><div class="meeting-head"><h3>${escape(group[0].venue)}<span>${bandLabel(group[0].session)}</span></h3><small>${dateLabel(group[0].date)} · ${group.length}R</small></div><div class="race-picker" data-picker-key="${escape(key)}" role="group" aria-label="${escape(group[0].venue)}のレース番号">${nav}</div><div class="race-detail">${raceDetail(chosen)}</div></section>`;
    }).join('');
    document.querySelectorAll('.race-picker').forEach(el => {const active = el.querySelector('[aria-pressed=true]'); el.scrollLeft = offsets.get(el.dataset.pickerKey) ?? (active ? Math.max(0, active.offsetLeft - el.offsetLeft - el.clientWidth / 2 + active.clientWidth / 2) : 0);});
  }

  function metricCard(name, values, sample, css) {
    return `<article class="metric-card ${css}"><h3>${name}</h3><div class="metric-pair"><div><span>的中率</span><strong>${pct(values.hit_rate)}</strong><small>${values.hits} / ${sample}レース</small></div><div><span>回収率</span><strong>${pct(values.roi)}</strong><small class="${values.profit_yen >= 0 ? 'positive' : 'negative'}">収支 ${sample ? profit(values.profit_yen) : '—'}</small></div></div><p class="metric-finance"><span>購入 <b>${sample ? yen(values.stake_yen) : '—'}</b></span><span>払戻 <b>${sample ? yen(values.payout_yen) : '—'}</b></span></p></article>`;
  }

  function comparisonCell(row, side) {
    const name = side === 'candidate' ? '10点方式' : '元の方式', v = settlement(row, side, state.method, state.budget);
    if (!v) return `<div class="comparison-cell ${side}"><div><strong>${name}</strong><span class="status">${hasSnapshot(row) ? state.method === 'allocated' && !allocation(row[side], state.budget) ? '配分なし' : status(row).label : '事前記録なし'}</span></div><p>払戻・収支 <b>—</b></p></div>`;
    return `<div class="comparison-cell ${side}"><div><strong>${name}</strong><span class="status ${v.hit ? 'hit' : 'miss'}">${v.hit ? '的中' : '不的中'}</span></div><p>購入 <b>${yen(v.stake)}</b></p><p>払戻 <b>${yen(v.returned)}</b></p><p>収支 <b class="${v.profit >= 0 ? 'positive' : 'negative'}">${profit(v.profit)}</b></p></div>`;
  }

  function renderResults(rows) {
    const summary = comparison(rows, state.method, state.budget), mode = state.method === 'flat' ? '各点100円' : state.budget === 'auto' ? '自動・共通予算' : `各${yen(Number(state.budget))}の配分`;
    byId('result-scope').textContent = `${summary.races}R · ${mode}`;
    byId('metrics').innerHTML = metricCard('元の方式', summary.baseline, summary.races, 'baseline') + metricCard('10点フォーメーション', summary.candidate, summary.races, 'candidate');
    byId('metrics-note').textContent = `${state.day === 'all' ? '保存期間すべて' : dateLabel(state.day)}・${bandLabel(state.band)}・${state.venue === 'all' ? '全開催場' : state.venue}。同じ${summary.races}Rで比較。${summary.saved}Rを事前保存、結果待ち${summary.pending}R、返還・精算確認中${summary.excluded}R。${state.method === 'allocated' ? `両方式で選択予算の配分・精算が確認できない確定レース${summary.unavailable}Rは集計外です。` : ''}過去32Rの研究値は含めません。`;
    const shown = rows.filter(r => r.status === 'settled' || ['excluded', 'invalid_result'].includes(r.status) || official(r) || epoch(r.close_at) <= Date.now()).slice().reverse();
    byId('result-list').innerHTML = shown.length ? shown.map(r => {
      const actual = official(r), resultLabel = actual ? actual.winners.map(t => `<span>${t.split('-').map(car).join(' ')}</span>`).join(' / ') : escape(status(r).label);
      return `<article class="result-card"><header><h4>${escape(r.venue)} ${escape(r.race_no)}R <small>${dateLabel(r.date)}</small></h4><div class="official"><span>公式結果</span>${resultLabel}</div></header><div class="comparison-row">${comparisonCell(r, 'baseline')}${comparisonCell(r, 'candidate')}</div>${r.reason ? `<p class="result-reason">${escape(r.reason)}</p>` : ''}</article>`;
    }).join('') : '<p class="empty-state">この条件では、まだ締切後のレースがありません。結果が確定するとここに表示します。</p>';
  }

  function renderControls() {
    const dates = [...new Set(state.feed.races.map(r => r.date))].sort().reverse();
    if (!state.day || (state.day !== 'all' && !dates.includes(state.day))) state.day = dates.includes(state.feed.date) ? state.feed.date : dates[0] || 'all';
    byId('day').innerHTML = dates.map(d => `<option value="${d}">${dateLabel(d)}</option>`).join('') + '<option value="all">保存期間すべて</option>';
    byId('day').value = state.day;
    const venues = [...new Set(scopeRows(true).map(r => r.venue))].filter(Boolean).sort((a, b) => a.localeCompare(b, 'ja'));
    if (!venues.includes(state.venue)) state.venue = 'all';
    byId('venue').innerHTML = '<option value="all">全開催場</option>' + venues.map(v => `<option value="${escape(v)}">${escape(v)}</option>`).join(''); byId('venue').value = state.venue;
    const base = scopeRows(false, true);
    byId('bands').innerHTML = BANDS.map(([key, label]) => `<button type="button" data-band="${key}" aria-pressed="${key === state.band}">${label}<span>${base.filter(r => key === 'all' || r.session === key).length}R</span></button>`).join('');
    const rows = scopeRows(), saved = rows.filter(hasSnapshot).length, open = rows.filter(r => epoch(r.close_at) > Date.now()).length;
    byId('coverage').innerHTML = `10点予想 <b>${saved} / ${rows.length}R 保存済み</b>　締切前 ${open}R`;
    const budgets = new Set([5000, 6000, 7000, 8000, 9000, 10000]);
    for (const r of rows) for (const side of ['candidate', 'baseline']) for (const key of Object.keys(r[side]?.allocations || {})) if (allocation(r[side], key)) budgets.add(Number(key));
    if (state.budget !== 'auto') budgets.add(Number(state.budget));
    byId('budget').innerHTML = '<option value="auto">自動・共通予算（5千〜1万円）</option>' + [...budgets].sort((a, b) => a - b).map(n => `<option value="${n}">${yen(n)}</option>`).join('');
    byId('budget').value = state.budget;
  }

  function render() {
    if (!state.feed) return;
    byId('updated').textContent = stamp(state.feed.updated_at);
    renderControls();
    document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === state.view)));
    document.querySelectorAll('[data-method]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.method === state.method)));
    byId('forecasts').hidden = state.view !== 'forecast'; byId('results').hidden = state.view !== 'results';
    const rows = scopeRows(); renderMeetings(rows); renderResults(rows);
    const totalSaved = state.feed.races.filter(hasSnapshot).length;
    byId('notice').classList.toggle('error', Boolean(state.error));
    byId('notice').textContent = state.error || (totalSaved ? `締切前に保存した10点予想で比較中です。比較開始 ${stamp(state.feed.policy.activated_at)}。フォーメーション・金額は保存時の記録です。` : '10点予想の事前保存はまだありません。レース一覧には、各レースの保存状況を表示しています。');
  }

  async function load() {
    if (state.loading) return;
    state.loading = true; byId('refresh').disabled = true;
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch('original/nexus_forward.json?_=' + Date.now(), {cache: 'no-store', signal: controller.signal});
      if (!response.ok) throw new Error('比較データを取得できませんでした。');
      state.feed = validateFeed(await response.json()); state.error = ''; render();
    } catch (error) {
      state.error = state.feed ? '最新データの取得に失敗しました。現在は上記の更新時刻に保存されたデータを表示しています。更新ボタンで再確認できます。' : '10点予想の比較データをまだ読み込めません。更新ボタンで再確認してください。元の予想は上の「NEXUS予想」から確認できます。';
      if (state.feed) render(); else {byId('notice').textContent = state.error; byId('notice').classList.add('error'); byId('updated').textContent = '未取得'; byId('meetings').innerHTML = '<p class="empty-state">公開データを確認中です。予想や成績を推定して表示することはありません。</p>';}
    } finally {clearTimeout(timer); state.loading = false; byId('refresh').disabled = false;}
  }

  document.addEventListener('click', event => {
    const b = event.target.closest('button'); if (!b) return;
    if (b.id === 'refresh') {load(); return;}
    if (b.dataset.view) {state.view = b.dataset.view; render(); return;}
    if (b.dataset.method) {state.method = b.dataset.method; render(); return;}
    if (b.dataset.band) {state.band = b.dataset.band; render(); return;}
    if (b.dataset.budgetRecommend) {state.budget = b.dataset.budgetRecommend; render(); return;}
    if (b.dataset.race && b.dataset.group) {state.selected.set(b.dataset.group, b.dataset.race); const raceId = b.dataset.race; renderMeetings(scopeRows()); document.querySelector(`[data-race="${raceId}"]`)?.focus({preventScroll: true});}
  });
  byId('day').addEventListener('change', event => {state.day = event.target.value; state.venue = 'all'; render();});
  byId('venue').addEventListener('change', event => {state.venue = event.target.value; render();});
  byId('budget').addEventListener('change', event => {state.budget = event.target.value; render();});
  load();
  setInterval(() => {if (!document.hidden) load();}, 60000);
})();
