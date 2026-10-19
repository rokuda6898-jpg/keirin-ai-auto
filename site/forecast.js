'use strict';
// Display only: predictions, stakes and official settlements stay fixed in the feed.
(() => {
  const BANDS = [['all', 'すべて'], ['morning', 'モーニング'], ['day', 'デイ'], ['night', 'ナイト'], ['midnight', 'ミッドナイト']];
  const LABELS = {before: '締切前', pending: '結果待ち', hit: '的中', miss: '不的中', held: '精算保留', skipped: '見送り', unavailable: '未配分', invalid: '確認中', missing: '予想待ち'};
  const isNumber = n => typeof n === 'number' && Number.isFinite(n);
  const epoch = v => typeof v === 'number' ? v * (v < 1e12 ? 1000 : 1) : Date.parse(v);
  const time = v => Number.isFinite(epoch(v)) ? new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', hour: '2-digit', minute: '2-digit', hour12: false}).format(epoch(v)) : '—';
  const stamp = v => Number.isFinite(epoch(v)) ? new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false}).format(epoch(v)) : '未確認';
  const today = () => new Intl.DateTimeFormat('sv-SE', {timeZone: 'Asia/Tokyo', year: 'numeric', month: '2-digit', day: '2-digit'}).format(Date.now());
  const yen = v => isNumber(v) ? Math.round(v).toLocaleString('ja-JP') + '円' : '—';
  const percent = v => isNumber(v) ? (v * 100).toFixed(1) + '%' : '—';
  const profit = v => isNumber(v) ? (v > 0 ? '+' : v === 0 ? '±' : '') + yen(v) : '—';
  const bandLabel = id => BANDS.find(x => x[0] === id)?.[1] || '時間帯未確認';
  const dayLabel = d => d === 'all' ? '保存期間すべて' : new Intl.DateTimeFormat('ja-JP', {month: 'numeric', day: 'numeric', weekday: 'short', timeZone: 'Asia/Tokyo'}).format(new Date(d + 'T12:00:00+09:00'));
  const bandKey = r => String(r.date) + '|' + String(r.venue);
  const validBuy = buy => typeof buy === 'string' && /^[1-9]-[1-9]-[1-9]$/.test(buy) && new Set(buy.split('-')).size === 3;
  function tickets(r) {
    const t = r.shadow?.tickets;
    if (!Array.isArray(t) || !t.length || t.some(x => !validBuy(x)) || new Set(t).size !== t.length) return [];
    return t;
  }
  function normalizeBand(raw) {
    const s = String(raw || '').toLowerCase();
    if (/midnight|ミッド|深夜/.test(s)) return 'midnight';
    if (/morning|モーニング|朝/.test(s)) return 'morning';
    if (/night|ナイト|ナイター|夜/.test(s)) return 'night';
    if (/day|デイ|昼/.test(s)) return 'day';
    return '';
  }
  function officialResults(r) {
    const withdrawn = new Set((r.cancelled_cars || []).map(String));
    return (Array.isArray(r.actual) ? r.actual : []).filter(b => validBuy(b) && !b.split('-').some(c => withdrawn.has(c)));
  }
  function meetingBands(races) {
    const groups = new Map(), result = new Map();
    for (const r of races) { const key = bandKey(r); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(r); }
    for (const [key, group] of groups) {
      const explicit = group.map(r => normalizeBand(r.meeting_time_band || r.meeting_session || r.session || r.time_band || r.band)).find(Boolean);
      if (explicit) { result.set(key, explicit); continue; }
      const starts = group.map(r => epoch(r.start_at)).filter(Number.isFinite).sort((a, b) => a - b);
      if (!starts.length) { result.set(key, 'unknown'); continue; }
      const minutes = ms => {const d = new Date(ms + 9 * 3600000); return d.getUTCHours() * 60 + d.getUTCMinutes();};
      const first = minutes(starts[0]), last = minutes(starts[starts.length - 1]);
      // Classify the whole meeting, including early races of a night meeting.
      const band = first >= 20 * 60 + 30 || last >= 22 * 60 ? 'midnight'
        : last >= 17 * 60 ? 'night' : first < 11 * 60 && last < 15 * 60 ? 'morning' : 'day';
      result.set(key, band);
    }
    return result;
  }
  function allocation(r) {
    const p = r.budget, selected = tickets(r);
    if (p?.status !== 'allocated' || !selected.length) return null;
    const list = p.allocations, total = p.total_stake_yen;
    if (!Array.isArray(list) || list.length !== selected.length || list.some(a => !a || typeof a !== 'object') || new Set(list.map(a => a.buy)).size !== selected.length
      || !Number.isSafeInteger(total) || total < 5000 || total > 10000 || total % 100 !== 0
      || list.some(a => !selected.includes(a.buy) || !Number.isSafeInteger(a.stake_yen) || a.stake_yen < 100 || a.stake_yen % 100 !== 0)
      || list.reduce((s, a) => s + a.stake_yen, 0) !== total) return null;
    return list;
  }
  function baseSettlement(r, now = Date.now()) {
    const buys = tickets(r), invested = buys.length * 100;
    if (!buys.length) return {status: 'missing', invested: null, paid: null, net: null, hit: null};
    const frozen = epoch(r.shadow.snapshot_at), close = epoch(r.close_at);
    if (!Number.isFinite(close) || !Number.isFinite(frozen) || frozen >= close) return {status: 'invalid', invested, paid: null, net: null, hit: null};
    const withdrawn = new Set((r.cancelled_cars || []).map(String));
    if (buys.some(b => b.split('-').some(c => withdrawn.has(c)))) return {status: 'held', invested, paid: null, net: null, hit: null};
    const actual = officialResults(r);
    if (!actual.length) return {status: Number.isFinite(close) && close > now ? 'before' : 'pending', invested, paid: null, net: null, hit: null};
    const winners = buys.filter(b => actual.includes(b));
    if (winners.some(b => !isNumber(r.payouts?.[b]) || r.payouts[b] < 0)) return {status: 'held', invested, paid: null, net: null, hit: Boolean(winners.length)};
    const paid = winners.reduce((sum, b) => sum + r.payouts[b], 0);
    return {status: winners.length ? 'hit' : 'miss', invested, paid, net: paid - invested, hit: Boolean(winners.length)};
  }
  function allocatedSettlement(r, now = Date.now()) {
    const p = r.budget;
    if (p?.status === 'infeasible') return {status: 'skipped', invested: 0, paid: null, net: null, hit: null};
    const list = allocation(r);
    if (!list) return {status: p?.status === 'allocated' ? 'invalid' : 'unavailable', invested: null, paid: null, net: null, hit: null};
    const invested = p.total_stake_yen, s = r.budget_settlement;
    if (s?.status === 'settled') {
      if (!isNumber(s.invested_yen) || s.invested_yen !== invested || !isNumber(s.payout_yen) || s.payout_yen < 0 || typeof s.hit !== 'boolean') return {status: 'invalid', invested, paid: null, net: null, hit: null};
      return {status: s.hit ? 'hit' : 'miss', invested, paid: s.payout_yen, net: s.payout_yen - invested, hit: s.hit};
    }
    if (['refund_pending', 'payout_pending', 'invalid_snapshot', 'not_purchased'].includes(s?.status)) return {status: 'held', invested, paid: null, net: null, hit: null};
    return {status: epoch(r.close_at) > now && !r.actual?.length ? 'before' : 'pending', invested, paid: null, net: null, hit: null};
  }
  function aggregate(races, type) {
    const states = races.map(r => type === 'allocated' ? allocatedSettlement(r) : baseSettlement(r));
    const done = states.filter(s => s.status === 'hit' || s.status === 'miss');
    const paid = done.reduce((n, s) => n + s.paid, 0), invested = done.reduce((n, s) => n + s.invested, 0);
    const pending = states.filter(s => ['before', 'pending'].includes(s.status));
    const held = states.filter(s => ['held', 'invalid'].includes(s.status));
    const hits = done.filter(s => s.hit).length;
    return {total: races.length, settled: done.length, hits, hitRate: done.length ? hits / done.length : null,
      roi: invested ? paid / invested : null, invested: done.length ? invested : null, paid: done.length ? paid : null,
      net: done.length ? paid - invested : null, pending: pending.length, open: pending.filter(s => s.status === 'before').length,
      awaitingResult: pending.filter(s => s.status === 'pending').length, pendingStake: pending.reduce((n, s) => n + (s.invested || 0), 0),
      held: held.length, skipped: states.filter(s => s.status === 'skipped').length,
      unavailable: states.filter(s => ['missing', 'unavailable'].includes(s.status)).length};
  }
  const API = {tickets, allocation, baseSettlement, allocatedSettlement, aggregate, meetingBands, normalizeBand, officialResults};
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  if (typeof document === 'undefined') return;
  window.ForecastUI = API;
  const $ = s => document.querySelector(s);
  const el = (tag, text, cls) => {const n = document.createElement(tag); if (text !== undefined && text !== null) n.textContent = text; if (cls) n.className = cls; return n;};
  const car = c => {const n = el('span', String(c), 'car c' + String(c)); n.setAttribute('aria-label', String(c) + '番'); return n;};
  const badge = status => el('span', LABELS[status] || '確認中', 'status ' + status);
  const moneyClass = n => n > 0 ? 'profit-positive' : n < 0 ? 'profit-negative' : '';
  const state = {data: null, bands: new Map(), date: '', band: 'all', venue: 'all', view: location.hash === '#results' ? 'results' : 'forecast', active: new Map(), expanded: new Set(), limit: 30, dates: [], loading: false, lastError: false};
  const filtered = (includeVenue = true, includeBand = true) => (state.data?.races || []).filter(r => (state.date === 'all' || r.date === state.date) && (!includeBand || state.band === 'all' || state.bands.get(bandKey(r)) === state.band) && (!includeVenue || state.venue === 'all' || r.venue === state.venue));
  function exactFormations(r, weighted) {
    const buys = tickets(r);
    if (!buys.length) return [];
    if (weighted) return allocation(r) ? window.ForecastMath.stakeFormations(allocation(r)) : [];
    return window.ForecastMath.toFormations(buys).map(f => ({...f, stake_yen: 100, total_stake_yen: f.count * 100}));
  }
  function formations(r, weighted) {
    const container = el('div', null, 'formation-list'), rows = exactFormations(r, weighted);
    rows.forEach((f, i) => {
      const winning = f.tickets.filter(t => officialResults(r).includes(t));
      const row = el('div', null, 'formation-row' + (winning.length ? ' winning' : ''));
      row.dataset.tickets = f.tickets.join(','); row.dataset.stakeYen = f.stake_yen;
      const meta = el('div', null, 'formation-meta');
      meta.append(el('span', 'フォーメーション ' + (i + 1) + ' · ' + f.count + '点'), el('strong', '各' + yen(f.stake_yen) + ' / 計' + yen(f.total_stake_yen)));
      const grid = el('div', null, 'formation-grid');
      [f.first, f.second, f.third].forEach((numbers, idx) => {
        if (idx) grid.append(el('span', '→', 'formation-arrow'));
        const leg = el('div', null, 'formation-leg'), nums = el('div', null, 'formation-numbers');
        leg.append(el('small', (idx + 1) + '着')); numbers.forEach(c => nums.append(car(c))); leg.append(nums); grid.append(leg);
      });
      row.append(meta, grid);
      if (winning.length) row.append(el('p', '的中を含む行', 'formation-hit'));
      container.append(row);
    });
    return container;
  }
  function resultBar(s) {
    const bar = el('div', null, 'purchase-result ' + s.status);
    if (s.status === 'hit' || s.status === 'miss') {
      bar.append(badge(s.status));
      for (const [label, value, cls] of [['払戻', yen(s.paid), ''], ['収支', profit(s.net), moneyClass(s.net)]]) {
        const box = el('span'); box.append(el('small', label), el('strong', value, cls)); bar.append(box);
      }
    } else {
      bar.append(badge(s.status), el('small', ['before', 'pending'].includes(s.status) ? '払戻・収支は結果確定後' : s.status === 'held' || s.status === 'invalid' ? '精算が確定するまで集計対象外' : '投資・払戻の集計対象外'));
    }
    return bar;
  }
  function purchase(r, weighted) {
    const selected = tickets(r), list = weighted ? allocation(r) : null;
    const total = weighted ? (list ? r.budget.total_stake_yen : null) : selected.length ? selected.length * 100 : null;
    const section = el('section', null, 'purchase-card ' + (weighted ? 'allocated' : 'uniform'));
    const header = el('div', null, 'purchase-header'), heading = el('div');
    heading.append(el('h6', weighted ? 'B 予算配分' : 'A 各点100円'), el('strong', yen(total)));
    header.append(heading, el('p', weighted ? '予算5,000〜10,000円 · 同じ' + selected.length + '点' : selected.length + '点 × 100円 · 均等購入'));
    section.append(header);
    if (!selected.length) {
      const box = el('div', null, 'purchase-state'); box.append(el('strong', '予想の保存待ち'), el('p', '買い目が保存され次第、フォーメーションを表示します。')); section.append(box);
    } else if (!weighted || list) {
      section.append(formations(r, weighted));
      if (weighted) {
        const p = r.budget;
        section.append(el('p', '保存オッズでの最低想定払戻 ' + yen(p.minimum_payout_yen) + ' · 最低想定収支 ' + profit(p.minimum_profit_yen), 'purchase-note'));
      } else section.append(el('p', '上から各行を購入 · 合計' + selected.length + '点（重複なし）', 'purchase-note'));
    } else {
      const p = r.budget, status = allocatedSettlement(r).status, box = el('div', null, 'purchase-state');
      box.append(badge(status));
      if (status === 'skipped') box.append(el('strong', 'このレースの配分は見送り'), el('p', '同じ' + selected.length + '点で、予算内に元本以上の想定払戻となる配分が成立しません。'));
      else if (status === 'invalid') box.append(el('strong', '保存配分を確認中'), el('p', '買い目と金額の整合性を確認できないため、配分額の使用を保留しています。'));
      else box.append(el('strong', epoch(r.close_at) > Date.now() && !r.actual?.length ? 'オッズ取得・配分待ち' : '締切前の配分記録なし'), el('p', '配分額は未確定です。保存できた金額だけを表示します。'));
      if (p?.required_stake_yen && status === 'skipped') box.append(el('p', '保存時に必要だった最低購入額 ' + yen(p.required_stake_yen)));
      section.append(box);
    }
    section.append(resultBar(weighted ? allocatedSettlement(r) : baseSettlement(r)));
    return section;
  }
  function riderMarks(r) {
    const result = new Map();
    for (const m of r.shadow?.marks || []) if (['◎', '○', '▲', '△', '☆'].includes(m.mark)) result.set(String(m.car), m.mark);
    // A stored model ranking can supply marks; missing ranks are never guessed from odds.
    const ranking = r.shadow?.rider_rank || r.shadow?.rider_ranking || r.shadow?.ranking;
    if (!result.size && Array.isArray(ranking)) ranking.slice(0, 5).forEach((v, i) => result.set(String(typeof v === 'object' ? v.car : v), ['◎', '○', '▲', '△', '☆'][i]));
    return result;
  }
  function raceContent(r) {
    const box = el('div', null, 'race-content'); box.dataset.raceId = r.id;
    const title = el('div', null, 'race-title'), heading = el('h4', r.venue + ' ' + r.number + 'R');
    heading.append(el('small', '発走 ' + time(r.start_at) + ' / 締切 ' + time(r.close_at))); title.append(heading, badge(baseSettlement(r).status)); box.append(title);
    const rh = el('div', null, 'rider-heading'); rh.append(el('h5', '出走選手・予想印'), el('span', (r.riders || []).length + '名 · ◎ ○ ▲ △ ☆')); box.append(rh);
    const grid = el('div', null, 'rider-grid'), marks = riderMarks(r), withdrawn = new Set((r.cancelled_cars || []).map(String));
    for (const rider of [...(r.riders || [])].sort((a, b) => Number(a.car) - Number(b.car))) {
      const c = String(rider.car), mark = marks.get(c), row = el('div', null, 'rider' + (withdrawn.has(c) ? ' cancelled' : ''));
      row.append(el('span', mark || '—', 'rider-mark' + (!mark ? ' unmarked' : '')), car(c));
      const name = el('span', rider.name || '選手名未取得', 'rider-name'); name.title = rider.name || '選手名未取得'; row.append(name);
      if (withdrawn.has(c)) row.setAttribute('aria-label', c + '番 ' + rider.name + ' 欠車'); grid.append(row);
    }
    box.append(grid);
    if (!r.riders?.length) box.append(el('p', '出走選手の確認待ちです。', 'race-warning'));
    if (withdrawn.size) box.append(el('p', '欠車 ' + [...withdrawn].join('・') + '番。保存時の予想です。欠車を含む買い目は精算確認まで保留しています。', 'race-warning'));
    const ph = el('div', null, 'picks-heading'); ph.append(el('h5', '3連単フォーメーション'), el('span', '2つの購入方法を同時に確認')); box.append(ph);
    const purchases = el('div', null, 'purchase-grid'); purchases.append(purchase(r, false), purchase(r, true)); box.append(purchases);
    if (r.actual?.length) {
      const official = el('div', null, 'official-result'); official.append(el('strong', '確定着順'));
      for (const actual of officialResults(r)) {
        const finish = el('span', null, 'official-finish'); actual.split('-').forEach((c, i) => {if (i) finish.append(el('span', '−')); finish.append(car(c));});
        finish.append(el('small', '100円払戻 ' + yen(r.payouts?.[actual]))); official.append(finish);
      }
      box.append(official);
    }
    const meta = el('details', null, 'race-meta'); meta.append(el('summary', '予想保存 ' + stamp(r.shadow?.snapshot_at) + ' · 配分・保存時刻の詳細'));
    meta.append(el('p', '予想：条件付き3連単・上位' + tickets(r).length + '点。'));
    if (r.budget?.odds_captured_at_jst) meta.append(el('p', (r.budget.odds_basis === 'initial_allocation_after_forecast' ? '初回配分時オッズ ' : '予想時オッズ ') + stamp(r.budget.odds_captured_at_jst)));
    if (r.budget?.allocation_frozen_at_jst) meta.append(el('p', '配分保存 ' + stamp(r.budget.allocation_frozen_at_jst)));
    meta.append(el('p', '配分後のオッズ下落で、的中しても購入額を下回る場合があります。')); box.append(meta);
    return box;
  }
  function performanceCard(races, weighted) {
    const s = aggregate(races, weighted ? 'allocated' : 'uniform'), card = el('article', null, 'performance-card ' + (weighted ? 'allocated' : 'uniform'));
    const h = el('h3'); h.append(el('span', weighted ? 'B' : 'A', 'variant-index'), document.createTextNode(weighted ? '予算配分の成績' : '各点100円の成績')); card.append(h);
    const rates = el('div', null, 'metric-rates');
    for (const [label, value, note] of [['的中率', percent(s.hitRate), s.hits + ' / ' + s.settled + 'R'], ['回収率', percent(s.roi), '精算済み ' + s.settled + 'R']]) {
      const d = el('div'); d.append(el('span', label), el('strong', value), el('small', note)); rates.append(d);
    }
    card.append(rates);
    const amounts = el('dl', null, 'metric-money');
    for (const [label, value, cls] of [['投資', yen(s.invested), ''], ['払戻', yen(s.paid), ''], ['収支', profit(s.net), moneyClass(s.net)]]) { const d = el('div'); d.append(el('dt', label), el('dd', value, cls)); amounts.append(d); }
    card.append(amounts);
    const notes = ['結果待ち ' + s.pending + 'R' + (s.pending ? '（' + yen(s.pendingStake) + '）' : ''), '精算保留 ' + s.held + 'R'];
    if (weighted) notes.push('見送り ' + s.skipped + 'R', '未配分 ' + s.unavailable + 'R'); else if (s.unavailable) notes.push('予想待ち ' + s.unavailable + 'R');
    card.append(el('p', notes.join(' / '), 'metric-status'));
    return card;
  }
  function renderControls() {
    for (const button of document.querySelectorAll('[data-view]')) { button.setAttribute('aria-selected', String(button.dataset.view === state.view)); button.tabIndex = button.dataset.view === state.view ? 0 : -1; }
    for (const link of document.querySelectorAll('[data-view-link]')) { if (link.dataset.viewLink === state.view) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current'); }
    $('#day').value = state.date;
    const i = state.dates.indexOf(state.date); $('#previous-day').disabled = i < 0 || i === state.dates.length - 1; $('#next-day').disabled = i <= 0;
    const dateRaces = filtered(false, false), bands = $('#band-tabs'); bands.replaceChildren();
    for (const [id, label] of BANDS) {
      const count = dateRaces.filter(r => id === 'all' || state.bands.get(bandKey(r)) === id).length;
      const b = el('button', label); b.type = 'button'; b.dataset.band = id; b.setAttribute('aria-pressed', String(id === state.band)); b.append(el('span', count + 'R')); bands.append(b);
    }
    const venueRaces = filtered(false), venues = [...new Set(venueRaces.map(r => r.venue))];
    if (state.venue !== 'all' && !venues.includes(state.venue)) state.venue = 'all';
    const holder = $('#venue-tabs'); holder.replaceChildren();
    for (const venue of ['all', ...venues]) {
      const count = venueRaces.filter(r => venue === 'all' || r.venue === venue).length;
      const b = el('button', venue === 'all' ? '全開催場' : venue); b.type = 'button'; b.dataset.venue = venue; b.setAttribute('aria-pressed', String(venue === state.venue)); b.append(el('small', count + 'R')); holder.append(b);
    }
  }
  function chooseRace(group) {
    const key = bandKey(group[0]), remembered = group.find(r => r.id === state.active.get(key));
    if (remembered) return remembered;
    return group.find(r => !r.actual?.length && epoch(r.close_at) > Date.now()) || [...group].reverse().find(r => r.actual?.length) || group[0];
  }
  function forecastVenue(group) {
    group.sort((a, b) => epoch(a.start_at) - epoch(b.start_at));
    const section = el('section', null, 'venue-section'); section.dataset.meeting = bandKey(group[0]);
    const heading = el('div', null, 'venue-heading'); heading.append(el('h3', group[0].venue), el('span', bandLabel(state.bands.get(bandKey(group[0])))), el('small', (state.date === 'all' ? dayLabel(group[0].date) + ' · ' : '') + group.length + 'R')); section.append(heading);
    const tabs = el('div', null, 'race-tabs'), active = chooseRace(group); tabs.setAttribute('role', 'tablist'); tabs.setAttribute('aria-label', group[0].venue + 'のレース');
    const contentId = 'race-panel-' + group[0].id;
    for (const r of group) {
      const status = baseSettlement(r).status, b = el('button'); b.type = 'button'; b.dataset.race = r.id; b.id = 'race-tab-' + r.id; b.setAttribute('role', 'tab'); b.setAttribute('aria-controls', contentId); b.setAttribute('aria-selected', String(r.id === active.id)); b.tabIndex = r.id === active.id ? 0 : -1; b.dataset.status = status;
      b.append(el('strong', r.number + 'R'), el('small', time(r.start_at)), el('span', LABELS[status] || '確認中')); tabs.append(b);
    }
    section.append(tabs);
    const content = raceContent(active); content.id = contentId; content.setAttribute('role', 'tabpanel'); content.setAttribute('aria-labelledby', 'race-tab-' + active.id); section.append(content);
    return section;
  }
  function renderForecast(races) {
    const holder = $('#races'); holder.replaceChildren();
    const grouped = new Map();
    for (const r of races) {const key = bandKey(r); if (!grouped.has(key)) grouped.set(key, []); grouped.get(key).push(r);}
    const groups = [...grouped.values()].sort((a, b) => epoch(a[0].start_at) - epoch(b[0].start_at));
    for (const id of [...BANDS.slice(1).map(b => b[0]), 'unknown']) {
      const matches = groups.filter(g => state.bands.get(bandKey(g[0])) === id); if (!matches.length) continue;
      const section = el('section', null, 'session-section'), h = el('h3', null, 'session-heading'); h.append(el('span', null, 'session-tag'), document.createTextNode(bandLabel(id)), el('small', matches.length + '開催')); section.append(h);
      for (const group of matches) section.append(forecastVenue(group)); holder.append(section);
    }
    $('#load-more').hidden = true;
    if (!races.length) holder.append(el('p', 'この条件のレースはありません。日付・時間帯・開催場を変更してください。', 'empty-state'));
  }
  function resultAmount(r, weighted) {
    const s = weighted ? allocatedSettlement(r) : baseSettlement(r), box = el('div', null, 'result-amount');
    box.append(el('small', weighted ? 'B 予算配分' : 'A 各点100円'));
    const row = el('div'); row.append(badge(s.status), el('strong', s.net === null ? '—' : profit(s.net), moneyClass(s.net))); box.append(row);
    box.append(el('span', s.paid === null ? ['before', 'pending', 'held'].includes(s.status) ? '払戻未確定' : '購入記録なし' : '投資 ' + yen(s.invested) + ' / 払戻 ' + yen(s.paid)));
    return box;
  }
  function renderResults(races) {
    const holder = $('#races'); holder.replaceChildren();
    const ordered = [...races].sort((a, b) => {
      const ad = Boolean(a.actual?.length), bd = Boolean(b.actual?.length);
      return Number(bd) - Number(ad) || epoch(b.start_at) - epoch(a.start_at);
    });
    const list = el('div', null, 'result-list');
    for (const r of ordered.slice(0, state.limit)) {
      const item = el('details', null, 'result-item'); item.dataset.raceId = r.id;
      const summary = el('summary'), label = el('div', null, 'result-race-label');
      label.append(el('strong', r.venue + ' ' + r.number + 'R'), el('small', dayLabel(r.date) + ' ' + time(r.start_at) + ' · ' + bandLabel(state.bands.get(bandKey(r)))));
      const finishes = officialResults(r);
      if (finishes.length) label.append(el('small', '着順 ' + finishes.map(b => b.replaceAll('-', '→')).join(' / '), 'result-finish'));
      summary.append(label, resultAmount(r, false), resultAmount(r, true)); item.append(summary);
      // Build the detailed formation only when opened; a full-history list stays light.
      item.addEventListener('toggle', () => {
        if (item.open) {state.expanded.add(r.id); if (!item.querySelector('.race-content')) item.append(raceContent(r));}
        else state.expanded.delete(r.id);
      });
      if (state.expanded.has(r.id)) {item.open = true; item.append(raceContent(r));}
      list.append(item);
    }
    holder.append(list); $('#load-more').hidden = ordered.length <= state.limit;
    $('#load-more').textContent = 'さらに結果を表示（残り' + Math.max(0, ordered.length - state.limit) + 'R）';
    if (!races.length) holder.append(el('p', 'この条件の記録はありません。日付・時間帯・開催場を変更してください。', 'empty-state'));
  }
  function updateClock() {
    if (!state.data) return;
    const next = state.data.races.filter(r => !r.actual?.length && epoch(r.close_at) > Date.now()).sort((a, b) => epoch(a.start_at) - epoch(b.start_at))[0];
    const b = $('#next-race'); b.hidden = !next;
    if (next) {b.dataset.race = next.id; $('#next-race-name').textContent = next.venue + ' ' + next.number + 'R'; $('#next-race-time').textContent = time(next.start_at) + ' 発走 · ' + time(next.close_at) + ' 締切';}
    const age = Date.now() - epoch(state.data.updated_at || state.data.generated_at), notice = $('#notice');
    if (state.lastError) {
      notice.hidden = false; notice.className = 'notice error'; notice.textContent = '更新できませんでした。最後に読み込めた保存データを表示しています。更新ボタンで再読み込みできます。';
    } else if (state.data.schedule_date && state.data.schedule_date !== today()) {
      notice.hidden = false; notice.className = 'notice stale'; notice.textContent = '現在表示できる最新開催日は' + state.data.schedule_date + 'です。当日分の更新を確認してください。';
    } else if (!Number.isFinite(age) || age > 45 * 60000) {
      notice.hidden = false; notice.className = 'notice stale'; notice.textContent = 'データ更新から' + (Number.isFinite(age) ? Math.max(0, Math.floor(age / 60000)) + '分経過しています。' : 'の時刻を確認できません。') + '結果・オッズは更新時点の記録です。';
    } else {notice.hidden = true; notice.className = 'notice';}
    for (const b of document.querySelectorAll('.race-tabs button[data-race]')) {
      const r = state.data.races.find(r => r.id === b.dataset.race); if (!r) continue;
      const status = baseSettlement(r).status; b.dataset.status = status; b.querySelector('span').textContent = LABELS[status] || '確認中';
    }
    for (const content of document.querySelectorAll('.race-content[data-race-id]')) {
      const r = state.data.races.find(r => r.id === content.dataset.raceId); if (!r) continue;
      content.querySelector('.race-title .status')?.replaceWith(badge(baseSettlement(r).status));
      content.querySelector('.purchase-card.uniform .purchase-result')?.replaceWith(resultBar(baseSettlement(r)));
      content.querySelector('.purchase-card.allocated .purchase-result')?.replaceWith(resultBar(allocatedSettlement(r)));
    }
  }
  function render() {
    if (!state.data) return;
    renderControls(); const races = filtered();
    $('#performance-scope').textContent = [dayLabel(state.date), state.band === 'all' ? '全時間帯' : bandLabel(state.band), state.venue === 'all' ? '全開催場' : state.venue].join(' · ');
    $('#performance-cards').replaceChildren(performanceCard(races, false), performanceCard(races, true));
    $('#race-heading').textContent = state.view === 'results' ? 'レース別の結果・収支' : '開催レースの予想'; $('#race-count').textContent = races.length + 'R';
    if (state.view === 'results') renderResults(races); else renderForecast(races);
    updateClock();
  }
  function setView(view) {
    state.view = view; state.limit = 30;
    if (location.hash !== '#' + view) history.replaceState(null, '', '#' + view);
    window.dispatchEvent(new Event('hashchange'));
    render();
  }
  async function load() {
    if (state.loading) return; state.loading = true; $('#refresh').disabled = true;
    try {
      const response = await fetch('original/conditional_live.json?_=' + Date.now(), {cache: 'no-store'});
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const data = await response.json();
      if (!Array.isArray(data.races) || !window.ForecastMath?.toFormations || !window.ForecastMath?.stakeFormations) throw new Error('Missing forecast data or formation support');
      const updated = data.updated_at || data.generated_at;
      const unchanged = Boolean(state.data && updated && updated === (state.data.updated_at || state.data.generated_at) && data.races.length === state.data.races.length);
      const oldScroll = window.scrollY, focusedId = document.activeElement?.id;
      const rowScroll = new Map([...document.querySelectorAll('.venue-section')].map(n => [n.dataset.meeting, n.querySelector('.race-tabs').scrollLeft]));
      state.data = data; state.lastError = false; state.bands = meetingBands(data.races); state.dates = [...new Set(data.races.map(r => r.date))].sort().reverse();
      if (unchanged) {updateClock(); return;}
      if (!state.date || state.date !== 'all' && !state.dates.includes(state.date)) state.date = state.dates.includes(today()) ? today() : data.schedule_date && state.dates.includes(data.schedule_date) ? data.schedule_date : state.dates[0] || 'all';
      $('#day').replaceChildren();
      for (const d of [...state.dates, 'all']) {const o = el('option', d === 'all' ? '保存期間すべて' : dayLabel(d) + (d === today() ? ' / 今日' : '')); o.value = d; $('#day').append(o);}
      $('#updated').textContent = stamp(data.updated_at || data.generated_at);
      render();
      for (const row of document.querySelectorAll('.venue-section')) if (rowScroll.has(row.dataset.meeting)) row.querySelector('.race-tabs').scrollLeft = rowScroll.get(row.dataset.meeting);
      if (focusedId) document.getElementById(focusedId)?.focus({preventScroll: true});
      if (oldScroll) window.scrollTo({top: oldScroll, behavior: 'instant'});
    } catch (error) {
      state.lastError = true;
      const notice = $('#notice'); notice.hidden = false; notice.className = 'notice error';
      notice.textContent = state.data ? '更新できませんでした。現在は最後に読み込めた保存データを表示しています。更新ボタンで再読み込みできます。' : '予想データを読み込めませんでした。更新ボタンで再読み込みしてください。';
      if (!state.data) $('#races').replaceChildren(el('p', 'データの再読み込み待ちです。', 'empty-state'));
    } finally {state.loading = false; $('#refresh').disabled = false;}
  }
  $('#refresh').addEventListener('click', load);
  $('#day').addEventListener('change', e => {state.date = e.target.value; state.venue = 'all'; state.limit = 30; render();});
  $('#previous-day').addEventListener('click', () => {const i = state.dates.indexOf(state.date); if (i >= 0 && i + 1 < state.dates.length) {state.date = state.dates[i + 1]; state.venue = 'all'; state.limit = 30; render();}});
  $('#next-day').addEventListener('click', () => {const i = state.dates.indexOf(state.date); if (i > 0) {state.date = state.dates[i - 1]; state.venue = 'all'; state.limit = 30; render();}});
  $('#band-tabs').addEventListener('click', e => {const b = e.target.closest('button[data-band]'); if (b) {state.band = b.dataset.band; state.venue = 'all'; state.limit = 30; render();}});
  $('#venue-tabs').addEventListener('click', e => {const b = e.target.closest('button[data-venue]'); if (b) {state.venue = b.dataset.venue; state.limit = 30; render();}});
  document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
  $('#races').addEventListener('click', e => {
    const b = e.target.closest('button[data-race]'); if (!b) return;
    const r = state.data.races.find(r => r.id === b.dataset.race); if (!r) return;
    state.active.set(bandKey(r), r.id);
    const venue = b.closest('.venue-section'), replacement = raceContent(r), old = venue.querySelector('.race-content');
    replacement.id = old.id; replacement.setAttribute('role', 'tabpanel'); replacement.setAttribute('aria-labelledby', b.id); old.replaceWith(replacement);
    venue.querySelectorAll('.race-tabs button').forEach(btn => {btn.setAttribute('aria-selected', String(btn === b)); btn.tabIndex = btn === b ? 0 : -1;});
  });
  document.addEventListener('keydown', e => {
    const button = e.target.closest('[role=tab]'); if (!button || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    const buttons = [...button.parentElement.querySelectorAll('[role=tab]')], i = buttons.indexOf(button);
    const target = buttons[e.key === 'Home' ? 0 : e.key === 'End' ? buttons.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length];
    e.preventDefault(); target.click(); target.focus();
  });
  $('#load-more').addEventListener('click', () => {state.limit += 30; renderResults(filtered());});
  $('#next-race').addEventListener('click', () => {
    const r = state.data?.races.find(r => r.id === $('#next-race').dataset.race); if (!r) return;
    state.date = r.date; state.band = state.bands.get(bandKey(r)) || 'all'; state.venue = r.venue; state.active.set(bandKey(r), r.id); setView('forecast');
    $('#race-area').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'}); $('#race-area').focus({preventScroll: true});
  });
  window.addEventListener('hashchange', () => {
    if (location.hash === '#model-details') {$('#model-details').open = true; return;}
    const view = location.hash === '#results' ? 'results' : location.hash === '#forecast' ? 'forecast' : null;
    if (view && view !== state.view) {state.view = view; state.limit = 30; render();}
  });
  if (location.hash === '#model-details') $('#model-details').open = true;
  setInterval(updateClock, 30000);
  setInterval(() => {if (document.visibilityState === 'visible') load();}, 60000);
  load();
})();
