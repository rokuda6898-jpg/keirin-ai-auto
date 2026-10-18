'use strict';
// Prospective 50x+ overlay. Read-only research; never amends original forecasts.
if (typeof document !== 'undefined' && document.body.dataset.feed === 'conditional_live.json') {
  (() => {
    const node = (tag, text, cls) => {
      const n = document.createElement(tag);
      if (text !== undefined) n.textContent = text;
      if (cls) n.className = cls;
      return n;
    };
    const money = n => Number(n).toLocaleString('ja-JP') + '円';
    const pct = n => n === null ? '—' : (n * 100).toFixed(2) + '%';
    const paired = r => r.high_value?.new_high?.length
      && r.budget_settlement?.status === 'settled'
      && r.high_value_settlement?.status === 'settled'
      && r.budget_settlement.invested_yen === 10000
      && r.high_value_settlement.invested_yen === 10000;
    function showSummary(races) {
      const target = document.querySelector('#high-value-stats');
      if (!target) return;
      const proposed = races.filter(r => r.high_value?.new_high?.length);
      const comparable = races.filter(paired);
      const sum = (key, name) => comparable.reduce((s, r) => s + r[key][name], 0);
      const baselinePaid = sum('budget_settlement', 'payout_yen');
      const alternatePaid = sum('high_value_settlement', 'payout_yen');
      const baseHits = comparable.filter(r => r.budget_settlement.hit).length;
      const altHits = comparable.filter(r => r.high_value_settlement.hit).length;
      const total = comparable.length * 10000;
      const vals = [
        ['高配当候補を追加', proposed.length + 'R', '既存12点のうち最大4点を差替え'],
        ['50〜99倍の追加', proposed.reduce((s,r) => s+r.high_value.new_high.filter(x=>x.odds_band==='50_to_99').length,0)+'点', '100倍以上の穴枠とは別集計'],
        ['100倍以上の追加', proposed.reduce((s,r) => s+r.high_value.new_high.filter(x=>x.odds_band==='100x_plus').length,0)+'点', '発走前オッズによる分類'],
        ['同条件で精算済み', comparable.length+'R', '元予想・追加案とも1万円配分成立'],
        ['元予想的中率', comparable.length?pct(baseHits/comparable.length):'—', baseHits+'/'+comparable.length+'R'],
        ['追加案的中率', comparable.length?pct(altHits/comparable.length):'—', altHits+'/'+comparable.length+'R'],
        ['元予想回収率', total?pct(baselinePaid/total):'—', total?'公式払戻 '+money(baselinePaid):'未検証'],
        ['追加案回収率', total?pct(alternatePaid/total):'—', total?'公式払戻 '+money(alternatePaid):'未検証']
      ];
      target.replaceChildren();
      vals.forEach(([label, value, hint]) => {
        const box = node('div', undefined, 'stat');
        box.append(node('span', label), node('strong', value), node('small', hint));
        target.append(box);
      });
      const desc = document.querySelector('#high-value-scope');
      if (desc) desc.textContent = '保存済みの同じレース・発走前の同一オッズで2方式を比較。両方式の配分が成立し、公式払戻まで確定したレースだけが回収率比較対象。対象外や未作成を勝ち負けに混ぜません。予測確率は未校正のため、表示スコアは期待利益の保証ではありません。';
    }
    function createPanel(r) {
      const section = node('section', undefined, 'conditional-budget high-value-details');
      section.append(node('h4', '50倍以上を拾う別選抜｜凍結テスト', 'pick-label'));
      const overlay = r.high_value;
      if (!overlay) {
        section.append(node('p', 'この予想を保存した時点では高配当比較を行っていません。後付け予想は作成しません。', 'saved'));
        return section;
      }
      if (overlay.status !== 'ready') {
        section.append(node('p', '比較不可：'+(overlay.reason || '全組み合わせの発走前オッズ未取得'), 'saved'));
        return section;
      }
      section.append(node('p','元の確率上位12点は変更せず、別枠で最大4点の50倍以上を探します。50〜99倍は中高配当、100倍以上を穴として区別。', 'saved'));
      if (!overlay.new_high.length) {
        section.append(node('p', '条件を満たす追加候補なし。低確率の高配当を無理に補充しません。', 'saved'));
        return section;
      }
      const ul = node('ul');
      overlay.new_high.forEach(x => ul.append(node('li',
        (x.odds_band === '100x_plus' ? '穴・100倍以上' : '中高配当・50〜99倍') +
        '　'+x.buy+'　'+Number(x.odds).toFixed(1)+'倍　AI順位 '+x.model_rank+'位')));
      section.append(ul);
      section.append(node('p', '差替え対象：'+overlay.replaced.join(' ／ ')+'。別案の合計は'+overlay.tickets.length+'点。全候補 '+overlay.market_count+'通りを同じ時刻のオッズで評価。', 'saved'));
      const b = r.high_value_budget, s = r.high_value_settlement;
      if (!b) {
        section.append(node('p','別案の配分記録なし。', 'saved'));
      } else if (b.status === 'allocated') {
        section.append(node('p','別案の仮想配分 '+money(b.total_stake_yen)+' ／ 保存オッズでの最低想定払戻 '+money(b.minimum_payout_yen), 'saved'));
        if (s?.status === 'settled') {
          section.append(node('p','公式払戻 '+money(s.payout_yen)+' ／ 収支 '+money(s.profit_yen)+' ／ 回収率 '+pct(s.roi), 'result'));
        } else {
          section.append(node('p','公式払戻・返還等の確定待ち。最終回収率は未確定。', 'saved'));
        }
      } else {
        section.append(node('p','別案は見送り：'+(b.reason||b.status), 'saved'));
      }
      section.append(node('p','これは前向き比較用の仮想予想です。既存の購入配分・買い目を変更しません。', 'saved'));
      return section;
    }
    function update() {
      if (typeof state === 'undefined' || !state.data) return;
      const races = selected(), index = new Map(races.map(r=>[r.id,r]));
      showSummary(races);
      for (const card of document.querySelectorAll('#races details.race')) {
        if (card.querySelector('.high-value-details')) continue;
        const r = index.get(card.dataset.raceId);
        if (r) card.querySelector('.race-body')?.append(createPanel(r));
      }
    }
    const target = document.querySelector('#races');
    if (target) new MutationObserver(update).observe(target, {childList: true});
    update();
  })();
}
