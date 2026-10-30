'use strict';
(() => {
  const root = document.currentScript.dataset.root || './';
  const items = [['モデルを選ぶ', 'index.html#model-switcher'], ['従来の予想', 'previous.html#forecast'], ['従来の結果', 'previous.html#results']];
  const nav = document.createElement('nav');
  nav.className = 'nexus-bottom-nav';
  nav.setAttribute('aria-label', 'サイト共通メニュー');
  for (const [label, path] of items) {
    const a = document.createElement('a');
    a.textContent = label; a.href = root + path; nav.append(a);
  }
  const update = () => {
    const home = new URL(root, location.href).pathname;
    const isHome = location.pathname === home || location.pathname === home + 'index.html' || location.pathname === home + 'previous.html';
    const hash = location.hash || '#forecast';
    for (const a of nav.querySelectorAll('a')) {
      if (isHome && new URL(a.href).hash === hash) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    }
  };
  update(); window.addEventListener('hashchange', update); document.body.append(nav);
  document.querySelectorAll('table').forEach(table => {
    if (table.parentElement.classList.contains('nexus-table-scroll')) return;
    const wrap = document.createElement('div'); wrap.className = 'nexus-table-scroll'; wrap.tabIndex = 0;
    wrap.setAttribute('role', 'region'); wrap.setAttribute('aria-label', '表。横にスクロールできます'); table.before(wrap); wrap.append(table);
  });
  const top = document.createElement('button'); top.type = 'button'; top.className = 'nexus-back-top'; top.textContent = '↑';
  top.setAttribute('aria-label', 'ページの先頭へ');
  top.addEventListener('click', () => window.scrollTo({top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'}));
  document.body.append(top);
})();
