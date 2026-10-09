'use strict';
(()=>{
 const root=document.currentScript.dataset.root||'./';
 const items=[['今日の予想','index.html'],['本線・穴','original/index.html'],['独立予想','independent.html'],['会社','company.html'],['結果・保管庫','archive.html']];
 const nav=document.createElement('nav');nav.className='nexus-bottom-nav';nav.setAttribute('aria-label','サイト共通メニュー');
 items.forEach(([label,path])=>{const a=document.createElement('a');a.textContent=label;a.href=root+path;if(new URL(a.href).pathname===location.pathname||(path==='index.html'&&location.pathname===new URL(root,location.href).pathname))a.setAttribute('aria-current','page');nav.append(a);});document.body.append(nav);
 document.querySelectorAll('table').forEach(table=>{if(table.parentElement.classList.contains('nexus-table-scroll'))return;const wrap=document.createElement('div');wrap.className='nexus-table-scroll';wrap.tabIndex=0;wrap.setAttribute('role','region');wrap.setAttribute('aria-label','表。横にスクロールできます');table.before(wrap);wrap.append(table);});
 const top=document.createElement('button');top.type='button';top.className='nexus-back-top';top.textContent='↑';top.setAttribute('aria-label','ページの先頭へ');top.addEventListener('click',()=>window.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'}));document.body.append(top);
})();
