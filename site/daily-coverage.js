'use strict';
/* Show clearly attributed pre-close NEXUS forecasts when a dedicated model has
   not published today's races. Never change a dedicated forecast or its ROI. */
(() => {
  const MODELS = {
    current: {id:'current471_joint_consensus_MAIN4_20261031',name:'現行モデル'},
    high_roi: {id:'ROI9891_fixed_EX_20261102',name:'高回収率モデル'},
    ex: {id:'ROI9234_MAIN6_fixed_EX_20261101',name:'EX予想'}
  };
  const mode = document.body?.dataset.coverageModel, config = MODELS[mode];
  if (!config) return;
  const today = () => new Date(Date.now()+9*3600_000).toISOString().slice(0,10);
  const esc = v => String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const stamp = v => v ? new Date(v).toLocaleString('ja-JP',{timeZone:'Asia/Tokyo',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '—';
  const deadline = v => v ? new Date(v).toLocaleTimeString('ja-JP',{timeZone:'Asia/Tokyo',hour:'2-digit',minute:'2-digit'}) : '—';
  const fmt = n => Number(n).toLocaleString('ja-JP');
  const label = {MAIN:'本線',MEDIUM:'中穴',HIGH:'大穴',FAVORITE:'本命'};
  const sessions = {morning:'モーニング',day:'デイ',night:'ナイト',midnight:'ミッドナイト'};
  const repo = 'https://raw.githubusercontent.com/rokuda6898-jpg/keirin-ai-auto/main/outputs/';
  function read(url) {
    return fetch(url+(url.includes('?')?'&':'?')+'t='+Date.now(),{
      cache:'no-store',signal:AbortSignal.timeout(16000)
    }).then(r=>{if(!r.ok)throw Error('HTTP '+r.status);return r.json();});
  }
  async function candidates(urls) {
    const results=await Promise.allSettled(urls.map(read));
    return results.filter(r=>r.status==='fulfilled').map(r=>r.value);
  }
  function ticketList(blocks) {
    const ts = new Set();
    for (const b of blocks) {
      if(!Array.isArray(b.first)||!Array.isArray(b.second)||!Array.isArray(b.third))return [];
      for (const a of b.first) for (const c of b.second) for (const d of b.third)
        if(new Set([a,c,d]).size===3)ts.add([a,c,d].join('-'));
    }
    return [...ts].sort();
  }
  function safeRace(r,day) {
    const p=r?.prediction, ev=p?.replay_evidence;
    if (!p || r.date!==day || !new RegExp('^\\d{4}'+day.replaceAll('-','')+'$').test(String(r.race_id))
        || p.model_id!=='nexus-formation-5385-v1' || ev?.evidence_kind!=='prospective_preclose_inference') return false;
    const close=Date.parse(r.close_at),captured=Date.parse(p.captured_at),inputs=Date.parse(p.input_captured_at);
    if(!Number.isFinite(close)||!Number.isFinite(captured)||!Number.isFinite(inputs)||captured>=close||inputs>=close)return false;
    if(!Array.isArray(p.blocks)||p.blocks.length>4||p.blocks.length<1||!Array.isArray(p.tickets))return false;
    const tickets=ticketList(p.blocks);
    return tickets.length===p.count && p.tickets.length===p.count
      && JSON.stringify(tickets)===JSON.stringify([...new Set(p.tickets)].sort());
  }
  function extractMain(d,day) {
    if(!d||!Array.isArray(d.races))return new Set();
    return new Set(d.races.filter(r=>r.date===day&&r.forecast?.model_id===config.id)
      .map(r=>String(r.race_id)));
  }
  function getSession(r) {
    if(sessions[r.session])return r.session;
    const hour=Number(new Date(r.start_at).toLocaleString('en-GB',{timeZone:'Asia/Tokyo',hour:'2-digit',hourCycle:'h23'}));
    return hour<11?'morning':hour<16?'day':hour<20?'night':'midnight';
  }
  function renderItem(r) {
    const p=r.prediction;
    const no=(r.race_no||Number(String(r.race_id).slice(0,2)));
    const formation=p.blocks.map((b,i)=>'<div class="daily-formation"><small>0'+(i+1)+'</small><span>'+esc(b.first.join('・'))+'</span><b>→</b><span>'+esc(b.second.join('・'))+'</span><b>→</b><span>'+esc(b.third.join('・'))+'</span></div>').join('');
    const marks=(r.riders||[]).map(car=>{
      const id=Number(car.car??car.car_no??car.number);
      return {id,name:car.name,mark:p.marks?.[id]};
    }).filter(x=>Number.isFinite(x.id)&&x.id>0).sort((a,b)=>a.id-b.id)
      .map(x=>'<span>'+esc(x.id)+'番 '+esc(x.mark||'印なし')+' '+esc(x.name||'')+'</span>').join('');
    const closed=Date.parse(r.close_at)<=Date.now();
    return '<details class="daily-race"><summary><strong>'+esc(r.venue)+' '+esc(no)+'R</strong><span>'+esc(sessions[getSession(r)])+' ／ 締切 '+deadline(r.close_at)+'</span><b>'+esc(label[p.category]||'NEXUS')+' '+fmt(p.count)+'点</b><span>'+ (closed?'締切済・保存済み':'締切前の保存予想') +'</span></summary>'+
      '<div class="daily-detail"><div class="daily-forms"><p><strong>NEXUS保存フォーメーション</strong>（着順指定）</p>'+formation+'</div>'+
      '<div class="daily-stake">重複除外 '+fmt(p.count)+'点 × 100円＝'+fmt(p.count*100)+'円</div>'+
      '<details><summary>全'+fmt(p.count)+'点の買い目</summary><p class="daily-tickets">'+p.tickets.map(esc).join(' ／ ')+'</p></details>'+
      (marks?'<div class="daily-marks"><strong>選手・予想印</strong><div>'+marks+'</div></div>':'')+
      '<p class="daily-when">元予想の入力保存 '+stamp(p.captured_at)+' ／ 適用履歴 '+esc(p.history_through||'—')+'まで'+
      ' ／ 元モデル '+esc(p.model_id)+'</p></div></details>';
  }
  let widget=null,cache=[],filterVenue='all',filterSession='all';
  function list() {
    if(!widget)return;
    const chosen=cache.filter(r=>(filterVenue==='all'||r.venue===filterVenue)
      &&(filterSession==='all'||getSession(r)===filterSession));
    widget.querySelector('#daily-counter').textContent='絞り込み '+chosen.length+' / '+cache.length+'レース';
    widget.querySelector('#daily-items').innerHTML=chosen.map(renderItem).join('')||'<p>この条件のレースはありません。</p>';
  }
  async function refresh() {
    const day=today(),part=mode==='high_roi'?'high_roi':mode;
    const [forecasts,reference]=await Promise.all([
      candidates(['original/'+part+'_live.json',repo+part+'_live.json']),
      candidates(['original/nexus_forecasts/'+day+'.json',repo+'nexus_forecasts/'+day+'.json'])
    ]);
    const actual=forecasts.sort((a,b)=>Date.parse(b.generated_at||0)-Date.parse(a.generated_at||0))[0];
    const dedicated=extractMain(actual,day);
    const refs=reference.filter(Array.isArray).sort((a,b)=>
      b.filter(r=>safeRace(r,day)).length-a.filter(r=>safeRace(r,day)).length);
    const saved=refs[0]?.filter(r=>safeRace(r,day))||[];
    const missing=saved.filter(r=>!dedicated.has(String(r.race_id)));
    if(!saved.length){
      // Keep a truthful error instead of silently reusing yesterday's record.
      if(widget){widget.remove();widget=null;}
      if(dedicated.size===0){
        const notice=document.getElementById('notice');
        if(notice)notice.textContent=day.replaceAll('-','/')+'の専用予想を確認できません。NEXUSの保存予想も取得できませんでした。';
      }
      return;
    }
    if(!missing.length){if(widget){widget.remove();widget=null;}return;}
    cache=missing.sort((a,b)=>Date.parse(a.close_at)-Date.parse(b.close_at)||String(a.race_id).localeCompare(String(b.race_id)));
    if(!widget) {
      widget=document.createElement('section');
      widget.id='daily-reference';
      widget.className='daily-reference';
      widget.setAttribute('aria-label','本日の未掲載レースとNEXUS参考予想');
      document.getElementById('races')?.before(widget);
    }
    if(!widget.parentElement)return;
    const venues=[...new Set(cache.map(r=>r.venue))];
    widget.innerHTML='<div class="daily-reference-heading"><strong>本日分の専用予想が未掲載です</strong><span>'+esc(day.replaceAll('-','/'))+'</span></div>'+
      '<p>'+esc(config.name)+'で締切前に公開済みの予想：<b>'+dedicated.size+' / '+saved.length+'レース</b>。未掲載の'+missing.length+'レースについて、別モデルの保存済み予想を参考表示しています。</p>'+
      '<p class="daily-disclaimer"><strong>重要：</strong>以下は専用の'+esc(config.name)+'ではなく、NEXUSの事前保存予想です。EX補助評価、専用方式の8・10・18点ルール、旧方式の回収率実績は適用されません。元データの予想を後付け作成したものではありません。</p>'+
      '<div class="daily-controls"><label>開催場 <select id="daily-venue"><option value="all">すべて</option>'+
      venues.map(v=>'<option value="'+esc(v)+'">'+esc(v)+'</option>').join('')+'</select></label>'+
      '<label>時間帯 <select id="daily-session"><option value="all">すべて</option>'+
      Object.entries(sessions).map(([k,v])=>'<option value="'+k+'">'+v+'</option>').join('')+'</select></label>'+
      '<span id="daily-counter"></span></div><div id="daily-items"></div>'+
      '<p class="daily-source">当日の専用予想が公開されると、対応するレースの参考表示は自動で非表示になります。<a href="previous.html">NEXUS予想ページ</a></p>';
    filterVenue=venues.includes(filterVenue)?filterVenue:'all';
    widget.querySelector('#daily-venue').value=filterVenue;
    widget.querySelector('#daily-session').value=filterSession;
    widget.querySelector('#daily-venue').addEventListener('change',e=>{filterVenue=e.target.value;list();});
    widget.querySelector('#daily-session').addEventListener('change',e=>{filterSession=e.target.value;list();});
    list();
  }
  refresh().catch(e=>{const status=document.getElementById('notice');if(status&&!status.textContent.includes('読み込めません'))status.setAttribute('data-reference-error',String(e?.name||'fetch-failed'));});
  setInterval(()=>{if(!document.hidden)refresh().catch(()=>{});},180000);
  if(typeof module!=='undefined'&&module.exports)module.exports={safeRace,ticketList,extractMain};
})();