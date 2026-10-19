(function(root){
  'use strict';
  const expand = blocks => [...new Set(blocks.flatMap(b => b.first.flatMap(a => b.second.flatMap(c => b.third.filter(d => new Set([a,c,d]).size === 3).map(d => [a,c,d].join('-'))))))].sort();
  const openPodium = p => Boolean(p?.tickets?.length && new Set(p.tickets.map(t => String(t).split('-')[1])).size >= 2);
  const closed = (r,now=Date.now()) => Boolean(r.closed || r.close_at && now >= Date.parse(r.close_at));
  const epoch = v => typeof v === 'number' ? v * (v < 1e12 ? 1000 : 1) : Date.parse(v);
  const validTicket = t => typeof t === 'string' && /^[1-9]-[1-9]-[1-9]$/.test(t) && new Set(t.split('-')).size === 3;
  function confidence(p){
    if(!p?.tickets?.length)return null;
    const c=p.confidence,m=c?.model_mass;
    if(c?.status==='assessed' && c.policy_id==='selected-mass-v1' && typeof m==='number' && Number.isFinite(m) && m>=0 && m<=1){
      return {...c,grade:m>=0.65?'S':m>=0.45?'A':'B',assessed:true};
    }
    return {grade:'B',status:'insufficient',assessed:false,reason:'自信度の根拠指標が保存されていないため未評価です。'};
  }
  // The conditional record is an alternative model's genuine pre-close forecast.
  // It must never replace the primary prediction or enter its settlement totals.
  function conditionalPrediction(r){
    const tickets=r?.shadow?.tickets,close=epoch(r?.close_at),saved=epoch(r?.shadow?.snapshot_at);
    if(!Array.isArray(tickets)||!tickets.length||tickets.some(t=>!validTicket(t))||new Set(tickets).size!==tickets.length||!Number.isFinite(close)||!Number.isFinite(saved)||saved>=close)return null;
    const b=r.budget,allocations={},allocated=b?.status==='allocated',budget=b?.total_stake_yen;
    const marks=Array.isArray(r.shadow.marks)?r.shadow.marks:[];
    const p={source_model:'conditional',model_id:'conditional_trifecta_v1',category:'CONDITIONAL',count:tickets.length,tickets:[...tickets],blocks:packTickets(tickets),marks:Object.fromEntries(marks.filter(m=>m&&['◎','○','▲','△','☆'].includes(m.mark)).map(m=>[m.car,m.mark])),captured_at:r.shadow.snapshot_at,confidence:r.shadow.confidence,allocations,allocation_policy:'conditional-saved',conditional_budget_status:b?.status||'unavailable',odds:b?.snapshot_odds||{},odds_at:b?.odds_captured_at_jst,allocation_saved_at:b?.allocation_frozen_at_jst};
    const oddsAt=epoch(b?.odds_captured_at_jst),allocatedAt=epoch(b?.allocation_frozen_at_jst);
    if(allocated && Number.isInteger(budget) && budget>=5000 && budget<=10000 && budget%100===0 && Number.isFinite(oddsAt) && oddsAt<close && Number.isFinite(allocatedAt) && allocatedAt<close && Array.isArray(b.allocations) && b.allocations.length===tickets.length && b.allocations.every(a=>a && typeof a==='object') && new Set(b.allocations.map(a=>a.buy)).size===tickets.length){
      const stakes=Object.fromEntries(b.allocations.map(a=>[a.buy,a.stake_yen]));
      allocations[String(budget)]={available:true,quote_break_even:Number.isFinite(b.minimum_payout_yen)&&b.minimum_payout_yen>=budget,budget,minimum_return:b.minimum_payout_yen,stakes};p.auto_budget=budget;
      if(!allocation(p,'auto')){delete allocations[String(budget)];delete p.auto_budget;}
    }
    return p;
  }
  function mergeForecastRows(primary,conditional){
    const rows=(primary.races||[]).map(r=>({...r})),byId=new Map(rows.map(r=>[r.race_id,r]));
    const days=new Set(rows.map(r=>r.date));days.add(primary.date);if(conditional?.schedule_date)days.add(conditional.schedule_date);
    for(const cr of conditional?.races||[]){
      if(!cr||!days.has(cr.date)||!cr.id)continue;
      let r=byId.get(cr.id);
      if(r && (r.date!==cr.date||Number(r.race_no)!==Number(cr.number)||r.venue!==cr.venue))continue;
      const p=conditionalPrediction(cr);
      if(!r){
        const start=epoch(cr.start_at),close=epoch(cr.close_at);
        r={race_id:cr.id,date:cr.date,venue:cr.venue,race_no:cr.number,start_at:Number.isFinite(start)?new Date(start).toISOString():null,close_at:Number.isFinite(close)?new Date(close).toISOString():null,riders:cr.riders||[],session:cr.session||cr.meeting_time_band||null,source_url:cr.budget?.source_url};
        byId.set(cr.id,r);rows.push(r);
      }
      r.conditional_prediction=p;
      // Source outcomes are shown without creating a primary-model settlement.
      if(!r.winners?.length && cr.actual?.length){r.winners=[...cr.actual];r.payouts={...cr.payouts};r.result_captured_at=cr.result_observed_at;}
    }
    const meetings=new Map();
    for(const r of rows){const k=r.date+'|'+r.venue;if(!meetings.has(k))meetings.set(k,[]);meetings.get(k).push(r);}
    for(const group of meetings.values()){
      let band=group.map(r=>r.session).find(s=>['morning','day','night','midnight'].includes(s));
      if(!band){
        const times=group.map(r=>epoch(r.start_at)).filter(Number.isFinite).sort((a,b)=>a-b),minutes=t=>{const d=new Date(t+9*3600000);return d.getUTCHours()*60+d.getUTCMinutes();};
        if(times.length){const first=minutes(times[0]),last=minutes(times.at(-1));band=first>=1230||last>=1320?'midnight':last>=1020?'night':first<660&&last<900?'morning':'day';}
      }
      for(const r of group)if(!['morning','day','night','midnight'].includes(r.session))r.session=band||'unknown';
    }
    return rows;
  }
  const settlement = (r,mode) => r.result_complete===false && !r.refund_all && !r.cancelled && !r.settlement?.flat?.void ? null : r.settlement?.[mode]||null;
  function resultState(r){
    if(r.refund_all || r.cancelled || r.settlement?.flat?.void)return 'void';
    if(r.result_complete===true || r.result_complete!==false && (r.settlement?.flat || r.winners?.length && r.winners.every(t=>Number.isInteger(r.payouts?.[t]) && r.payouts[t]>0)))return 'confirmed';
    return r.winners?.length?'payout_pending':'pending';
  }
  function allocation(p,mode){
    if(!p)return null;
    const key = mode === 'auto' ? p.auto_budget : Number(mode);
    if(!key)return null;
    const a=p.allocations?.[String(key)];
    if(!a?.available || a.quote_break_even !== true || a.minimum_return < Number(key))return null;
    const stakes=a.stakes||{}, keys=Object.keys(stakes);
    if(keys.length!==p.tickets.length || keys.some(t=>!p.tickets.includes(t)) || p.tickets.some(t=>!Number.isInteger(stakes[t]) || stakes[t]<100 || stakes[t]%100) || Object.values(stakes).reduce((s,n)=>s+n,0)!==Number(key))return null;
    return {...a,key:String(key)};
  }
  // Each amount group is packed independently, preserving every original ticket.
  function packTickets(tickets){
    let blocks=[...new Set(tickets)].sort().map(t=>{const [a,b,c]=t.split('-').map(Number);return {first:[a],second:[b],third:[c]};});
    for(;;){
      let changed=false;
      outer:for(let i=0;i<blocks.length;i++)for(let j=i+1;j<blocks.length;j++){
        const merged=Object.fromEntries(['first','second','third'].map(k=>[k,[...new Set([...blocks[i][k],...blocks[j][k]])].sort((a,b)=>a-b)]));
        const before=expand([blocks[i],blocks[j]]),after=expand([merged]);
        if(before.length!==after.length || before.some((t,k)=>t!==after[k]))continue;
        blocks.splice(j,1);blocks[i]=merged;changed=true;break outer;
      }
      if(!changed)return blocks.map(b=>({...b,count:expand([b]).length}));
    }
  }
  function amountGroups(stakes){
    const by=new Map();
    for(const [ticket,amount] of Object.entries(stakes||{})){if(!by.has(amount))by.set(amount,[]);by.get(amount).push(ticket);}
    return [...by].sort(([a],[b])=>a-b).map(([amount,tickets])=>({amount,count:tickets.length,blocks:packTickets(tickets)}));
  }
  function totals(rows,mode){
    const out={hits:0,settled:0,stake:0,returned:0,refund:0,pending:0,unavailable:0,unpredicted:0,void:0};
    for(const r of rows){
      const p=r.prediction;
      if(!p){out.unpredicted++;continue;}
      const settled=settlement(r,mode);
      if(settled){
        if(settled.void){out.void++;continue;}
        out.settled++;out.hits+=Number(Boolean(settled.hit));out.stake+=Number(settled.net_stake??(settled.stake-(settled.refund||0)));out.returned+=Number(settled.winnings??(settled.returned-(settled.refund||0)));out.refund+=Number(settled.refund||0);
      }else if(mode==='flat' || allocation(p,mode))out.pending++;
      else out.unavailable++;
    }
    out.hitRate=out.settled?out.hits/out.settled:null;out.roi=out.stake?out.returned/out.stake:null;out.profit=out.returned-out.stake;return out;
  }
  const latestFeed=feeds=>feeds.filter(f=>f?.schema===1&&Array.isArray(f.races)).sort((a,b)=>(Date.parse(b.updated_at)||0)-(Date.parse(a.updated_at)||0)||(Date.parse(b.policy_updated_at)||0)-(Date.parse(a.policy_updated_at)||0)||(Date.parse(b.results_updated_at)||0)-(Date.parse(a.results_updated_at)||0)||(Date.parse(b.confidence_updated_at)||0)-(Date.parse(a.confidence_updated_at)||0))[0]||null;
  const api={expand,openPodium,closed,confidence,conditionalPrediction,mergeForecastRows,resultState,settlement,allocation,packTickets,amountGroups,totals,latestFeed};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.NexusCore=api;
})(typeof globalThis==='undefined'?this:globalThis);
