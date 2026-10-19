(function(root){
  'use strict';
  const expand = blocks => [...new Set(blocks.flatMap(b => b.first.flatMap(a => b.second.flatMap(c => b.third.filter(d => new Set([a,c,d]).size === 3).map(d => [a,c,d].join('-'))))))].sort();
  const openPodium = p => Boolean(p?.tickets?.length && new Set(p.tickets.map(t => String(t).split('-')[1])).size >= 2);
  const closed = (r,now=Date.now()) => Boolean(r.closed || r.close_at && now >= Date.parse(r.close_at));
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
      const settled=r.settlement?.[mode];
      if(settled){
        if(settled.void){out.void++;continue;}
        out.settled++;out.hits+=Number(Boolean(settled.hit));out.stake+=Number(settled.net_stake??(settled.stake-(settled.refund||0)));out.returned+=Number(settled.winnings??(settled.returned-(settled.refund||0)));out.refund+=Number(settled.refund||0);
      }else if(mode==='flat' || allocation(p,mode))out.pending++;
      else out.unavailable++;
    }
    out.hitRate=out.settled?out.hits/out.settled:null;out.roi=out.stake?out.returned/out.stake:null;out.profit=out.returned-out.stake;return out;
  }
  const api={expand,openPodium,closed,allocation,packTickets,amountGroups,totals};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.NexusCore=api;
})(typeof globalThis==='undefined'?this:globalThis);
