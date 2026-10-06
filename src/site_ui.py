"""Shared, accessible mobile presentation for the company keirin site."""
import html
import json
import math
import re
from pathlib import Path
from bs4 import BeautifulSoup
from common import OUTPUT_DIR

CSS = """
:root{--ink:#18324f;--blue:#247be6;--muted:#6b7e95;--line:#dfE8f2}body{margin:0;background:#f5f8fc;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.6}main{max-width:900px;margin:auto;padding:18px 14px 112px!important}header{background:#fff!important;color:var(--ink)!important;border-bottom:1px solid var(--line)!important;padding:16px 20px!important}.brand{background:none!important;color:var(--ink)!important;box-shadow:none!important;font-size:24px!important;letter-spacing:.08em!important;padding:0!important;min-width:0!important}.brand:before{content:'N';background:#247be6;border-radius:10px;color:#fff;padding:2px 9px;margin-right:10px;font-size:25px}.hero{background:#fff!important;color:var(--ink)!important;border:1px solid var(--line);border-radius:20px!important;box-shadow:none!important}.hero p{opacity:1!important;color:var(--muted);margin:6px 0!important}.hero a{color:var(--blue)!important}.eyebrow{color:var(--blue)!important}.trust{background:#f0f6ff!important;border-color:#dbe8fa!important}.top-nav{display:flex!important;border:0!important;gap:8px!important;flex-wrap:wrap}.top-nav a{border:1px solid var(--line);border-radius:10px;padding:9px 12px!important;font-size:13px}.mobile-nav{position:fixed;bottom:12px;left:50%;transform:translateX(-50%);width:calc(100% - 24px);max-width:680px;display:grid!important;grid-template-columns:repeat(4,1fr);background:#ffffffef!important;backdrop-filter:blur(12px);border:1px solid #d5e1ef!important;border-radius:22px;box-shadow:0 6px 30px #244b7420;padding:8px!important;z-index:100;margin:0!important}.mobile-nav a{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:7px 3px!important;font-weight:800;color:var(--muted)!important;text-decoration:none;font-size:14px;min-height:48px}.mobile-nav a.active{color:var(--blue)!important;background:#edf5ff;border-radius:14px}.mobile-nav small{font-size:9px;letter-spacing:.12em;display:block}.car-box{display:inline-grid;place-items:center;width:45px;height:48px;border-radius:12px;border:1px solid #cdd5df;font-size:27px;font-weight:900;background:#fff;color:#162237;box-shadow:0 3px 8px #10233f10;flex-shrink:0}.car-box.n2{background:#202126;color:#fff}.car-box.n3{background:#e72d42;color:#fff}.car-box.n4{background:#247be6;color:#fff}.car-box.n5{background:#f2d538;color:#15213b}.car-box.n6{background:#22985c;color:#fff}.car-box.n7{background:#f68a29;color:#162237}.car-box.n8{background:#ec75b0;color:#162237}.car-box.n9{background:#9664c9;color:#fff}.ticket-numbers{display:flex;align-items:center;gap:8px;flex-wrap:nowrap}.ticket-dash{color:#8493a6;font-size:21px}.ticket-card{display:grid!important;grid-template-columns:1fr auto!important;gap:10px!important;border:1px solid var(--line);border-radius:17px;padding:16px!important;margin:10px 0!important;background:#fff}.ticket-card.hole{background:#fffbf1;border-color:#ebd8ad}.ticket-card>b{grid-column:1/-1;font-size:12px!important;color:var(--blue)!important}.ticket-card>strong{font-size:inherit!important}.ticket-card>.ticket-amount{font-weight:900;font-size:20px;white-space:nowrap;align-self:center}.ticket-card>small{grid-column:1/-1!important;color:var(--muted)!important;font-size:13px;letter-spacing:0}.race-actions button{min-height:46px}.race-actions button.active{background:var(--blue)!important;color:#fff!important}.race.open{border-color:#a5c9f3}.history-intro{font-size:13px;color:var(--muted)}.history-filters{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin:20px 0 12px}.history-filters label{font-size:12px;font-weight:700;color:var(--muted)}.history-filters select{width:100%;padding:11px 8px;background:#fff;border:1px solid var(--line);border-radius:11px;color:var(--ink);font:inherit;min-height:44px}.history-count{font-size:13px;color:var(--muted);margin:8px 0}.history-race{background:#fff;border:1px solid var(--line);border-radius:18px;margin:12px 0;overflow:hidden}.history-race summary{cursor:pointer;padding:16px;list-style:none}.history-race summary::-webkit-details-marker{display:none}.history-race summary:after{content:'買い目を見る ＋';display:block;color:var(--blue);font-size:12px;margin-top:9px}.history-race[open] summary:after{content:'買い目を閉じる −'}.race-title{display:flex;align-items:center;justify-content:space-between;gap:10px}.race-title b{font-size:18px}.race-date{color:var(--muted);font-size:12px}.result-badge{padding:5px 10px;border-radius:99px;font-size:12px;font-weight:800;white-space:nowrap;background:#eef3f9;color:var(--muted)}.result-badge.hit{color:#11774a;background:#e8f7ee}.result-badge.miss{color:#b63744;background:#fff0f1}.race-money{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:13px}.race-money small{display:block;color:var(--muted);font-size:11px}.race-money strong{font-size:16px;white-space:nowrap}.history-tickets{border-top:1px solid var(--line);padding:4px 14px 10px}.history-ticket{padding:13px 0;border-bottom:1px solid #edf2f7}.history-ticket:last-child{border-bottom:0}.history-ticket-top{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}.history-ticket-top small{color:var(--muted)}.history-ticket .car-box{width:35px;height:39px;font-size:23px;border-radius:9px}.ticket-money{font-size:12px;color:var(--muted);margin-top:9px;display:flex;gap:14px;flex-wrap:wrap}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.stats div{padding:13px;background:#fff;border:1px solid var(--line);border-radius:14px}.stats small{display:block;color:var(--muted);font-size:11px}.stats b{font-size:20px}[hidden]{display:none!important}.history-more{display:block;margin:18px auto;border:1px solid #a8c7ed;background:#edf5ff;border-radius:12px;color:var(--blue);padding:12px 26px;font-weight:800;min-height:44px}.empty{padding:24px;text-align:center;background:#fff;border-radius:16px}button,a,summary,select{touch-action:manipulation}a:focus-visible,button:focus-visible,summary:focus-visible,select:focus-visible{outline:3px solid #247be6;outline-offset:3px}@media(max-width:420px){.car-box{width:38px;height:43px;font-size:25px}.ticket-card{padding:13px!important}.ticket-card>.ticket-amount{font-size:18px}.history-filters{gap:6px}.race-money strong{font-size:14px}.hero h1{font-size:23px}.stats b{font-size:18px}.mobile-nav{bottom:max(8px,env(safe-area-inset-bottom))}} 
"""

def assets():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUTPUT_DIR/"site-ui.css").write_text(CSS,encoding="utf-8")

def numbers(ticket):
    text=str(ticket)
    if not re.fullmatch(r"[1-9](?:-[1-9]){1,2}",text):
        return html.escape(text)
    return '<span class="ticket-numbers">'+('<span class="ticket-dash">−</span>'.join(f'<span class="car-box n{n}">{n}</span>' for n in text.split('-')))+'</span>'

def nav(active="today"):
    links=[("today","index.html","今日","RACE"),("picks","index.html#picks","買い目","PICKS"),("company","company/annual_department_report.html","会社","COMPANY"),("history","history.html","履歴","HISTORY")]
    return '<nav class="mobile-nav" aria-label="メインメニュー">'+''.join(f'<a class="{"active" if key==active else ""}" href="{url}">{label}<small>{en}</small></a>' for key,url,label,en in links)+'</nav>'

def enhance_today(document):
    assets()
    soup=BeautifulSoup(document,"html.parser")
    if soup.select_one('link[href="site-ui.css"]'):
        return document
    soup.head.append(soup.new_tag('link',rel='stylesheet',href='site-ui.css'))
    old=soup.select_one('main > nav')
    if old:old['class']=['top-nav']
    for card in soup.select('.bet'):
        card['class']=card.get('class',[])+['ticket-card']
        label=card.find('b')
        if label and '穴' in label.get_text():card['class'].append('hole')
        ticket=card.find('strong')
        if ticket:
            value=ticket.get_text(strip=True);ticket.clear();ticket.append(BeautifulSoup(numbers(value),'html.parser'))
        amount=card.find('span',recursive=False)
        if amount:amount['class']=['ticket-amount']
    soup.body.append(BeautifulSoup(nav(),'html.parser'))
    script=soup.new_tag('script')
    script.string="""
function activatePicks(){var card=document.querySelector('.race.venue-visible');if(!card)return;card.classList.add('open');var btn=card.querySelector('.race-actions button');if(btn)showPanel(btn,'picks');}
document.addEventListener('DOMContentLoaded',function(){var first=document.querySelector('#venueJump button');if(first)first.click();if(location.hash==='#picks')activatePicks();document.querySelectorAll('.race-select').forEach(function(btn){btn.addEventListener('click',function(){var card=btn.closest('.race');if(card.classList.contains('open')){var pick=card.querySelector('.race-actions button');if(pick)showPanel(pick,'picks');}})});document.querySelectorAll('.race-actions button').forEach(function(btn){btn.addEventListener('click',function(){btn.parentNode.querySelectorAll('button').forEach(function(b){b.classList.toggle('active',b===btn)})})})});window.addEventListener('hashchange',function(){if(location.hash==='#picks')activatePicks()});
"""
    soup.body.append(script)
    return str(soup)

def numeric(value):
    try:
        number=float(value)
        return number if math.isfinite(number) else None
    except (ValueError,TypeError):return None

def money(value, signed=False):
    number=numeric(value)
    return '—' if number is None else (f'{number:+,.0f}円' if signed else f'{number:,.0f}円')

def history_document(settled):
    assets()
    view=settled.copy()
    if "is_prospective" in view:
        view=view[view["is_prospective"].fillna(False).eq(True)].copy()
    else:
        view=view.iloc[0:0]
    if "prediction_created_at_jst" in view and "race_id" in view:
        import pandas as pd
        view["_snapshot"]=pd.to_datetime(view["prediction_created_at_jst"],errors="coerce",utc=True)
        latest=view.groupby("race_id")["_snapshot"].transform("max")
        view=view[view["_snapshot"].eq(latest) | latest.isna()].copy()
    if {"race_id","bet_type","buy"}.issubset(view.columns):
        view=view.drop_duplicates(["race_id","bet_type","buy"],keep="last")
    rows=view.to_dict('records')
    groups={}
    for row in rows:
        if not bool(row.get('is_prospective',False)):continue
        key=(str(row.get('date','')),str(row.get('venue','')),str(row.get('race_no','')),str(row.get('race_id','')))
        groups.setdefault(key,[]).append(row)
    races=[]
    for (day,venue,no,rid),tickets in sorted(groups.items(),key=lambda item:(item[0][0],item[0][1],numeric(item[0][2]) or 0),reverse=True):
        decided=all(bool(t.get('is_decided',False)) for t in tickets)
        hit=any(bool(t.get('is_hit',False)) and bool(t.get('is_decided',False)) for t in tickets)
        status='hit' if hit else ('miss' if decided else 'pending')
        stake=sum(numeric(t.get('stake_yen')) or 0 for t in tickets)
        ret=sum(numeric(t.get('actual_return_yen')) or 0 for t in tickets) if decided else None
        profit=ret-stake if ret is not None else None
        cards=[]
        for t in tickets:
            t_decided=bool(t.get('is_decided',False))
            tstatus='hit' if t_decided and bool(t.get('is_hit',False)) else ('miss' if t_decided else 'pending')
            label={'hit':'的中','miss':'不的中','pending':'結果待ち'}[tstatus]
            cards.append('<div class="history-ticket"><div class="history-ticket-top"><small>'+html.escape(str(t.get('bet_label') or t.get('bet_type') or '買い目'))+'</small><span class="result-badge '+tstatus+'">'+label+'</span></div>'+numbers(t.get('buy',''))+'<div class="ticket-money"><span>配分 '+money(t.get('stake_yen'))+'</span><span>払戻 '+(money(t.get('actual_return_yen')) if t_decided else '—')+'</span><span>損益 '+(money(t.get('actual_profit_yen'),True) if t_decided else '—')+'</span></div></div>')
        label={'hit':'的中あり' if decided else '的中あり・一部結果待ち','miss':'不的中','pending':'結果待ち'}[status]
        title=html.escape(venue)+' '+html.escape(str(int(float(no))))+'R'
        card='<details class="history-race"><summary><div class="race-date">'+html.escape(day)+' · '+str(len(tickets))+'点</div><div class="race-title"><b>'+title+'</b><span class="result-badge '+status+'">'+label+'</span></div><div class="race-money"><div><small>配分額・試算</small><strong>'+money(stake)+'</strong></div><div><small>払戻額</small><strong>'+money(ret)+'</strong></div><div><small>損益</small><strong>'+money(profit,True)+'</strong></div></div></summary><div class="history-tickets">'+''.join(cards)+'</div></details>'
        races.append({'date':day,'venue':venue,'status':status,'html':card,'decided':decided,'hit':hit,'stake':stake,'return':ret})
    payload=json.dumps(races,ensure_ascii=False).replace('<',r'\u003c').replace('>',r'\u003e').replace('&',r'\u0026')
    return """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>NEXUS | 買い目履歴</title><link rel="stylesheet" href="site-ui.css"></head><body><header><div class="brand">NEXUS</div></header><main><h1>買い目の履歴</h1><p class="history-intro">各レースの最後に保存した締切前予想の検証結果です。配分額・払戻額は試算で、実購入の成績ではありません。</p><a href="performance.html">的中率・回収率の詳しい集計</a><div class="history-filters"><label>日付<select id="historyDate"><option value="">すべての日</option></select></label><label>競輪場<select id="historyVenue"><option value="">すべての場</option></select></label><label>結果<select id="historyStatus"><option value="">すべて</option><option value="hit">的中あり</option><option value="miss">不的中</option><option value="pending">結果待ち</option></select></label></div><section class="stats" id="historyStats"></section><p class="history-count" id="historyCount" aria-live="polite"></p><div id="historyList"></div><button class="history-more" id="historyMore" type="button">もっと見る</button><noscript>履歴の絞り込みにはJavaScriptが必要です。<a href="prediction_history.csv">履歴CSVを開く</a></noscript></main>"""+nav('history')+'<script type="application/json" id="historyData">'+payload+"""</script><script>
const historyRaces=JSON.parse(document.getElementById('historyData').textContent);let historyLimit=30;
function historyOptions(id,values){let box=document.getElementById(id);values.forEach(value=>{let option=document.createElement('option');option.value=value;option.textContent=value;box.appendChild(option)})}
historyOptions('historyDate',[...new Set(historyRaces.map(r=>r.date))]);historyOptions('historyVenue',[...new Set(historyRaces.map(r=>r.venue))].sort());
function renderHistory(){const date=document.getElementById('historyDate').value,venue=document.getElementById('historyVenue').value,status=document.getElementById('historyStatus').value;const rows=historyRaces.filter(r=>(!date||r.date===date)&&(!venue||r.venue===venue)&&(!status||r.status===status));const decided=rows.filter(r=>r.decided),hits=decided.filter(r=>r.hit).length;document.getElementById('historyStats').innerHTML='<div><small>対象レース</small><b>'+rows.length+'R</b></div><div><small>確定レース</small><b>'+decided.length+'R</b></div><div><small>的中レース</small><b>'+hits+'R</b></div>';document.getElementById('historyCount').textContent=rows.length+'レース中 '+Math.min(historyLimit,rows.length)+'レースを表示';document.getElementById('historyList').innerHTML=rows.slice(0,historyLimit).map(r=>r.html).join('')||'<div class="empty">該当する履歴はありません。</div>';document.getElementById('historyMore').hidden=rows.length<=historyLimit;}
['historyDate','historyVenue','historyStatus'].forEach(id=>document.getElementById(id).addEventListener('change',()=>{historyLimit=30;renderHistory()}));document.getElementById('historyMore').addEventListener('click',()=>{historyLimit+=30;renderHistory()});renderHistory();
</script></body></html>"""
