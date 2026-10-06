"""Shared, accessible mobile presentation for the company keirin site."""
import html
import json
import math
import re
from pathlib import Path
from bs4 import BeautifulSoup
from common import OUTPUT_DIR


def odds_provenance(row):
    sources=row.get('odds_sources')
    captured=row.get('odds_captured_at_jst')
    if not isinstance(sources,str) or not sources.strip() or not isinstance(captured,str) or not captured.strip():
        return '<small class="odds-provenance">価格の取得元・時刻は未保存です。</small>'
    from datetime import datetime
    try:
        timestamp=datetime.fromisoformat(captured)
        if timestamp.tzinfo is None:raise ValueError('timezone missing')
    except ValueError:
        return '<small class="odds-provenance">価格の取得時刻を確認できません。</small>'
    names={'winticket':'WINTICKET','kdreams':'Kドリームス','oddspark':'オッズパーク','netkeirin':'netkeirin'}
    labels='・'.join(names.get(name.strip(),name.strip()) for name in sources.split('|'))
    return '<small class="odds-provenance" data-captured="'+html.escape(captured,quote=True)+'">取得元 '+html.escape(labels)+' ／ '+html.escape(timestamp.strftime('%m/%d %H:%M:%S'))+'<span class="odds-age"></span></small>'

CSS = """
:root{--ink:#18324f;--blue:#247be6;--muted:#6b7e95;--line:#dfE8f2}body{margin:0;background:#f5f8fc;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.6}main{max-width:900px;margin:auto;padding:18px 14px 112px!important}header{background:#fff!important;color:var(--ink)!important;border-bottom:1px solid var(--line)!important;padding:16px 20px!important}.brand{background:none!important;color:var(--ink)!important;box-shadow:none!important;font-size:24px!important;letter-spacing:.08em!important;padding:0!important;min-width:0!important}.brand:before{content:'N';background:#247be6;border-radius:10px;color:#fff;padding:2px 9px;margin-right:10px;font-size:25px}.hero{background:#fff!important;color:var(--ink)!important;border:1px solid var(--line);border-radius:20px!important;box-shadow:none!important}.hero p{opacity:1!important;color:var(--muted);margin:6px 0!important}.hero a{color:var(--blue)!important}.eyebrow{color:var(--blue)!important}.trust{background:#f0f6ff!important;border-color:#dbe8fa!important}.top-nav{display:flex!important;border:0!important;gap:8px!important;flex-wrap:wrap}.top-nav a{border:1px solid var(--line);border-radius:10px;padding:9px 12px!important;font-size:13px}.mobile-nav{position:fixed;bottom:12px;left:50%;transform:translateX(-50%);width:calc(100% - 24px);max-width:680px;display:grid!important;grid-template-columns:repeat(4,1fr);background:#ffffffef!important;backdrop-filter:blur(12px);border:1px solid #d5e1ef!important;border-radius:22px;box-shadow:0 6px 30px #244b7420;padding:8px!important;z-index:100;margin:0!important}.mobile-nav a{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:7px 3px!important;font-weight:800;color:var(--muted)!important;text-decoration:none;font-size:14px;min-height:48px}.mobile-nav a.active{color:var(--blue)!important;background:#edf5ff;border-radius:14px}.mobile-nav small{font-size:9px;letter-spacing:.12em;display:block}.car-box{display:inline-grid;place-items:center;width:45px;height:48px;border-radius:12px;border:1px solid #cdd5df;font-size:27px;font-weight:900;background:#fff;color:#162237;box-shadow:0 3px 8px #10233f10;flex-shrink:0}.car-box.n2{background:#202126;color:#fff}.car-box.n3{background:#e72d42;color:#fff}.car-box.n4{background:#247be6;color:#fff}.car-box.n5{background:#f2d538;color:#15213b}.car-box.n6{background:#22985c;color:#fff}.car-box.n7{background:#f68a29;color:#162237}.car-box.n8{background:#ec75b0;color:#162237}.car-box.n9{background:#9664c9;color:#fff}.ticket-numbers{display:flex;align-items:center;gap:8px;flex-wrap:nowrap}.ticket-dash{color:#8493a6;font-size:21px}.ticket-card{display:grid!important;grid-template-columns:1fr auto!important;gap:10px!important;border:1px solid var(--line);border-radius:17px;padding:16px!important;margin:10px 0!important;background:#fff}.ticket-card.hole{background:#fffbf1;border-color:#ebd8ad}.ticket-card>b{grid-column:1/-1;font-size:12px!important;color:var(--blue)!important}.ticket-card>strong{font-size:inherit!important}.ticket-card>.ticket-amount{font-weight:900;font-size:20px;white-space:nowrap;align-self:center}.ticket-card>small{grid-column:1/-1!important;color:var(--muted)!important;font-size:13px;letter-spacing:0}.race-actions button{min-height:46px}.race-actions button.active{background:var(--blue)!important;color:#fff!important}.race.open{border-color:#a5c9f3}.history-intro{font-size:13px;color:var(--muted)}.history-filters{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin:20px 0 12px}.history-filters label{font-size:12px;font-weight:700;color:var(--muted)}.history-filters select{width:100%;padding:11px 8px;background:#fff;border:1px solid var(--line);border-radius:11px;color:var(--ink);font:inherit;min-height:44px}.history-count{font-size:13px;color:var(--muted);margin:8px 0}.history-race{background:#fff;border:1px solid var(--line);border-radius:18px;margin:12px 0;overflow:hidden}.history-race summary{cursor:pointer;padding:16px;list-style:none}.history-race summary::-webkit-details-marker{display:none}.history-race summary:after{content:'買い目を見る ＋';display:block;color:var(--blue);font-size:12px;margin-top:9px}.history-race[open] summary:after{content:'買い目を閉じる −'}.race-title{display:flex;align-items:center;justify-content:space-between;gap:10px}.race-title b{font-size:18px}.race-date{color:var(--muted);font-size:12px}.result-badge{padding:5px 10px;border-radius:99px;font-size:12px;font-weight:800;white-space:nowrap;background:#eef3f9;color:var(--muted)}.result-badge.hit{color:#11774a;background:#e8f7ee}.result-badge.miss{color:#b63744;background:#fff0f1}.race-money{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:13px}.race-money small{display:block;color:var(--muted);font-size:11px}.race-money strong{font-size:16px;white-space:nowrap}.history-tickets{border-top:1px solid var(--line);padding:4px 14px 10px}.history-ticket{padding:13px 0;border-bottom:1px solid #edf2f7}.history-ticket:last-child{border-bottom:0}.history-ticket-top{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}.history-ticket-top small{color:var(--muted)}.history-ticket .car-box{width:35px;height:39px;font-size:23px;border-radius:9px}.ticket-money{font-size:12px;color:var(--muted);margin-top:9px;display:flex;gap:14px;flex-wrap:wrap}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.stats div{padding:13px;background:#fff;border:1px solid var(--line);border-radius:14px}.stats small{display:block;color:var(--muted);font-size:11px}.stats b{font-size:20px}#venueJump button small{display:block;font-size:11px;color:#647d98;margin-top:3px}#venueJump button.active{background:#e5f1ff;border-color:#6ea9ee}.saved-picks-note{background:#edf5ff;border:1px solid #ccdef6;border-radius:12px;padding:12px;font-size:12px;color:#526c87}.continuous-results{background:#fff;border:1px solid #dfe8f2;border-radius:18px;padding:16px;margin:16px 0}.continuous-results h2{font-size:18px;margin:0 0 10px}.continuous-results select{padding:8px;border:1px solid #cddcea;border-radius:9px;background:#fff;color:#18324f}.result-table{overflow-x:auto}.result-table table{width:100%;border-collapse:collapse;font-size:13px}.result-table td,.result-table th{padding:10px 5px;text-align:center;border-bottom:1px solid #edf2f7;white-space:nowrap}.continuous-results small,.result-asof,.hole-picks p{font-size:12px;color:#647d98}.hole-picks{padding-top:12px}.hole-picks h4{color:#526c87}[hidden]{display:none!important}.history-more{display:block;margin:18px auto;border:1px solid #a8c7ed;background:#edf5ff;border-radius:12px;color:var(--blue);padding:12px 26px;font-weight:800;min-height:44px}.empty{padding:24px;text-align:center;background:#fff;border-radius:16px}button,a,summary,select{touch-action:manipulation}a:focus-visible,button:focus-visible,summary:focus-visible,select:focus-visible{outline:3px solid #247be6;outline-offset:3px}@media(max-width:420px){.car-box{width:38px;height:43px;font-size:25px}.ticket-card{padding:13px!important}.ticket-card>.ticket-amount{font-size:18px}.history-filters{gap:6px}.race-money strong{font-size:14px}.hero h1{font-size:23px}.stats b{font-size:18px}.mobile-nav{bottom:max(8px,env(safe-area-inset-bottom))}} 
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
    links=[("today","index.html","今日","RACE"),("picks","index.html#picks","買い目","PICKS"),("company","company/operations.html","会社","COMPANY"),("history","history.html","履歴","HISTORY")]
    return '<nav class="mobile-nav" aria-label="メインメニュー">'+''.join(f'<a class="{"active" if key==active else ""}" href="{url}">{label}<small>{en}</small></a>' for key,url,label,en in links)+'</nav>'

def saved_race_tickets(race_id, now_epoch=None):
    """Restore only current-strategy pre-close snapshots for finished races."""
    import pandas as pd
    import time
    from betting_logic import STRATEGY_VERSION
    now_epoch=time.time() if now_epoch is None else now_epoch
    rid=str(race_id)
    if not re.fullmatch(r"\d{12}",rid):return []
    day=rid[4:]
    snapshots=[]
    for path in OUTPUT_DIR.glob(f"shadow_bets_{day}_*.csv"):
        try:
            frame=pd.read_csv(path,dtype={"race_id":str,"buy":str})
            if "strategy_version" not in frame.columns:
                continue
            frame=frame[
                frame["race_id"].eq(rid)
                & frame["strategy_version"].eq(STRATEGY_VERSION)
            ].copy()
            if frame.empty:continue
            close=pd.to_numeric(frame["close_at"],errors="coerce")
            created=pd.to_datetime(frame["prediction_created_at_jst"],errors="coerce",utc=True)
            frame=frame[close.lt(now_epoch) & created.notna() & created.map(lambda t:t.timestamp() if pd.notna(t) else float("inf")).lt(close)]
            if len(frame):snapshots.append((created.max(),frame))
        except (OSError,ValueError,KeyError,pd.errors.EmptyDataError):continue
    if not snapshots:return []
    _,frame=max(snapshots,key=lambda item:item[0])
    return frame.drop_duplicates(["bet_type","buy"],keep="last").to_dict("records")

def hole_predictions(race_id, path=None):
    import pandas as pd
    path=Path(path) if path is not None else OUTPUT_DIR/"latest_bet_candidates.csv"
    try:frame=pd.read_csv(path,dtype={"race_id":str,"buy":str})
    except (OSError,pd.errors.EmptyDataError):return []
    frame=frame[frame["race_id"].eq(str(race_id)) & frame["bet_type"].eq("trifecta")].copy()
    if frame.empty:return []
    frame["prob"]=pd.to_numeric(frame["prob"],errors="coerce")
    frame=frame[frame["prob"].notna() & frame["prob"].gt(0) & frame["buy"].str.fullmatch(r"[1-9]-[1-9]-[1-9]")].copy()
    truth=lambda col:frame[col].fillna(False).astype(str).str.lower().isin(["true","1"])
    frame["odds_used"]=pd.to_numeric(frame.get("odds_used"),errors="coerce")
    # 穴予想 is a strict longshot lane: known odds of 100x or higher only.
    holes=frame[truth("hole_formation") & frame["odds_used"].ge(100)].copy()
    if holes.empty:return []
    if "ev" in holes.columns:
        holes["ev_rank"]=pd.to_numeric(holes["ev"],errors="coerce")
    else:
        holes["ev_rank"]=float("nan")
    holes["ev_rank"]=holes["ev_rank"].fillna(holes["prob"]*holes["odds_used"])
    holes=holes.sort_values(["ev_rank","prob","buy"],ascending=[False,False,True],na_position="last").head(12)
    return [{**row.to_dict(),"prediction_group":"穴予想"} for _,row in holes.iterrows()]


def continuous_section(relative=""):
    return '<section class="continuous-results" data-report="'+relative+'company/ticket_return_department.json"><h2>買い目の継続成績</h2><label>集計期間 <select class="result-window"><option value="today">今日1日</option><option value="all">全期間・現行戦略</option><option value="last50">直近50レース</option><option value="last100">直近100レース</option></select></label><p class="result-asof"></p><div class="result-table"><p>成績を取得中です。</p></div><small>締切前に保存した購入条件を満たす候補の検証です。実購入ではありません。回収率は各点100円の試算。穴予想の未購入候補・見送り・結果未確定は的中率と回収率の分母に含めません。</small></section>'

CONTINUOUS_JS="""
document.querySelectorAll('.continuous-results').forEach(async box=>{try{const response=await fetch(box.dataset.report+'?t='+Date.now(),{cache:'no-store'});if(!response.ok)throw Error('取得失敗');const report=await response.json();function show(){const period=box.querySelector('.result-window').value,window=report.continuous&&report.continuous[period];const pct=value=>value===null||value===undefined?'未集計':(Number(value)*100).toFixed(1)+'%';const currentDate=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Tokyo'}).format(new Date()),isTodayStale=period==='today'&&report.today_date!==currentDate;const rows=isTodayStale?{}:window||(period==='all'&&report.shadow?{total:report.shadow.total,main:report.shadow.main,hole:report.shadow.hole}:{});box.querySelector('.result-asof').textContent=(period==='today'?'対象日 '+currentDate+(isTodayStale?' · 今日の集計待ち':'')+' · ':'')+'更新 '+report.updated_at_jst+' · 現行戦略';const table=document.createElement('table');table.innerHTML='<thead><tr><th>対象</th><th>的中率</th><th>回収率</th><th>的中／確定</th></tr></thead>';const body=document.createElement('tbody');[['total','本線＋穴'],['main','本線'],['hole','穴']].forEach(([key,label])=>{const data=rows[key]||{};const tr=document.createElement('tr');[label,pct(data.hit_rate),pct(data.flat_100yen&&data.flat_100yen.return_rate),(data.hits||0)+'／'+(data.bet_races||0)+'R'].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td)});body.append(tr)});table.append(body);box.querySelector('.result-table').replaceChildren(table)}box.querySelector('.result-window').addEventListener('change',show);show()}catch(error){box.querySelector('.result-table').textContent='成績を取得できません。再読み込みしてください。'}});
"""

def enhance_today(document):
    assets()
    soup=BeautifulSoup(document,"html.parser")
    if not soup.select_one('link[href="site-ui.css"]'):
        soup.head.append(soup.new_tag('link',rel='stylesheet',href='site-ui.css'))
    old=soup.select_one('main > nav')
    if old:old['class']=['top-nav']
    for race in soup.select('article.race'):
        panel=race.select_one('.picks-panel')
        if panel and not panel.select_one('.bet'):
            rid=race.get('id','').removeprefix('race-')
            stored=saved_race_tickets(rid)
            if stored:
                for item in list(panel.children):
                    if getattr(item,'name',None)!='h3':item.extract()
                stamp=str(stored[0].get('prediction_created_at_jst',''))
                notice='<p class="saved-picks-note">締切前に保存した検証用予想 · '+html.escape(stamp)+'<br>保存時のオッズ・配分額です。現在の購入候補ではありません。</p>'
                panel.append(BeautifulSoup(notice,'html.parser'))
                for row in stored:
                    odds=numeric(row.get('odds_used'));ev=numeric(row.get('ev'))
                    detail=(f'保存時オッズ {odds:.1f}倍' if odds is not None else '保存時オッズ未取得')+(f' · 推定EV {ev:.2f}' if ev is not None else '')
                    card='<div class="bet"><b>'+html.escape(str(row.get('ticket_group','')))+' '+html.escape(str(row.get('bet_label','')))+'</b><strong>'+html.escape(str(row.get('buy','')))+'</strong><span>'+money(row.get('stake_yen'))+'</span><small>'+detail+'</small>'+odds_provenance(row)+'</div>'
                    panel.append(BeautifulSoup(card,'html.parser'))
            else:
                waiting=panel.select_one('.waiting')
                if waiting:
                    waiting.string='現在は買い目候補なし。対象時間外、オッズ未取得・不一致、または期待値条件を満たしていません。条件がそろったレースから表示します。'
    import time
    for race in soup.select('article.race'):
        if (numeric(race.get('data-start')) or 0)<=time.time():continue
        panel=race.select_one('.picks-panel')
        if panel is None or panel.select_one('.hole-picks'):continue
        holes=hole_predictions(race.get('id','').removeprefix('race-'))
        if not holes:continue
        html_hole='<section class="hole-picks"><h4>穴予想</h4><p>100倍以上だけを対象。AIの期待値と確率で最大12点まで表示します。購入条件を満たした買い目は上段の買い目にも反映します。</p>'
        for row in holes:
            odds=numeric(row.get('odds_used'))
            market=f'取得時オッズ {odds:.1f}倍'
            html_hole+='<div class="bet hole"><b>'+html.escape(row['prediction_group'])+'</b><strong>'+html.escape(row['buy'])+'</strong><small>'+market+'</small>'+odds_provenance(row)+'</div>'
        html_hole+='</section>'
        panel.append(BeautifulSoup(html_hole,'html.parser'))
    if not soup.select_one('.continuous-results'):
        hero=soup.select_one('.hero')
        section=BeautifulSoup(continuous_section(),'html.parser')
        if hero:hero.insert_after(section)
        else:soup.main.append(section)
    for card in soup.select('.bet'):
        if not card.select_one('.odds-provenance'):
            card.append(BeautifulSoup(odds_provenance({}),'html.parser'))
        card['class']=list(dict.fromkeys(card.get('class',[])+['ticket-card']))
        label=card.find('b')
        if label and '穴' in label.get_text() and 'hole' not in card['class']:card['class'].append('hole')
        ticket=card.find('strong')
        if ticket and not ticket.select_one('.car-box'):
            value=ticket.get_text(strip=True);ticket.clear();ticket.append(BeautifulSoup(numbers(value),'html.parser'))
        amount=card.find('span',recursive=False)
        if amount:amount['class']=['ticket-amount']
    old_navigation=soup.select_one('.mobile-nav')
    if old_navigation:old_navigation.decompose()
    soup.body.append(BeautifulSoup(nav(),'html.parser'))
    for race in soup.select('article.race'):
        previous=race.select_one('.odds-audit')
        if previous:previous.decompose()
        rid=race.get('id','').removeprefix('race-')
        if not re.fullmatch(r'\d{12}',rid):continue
        try:report=json.loads((OUTPUT_DIR/f'market_odds_{rid}.json').read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):continue
        if str(report.get('race_id'))!=rid:continue
        conflict_count=sum(bool(re.fullmatch(r'[1-9]-[1-9]-[1-9]',key)) for key in report.get('conflicts',{}))
        note='<details class="odds-audit"><summary>オッズ取得・不一致の確認記録</summary><p>最新の取得記録 '+html.escape(str(report.get('captured_at_jst','未取得')))+'。各買い目の価格取得時刻とは区別して表示します。</p><p>使用可能 '+str(report.get('usable_count',0))+'組 ／ サイト間の価格不一致 '+str(conflict_count)+'組（価格として使用しません）。</p></details>'
        panel=race.select_one('.picks-panel')
        if panel:panel.append(BeautifulSoup(note,'html.parser'))
    for script in soup.find_all('script'):
        if script.get('id')=='site-navigation' or 'function activatePicks()' in (script.string or ''):script.extract()
    script=soup.new_tag('script',id='site-navigation')
    script.string="""
function raceTime(epoch){return Number(epoch)>0&&Number(epoch)<9e10?new Intl.DateTimeFormat('ja-JP',{timeZone:'Asia/Tokyo',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(Number(epoch)*1000)):'時刻未取得';}
function buildVenueJump(){const box=document.getElementById('venueJump');if(!box)return;box.replaceChildren();const cards=[...document.querySelectorAll('article.race')],now=Date.now()/1000;const dayFormat=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Tokyo'}),today=dayFormat.format(new Date()),raceDay=cards.length?dayFormat.format(new Date(Number(cards[0].dataset.start)*1000)):today;const heading=document.querySelector('.hero h1');if(heading)heading.textContent=raceDay===today?'今日のレース':raceDay+'のレース（今日のデータ待ち）';const groups=new Map();cards.forEach(r=>{const status=r.querySelector('.race-select small');if(status&&!r.querySelector('.race-result.decided'))status.textContent=Number(r.dataset.start)<=now?'発走済み・結果待ち':raceTime(r.dataset.start)+' 発走';const venue=r.dataset.venue;if(!groups.has(venue))groups.set(venue,[]);groups.get(venue).push(r)});const venues=[...groups].map(([name,rows])=>({name,rows:rows.sort((a,b)=>Number(a.dataset.start)-Number(b.dataset.start))})).sort((a,b)=>Number(a.rows[0].dataset.start)-Number(b.rows[0].dataset.start));const upcoming=cards.filter(r=>Number(r.dataset.start)>now).sort((a,b)=>Number(a.dataset.start)-Number(b.dataset.start))[0];venues.forEach(v=>{const first=v.rows[0],last=v.rows[v.rows.length-1],next=v.rows.find(r=>Number(r.dataset.start)>now),button=document.createElement('button');button.dataset.venue=v.name;button.append(document.createTextNode(v.name+' '+raceTime(first.dataset.start)+'〜'+raceTime(last.dataset.start)));const detail=document.createElement('small');detail.textContent=next?'次 '+raceTime(next.dataset.start)+'発走':(raceDay===today?'本日終了':raceDay+' 終了');button.append(detail);button.onclick=()=>{box.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b===button));cards.forEach(r=>{r.classList.toggle('venue-visible',r.dataset.venue===v.name);r.classList.remove('open')});};box.append(button)});const selected=upcoming?box.querySelector('button[data-venue="'+upcoming.dataset.venue+'"]'):box.querySelector('button');if(selected)selected.click();}
function activatePicks(){const cards=[...document.querySelectorAll('.race.venue-visible')],now=Date.now()/1000;const card=cards.find(r=>Number(r.dataset.start)>now)||cards[0];if(!card)return;card.classList.add('open');const button=card.querySelector('.race-actions button');if(button)showPanel(button,'picks');}
document.addEventListener('DOMContentLoaded',function(){if(location.hash==='#picks')activatePicks();document.querySelectorAll('.race-select').forEach(function(btn){btn.addEventListener('click',function(){var card=btn.closest('.race');if(card.classList.contains('open')){var pick=card.querySelector('.race-actions button');if(pick)showPanel(pick,'picks');}})});document.querySelectorAll('.race-actions button').forEach(function(btn){btn.addEventListener('click',function(){btn.parentNode.querySelectorAll('button').forEach(function(b){b.classList.toggle('active',b===btn)})})})});window.addEventListener('hashchange',function(){if(location.hash==='#picks')activatePicks()});
"""
    soup.body.append(script)
    existing_metrics=soup.select_one('script#continuous-results-script')
    if existing_metrics:
        existing_metrics.decompose()
    metrics=soup.new_tag("script",id="continuous-results-script");metrics.string=CONTINUOUS_JS;soup.body.append(metrics)
    previous_age=soup.select_one('#odds-age-script')
    if previous_age:previous_age.decompose()
    age=soup.new_tag('script',id='odds-age-script')
    age.string="function updateOddsAge(){document.querySelectorAll('.odds-provenance[data-captured]').forEach(e=>{const age=(Date.now()-Date.parse(e.dataset.captured))/1000;const label=e.querySelector('.odds-age');if(label)label.textContent=!Number.isFinite(age)||age<0?' ／ 時刻要確認':age>=120?' ／ 2分以上前の取得・更新確認':' ／ 取得から'+Math.floor(age)+'秒'})};updateOddsAge();setInterval(updateOddsAge,30000);"
    soup.body.append(age)
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
