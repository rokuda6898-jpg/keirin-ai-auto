"""Public trifecta sources. Ticket order and race identity must be explicit."""
import json
import math
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup
from common import OUTPUT_DIR

VERSION = "multi_source_trifecta_v1"

def valid_price(value):
    try:
        price = float(value)
        return price if math.isfinite(price) and 1 < price < 9999.9 else None
    except (TypeError, ValueError):
        return None

def normalize(odds, cars):
    out = {}
    for ticket, value in odds.items():
        if not re.fullmatch(r"[1-9]-[1-9]-[1-9]", str(ticket)):
            continue
        numbers = list(map(int, ticket.split("-")))
        price = valid_price(value)
        if len(set(numbers)) == 3 and set(numbers).issubset(cars) and price is not None:
            out[ticket] = price
    return out

def parse_oddspark(html, day, race_no, venue):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text() if soup.title else ""
    y,m,d = day[:4], int(day[4:6]), int(day[6:])
    if f"{y}年{m}月{d}日" not in title or venue+"競輪" not in title or not re.search(rf"\b{race_no}R\b", title):
        raise ValueError("oddspark race identity mismatch")
    if f"{race_no}R 3連単オッズ" not in soup.get_text(" ", strip=True):
        raise ValueError("oddspark bet type mismatch")
    out = {}
    for row in soup.select("table.tb50 tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) != 3:
            continue
        ordered = cells[1].select_one("ul.trio")
        if ordered is None or ordered.get_text(strip=True) != "→→":
            continue
        numbers = [c[1:] for li in ordered.select("li") for c in li.get("class", []) if re.fullmatch(r"n[1-9]",c)]
        if len(numbers) == 3:
            out["-".join(numbers)] = cells[2].get_text(strip=True)
    return out

def parse_netkeirin(payload, day, code, race_no):
    key = "nkrace_odds::"+day+code+f"{race_no:02d}"
    if payload.get("status") != "OK" or key not in payload.get("data", {}):
        raise ValueError("netkeirin race identity mismatch")
    data = payload["data"][key]
    out = {}
    for row in data.get("list_9", []):
        if len(row) >= 2 and re.fullmatch(r"0[1-9]0[1-9]0[1-9]",str(row[0])):
            out["-".join(str(int(row[0][i:i+2])) for i in (0,2,4))] = row[1]
    return out, data.get("official_dt")

def reconcile(sources):
    """Use the lowest agreeing quote; disagreements never create an EV price."""
    quotes = {}
    for source in sources:
        for ticket, price in source.get("odds", {}).items():
            quotes.setdefault(ticket, []).append((source["provider"],price))
    chosen, conflicts, evidence = {}, {}, {}
    for ticket, rows in quotes.items():
        values = [price for _,price in rows]
        # No optimistic averaging or maximum-price selection across update delays.
        if max(values)-min(values) > max(0.1,min(values)*0.05):
            conflicts[ticket] = dict(rows)
        else:
            chosen[ticket] = min(values)
            evidence[ticket] = [name for name,_ in rows]
    return chosen, conflicts, evidence

def verify_market(entry_rows, odds_rows, fetch_html=None, report_dir=None):
    """Capture only pre-close data; historical/backtest requests stay offline."""
    base = entry_rows[0]
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    if base["date"] != now.strftime("%Y-%m-%d") or float(base.get("close_at") or 0) <= now.timestamp():
        return odds_rows
    rid = str(base["race_id"])
    if not re.fullmatch(r"\d{12}", rid):
        raise ValueError("invalid market race id")
    race_no, code, day = int(rid[:2]),rid[2:4],rid[4:]
    if day != base["date"].replace("-", "") or race_no != int(base["race_no"]):
        raise ValueError("market race identity mismatch")
    cars = {int(row["car_no"]) for row in entry_rows}
    parsed_url = urlsplit(base["source_url"])
    match = re.fullmatch(r"/keirin/([^/]+)/racecard/(\d{10})/(\d+)/(\d+)", parsed_url.path)
    if match is None or match[2][-2:] != code or int(match[4]) != race_no:
        raise ValueError("market meeting identity mismatch")
    slug,cup,index,_ = match.groups()
    kd_id = code+cup[:8]+f"{int(index):02d}00{race_no:02d}"
    urls = {
        "kdreams":f"https://keirin.kdreams.jp/{slug}/racedetail/{kd_id}/?pageType=odds&kakeshikiType=3rentan",
        "oddspark":f"https://www.oddspark.com/keirin/Odds.do?betType=9&joCode={code}&kaisaiBi={day}&raceNo={race_no}&viewType=1",
        "netkeirin":"https://keirin.netkeiba.com/api/race/",
    }
    sources = [{"provider":"winticket","source_url":base["source_url"],"status":"verified","odds":normalize({r["buy"]:r["odds_used"] for r in odds_rows if r["bet_type"] == "trifecta"},cars)}]
    def fetch(pair):
        name,url = pair
        item = {"provider":name,"source_url":url,"odds":{}}
        try:
            if name == "netkeirin":
                response = requests.post(url, data={"class":"AplRaceOdds","method":"get","compress":"0","race_id":day+code+f"{race_no:02d}","input":"UTF-8","output":"json"},timeout=(4,8))
                response.raise_for_status()
                odds, official = parse_netkeirin(response.json(),day,code,race_no)
                item["source_updated_at"] = official
            else:
                response = requests.get(url,timeout=(4,8)) if fetch_html is None else None
                if response is not None:
                    response.raise_for_status();response.encoding="utf-8";html=response.text
                else:html=fetch_html(url)
                if name == "oddspark":odds=parse_oddspark(html,day,race_no,base["venue"])
                else:
                    soup=BeautifulSoup(html,"html.parser");title=soup.title.get_text() if soup.title else ""
                    if f"{day[:4]}年{day[4:6]}月{day[6:]}日" not in title or not re.search(rf"\b{race_no}R\b",title):
                        raise ValueError("kdreams race identity mismatch")
                    odds=parse_kdreams(html)
            item["odds"] = normalize(odds,cars)
            item["status"] = "verified" if item["odds"] else "no_odds"
        except Exception as exc:
            item["status"]="unavailable";item["error"]=type(exc).__name__+": "+str(exc)
        item["captured_at_jst"] = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
        return item
    with ThreadPoolExecutor(max_workers=3) as executor:
        sources.extend(executor.map(fetch,urls.items()))
    for item in sources:
        item.setdefault("captured_at_jst",now.isoformat(timespec="seconds"))
        item["count"] = len(item["odds"])
    verified, conflicts, evidence = reconcile(sources)
    if datetime.now(ZoneInfo("Asia/Tokyo")).timestamp() >= float(base["close_at"]):
        verified={};conflicts["capture"]={"reason":"closed_during_fetch"}
    report={"race_id":rid,"captured_at_jst":now.isoformat(timespec="seconds"),"parser_version":VERSION,"sources":sources,"conflicts":conflicts,"ticket_sources":evidence,"usable_count":len(verified)}
    destination = OUTPUT_DIR if report_dir is None else Path(report_dir)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/f"market_odds_{rid}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    # Keep non-trifecta pools unchanged. Invalid/conflicting trifecta prices are
    # represented explicitly by NaN, so cached prices cannot resurrect them.
    result=[r for r in odds_rows if r["bet_type"] != "trifecta"]
    for ticket in sorted(set(verified)|{r["buy"] for r in odds_rows if r["bet_type"]=="trifecta"}|{t for t in conflicts if re.fullmatch(r"[1-9]-[1-9]-[1-9]",t)}):
        price=verified.get(ticket,float("nan"))
        providers=evidence.get(ticket,[])
        captured=min((s["captured_at_jst"] for s in sources if s["provider"] in providers),default=None)
        result.append({"date":base["date"],"venue":base["venue"],"race_no":race_no,"race_id":rid,"bet_type":"trifecta","buy":ticket,"odds":price,"min_odds":0,"max_odds":0,"odds_used":price,"popularity_order":None,"source_url":" | ".join(urls[n] if n in urls else base["source_url"] for n in providers),
                       "odds_sources":" | ".join(providers),"odds_captured_at_jst":captured,
                       "odds_verification_status":"verified" if ticket in verified else "excluded"})
    return result

def parse_kdreams(html):
    """KDreams bt5: block=first, column=second, row=third. Never guess."""
    soup = BeautifulSoup(html, "html.parser")
    odds = {}
    for table in soup.select("table.odds_table.bt5"):
        title = table.select_one("th[colspan] span.number")
        if title is None or not re.fullmatch(r"[1-9]", title.get_text(strip=True)):
            continue
        first = int(title.get_text(strip=True))
        columns = None
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                continue
            headers = [c for c in cells if c.name == "th" and re.fullmatch(r"[1-9]", c.get_text(strip=True))]
            values = [c for c in cells if c.name == "td"]
            if headers and not values and not any(c.has_attr("colspan") for c in cells):
                columns = [int(c.get_text(strip=True)) for c in headers]
                if len(set(columns)) != len(columns) or first in columns:
                    columns = None
                continue
            if columns is None or not headers or len(values) != len(columns):
                continue
            third = int(headers[0].get_text(strip=True))
            if third not in columns:
                continue
            for second, cell in zip(columns, values):
                if len({first, second, third}) != 3:
                    continue
                value = cell.get_text(strip=True).replace(",", "")
                if not re.fullmatch(r"\d+(?:\.\d+)?", value):
                    continue
                odd = float(value)
                # Display ceiling is censored, not a usable exact market price.
                if 1 < odd < 9999.9:
                    odds[f"{first}-{second}-{third}"] = odd
    return odds
