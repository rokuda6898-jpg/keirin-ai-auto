"""Inspect actual WINTICKET results payload shapes; bounded public probe."""
import json
import re
from pathlib import Path

from fetch_today_entries import http_get, extract_preloaded_state, tanstack_queries

BASE="https://www.winticket.jp/keirin/gifu/racecard/2026101043"
URLS=[BASE+"/results",BASE+"/1/1",BASE+"/2/results",BASE+"/results/2",BASE+"/results?index=2"]
DEST=Path("research/method_probe")
DEST.mkdir(parents=True,exist_ok=True)
report={}
for url in URLS:
    slot=url.split("/racecard/2026101043")[-1].replace("/","_").replace("?","_") or "root"
    out={"url":url}
    try:
        html=http_get(url,attempts=2)
        out["html_bytes"]=len(html)
        out["result_links"]=sorted(set(re.findall(r'href="([^"]*2026101043[^"]*)"',html)))[:80]
        pos=html.find("着順・決まり手")
        out["text_decisive_excerpt"]=re.sub("<[^>]+>"," ",html[max(0,pos-100):pos+400]) if pos>=0 else "not found"
        state=extract_preloaded_state(html)
        queries=[]
        for q in tanstack_queries(state):
            key=q.get("queryKey")
            name=json.dumps(key,ensure_ascii=False)
            d=q.get("state",{}).get("data")
            rec={"key":key,"data_type":type(d).__name__,
                 "root_fields":list(d)[:35] if isinstance(d,dict) else []}
            if isinstance(d,dict):
                for field in ("race","results","races","schedules","entries","cup","raceResults","raceResult","raceResultOverview"):
                    v=d.get(field)
                    if isinstance(v,list) and v:
                        rec["sample_"+field]=v[0] if field not in ("entries",) else v[:2]
                    elif isinstance(v,dict):
                        rec["sample_"+field]={k:(z[:2] if isinstance(z,list) else z) for k,z in list(v.items())[:7]}
            if any(m in name.upper() for m in ("RESULT","RACE","CUP")):
                queries.append(rec)
        out["queries"]=queries
    except Exception as exc:
        out["error"]=repr(exc)
    report[slot]=out
(DEST/"schema.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
for k,v in report.items():
    print("PAGE",k,"bytes",v.get("html_bytes"),"error",v.get("error"))
    for q in v.get("queries",[]):
        print("QUERY",json.dumps(q,ensure_ascii=False)[:2600])
