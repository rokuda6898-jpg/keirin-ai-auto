"""Bounded read-only diagnostic for public Gamboo venue decision-method tables."""
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
import requests

ROOT=Path("research/gamboo_probe")
ROOT.mkdir(parents=True,exist_ok=True)
urls={
    "LP":"https://gamboo.jp/pages/?tid=tohyama_bank_LP",
    "75":"https://gamboo.jp/pages/?tid=tohyama_bank_75",
    "53":"https://gamboo.jp/pages/?tid=tohyama_bank_53",
    "24":"https://gamboo.jp/pages/?tid=tohyama_bank_24"
}
result={}
for code,url in urls.items():
    res={"url":url}
    try:
        response=requests.get(url,timeout=25,headers={"User-Agent":"Mozilla/5.0 (compatible; research-check/1.0)","Accept-Language":"ja-JP,ja;q=0.9"})
        response.encoding=response.apparent_encoding
        html=response.text
        soup=BeautifulSoup(html,"html.parser")
        plain=soup.get_text(" ",strip=True)
        res.update({"http_status":response.status_code,"bytes":len(response.content),
                    "encoding":response.encoding,"title":soup.title.get_text(" ",strip=True) if soup.title else None,
                    "tables":len(soup.find_all("table")),
                    "text_occurrences":plain.count("決まり手"),
                    "text_excerpt":plain[max(0,plain.find("決まり手データ")-40):plain.find("決まり手データ")+2300]})
        for keyword in ("S級戦・9車立て","A12班戦・7車立て","1着決まり手比率","チャレンジ・7車立て","決まり手データ"):
            found=soup.find(string=lambda t:t and keyword in str(t))
            if found:
                parent=found.parent
                res["html_"+keyword]=str(parent.parent)[:5000]
        res["scripts_src"]=[s.get("src") for s in soup.select("script[src]")[-20:]]
        (ROOT/f"{code}.html").write_text(html,encoding="utf-8")
    except Exception as exc:
        res["error"]=str(exc)
    result[code]=res
    print("PROBE",code, json.dumps({k:v for k,v in res.items() if k.startswith("http_") or k in ("bytes","tables","title","text_occurrences","error","text_excerpt")},ensure_ascii=False)[:2600],flush=True)
(ROOT/"probe.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
