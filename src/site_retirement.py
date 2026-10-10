"""Remove the retired fusion-independent product from published navigation.

Historical JSON/JSONL evidence is retained. Conditional trifecta is not retired.
"""
import re
from pathlib import Path


def remove_legacy_navigation(text):
    text = re.sub(
        r'<a\b(?=[^>]*\bhref=[\"\'][^\"\']*(?:independent\.html|fusion_shadow_live_report\.html)[^\"\']*[\"\'])[^>]*>.*?</a\s*>',
        '', text, flags=re.I|re.S)
    text = re.sub(r'<button\b(?=[^>]*\bid=[\"\']shadow-mode[\"\'])[^>]*>.*?</button\s*>',
                  '', text, flags=re.I|re.S)
    return text


def apply_retirement(destination):
    destination = Path(destination)
    for path in destination.rglob('*.html'):
        text = remove_legacy_navigation(path.read_text(encoding='utf-8-sig'))
        if path == destination/'conditional.html':
            text = text.replace('独立予想', '条件付き3連単')
        if path == destination/'company.html':
            text = text.replace('会議の最終予想・独立予想', '会社の最終予想')
            text = text.replace('それぞれの予想印と買い目を切り替え。', '全部署の判断を集約した予想印と買い目。')
        if path == destination/'index.html':
            text = text.replace('38.77％は過去の検証値で、将来の的中を保証しません。', '')
        text = re.sub(r'app\.js\?v=[^\"\']+', 'app.js?v=20261011-retirement', text)
        text = re.sub(r'site-wide\.js(?:\?[^\"\']*)?(?=[\"\'])',
                      'site-wide.js?v=20261011-retirement', text)
        path.write_text(text, encoding='utf-8')
    app = destination/'app.js'
    if app.exists():
        text = app.read_text(encoding='utf-8-sig')
        text = text.replace("shadow:'結果・成績｜独立予想'", "shadow:'結果・成績｜条件付き3連単'")
        text = text.replace("state.mode==='shadow'?'独立予想'", "state.mode==='shadow'?'条件付き3連単'")
        text = text.replace('independent.html#forecast', 'conditional.html#forecast')
        app.write_text(text, encoding='utf-8')
    (destination/'independent.html').write_text(
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="refresh" content="0;url=conditional.html">'
        '<title>従来の独立予想は終了しました</title></head><body>'
        '<p>従来の独立予想は終了しました。</p>'
        '<a href="conditional.html">条件付き3連単へ</a>'
        '<a href="company.html">会社・全部署へ</a></body></html>', encoding='utf-8')
