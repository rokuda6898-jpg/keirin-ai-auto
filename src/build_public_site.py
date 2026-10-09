"""Publish the original company UI unchanged alongside the lightweight home."""
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT, destination=None):
    destination = destination or root / 'public-build'
    if destination.exists():
        raise ValueError('Use a fresh build directory')
    shutil.copytree(root / 'site', destination)
    # Only generated public output, never source credentials or model files.
    shutil.copytree(root / 'outputs', destination / 'original')
    stamp = datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')
    for path in (destination / 'original').rglob('*.html'):
        depth = len(path.relative_to(destination).parts) - 1
        home = '../' * depth
        nav = ('<div class="nexus-site-header">'
               f'<a href="{home}index.html">NEXUS トップ</a><nav aria-label="サイト案内">'
               f'<a href="{home}original/index.html">本線・穴</a>'
               f'<a href="{home}independent.html">独立予想の買い目</a>'
               f'<a href="{home}company.html">会社・全部署</a>'
               f'<a href="{home}archive.html">過去レース保管庫</a></nav></div>'
               f'<small class="nexus-published">画面保存 {stamp} ／ 各予想の保存時刻は本文に表示</small>')
        text = path.read_text(encoding='utf-8-sig')
        import re
        assets = f'<link rel="stylesheet" href="{home}site-wide.css">'
        script = f'<script src="{home}site-wide.js" data-root="{home}"></script>'
        if re.search(r'</head>', text, re.I):
            text = re.sub(r'</head>', lambda m: assets+m[0], text, count=1, flags=re.I)
        else:
            text = assets + text
        if not re.search(r'<body\b', text, re.I):
            # Some original reports omit body tags; give them the same shell.
            if re.search(r'</head>', text, re.I):
                text = re.sub(r'</head>', lambda m: m[0]+'<body>', text, count=1, flags=re.I)
            else:
                text = '<body>' + text
        def body_tag(match):
            tag = match[0]
            if re.search(r'class=[\"\']', tag):
                tag = re.sub(r'(class=[\"\'])', r'\1nexus-report ', tag, count=1)
            else:
                tag = tag[:-1] + ' class="nexus-report">'
            return tag + nav
        text = re.sub(r'<body\b[^>]*>', body_tag, text, count=1, flags=re.I)
        if re.search(r'</body>', text, re.I):
            text = re.sub(r'</body>', lambda m: script+m[0], text, count=1, flags=re.I)
        else:
            text += script + '</body>'
        path.write_text(text, encoding='utf-8')
    (destination / 'published.json').write_text(json.dumps({'published_at': stamp}), encoding='utf-8')
    print(f'Published complete company pages: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    build(destination=args.destination)
