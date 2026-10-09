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
        nav = ('<nav style="padding:12px;background:#102822;color:white">'
               f'<a style="color:white" href="{home}index.html">NEXUS トップ</a>　'
               f'<a style="color:white" href="{home}archive.html">過去レース保管庫</a>'
               f'<small>　画面保存 {stamp}（各予想の保存時刻は本文に表示）</small></nav>')
        text = path.read_text(encoding='utf-8-sig')
        import re
        text = re.sub(r'(<body\b[^>]*>)', lambda m: m[0] + nav, text, count=1, flags=re.I)
        path.write_text(text, encoding='utf-8')
    (destination / 'published.json').write_text(json.dumps({'published_at': stamp}), encoding='utf-8')
    print(f'Published complete company pages: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    build(destination=args.destination)
