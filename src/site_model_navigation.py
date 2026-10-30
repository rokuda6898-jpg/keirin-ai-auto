"""Keep existing prediction products directly accessible beside the current one."""
import re
from pathlib import Path

MODELS = (
    ('index.html', '現行モデル', '8・10・18点の予想'),
    ('previous.html', '従来NEXUS', '予想印・S/A/B・配分'),
    ('conditional.html', '条件付き3連単', '回収重視8点／従来12点'),
    ('nexus-forward.html', '10点予想・比較', '保存した予想と成績'),
    ('graded.html', '重賞部', '通常・穴・最良式'),
    ('original/index.html', '本線・穴／最終予想', '従来の本線・穴と会社判断'),
    ('company.html', '会社・各部署', '7部署の予想と判断'),
)


def apply_model_navigation(destination):
    destination = Path(destination)
    # Every destination must already exist; no placeholder or retired model links.
    for path, _, _ in MODELS:
        if not (destination / path).is_file():
            raise ValueError(f'Missing model page: {path}')
    paths = list(destination.glob('*.html'))
    paths += list((destination / 'original').rglob('*.html'))
    for path in paths:
        if path.name == 'independent.html':
            continue  # Preserve the pre-existing retirement redirect.
        rel = path.relative_to(destination)
        home = '../' * (len(rel.parts) - 1)
        active = rel.as_posix()
        if active == 'current.html':
            active = 'index.html'
        elif active.startswith('original/company/'):
            active = 'company.html'
        links = ''.join(
            f'<a href="{home}{target}"'
            + (' aria-current="page"' if target == active else '')
            + f'><strong>{label}</strong><span>{detail}</span></a>'
            for target, label, detail in MODELS
        )
        nav = ('<nav class="model-switcher" id="model-switcher" aria-label="予想モデルの切り替え">'
               '<div class="model-switcher-title">モデルを選ぶ</div>'
               f'<div class="model-switcher-links">{links}</div></nav>')
        text = path.read_text(encoding='utf-8')
        if len(rel.parts) == 1 and path.name not in ('index.html', 'current.html'):
            # These old anchors belong to the preserved NEXUS page, not the new home.
            text = re.sub(r'href="index\.html#(forecast|results|model|model-details)"',
                          r'href="previous.html#\1"', text)
        style = f'<link rel="stylesheet" href="{home}model-navigation.css?v=20261030-1">'
        text = re.sub(r'</head>', lambda m: style + m[0], text, count=1, flags=re.I)
        if re.search(r'</header>', text, re.I):
            text = re.sub(r'</header>', lambda m: m[0] + nav, text, count=1, flags=re.I)
        else:
            text = re.sub(r'<body\b[^>]*>', lambda m: m[0] + nav, text, count=1, flags=re.I)
        path.write_text(text, encoding='utf-8')
