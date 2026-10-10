"""Render the frozen retrospective evidence, separate from prospective scores."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    result = json.loads((root/'outputs/conditional_research/results.json').read_text(encoding='utf-8'))
    own, base = result['methods']['selected'], result['methods']['previous_best']
    parts = ['<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>条件付き3連単｜過去1年の検証</title><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="site-wide.css"></head><body class="home-page"><header><a class="brand" href="./">N<span>KEIRIN<br><strong>NEXUS</strong></span></a><nav class="header-nav"><a href="conditional.html">今日の条件付き3連単 →</a></nav></header><main>',
        '<section class="home-heading"><div><p class="eyebrow">HISTORICAL STUDY</p><h1>過去1年の検証</h1><p class="intro">条件付き3連単・各レース12点</p></div></section>',
        '<section class="stats">']
    for label, value, note in [('的中率', f"{own['hit_rate']:.2%}", '過去データでの再検証'),
                               ('的中', f"{own['hits']:,} / {own['races']:,}R", '同じレースを比較'),
                               ('改善', f"+{own['hits']-base['hits']:,}R", f"前の最良式 {base['hit_rate']:.2%}"),
                               ('回収率', '未検証', '的中率と別に評価')]:
        parts.append(f'<div class="stat"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>')
    parts.extend(['</section><section class="performance-panel"><h2>何を変えたか</h2>',
        '<p>3連単の評価 ＝ 1着の確率 × その1着のもとでの2着確率 × その1・2着のもとでの3着確率。</p>',
        '<p>選手の成績・脚質・同地区などの関係を学び、全着順を評価して上位12点を選びます。同じ学習方法で各順位を独立に評価した場合は39.15%。条件付きの組み合わせ評価で45.86%になりました。点数は増やしていません。</p>',
        '<p>前の最良式が外した1,544レースを拾い、逆に644レースを落としました。</p></section>',
        '<section class="performance-panel"><h2>検証条件</h2><ul>',
        '<li>予想期間：2025年10月9日〜2026年10月8日。</li>',
        '<li>最終学習：2022年10月9日〜2025年10月8日の30,232レース。古い履歴から作る選手別の事前集計は学習締切以前に限定。</li>',
        '<li>直近の成績を重く扱い、1年前の重みを半分にする方式。設定の選択は2025年4月9日〜10月8日の別期間で完了。</li>',
        '<li>予想13,742レース。1〜3着を一意に確定できない130レースは、全方式共通で成績集計から除外。残り13,612レースをすべて比較。</li>',
        f"<li>除外130レースもすべて不的中と数えた場合でも {own['hits']/result['predicted_races']:.2%}。</li>",
        '<li>各方式12点・各点100円。穴100倍以上の条件や6,000円の配分を評価した成績ではありません。</li>',
        '</ul><p>入力は過去ページからの復元で、実際に締切前に保存された予想ではありません。比較対象の直近1年は以前の研究でも参照済みです。この方式の設定選択には使っていませんが、完全に未参照の将来検証とは区別します。</p>',
        '<p>45.86%が今後も続くことや、回収率が上がることは未確認です。今日以降の買い目と成績は別台帳で検証します。</p></section>'])
    for prefix, title in [('month:', '月別の的中率'), ('field:', '出走人数別'), ('class:', 'クラス別')]:
        parts.append(f'<section class="performance-panel"><h2>{title}</h2><table><thead><tr><th>対象</th><th>レース数</th><th>条件付き3連単</th><th>前の最良式</th></tr></thead><tbody>')
        for key, group in sorted(result['groups'].items()):
            if not key.startswith(prefix):
                continue
            a, b = group['selected'], group['previous_best']
            parts.append(f"<tr><td>{html.escape(key[len(prefix):])}</td><td>{a['races']:,}</td><td>{a['hits']/a['races']:.2%} ({a['hits']:,})</td><td>{b['hits']/b['races']:.2%} ({b['hits']:,})</td></tr>")
        parts.append('</tbody></table></section>')
    parts.append('<footer><a href="conditional.html">今日の条件付き3連単へ →</a><p><a href="original/conditional_research/results.json">集計データ</a> ／ <a href="original/conditional_research/frozen_manifest.json">学習・選択の記録</a></p></footer></main><script src="site-wide.js" data-root="./"></script></body></html>')
    (root/'site/conditional-study.html').write_text(''.join(parts), encoding='utf-8')


if __name__ == '__main__':
    build()
