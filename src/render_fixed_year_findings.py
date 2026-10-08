"""Present a completed fixed-year artifact without changing its model or scores."""
import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


def render(source,destination):
    read=lambda name:json.loads((source/name).read_text(encoding='utf-8'))
    result=read('results.json');inventory=read('inventory.json');frozen=read('frozen_manifest.json')
    prediction=read('prediction_manifest.json');execution=read('execution_provenance.json')
    if result['model_sha256']!=prediction['model_sha256'] or result['model_sha256']!=frozen['model_sha256']:
        raise ValueError('model evidence does not match')
    if result['predictions_sha256']!=prediction['predictions_sha256']:raise ValueError('prediction evidence does not match')
    if result['model_update_during_holdout'] or result['ceo_integration']:raise ValueError('unexpected research status')
    total=result['evaluated_races'];overall=result['summary']['all']['methods'];risk=overall['risk'];combined=overall['fusion']
    hits=lambda name,k:round(overall[name][f'top{k}_hit']*total)
    gained=hits('fusion',12)-hits('risk',12)
    pct=lambda v:f'{v*100:.2f}％'
    point=lambda v:f'{v*100:+.2f}ポイント'
    names={'data':'データ部','pace':'展開部','line':'ライン部'}
    pairs=''.join(f'<tr><th>{label}</th><td>{pct(overall[d+"_legacy"]["top12_hit"])}</td><td>{pct(overall[d]["top12_hit"])}</td><td>{point(overall[d]["top12_hit"]-overall[d+"_legacy"]["top12_hit"])}</td></tr>' for d,label in names.items())
    fields=''.join(f'<tr><th>{html.escape(g.removeprefix("class:").removeprefix("field:"))}{"車" if g.startswith("field:") else ""}</th><td>{v["races"]:,}</td><td>{pct(v["methods"]["risk"]["top12_hit"])}</td><td>{pct(v["methods"]["fusion"]["top12_hit"])}</td></tr>' for g,v in result['summary'].items() if g.startswith(('field:','class:')))
    comparison=result['comparisons_to_risk_formula']['fusion']['day_bootstrap_95_percent_interval']
    cutoff=frozen['phases'][0]['last']
    additions=f'''<section><h2>この検証で分かったこと</h2>
<p><b>統合式は {hits('fusion',12):,}／{total:,} レースで正解を最大12候補に含めました。</b>基準のリスク部評価式は {hits('risk',12):,} レースで、差は {gained:,} レース、{point(combined['top12_hit']-risk['top12_hit'])}です。これは当時の実際の購入成績ではありません。</p>
<p>同じ日のレースをまとめて再抽出した参考区間は {point(comparison[0])} 〜 {point(comparison[1])}。複数方式を検討した過去検証の参考値であり、そのまま本番採用を決める区間ではありません。</p>
<p>保存済み {result['predicted_races']:,} レースをすべて予想し、１〜３着を一意に確認できなかった {sum(result['outcome_exclusions'].values()):,} レースを答え合わせから除外しました。各方式の比較対象は同じ {total:,} レースです。</p>
<h3>① 部署の評価を残すだけでは、この条件では改善しなかった</h3>
<p>１着評価と候補数を揃え、２・３着評価の保持だけを比較しました。部署を残したまま、専門評価そのものの学習方法と古さを見直す必要があります。</p>
<table><tr><th>部署</th><th>従来式で上書き</th><th>部署の評価を保持</th><th>差</th></tr>{pairs}</table>
<h3>② 先着関係式の有無で差が出た</h3>
<p>統合式 {pct(combined['top12_hit'])} に対し、先着関係式を合成から外すと {pct(overall['without_pairwise']['top12_hit'])}。部署の分布を外すと {pct(overall['without_departments']['top12_hit'])}、今回の組み合わせ式を外すと {pct(overall['without_joint']['top12_hit'])} でした。これは残った重みを正規化した比較で、再学習や、関連特徴の完全な除去を伴う因果的な比較ではありません。</p>
<h3>③ １着の外れと、９車・Ｓ級を重点的に調べる</h3>
<p>統合式の３連単最上位候補は、全レースの {pct(combined['first_error_at_1'])} で１着から不一致でした。３着の調整だけで改善したと判断せず、１着の評価も調べる必要があります。９車とＳ級は別の切り口なので、下の件数を足し合わせることはできません。</p>
<table><tr><th>条件</th><th>レース数</th><th>リスク部の評価式</th><th>統合式</th></tr>{fields}</table>
<h3>学習期間と入力の制限</h3>
<p>基礎モデルと部署知識は学習内部の最初の期間（{cutoff} まで）で固定し、後の古い期間で組み合わせ式・配分・確率補正を学習しました。すべての部品を2025年10月8日まで一律に再学習した実験ではありません。部署評価の悪化には、この知識の古さも影響し得るため、評価保持そのものの一般的な否定には使えません。</p>
<p>古いデータの並び情報には、検証済みを示す印がありませんでした。今回の学習では、確認できない並びの関係を新しい組み合わせ式の直接の関係特徴にしていません。部署の評価経由の影響は残っています。過去の並びの再検証も、次に調べる課題です。</p>
<p>順位別市場支持・100倍以上の穴・回収率・高配当的中への依存度は、このデータでは未検証です。既存７部署・リスク部の本番ロジックを保持し、この研究結果だけで社長への組み込みは行っていません。</p></section>'''
    doc=(source/'fixed_year_report.html').read_text(encoding='utf-8')
    doc=doc.replace('<section><h2>同じ検証レースでの比較</h2>',additions+'<section><h2>同じ検証レースでの比較</h2>',1)
    footer=f'<p><a href="{html.escape(execution["run_url"])}">実データ実行記録</a> ／ <a href="fixed_year_results.json">集計結果</a> ／ <a href="fixed_year_frozen_manifest.json">学習期間と固定した配分</a> ／ <a href="fixed_year_inventory.json">対象データの範囲</a> ／ <a href="annual_fusion_report.html">事前予想の統合研究</a></p>'
    doc=doc.replace('</html>',footer+'</html>')
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'fixed_year_report.html').write_text(doc,encoding='utf-8')
    for original,published in [('results.json','fixed_year_results.json'),('inventory.json','fixed_year_inventory.json'),
                               ('frozen_manifest.json','fixed_year_frozen_manifest.json'),
                               ('prediction_manifest.json','fixed_year_prediction_manifest.json'),
                               ('execution_provenance.json','fixed_year_execution_provenance.json')]:
        shutil.copyfile(source/original,destination/published)
    return {'evaluated_races':total,'gained_top12_hits':gained,
        'published_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.glob('fixed_year_*'))}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();print(json.dumps(render(args.source,args.destination),ensure_ascii=False,indent=2))
