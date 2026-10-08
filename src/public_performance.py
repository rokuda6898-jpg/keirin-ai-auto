"""Human-readable performance definitions using existing audited results only."""
import html
import json
from common import OUTPUT_DIR


def percentage(value):
    return "未集計" if value is None else f"{value * 100:.1f}%"


def build_performance_page(output_dir=OUTPUT_DIR):
    def read(name):
        path = output_dir / "company" / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    tickets = read("ticket_return_department.json")
    winners = read("verified_live_audit.json")
    total = tickets.get("shadow", {}).get("total", {})
    flat = total.get("flat_100yen", {})
    count = total.get("bet_races", 0)
    hits = total.get("hits", 0)
    rate = percentage(total.get("hit_rate")) if count else "未集計"
    roi = percentage(flat.get("return_rate")) if flat.get("stake_yen", 0) else "未集計"
    winner_count = winners.get("timestamp_verified_races", 0)
    winner_rate = percentage(winners.get("hit_rate")) if winner_count else "未集計"
    updated = html.escape(str(tickets.get("updated_at_jst", "未取得")))
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>成績と数字の見方</title><style>
body{{font-family:system-ui;background:#f4f7fb;color:#172b45;margin:0;padding:20px}}main{{max-width:880px;margin:auto}}section{{background:white;border-radius:16px;padding:22px;margin:16px 0}}h1{{font-size:24px}}h2{{font-size:20px}}p{{line-height:1.8}}.numbers{{display:flex;gap:25px;flex-wrap:wrap}}.numbers strong{{display:block;font-size:30px}}small{{color:#52657e}}a{{color:#0965c7}}
</style></head><body><main><a href="index.html">今日の予想</a> ／ <a href="history.html">買い目履歴</a><h1>成績と数字の見方</h1>
<section><h2>3連単の買い目成績（検証用）</h2><div class="numbers"><div>買い目の的中率<strong>{rate}</strong></div><div>回収率・各点100円の試算<strong>{roi}</strong></div></div>
<p>対象 {count}レース ／ 的中 {hits}レース<br>試算の購入額 {flat.get('stake_yen', 0):,}円 ／ 公式払戻による試算 {flat.get('return_yen', 0):,}円</p>
<p>締切前に保存した、本線と穴を合わせた買い目が対象です。1レースで1点でも当たれば「的中1レース」。見送り・結果未確定・公式払戻未取得は、この的中率と回収率に含めません。</p>
<p>的中率＝的中レース数 ÷ 買い目がある確定レース数。<br>回収率＝払戻の合計 ÷ 全買い目の購入額 × 100。100%で収支が同額、120%なら1万円に対して1万2千円の払戻です。</p>
<p>実際に車券を購入した成績ではありません。対象0レースのときは「未集計」です。</p><a href="company/ticket_return_department.html">本線・穴の内訳と配分額の詳細</a></section>
<section><h2>1着候補の的中率（参考）</h2><div class="numbers"><div>1着候補が実際に勝った割合<strong>{winner_rate}</strong></div></div>
<p>的中 {winners.get('hits', 0)} ／ 締切前時刻を確認できた {winner_count}レース。3連単が当たった割合ではなく、この数字から回収率は分かりません。</p>
<p>従来記録のうち、締切前時刻を確認できない {winners.get('unverified_rows', 0)}件は除外。旧記録の時刻確認であり、変更できない事前保存の証明とは区別しています。上の買い目成績とは対象と期間が異なります。</p></section>
<section><h2>予想画面の％・部署の成績</h2><p>選手横の％は、AIが推定する「今回1着になる確率」です。過去の的中率ではありません。<br>各部署の成績は検証用予想の比較です。上の成績に合算しません。<br>過去データでの再計算やモデルのテスト成績も、締切前に保存した買い目成績とは分けて扱います。</p>
<a href="company/annual_department_report.html">各部署の検証用成績</a> ／ <a href="company/prediction_quality.html">外れ方とAI確率の検証</a> ／ <a href="company/mark_performance.html">◎○▲△☆の印別成績（発走前固定）</a></section>
<p>回収率120%は目標です。達成を確認した成績ではありません。<br><small>買い目成績の更新 {updated}</small></p></main></body></html>"""
    from site_ui import continuous_section,CONTINUOUS_JS,assets
    assets()
    page=page.replace('</head>','<link rel="stylesheet" href="site-ui.css"></head>')
    page=page.replace('<h1>成績と数字の見方</h1>','<h1>成績と数字の見方</h1><p><a href="company/validation_coverage.html">検証の抜けと条件別成績を見る</a></p>'+continuous_section())
    page=page.replace('</body>','<script>'+CONTINUOUS_JS+'</script></body>')
    (output_dir / "performance.html").write_text(page, encoding="utf-8")
    return page
