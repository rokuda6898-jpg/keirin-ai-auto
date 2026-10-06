"""Company oversight based on measured evidence, not fictional autonomous employees."""
import html
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from common import OUTPUT_DIR


def load(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def build_company_operations(output_dir=OUTPUT_DIR):
    folder=output_dir/'company';folder.mkdir(parents=True,exist_ok=True)
    coverage=load(folder/'validation_coverage.json')
    research=load(folder/'selection_research.json')
    quality=load(folder/'prediction_quality.json')
    manager=load(output_dir/'manager_status.json')
    checks=coverage.get('time_checks',{})
    roles=[{'role':'データ品質担当','status':'要確認' if checks.get('quote_after_snapshot_or_close',0)>0 else '検証中',
            'task':'オッズの鮮度・時刻・不一致を確認。未来時刻や5分を超えた価格を候補生成から除外。',
            'evidence':checks,'report':'validation_coverage.html'},
           {'role':'検証監査担当','status':'記録を蓄積中','task':'開催予定に対する保存・公式結果・成績照合の抜けを確認。',
            'evidence':coverage.get('coverage',{}),'report':'validation_coverage.html'},
           {'role':'着順分析担当','status':'比較検証中','task':'1着固定、期待値順、2・3着分散案を別々に検証。',
            'evidence':quality.get('reason_counts',{}),'report':'selection_research.html'},
           {'role':'回収率審査担当','status':'自動採用停止・検証不足','task':'同じレース・同じ100円単位で現行案と比較。少数の好成績を本番採用しない。',
            'evidence':{'probability_first_paired_races':research.get('paired_races',0),
                        'lower_paired_races':research.get('lower_comparison',{}).get('paired_races',0),
                        'axis_paired_races':research.get('axis_comparison',{}).get('paired_races',0)},'report':'selection_research.html'}]
    now=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')
    report={'updated_at_jst':now,'implementation':'自動集計・ルールによる担当機能。独立した人間や会話AIを雇ったものではない。',
            'site_status':manager.get('status','未確認'),'roles':roles,
            'release_policy':{'minimum_same_race_comparisons':300,'chronological_holdout_required':True,
                              'profit_gate_required':True,'automatic_promotion':False},
            'target_return_rate':1.20,'target_validated':False,
            'unresolved':['現行買い目の実戦検証は少数。回収率120%の達成は未確認。',
                          '全特徴量とモデル学習の利用可能時刻は未確認。',
                          '全国の全レース取得と全オッズ取得を保証できる状況ではない。',
                          '自動更新の実行時刻はGitHub側の待ち時間によって遅れる場合がある。']}
    (folder/'operations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    esc=lambda value:html.escape(str(value))
    body='<h1>会社の運用・改善状況</h1><p>サイト稼働状況：'+esc(report['site_status'])+'</p><p>稼働が正常でも、予想精度・回収率の改善は別に検証します。</p>'
    links=[('annual_department_report.html','4部署の予想と選手別1〜3年の成績'),('annual_strategist_report.html','軍師の分析と比較'),
           ('selection_research.html','買い目・確率補正の比較検証'),('validation_coverage.html','検証の抜けと条件別成績'),
           ('ticket_return_department.html','本線・穴の的中率と回収率')]
    body+='<nav>'+''.join('<p><a href="'+path+'">'+label+'</a></p>' for path,label in links)+'</nav><h2>担当機能</h2>'
    for role in roles:
        body+='<section><h3>'+role['role']+'</h3><p>'+role['status']+'</p><p>'+role['task']+'</p><a href="'+role['report']+'">確認結果を見る</a></section>'
    body+='<h2>本番採用の条件</h2><p>同一レースで最低300件の比較、期間を分けた検証、回収率の審査を経て判断します。部署の多数決や数レースの的中だけで自動採用しません。</p><h2>残っている未確認事項</h2><ul>'
    for issue in report['unresolved']:body+='<li>'+esc(issue)+'</li>'
    body+='</ul><p>'+esc(report['implementation'])+'</p><p>更新 '+esc(now)+'</p><a href="../index.html">今日の予想へ</a>'
    page='<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>会社の運用・改善状況</title><link rel="stylesheet" href="../site-ui.css"></head><body><main>'+body+'</main></body></html>'
    (folder/'operations.html').write_text(page,encoding='utf-8')
    return report
