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
    from betting_logic import STRATEGY_VERSION
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
    from race_meeting import build_race_meetings
    race_meetings=build_race_meetings(output_dir, datetime.fromisoformat(now))
    report={'updated_at_jst':now,'implementation':'自動集計・ルールによる担当機能。独立した人間や会話AIを雇ったものではない。',
            'strategy_review': {'strategy_version': STRATEGY_VERSION,
                'weights': {'line_development':33, 'recent_form':15, 'race_score':15, 'riding_style':20, 'track_fit':8, 'opponents':9},
                'weight_scope':'既存モデルへの補正配分。最終的な寄与率や最適値の証明ではない。',
                'optimality_validated':False,
                'risk_budget_policy':'候補は広げるが荒れ指数だけで本線の点数を増やさない。期待値基準を下げない。',
                'fixed_axis_policy':'勝率の独立した校正検証が済むまで固定停止。',
                'agenda':[
                    {'department':'着順分析・軍師','decision':'2着以内・3着以内をちょうど2着・3着へ変換する比較案を締切前保存。', 'evidence':research.get('exact_position_comparison',{}), 'adoption':'比較期間が不足しているため採用保留'},
                    {'department':'リスク・回収率','decision':'過信した固定軸と点数の強制増加を抑制。配分は据え置き。', 'evidence':quality.get('fixed_axis_audit',{}), 'adoption':'保護ルールを適用。精度改善は未確認'},
                    {'department':'検証監査','decision':'開催日を分けて同じレースの的中率・回収率・確率誤差を比較し、変更を審査。', 'evidence':research.get('calibration',{}), 'adoption':'少数の好成績では配分を変更しない'}]},
            'site_status':manager.get('status','未確認'),'roles':roles,
            'employee_workflow': {'roles':['データ担当','予想担当','軍師','リスク担当','検証担当'],
                'equal_speaking_rights':True, 'decision_rule':'発言権は平等。採用は検証根拠で決め、多数決で変更しない。',
                'race_meetings':race_meetings},
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
    review=report['strategy_review']
    body+='<h2>予想方針の会議・審査結果</h2><p>補正配分：ライン・展開33%／調子15%／競走得点15%／脚質20%／場との相性8%／相手関係9%。最適値は未検証です。最終予想の寄与率とは異なります。</p><p>'+esc(review['risk_budget_policy'])+'</p><p>'+esc(review['fixed_axis_policy'])+'</p>'
    for entry in review['agenda']:
        body+='<section><h3>'+esc(entry['department'])+'</h3><p>'+esc(entry['decision'])+'</p><p>判断：'+esc(entry['adoption'])+'</p></section>'
    body+='<p>この会議は各部門の集計結果を使った自動審査です。別々の会話AIが討論する機能ではありません。更新ごとに判断材料を集計します。</p>'
    body+='<h2>レース別の担当報告・会議</h2><p>データ担当→予想担当→軍師→リスク担当→検証担当。発言権は全員同じです。採用は多数決ではなく検証根拠で決めます。以下は保存候補の説明で、現在の購入承認ではありません。</p>'
    for meeting in race_meetings:
        body+='<details><summary>'+esc(meeting['venue'])+' '+esc(meeting['race_no'])+'R｜本線'+str(meeting['main_count'])+'点・穴'+str(meeting['hole_count'])+'点</summary>'
        for key,label in [('main_basis','本線の根拠'),('failure_scenario','外れる展開'),('hole_reason','穴を入れる理由'),('information_gaps','情報不足')]:
            value=meeting['topics'][key]
            body+='<h3>'+label+'</h3><p>'+esc('／'.join(value) if isinstance(value,list) else value)+'</p>'
        for opinion in meeting['opinions']:
            body+='<p><b>'+esc(opinion['role'])+'</b>：'+esc(opinion['statement'])+'</p>'
        body+='</details>'
    for role in roles:
        body+='<section><h3>'+role['role']+'</h3><p>'+role['status']+'</p><p>'+role['task']+'</p><a href="'+role['report']+'">確認結果を見る</a></section>'
    body+='<h2>本番採用の条件</h2><p>同一レースで最低300件の比較、期間を分けた検証、回収率の審査を経て判断します。部署の多数決や数レースの的中だけで自動採用しません。</p><h2>残っている未確認事項</h2><ul>'
    for issue in report['unresolved']:body+='<li>'+esc(issue)+'</li>'
    body+='</ul><p>'+esc(report['implementation'])+'</p><p>更新 '+esc(now)+'</p><a href="../index.html">今日の予想へ</a>'
    page='<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>会社の運用・改善状況</title><link rel="stylesheet" href="../site-ui.css"></head><body><main>'+body+'</main></body></html>'
    (folder/'operations.html').write_text(page,encoding='utf-8')
    return report
