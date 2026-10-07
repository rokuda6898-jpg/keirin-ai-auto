"""Evidence-based department reviews; no simulated employee conversations."""
import csv
import json
from datetime import datetime


def records(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def build_race_meetings(output_dir, now):
    path = output_dir / 'latest_race_strategy.json'
    plans = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    candidates = records(output_dir / 'latest_bet_candidates.csv')
    meetings = []
    for plan in plans:
        rid = str(plan['race_id'])
        pool = [r for r in candidates if r.get('race_id') == rid and r.get('bet_type') == 'trifecta'
                and r.get('strategy_version') == plan.get('strategy_version')]
        selected = [r for r in pool if r.get('is_selected', '').lower() == 'true']
        main = [r for r in selected if r.get('ticket_group') == '本線']
        holes = [r for r in selected if r.get('ticket_group') == '穴']
        gaps = ['並び・欠場の取得元との一致は、この会議では未検証。']
        if not pool:
            gaps.append('同じ予想版の候補データなし。')
        stale = 0
        for row in selected:
            try:
                stamp = datetime.fromisoformat(row.get('odds_captured_at_jst', ''))
                age = (now - stamp).total_seconds()
                verified = row.get('odds_verification_status') == 'verified'
                stale += int(not verified or not 0 <= age <= 300)
            except (ValueError, TypeError):
                stale += 1
        if stale:
            gaps.append(f'保存買い目のうち{stale}点は現在のオッズ鮮度・確認状態が不足。')
        if not selected:
            gaps.append('正式な条件成立買い目なし。')
        formation = plan.get('formation', {})
        basis = (f'1着候補 {formation.get("first", [])}。本線{len(main)}点／上限{plan.get("main_limit", 0)}点。'
                 f'推定確率を優先し、EV≥{plan.get("main_ev", 1.10)}を満たした保存候補。'
                 if main else '本線なし：' + str(plan.get('skip_reason') or '条件成立なし。'))
        failure = f'1着候補以外の勝利、2着・3着の組み合わせ違いに注意。荒れ指数{plan.get("chaos_index", "未取得")}。'
        if plan.get('fixed_policy_status') == 'calibration_unverified':
            failure += '勝率の校正未確認のため1着固定は停止。'
        hole_reason = (f'100倍以上・EV≥{plan.get("hole_ev", 1.25)}の保存候補{len(holes)}点。高配当だけで価値を保証しない。'
                       if holes else '穴なし：条件成立なし。穴を強制追加しない。')
        topics = {'main_basis': basis, 'failure_scenario': failure, 'hole_reason': hole_reason, 'information_gaps': gaps}
        opinions = [{'role': role, 'speaking_weight': 1, 'statement': statement} for role, statement in [
            ('データ担当', '／'.join(gaps)), ('予想担当', basis), ('軍師', hole_reason),
            ('リスク担当', failure), ('検証担当', '締切前保存と公式結果で照合し、同じ予算・異なる開催日で比較。少数的中では配分を変えない。')]]
        meetings.append({'race_id': rid, 'venue': plan.get('venue'), 'race_no': plan.get('race_no'),
                         'strategy_version': plan.get('strategy_version'), 'reviewed_at_jst': now.isoformat(timespec='seconds'),
                         'topics': topics, 'opinions': opinions, 'main_count': len(main), 'hole_count': len(holes),
                         'decision': '保存候補の説明。新しい買い目を生成・承認する会議ではない。',
                         'automatic_promotion': False})
    return meetings
