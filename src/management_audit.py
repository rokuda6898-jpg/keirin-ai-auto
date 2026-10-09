"""Independent, read-only operational owners. Never generates or changes picks."""
import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OWNERS = {
    'data': 'データ品質責任者',
    'coverage': '全レース予想責任者',
    'timing': '締切・保存責任者',
    'settlement': '成績・払戻責任者',
    'operations': '稼働・通知責任者',
}
WORKFLOWS = ('site-manager.yml','manager-watchdog.yml','daily.yml','intraday.yml',
             'fusion-shadow-live.yml','quick-results.yml','settle.yml','public-live-data.yml','pages.yml')


def workflow_findings(workflow, runs, now_s):
    """An old failure remains open until a later successful completed run."""
    meaningful = [r for r in runs if r.get('conclusion') != 'skipped']
    completed = next((r for r in meaningful if r['status'] == 'completed'), None)
    stalled = next((r for r in meaningful if r['status'] != 'completed'
                    and now_s-(timestamp(r.get('created_at')) or now_s)>40*60), None)
    run = stalled or (completed if completed and completed.get('conclusion') != 'success' else None)
    if not run:
        return []
    return [{'owner':'operations','code':'workflow_'+workflow,'level':'error',
             'message':workflow+(' が40分以上待機・実行中です。' if stalled else ' の直近完了処理が失敗しています。'),
             'race_ids':[], 'run_url':run.get('html_url','')}]


def timestamp(value):
    try:
        stamp = datetime.fromisoformat(str(value))
        return stamp.timestamp() if stamp.tzinfo else None
    except (ValueError, TypeError):
        return None


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def valid_ticket(buy):
    parts = str(buy).split('-')
    return len(parts) == 3 and len(set(parts)) == 3 and all(p in '123456789' and len(p) == 1 for p in parts)


def audit(feed, now):
    findings = []
    def add(owner, code, message, races=(), level='error'):
        findings.append(dict(owner=owner, code=code, level=level, message=message,
                             race_ids=sorted(set(races))))
    today = now.date().isoformat()
    current = [r for r in feed.get('races', []) if r.get('date') == today]
    now_s = now.timestamp()
    if feed.get('schema') != 1 or not isinstance(feed.get('races'), list):
        add('data', 'invalid_feed', '公開データの形式が不正です。')
    if 7 <= now.hour <= 23 and feed.get('schedule_date') != today:
        add('data', 'schedule_stale', '本日の開催日程がまだ反映されていません。休催日か取得失敗か確認が必要です。')
    active = any((number(r.get('close_at')) or 0) > now_s for r in current)
    updated = timestamp(feed.get('updated_at'))
    if active and (updated is None or now_s-updated > 45*60 or updated > now_s+60):
        add('data', 'feed_stale', '締切前レースがありますが公開データが45分以上古い、または時刻が不正です。')
    ids = [r.get('id') for r in feed.get('races', [])]
    if len(set(ids)) != len(ids):
        add('data', 'duplicate_races', '公開データにレースIDの重複があります。')
    missing, late, invalid, unpaid, overdue = {}, [], [], [], []
    coverage = {}
    for mode in ('company', 'shadow'):
        missing[mode] = []
        due_missing = []
        for r in current:
            close = number(r.get('close_at'))
            if close is None:
                invalid.append(r['id'])
            prediction = r.get(mode)
            tickets = prediction.get('tickets', []) if prediction else []
            if not tickets:
                missing[mode].append(r['id'])
                if close and close <= now_s+15*60:
                    due_missing.append(r['id'])
        coverage[mode] = {'total': len(current), 'saved': len(current)-len(missing[mode]),
                          'missing': len(missing[mode])}
        if due_missing:
            add('coverage', 'missing_'+mode, ('会社' if mode == 'company' else '独立')+
                '予想が締切15分前または締切後にも未作成です。後付けせず未作成として記録します。', due_missing)
    for r in feed.get('races', []):
        for mode in ('company', 'shadow'):
            p = r.get(mode)
            if not p:
                continue
            tickets = p.get('tickets', [])
            cars = {str(x.get('car')) for x in r.get('riders', [])}
            if not tickets or len(tickets) > 12 or len(set(tickets)) != len(tickets) or any(
                not valid_ticket(b) or (cars and not set(b.split('-')) <= cars) for b in tickets
            ):
                invalid.append(r['id'])
            stamp, close = timestamp(p.get('snapshot_at')), number(r.get('close_at'))
            if stamp is None or close is None or stamp >= close or stamp > now_s+60:
                late.append(r['id'])
            for buy in set(tickets) & set(r.get('actual', [])):
                if (number(r.get('payouts', {}).get(buy)) or 0) <= 0:
                    unpaid.append(r['id'])
        if r.get('date') == today and (number(r.get('start_at')) or now_s) < now_s-120*60 and not r.get('actual'):
            overdue.append(r['id'])
    if invalid:
        add('coverage', 'invalid_tickets', '車番・重複・点数・締切時刻に不正があります。', invalid)
    if late:
        add('timing', 'late_snapshot', '締切前保存を証明できない予想があります。成績比較に使用しないでください。', late)
    if unpaid:
        add('settlement', 'unknown_payout', '的中した買い目の公式払戻が未取得です。回収率を確定できません。', unpaid)
    if overdue:
        add('settlement', 'results_overdue', '発走から2時間以上経過しても結果未確認です。中止・延期も確認してください。', overdue, 'warning')
    # Matched cohort prevents a method from benefiting from easier/missing races.
    excluded = set(late) | set(invalid)
    common = [r for r in current if r['id'] not in excluded and r.get('actual') and all(r.get(m, {}).get('tickets') if r.get(m) else False for m in ('company','shadow'))]
    comparison = {}
    for mode in ('company', 'shadow'):
        hits, stake, returned, known, returns = 0, 0, 0, True, []
        for r in common:
            tickets = r[mode]['tickets']
            winning = set(tickets) & set(r['actual'])
            hits += bool(winning)
            stake += len(tickets)*100
            values = [number(r.get('payouts', {}).get(b)) for b in winning]
            if any(v is None or v <= 0 for v in values):
                known = False
            payout = sum(v for v in values if v is not None and v > 0)
            returns.append(payout)
            returned += payout
        comparison[mode] = {'races':len(common), 'hits':hits, 'stake_yen':stake,
                            'return_yen':returned if known else None,
                            'largest_hit_return_share':max(returns, default=0)/returned if known and returned else None}
    overlaps = []
    for r in current:
        opinions = (r.get('company') or {}).get('opinions', [])
        for i, a in enumerate(opinions):
            for b in opinions[i+1:]:
                x, y = set(a['tickets']), set(b['tickets'])
                if x or y:
                    overlaps.append(len(x & y)/len(x | y))
    return {'checked_at':now.isoformat(timespec='seconds'), 'owners':OWNERS,
            'status':'error' if any(f['level']=='error' for f in findings) else 'warning' if findings else 'healthy',
            'findings':findings, 'coverage':coverage, 'matched_comparison':comparison,
            'comparison_scope':'同じレース・初回締切前予想・各点100円。点数差は投資額に反映。因果的な優劣判定ではありません。',
            'department_mean_ticket_overlap':sum(overlaps)/len(overlaps) if overlaps else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    now = datetime.now(ZoneInfo('Asia/Tokyo'))
    try:
        feed = json.loads((args.root/'outputs/public_live.json').read_text(encoding='utf-8'))
        report = audit(feed, now)
        with (args.root/'outputs/latest_race_schedule.csv').open(encoding='utf-8-sig', newline='') as stream:
            expected = {r['race_id'] for r in csv.DictReader(stream) if r['date'] == now.date().isoformat()}
        omitted = expected - {r['id'] for r in feed['races']}
        if omitted:
            report['status'] = 'error'
            report['findings'].append({'owner':'coverage','code':'omitted_races','level':'error',
                'message':'取得済み開催日程のレースが公開一覧から欠落しています。','race_ids':sorted(omitted)})
    except (OSError, ValueError, TypeError, KeyError) as error:
        report = {'checked_at':now.isoformat(), 'owners':OWNERS, 'status':'error', 'findings':[
            {'owner':'data','code':'audit_input_error','level':'error','message':'監査入力の読込・検査に失敗: '+type(error).__name__,'race_ids':[]}]}
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        for workflow in WORKFLOWS:
            try:
                url = 'https://api.github.com/repos/'+os.environ['GITHUB_REPOSITORY']+'/actions/workflows/'+workflow+'/runs?branch=main&per_page=15'
                request = Request(url, headers={'Authorization':'Bearer '+os.environ['GITHUB_TOKEN'], 'Accept':'application/vnd.github+json'})
                with urlopen(request, timeout=20) as response:
                    runs = json.load(response)['workflow_runs']
                report['findings'].extend(workflow_findings(workflow,runs,now.timestamp()))
            except Exception:
                report['findings'].append({'owner':'operations','code':'workflow_check_unavailable', 'level':'error',
                    'message':'更新処理の稼働確認に失敗しました。正常とは判断できません。','race_ids':[]})
                break
        if any(f['level']=='error' for f in report['findings']):
            report['status'] = 'error'
    report['email_configured'] = all(os.environ.get(k) for k in ('SMTP_HOST','SMTP_USER','SMTP_PASSWORD','ALERT_FROM','ALERT_TO'))
    target = args.root/'outputs/management_status.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status':report['status'], 'findings':len(report['findings']), 'email_configured':report['email_configured']}))


if __name__ == '__main__':
    main()
