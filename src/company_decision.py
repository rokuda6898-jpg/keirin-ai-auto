"""Auditable, prospective company decisions; legacy risk tickets stay intact.

Every department submits its own order/tickets. The strategist aggregates
evidence-family reciprocal ranks; the president freezes at most twelve unique
proposals. Scores are votes, not calibrated probabilities or expected returns.
"""
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from department_coverage import DEPARTMENTS, LABELS
from forecast_coverage import reconcile_coverage, schedule_rows

VERSION = 'company_evidence_family_rrf_v2'


def ticket_valid(buy, cars):
    parts = str(buy).split('-')
    return len(parts) == 3 and len(set(parts)) == 3 and all(p in cars for p in parts)


def decide(group, cars, now, close):
    """No confidence/EV selection of races, no copied missing department votes."""
    if not math.isfinite(close) or now.timestamp() >= close:
        raise ValueError('締切後の新規予想は禁止')
    submissions = {}
    for row in group:
        name = row['department']
        if name not in DEPARTMENTS or name in submissions:
            raise ValueError('部署の重複または不明な部署')
        stamp = datetime.fromisoformat(row['snapshot_at'])
        if stamp.tzinfo is None or not stamp.timestamp() <= now.timestamp() < float(row['close_at']):
            raise ValueError('部署の提出時刻・締切が不正')
        if not row.get('forecast_available'):
            raise ValueError('未提出の部署あり')
        ordered = ['-'.join(map(str, row.get('top3_cars', [])))]
        ordered += [str(t['buy']) for t in row.get('tickets', [])]
        ordered = list(dict.fromkeys(b for b in ordered if ticket_valid(b, cars)))
        if not ordered:
            raise ValueError('部署の有効な買い目なし')
        submissions[name] = ordered
    if set(submissions) != set(DEPARTMENTS):
        raise ValueError('全部署の提出が必要')
    scores = defaultdict(float)
    evidence = defaultdict(dict)
    origins = {r['department']:r.get('opinion_origin',r['department']) for r in group}
    # Strategist is a synthesis of existing votes, not new evidence. Only
    # explicitly shared fallbacks with exactly the same ranking share a vote;
    # genuine independent agreement is not penalized or forced apart.
    families = defaultdict(list)
    for department, tickets in submissions.items():
        if department == 'strategist_department':
            continue
        origin = origins[department]
        key = ('shared_model_fallback', tuple(tickets)) if origin == 'shared_model_fallback' else (department,)
        families[key].append(department)
    weights = {department:1/len(members) for members in families.values() for department in members}
    weights['strategist_department'] = 0.0
    for department, tickets in submissions.items():
        # Each independent evidence family has one total vote, regardless of
        # ticket count. A derived strategist summary has no additional vote.
        total = sum(1/(rank+1) for rank in range(len(tickets)))
        for rank, buy in enumerate(tickets):
            contribution = weights[department]*(1/(rank+1))/total
            scores[buy] += contribution
            evidence[buy][department] = contribution
    ranked = sorted((buy for buy in scores if scores[buy] > 0), key=lambda buy: (-scores[buy], tuple(map(int, buy.split('-')))))[:12]
    return {'version': VERSION, 'top12': ranked, 'department_submissions': submissions,
            'department_evidence': {r['department']: {'origin': origins[r['department']],
                'context_status': r.get('specialist_evidence', {}).get('status'),
                'vote_weight': weights[r['department']]} for r in group},
            'strategist': {'rule': '独立した根拠を1票として順位集約。同じ共通補完は分割し、軍師の集約結果を再加算しない。',
                           'department_vote_weights': weights,
                           'scores': {buy: scores[buy] for buy in ranked}},
            'president': {'decision': '全部署提出・締切前・有効車番を確認し上位最大12点を確定',
                          'adopted': ranked},
            'ticket_evidence': {buy: evidence[buy] for buy in ranked},
            'purchase_authorized': False}


def build_company_decisions(pred, report, now, output_dir):
    started = time.monotonic()
    folder = output_dir/'company'
    folder.mkdir(parents=True, exist_ok=True)
    ledger = folder/'company_decision_ledger.jsonl'
    rows = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines() if line.strip()] if ledger.exists() else []
    saved = {}
    for row in sorted(rows, key=lambda r: r['snapshot_at_jst']):
        stamp = datetime.fromisoformat(row['snapshot_at_jst'])
        if stamp.tzinfo is None or stamp.timestamp() >= float(row['close_at']):
            raise ValueError('会社予想の保存時刻が不正')
        saved.setdefault(str(row['race_id']), row)
    grouped = defaultdict(list)
    for row in report['predictions']:
        grouped[str(row['race_id'])].append(row)
    coverage, fresh = [], []
    for rid, race in pred.groupby('race_id', sort=False):
        rid = str(rid)
        item = race.iloc[0]
        entry = {key: str(item.get(key, '')) for key in ('date', 'venue', 'race_no')}
        entry.update(race_id=rid, close_at=float(item.close_at), status='予想なし', reason='')
        if rid not in saved:
            try:
                decision = decide(grouped[rid], {str(int(c)) for c in race.car_no}, now, float(item.close_at))
                completed = now + timedelta(seconds=time.monotonic() - started)
                if completed.timestamp() >= float(item.close_at):
                    raise ValueError('計算中に締切を経過')
                row = {**entry, **decision, 'snapshot_at_jst': completed.isoformat(timespec='seconds')}
                fresh.append(row)
                saved[rid] = row
            except (ValueError, KeyError, TypeError) as error:
                entry['reason'] = str(error)
        if rid in saved:
            entry.update(status='買い目固定済み', reason='')
        coverage.append(entry)
    if fresh:
        with ledger.open('a', encoding='utf-8') as stream:
            for row in fresh:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    coverage = reconcile_coverage(coverage, schedule_rows(output_dir/'latest_race_schedule.csv', now.date().isoformat()), set(saved), now)
    result = {'updated_at_jst': now.isoformat(), 'coverage': coverage,
              'predictions': [r for r in saved.values() if r['date'] == now.date().isoformat()]}
    (folder/'company_decision_predictions.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result


def add_company_decisions(soup, output_dir):
    path = output_dir/'company/company_decision_predictions.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    saved = {str(r['race_id']): r for r in data.get('predictions', [])}
    for previous in soup.select('.company-final-decision'):
        previous.decompose()
    for race in soup.select('article.race'):
        rid = race.get('id', '').removeprefix('race-')
        panel = race.select_one('.picks-panel')
        if panel is None:
            continue
        baseline = panel.find('h3', recursive=False)
        if baseline:
            baseline.string = '従来の本線・穴（比較用）'
        section = soup.new_tag('section', attrs={'class': 'company-final-decision'})
        title = soup.new_tag('h3'); title.string = '全部署会議・社長の最終予想'; section.append(title)
        row = saved.get(rid)
        if row:
            tickets = soup.new_tag('p'); tickets.string = ' ／ '.join(row['top12']); section.append(tickets)
            details = soup.new_tag('details'); heading = soup.new_tag('summary'); heading.string = '部署の意見・軍師の集約を見る'; details.append(heading)
            for name, buys in row['department_submissions'].items():
                line = soup.new_tag('p'); line.string = LABELS[name]+': '+' ／ '.join(buys); details.append(line)
            line = soup.new_tag('p'); line.string = row['strategist']['rule']+' '+row['president']['decision']; details.append(line)
            section.append(details)
            line = soup.new_tag('small'); line.string = '締切前保存 '+row['snapshot_at_jst']+'。集約点は確率ではありません。'; section.append(line)
        else:
            line = soup.new_tag('p'); line.string = '会議予想の保存待ち。締切後の後付け予想は作成しません。'; section.append(line)
        panel.insert(0, section)
    return soup
