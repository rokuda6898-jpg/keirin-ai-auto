"""Bounded continued exploration of existing conditional trifecta evidence.

This is NOT a new holdout. All dates/results were previously inspected.
No model fitting, odds estimation, allocation changes, or live-file writes.
This standalone research runner is not connected to any production publisher.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from itertools import permutations
import csv
import hashlib
import json
import math
from pathlib import Path

BASE = 'saved_top12'
SAVED_POLICIES = (BASE, 'top10', 'retain90_min6')
FULL_POLICIES = (BASE, 'pair_cap3_top12', 'third_temper_half_top12', 'two_triple_boxes12')
DESCRIPTIONS = {
    BASE: 'Original frozen probability-ranked top twelve, unchanged.',
    'top10': 'Keep the first ten of the original frozen twelve.',
    'retain90_min6': 'Smallest probability-ranked prefix retaining at least 90% of original top-twelve model mass, minimum six and maximum twelve tickets.',
    'pair_cap3_top12': 'Greedily take the highest full-q tickets with at most three tickets for each ordered first-second pair, until twelve tickets.',
    'third_temper_half_top12': 'Preserve each ordered first-second pair mass; replace its conditional third-place distribution by normalized square roots; select the twelve largest resulting scores.',
    'two_triple_boxes12': 'Rank unordered three-rider groups by sum of all six ordered full-q probabilities, then buy all six permutations of the two leading groups.',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ticket_tuple(ticket):
    parts = tuple(int(x) for x in ticket.split('-'))
    assert len(parts) == len(set(parts)) == 3 and all(1 <= x <= 9 for x in parts)
    return parts


def whole_yen(value):
    number = Decimal(str(value))
    assert number.is_finite() and number > 0 and number == int(number)
    return int(number)


def selected_saved(tickets, probabilities, policy):
    assert len(tickets) == len(probabilities) == 12 and len(set(tickets)) == 12
    if policy == BASE:
        return list(tickets)
    if policy == 'top10':
        return list(tickets[:10])
    if policy == 'retain90_min6':
        target = math.fsum(probabilities) * .90
        count = next(n for n in range(6, 13) if math.fsum(probabilities[:n]) >= target)
        return list(tickets[:count])
    raise ValueError(policy)


def third_tempered_scores(q):
    groups = defaultdict(list)
    for ticket, value in q.items():
        groups[ticket_tuple(ticket)[:2]].append((ticket, value))
    scores = {}
    for rows in groups.values():
        mass = math.fsum(value for _, value in rows)
        denominator = math.fsum(math.sqrt(value) for _, value in rows)
        for ticket, value in rows:
            scores[ticket] = mass * math.sqrt(value) / denominator if denominator else 0.0
    return scores


def selected_full(q, policy):
    order = sorted(q, key=lambda ticket: (-q[ticket], ticket))
    if policy == BASE:
        return order[:12]
    if policy == 'pair_cap3_top12':
        counts, selected = Counter(), []
        for ticket in order:
            pair = ticket_tuple(ticket)[:2]
            if counts[pair] < 3:
                selected.append(ticket)
                counts[pair] += 1
            if len(selected) == 12:
                return selected
        raise ValueError('Unable to fill twelve tickets under pair cap')
    if policy == 'third_temper_half_top12':
        scores = third_tempered_scores(q)
        return sorted(scores, key=lambda ticket: (-scores[ticket], ticket))[:12]
    if policy == 'two_triple_boxes12':
        groups = defaultdict(list)
        for ticket in q:
            groups[tuple(sorted(ticket_tuple(ticket)))].append(ticket)
        assert all(len(tickets) == 6 for tickets in groups.values())
        ranked = sorted(groups, key=lambda group: (-math.fsum(q[t] for t in groups[group]), group))
        selected = [ticket for group in ranked[:2] for ticket in sorted(groups[group])]
        assert len(selected) == len(set(selected)) == 12
        return selected
    raise ValueError(policy)


def settle(tickets, outcome):
    assert tickets and len(tickets) == len(set(tickets))
    winners = outcome['winning_buys']
    assert winners and len(winners) == len(set(winners))
    payouts = {t: whole_yen(outcome['payouts'][t]) for t in winners}
    hit_tickets = sorted(set(tickets).intersection(winners))
    paid = sum(payouts[t] for t in hit_tickets)
    return dict(tickets=len(tickets), invested_yen=len(tickets) * 100,
                payout_yen=paid, hit=bool(hit_tickets), winning_selected=hit_tickets,
                torigami=bool(hit_tickets) and paid < len(tickets) * 100)


def totals(records, policy):
    values = [r['policies'][policy] for r in records]
    invested = sum(r['invested_yen'] for r in values)
    paid = sum(r['payout_yen'] for r in values)
    hits = sum(r['hit'] for r in values)
    return dict(races=len(values), hits=hits, hit_rate=hits / len(values) if values else None,
                invested_yen=invested, payout_yen=paid, profit_yen=paid-invested,
                roi=paid/invested if invested else None,
                average_tickets=invested/100/len(values) if values else None,
                torigami=sum(r['torigami'] for r in values))


def comparisons(records, policies):
    baseline = totals(records, BASE)
    answer = {}
    for policy in policies:
        v = totals(records, policy)
        v['hit_change_pp'] = 100 * (v['hit_rate'] - baseline['hit_rate']) if records else None
        v['roi_change_pp'] = 100 * (v['roi'] - baseline['roi']) if records else None
        v['changed_ticket_set_races'] = sum(set(r['decisions'][policy]) != set(r['decisions'][BASE]) for r in records)
        v['hits_gained'] = sum(r['policies'][policy]['hit'] and not r['policies'][BASE]['hit'] for r in records)
        v['hits_lost'] = sum(r['policies'][BASE]['hit'] and not r['policies'][policy]['hit'] for r in records)
        v['jointly_improves_hit_rate_and_roi'] = bool(records and v['hit_change_pp'] > 0 and v['roi_change_pp'] > 0)
        if len(records) > 1:
            largest = min(records, key=lambda r: (-r['policies'][policy]['payout_yen'], r['race_id']))
            reduced = [r for r in records if r['race_id'] != largest['race_id']]
            sensitivity_base, sensitivity = totals(reduced, BASE), totals(reduced, policy)
            v['remove_largest_candidate_return'] = dict(race_id=largest['race_id'],
                removed_return_yen=largest['policies'][policy]['payout_yen'],
                candidate=sensitivity, baseline=sensitivity_base,
                roi_change_pp=100*(sensitivity['roi']-sensitivity_base['roi']))
        answer[policy] = v
    return answer


def correction_outcome(raw, report, row):
    # Independently validate the immutable official payoff used by the existing
    # correction audit. Do not infer a winning ticket from final odds.
    blob_sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    evidence = report['correction_evidence']
    assert blob_sha == evidence['raw_blob_sha']
    page = json.loads(raw)
    data, race = page['data'], page['data']['race']
    assert page['sha256'] == evidence['raw_source_sha256']
    assert page['captured_at'] == evidence['captured_at']
    assert race['id'] == row['race_id'] == evidence['race_id']
    assert page['meta']['race_id'] == row['race_id'] and page['meta']['date'] == row['date']
    assert race['closeAt'] == row['close_at'] and race['cancel'] is False
    assert data['trifectaPayoffStatus'] == 1 and not any(e['absent'] for e in data['entries'])
    assert datetime.fromisoformat(page['captured_at']).timestamp() > float(row['close_at'])
    winning_ids = data['trifectaWinningOddsIds']
    assert winning_ids and len(winning_ids) == len(set(winning_ids))
    quotes = data['trifecta']
    assert len(quotes) == len({q['id'] for q in quotes}) == race['entriesNumber']*(race['entriesNumber']-1)*(race['entriesNumber']-2)
    payouts = {}
    for quote in quotes:
        assert quote['raceId'] == row['race_id'] and quote['unitPrice'] == 100 and quote['absent'] is False
        if quote['id'] in winning_ids:
            payouts['-'.join(map(str, quote['key']))] = whole_yen(quote['payoffUnitPrice'])
        else:
            assert quote['payoffUnitPrice'] == 0
    assert len(payouts) == len(winning_ids) and payouts == evidence['payouts']
    return dict(winning_buys=list(payouts), payouts=payouts)


def run(work_root, output_prefix):
    """Read an explicit evidence directory and write only explicit result artifacts."""
    work_root = Path(work_root).resolve()
    output_prefix = Path(output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    relative_paths = ['data/conditional_live_ledger.jsonl', 'analysis_rows_2026-10-20.json',
        'replay/reconstructed_joint_distributions.jsonl', 'replay/reconstruction_check.json',
        'data/outcome_096220261019.json', 'research/outer_blocks_eight_v1_settlement_corrected.json']
    sources = {name: (work_root/name).read_bytes() for name in relative_paths}
    ledger_rows = [json.loads(line) for line in sources[relative_paths[0]].decode().splitlines() if line]
    ledger = {row['race_id']: row for row in ledger_rows}
    assert len(ledger) == len(ledger_rows)
    audited = {r['id']: r for r in json.loads(sources[relative_paths[1]])['flat_100']}
    reconstructed = [json.loads(line) for line in sources[relative_paths[2]].decode().splitlines() if line]
    reconstruction = json.loads(sources[relative_paths[3]])
    corrected = json.loads(sources[relative_paths[5]])
    assert digest(sources[relative_paths[0]]) == corrected['source_ledger_sha256'] == reconstruction['ledger_sha256']
    assert digest(sources[relative_paths[2]]) == reconstruction['reconstruction_sha256']
    assert len(reconstructed) == len({r['race_id'] for r in reconstructed}) == reconstruction['races']
    correction_id = corrected['added_race_ids'][0]
    assert corrected['added_race_ids'] == ['096220261019'] and correction_id not in audited
    repaired = correction_outcome(sources[relative_paths[4]], corrected, ledger[correction_id])
    clean_ids = set(audited) | {correction_id}
    assert len(clean_ids) == corrected['all']['flat_100']['primary']['initial']['races'] == 646
    forecast_inputs, outcomes = {}, {}
    for rid in sorted(clean_ids):
        row = ledger[rid]
        assert datetime.fromisoformat(row['snapshot_at_jst']).timestamp() < float(row['close_at'])
        assert row['training_cutoff_exclusive'] <= row['date'] and not row.get('cancelled_cars') and not row.get('returned_buys')
        tickets, probabilities = row['top12'], row['probabilities']
        assert len(tickets) == len(probabilities) == 12 and len(set(tickets)) == 12
        assert all(math.isfinite(q) and 0 <= q <= 1 for q in probabilities) and math.fsum(probabilities) <= 1 + 1e-12
        assert all(a >= b for a,b in zip(probabilities, probabilities[1:]))
        for ticket in tickets:
            ticket_tuple(ticket)
        forecast_inputs[rid] = dict(race_id=rid, date=row['date'], tickets=list(tickets), probabilities=list(probabilities))
        outcomes[rid] = repaired if rid == correction_id else dict(winning_buys=list(row['actual']), payouts=dict(row['payouts']))
    # The selectors receive only forecast probabilities/tickets, never outcomes,
    # race identity, final odds, grade, venue, or dates.
    saved_decisions = {rid: {policy: selected_saved(r['tickets'], r['probabilities'], policy) for policy in SAVED_POLICIES}
                       for rid, r in forecast_inputs.items()}
    q_by_race, full_decisions, full_excluded = {}, {}, []
    for replay in reconstructed:
        rid = replay['race_id']
        if rid not in clean_ids:
            full_excluded.append(rid)
            continue
        row = ledger[rid]
        assert replay['input_preclose_provenance_verified'] and replay['top12_match'] and replay['probabilities_exact_match']
        assert replay['model_sha256'] == row['model_sha256'] and replay['frozen_snapshot_at_jst'] == row['snapshot_at_jst']
        assert replay['top12'] == row['top12'] and replay['top12_probabilities'] == row['probabilities']
        q = dict(zip(replay['keys'], replay['probabilities']))
        assert len(q) == len(replay['keys']) == len(replay['probabilities']) == math.perm(replay['field'], 3)
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in q.values()) and abs(math.fsum(q.values())-1) < 1e-10
        assert selected_full(q, BASE) == row['top12']
        assert all(q[t] == p for t,p in zip(row['top12'], row['probabilities']))
        q_by_race[rid] = q
        full_decisions[rid] = {policy: selected_full(q, policy) for policy in FULL_POLICIES}
    decisions_output = [dict(cohort=kind, race_id=rid, date=forecast_inputs[rid]['date'], decisions=decisions)
                        for kind, collection in [('saved646', saved_decisions), ('full_q_oct19', full_decisions)]
                        for rid, decisions in sorted(collection.items())]
    decision_path = Path(str(output_prefix) + '_decisions_v1.jsonl')
    decision_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in decisions_output))
    records = {}
    for kind, collection in [('saved646', saved_decisions), ('full_q_oct19', full_decisions)]:
        records[kind] = []
        for rid, decisions in sorted(collection.items()):
            values = {policy: settle(tickets, outcomes[rid]) for policy, tickets in decisions.items()}
            if rid in audited:
                assert values[BASE]['payout_yen'] == audited[rid]['payout_yen'] and values[BASE]['hit'] == audited[rid]['hit']
            records[kind].append(dict(race_id=rid, date=forecast_inputs[rid]['date'], decisions=decisions, policies=values))
    saved_totals = totals(records['saved646'], BASE)
    previous_baseline = corrected['all']['flat_100']['primary']['initial']
    assert all(saved_totals[k] == previous_baseline[k] for k in ['races','hits','invested_yen','payout_yen'])
    report = dict(policy_id='recent-limited-ticket-exploration-v1', primary_metric='All eligible races; 100 yen per selected ticket; actual official payout / actual investment.',
        research_status='continued_exploration_all_outcomes_previously_seen', pristine_holdout=False,
        original_annual_27351_race_reproduction=False, production_adopted=False,
        candidate_count_excluding_baselines=5, candidate_definitions=DESCRIPTIONS,
        source_scope=dict(date_from=min(r['date'] for r in forecast_inputs.values()), date_to=max(r['date'] for r in forecast_inputs.values()),
                          model_sha256s=sorted({ledger[rid]['model_sha256'] for rid in clean_ids}),
                          training_cutoffs_exclusive=sorted({ledger[rid]['training_cutoff_exclusive'] for rid in clean_ids})),
        source_sha256={name:digest(raw) for name,raw in sources.items()},
        decisions_sha256=digest(decision_path.read_bytes()),
        correction=dict(race_id=correction_id, payouts=repaired['payouts'], source_ref=corrected['source_ref'], raw_blob_sha=corrected['correction_evidence']['raw_blob_sha']),
        eligibility=dict(stored_rows=len(ledger), saved_cohort=len(clean_ids), excluded_ids=sorted(set(ledger)-clean_ids),
                         full_q_available=len(reconstructed), full_q_eligible=len(full_decisions), full_q_common_excluded_ids=full_excluded,
                         full_q_cohort_reconciliation=dict(original_audit_complete_rows=sum(rid in audited for rid in full_decisions),
                             officially_repaired_rows=int(correction_id in full_decisions),
                             withdrawal_exclusion={'082720261019': 'Keiokaku 8R; car 3 withdrawn; original refund settlement pending; excluded for every candidate'}),
                         policy_specific_race_exclusions=False, reconstructed_full_q_was_published_preclose=False,
                         reconstructed_original_top12_and_probabilities_exactly_match=True),
        all_saved_races=comparisons(records['saved646'], SAVED_POLICIES),
        earlier_continued_exploration_through_oct16=comparisons([r for r in records['saved646'] if r['date'] <= '2026-10-16'], SAVED_POLICIES),
        later_continued_exploration_oct17_to_oct19=comparisons([r for r in records['saved646'] if r['date'] >= '2026-10-17'], SAVED_POLICIES),
        full_q_oct19_only=comparisons(records['full_q_oct19'], FULL_POLICIES),
        by_date={day: comparisons([r for r in records['saved646'] if r['date'] == day], SAVED_POLICIES)
                 for day in sorted({r['date'] for r in records['saved646']})},
        limitations=['Ten days of existing saved predictions; results were already viewed before this continued exploration.',
                     'Reconstructed full q covers one day only; source input was saved preclose and the original twelve tickets/probabilities match exactly, but the full vector itself was reconstructed after the races.',
                     'No odds are used or inferred; this evaluates realized flat-stake returns, not expected value or no-loss allocations.',
                     'All candidates use the same eligible identities within each cohort; no outcome or grade based race filtering.',
                     'Five new candidate policies in this bounded comparison; prior eight-ticket and allocation searches remain additional multiple-testing history.',
                     'A favorable retrospective result alone is insufficient to establish prospective profitability.'], records=records)
    assert all((work_root/name).read_bytes() == raw for name,raw in sources.items())
    report['source_files_unchanged'] = True
    result_path = Path(str(output_prefix) + '_results_v1.json')
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    table = []
    for scope in ['all_saved_races','earlier_continued_exploration_through_oct16','later_continued_exploration_oct17_to_oct19','full_q_oct19_only']:
        for policy, summary in report[scope].items():
            table.append(dict(scope=scope, policy=policy, **{k:v for k,v in summary.items() if not isinstance(v,dict)}))
    with Path(str(output_prefix) + '_comparison_v1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print(json.dumps(dict(eligibility=report['eligibility'], comparison=table), ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True,
                        help='Directory containing the frozen data/, replay/, research/ and analysis_rows_2026-10-20.json evidence.')
    parser.add_argument('--output-prefix', type=Path, required=True,
                        help='Explicit path prefix for _decisions_v1.jsonl, _results_v1.json and _comparison_v1.csv; never a live feed.')
    args = parser.parse_args()
    run(args.work_root, args.output_prefix)
