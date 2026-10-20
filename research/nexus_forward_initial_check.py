"""Recompute the fixed 10-point candidate on the seven late-received results.

This is a retrospective audit. It does not publish predictions, fetch data,
train models, choose a new policy, or place bets. All input paths are explicit.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

from nexus_recent_roi import outcome, totals
from nexus_roi_research import select_tickets

IDS = ('052320261020', '062320261020', '063620261020', '072320261020',
       '073620261020', '082320261020', '083620261020')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def stamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('timezone-aware timestamps are required')
    return parsed


def expanded(blocks):
    if not 1 <= len(blocks) <= 4:
        raise ValueError('one to four complete formation blocks required')
    values = set()
    for block in blocks:
        groups = [block[k] for k in ('first', 'second', 'third')]
        if any(not g or len(g) != len(set(g)) for g in groups):
            raise ValueError('invalid formation group')
        part = {t for t in itertools.product(*groups) if len(set(t)) == 3}
        if block.get('count', len(part)) != len(part):
            raise ValueError('formation count mismatch')
        values.update('-'.join(map(str, t)) for t in part)
    return values


def input_hash(snapshot):
    data, meta = snapshot['data'], snapshot['meta']
    model_data = {k: data[k] for k in ('players', 'entries', 'records', 'linePrediction') if k in data}
    model_data['race'] = {k: data['race'].get(k) for k in ('id', 'distance', 'startAt', 'closeAt')}
    model_data['schedule'] = {k: data['schedule'].get(k) for k in ('date', 'cupId', 'index', 'day')}
    model_meta = {k: v for k, v in meta.items() if k not in ('source_url', 'session')}
    return digest([model_meta, model_data])


def same_mapping(actual, expected):
    actual = {str(k): v for k, v in actual.items()}
    expected = {str(k): v for k, v in expected.items()}
    return (set(actual) == set(expected)
            and all(abs(actual[k] - expected[k]) <= 1e-12 for k in expected))


def rebuild_q(row, inputs_dir, engine, history):
    path = inputs_dir / (row['race_id'] + '.json')
    if not path.is_file():
        path = inputs_dir / ('input_' + row['race_id'] + '.json')
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    original = row['original_prediction']
    if (input_hash(snapshot) != snapshot['input_sha256']
            or snapshot['input_sha256'] != original['input_sha256']
            or snapshot['sha256'] != row['snapshot']['source_sha256']
            or stamp(snapshot['captured_at']) >= stamp(row['close_at'])
            or snapshot['data'].get('results') or snapshot['data']['race'].get('decidedAt')):
        raise ValueError('saved preclose input does not match ' + row['race_id'])
    from nexus_model import inputs
    clean = copy.deepcopy(snapshot['data'])
    for key in ('results', 'trifecta', 'finalOdds', 'trifectaWinningOddsIds', 'trifectaPayoffStatus'):
        clean.pop(key, None)
    matrix, _ = inputs.build(history, clean, snapshot['meta'])
    predicted = engine.predict(matrix)
    cars = [int(c) for c in matrix['rows'][0]['cars']]
    q = [float(v) for v in predicted['q']]
    if (cars != row['cars'] or len(q) != len(row['q'])
            or any(abs(a - b) > 1e-12 for a, b in zip(q, row['q']))
            or sorted(predicted['tickets']) != sorted(original['tickets'])
            or {str(k): v for k, v in predicted['marks'].items()} != original['marks']
            or predicted['count'] != original['count']
            or predicted['category'] != original['category']
            or predicted['policy_id'] != original['policy_id']
            or not same_mapping(predicted['first_probabilities'], original['first_probabilities'])
            or not same_mapping(predicted['band_probabilities'], original['band_probabilities'])
            or abs(predicted['confidence']['model_mass'] - original['confidence']['model_mass']) > 1e-12):
        raise ValueError('original model replay differs for ' + row['race_id'])


def run(evidence_path, prior_path, inputs_dir=None, model_dir=None, history_path=None):
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    records = evidence['records']
    if (len(records) != 7 or {r['race_id'] for r in records} != set(IDS)
            or digest(records) != evidence['records_sha256']
            or evidence['candidate']['points'] != 10
            or evidence['candidate']['max_formations'] != 4
            or evidence.get('prospective_preclose_candidate') is not False):
        raise ValueError('the fixed evidence contract does not match')
    prior_bytes = prior_path.read_bytes()
    if hashlib.sha256(prior_bytes).hexdigest() != evidence['prior32_report']['sha256']:
        raise ValueError('the original 32-race report differs')
    prior = json.loads(prior_bytes)
    engine, history = None, None
    if any(p is not None for p in (inputs_dir, model_dir, history_path)):
        if not all(p is not None for p in (inputs_dir, model_dir, history_path)):
            raise ValueError('model replay requires --inputs-dir, --model-dir and --history')
        from nexus_model.engine import Engine
        from nexus_model.inputs import History
        if any(hashlib.sha256((model_dir / 'manifest.json').read_bytes()).hexdigest()
               != r['replay_audit']['model_manifest_sha256'] for r in records):
            raise ValueError('frozen model manifest hash mismatch')
        if any(hashlib.sha256(history_path.read_bytes()).hexdigest()
               != r['replay_audit']['history_sha256'] for r in records):
            raise ValueError('frozen history hash mismatch')
        engine, history = Engine(model_dir), History.load(history_path)
    decisions = []
    for row in records:
        cars, q = row['cars'], row['q']
        original = row['original_prediction']
        joint = {k: row[k] for k in ('cars', 'q', 'q_order')}
        if (len(cars) != len(set(cars)) or any(type(c) is not int or not 1 <= c <= 9 for c in cars)
                or len(q) != math.perm(len(cars), 3)
                or any(isinstance(v, bool) or not isinstance(v, (float, int))
                       or not math.isfinite(v) or not 0 <= v <= 1 for v in q)
                or abs(math.fsum(q) - 1) > 1e-8
                or row['q_order'] != 'itertools.permutations(cars,3)'
                or digest(joint) != row['joint_distribution_sha256']):
            raise ValueError('full probability evidence differs')
        if (not stamp(original['input_captured_at']) <= stamp(original['captured_at']) < stamp(row['close_at'])
                or stamp(row['snapshot']['captured_at']) >= stamp(row['close_at'])
                or original['history_through'] >= row['date']
                or not row['replay_audit']['exact_replay_match']
                or not all(row['replay_audit']['match_flags'].values())):
            raise ValueError('preclose/equality evidence is invalid')
        marginals = dict.fromkeys(cars, 0.0)
        for triple, probability in zip(itertools.permutations(cars, 3), q):
            marginals[triple[0]] += probability
        if not same_mapping(marginals, original['first_probabilities']):
            raise ValueError('probability car ordering differs')
        if engine is not None:
            rebuild_q(row, inputs_dir, engine, history)
        tickets, blocks = select_tickets(cars, q, 10)
        tickets = ['-'.join(map(str, t)) for t in tickets]
        if (len(tickets) != 10 or len(set(tickets)) != 10
                or expanded(blocks) != set(tickets)
                or sorted(tickets) != sorted(row['candidate_plan']['tickets'])
                or canonical(blocks) != canonical(row['candidate_plan']['blocks'])
                or original['count'] != len(original['tickets'])
                or expanded(original['blocks']) != set(original['tickets'])):
            raise ValueError('saved formation selection differs')
        decisions.append((row, tickets))
    # Selection above receives cars/q only. Now read official settlement values.
    comparisons = []
    for row, tickets in decisions:
        if (row.get('refund_all') or row.get('refunded') or row.get('cancelled')
                or not row['payouts'] or set(row['payouts']) != set(row['winners'])
                or any(type(v) is not int or v <= 0 for v in row['payouts'].values())):
            raise ValueError('invalid official settlement')
        baseline = outcome(row, row['original_prediction']['tickets'])
        candidate = outcome(row, tickets)
        if baseline != row['baseline'] or candidate != row['candidate']:
            raise ValueError('saved settlement differs')
        comparisons.append(dict(race_id=row['race_id'], baseline=baseline, candidate=candidate))
    current = dict(baseline=totals([r['baseline'] for r in comparisons]),
                   candidate_totals=totals([r['candidate'] for r in comparisons]))
    if any(current[k] != evidence[k] for k in current):
        raise ValueError('saved seven-race aggregate differs')
    combined = {}
    for name, old_key, new_key in (('baseline', 'published', 'baseline'),
                                    ('candidate', 'formation_10', 'candidate_totals')):
        old, new = prior['summary'][old_key], current[new_key]
        n, hits = old['races'] + new['races'], old['hits'] + new['hits']
        stake, paid = old['stake_yen'] + new['stake_yen'], old['payout_yen'] + new['payout_yen']
        combined[name] = dict(races=n, hits=hits, hit_rate=hits/n, stake_yen=stake,
                              payout_yen=paid, profit_yen=paid-stake, roi=paid/stake)
    if combined != evidence['combined_previous32_plus_new7']:
        raise ValueError('saved combined 39-race aggregate differs')
    return dict(protocol_id=evidence['protocol_id'], verified_at=datetime.now(timezone.utc).isoformat(),
                source_ref=evidence['source_ref'], evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                all_seven_selections_and_settlements_match=True, model_q_rebuilt=engine is not None,
                prospective_preclose_candidate=False, untouched_holdout=False, annual_target_achieved=False,
                **current, combined_previous32_plus_new7=combined, records=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--prior-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs-dir', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--history', type=Path)
    args = parser.parse_args()
    report = run(args.evidence, args.prior_report, args.inputs_dir, args.model_dir, args.history)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, ensure_ascii=False, indent=2))
