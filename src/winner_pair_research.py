"""Score fixed winner/pair decisions from immutable historical distributions.

This changes the decision target, not the fitted model or the trifecta tickets.
Prepare saves every decision before assess reads any per-race outcome.
"""
import argparse
import gzip
import hashlib
import html
import json
import math
import random
from collections import Counter, defaultdict
from itertools import permutations, zip_longest
from pathlib import Path

VERSION = 'winner_pair_marginals_v1'
NAMES = {'risk': 'リスク部', 'fusion': '従来の統合式',
         'refit_first_anchor': '独立予想の基になった改善式'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f'evidence hash mismatch: {Path(path).name}')


def choices(keys, probabilities):
    """All decisions depend only on a complete trifecta probability table."""
    triples = [tuple(map(int, k.split('-'))) for k in keys]
    cars = sorted({c for t in triples for c in t})
    if (not 3 <= len(cars) <= 9 or not all(1 <= c <= 9 for c in cars)
            or len(set(triples)) != len(triples)
            or set(triples) != set(permutations(cars, 3))):
        raise ValueError('complete distinct-car triples required')
    if (len(probabilities) != len(triples)
            or any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities)
            or not math.isclose(math.fsum(probabilities), 1., rel_tol=0, abs_tol=1e-8)):
        raise ValueError('normalized finite probabilities required')
    distribution = dict(zip(triples, probabilities))
    first_groups, pair_groups = defaultdict(list), defaultdict(list)
    for triple, probability in distribution.items():
        first_groups[triple[0]].append(probability)
        pair_groups[triple[:2]].append(probability)
    first = {a: math.fsum(v) for a, v in first_groups.items()}
    pair = {ab: math.fsum(v) for ab, v in pair_groups.items()}
    ranked = sorted(distribution, key=lambda t: (-distribution[t], t))
    winner = min(first, key=lambda a: (-first[a], a))
    ordered_pair = min(pair, key=lambda ab: (-pair[ab], ab))
    return {'triple_first': ranked[0][0], 'triple_pair': list(ranked[0][:2]),
            'marginal_first': winner, 'marginal_pair': list(ordered_pair),
            'first_probability': first[winner], 'pair_probability': pair[ordered_pair],
            'top12': ['-'.join(map(str, t)) for t in ranked[:12]]}


def prepare(evidence_root, output):
    original = evidence_root / 'outputs/fixed_year_20261009'
    repair = evidence_root / 'outputs/fusion_input_repair_20261009/refit'
    paths = [original / 'holdout_predictions.jsonl.gz', repair / 'predictions.jsonl.gz']
    manifests = [read(original / 'prediction_manifest.json'), read(repair / 'prediction_manifest.json')]
    if manifests[1]['source_predictions_sha256'] != manifests[0]['predictions_sha256']:
        raise ValueError('repair and original have different source forecasts')
    for p, m in zip(paths, manifests):
        verified(p, m['predictions_sha256'])
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'decisions.jsonl.gz'
    seen = set()
    with gzip.open(paths[0], 'rt', encoding='utf-8') as a, gzip.open(paths[1], 'rt', encoding='utf-8') as b, gzip.open(target, 'wt', encoding='utf-8') as out:
        for left, right in zip_longest(a, b):
            if left is None or right is None:
                raise ValueError('prediction source lengths differ')
            base, refit = json.loads(left), json.loads(right)
            if any(base[k] != refit[k] for k in ('race_id', 'date', 'field', 'race_class', 'keys')):
                raise ValueError('prediction cohorts differ')
            if base['race_id'] in seen:
                raise ValueError('duplicate race')
            seen.add(base['race_id'])
            row = {k: base[k] for k in ('race_id', 'date', 'field', 'race_class')}
            row['choices'] = {name: choices(base['keys'], base['distributions'][name]) for name in ('risk', 'fusion')}
            row['choices']['refit_first_anchor'] = choices(refit['keys'], refit['distributions']['first_anchor'])
            out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
            if len(seen) % 2000 == 0:
                print(f'WINNER_PAIR_DECISIONS {len(seen)}', flush=True)
    if any(len(seen) != m['predicted_races'] for m in manifests):
        raise ValueError('incomplete forecast cohort')
    manifest = {'version': VERSION, 'predicted_races': len(seen), 'labels_read': False,
                'model_refitted': False, 'sources': {str(p): m['predictions_sha256'] for p, m in zip(paths, manifests)},
                'decisions_sha256': digest(target), 'source_code_sha256': digest(__file__),
                'selection': 'one winner and one ordered pair; numeric lexicographic tie break',
                'retrospective_reused_year': True, 'production_changed': False}
    write(output / 'manifest.json', manifest)
    return manifest


def day_interval(daily, resamples=4000):
    """Paired day bootstrap; descriptive, not a promotion threshold."""
    rows = list(daily.values())
    rng = random.Random(20261010)
    draws = []
    for _ in range(resamples):
        sampled = rng.choices(rows, k=len(rows))
        draws.append(sum(d for d, n in sampled) / sum(n for d, n in sampled))
    draws.sort()
    return [draws[int(.025 * resamples)], draws[int(.975 * resamples)]]


def assess(evidence_root, output):
    manifest = read(output / 'manifest.json')
    verified(__file__, manifest['source_code_sha256'])
    verified(output / 'decisions.jsonl.gz', manifest['decisions_sha256'])
    repair = evidence_root / 'outputs/fusion_input_repair_20261009'
    historical = read(repair / 'results.json')
    truth_path = repair / 'assessed_races.jsonl.gz'
    verified(truth_path, historical['assessed_races_sha256'])
    totals = {n: Counter() for n in NAMES}
    daily = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    groups = defaultdict(lambda: {n: Counter() for n in NAMES})
    seen, excluded = set(), Counter()
    dates = []
    with gzip.open(truth_path, 'rt', encoding='utf-8') as outcomes, gzip.open(output / 'decisions.jsonl.gz', 'rt', encoding='utf-8') as decisions:
        for decision, result in zip_longest(decisions, outcomes):
            if decision is None or result is None:
                raise ValueError('outcome and prediction lengths differ')
            row, truth = json.loads(decision), json.loads(result)
            if row['race_id'] != truth['race_id']:
                raise ValueError('outcome and forecast identity mismatch')
            if row['race_id'] in seen:
                raise ValueError('duplicate assessment race')
            seen.add(row['race_id'])
            if truth.get('exclusion'):
                excluded[truth['exclusion']] += 1
                continue
            if any(row[k] != truth[k] for k in ('date', 'field', 'race_class')):
                raise ValueError('outcome and forecast metadata mismatch')
            actual = tuple(map(int, truth['actual'].split('-')))
            if len(actual) != 3 or len(set(actual)) != 3:
                raise ValueError('invalid outcome')
            dates.append(row['date'])
            for name, picked in row['choices'].items():
                counts = {'races': 1, 'top12_hits': int(truth['actual'] in picked['top12'])}
                for kind, correct in (('first', actual[0]), ('pair', list(actual[:2]))):
                    old = int(picked[f'triple_{kind}'] == correct)
                    new = int(picked[f'marginal_{kind}'] == correct)
                    counts.update({f'old_{kind}_hits': old, f'new_{kind}_hits': new,
                                   f'{kind}_rescued': int(new and not old), f'{kind}_lost': int(old and not new),
                                   f'{kind}_picks_changed': int(picked[f'triple_{kind}'] != picked[f'marginal_{kind}'])})
                    daily[name][kind][row['date']][0] += new - old
                    daily[name][kind][row['date']][1] += 1
                # Independent reproduction against the previous per-race audit.
                old_check = truth['methods'][name]
                if (counts['top12_hits'] != old_check['top12_hit']
                        or counts['new_first_hits'] != old_check['marginal_first_hit']):
                    raise ValueError('historical per-race audit cannot be reproduced')
                totals[name].update(counts)
                for key in ('month:' + row['date'][:7], 'field:' + str(row['field']), 'class:' + row['race_class']):
                    groups[key][name].update(counts)
    expected = historical['evaluated_races']
    if len(seen) != manifest['predicted_races'] or dict(excluded) != historical['outcome_exclusions']:
        raise ValueError('historical evaluation cohort differs')
    for name, t in totals.items():
        old = historical['summary']['all']['methods'][name]
        if t['races'] != expected:
            raise ValueError('incomplete method coverage')
        for count, metric in (('old_first_hits', 'first_hit'), ('old_pair_hits', 'first_pair_hit'), ('top12_hits', 'top12_hit')):
            if t[count] != round(old[metric] * expected):
                raise ValueError(f'baseline counts differ: {name} {metric}')
    def rates(counts):
        return {**counts, **{k.removesuffix('_hits') + '_rate': v / counts['races'] for k, v in counts.items() if k.endswith('_hits')}}
    result = {'version': VERSION, 'status': 'retrospective_decision_rule_comparison',
              'first': min(dates), 'last': max(dates), 'days': len(set(dates)),
              'predicted_races': len(seen), 'evaluated_races': expected, 'outcome_exclusions': dict(excluded),
              'methods': {n: rates(t) for n, t in totals.items()},
              'comparisons': {n: {kind: {'extra_hits': totals[n][f'new_{kind}_hits'] - totals[n][f'old_{kind}_hits'],
                                           'difference': (totals[n][f'new_{kind}_hits'] - totals[n][f'old_{kind}_hits']) / expected,
                                           'paired_day_bootstrap_95_interval': day_interval(daily[n][kind])}
                                    for kind in ('first', 'pair')} for n in NAMES},
              'groups': {g: {n: rates(t) for n, t in ns.items()} for g, ns in groups.items()},
              'baseline_counts_reproduced': True, 'prior_per_race_audit_reproduced': True,
              'truth_sha256': historical['assessed_races_sha256'], 'decision_manifest': manifest,
              'roi': None, 'new_three_rider_model_tested': False, 'new_winning_method_model_tested': False,
              'limitations': ['既存の凍結済み確率から選び方だけを変更。新しい相互作用・決まり手モデルの成績ではない。',
                             '既に何度も検討した1年の探索比較。将来の未使用期間での実績ではない。',
                             '全13,742レースに予想を作成。元検証と同じ13,612レースを採点し、同着等の130レースを同じ理由で除外。',
                             '1着は1人、1〜2着は順序を含む1組。それぞれ別の目的で最適化するため1着候補が異なることがある。',
                             '3連単12点は一切変更していない。1着・1〜2着の改善を3連単的中率の改善とは扱わない。',
                             '同時点のオッズ・払戻がないため回収率は算出しない。区間は日単位再抽出の参考値。']}
    write(output / 'results.json', result)
    render(result, output / 'report.html')
    print(json.dumps({k: result[k] for k in ('evaluated_races', 'methods', 'comparisons')}, ensure_ascii=False), flush=True)
    return result


def render(result, path):
    rows = []
    for name, m in result['methods'].items():
        for kind, title in (('first', '1着・1人'), ('pair', '1〜2着・順序通り1組')):
            d = result['comparisons'][name][kind]
            lo, hi = d['paired_day_bootstrap_95_interval']
            rows.append(f'<tr><td>{NAMES[name]}</td><td>{title}</td><td>{m[f"old_{kind}_rate"]:.2%}<small>{m[f"old_{kind}_hits"]:,}的中</small></td>'
                        f'<td><b>{m[f"new_{kind}_rate"]:.2%}</b><small>{m[f"new_{kind}_hits"]:,}的中</small></td>'
                        f'<td>{d["extra_hits"]:+,}件 / {100*d["difference"]:+.2f}ポイント<small>95%区間 {lo*100:+.2f}〜{hi*100:+.2f}ポイント</small></td></tr>')
    notes = ''.join(f'<li>{html.escape(note)}</li>' for note in result['limitations'])
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>1着・1〜2着の予想検証</title>
<style>body{{font-family:system-ui,sans-serif;color:#182537;background:#f3f6fb;margin:0;padding:32px}}main{{max-width:1050px;margin:auto}}h1{{font-size:28px}}p,li{{line-height:1.8}}.table{{overflow:auto;background:white;border-radius:12px}}table{{border-collapse:collapse;width:100%;min-width:820px}}th,td{{padding:16px;text-align:left;border-bottom:1px solid #dbe3ef}}th{{background:#163455;color:white}}small{{display:block;color:#53657b;font-size:12px;margin-top:5px}}b{{color:#11654f}}li{{margin:8px 0}}</style>
<main><h1>1着・1〜2着の予想検証</h1><p>{result['first']}〜{result['last']} / 同じ {result['evaluated_races']:,} レース<br>旧方式：最上位3連単の先頭を採用。新方式：残りの着順の確率を合計して選択。</p>
<div class="table"><table><thead><tr><th>モデル</th><th>的中条件</th><th>旧方式</th><th>確率を合計</th><th>差</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><ul>{notes}</ul></main></html>'''
    Path(path).write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'assess'), required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.evidence_root, args.output)
    else:
        assess(args.evidence_root, args.output)


if __name__ == '__main__':
    main()
