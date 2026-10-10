"""Past-only selection of nonlinear conditional trifecta models.

P(i,j,k)=P(i)*P(j|i)*P(k|i,j). Training uses observed prefixes; prediction
enumerates every hypothetical prefix and never reads the realized result.
This research CLI cannot publish or replace a production model.
"""
import argparse
import gzip
import json
import math
import platform
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import permutations
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import fixed_year_features as old_features
from fixed_year_study import eligible, outcome
from winner_pair_research import digest, read, verified, write

VERSION = 'conditional_trifecta_past_only_v1'
CONFIG = {'train_cutoff': '2025-10-09', 'development_start': '2025-04-09',
          'target_end': '2026-10-09', 'fit_years': 3, 'half_life_days': 365.,
          'leaves': [31, 63], 'chain_weights': [0., .5, 1.], 'trees': 400,
          'learning_rate': .035, 'min_child_samples': 100, 'reg_lambda': 3.,
          'threads': 2, 'tickets': 12, 'stake_per_ticket_yen': 100,
          'already_inspected_target_year': True, 'automatic_promotion': False}
CONTEXT = ['score', 'win_rate', 'place2_rate', 'place3_rate', 'back_count',
           'front_runner_count', 'stalker_count', 'deep_closer_count', 'marker_count',
           'recent_avg_finish', 'prediction_mark', 'age', 'term', 'style_code',
           'region_id', 'prefecture_code', 'car_no', 'gear_ratio',
           'player_prior_win_rate', 'player_prior_place2_rate', 'player_prior_place3_rate',
           'score_rank', 'score_gap_to_best', 'rider_strength']


def source_hashes():
    return {n: digest(Path(__file__).with_name(n)) for n in
            ('trifecta_chain_research.py', 'fixed_year_features.py', 'fixed_year_study.py',
             'common.py', 'winner_pair_research.py')}


def frame_order(frame):
    return frame.sort_values(['date', 'race_id', 'car_no'], kind='stable').reset_index(drop=True)


def feature_matrix(frame, feature_config=None):
    """Only the reviewed input schema is admitted; result/identity keys are absent."""
    X, config = old_features.matrix(old_features.mask_outcomes(frame), feature_config)
    dropped = {'player_id_code', 'player_prior_days_since_last_race', 'wind_speed', 'weather_code'}
    X = X[[c for c in X if c not in dropped and not c.startswith('weather_')]].copy()
    for c in ('score', 'win_rate', 'place2_rate', 'place3_rate', 'back_count',
              'front_runner_count', 'stalker_count', 'deep_closer_count', 'marker_count',
              'recent_avg_finish', 'age', 'prediction_mark'):
        group = X[c].groupby(frame.race_id.to_numpy(), sort=False)
        X[c+'_field_delta'] = X[c] - group.transform('mean')
        X[c+'_field_rank'] = group.rank(method='average', pct=True)
    return X.astype(np.float32), config


def enrich_training(frame, cutoff):
    if frame.empty or not frame.date.lt(cutoff).all():
        raise ValueError('training reaches prediction cutoff')
    frame = frame_order(frame)
    enriched, reference = old_features.daily_priors(frame)
    if not enriched.race_id.equals(frame.race_id) or not enriched.player_id.equals(frame.player_id):
        raise ValueError('training feature alignment changed')
    return frame, enriched, reference


def design(X, target, first=None, second=None):
    """X contains covariates only. Prefix indices are hypotheses, not result fields."""
    target = np.asarray(target, dtype=int)
    values = X.to_numpy(copy=False)
    context = values[:, [X.columns.get_loc(c) for c in CONTEXT]]
    blocks = [values[target]]
    for prefix in (first, second):
        if prefix is None:
            continue
        prefix = np.asarray(prefix, dtype=int)
        if len(prefix) != len(target) or np.any(prefix == target):
            raise ValueError('prefix must be distinct from candidate')
        blocks.extend((context[prefix], context[target] - context[prefix]))
        # Shared attributes are not claims of verified physical line membership.
        same = []
        for col in ('region_id', 'prefecture_code', 'style_code'):
            k = X.columns.get_loc(col)
            same.append((values[target, k] == values[prefix, k]).astype(np.float32))
        blocks.append(np.column_stack(same))
    if second is not None:
        if first is None or np.any(np.asarray(first) == np.asarray(second)):
            raise ValueError('ordered prefix must have distinct first and second')
        blocks.extend((context[first]-context[second], context[first]*context[second]))
    return np.column_stack(blocks).astype(np.float32, copy=False)


def training_indices(frame, position):
    """One candidate per remaining rider, conditional on the observed prefix."""
    targets, first, second, labels = [], [], [], []
    for _, race in frame.groupby('race_id', sort=False):
        finish = pd.to_numeric(race.finish_pos, errors='coerce')
        if any(int(finish.eq(p).sum()) != 1 for p in (1, 2, 3)):
            raise ValueError('ambiguous training podium')
        ordered = [int(race.index[finish.eq(p)][0]) for p in (1, 2, 3)]
        prefix = ordered[:position-1]
        for idx in race.index:
            if idx in prefix:
                continue
            targets.append(int(idx)); labels.append(int(idx == ordered[position-1]))
            if position > 1: first.append(ordered[0])
            if position > 2: second.append(ordered[1])
    return (np.array(targets), np.array(first) if first else None,
            np.array(second) if second else None, np.array(labels))


def fit(frame, cutoff, leaves, output, fast=False):
    clean, excluded = eligible(frame, labels=True)
    clean, enriched, reference = enrich_training(clean, cutoff)
    earliest = str((pd.Timestamp(cutoff)-pd.DateOffset(years=CONFIG['fit_years'])).date())
    fit_mask = clean.date.ge(earliest).to_numpy()
    clean = clean.loc[fit_mask].reset_index(drop=True)
    enriched = enriched.loc[fit_mask].reset_index(drop=True)
    X, feature_config = feature_matrix(enriched)
    dates = pd.to_datetime(clean.date)
    recency = np.exp(-np.log(2)*(pd.Timestamp(cutoff)-dates).dt.days.to_numpy()/CONFIG['half_life_days'])
    field = clean.groupby('race_id').race_id.transform('size').to_numpy()
    models = {}
    for name, position, conditional in [('first', 1, False), ('second', 2, False),
                                        ('third', 3, False), ('given_first', 2, True),
                                        ('given_pair', 3, True)]:
        if conditional:
            target, first, second, y = training_indices(clean, position)
            data = design(X, target, first, second)
            weight = recency[target]*7/(field[target]-position+1)
        else:
            y = clean.finish_pos.eq(position).to_numpy(dtype=int)
            data = X.to_numpy(copy=False)
            weight = recency*7/field
        model = lgb.LGBMClassifier(n_estimators=8 if fast else CONFIG['trees'],
            learning_rate=CONFIG['learning_rate'], num_leaves=leaves,
            min_child_samples=CONFIG['min_child_samples'], reg_lambda=CONFIG['reg_lambda'],
            n_jobs=CONFIG['threads'], random_state=401+position, verbosity=-1,
            deterministic=True, force_col_wise=True)
        model.fit(data, y, sample_weight=weight)
        models[name] = model
        print(f'CHAIN_FIT {cutoff} leaves={leaves} {name} rows={len(y)}', flush=True)
        del data
    bundle = {'models': models, 'reference': reference, 'feature_config': feature_config,
              'features': list(X), 'cutoff': cutoff, 'leaves': leaves,
              'fit_first': str(clean.date.min()), 'fit_last': str(clean.date.max()),
              'fit_races': int(clean.race_id.nunique()), 'training_exclusions': excluded,
              'source_hashes': source_hashes()}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output/'model.joblib', compress=3)
    return bundle


@lru_cache(maxsize=7)
def hypotheses(n):
    return np.asarray(list(permutations(range(n), 2))), np.asarray(list(permutations(range(n), 3)))


def normalized(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-12)
    return p/p.sum()


def combine(first, second, third, pair_predictions, triple_predictions, pairs, triples):
    """Complete normalized joint tables; each prefix has its own denominator."""
    n = len(first)
    p1, p2, p3 = map(normalized, (first, second, third))
    pair = np.zeros((n, n))
    pair[pairs[:, 0], pairs[:, 1]] = np.maximum(pair_predictions, 1e-12)
    pair /= pair.sum(axis=1)[:, None]
    t = np.zeros((n, n, n))
    t[triples[:, 0], triples[:, 1], triples[:, 2]] = np.maximum(triple_predictions, 1e-12)
    sums = t.sum(axis=2)
    for i, j in pairs:
        t[i, j] /= sums[i, j]
    i, j, k = triples.T
    independent = p1[i]*(p2[j]/(1-p2[i]))*(p3[k]/(1-p3[i]-p3[j]))
    chain = p1[i]*pair[i, j]*t[i, j, k]
    return normalized(independent), normalized(chain)


def predict(frame, bundle, batch_size=128):
    clean = old_features.mask_outcomes(frame)
    if clean.empty or not clean.date.ge(bundle['cutoff']).all():
        raise ValueError('prediction dates precede training cutoff')
    clean, excluded = eligible(clean)
    if excluded:
        raise ValueError(f'input exclusions require explicit cohort audit: {excluded}')
    clean = frame_order(clean)
    enriched = old_features.frozen_priors(clean, bundle['reference'])
    X, _ = feature_matrix(enriched, bundle['feature_config'])
    if list(X) != bundle['features']:
        raise ValueError('feature schema changed')
    base = {n: bundle['models'][n].booster_.predict(X.to_numpy(copy=False), num_threads=CONFIG['threads'])
            for n in ('first', 'second', 'third')}
    groups = list(clean.groupby('race_id', sort=False))
    for start in range(0, len(groups), batch_size):
        batch = groups[start:start+batch_size]
        pidx, tidx, lengths = [], [], []
        for rid, race in batch:
            pair, triple = hypotheses(len(race))
            pidx.append(race.index.to_numpy()[pair]); tidx.append(race.index.to_numpy()[triple])
            lengths.append((len(pair), len(triple)))
        pidx, tidx = np.concatenate(pidx), np.concatenate(tidx)
        p2 = bundle['models']['given_first'].booster_.predict(design(X, pidx[:, 1], pidx[:, 0]), num_threads=CONFIG['threads'])
        p3 = bundle['models']['given_pair'].booster_.predict(design(X, tidx[:, 2], tidx[:, 0], tidx[:, 1]), num_threads=CONFIG['threads'])
        poff, toff = 0, 0
        for (rid, race), (plen, tlen) in zip(batch, lengths):
            pair, triple = hypotheses(len(race))
            independent, chain = combine(*(base[n][race.index] for n in ('first', 'second', 'third')),
                p2[poff:poff+plen], p3[toff:toff+tlen], pair, triple)
            poff += plen; toff += tlen
            cars = race.car_no.astype(int).to_numpy()
            keys = ['-'.join(map(str, cars[t])) for t in triple]
            yield {'race_id': str(rid), 'date': str(race.date.iloc[0]), 'field': len(race),
                   'race_class': str(race.race_class.iloc[0]), 'keys': keys,
                   'independent': independent.tolist(), 'chain': chain.tolist()}
        if (start//batch_size) % 10 == 0:
            print(f'CHAIN_PREDICT {min(start+batch_size, len(groups))}/{len(groups)}', flush=True)


def tickets(keys, probabilities):
    if not math.isclose(math.fsum(probabilities), 1., abs_tol=1e-8) or not np.isfinite(probabilities).all():
        raise ValueError('invalid joint probabilities')
    ordered = sorted(range(len(keys)), key=lambda i: (-probabilities[i], tuple(map(int, keys[i].split('-')))))
    chosen = [keys[i] for i in ordered[:CONFIG['tickets']]]
    if len(chosen) != CONFIG['tickets'] or len(set(chosen)) != CONFIG['tickets']:
        raise ValueError('exactly 12 distinct tickets required')
    return chosen


def train(history, output, raw_sha):
    if (output/'frozen_manifest.json').exists():
        raise ValueError('refusing to overwrite frozen research')
    output.mkdir(parents=True, exist_ok=True)
    write(output/'declaration.json', {'version': VERSION, 'config': CONFIG, 'history_sha256': raw_sha,
        'source_hashes': source_hashes(), 'target_year_used_for_selection': False,
        'note': 'Target year was inspected by earlier studies; no fresh validation claim.'})
    older = history[history.date.lt(CONFIG['development_start'])]
    development = history[history.date.ge(CONFIG['development_start']) & history.date.lt(CONFIG['train_cutoff'])]
    valid, exclusions = eligible(development, labels=True)
    answers = {str(rid): outcome(g) for rid, g in valid.groupby('race_id', sort=False)}
    trials = []
    for leaves in CONFIG['leaves']:
        bundle = fit(older, CONFIG['development_start'], leaves, output/f'development_{leaves}')
        counts = Counter()
        for row in predict(development, bundle):
            if row['race_id'] not in answers:
                continue
            for weight in CONFIG['chain_weights']:
                p = (1-weight)*np.asarray(row['independent'])+weight*np.asarray(row['chain'])
                counts[str(weight)] += int(answers[row['race_id']] in tickets(row['keys'], p))
        for weight in CONFIG['chain_weights']:
            trials.append({'leaves': leaves, 'chain_weight': weight, 'hits': counts[str(weight)],
                           'races': len(answers), 'rate': counts[str(weight)]/len(answers)})
        write(output/'development.json', {'trials': trials, 'exclusions': exclusions})
        print('CHAIN_DEVELOPMENT '+json.dumps(trials[-len(CONFIG['chain_weights']):]), flush=True)
    selected = max(trials, key=lambda x: (x['rate'], -x['leaves'], -x['chain_weight']))
    write(output/'selected.json', selected)
    final = fit(history[history.date.lt(CONFIG['train_cutoff'])], CONFIG['train_cutoff'], selected['leaves'], output/'final')
    manifest = {'version': VERSION, 'config': CONFIG, 'selected': selected, 'trials': trials,
        'history_sha256': raw_sha, 'source_hashes': source_hashes(),
        'model_sha256': digest(output/'final/model.joblib'),
        'declaration_sha256': digest(output/'declaration.json'), 'selected_sha256': digest(output/'selected.json'),
        'fit_first': final['fit_first'], 'fit_last': final['fit_last'], 'fit_races': final['fit_races'],
        'development_first': str(development.date.min()), 'development_last': str(development.date.max()),
        'target_year_used_for_selection': False, 'target_year_previously_inspected': True,
        'runtime': {'python': platform.python_version(), 'lightgbm': lgb.__version__, 'numpy': np.__version__},
        'production_changed': False}
    write(output/'frozen_manifest.json', manifest)
    return manifest


def frozen(output, raw_sha):
    m = read(output/'frozen_manifest.json')
    if m['source_hashes'] != source_hashes() or m['history_sha256'] != raw_sha:
        raise ValueError('frozen code/archive differs')
    verified(output/'final/model.joblib', m['model_sha256'])
    verified(output/'declaration.json', m['declaration_sha256'])
    verified(output/'selected.json', m['selected_sha256'])
    return m


def forecast(history, output, raw_sha):
    m = frozen(output, raw_sha)
    if (output/'prediction_manifest.json').exists():
        raise ValueError('predictions already frozen')
    bundle = joblib.load(output/'final/model.joblib')
    target = history[history.date.ge(CONFIG['train_cutoff']) & history.date.lt(CONFIG['target_end'])]
    selected_weight = m['selected']['chain_weight']
    path, seen = output/'predictions.jsonl.gz', set()
    with gzip.open(path, 'wt', encoding='utf-8') as out:
        for row in predict(target, bundle):
            if row['race_id'] in seen: raise ValueError('duplicate prediction')
            seen.add(row['race_id'])
            selected = (1-selected_weight)*np.array(row['independent'])+selected_weight*np.array(row['chain'])
            row['tickets'] = {n: tickets(row['keys'], p) for n, p in
                             [('selected', selected), ('independent', row['independent']), ('chain', row['chain'])]}
            out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    if len(seen) != target.race_id.nunique(): raise ValueError('incomplete cohort')
    write(output/'prediction_manifest.json', {'predicted_races': len(seen),
        'model_manifest_sha256': digest(output/'frozen_manifest.json'), 'predictions_sha256': digest(path),
        'outcomes_accessed_by_predictor': False, 'target_year_previously_inspected': True})


def assess(history, output, raw_sha, evidence_root):
    m = frozen(output, raw_sha)
    pm = read(output/'prediction_manifest.json')
    verified(output/'frozen_manifest.json', pm['model_manifest_sha256'])
    verified(output/'predictions.jsonl.gz', pm['predictions_sha256'])
    previous = read(evidence_root/'outputs/trifecta_consistency_20261010/results.json')
    path = evidence_root/'outputs/trifecta_consistency_20261010/predictions.jsonl.gz'
    verified(path, previous['prediction_manifest']['predictions_sha256'])
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        old = {r['race_id']: r for r in map(json.loads, f)}
    target = history[history.race_id.isin(old)]
    answers = {str(rid): outcome(g) for rid, g in target.groupby('race_id', sort=False)}
    totals, groups, excluded, seen = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter)), Counter(), set()
    with gzip.open(output/'predictions.jsonl.gz', 'rt', encoding='utf-8') as f:
        for row in map(json.loads, f):
            rid = row['race_id']
            if rid in seen or rid not in old: raise ValueError('different cohort')
            seen.add(rid)
            if any(row[k] != old[rid][k] for k in ('date', 'field', 'race_class')):
                raise ValueError('metadata differs')
            actual = answers.get(rid)
            if actual is None:
                excluded['ambiguous_or_missing_podium'] += 1
                continue
            chosen = {**row['tickets'], 'previous_best': old[rid]['tickets']['uncertainty']}
            baseline = actual in chosen['previous_best']
            for n, bets in chosen.items():
                hit = actual in bets
                v = {'races': 1, 'hits': int(hit), 'rescued': int(hit and not baseline), 'lost': int(baseline and not hit)}
                totals[n].update(v)
                for group in ('month:'+row['date'][:7], 'field:'+str(row['field']), 'class:'+row['race_class']):
                    groups[group][n].update(v)
    if seen != set(old) or dict(excluded) != previous['outcome_exclusions']:
        raise ValueError('cohort/exclusions changed')
    if totals['previous_best']['hits'] != previous['methods']['uncertainty']['hits']:
        raise ValueError('previous result not reproduced')
    for v in totals.values():
        v['hit_rate'] = v['hits']/v['races']
        v['stake_yen'] = v['races']*CONFIG['tickets']*CONFIG['stake_per_ticket_yen']
    report = {'version': VERSION, 'scope': 'past_only_tuning_but_reused_retrospective_target',
        'first': CONFIG['train_cutoff'], 'end_exclusive': CONFIG['target_end'],
        'predicted_races': len(seen), 'outcome_exclusions': dict(excluded), 'methods': dict(totals),
        'groups': {k: dict(v) for k, v in groups.items()}, 'frozen_selection': m['selected'],
        'selected_exceeds_45_percent': totals['selected']['hit_rate'] > .45,
        'fresh_validation': False, 'production_promotion_allowed': False,
        'production_changed': False, 'roi': None,
        'limitations': ['入力は過去ページからの復元。予想当時の保存時刻を保証する記録ではない。',
            '選び方・学習期間は直近1年より前で固定。ただし比較する直近1年は過去の研究で参照済み。',
            '同じ13,612レース・各12点・1点100円。回収率と100倍以上の穴の別評価は未検証。']}
    write(output/'results.json', report)
    print('CHAIN_RESULTS '+json.dumps({k: report[k] for k in ('methods', 'selected_exceeds_45_percent', 'fresh_validation')}, ensure_ascii=False), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--stage', choices=('train', 'predict', 'assess'), required=True)
    args = p.parse_args()
    raw_sha = digest(args.history)
    allowed = None if args.stage != 'predict' else lambda c: c not in ('finish_pos', 'is_dnf', 'odds_win', 'payout') and not c.startswith(('result_', 'official_', 'actual_'))
    frame = pd.read_csv(args.history, dtype={'race_id': str, 'player_id': str}, low_memory=False, usecols=allowed)
    if args.stage == 'train': train(frame, args.output, raw_sha)
    elif args.stage == 'predict': forecast(frame, args.output, raw_sha)
    else: assess(frame, args.output, raw_sha, args.evidence_root)


if __name__ == '__main__':
    main()
