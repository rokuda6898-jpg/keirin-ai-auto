"""Paired retrospective ablation of conditional-trifecta prior freshness.

Research only: fixed model, all admissible races, exactly twelve tickets.
Never updates production models, ledgers, site files, or declares a reused
retrospective period to be fresh holdout validation.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

VERSION = 'conditional_58_prior_freshness_ablation_v1'
MODES = ('frozen', 'expanding_daily', 'rolling_365d')
START, END = '2025-10-09', '2026-10-09'
PRIOR = ('player_prior_races', 'player_prior_win_rate', 'player_prior_place2_rate',
         'player_prior_place3_rate', 'player_prior_avg_finish',
         'player_prior_days_since_last_race')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def daily_prior_table(history, horizon_days=None):
    """No same-day labels: emit covariates before absorbing that day's outcomes.

    Rows without a legal observed finish contribute zero exposure, matching the
    existing baseline convention. Date membership, not race-id ordering, is the
    temporal boundary. Rolling windows are [date-horizon, date).
    """
    needed = {'player_id', 'race_id', 'date', 'finish_pos'}
    if not needed.issubset(history):
        raise ValueError('missing historical identity or result columns')
    if history.duplicated(['race_id', 'player_id']).any():
        raise ValueError('duplicate historical rider')
    if horizon_days is not None and horizon_days <= 0:
        raise ValueError('positive horizon required')
    frame = history.copy()
    dates = pd.to_datetime(frame.date, errors='raise')
    if not dates.dt.strftime('%Y-%m-%d').eq(frame.date.astype(str)).all():
        raise ValueError('ISO race dates required')
    field = frame.groupby('race_id').race_id.transform('size')
    finish = pd.to_numeric(frame.finish_pos, errors='coerce')
    valid = finish.between(1, field) & finish.mod(1).eq(0)
    finish = finish.where(valid)
    work = pd.DataFrame({'player_id': frame.player_id.astype(str), 'date': frame.date.astype(str),
        'n': valid.astype(float), 'win': finish.eq(1).astype(float),
        'p2': finish.le(2).astype(float), 'p3': finish.le(3).astype(float),
        'finish': finish.fillna(0)})
    daily = work.groupby(['player_id', 'date'], as_index=False).sum().sort_values(['player_id', 'date'])
    columns = ['n', 'win', 'p2', 'p3', 'finish']
    records = []
    for pid, group in daily.groupby('player_id', sort=False):
        days = pd.to_datetime(group.date).to_numpy(dtype='datetime64[D]')
        values = group[columns].to_numpy(dtype=float)
        cumulative = np.vstack([np.zeros((1, len(columns))), values.cumsum(axis=0)])
        upper = np.arange(len(group))  # prefix before this date, excludes today
        lower = np.zeros(len(group), dtype=int) if horizon_days is None else np.searchsorted(
            days, days - np.timedelta64(horizon_days, 'D'), side='left')
        previous = cumulative[upper] - cumulative[lower]
        table = group[['player_id', 'date']].copy()
        table['player_prior_races'] = previous[:, 0]
        for output, k in (('win_rate', 1), ('place2_rate', 2), ('place3_rate', 3), ('avg_finish', 4)):
            table['player_prior_' + output] = np.divide(previous[:, k], previous[:, 0],
                out=np.full(len(group), np.nan), where=previous[:, 0] > 0)
        gaps = np.full(len(group), np.nan)
        if len(group) > 1:
            gaps[1:] = np.diff(days).astype('timedelta64[D]').astype(float)
        table['player_prior_days_since_last_race'] = gaps
        records.append(table)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=['player_id', 'date', *PRIOR])


def attach_priors(frame, table):
    import fixed_year_features as ff
    out = ff.mask_outcomes(frame).drop(columns=list(PRIOR), errors='ignore').copy()
    out['player_id'] = out.player_id.astype(str)
    out['_research_order'] = np.arange(len(out))
    out = out.merge(table, on=['player_id', 'date'], how='left', validate='many_to_one', indicator=True)
    if not out['_merge'].eq('both').all():
        raise ValueError('missing causal reference rows')
    return out.sort_values('_research_order').drop(columns=['_research_order', '_merge']).reset_index(drop=True)


def predict_enriched(clean, enriched, bundle):
    """Same baseline factorization and models; only prepared prior inputs differ."""
    import trifecta_chain_research as chain
    if not clean[['race_id', 'player_id', 'car_no']].equals(enriched[['race_id', 'player_id', 'car_no']]):
        raise ValueError('feature alignment mismatch')
    X, _ = chain.feature_matrix(enriched, bundle['feature_config'])
    if list(X) != bundle['features']:
        raise ValueError('model feature schema changed')
    threads = chain.CONFIG['threads']
    base = {name: bundle['models'][name].booster_.predict(X.to_numpy(copy=False), num_threads=threads)
            for name in ('first', 'second', 'third')}
    groups = list(clean.groupby('race_id', sort=False))
    for start in range(0, len(groups), 128):
        batch = groups[start:start + 128]
        pidx, tidx, lengths = [], [], []
        for _, race in batch:
            pairs, triples = chain.hypotheses(len(race))
            pidx.append(race.index.to_numpy()[pairs])
            tidx.append(race.index.to_numpy()[triples])
            lengths.append((len(pairs), len(triples)))
        pidx, tidx = np.concatenate(pidx), np.concatenate(tidx)
        p2 = bundle['models']['given_first'].booster_.predict(
            chain.design(X, pidx[:, 1], pidx[:, 0]), num_threads=threads)
        p3 = bundle['models']['given_pair'].booster_.predict(
            chain.design(X, tidx[:, 2], tidx[:, 0], tidx[:, 1]), num_threads=threads)
        poff = toff = 0
        for (rid, race), (plen, tlen) in zip(batch, lengths):
            pairs, triples = chain.hypotheses(len(race))
            independent, conditional = chain.combine(*(base[n][race.index] for n in ('first', 'second', 'third')),
                p2[poff:poff + plen], p3[toff:toff + tlen], pairs, triples)
            poff += plen
            toff += tlen
            keys = ['-'.join(map(str, race.car_no.astype(int).to_numpy()[t])) for t in triples]
            yield {'race_id': str(rid), 'keys': keys, 'independent': independent, 'chain': conditional}
        if start % 1280 == 0:
            print(f'PRIOR_PREDICT {min(start + 128, len(groups))}/{len(groups)}', flush=True)


def summarize(predictions, answers, meta, baseline):
    by_group = defaultdict(Counter)
    rows = []
    for rid, tickets in predictions.items():
        actual = answers.get(rid)
        if actual is None:
            continue
        hit, prior = actual in tickets, actual in baseline[rid]
        flags = {'races': 1, 'hits': int(hit), 'rescued': int(hit and not prior), 'lost': int(prior and not hit)}
        if not hit:
            prefix = actual.split('-')
            if not any(t.split('-')[0] == prefix[0] for t in tickets):
                flags['miss_first_not_covered'] = 1
            elif not any(t.split('-')[:2] == prefix[:2] for t in tickets):
                flags['miss_second_not_covered'] = 1
            else:
                flags['miss_third_not_covered'] = 1
        for key in ('overall', 'month:' + meta[rid]['date'][:7], 'field:' + str(meta[rid]['field']),
                    'class:' + meta[rid]['class']):
            by_group[key].update(flags)
        rows.append({'race_id': rid, 'date': meta[rid]['date'], 'hit': int(hit), 'baseline_hit': int(prior)})
    result = {}
    for key, counts in by_group.items():
        n, hits = counts['races'], counts['hits']
        p, z = hits/n, 1.959963984540054
        center = (p + z*z/(2*n))/(1 + z*z/n)
        radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
        result[key] = {**counts, 'hit_rate': p, 'flat_stake_yen': n*1200,
                       'hit_rate_wilson95_descriptive_only': [center-radius, center+radius]}
    daily = defaultdict(lambda: [0, 0])
    for row in rows:
        daily[row['date']][0] += row['hit'] - row['baseline_hit']
        daily[row['date']][1] += 1
    values = np.asarray(list(daily.values()))
    rng = np.random.default_rng(58011)
    samples = values[rng.integers(0, len(values), (1000, len(values)))].sum(axis=1)
    delta = samples[:, 0]/samples[:, 1]
    result['paired_daily_bootstrap_delta95_descriptive_only'] = np.quantile(delta, [.025, .975]).tolist()
    return result, rows


def run(root, input_dir, output):
    import trifecta_chain_research as chain
    import fixed_year_features as ff
    from fixed_year_study import eligible, outcome
    root, input_dir, output = map(Path, (root, input_dir, output))
    if output.exists():
        raise ValueError('use a fresh research output directory')
    manifest = json.loads((input_dir/'manifest.json').read_text())
    compressed = input_dir/'history.csv.gz'
    if sha(compressed) != manifest['gzip_sha256']:
        raise ValueError('history gzip checksum mismatch')
    with gzip.open(compressed, 'rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest['raw_sha256']:
            raise ValueError('history raw checksum mismatch')
    model_path = root/'models/conditional_trifecta_v1.joblib'
    model_manifest = json.loads((root/'models/conditional_trifecta_v1.json').read_text())
    if sha(model_path) != model_manifest['model_sha256']:
        raise ValueError('model checksum mismatch')
    # No pickled model from an arbitrary user-controlled input path is loaded.
    for name, digest in model_manifest['source_hashes'].items():
        normalized = hashlib.sha256((root/'src'/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if normalized != digest:
            raise ValueError('baseline source checksum mismatch: ' + name)
    bundle = joblib.load(model_path)
    if bundle['cutoff'] != START:
        raise ValueError('unexpected trained baseline cutoff')
    history = pd.read_csv(compressed, dtype={'race_id': str, 'player_id': str}, low_memory=False)
    if len(history) != manifest['rows'] or history.race_id.nunique() != manifest['races']:
        raise ValueError('historical inventory mismatch')
    if history.duplicated(['race_id', 'car_no']).any():
        raise ValueError('duplicate race car')
    target = history[history.date.ge(START) & history.date.lt(END)]
    clean, input_exclusions = eligible(target, labels=False)
    clean = chain.frame_order(clean)
    if clean.groupby('race_id').size().min() < 4:
        raise ValueError('fewer than 12 possible triples; requires explicit cohort policy')
    # Features may use only valid earlier-day result observations. Current-day
    # outcomes are not part of the ranking frame or of their own prior counts.
    reference_history, reference_exclusions = eligible(history, labels=True)
    # Include outcome-ambiguous target races as zero-label queries so that they
    # still receive forecasts and appear as explicit settlement exclusions.
    known = set(reference_history.race_id)
    queries = ff.mask_outcomes(clean[~clean.race_id.isin(known)])
    prior_history = pd.concat([reference_history, queries], ignore_index=True, sort=False)
    output.mkdir(parents=True)
    declaration = {'version': VERSION, 'start': START, 'end_exclusive': END,
        'modes_fixed_before_run': list(MODES), 'target': .58, 'tickets_per_race': 12,
        'race_selection': 'all baseline-admissible races; no confidence filter',
        'target_period_previously_inspected': True, 'fresh_holdout': False,
        'production_changed': False, 'automatic_promotion': False,
        'model_sha256': sha(model_path), 'history_raw_sha256': manifest['raw_sha256'],
        'code_sha256': sha(__file__), 'input_exclusions': input_exclusions,
        'reference_outcome_exclusions': reference_exclusions,
        'result_availability_assumption': 'race results available by following JST day; retrospective archive, not logged availability',
        'return_rate': None, 'return_rate_reason': 'no verified per-race trifecta payout join in this ablation'}
    write(output/'declaration.json', declaration)
    forecasts = {}
    for mode in MODES:
        if mode == 'frozen':
            enriched = ff.frozen_priors(clean, bundle['reference'])
        else:
            priors = daily_prior_table(prior_history, 365 if mode == 'rolling_365d' else None)
            enriched = attach_priors(clean, priors)
        predictions = {}
        for row in predict_enriched(clean, enriched, bundle):
            rid = row['race_id']
            if rid in predictions:
                raise ValueError('duplicate forecast')
            weight = model_manifest['chain_weight']
            p = (1-weight)*row['independent'] + weight*row['chain']
            predictions[rid] = chain.tickets(row['keys'], p)
        if set(predictions) != set(clean.race_id):
            raise ValueError('not all admissible races forecast')
        with gzip.open(output/(mode + '.jsonl.gz'), 'wt', encoding='utf-8') as stream:
            for rid, tickets in sorted(predictions.items()):
                stream.write(json.dumps({'race_id': rid, 'tickets': tickets}, ensure_ascii=False) + '\n')
        forecasts[mode] = predictions
        print('PRIOR_MODE_SAVED ' + json.dumps({'mode': mode, 'races': len(predictions)}), flush=True)
    # Outcomes are joined only after every mode's complete forecast is saved.
    answers = {str(rid): outcome(race) for rid, race in clean.groupby('race_id', sort=False)}
    meta = {str(rid): {'date': race.date.iloc[0], 'field': len(race), 'class': str(race.race_class.iloc[0])}
            for rid, race in clean.groupby('race_id', sort=False)}
    results = {'declaration': declaration, 'predicted_races': len(answers),
               'ambiguous_or_missing_podium': sum(v is None for v in answers.values()), 'methods': {}}
    paired = []
    for mode, predictions in forecasts.items():
        results['methods'][mode], rows = summarize(predictions, answers, meta, forecasts['frozen'])
        paired.extend(dict(mode=mode, **r) for r in rows)
    base = results['methods']['frozen']['overall']
    results['baseline_reproduced'] = (base['races'] == model_manifest['historical_races']
                                      and base['hits'] == model_manifest['historical_hits'])
    for mode in MODES:
        current = results['methods'][mode]['overall']
        current['delta_percentage_points'] = 100*(current['hit_rate'] - base['hit_rate'])
    results['fresh_58_percent_confirmed'] = False
    write(output/'results.json', results)
    pd.DataFrame(paired).to_csv(output/'paired_results.csv', index=False)
    print('PRIOR_ABLATION_RESULT ' + json.dumps({'baseline_reproduced': results['baseline_reproduced'],
        'methods': {m: results['methods'][m]['overall'] for m in MODES}}, ensure_ascii=False), flush=True)
    if not results['baseline_reproduced']:
        raise ValueError('published baseline not reproduced; investigate before comparing improvements')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.input, args.output)
