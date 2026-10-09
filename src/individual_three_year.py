"""Isolated three-calendar-year training with race weights 4 : 2 : 1.

No live department model is overwritten. Every stacking input is predicted by
a base model fitted strictly before its date. Calibration is not a holdout.
"""
import argparse
import gc
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import fixed_year_features as features
import fixed_year_models as experts
import fixed_year_study as study
import fusion_equations as fusion
import fusion_input_repair as repair
import fusion_stage_context as stage

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'models/individual_three_year'
VERSION = 'three_calendar_years_421_v1'


def weights(dates, cutoff):
    end = pd.Timestamp(cutoff)
    dates = pd.to_datetime(pd.Series(dates), errors='raise')
    if not (dates.ge(end - pd.DateOffset(years=3)) & dates.lt(end)).all():
        raise ValueError('training date outside the three-year window')
    return np.where(dates.ge(end - pd.DateOffset(years=1)), 4.,
                    np.where(dates.ge(end - pd.DateOffset(years=2)), 2., 1.))


def hashes():
    names = ('individual_three_year.py', 'fusion_input_repair.py',
             'fusion_stage_context.py', 'archive_50000.py')
    return {**study.source_hashes(), **{n: hashlib.sha256(
        (ROOT / 'src' / n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in names}}


def profiles(frame, cutoff, reference_day):
    work = frame.copy()
    work['_w'] = weights(work.date, cutoff)
    pos = pd.to_numeric(work.finish_pos, errors='coerce')
    work['_valid'] = pos.between(1, work.groupby('race_id').race_id.transform('size')) & pos.mod(1).eq(0)
    for p in (1, 2, 3):
        work[f'_p{p}'] = pos.eq(p).astype(float)
    def stats(rows):
        good = rows[rows._valid]
        mass = float(good._w.sum())
        return {'races': len(good), 'effective_races': mass,
                'rates': [float((good[f'_p{p}'] * good._w).sum() / mass) if mass else 0. for p in (1, 2, 3)]}
    result = {}
    for pid, rows in work.groupby('player_id', sort=False):
        base = stats(rows)
        if not base['races']:
            continue
        profile = {'evaluation': {**base, 'entries': len(rows),
            'effective_entries': float(rows._w.sum()),
            'effective_unplaced': float(rows.loc[~rows._valid, '_w'].sum())},
            'recent90': stats(rows[pd.to_datetime(rows.date).ge(pd.Timestamp(reference_day) - pd.Timedelta(days=90))]),
            'line_positions': {}, 'events': {}, 'reference_years': 3, 'year_weights': [4., 2., 1.]}
        if 'line_position' in rows:
            verified = rows.get('line_verification_status', pd.Series('', index=rows.index)).eq('verified')
            for position, group in rows[verified].groupby('line_position'):
                if pd.notna(position):
                    profile['line_positions'][str(int(float(position)))] = stats(group)
        for c in rows:
            if c.startswith('result_event_'):
                observed = rows[c].notna()
                yes = rows[c].astype(str).str.lower().isin(('true', '1', '1.0'))
                profile['events'][c.removeprefix('result_event_')] = {
                    'observed': int(observed.sum()), 'effective_observed': float(rows.loc[observed, '_w'].sum()),
                    'effective_true': float(rows.loc[yes, '_w'].sum()), 'true_results': stats(rows[yes])}
        result[str(pid)] = profile
    return result


def fit_base(frame, cutoff):
    enriched, reference = features.daily_priors(frame)
    X, config = features.matrix(enriched)
    selected = [c for c in config['features'] if c not in repair.ARMS['refit']]
    # Each race contributes one recency weight, irrespective of field size.
    w = weights(frame.date, cutoff) / frame.groupby('race_id').race_id.transform('size').to_numpy()
    w /= w.mean()
    target = pd.to_numeric(frame.finish_pos, errors='coerce')
    fitted = {}
    for p in (1, 2, 3):
        model = HistGradientBoostingClassifier(max_iter=180, learning_rate=.05,
            max_leaf_nodes=31, l2_regularization=.03, early_stopping=False, random_state=41+p)
        model.fit(X[selected], target.eq(p).astype(int), sample_weight=w)
        fitted[p] = repair.FeatureSubsetClassifier(model, selected)
    print('THREE_YEAR_BASE ' + str(max(frame.date)), flush=True)
    return {'classifiers': fitted, 'feature_config': config, 'reference': reference,
            'profiles': profiles(frame, cutoff, repair.boundary(frame)),
            'base_training_last': max(frame.date), 'selected_features': selected}


def joint_fit(X, lengths, targets, race_weights):
    starts = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    target_rows = starts + np.asarray(targets)
    w = np.asarray(race_weights, dtype=X.dtype); w /= w.sum()
    beta = np.zeros(X.shape[1], dtype=X.dtype)
    for _ in range(fusion.CONFIG['iterations']):
        logits = X @ beta
        exp = np.exp(logits - np.repeat(np.maximum.reduceat(logits, starts), lengths))
        residual = exp / np.repeat(np.add.reduceat(exp, starts), lengths)
        residual[target_rows] -= 1
        residual *= np.repeat(w, lengths)
        beta -= fusion.CONFIG['learning_rate'] * (X.T @ residual + fusion.CONFIG['ridge'] * beta)
    return beta.astype(float).tolist()


def fit_equations(frame, bundle, folder, cutoff):
    predicted = repair.later_prediction(frame, bundle)
    answers = {str(rid): study.outcome(r) for rid, r in frame.groupby('race_id', sort=False)}
    lengths = [len(r)*(len(r)-1)*(len(r)-2) for _, r in predicted.groupby('race_id', sort=False)]
    scratch = folder / 'joint_design.f32'
    X = np.memmap(scratch, dtype=np.float32, mode='w+', shape=(sum(lengths), 122))
    targets, rw, px, py, pw = [], [], [], [], []
    offset = 0
    for count, (rid, race) in enumerate(predicted.groupby('race_id', sort=False), 1):
        packet = experts.make_packet(race, bundle['profiles'])
        triples, context = fusion.triples(packet), fusion.feature_context(packet)
        X[offset:offset+len(triples)] = np.asarray([fusion.joint_features(packet, t, context) for t in triples])
        offset += len(triples)
        actual = answers[str(rid)]
        targets.append(fusion.keys(packet).index(actual))
        weight = float(weights([str(race.iloc[0].date)], cutoff)[0]); rw.append(weight)
        from equation_models import rider_features
        fs = rider_features(packet['source']['evidence']['inputs']['risk_department'])
        ranks = {car: i for i, car in enumerate(map(int, actual.split('-')))}
        pairs = [(a, b) for a, b in combinations(sorted(fs), 2) if a in ranks or b in ranks]
        for a, b in pairs:
            px.append([u-v for u, v in zip(fs[a], fs[b])])
            py.append(float(ranks.get(a, 3) < ranks.get(b, 3))); pw.append(weight/len(pairs))
        if count % 1000 == 0:
            print(f'THREE_YEAR_DESIGN {count}/{len(lengths)}', flush=True)
    model = {'joint_beta': joint_fit(X, lengths, targets, rw),
             'pair_beta': experts.fit_pairwise(px, py, pw)}
    del X, predicted, px, py, pw
    gc.collect(); scratch.unlink()
    return model


def pool_weights(p, sample_weights, prior=None):
    p = np.asarray(p, dtype=float)
    sw = np.asarray(sample_weights, dtype=float); sw /= sw.sum()
    if p.ndim != 2 or not np.isfinite(p).all() or (p <= 0).any() or (sw <= 0).any():
        raise ValueError('invalid weighted pool observations')
    prior = np.full(p.shape[1], 1/p.shape[1]) if prior is None else np.asarray(prior)
    error = -np.log(p); error -= np.sum(error * sw[:, None], axis=0)
    error *= np.sqrt(sw[:, None]); error /= np.maximum(np.linalg.norm(error, axis=0), 1e-12)
    correlation = error.T @ error
    logits = np.log(prior)
    for _ in range(fusion.CONFIG['pool_iterations']):
        w = np.exp(logits-logits.max()); w /= w.sum()
        grad = np.sum((1-p/(p@w)[:, None]) * sw[:, None], axis=0)
        grad += fusion.CONFIG['pool_redundancy'] * (correlation@w)
        grad += fusion.CONFIG['pool_prior_penalty'] * (np.log(w/prior)+1)
        logits -= .2*w*(grad-float(w@grad))
    w = np.exp(logits-logits.max())
    return w/w.sum()


def component_rows(frame, bundle, equations):
    answers = {str(rid): study.outcome(r) for rid, r in frame.groupby('race_id', sort=False)}
    for count, (rid, race) in enumerate(repair.later_prediction(frame, bundle).groupby('race_id', sort=False), 1):
        packet = experts.make_packet(race, bundle['profiles'])
        yield str(race.iloc[0].date), stage.context_key(len(race), str(race.iloc[0].get('race_class', 'missing'))), answers[str(rid)], experts.components(equations, packet)
        if count % 1000 == 0:
            print(f'THREE_YEAR_STACK {count}', flush=True)


def fit_pool_and_stage(frame, bundle, equations, cutoff):
    true, conditional, dates, contexts = [], [], [], []
    names, fingerprints = None, {}
    for day, context, actual, available in component_rows(frame, bundle, equations):
        if names is None:
            names = sorted(available); fingerprints = {n: hashlib.sha256() for n in names}
        if set(names) != set(available):
            raise ValueError('inconsistent expert availability')
        keys = sorted(available[names[0]], key=lambda k: tuple(map(int, k.split('-'))))
        matrix = np.asarray([[available[n][k] for k in keys] for n in names])
        for name in names:
            fingerprints[name].update(json.dumps(sorted((k, round(p, 12)) for k, p in available[name].items())).encode())
        index = keys.index(actual)
        true.append(matrix[:, index]); conditional.append(stage.decompose(keys, matrix)[:, :, index])
        dates.append(day); contexts.append(context)
    clusters = {}
    for name in names:
        clusters.setdefault(fingerprints[name].hexdigest(), []).append(name)
    aliases = {group[0]: group for group in clusters.values()}
    chosen = sorted(aliases); ix = [names.index(n) for n in chosen]
    joint = np.asarray(true)[:, ix]; stages = np.asarray(conditional)[:, ix, :]
    w = weights(dates, cutoff)
    global_joint = pool_weights(joint, w)
    pool = {'weights': dict(zip(chosen, global_joint.tolist())), 'aliases': aliases, 'fitted_races': len(dates)}
    global_stage = np.array([pool_weights(stages[:, :, s], w) for s in range(3)])
    groups = {}
    for context in sorted(set(contexts)):
        mask = np.asarray(contexts) == context
        count = int(mask.sum())
        if count < stage.CONFIG['minimum_group_races'] or len(set(np.asarray(dates)[mask])) < stage.CONFIG['minimum_group_days']:
            continue
        strength = count / (count + stage.CONFIG['group_shrinkage_races'])
        local_joint = pool_weights(joint[mask], w[mask], global_joint)
        local_stage = np.array([pool_weights(stages[mask, :, s], w[mask], global_stage[s]) for s in range(3)])
        groups[context] = {'races': count, 'joint': ((1-strength)*global_joint+strength*local_joint).tolist(),
                           'stages': ((1-strength)*global_stage+strength*local_stage).tolist()}
    return pool, {'names': chosen, 'aliases': aliases, 'global_joint': global_joint.tolist(),
        'global_stages': global_stage.tolist(), 'groups': groups, 'fit_first': min(dates), 'fit_last': max(dates)}


def calibrate(frame, bundle, equations, stage_model, cutoff):
    temperatures = fusion.CONFIG['temperatures']
    losses = np.zeros(len(temperatures))
    grids = [(alpha, tau, base_tau) for alpha in stage.CONFIG['blend_grid'] for tau in temperatures for base_tau in temperatures]
    stage_losses = {n: np.zeros(len(grids)) for n in stage.VARIANTS}
    total = 0.
    for day, context, actual, available in component_rows(frame, bundle, equations):
        keys, matrix = stage.expert_matrix(equations['pool'], available)
        base = fusion.mix(equations['pool'], available)
        baselines = {tau: fusion.temperature(base, tau)[actual] for tau in temperatures}
        weight = float(weights([day], cutoff)[0]); total += weight
        losses -= weight*np.log([baselines[tau] for tau in temperatures])
        raw = stage.raw_predictions(stage_model, keys, matrix, context)
        index = keys.index(actual)
        for name, p in raw.items():
            candidates = {tau: stage.temper(p, tau)[index] for tau in temperatures}
            stage_losses[name] -= weight*np.log([(1-a)*baselines[b]+a*candidates[t] for a,t,b in grids])
    if not total:
        raise ValueError('no calibration races')
    best = int(np.argmin(losses)); equations['temperature'] = temperatures[best]
    stage_model['calibration'] = {'variants': {}}
    for name, values in stage_losses.items():
        candidates = [i for i, item in enumerate(grids) if item[2] == temperatures[best]]
        selected = min(candidates, key=lambda i: (values[i], grids[i][0], abs(grids[i][1]-1)))
        alpha, tau, _ = grids[selected]
        stage_model['calibration']['variants'][name] = {'blend': alpha, 'temperature': tau, 'nll': float(values[selected]/total)}
    return {'metric': 'recency_weighted_nll', 'base_temperature': temperatures[best],
            'calibration_nll': float(losses[best]/total), 'is_independent_test': False}


def fit_archive(frame, cutoff):
    from archive_50000 import rider_vectors
    X, ys, w, px, py, pw = [], [[], [], []], [], [], [], []
    for _, race in frame.groupby('race_id', sort=False):
        fs = rider_vectors(race.to_dict('records'))
        ranks = {int(r.car_no): int(r.finish_pos) for r in race.itertuples() if r.finish_pos in (1, 2, 3)}
        weight = float(weights([race.iloc[0].date], cutoff)[0])
        for car in sorted(fs):
            X.append(fs[car]); w.append(weight/len(fs))
            for p in range(3):
                ys[p].append(int(ranks.get(car) == p+1))
        pairs = [(a,b) for a,b in combinations(sorted(fs),2) if a in ranks or b in ranks]
        for a,b in pairs:
            px.append([u-v for u,v in zip(fs[a],fs[b])]); py.append(int(ranks.get(a,99)<ranks.get(b,99))); pw.append(weight/len(pairs))
    betas, intercepts = [], []
    for y in ys:
        model = LogisticRegression(C=3, max_iter=120, random_state=2026).fit(X, y, sample_weight=w)
        betas.append(model.coef_[0].tolist()); intercepts.append(float(model.intercept_[0]))
    pairs = LogisticRegression(C=3, max_iter=120, fit_intercept=False, random_state=2026).fit(px, py, sample_weight=pw)
    return {'position_betas': betas, 'position_intercepts': intercepts, 'pairwise_beta': pairs.coef_[0].tolist()}


def train(history_path, cutoff, folder=FOLDER):
    history = pd.read_csv(history_path, dtype={'race_id': str, 'player_id': str}, low_memory=False)
    dates = pd.to_datetime(history.date, errors='coerce')
    end = pd.Timestamp(cutoff); start = end-pd.DateOffset(years=3)
    history = history[dates.ge(start) & dates.lt(end)].copy()
    history['date'] = pd.to_datetime(history.date).dt.strftime('%Y-%m-%d')
    clean, excluded = study.eligible(history, labels=True)
    clean['finish_pos'] = pd.to_numeric(clean.finish_pos, errors='coerce')
    clean['car_no'] = pd.to_numeric(clean.car_no, errors='raise').astype(int)
    if clean.date.nunique() < 900 or clean.race_id.nunique() < 10000:
        raise ValueError('three-year real archive coverage is insufficient')
    days = sorted(clean.date.unique()); cuts = [0, int(len(days)*.45), int(len(days)*.75), int(len(days)*.9), len(days)]
    phases = [clean[clean.date.isin(days[a:b])].reset_index(drop=True) for a,b in zip(cuts,cuts[1:])]
    info = [{'phase': n, 'first': min(f.date), 'last': max(f.date), 'races': int(f.race_id.nunique())}
            for n,f in zip(('base','equations','mixture','calibration'), phases)]
    print('THREE_YEAR_PHASES '+json.dumps(info), flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    base = fit_base(phases[0], cutoff)
    equations = fit_equations(phases[1], base, folder, cutoff)
    base = fit_base(pd.concat(phases[:2], ignore_index=True), cutoff)
    pool, stage_model = fit_pool_and_stage(phases[2], base, equations, cutoff)
    equations['pool'] = pool
    base = fit_base(pd.concat(phases[:3], ignore_index=True), cutoff)
    calibration = calibrate(phases[3], base, equations, stage_model, cutoff)
    base = fit_base(clean, cutoff)
    base.update(equations=equations, stage=stage_model, archive=fit_archive(clean, cutoff))
    joblib.dump(base, folder/'model.joblib', compress=3)
    race_days = clean.drop_duplicates('race_id').date
    ww = weights(race_days, cutoff)
    manifest = {'version': VERSION, 'training_cutoff_exclusive': cutoff,
        'training_window_start': str(start.date()), 'training_first': min(clean.date),
        'training_last': max(clean.date), 'training_races': int(clean.race_id.nunique()),
        'year_weights_recent_to_old': [4, 2, 1],
        'year_races_recent_to_old': [int((ww == w).sum()) for w in (4, 2, 1)],
        'exclusions': excluded, 'phases': info, 'calibration': calibration,
        'model_sha256': study.digest_file(folder/'model.joblib'), 'source_hashes': hashes(),
        'history_sha256': study.digest_file(history_path), 'trained_equations': 20,
        'market_residual_training': False, 'measured_path_training': False,
        'historical_roi_validation': 'unavailable_without_preclose_trifecta_odds',
        'production_risk_changed': False, 'purchase_authorized': False}
    study.write_json(folder/'manifest.json', manifest)
    print('THREE_YEAR_COMPLETE '+json.dumps({k:v for k,v in manifest.items() if k not in ('source_hashes',)}), flush=True)
    return manifest


def load(asof, folder=FOLDER):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['version'] != VERSION or manifest['source_hashes'] != hashes() or manifest['model_sha256'] != study.digest_file(folder/'model.joblib'):
        raise ValueError('three-year model provenance mismatch')
    if manifest['training_cutoff_exclusive'] > str(asof):
        raise ValueError('model trained after forecast date')
    return manifest, joblib.load(folder/'model.joblib')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=ROOT/'data/raw/history.csv')
    parser.add_argument('--asof', required=True)
    args = parser.parse_args()
    train(args.history, args.asof)
