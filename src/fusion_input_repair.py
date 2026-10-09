"""Research-only repairs of frozen-year inputs and chronological refitting.

The original code, frozen risk forecast, and production tickets are immutable
comparators. A removed feature is removed during both fitting and prediction.
"""
import argparse
import gc
import gzip
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import fixed_year_features as features
import fixed_year_models as experts
import fixed_year_study as old
import fusion_equations as fusion
import equation_models as legacy
from annual_knowledge import build_annual_profiles
from fixed_year_research import split_history
from fusion_stage_study import verify_source, read_json

VERSION = 'fixed_year_input_repair_v1'
ARMS = {
    'gap': ('player_prior_days_since_last_race',),
    'identity': ('player_id_code',),
    'both': ('player_prior_days_since_last_race', 'player_id_code'),
    'refit': ('player_prior_days_since_last_race', 'player_id_code'),
}
LABELS = {'risk': '保存済みリスク部の現行評価式', 'fusion': '保存済み従来統合式',
          'original': '保存済み順位別モデル', 'gap': '① 日数項目の修正',
          'identity': '② 選手ID項目の修正', 'both': '③ 両方を修正',
          'refit': '④ 両方修正＋時系列で再学習',
          'refit_first_anchor': '参考：④の1着を再学習モデルに合わせる'}


def source_hashes():
    result = old.source_hashes()
    for name in ('fusion_input_repair.py', 'fusion_input_repair_assess.py', 'fusion_stage_study.py'):
        result[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return result


class FeatureSubsetClassifier:
    """Explicit saved input contract; no mutation of the legacy feature schema."""
    def __init__(self, estimator, columns):
        self.estimator = estimator
        self.columns = list(columns)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X.loc[:, self.columns])


def fit_base(frame, dropped, folder, cutoff, fast=False):
    if not pd.to_datetime(frame.date).lt(pd.Timestamp(cutoff)).all():
        raise ValueError('base training reaches its prediction cutoff')
    folder.mkdir(parents=True, exist_ok=True)
    enriched, reference = features.daily_priors(frame)
    X, config = features.matrix(enriched)
    selected = [c for c in config['features'] if c not in dropped]
    models = {}
    target = pd.to_numeric(frame.finish_pos, errors='coerce')
    for p in (1, 2, 3):
        model = HistGradientBoostingClassifier(max_iter=3 if fast else 180, learning_rate=.05,
            max_leaf_nodes=31, l2_regularization=.03, early_stopping=False, random_state=41+p)
        model.fit(X[selected], target.eq(p).astype(int))
        models[p] = FeatureSubsetClassifier(model, selected)
    del X, enriched
    gc.collect()
    profile_input = folder / 'profile_training.csv'
    frame.to_csv(profile_input, index=False)
    knowledge = build_annual_profiles(cutoff, profile_input, folder / 'profiles')
    profile_input.unlink()
    bundle = {'version': VERSION, 'classifiers': models, 'feature_config': config,
              'selected_features': selected, 'dropped_features': list(dropped),
              'reference': reference, 'profiles': knowledge['profiles'],
              'base_training_first': min(frame.date), 'base_training_last': max(frame.date),
              'base_training_races': int(frame.race_id.nunique()),
              'profile_cutoff_exclusive': cutoff}
    print('REPAIR_BASE ' + json.dumps({k: bundle[k] for k in ('base_training_last', 'base_training_races', 'dropped_features')}), flush=True)
    return bundle


def boundary(frame):
    return str((pd.Timestamp(max(frame.date)) + pd.Timedelta(days=1)).date())


def later_prediction(frame, bundle):
    if not pd.to_datetime(frame.date).gt(pd.Timestamp(bundle['base_training_last'])).all():
        raise ValueError('a stacking forecast is not strictly later than its base training')
    return old.base_predict(features.mask_outcomes(frame), bundle)


def fit_equations(frame, bundle, folder, fast=False):
    answers = {str(rid): old.outcome(r) for rid, r in frame.groupby('race_id', sort=False)}
    predicted = later_prediction(frame, bundle)
    lengths = [len(r)*(len(r)-1)*(len(r)-2) for _, r in predicted.groupby('race_id', sort=False)]
    scratch = folder / 'joint_design.f32'
    X = np.memmap(scratch, dtype=np.float32, mode='w+', shape=(sum(lengths), 122))
    offset = 0
    targets, px, py, pw = [], [], [], []
    for count, (rid, race) in enumerate(predicted.groupby('race_id', sort=False), 1):
        packet = experts.make_packet(race, bundle['profiles'])
        triples, ctx = fusion.triples(packet), fusion.feature_context(packet)
        design = np.asarray([fusion.joint_features(packet, t, ctx) for t in triples], dtype=np.float32)
        if design.shape[1] != 122:
            raise ValueError('original joint feature contract changed')
        X[offset:offset+len(triples)] = design
        offset += len(triples)
        actual = answers[str(rid)]
        targets.append(fusion.keys(packet).index(actual))
        fs = legacy.rider_features(packet['source']['evidence']['inputs']['risk_department'])
        ranks = {car: i for i, car in enumerate(map(int, actual.split('-')))}
        pairs = [(a, b) for a, b in combinations(sorted(fs), 2) if a in ranks or b in ranks]
        for a, b in pairs:
            px.append([u-v for u, v in zip(fs[a], fs[b])])
            py.append(float(ranks.get(a, 3) < ranks.get(b, 3)))
            pw.append(1/len(pairs))
        if count % 2000 == 0:
            print(f'REPAIR_DESIGNS {count}/{len(lengths)}', flush=True)
    model = {'joint_beta': experts.fit_joint_matrix(X, lengths, targets, iterations=2 if fast else None),
             'pair_beta': experts.fit_pairwise(px, py, pw)}
    del X, predicted, px, py, pw
    gc.collect()
    scratch.unlink()
    return model


def training_records(frame, bundle, model):
    answers = {str(rid): old.outcome(r) for rid, r in frame.groupby('race_id', sort=False)}
    for count, (rid, race) in enumerate(later_prediction(frame, bundle).groupby('race_id', sort=False), 1):
        packet = experts.make_packet(race, bundle['profiles'])
        yield answers[str(rid)], experts.components(model, packet)
        if count % 2000 == 0:
            print(f'REPAIR_STACKING {count}/{frame.race_id.nunique()}', flush=True)


def train(older, source, folder, arm, fast=False):
    if arm not in ARMS:
        raise ValueError('unknown predefined arm')
    original = read_json(source / 'frozen_manifest.json')
    cutoff = original['training_cutoff_exclusive']
    if not pd.to_datetime(older.date).lt(pd.Timestamp(cutoff)).all():
        raise ValueError('training contains target-year rows')
    clean, excluded = old.eligible(older, labels=True)
    phases = old.phase_split(clean)
    if excluded != original['training_exclusions']:
        raise ValueError('older training cohort changed')
    for frame, expected in zip(phases, original['phases']):
        if min(frame.date) != expected['first'] or max(frame.date) != expected['last'] or frame.race_id.nunique() != expected['races']:
            raise ValueError('older phase changed')
    folder.mkdir(parents=True, exist_ok=True)
    dropped = ARMS[arm]
    base = fit_base(phases[0], dropped, folder/'base', boundary(phases[0]), fast)
    model = fit_equations(phases[1], base, folder, fast)
    provenance = [{'purpose': 'equations', 'fit_last': base['base_training_last'],
                   'predict_first': min(phases[1].date), 'predict_last': max(phases[1].date)}]
    for phase_index, purpose in ((2, 'mixture'), (3, 'temperature')):
        if arm == 'refit':
            past = pd.concat(phases[:phase_index], ignore_index=True)
            base = fit_base(past, dropped, folder/purpose, boundary(past), fast)
        frame = phases[phase_index]
        provenance.append({'purpose': purpose, 'fit_last': base['base_training_last'],
                           'predict_first': min(frame.date), 'predict_last': max(frame.date)})
        if purpose == 'mixture':
            records = list(training_records(frame, base, model))
            model['pool'] = fusion.fit_pool(records)
            del records
            gc.collect()
        else:
            losses = {tau: 0. for tau in fusion.CONFIG['temperatures']}
            n = 0
            for actual, available in training_records(frame, base, model):
                distribution = fusion.mix(model['pool'], available)
                for tau in losses:
                    losses[tau] -= math.log(fusion.temperature(distribution, tau)[actual])
                n += 1
            model['calibration_fit_losses'] = {tau: loss/n for tau, loss in losses.items()}
            model['temperature'] = min(losses, key=lambda tau: (losses[tau], abs(tau-1)))
        print('REPAIR_PHASE_FINISHED ' + arm + ' ' + purpose, flush=True)
    if arm == 'refit':
        base = fit_base(clean, dropped, folder/'final', cutoff, fast)
    base['equations'] = model
    base['training_cutoff_exclusive'] = cutoff
    joblib.dump(base, folder/'model.joblib', compress=3)
    manifest = {'version': VERSION, 'arm': arm, 'dropped_features': list(dropped),
                'source_hashes': source_hashes(), 'source_model_sha256': original['model_sha256'],
                'model_sha256': old.digest_file(folder/'model.joblib'),
                'training_cutoff_exclusive': cutoff, 'original_phases': original['phases'],
                'training_exclusions': excluded, 'chronological_base_forecasts': provenance,
                'final_base_training_last': base['base_training_last'],
                'final_base_training_races': base['base_training_races'],
                'final_profile_cutoff_exclusive': base['profile_cutoff_exclusive'],
                'selected_features': base['selected_features'], 'temperature': model['temperature'],
                'calibration_fit_losses': model['calibration_fit_losses'],
                'base_fit_policy': 'expanding_past_only_then_all_older' if arm == 'refit' else 'same_original_base_period',
                'equations_fit_policy': 'original_separate_equation_period_for_all_arms',
                'selection_uses_target_outcomes': False, 'model_update_during_target': False,
                'already_inspected_target_year': True, 'ceo_integration': False,
                'risk_logic_changed': False, 'purchase_authorized': False}
    old.write_json(folder/'manifest.json', manifest)
    print('REPAIR_FROZEN ' + json.dumps({k: manifest[k] for k in ('arm', 'model_sha256', 'final_base_training_last')}), flush=True)
    return base


def anchor_first(distribution, reference):
    """Replace first marginals only, retaining every conditional lower order."""
    heads, desired = {}, {}
    if set(distribution) != set(reference):
        raise ValueError('anchor supports differ')
    for key, p in distribution.items():
        first = key.split('-')[0]
        heads[first] = heads.get(first, 0) + p
        desired[first] = desired.get(first, 0) + reference[key]
    return {key: p / heads[key.split('-')[0]] * desired[key.split('-')[0]] for key, p in distribution.items()}


def checked_bundle(folder):
    manifest = read_json(folder/'manifest.json')
    if source_hashes() != manifest['source_hashes']:
        raise ValueError('research source changed after training')
    if old.digest_file(folder/'model.joblib') != manifest['model_sha256']:
        raise ValueError('frozen repair model changed')
    return manifest, joblib.load(folder/'model.joblib')


def predict(target, source, folder):
    manifest, bundle = checked_bundle(folder)
    clean, exclusions = old.eligible(features.mask_outcomes(target))
    inventory = read_json(source/'inventory.json')
    if not pd.to_datetime(clean.date).between(pd.Timestamp(inventory['holdout_start_inclusive']),
             pd.Timestamp(inventory['asof_exclusive']), inclusive='left').all():
        raise ValueError('prediction dates outside the original target year')
    original = read_json(source/'prediction_manifest.json')
    original_path = source/'holdout_predictions.jsonl.gz'
    if old.digest_file(original_path) != original['predictions_sha256']:
        raise ValueError('original predictions modified')
    rows = later_prediction(clean, bundle)
    count = 0
    with gzip.open(original_path, 'rt', encoding='utf-8') as baseline, gzip.open(folder/'predictions.jsonl.gz', 'wt', encoding='utf-8') as handle:
        for rid, race in rows.groupby('race_id', sort=False):
            packet = experts.make_packet(race, bundle['profiles'])
            available = experts.components(bundle['equations'], packet)
            combined = fusion.temperature(fusion.mix(bundle['equations']['pool'], available), bundle['equations']['temperature'])
            distribution = {'base': available['original:model_positions'], 'fusion': combined}
            if manifest['arm'] == 'refit':
                distribution['first_anchor'] = anchor_first(combined, distribution['base'])
            keys = fusion.keys(packet)
            saved = json.loads(next(baseline))
            if str(rid) != saved['race_id'] or keys != saved['keys'] or str(race.iloc[0].date) != saved['date']:
                raise ValueError('forecast race/key/date mismatch with immutable baseline')
            if any(not fusion.valid_distribution(p, keys) for p in distribution.values()):
                raise ValueError('invalid forecast distribution')
            row = {k: saved[k] for k in ('race_id', 'date', 'field', 'race_class', 'keys')}
            row['distributions'] = {name: [p[k] for k in keys] for name, p in distribution.items()}
            row['department_scenarios'] = packet['auxiliary']['departments']
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
            count += 1
            if count % 2000 == 0:
                print(f'REPAIR_PREDICTED {manifest["arm"]} {count}', flush=True)
        if next(baseline, None) is not None:
            raise ValueError('missing target forecasts')
    if count != original['predicted_races'] or exclusions != original['input_exclusions']:
        raise ValueError('target input cohort changed')
    checked_bundle(folder)
    result = {'arm': manifest['arm'], 'predicted_races': count, 'input_exclusions': exclusions,
              'model_sha256': manifest['model_sha256'], 'model_manifest_sha256': old.digest_file(folder/'manifest.json'),
              'predictions_sha256': old.digest_file(folder/'predictions.jsonl.gz'),
              'source_predictions_sha256': original['predictions_sha256'],
              'outcomes_accessed': False, 'model_updated': False, 'risk_predictions_replaced': False}
    old.write_json(folder/'prediction_manifest.json', result)
    print('REPAIR_PREDICTION_FINISHED ' + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('train', 'predict', 'assess'), required=True)
    parser.add_argument('--arm', choices=tuple(ARMS))
    parser.add_argument('--history', type=Path, default=Path('data/raw/history.csv'))
    parser.add_argument('--source', type=Path, default=Path('research/frozen_original'))
    parser.add_argument('--output-dir', type=Path, default=Path('research/fusion_input_repair'))
    args = parser.parse_args()
    _, inventory = verify_source(args.source, args.history)
    history = pd.read_csv(args.history, dtype={'race_id': str, 'player_id': str}, low_memory=False)
    older, target = split_history(history, inventory['asof_exclusive'])
    if args.stage == 'assess':
        from fusion_input_repair_assess import assess
        assess(target, args.source, args.output_dir)
    else:
        if args.arm is None:
            parser.error('--arm is required for train/predict')
        folder = args.output_dir/args.arm
        if args.stage == 'train':
            train(older, args.source, folder, args.arm)
        else:
            predict(target, args.source, folder)


if __name__ == '__main__':
    # Pickles must refer to the importable module, not an ephemeral __main__.
    from fusion_input_repair import main as run
    run()
