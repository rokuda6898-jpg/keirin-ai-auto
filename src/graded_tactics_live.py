"""Frozen best tactical equation for pre-close G1/G2/G3 forecasts."""
import hashlib
import math
from itertools import permutations
from pathlib import Path

import joblib
import numpy as np

import graded_retrospective as base
from graded_tactics import conditional_tactical_features, tactical_inputs

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'graded_tactical_v1.joblib'
MODEL_SHA256 = '2d2fc06e43dc7ea54369dab71865fda36de1f93901686cb425b4b840df045657'
_BUNDLE = None
_TACTICS = {}
_ORIGINAL_FEATURES = base.features


def load_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError('Frozen tactical equation model is missing')
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            raise ValueError('Frozen tactical equation model hash mismatch')
        bundle = joblib.load(MODEL_PATH)
        if len(bundle.get('tactical', [])) != 3 or not bundle.get('stages'):
            raise ValueError('Frozen tactical equation model schema mismatch')
        _BUNDLE = bundle
    return _BUNDLE


def _features(records, candidate, prefix, stages, contextual):
    features = _ORIGINAL_FEATURES(records, candidate, prefix, stages, contextual)
    info = _TACTICS.get(str(records[0]['race_id']))
    extra = conditional_tactical_features(info, candidate, prefix) if info else [np.nan] * 8
    return features + extra


def predict(entries, race_data, grade):
    """Return exactly the model's 12 best ordered triples and readable marks."""
    rows = [dict(row) for row in entries]
    cars = [int(row['car_no']) for row in rows]
    if not 3 <= len(cars) <= 9 or len(set(cars)) != len(cars):
        raise ValueError('Invalid active rider list for tactical equation')
    race_id = str(race_data.get('race', {}).get('id') or rows[0].get('race_id') or '')
    if not race_id or any(str(row.get('race_id', race_id)) != race_id for row in rows):
        raise ValueError('Tactical equation race identity mismatch')
    for row in rows:
        row['race_id'] = race_id
        row['grade'] = grade
    tactical = tactical_inputs(race_data)
    _TACTICS[race_id] = tactical
    bundle = load_bundle()
    previous = base.features
    try:
        base.features = _features
        distribution = base.distribution(rows, bundle['tactical'], 'stage_conditional', bundle['stages'])
    finally:
        base.features = previous
        _TACTICS.pop(race_id, None)
    if not distribution or not math.isclose(sum(distribution.values()), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError('Tactical equation probability distribution is invalid')
    ordered = sorted(distribution, key=lambda ticket: (-distribution[ticket], ticket))
    tickets = ['-'.join(map(str, ticket)) for ticket in ordered[:12]]
    if len(tickets) != 12 or len(set(tickets)) != 12:
        raise ValueError('Tactical equation did not produce 12 distinct tickets')
    if any(len(set(ticket.split('-'))) != 3 for ticket in tickets):
        raise ValueError('Tactical equation produced an invalid ordered triple')
    return {
        'tickets': tickets,
        'probabilities': [float(distribution[ticket]) for ticket in ordered[:12]],
        'distribution_mass': float(sum(distribution.values())),
        'line_status': tactical.get('line_status', 'missing'),
        'model_version': 'graded_tactical_v1',
    }
