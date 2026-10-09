"""Independent archive-trained position and pairwise forecasts.

Research only. Historical odds and measured in-race paths do not exist in
the central archive; neither market-residual nor trajectory training is claimed.
This module never writes purchase or CEO forecasts.
"""
import hashlib
import json
import math
from itertools import permutations
from pathlib import Path

VERSION = 'archive_50000_v1'
NAMES = {
    'archive_positions': '過去5万レース・順位別式',
    'archive_pairwise': '過去5万レース・先着関係式',
}
FEATURE_NAMES = (
    'score_centered', 'win_rate', 'place2_rate', 'place3_rate',
    'back_count', 'recent_finish', 'score_missing', 'recent_missing',
    'field_size',
)

def numeric(value, default=0.0):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (ValueError, TypeError):
        return default

def rate(value):
    x = numeric(value)
    if x > 1:
        x /= 100.0
    return min(1.0, max(0.0, x))

def rider_vectors(riders):
    """Same pre-event schema for archive fitting and current inference."""
    n = len(riders)
    if not 3 <= n <= 9:
        raise ValueError('unsupported field')
    cars = [int(r['car_no']) for r in riders]
    if len(set(cars)) != n or any(not 1 <= c <= 9 for c in cars):
        raise ValueError('duplicate or invalid car')
    scores = [numeric(r.get('score'), 100.0) for r in riders]
    center = sum(scores) / n
    result = {}
    for r, s in zip(riders, scores):
        result[int(r['car_no'])] = [
            max(-3.0, min(3.0, (s - center) / 15)),
            rate(r.get('win_rate')),
            rate(r.get('place2_rate')),
            rate(r.get('place3_rate')),
            min(1.0, max(0.0, numeric(r.get('back_count')) / 20)),
            min(1.0, max(0.0, numeric(r.get('recent_avg_finish'), 5.0) / 9)),
            float(r.get('score') is None),
            float(r.get('recent_avg_finish') is None),
            float(n) / 9,
        ]
    return result

def log_sigmoid(x):
    return -max(0.0, -x) - math.log1p(math.exp(-abs(x)))

def normalized_scores(logits):
    largest = max(logits.values())
    weights = {k: math.exp(v - largest) for k, v in logits.items()}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}

def probabilities(method, model, riders):
    fs = rider_vectors(riders)
    cars = sorted(fs)
    combos = list(permutations(cars, 3))
    if method == 'archive_pairwise':
        beta = model['pairwise_beta']
        pair = {}
        for a, b in permutations(cars, 2):
            delta = sum(w * (u-v) for w,u,v in zip(beta, fs[a], fs[b]))
            pair[a,b] = log_sigmoid(delta)
        logits = {}
        for a,b,c in combos:
            v = pair[a,b] + pair[a,c] + pair[b,c]
            v += sum(pair[w, other] for other in cars
                     if other not in (a,b,c) for w in (a,b,c))
            logits[f'{a}-{b}-{c}'] = v
        return normalized_scores(logits)
    if method != 'archive_positions':
        raise ValueError('unknown archive method')
    position = {}
    for i,(beta,intercept) in enumerate(zip(model['position_betas'],model['position_intercepts'])):
        logits = {car: sum(b*f for b,f in zip(beta,fs[car]))+intercept for car in cars}
        weights = {c: 1.0/(1.0+math.exp(-max(-30,min(30,v)))) for c,v in logits.items()}
        total = sum(weights.values())
        position[i] = {c: w/total for c,w in weights.items()}
    logits = {}
    for a,b,c in combos:
        second_denominator = max(1e-12, 1 - position[1][a])
        third_denominator = max(1e-12, 1 - position[2][a] - position[2][b])
        logits[f'{a}-{b}-{c}'] = (
            math.log(max(1e-15,position[0][a]))
            + math.log(max(1e-15,position[1][b])) - math.log(second_denominator)
            + math.log(max(1e-15,position[2][c])) - math.log(third_denominator))
    return normalized_scores(logits)

def body_digest(record):
    canonical = json.dumps({k:v for k,v in record.items() if k != 'model_sha256'},
                           ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

def validated_model(model, race_date):
    if not isinstance(model,dict) or model.get('version') != VERSION:
        return False
    if model.get('trained_races') != 50000 or model.get('feature_names') != list(FEATURE_NAMES):
        return False
    if model.get('training_end','9999-99-99') >= str(race_date) or model.get('train_holdout_overlap'):
        return False
    if model.get('model_sha256') != body_digest(model):
        return False
    try:
        if len(model['pairwise_beta']) != len(FEATURE_NAMES) or len(model['position_betas']) != 3:
            return False
        if len(model['position_intercepts']) != 3 or any(len(x)!=len(FEATURE_NAMES) for x in model['position_betas']):
            return False
        if any(not math.isfinite(float(x)) for row in
               [model['pairwise_beta'],*model['position_betas'],model['position_intercepts']] for x in row):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True

def load_model(path, race_date):
    path = Path(path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    return data if validated_model(data,race_date) else None
