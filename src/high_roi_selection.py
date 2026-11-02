"""Exact frozen research selection, used only as a label on all forecasts."""
import json
import math
from pathlib import Path

RULE_ID = 'SC10955_native_20261102'
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models/high_roi_selection.json'


def evaluate(*, score, return_mass, points, high_probability, cars, graded, category, model=None):
    model = model or json.loads(MODEL_PATH.read_text(encoding='utf8'))
    if model['rule_id'] != RULE_ID or category not in ('MAIN', 'MEDIUM', 'HIGH'):
        raise ValueError('Unknown selection model or category')
    if not math.isfinite(score) or not 0 < score <= 1 + 1e-12:
        raise ValueError('Missing or invalid native ticket probability')
    if not math.isfinite(high_probability) or not 0 <= high_probability <= 1:
        raise ValueError('Invalid native HIGH probability')
    if points != (18 if graded else 10 if category == 'HIGH' else 8) or not 3 <= cars <= 9:
        raise ValueError('Ticket policy mismatch')
    grade = 'S' if score >= .30 else 'A' if score >= .20 else 'B' if score >= .10 else 'C'
    preference = None
    features = None
    if category == 'MAIN':
        if return_mass is None or not math.isfinite(return_mass) or not 0 < return_mass <= 1 + 1e-12:
            raise ValueError('Missing or invalid native return mass')
        features = [math.log(score), math.log(return_mass), math.log(points), high_probability,
                    float(cars), float(graded), float(grade == 'S')]
        scaled = [1.] + [max(-8., min(8., (v - mean) / sd)) for v, mean, sd in
                         zip(features, model['scaler']['mean'], model['scaler']['std'], strict=True)]
        logit = math.fsum(w * x for w, x in zip(model['coefficients'], scaled, strict=True))
        preference = 1 / (1 + math.exp(-logit))
    selected = grade in ('S', 'C') and (category != 'MAIN' or preference > model['threshold'])
    return dict(rule_id=RULE_ID, selected=selected, confidence_grade=grade,
                native_ticket_probability=score, purchase_preference=preference, features=features,
                basis='before_EX', all_forecasts_visible=True)
