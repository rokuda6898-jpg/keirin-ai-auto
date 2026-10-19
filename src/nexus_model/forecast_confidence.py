"""Versioned model support for the saved ticket set, never an observed hit rate.

These initial thresholds are a fixed display policy, not fitted S/A/B outcome
calibration. No category, market odds, result or stake amount enters the grade.
"""

from datetime import datetime, timezone
import itertools
import math


POLICY_ID = 'selected-mass-v1'
REASONS = {
    'missing_full_order_scores': '自信度の根拠となる全着順評価が保存されていません',
    'missing_saved_probabilities': '買い目に対応するモデル評価が保存されていません',
    'invalid_joint_distribution': '全着順のモデル評価を確認できません',
    'invalid_selected_probabilities': '買い目のモデル評価を確認できません',
    'invalid_tickets': '自信度を計算する買い目を確認できません',
    'frozen_snapshot_mismatch': '保存予想と表示中の買い目・保存時刻が一致しません',
    'unverified_preclose_snapshot': '締切前に保存されたモデル評価を確認できません',
}


def _annotation_time(value=None):
    if isinstance(value, datetime):
        return value.isoformat()
    return value or datetime.now(timezone.utc).isoformat()


def insufficient(reason_code='missing_full_order_scores', *, annotated_at=None,
                 source_captured_at=None):
    result = dict(grade='B', status='insufficient', policy_id=POLICY_ID,
                  model_mass=None, reason_code=reason_code,
                  reason=REASONS[reason_code], empirical_grade_validation=False,
                  annotated_at=_annotation_time(annotated_at))
    if source_captured_at:
        result['source_captured_at'] = source_captured_at
    return result


def _tickets(values):
    if not isinstance(values, (list, tuple)) or not values:
        return None
    result = []
    for value in values:
        parts = value.split('-') if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else []
        parts = tuple(str(car) for car in parts)
        if len(parts) != 3 or len(set(parts)) != 3 or any(car not in '123456789' or len(car) != 1 for car in parts):
            return None
        result.append('-'.join(parts))
    return result if len(set(result)) == len(result) else None


def _numbers(values):
    try:
        if any(isinstance(value, bool) for value in values):
            return None
        numbers = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        return None
    return numbers if numbers and all(math.isfinite(value) and 0 <= value <= 1 for value in numbers) else None


def _assessed(mass, count, *, annotated_at=None, source_captured_at=None, entropy=None):
    grade = 'S' if mass >= .65 else 'A' if mass >= .45 else 'B'
    level = {'S': 'high', 'A': 'medium', 'B': 'low'}[grade]
    reason = {'S': '選択買い目へのモデル支持が高い',
              'A': '選択買い目へのモデル支持は中程度',
              'B': 'モデル評価が買い目の外にも分散しています'}[grade]
    result = dict(grade=grade, status='assessed', policy_id=POLICY_ID,
                  model_mass=float(mass), ticket_count=count,
                  reason_code=level + '_selected_mass', reason=reason,
                  empirical_grade_validation=False,
                  annotated_at=_annotation_time(annotated_at))
    if source_captured_at:
        result['source_captured_at'] = source_captured_at
    if entropy is not None:
        result['normalized_entropy'] = min(1.0, max(0.0, float(entropy)))
    return result


def from_joint(cars, probabilities, tickets, *, annotated_at=None):
    """Assess the FINAL formation using the engine's permutation-ordered q."""
    selected = _tickets(tickets)
    if selected is None:
        return insufficient('invalid_tickets', annotated_at=annotated_at)
    try:
        cars = [str(car) for car in cars]
        valid_cars = 3 <= len(cars) <= 9 and len(set(cars)) == len(cars) and all(car in '123456789' and len(car) == 1 for car in cars)
        values = list(probabilities)
    except TypeError:
        valid_cars, values = False, []
    q = _numbers(values)
    size = len(cars) * (len(cars) - 1) * (len(cars) - 2) if valid_cars else 0
    if not valid_cars or q is None or len(q) != size or not math.isclose(math.fsum(q), 1.0, rel_tol=0, abs_tol=1e-8):
        return insufficient('invalid_joint_distribution', annotated_at=annotated_at)
    total = math.fsum(q)
    q = [value / total for value in q]
    scores = dict(zip(('-'.join(order) for order in itertools.permutations(cars, 3)), q))
    if any(ticket not in scores for ticket in selected):
        return insufficient('invalid_tickets', annotated_at=annotated_at)
    mass = math.fsum(scores[ticket] for ticket in selected)
    entropy = -math.fsum(value * math.log(value) for value in q if value) / math.log(size)
    return _assessed(min(1.0, mass), len(selected), annotated_at=annotated_at, entropy=entropy)


def _aware_time(value):
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, timezone.utc)
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed if parsed.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError, OSError):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            return None


def from_saved(row, *, tickets=None, snapshot_at=None, annotated_at=None):
    """Use frozen conditional top12/probabilities, with their actual receipt."""
    row = row if isinstance(row, dict) else {}
    captured_at = row.get('snapshot_at_jst')
    kwargs = dict(annotated_at=annotated_at, source_captured_at=captured_at)
    selected = _tickets(row.get('top12'))
    if selected is None:
        return insufficient('missing_saved_probabilities', **kwargs)
    if tickets is not None and _tickets(tickets) != selected:
        return insufficient('frozen_snapshot_mismatch', **kwargs)
    when, close = _aware_time(captured_at), _aware_time(row.get('close_at'))
    if snapshot_at is not None and (when is None or _aware_time(snapshot_at) != when):
        return insufficient('frozen_snapshot_mismatch', **kwargs)
    if when is None or close is None or when >= close:
        return insufficient('unverified_preclose_snapshot', **kwargs)
    probabilities = row.get('probabilities')
    if not isinstance(probabilities, (list, tuple)) or len(probabilities) != len(selected):
        return insufficient('missing_saved_probabilities', **kwargs)
    values = _numbers(probabilities)
    if values is None or math.fsum(values) > 1 + 1e-8:
        return insufficient('invalid_selected_probabilities', **kwargs)
    # These are selected-ticket probabilities only. Never normalize their sum
    # to one, which would turn every saved top12 into a spurious S rating.
    return _assessed(min(1.0, math.fsum(values)), len(selected), **kwargs)


def _existing(value):
    return (isinstance(value, dict) and value.get('grade') in ('S', 'A', 'B')
            and value.get('status') in ('assessed', 'insufficient')
            and value.get('policy_id') == POLICY_ID and bool(value.get('annotated_at')))


def annotate_prediction(prediction, *, annotated_at=None):
    """Add only confidence metadata; leave a missing forecast missing."""
    if not isinstance(prediction, dict) or not prediction.get('tickets'):
        return None
    if not _existing(prediction.get('confidence')):
        prediction['confidence'] = insufficient(annotated_at=annotated_at,
            source_captured_at=prediction.get('captured_at'))
    elif prediction.get('captured_at'):
        prediction['confidence'].setdefault('source_captured_at', prediction['captured_at'])
    return prediction['confidence']


def annotate_conditional_view(view, row, *, annotated_at=None):
    """Attach a grade to a public view without modifying the frozen ledger."""
    if not isinstance(view, dict) or not view.get('tickets'):
        return None
    if not _existing(view.get('confidence')):
        view['confidence'] = from_saved(row, tickets=view['tickets'],
            snapshot_at=view.get('snapshot_at'), annotated_at=annotated_at)
    return view['confidence']
