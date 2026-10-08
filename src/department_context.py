"""Observed opponent histories and explicit, uncalibrated race-context hypotheses.

Ratings are updated after each historical DAY. Current/future results are never
used. Incomplete fields do not create opponent-strength observations.
"""
import math
from collections import defaultdict


def class_key(value):
    try:
        number = float(value)
        return str(int(number)) if math.isfinite(number) and number.is_integer() else str(value).strip()
    except (TypeError, ValueError):
        return str(value or '').strip()


def opponent_profiles(history, cutoff):
    import pandas as pd
    if history.empty or not {'race_id', 'player_id', 'date', 'finish_pos'}.issubset(history.columns):
        return {}, {'status': 'missing_history', 'complete_races': 0}
    frame = history.copy()
    frame['_day'] = pd.to_datetime(frame.date, errors='coerce').dt.strftime('%Y-%m-%d')
    frame = frame[frame._day.lt(str(cutoff))].drop_duplicates(['race_id', 'player_id'], keep='last')
    ratings, buckets = {}, defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    complete = excluded = 0
    for _, day in frame.sort_values(['_day', 'race_id']).groupby('_day', sort=True):
        deltas = defaultdict(list)
        for _, race in day.groupby('race_id', sort=True):
            finish = pd.to_numeric(race.finish_pos, errors='coerce')
            declared = pd.to_numeric(race.get('entries_number', pd.Series(float('nan'), index=race.index)), errors='coerce')
            if (len(race) < 3 or declared.isna().any() or declared.ne(len(race)).any() or
                    not finish.between(1, len(race)).all() or not finish.mod(1).eq(0).all()):
                excluded += 1
                continue
            complete += 1
            ids = race.player_id.astype(str).tolist()
            positions = finish.tolist()
            classes = race.get('player_class', pd.Series('', index=race.index)).fillna('').map(class_key).tolist()
            before = [ratings.get(pid, 1500.0) for pid in ids]
            for i, pid in enumerate(ids):
                opponents = [j for j in range(len(ids)) if j != i]
                strength = sum(before[j] for j in opponents) / len(opponents)
                key = (classes[i], len(ids), int(round(strength / 100) * 100))
                bucket = buckets[pid][key]
                bucket[0] += 1
                for pos in (1, 2, 3):
                    bucket[pos] += positions[i] == pos
                residual = sum((1 if positions[i] < positions[j] else .5 if positions[i] == positions[j] else 0)
                               - 1 / (1 + 10 ** ((before[j] - before[i]) / 400)) for j in opponents) / len(opponents)
                deltas[pid].append(16 * residual)
        for pid, changes in deltas.items():
            ratings[pid] = ratings.get(pid, 1500.0) + sum(changes) / len(changes)
    profiles = {pid: {'rating': ratings[pid], 'buckets': [
        {'class': cls, 'field_size': size, 'opponent_rating': strength, 'races': values[0],
         'rates': [v / values[0] for v in values[1:]]}
        for (cls, size, strength), values in sorted(buckets[pid].items())]}
        for pid in ratings}
    return profiles, {'status': 'observed_history', 'complete_races': complete,
                      'excluded_incomplete_or_invalid_races': excluded, 'cutoff_exclusive': str(cutoff),
                      'method': 'daily_frozen_pairwise_rating_v1', 'calibrated': False}


def context_for(department, race, profiles):
    """Return a prefix-sensitive multiplier and auditable availability evidence."""
    records = {int(r['car_no']): r for r in race.to_dict('records')}
    evidence = {'department': department, 'hypothesis': 'observed_context_v2', 'calibrated': False}
    factors, sample = {}, {}
    def number(value, default=0):
        try:
            value = float(value)
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default
    if department == 'data_department':
        ratings = {c: profiles.get(str(r['player_id']), {}).get('opponent_context', {}).get('rating') for c, r in records.items()}
        if any(value is None for value in ratings.values()):
            return None, {**evidence, 'status': 'opponent_history_missing'}
        for car, rider in records.items():
            profile = profiles.get(str(rider['player_id']), {})
            strength = sum(v for c, v in ratings.items() if c != car) / (len(records) - 1)
            candidates = profile.get('opponent_context', {}).get('buckets', [])
            candidates = [b for b in candidates if b['field_size'] == len(records)
                          and b['class'] == class_key(rider.get('player_class', ''))]
            weights = [b['races'] * math.exp(-.5 * ((b['opponent_rating'] - strength) / 100) ** 2) for b in candidates]
            mass = sum(weights)
            if mass < 5:
                return None, {**evidence, 'status': 'matched_opponent_sample_insufficient'}
            base = profile.get('evaluation', profile).get('rates', [0, 0, 0])
            adjusted = [sum(b['rates'][p] * w for b, w in zip(candidates, weights)) / mass for p in range(3)]
            factors[car] = {p: max(.75, min(1.25, (adjusted[p-1] * mass + number(base[p-1]) * 20 + .5)
                                          / ((mass + 20) * number(base[p-1]) + .5))) for p in (2, 3)}
            sample[car] = {'matched_effective_races': mass, 'opponent_rating': strength}
        return lambda a, b, c, p: factors[c][p], {**evidence, 'status': 'available', 'matched_history': sample}
    verified = all(r.get('line_verification_status') == 'verified' and r.get('line_id') is not None
                   and str(r.get('line_id')).strip().lower() not in ('', 'nan', 'none')
                   and number(r.get('line_position')) >= 1 for r in records.values())
    if not verified:
        return None, {**evidence, 'status': 'verified_line_required'}
    leaders = [c for c, r in records.items() if number(r.get('line_position')) == 1
               and (str(r.get('style', '')) == '逃' or number(r.get('front_runner_count')) > 0)]
    if department == 'pace_department' and len({records[c]['line_id'] for c in leaders}) < 2:
        return None, {**evidence, 'status': 'observed_competing_leaders_required'}
    for car, rider in records.items():
        profile = profiles.get(str(rider['player_id']), {})
        if department == 'line_department':
            support = profile.get('line_positions', {}).get(str(int(number(rider['line_position']))), {})
            mass = number(support.get('effective_races', support.get('races')))
        else:
            event = profile.get('events', {}).get('back', {})
            mass = number(event.get('effective_observed', event.get('observed')))
        sample[car] = mass
    if max(sample.values(), default=0) < 10:
        return None, {**evidence, 'status': 'observed_context_sample_insufficient'}
    def multiplier(a, b, c, position):
        rider = records[c]
        same = rider['line_id'] == records[a]['line_id']
        confidence = sample[c] / (sample[c] + 20)
        if department == 'line_department':
            # If the first two belong to one line, evaluate its next rider;
            # if they cross lines, support the followers of either survivor.
            if b is None:
                signal = 1 if same and number(rider['line_position']) > number(records[a]['line_position']) else 0
            elif records[a]['line_id'] == records[b]['line_id']:
                following = same and number(rider['line_position']) > max(number(records[a]['line_position']), number(records[b]['line_position']))
                signal = (1 if number(records[a]['line_position']) > number(records[b]['line_position']) else .5) if following else -.5
            else:
                signal = 1 if rider['line_id'] in (records[a]['line_id'], records[b]['line_id']) and number(rider['line_position']) > 1 else -.5
        else:
            signal = -1 if c in leaders and not same else 1 if number(rider['line_position']) > 1 and same else 0
            if b is not None and rider['line_id'] == records[b]['line_id'] and number(rider['line_position']) > number(records[b]['line_position']):
                signal = 1
        return 1 + .15 * confidence * signal
    return multiplier, {**evidence, 'status': 'available', 'observed_samples': sample, 'competing_leaders': leaders,
                        'assumption': 'bounded prefix-dependent scenario hypothesis; no fitted causal probability'}
