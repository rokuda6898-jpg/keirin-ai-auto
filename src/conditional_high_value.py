"""Prospective high-odds *shadow* selector for the frozen conditional model.

Never replaces model weights or previously frozen live tickets. Only a complete,
same-snapshot trifecta quote can produce a selection. Score is a research proxy
from uncalibrated model probabilities, NOT validated expected return.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
import re

VERSION = 'conditional-50plus-overlay-v1'
MIN_ODDS = Decimal('50')
CENSORED_ODDS = Decimal('9999.9')
MIN_PROXY = 1.05
MAX_NEW_HIGH = 4


def _decimal(value):
    try:
        d = Decimal(str(value))
    except (TypeError, ValueError, InvalidOperation):
        raise ValueError('invalid market quote') from None
    if not d.is_finite() or d <= 0:
        raise ValueError('invalid market quote')
    return d


def unavailable(reason):
    return {'version': VERSION, 'status': 'unavailable',
            'reason': str(reason)[:240], 'tickets': [], 'new_high': []}


def select(keys, probabilities, baseline, quote, max_new_high=MAX_NEW_HIGH):
    """Replace up to four *lowest ranked* baseline tickets with new 50x+ picks.

    The fixed model's joint distribution stays unchanged. Capped prices are
    present for market completeness but never treated as executable quotes.
    Missing any candidate price is a data failure, not evidence of low value.
    """
    if (type(max_new_high) is not int or not 0 <= max_new_high <= 12
            or len(keys) != len(probabilities) or len(set(keys)) != len(keys)
            or not 24 <= len(keys) <= 504
            or len(baseline) != 12 or len(set(baseline)) != 12
            or not set(baseline).issubset(keys)):
        raise ValueError('invalid candidate set')
    if any(not isinstance(key, str) or
           not re.fullmatch(r'[1-9]-[1-9]-[1-9]', key) or
           len(set(key.split('-'))) != 3 for key in keys):
        raise ValueError('invalid trifecta combination')
    if (any(not math.isfinite(float(p)) or float(p) < 0 for p in probabilities)
            or not math.isclose(math.fsum(map(float, probabilities)), 1, abs_tol=1e-8)):
        raise ValueError('invalid model probability mass')
    probs = dict(zip(keys, map(float, probabilities)))
    ranked = sorted(keys, key=lambda key: (-probs[key], key))
    if baseline != ranked[:12]:
        raise ValueError('baseline must be the immutable model top12')
    if (not isinstance(quote, dict) or quote.get('bet_type') != 'trifecta'
            or not quote.get('captured_at_jst') or not quote.get('source_url')
            or not quote.get('provider') or not isinstance(quote.get('odds'), dict)):
        raise ValueError('unverified market source')
    market = quote['odds']
    if not set(keys).issubset(market):
        raise ValueError('full market is incomplete')
    prices = {key: _decimal(market[key]) for key in keys}
    if any(prices[key] >= CENSORED_ODDS for key in baseline):
        raise ValueError('baseline quote is capped')
    ranks = {key: index + 1 for index, key in enumerate(ranked)}
    candidates = []
    for key in keys:
        odds = prices[key]
        if key in baseline or odds < MIN_ODDS or odds >= CENSORED_ODDS:
            continue
        p = probs[key]
        proxy = p * float(odds)
        if proxy < MIN_PROXY:
            continue
        # Probability-regularized pricing signal: reduces tail-model noise
        # versus sorting by p*odds alone. This is NOT a calibrated EV.
        score = p ** 1.25 * float(odds)
        candidates.append((key, odds, p, proxy, score))
    candidates.sort(key=lambda item: (-item[4], -item[2], ranks[item[0]], item[0]))
    chosen, prefixes = [], Counter()
    for key, odds, p, proxy, score in candidates:
        prefix = tuple(key.split('-')[:2])
        if prefixes[prefix] >= 2:
            continue
        prefixes[prefix] += 1
        chosen.append({'buy': key, 'odds': str(odds),
                       'odds_band': '100x_plus' if odds >= 100 else '50_to_99',
                       'model_rank': ranks[key],
                       'model_probability': p,
                       'probability_x_odds_proxy': round(proxy, 6),
                       'ranking_signal': round(score, 8)})
        if len(chosen) >= max_new_high:
            break
    new_tickets = [item['buy'] for item in chosen]
    kept = baseline[:12 - len(new_tickets)]
    tickets = kept + new_tickets
    if len(tickets) != 12 or len(set(tickets)) != 12:
        raise ValueError('invalid shadow tickets')
    digest = sha256(json.dumps(
        [(key, str(prices[key])) for key in sorted(keys)],
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return {'version': VERSION, 'status': 'ready',
            'tickets': tickets, 'baseline12': baseline.copy(),
            'new_high': chosen, 'replaced': baseline[12 - len(new_tickets):],
            'eligible_high_count': len(candidates),
            'market_count': len(keys), 'priced_count': len(prices),
            'censored_count': sum(p >= CENSORED_ODDS for p in prices.values()),
            'market_sha256': digest,
            'source_url': quote['source_url'], 'provider': quote['provider'],
            'captured_at_jst': quote['captured_at_jst'],
            'selection_rule': '8-12 probability tickets + up to 4 new >=50x candidates; p*odds>=1.05; rank=p^1.25*odds; pair cap=2',
            'validated_roi': False}
