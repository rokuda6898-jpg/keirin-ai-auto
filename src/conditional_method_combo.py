"""Read-only research transforms of a complete conditional three-place distribution.

No outcomes, observed odds, or hindsight decisions enter these transformations.
"""
from __future__ import annotations

from itertools import permutations
import math
import numpy as np

K_VALUES = (3, 6, 8, 10, 12, 18)
VARIANTS = (
    'conditional', 'rider_020', 'rider_035', 'rider_070',
    'rider_035_first_locked', 'rider_070_first_locked',
    'verified_line_035', 'archive_line_010', 'archive_line_030',
    'archive_line_030_first_locked', 'confidence_first_locked',
)


def validated(base, keys):
    arr = np.asarray(base, dtype=np.float64)
    if len(arr) != len(keys) or not len(arr) or not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError('nonpositive, incomplete or nonfinite conditional distribution')
    triples = [tuple(map(int, k.split('-'))) for k in keys]
    cars = sorted({c for t in triples for c in t})
    if len(triples) != len(set(triples)) or set(triples) != set(permutations(cars,3)):
        raise ValueError('not the complete ordered trifecta field')
    if not math.isclose(math.fsum(arr), 1.0, abs_tol=1e-8):
        raise ValueError('conditional probabilities not normalized')
    return arr, triples


def pair_adjust(base, triples, factors, alpha, *, preserve_first=False):
    """Adjust the 1->2 pair only; 3rd given a fixed pair is unchanged."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('bad preregistered exponent')
    p = np.asarray(base, dtype=np.float64)
    if alpha == 0:
        return p.copy()
    weighted = p * np.fromiter((float(factors[a,b])**alpha for a,b,c in triples),dtype=np.float64,count=len(triples))
    weighted /= weighted.sum()
    if preserve_first:
        first = np.fromiter((t[0] for t in triples), dtype=np.int32,count=len(triples))
        x = np.bincount(first,weights=p,minlength=10)
        y = np.bincount(first,weights=weighted,minlength=10)
        if np.any((x>0)&(y<=0)):
            raise ValueError('zero winner mass after correction')
        ratio = np.divide(x,y,out=np.ones_like(x),where=y>0)
        weighted *= ratio[first]
        weighted /= weighted.sum()
    if not np.isfinite(weighted).all() or (weighted <= 0).any() or not np.isclose(weighted.sum(),1.0,atol=1e-9):
        raise ValueError('adjustment broke the joint probability distribution')
    return weighted


def build_variants(base,keys,pure,verified_line,archive_line,*,alpha_high_conf=.20,alpha_low_conf=.70):
    base, triples = validated(base,keys)
    if set(pure) != set((a,b) for a,b,c in triples) or set(verified_line) != set(pure) or set(archive_line) != set(pure):
        raise ValueError('incomplete first/second candidate factors')
    if any(not math.isfinite(float(f)) or float(f)<=0 for collection in (pure,verified_line,archive_line) for f in collection.values()):
        raise ValueError('scenario factors must be finite positive numbers')
    marginal = np.bincount(np.fromiter((t[0] for t in triples),dtype=np.int32),weights=base,minlength=10)
    alpha = alpha_high_conf if marginal.max()>=.45 else alpha_low_conf
    result = {
        'conditional':base,
        'rider_020':pair_adjust(base,triples,pure,.20),
        'rider_035':pair_adjust(base,triples,pure,.35),
        'rider_070':pair_adjust(base,triples,pure,.70),
        'rider_035_first_locked':pair_adjust(base,triples,pure,.35,preserve_first=True),
        'rider_070_first_locked':pair_adjust(base,triples,pure,.70,preserve_first=True),
        'verified_line_035':pair_adjust(base,triples,verified_line,.35),
        'archive_line_010':pair_adjust(base,triples,archive_line,.35),
        'archive_line_030':pair_adjust(base,triples,archive_line,.35),
        'archive_line_030_first_locked':pair_adjust(base,triples,archive_line,.35,preserve_first=True),
        'confidence_first_locked':pair_adjust(base,triples,pure,alpha,preserve_first=True),
    }
    # archive_line_010 and archive_line_030 are supplied different factor families
    # by the runner and recomputed there; this helper is kept generic for tests.
    return result, {'top1_first_mass':float(marginal.max()),'adaptive_alpha':alpha}


def rerank_all(base,keys,factors_rider,factors_verified,factors_archive_010,factors_archive_030):
    """Predeclared 11-way study with separate historical-line strengths."""
    p,triples=validated(base,keys)
    first=np.fromiter((t[0] for t in triples),dtype=np.int32)
    confidence=float(np.bincount(first,weights=p,minlength=10).max())
    alpha=.20 if confidence>=.45 else .70
    families={
        'conditional':p,
        'rider_020':pair_adjust(p,triples,factors_rider,.20),
        'rider_035':pair_adjust(p,triples,factors_rider,.35),
        'rider_070':pair_adjust(p,triples,factors_rider,.70),
        'rider_035_first_locked':pair_adjust(p,triples,factors_rider,.35,preserve_first=True),
        'rider_070_first_locked':pair_adjust(p,triples,factors_rider,.70,preserve_first=True),
        'verified_line_035':pair_adjust(p,triples,factors_verified,.35),
        'archive_line_010':pair_adjust(p,triples,factors_archive_010,.35),
        'archive_line_030':pair_adjust(p,triples,factors_archive_030,.35),
        'archive_line_030_first_locked':pair_adjust(p,triples,factors_archive_030,.35,preserve_first=True),
        'confidence_first_locked':pair_adjust(p,triples,factors_rider,alpha,preserve_first=True),
    }
    if tuple(families)!=VARIANTS:raise ValueError('variant set changed without updating preregistration')
    return families,{'top1_first_mass':confidence,'adaptive_alpha':alpha}


def rank(keys, probs, k):
    return [keys[i] for i in sorted(range(len(keys)),key=lambda i:(-probs[i],tuple(map(int,keys[i].split('-')))))[:k]]


def select_value_candidates(keys,probs,odds,*,min_odds=50.0,limit=12):
    """Only valid full contemporaneous PRE-CLOSE quoted snapshots are admissible.

    A provenance/time gate must precede this function. Do not use settlement payouts.
    """
    if set(keys)!=set(odds) or any(not math.isfinite(float(v)) or v<=1 or v>=9999.9 for v in odds.values()):
        raise ValueError('partial/censored/nonpositive odds snapshot')
    return [k for k in sorted(keys,key=lambda k:(-probs[keys.index(k)]*float(odds[k]),-probs[keys.index(k)],k))
            if float(odds[k])>=min_odds][:limit]
