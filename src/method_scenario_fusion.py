"""Research-only first/second method scenario adjustment to a FULL trifecta distribution.

Observed past finishers' method labels are used exclusively to fit priors.
At prediction only the candidate riders' IDs, verified forecast lineup, and
race metadata are read. Heuristic lineup signals are NOT fitted probabilities.
No purchase or production integration is performed here.
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from datetime import date
from itertools import permutations
from pathlib import Path

from venue_time_learning import classify, venue_name

FIRST = ('逃', '捲', '差')
SECOND = ('逃', '捲', '差', 'マ')
PAIRS = tuple(a+b for a in FIRST for b in SECOND)
VERSION = 'method_scenario_rider_line_shadow_v1'


def _date(value):
    return date.fromisoformat(str(value)[:10])


def _distribution(counts, labels, prior=None, strength=1.):
    if prior is None:
        prior = {m: 1/len(labels) for m in labels}
    n = sum(counts.values())
    return {m: (counts[m] + strength*prior[m])/(n+strength) for m in labels}


def _valid_player(value):
    return value is not None and str(value).strip().lower() not in ('', 'none', 'nan')


def _verified_line(entrants):
    """Only forecast geometry verified for *every* active starter is usable."""
    grouping = defaultdict(list)
    for row in entrants:
        if row.get('line_verification_status') != 'verified':
            return None
        if not _valid_player(row.get('line_id')):
            return None
        try:
            pos = float(row['line_position'])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(pos) or pos != int(pos) or pos < 1:
            return None
        grouping[str(row['line_id'])].append((int(pos), int(row['car_no'])))
    if any(sorted(p for p,_ in members) != list(range(1,len(members)+1)) for members in grouping.values()):
        return None
    return {int(row['car_no']): (str(row['line_id']), int(float(row['line_position'])))
            for row in entrants}


def _line_signal(first_method, second_method, a, b, lines):
    """Explicit *uncalibrated* hypothesis score, not a scenario probability."""
    if lines is None:
        return 0.
    line_a, pos_a = lines[a]
    line_b, pos_b = lines[b]
    follows = line_a == line_b and pos_b == pos_a+1
    signal = 0.
    if first_method == '逃':
        signal += 0.8 if pos_a == 1 else -0.45
    elif first_method == '差':
        signal += 0.50 if pos_a >= 2 else -0.25
    elif first_method == '捲':
        signal += 0.25 if pos_a == 1 else 0.
    if second_method == 'マ':
        signal += 1. if follows else (-0.2 if line_a != line_b else 0.)
    elif second_method == '逃':
        signal += 0.3 if pos_b == 1 else -0.1
    return signal


class MethodScenarioFusion:
    """Frozen prior from all confirmed observations STRICTLY before cutoff."""

    def __init__(self, history, cutoff):
        self.cutoff = _date(cutoff)
        self.first_global = Counter()
        self.second_global = Counter()
        self.pair_global = Counter()
        self.pair_national = defaultdict(Counter)
        self.pair_venue = defaultdict(Counter)
        self.rider_first = defaultdict(Counter)
        self.rider_second = defaultdict(Counter)
        self.training_races = 0
        self.pair_races = 0
        seen = set()
        for row in history:
            if _date(row['date']) >= self.cutoff or row.get('method_status') != 'confirmed_by_podium_ids':
                continue
            rid = str(row['race_id'])
            if rid in seen:
                raise ValueError('duplicate official method race_id in training archive')
            seen.add(rid)
            first, second = row.get('winner_method'), row.get('second_method')
            first = first if first in FIRST else None
            second = second if second in SECOND else None
            if first is None and second is None:
                continue
            self.training_races += 1
            if first is not None:
                self.first_global[first] += 1
                pid = row.get('winner_player_id')
                if _valid_player(pid):
                    self.rider_first[str(pid)][first] += 1
            if second is not None:
                self.second_global[second] += 1
                pid = row.get('second_player_id')
                if _valid_player(pid):
                    self.rider_second[str(pid)][second] += 1
            if first is not None and second is not None:
                try:
                    field = int(float(row['field']))
                except (TypeError, ValueError):
                    continue
                pair = first+second
                national = (str(row['division']),field)
                venue = (venue_name(row['venue']),*national)
                self.pair_global[pair] += 1
                self.pair_national[national][pair] += 1
                self.pair_venue[venue][pair] += 1
                self.pair_races += 1
        if self.training_races < 100 or self.pair_races < 100:
            raise ValueError('insufficient confirmed, chronological method history')
        self.p_first = _distribution(self.first_global,FIRST)
        self.p_second = _distribution(self.second_global,SECOND)
        self.p_pair = _distribution(self.pair_global,PAIRS)

    @classmethod
    def from_csv(cls, file_path, cutoff):
        with Path(file_path).open(encoding='utf-8',newline='') as handle:
            return cls(csv.DictReader(handle),cutoff)

    def _prior(self, race):
        tier, division = classify(race)
        if tier == 'unknown':
            raise ValueError('unknown pre-race division; no fallback from results')
        national = (division,int(race['field']))
        venue = (venue_name(race['venue']),*national)
        p = _distribution(self.pair_national[national],PAIRS,self.p_pair,100.)
        return _distribution(self.pair_venue[venue],PAIRS,p,90.)

    def _rider(self, player_id, side):
        prior = self.p_first if side == 1 else self.p_second
        counts = self.rider_first.get(str(player_id),Counter()) if side == 1 else self.rider_second.get(str(player_id),Counter())
        # 20 prior podium observations protect scarce riders from large jumps.
        fitted = _distribution(counts,FIRST if side == 1 else SECOND,prior,20.)
        return {m: min(3.,max(1/3,fitted[m]/prior[m])) for m in prior}

    def pair_factors(self, entrants, race, *, line_weight=0.):
        if not math.isfinite(line_weight) or not 0 <= line_weight <= 1:
            raise ValueError('line weight must be a finite hypothesis coefficient in [0,1]')
        cars = [int(r['car_no']) for r in entrants]
        if not 3 <= len(cars) <= 9 or len(set(cars)) != len(cars) or any(not 1<=c<=9 for c in cars):
            raise ValueError('invalid complete active starters')
        if len({str(r.get('player_id')) for r in entrants}) != len(cars) or any(not _valid_player(r.get('player_id')) for r in entrants):
            raise ValueError('missing or duplicated entrant player IDs')
        if _date(race['date']) < self.cutoff:
            raise ValueError('held-out race predates model fit cutoff')
        inputs = {'venue':race['venue'], 'race_class':race['race_class'], 'race_type':race.get('race_type',''),
                  'is_grade_race':race.get('is_grade_race',False), 'field':len(cars)}
        prior = self._prior(inputs)
        lines = _verified_line(entrants)
        if lines is None:
            line_weight = 0.
        riders = {int(r['car_no']): r for r in entrants}
        p1 = {car:self._rider(row['player_id'],1) for car,row in riders.items()}
        p2 = {car:self._rider(row['player_id'],2) for car,row in riders.items()}
        factors = {}
        for a in cars:
            for b in cars:
                if a == b:
                    continue
                factors[a,b] = sum(prior[m+n]*p1[a][m]*p2[b][n]*
                                   math.exp(line_weight*_line_signal(m,n,a,b,lines))
                                   for m in FIRST for n in SECOND)
        return factors, {'line_verified':lines is not None,
                         'first_riders_with_history':sum(bool(self.rider_first.get(str(r['player_id']))) for r in entrants),
                         'second_riders_with_history':sum(bool(self.rider_second.get(str(r['player_id']))) for r in entrants)}

    def rerank(self, baseline, entrants, race, *, strength=.35, line_weight=0.):
        """Use strictly pre-race rows and all N*(N-1)*(N-2) ordered triples."""
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('strength must be finite in [0,1]')
        cars = sorted(int(r['car_no']) for r in entrants)
        expected = {'-'.join(map(str,k)) for k in permutations(cars,3)}
        if set(baseline) != expected or any(not math.isfinite(float(p)) or float(p)<=0 for p in baseline.values()):
            raise ValueError('baseline must have all strictly positive ordered triple probabilities')
        if not math.isclose(sum(map(float,baseline.values())),1.,rel_tol=0,abs_tol=1e-8):
            raise ValueError('baseline must already be normalized')
        factors, diagnostics = self.pair_factors(entrants,race,line_weight=line_weight)
        if strength == 0:
            return dict(baseline),diagnostics
        adjusted = {}
        for ticket,p in baseline.items():
            a,b,_ = map(int,ticket.split('-'))
            adjusted[ticket] = float(p)*factors[a,b]**strength
        total = sum(adjusted.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError('invalid scenario-weighted distribution')
        return {k:v/total for k,v in adjusted.items()},diagnostics
