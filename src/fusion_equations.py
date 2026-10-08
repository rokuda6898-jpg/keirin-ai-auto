"""Research-only joint forecasts and chronological, duplicate-aware stacking.

No department is a voter. Every component is a distribution or an explicitly
observed scenario feature. Neither missing probabilities nor paths are invented.
Inference/report replay uses the standard library; fitting imports numpy lazily.
"""
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import permutations

from equation_models import fit as fit_legacy, predict as predict_legacy

DEPARTMENTS = ('data_department', 'pace_department', 'line_department', 'risk_department',
               'prediction_department', 'strategist_department', 'high_payout_department')
CONFIG = {'version': 'fusion_joint_v1', 'minimum_phase_races': 50,
          'minimum_phase_days': 7, 'maximum_training_races': 900,
          'lookback_days': 180, 'iterations': 100, 'learning_rate': .12,
          'ridge': .04, 'pool_iterations': 160, 'pool_redundancy': .08,
          'pool_prior_penalty': .02, 'temperatures': [.6, .8, 1., 1.25, 1.6, 2., 3.]}


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def softmax(values):
    high = max(values)
    weights = [math.exp(max(-700, x-high)) for x in values]
    total = sum(weights)
    return [x/total for x in weights]


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('fusion feature schema mismatch')
    return sum(x*y for x, y in zip(a, b))


def triples(packet):
    return list(permutations(sorted(int(r['car_no']) for r in
        packet['source']['evidence']['inputs']['risk_department']), 3))


def keys(packet):
    return ['-'.join(map(str, t)) for t in triples(packet)]


def valid_distribution(values, expected):
    return (isinstance(values, dict) and set(values) == set(expected)
            and all(finite(p) is not None and 0 < p <= 1 for p in values.values())
            and abs(sum(values.values())-1) < 1e-8)


def normalize(values):
    if not values or any(finite(v) is None or v < 0 for v in values.values()) or sum(values.values()) <= 0:
        return None
    total = sum(max(1e-12, v) for v in values.values())
    return {k: max(1e-12, v)/total for k,v in values.items()}


def position_distribution(rows, preserved=False):
    if not rows:
        return None
    maps = []
    for pos, name in enumerate(('p_win', 'p_second', 'p_third')):
        values = {}
        for r in rows:
            field = ('department_score_second' if pos == 1 else 'department_score_third') if preserved and pos else name
            value = finite(r.get(field))
            if value is None:
                return None
            values[int(r['car_no'])] = value
        normalized = normalize(values)
        if normalized is None:
            return None
        maps.append(normalized)
    result = {}
    for a,b,c in permutations(sorted(maps[0]), 3):
        result[f'{a}-{b}-{c}'] = maps[0][a] * maps[1][b]/sum(v for k,v in maps[1].items() if k != a) * maps[2][c]/sum(v for k,v in maps[2].items() if k not in (a,b))
    return result


def market(packet):
    prices = packet['source']['evidence']['usable_quotes']
    if (not packet['source']['quote_quality']['complete_single_snapshot'] or set(prices) != set(keys(packet)) or
            any(finite(v) is None or not 1 <= v < 9999.9 for v in prices.values())):
        return None
    return normalize({k: 1/v for k,v in prices.items()})


def static_experts(packet):
    """Reuse frozen specialist distributions, not their filtered ticket lists."""
    source = packet['source']
    expected = keys(packet)
    result = {}
    q = market(packet)
    if q:
        result['market:only'] = q
    raw = position_distribution(packet.get('auxiliary', {}).get('riders', []))
    if raw and valid_distribution(raw, expected):
        result['original:model_positions'] = raw
    for department, rows in source['evidence']['inputs'].items():
        p = position_distribution(rows, preserved=True)
        if p and valid_distribution(p, expected):
            result['positions:'+department] = p
    for name, arm in source.get('arms', {}).items():
        p = arm.get('distribution')
        if not valid_distribution(p, expected):
            continue
        department, variant = name.split(':')
        family = {'baseline':'legacy_positions', 'preserve':'positions',
                  'conditional':'market_conditionals', 'combined':'market_conditionals',
                  'context':'specialist_context', 'market':'market'}[variant]
        # Preserve and market duplicates are deliberately kept as aliases at fit.
        result[f'{family}:{department}:{variant}'] = p
    return result


def feature_context(packet):
    evidence = packet['source']['evidence']
    by_car = {int(r['car_no']): r for r in evidence['inputs']['risk_department']}
    individual, lines, roles = {}, {}, {}
    for car, r in by_car.items():
        f = []
        for name, scale in (('p_win', 1), ('score', 120), ('back_count', 20),
                            ('place2_rate', 1), ('place3_rate', 1), ('recent_avg_finish', 9)):
            value = finite(r.get(name))
            signal = 0 if value is None else max(-3, min(3, value/scale))
            if name == 'p_win' and value is not None:
                signal = math.log(max(1e-8, value))/10
            f.extend((signal, float(value is None)))
        verified = (r.get('line_verification_status') == 'verified'
                    and str(r.get('line_id', '')).lower() not in ('', 'none', 'nan')
                    and finite(r.get('line_position')) is not None
                    and float(r['line_position']) >= 1)
        lines[car] = str(r['line_id']) if verified else None
        roles[car] = int(r['line_position']) if verified else None
        f.extend((float(not verified), float(verified and roles[car] == 1),
                  float(verified and roles[car] == 2), float(verified and roles[car] >= 3)))
        individual[car] = f
    department_maps = {}
    for d in DEPARTMENTS[:4]:
        rows = evidence['inputs'].get(d, [])
        maps = []
        for field in ('p_win', 'department_score_second', 'department_score_third'):
            vals = {int(r['car_no']): finite(r.get(field)) for r in rows}
            maps.append(normalize(vals) if set(vals) == set(by_car) and all(v is not None for v in vals.values()) else None)
        department_maps[d] = maps
    return {'individual': individual, 'lines': lines, 'roles': roles,
            'department_maps': department_maps, 'scenarios': packet.get('auxiliary', {}).get('departments', {}),
            'market': market(packet), 'rows': by_car}


def joint_features(packet, triple, ctx=None):
    ctx = ctx or feature_context(packet)
    a,b,c = triple
    f = [x for car in triple for x in ctx['individual'][car]]
    for d in DEPARTMENTS[:4]:
        for pos, car in enumerate(triple):
            weights = ctx['department_maps'][d][pos]
            p = weights[car]/sum(v for k,v in weights.items() if k not in triple[:pos]) if weights else None
            f.extend((math.log(max(1e-12,p))/10 if p is not None else 0, float(p is None)))
    lines, roles = ctx['lines'], ctx['roles']
    def same(x,y): return lines[x] is not None and lines[x] == lines[y]
    def follows(x,y): return same(x,y) and roles[y] == roles[x]+1
    for x,y in ((a,b),(a,c),(b,c)):
        f.extend((float(lines[x] is None or lines[y] is None), float(same(x,y)),
                  float(follows(x,y)), float(follows(y,x))))
    fronts = [car for car in lines if roles[car] == 1]
    rival_pressure = sum((finite(ctx['rows'][car].get('back_count')) or 0)/20
                         for car in fronts if not same(a,car) and car != a)
    rival_pressure = min(3, rival_pressure)
    f.extend((float(follows(b,c) and same(a,b)),
              float(not same(a,b) and follows(a,c)), float(not same(a,b) and follows(b,c)),
              rival_pressure*float(roles[a] == 1), rival_pressure*float(roles[b] == 2),
              rival_pressure*float(roles[c] == 2)))
    for d in DEPARTMENTS:
        order = ctx['scenarios'].get(d, {}).get('top3_cars', [])
        known = len(order) == 3 and len(set(order)) == 3
        # A scenario is an observed opinion feature, never a fabricated probability.
        f.extend(float(known and int(order[pos]) == car) for pos,car in enumerate(triple))
        f.append(float(not known))
    # Field size / measured wind can change the effect of attacking/following.
    wind = finite(ctx['rows'][a].get('wind_speed'))
    f.extend((len(ctx['rows'])/9 * ctx['individual'][a][4],
              0 if wind is None else min(3,wind/10)*float(roles[a] == 1),
              0 if wind is None else min(3,wind/10)*float(roles[b] == 2),
              float(wind is None)*float(roles[a] == 1)))
    return f


def transition_features(previous, target):
    return [float(previous[i] == target[j]) for i in range(3) for j in range(3)]


def fit_linear(designs, targets):
    import numpy as np
    beta = np.zeros(len(designs[0][0]))
    matrices = [np.asarray(x, dtype=float) for x in designs]
    for _ in range(CONFIG['iterations']):
        grad = np.zeros_like(beta)
        for X, target in zip(matrices, targets):
            logits = X @ beta
            exp = np.exp(logits-logits.max())
            grad += X.T @ (exp/exp.sum()) - X[target]
        beta -= CONFIG['learning_rate']*(grad/len(matrices) + CONFIG['ridge']*beta)
    return beta.tolist()


def fit_base(samples, paths):
    designs, targets = [], []
    for s in samples:
        ts, ctx = triples(s), feature_context(s)
        designs.append([joint_features(s,t,ctx) for t in ts])
        targets.append(keys(s).index(s['actual']))
    complete = [s for s in samples if market(s)]
    legacy = fit_legacy([{'race_id':s['source']['race_id'], 'date':s['date'], 'actual':s['actual'],
        'riders':s['source']['evidence']['inputs']['risk_department'],
        'quotes':s['source']['evidence']['usable_quotes']} for s in complete], paths)
    path_map = {p['race_id']:p for p in paths}
    process_groups = {}
    for n in sorted({len(s['source']['evidence']['inputs']['risk_department']) for s in samples}):
        observed = [s for s in samples if len(s['source']['evidence']['inputs']['risk_department']) == n
                    and s['source']['race_id'] in path_map]
        if not enough(observed):
            continue
        coefficients = []
        for stage, name in enumerate(('start','bell','back','finish')):
            X, y = [], []
            for s in observed:
                ts, ctx = triples(s), feature_context(s)
                order = path_map[s['source']['race_id']]['orders']
                prev = order[('start','bell','back')[stage-1]] if stage else None
                X.append([joint_features(s,t,ctx)+(transition_features(prev,t) if prev else []) for t in ts])
                y.append(ts.index(tuple(order[name])))
            coefficients.append(fit_linear(X,y))
        process_groups[str(n)] = {'beta':coefficients, 'observed_races':len(observed)}
    return {'joint_beta':fit_linear(designs,targets), 'legacy':legacy,
            'process_groups':process_groups}


def predict_process(base, packet, features=None):
    ts = triples(packet)
    group = base['process_groups'].get(str(len(packet['source']['evidence']['inputs']['risk_department'])))
    if not group:
        return None
    ctx = feature_context(packet)
    features = features or [joint_features(packet,t,ctx) for t in ts]
    mass = softmax([dot(group['beta'][0],f) for f in features])
    for beta in group['beta'][1:]:
        static = [dot(beta[:-9],f) for f in features]
        nxt = [0.0]*len(ts)
        for prev,prob in zip(ts,mass):
            transition = softmax([score+dot(beta[-9:],transition_features(prev,t)) for score,t in zip(static,ts)])
            for i,p in enumerate(transition):
                nxt[i] += prob*p
        mass = nxt
    return dict(zip(keys(packet),mass))


def experts(base, packet):
    result = static_experts(packet)
    if base is None:
        return result
    ctx = feature_context(packet)
    features = [joint_features(packet,t,ctx) for t in triples(packet)]
    result['joint:prefix_context'] = dict(zip(keys(packet),softmax([dot(base['joint_beta'],f) for f in features])))
    if market(packet):
        for name, model in base['legacy']['models'].items():
            p = predict_legacy(name, model, packet['source']['evidence']['inputs']['risk_department'],
                               packet['source']['evidence']['usable_quotes'])
            if p:
                result[{'market_residual':'trained_market','pairwise_order':'pairwise','state_paths':'paths'}[name]+':'+name] = p
    process = predict_process(base,packet,features)
    if process:
        result['paths:conditional_process'] = process
    return result


def enough(samples):
    return len(samples) >= CONFIG['minimum_phase_races'] and len({s['date'] for s in samples}) >= CONFIG['minimum_phase_days']


def split_days(samples):
    days = sorted({s['date'] for s in samples})
    n = len(days)//3
    groups = (set(days[:len(days)-2*n]),set(days[len(days)-2*n:len(days)-n]),set(days[len(days)-n:])) if n else (set(),set(),set())
    return [[s for s in samples if s['date'] in days] for days in groups]


def cluster_values(pool, available):
    """Each duplicate cluster has one total weight, irrespective of alias count."""
    values = {}
    for leader, aliases in pool['aliases'].items():
        unique = {}
        for key in aliases:
            if key in available:
                p = available[key]
                signature = tuple((k,round(v,12)) for k,v in sorted(p.items()))
                unique.setdefault(signature,p)
        if unique:
            values[leader] = {k:sum(p[k] for p in unique.values())/len(unique) for k in next(iter(unique.values()))}
    return values


def mix(pool, available, excluded=()):
    available = {k:v for k,v in available.items() if k.split(':')[0] not in excluded}
    groups = cluster_values(pool,available)
    total = sum(pool['weights'][k] for k in groups)
    if total <= 0:
        return None
    return {t:sum(pool['weights'][k]*p[t] for k,p in groups.items())/total for t in next(iter(groups.values()))}


def fit_pool(records):
    import numpy as np
    names = sorted({k for _, es in records for k in es})
    signature_groups = defaultdict(list)
    for name in names:
        signature = [None if name not in es else sorted((k,round(p,12)) for k,p in es[name].items()) for _,es in records]
        digest = hashlib.sha256(json.dumps(signature,separators=(',',':')).encode()).hexdigest()
        signature_groups[digest].append(name)
    aliases = {group[0]:group for group in signature_groups.values()}
    names = sorted(aliases)
    # An alias with a different department/family name must not change the prior.
    prior = np.full(len(names), 1/len(names))
    p = np.array([[es[k].get(actual,1e-12) if k in es else 0 for k in names] for actual,es in records])
    present = (p > 0).astype(float)
    # Gram matrix of centered observed errors is PSD. Missing errors contribute
    # zero to redundancy estimation, not a fabricated forecast to the mixture.
    error = np.zeros_like(p)
    for j in range(len(names)):
        seen = present[:,j] > 0
        losses = -np.log(p[seen,j])
        error[seen,j] = losses-losses.mean()
    norms = np.sqrt((error*error).sum(axis=0))
    error /= np.maximum(norms,1e-12)
    correlation = error.T @ error
    logits = np.log(prior)
    for _ in range(CONFIG['pool_iterations']):
        w = np.exp(logits-logits.max());w /= w.sum()
        numerator,denominator = p@w,present@w
        grad = (present/denominator[:,None]-p/numerator[:,None]).mean(axis=0)
        grad += CONFIG['pool_redundancy']*(correlation@w)
        grad += CONFIG['pool_prior_penalty']*(np.log(w/prior)+1)
        logits -= .2*w*(grad-float(w@grad))
    w = np.exp(logits-logits.max());w /= w.sum()
    return {'weights':dict(zip(names,w.tolist())), 'aliases':aliases,
            'error_correlation':correlation.tolist(), 'correlation_order':names,
            'fitted_races':len(records)}


def temperature(distribution, value):
    if distribution is None:
        return None
    return dict(zip(distribution,softmax([math.log(p)/value for p in distribution.values()])))


def train(samples, paths=()):
    phases = split_days(samples)
    status = {'config':dict(CONFIG), 'phase_counts':[{'races':len(s),'days':len({r['date'] for r in s}),
        'first':min((r['date'] for r in s),default=None), 'last':max((r['date'] for r in s),default=None)} for s in phases],
        'model':None, 'reason':'insufficient_separate_base_stack_calibration_periods'}
    if not all(enough(s) for s in phases):
        return status
    base_samples,stack_samples,cal_samples = phases
    allowed = {s['source']['race_id'] for s in base_samples}
    base = fit_base(base_samples,[p for p in paths if p['race_id'] in allowed])
    pool = fit_pool([(s['actual'],experts(base,s)) for s in stack_samples])
    calibration = [(s['actual'],mix(pool,experts(base,s))) for s in cal_samples]
    loss = {tau:sum(-math.log(temperature(p,tau)[y]) for y,p in calibration)/len(calibration) for tau in CONFIG['temperatures']}
    tau = min(loss,key=lambda t:(loss[t],abs(t-1)))
    status.update(reason='fitted_research_only',model={'base':base,'pool':pool,'temperature':tau,
        'calibration_fit_losses':loss,'calibration_status':'fitted_on_separate_period_not_prospectively_validated'})
    return status


def infer(model, packet):
    if model is None:
        return {'distribution':None, 'components':static_experts(packet), 'ablations':{},
                'status':'training_required', 'effective_weights':{}, 'missing_families':
                ['trained_market','pairwise','paths','joint']}
    available = experts(model['base'],packet)
    distribution = temperature(mix(model['pool'],available),model['temperature'])
    groups = cluster_values(model['pool'],available)
    total = sum(model['pool']['weights'][k] for k in groups)
    weights = {k:model['pool']['weights'][k]/total for k in groups}
    removals = {'without_departments':('positions','legacy_positions','specialist_context','market_conditionals'),
                'without_market':('market','trained_market','market_conditionals'),
                'without_pairwise':('pairwise',), 'without_paths':('paths',),
                'without_joint_context':('joint','specialist_context')}
    ablations = {name:temperature(mix(model['pool'],available,families),model['temperature']) for name,families in removals.items()}
    return {'distribution':distribution,'components':available,'ablations':ablations,
        'ablation_scope':'remove_components_without_refit; not a causal or complete feature ablation',
        'status':'research_only', 'effective_weights':weights,
        'missing_families':[f for f in ('trained_market','pairwise','paths','joint') if not any(k.startswith(f+':') for k in available)]}


def probability_scores(distribution, actual):
    """A proper-score decomposition: NLL = first + second|first + third|pair."""
    a,b,c = map(int,actual.split('-'))
    first = sum(p for t,p in distribution.items() if int(t.split('-')[0]) == a)
    pair = sum(p for t,p in distribution.items() if tuple(map(int,t.split('-')[:2])) == (a,b))
    exact = distribution[actual]
    return {'nll':-math.log(exact),'first_nll':-math.log(first),
            'second_given_first_nll':-math.log(pair/first), 'third_given_pair_nll':-math.log(exact/pair),
            'brier':sum((p-float(t==actual))**2 for t,p in distribution.items())}
