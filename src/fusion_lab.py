"""Append-only prospective fusion research. No authority over CEO or purchases."""
import argparse
import hashlib
import html
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import fusion_equations as eq
from department_experiment_v2 import (LEDGER as SOURCES, OUTCOMES, append, digest,
    frame_records, read_lines, valid_record, seal, code_provenance)
from equation_lab import PATHS, checked, performance
from official_outcomes import ticket_return

ROOT = Path(__file__).resolve().parents[1]
JST = ZoneInfo('Asia/Tokyo')
INPUTS = 'annual_fusion_inputs.jsonl'
MODELS = 'annual_fusion_models.jsonl'
LEDGER = 'annual_fusion_ledger.jsonl'
ATTEMPTS = 'annual_fusion_attempts.jsonl'
TRIALS = 'annual_fusion_confirmations.jsonl'
FILES = (INPUTS, MODELS, LEDGER, ATTEMPTS, TRIALS)
RULES = {'version':'fusion_research_v1','maximum_tickets':12,'stake_yen':100,
         'minimum_hole_odds':100,'minimum_candidate_probability':.001,
         'selection':'probability_ranking_research_only_no_EV_purchase_gate',
         'capture_minutes':[5,40],'quote_age_seconds':300,
         'automatic_promotion':False,'ceo_integration':False,'purchase_authorized':False,
         'confirmation_days':56,'minimum_confirmation_pairs':500,'minimum_confirmation_days':28,
         'bootstrap_block_days':7,'bootstrap_resamples':12000,'lifetime_alpha':.05}
GROUPS = ('本線','穴')
METHODS = ('fusion','market','original','joint','market_residual','pairwise_order',
           'state_paths','conditional_process','without_departments','without_market',
           'without_pairwise','without_paths','without_joint_context')
LABELS = {'fusion':'統合版','market':'市場支持のみ','original':'元の着順モデル','joint':'部署・組み合わせモデル',
          'market_residual':'初期市場補正式','pairwise_order':'初期先着関係式',
          'state_paths':'初期隊列頻度モデル','conditional_process':'条件付き隊列モデル',
          'without_departments':'部署成分を除外','without_market':'市場成分を除外',
          'without_pairwise':'先着成分を除外','without_paths':'隊列成分を除外',
          'without_joint_context':'組み合わせ成分を除外'}


def clock():
    return datetime.now(JST)


def code_id():
    names = ('fusion_equations.py','fusion_lab.py','equation_models.py','department_experiment_v2.py',
             'department_ticket_v2.py','department_context.py','department_coverage.py','official_outcomes.py')
    return digest({'rules':RULES,'config':eq.CONFIG,'sources':{
        n:hashlib.sha256(Path(__file__).with_name(n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in names}})


def source_map(folder):
    result = {}
    for r in read_lines(folder/SOURCES):
        if not valid_record(r):
            raise ValueError('invalid canonical source')
        result[r['record_sha256']] = r
    return result


def time_ok(source, now):
    return (now.tzinfo is not None and 300 < source['close_at']-now.timestamp() <= 2400
            and 0 <= (now-datetime.fromisoformat(source['snapshot_at'])).total_seconds() <= 300
            and all(0 <= now.timestamp()-datetime.fromisoformat(t).timestamp() <= 300
                    for t in source['evidence']['quote_times'].values()))


def packets(folder):
    sources = source_map(folder)
    result = []
    for row in checked(folder/INPUTS):
        source = sources.get(row['source_record'])
        stamp = datetime.fromisoformat(row['captured_at'])
        if (source is None or not time_ok(source,stamp) or row['race_id'] != source['race_id']
                or row['code'] is None or row['purchase_authorized'] is not False):
            raise ValueError('invalid fusion input provenance')
        aux = row['auxiliary']
        cars = {int(r['car_no']) for r in source['evidence']['inputs']['risk_department']}
        if len(aux['riders']) != len(cars) or {int(r['car_no']) for r in aux['riders']} != cars:
            raise ValueError('auxiliary rider mismatch')
        for rider in aux['riders']:
            if rider.get('finish_pos') is not None or rider.get('official_finish_pos') is not None or rider.get('result_available') not in (None,0,False):
                raise ValueError('result data in fusion input')
        if set(aux['departments']) != set(eq.DEPARTMENTS):
            raise ValueError('missing department availability record')
        for d, item in aux['departments'].items():
            order = item['top3_cars']
            if order and (len(order) != 3 or len(set(order)) != 3 or not set(order) <= cars):
                raise ValueError('invalid department scenario')
            if order:
                opinion_at = datetime.fromisoformat(item['snapshot_at'])
                if (opinion_at.tzinfo is None or not 0 <= (stamp-opinion_at).total_seconds() <= 300
                        or item['availability'] != 'observed_current_cycle'):
                    raise ValueError('stale or future department scenario')
        result.append({**row,'source':source})
    return result


def outcomes_asof(folder, cutoff):
    found, conflicts = {}, set()
    for r in checked(folder/OUTCOMES):
        stamp = datetime.fromisoformat(r['observed_at'])
        if stamp.tzinfo is None:
            raise ValueError('outcome observation has no timezone')
        if stamp >= cutoff:
            continue
        rid, outcome = r['race_id'],r['outcome']
        # A subsequently completed payout may enrich an unchanged winning order.
        if rid in found:
            previous = found[rid]['outcome']
            if (set(previous['winning_buys']) != set(outcome['winning_buys']) or
                    any(k in previous['payouts'] and not math.isclose(
                            float(previous['payouts'][k]), float(v), rel_tol=1e-12, abs_tol=1e-9)
                        for k,v in outcome['payouts'].items())):
                conflicts.add(rid)
        if rid not in found or r['observed_at'] > found[rid]['observed_at']:
            found[rid] = r
    return {rid:r for rid,r in found.items() if rid not in conflicts}, sorted(conflicts)


def prior_packets(output_dir, now):
    folder = Path(output_dir)/'company'
    cutoff = datetime.combine(now.astimezone(JST).date(),datetime.min.time(),JST)
    lower = cutoff-timedelta(days=eq.CONFIG['lookback_days'])
    latest = {}
    for p in packets(folder):
        stamp = datetime.fromisoformat(p['captured_at'])
        rid = p['race_id']
        if lower <= stamp < cutoff and p['source']['close_at'] < cutoff.timestamp():
            if rid not in latest or p['captured_at'] > latest[rid]['captured_at']:
                latest[rid] = p
    outcomes, conflicts = outcomes_asof(folder,cutoff)
    samples = []
    for rid,p in latest.items():
        observed = outcomes.get(rid)
        if observed is None or datetime.fromisoformat(observed['observed_at']).timestamp() < p['source']['close_at']:
            continue
        winning = observed['outcome']['winning_buys']
        if len(winning) != 1 or winning[0] not in eq.keys(p):
            continue
        samples.append({**p,'date':datetime.fromisoformat(p['captured_at']).astimezone(JST).date().isoformat(),
                        'actual':winning[0],'outcome_record':observed['record_sha256']})
    samples = sorted(samples,key=lambda s:(s['date'],s['race_id']))[-eq.CONFIG['maximum_training_races']:]
    by_id = {p['race_id']:p for p in samples}
    paths, raw_paths, bad_paths = [], {}, set()
    for r in read_lines(folder/PATHS):
        if not r:
            continue
        rid = r.get('race_id')
        if rid in raw_paths and digest(raw_paths[rid]) != digest(r):
            bad_paths.add(rid)
        raw_paths[rid] = r
    for rid,r in raw_paths.items():
        try:
            s = by_id[rid]
            stamp = datetime.fromisoformat(r['observed_at'])
            orders = [r['orders'][name] for name in ('start','bell','back','finish')]
            cars = set(t[0] for t in eq.triples(s))
            if (rid in bad_paths or r.get('verified') is not True or r.get('schema') != 'observed_top3_stages_v1'
                    or not str(r.get('source_url','')).startswith('https://') or stamp.tzinfo is None
                    or not s['source']['close_at'] < stamp.timestamp() < cutoff.timestamp()
                    or any(not isinstance(o,list) or len(o)!=3 or len(set(o))!=3 or not set(o)<=cars for o in orders)
                    or '-'.join(map(str,orders[-1])) != s['actual']):
                continue
            paths.append(r)
        except (ValueError,KeyError,TypeError):
            continue
    return samples,paths,{'conflicting_results':conflicts,'conflicting_paths':sorted(bad_paths)}


def prepare_model(output_dir, now=None):
    now = now or clock()
    started = time.monotonic()
    folder = Path(output_dir)/'company'
    code,day = code_id(),now.astimezone(JST).date().isoformat()
    existing = [m for m in checked(folder/MODELS) if m['code']==code and m['cutoff_exclusive']==day]
    if existing:
        return min(existing,key=lambda m:(m['created_at'],m['record_sha256']))
    samples,paths,diagnostics = prior_packets(output_dir,now)
    fitted = eq.train(samples,paths)
    model = seal({'code':code,'cutoff_exclusive':day,'created_at':(now+timedelta(seconds=time.monotonic()-started)).isoformat(),
        'training':fitted,'training_manifest':[{'input_record':s['record_sha256'],'outcome_record':s['outcome_record'],
            'date':s['date'],'race_id':s['race_id']} for s in samples],
        'observed_path_hashes':[digest(p) for p in paths],'diagnostics':diagnostics,
        'purchase_authorized':False,'ceo_integration':False})
    append(folder/MODELS,model)
    return model


def distributions(forecast):
    components = forecast['components']
    mapping = {'market':'market:only','original':'original:model_positions','joint':'joint:prefix_context',
        'market_residual':'trained_market:market_residual','pairwise_order':'pairwise:pairwise_order',
        'state_paths':'paths:state_paths','conditional_process':'paths:conditional_process'}
    return {'fusion':forecast['distribution'],**{k:components.get(v) for k,v in mapping.items()},
            **{k:forecast['ablations'].get(k) for k in METHODS if k.startswith('without_')}}


def choose(p, source, group, count):
    if not 0 < count <= 12:
        return None,'risk_abstained_or_invalid_count'
    if p is None:
        return None,'model_unavailable'
    prices = source['evidence']['usable_quotes']
    candidates = [{'buy':k,'prob':prob,'odds':prices[k],'group':group,'stake_yen':100}
                  for k,prob in p.items() if k in prices and prob >= RULES['minimum_candidate_probability']
                  and ('穴' if prices[k] >= 100 else '本線') == group]
    candidates.sort(key=lambda t:(-t['prob'],t['buy']))
    if len(candidates) < count:
        return None,'insufficient_priced_probability_candidates'
    return candidates[:count],'matched'


def make_views(forecast,source):
    views,reasons = {},{}
    for group in GROUPS:
        risk = [t for t in source['risk_tickets'] if t['group']==group]
        good = (0 < len(risk) <= 12 and all(t['stake_yen']==100 and t['odds']==source['evidence']['usable_quotes'].get(t['buy']) for t in risk))
        views[group],reasons[group] = {'risk':risk if good else None},{}
        for method,p in distributions(forecast).items():
            selected,reason = choose(p,source,group,len(risk) if good else 0)
            views[group][method],reasons[group][method] = selected,reason
    return views,reasons


def capture_packet(packet, output_dir, now=None):
    now = now or clock()
    started = time.monotonic()
    source = packet['source']
    if not time_ok(source,now):
        return 'outside_fresh_capture_window'
    folder = Path(output_dir)/'company'
    input_row = seal({'source_record':source['record_sha256'],'race_id':source['race_id'],
        'captured_at':now.isoformat(),'auxiliary':packet['auxiliary'],'code':code_id(), 'purchase_authorized':False})
    # Inputs, including explicit missing opinions, are retained even if fitting
    # times out. They are not a retroactively generated forecast.
    append(folder/INPUTS,input_row)
    packet = {**input_row,'source':source}
    model = prepare_model(output_dir,now)
    forecast = eq.infer(model['training']['model'],packet)
    completed = now+timedelta(seconds=time.monotonic()-started)
    if not time_ok(source,completed) or completed.astimezone(JST).date().isoformat()!=model['cutoff_exclusive']:
        return 'expired_during_fusion_calculation'
    views,reasons = make_views(forecast,source)
    append(folder/LEDGER,{'input_record':input_row['record_sha256'],'model_record':model['record_sha256'],
        'snapshot_at':completed.isoformat(),'race_id':source['race_id'], 'code':model['code'],
        'cohort':digest([model['code'],source['code']['sha256']]),'rules':RULES,
        'forecast':forecast,'views':views,'selection_reasons':reasons,
        'purchase_authorized':False,'ceo_integration':False})
    return 'captured_fusion' if forecast['distribution'] else 'captured_components_waiting_for_training'


def capture_cycle(pred, department_report, cycle_started, output_dir=ROOT/'outputs', now=None):
    """Only called by predict, after all seven current-cycle opinions exist."""
    folder = Path(output_dir)/'company'
    sources = {}
    for r in source_map(folder).values():
        if datetime.fromisoformat(r['snapshot_at']) >= cycle_started:
            rid=r['race_id']
            if rid not in sources or r['snapshot_at'] > sources[rid]['snapshot_at']:
                sources[rid]=r
    reports = defaultdict(dict)
    for r in department_report['predictions']:
        reports[str(r['race_id'])][r['department']] = r
    outcomes = []
    for rid,race in pred.groupby('race_id',sort=False):
        rid=str(rid)
        source=sources.get(rid)
        stamp=clock() if now is None else now
        if source is None or not time_ok(source,stamp):
            append(folder/ATTEMPTS,{'race_id':rid,'at':stamp.isoformat(),
                'reason':'no_fresh_current_cycle_source','code':code_id()})
            continue
        aux={'riders':frame_records(race),'departments':{}}
        for d in eq.DEPARTMENTS:
            opinion=reports[rid].get(d,{})
            order=opinion.get('top3_cars',[]) if opinion.get('forecast_available') else []
            opinion_stamp=datetime.fromisoformat(opinion['snapshot_at']) if opinion.get('snapshot_at') else None
            fresh = (opinion_stamp is not None and opinion_stamp.tzinfo is not None
                     and opinion_stamp >= cycle_started.replace(microsecond=0)
                     and 0 <= (stamp-opinion_stamp).total_seconds() <= 300)
            aux['departments'][d]={'top3_cars':order if fresh else [],
                'source':opinion.get('display_status','not_submitted'),
                'snapshot_at':opinion.get('snapshot_at'),'availability':'observed_current_cycle' if fresh and order else 'missing_current_cycle'}
        try:
            reason=capture_packet({'source':source,'auxiliary':aux},output_dir,stamp)
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            reason='fusion_unavailable:'+type(exc).__name__
        append(folder/ATTEMPTS,{'race_id':rid,'at':stamp.isoformat(),'reason':reason,'code':code_id()})
        outcomes.append({'race_id':rid,'reason':reason})
    return outcomes


def validated_rows(folder):
    inputs={p['record_sha256']:p for p in packets(folder)}
    models={m['record_sha256']:m for m in checked(folder/MODELS)}
    outcome_hashes={r['record_sha256']:r for r in checked(folder/OUTCOMES)}
    for model in models.values():
        cutoff=datetime.fromisoformat(model['cutoff_exclusive']).replace(tzinfo=JST)
        if (model['purchase_authorized'] is not False or model['ceo_integration'] is not False
                or datetime.fromisoformat(model['created_at']) < cutoff):
            raise ValueError('invalid fusion model provenance')
        for item in model['training_manifest']:
            p=inputs.get(item['input_record']);o=outcome_hashes.get(item['outcome_record'])
            if (p is None or o is None or p['race_id'] != item['race_id'] or o['race_id'] != item['race_id']
                    or datetime.fromisoformat(p['captured_at']) >= cutoff
                    or not p['source']['close_at'] <= datetime.fromisoformat(o['observed_at']).timestamp() < cutoff.timestamp()
                    or item['date'] != datetime.fromisoformat(p['captured_at']).astimezone(JST).date().isoformat()):
                raise ValueError('fusion training manifest is not as-of evidence')
        path_records = {digest(p):p for p in read_lines(folder/PATHS) if p}
        for path_hash in model['observed_path_hashes']:
            path = path_records.get(path_hash)
            if path is None or datetime.fromisoformat(path['observed_at']) >= cutoff:
                raise ValueError('fusion path manifest is not as-of evidence')
    rows=[]
    for row in checked(folder/LEDGER):
        p=inputs.get(row['input_record']);model=models.get(row['model_record'])
        stamp=datetime.fromisoformat(row['snapshot_at'])
        if (p is None or model is None or not time_ok(p['source'],stamp) or row['rules']!=RULES
                or row['purchase_authorized'] is not False or row['ceo_integration'] is not False
                or row['race_id']!=p['race_id'] or row['code']!=model['code']
                or row['code']!=p['code'] or datetime.fromisoformat(p['captured_at']) > stamp
                or row['cohort']!=digest([row['code'],p['source']['code']['sha256']])
                or datetime.fromisoformat(model['created_at']) > stamp
                or model['cutoff_exclusive']!=stamp.astimezone(JST).date().isoformat()):
            raise ValueError('invalid fusion forecast provenance')
        expected=eq.infer(model['training']['model'],p)
        # Pure-Python deterministic inference reproduces the actual frozen model.
        if digest(expected)!=digest(row['forecast']):
            raise ValueError('fusion forecast differs from frozen model')
        views,reasons=make_views(expected,p['source'])
        if row['views']!=views or row['selection_reasons']!=reasons:
            raise ValueError('risk, portfolio or exclusion reason was changed')
        rows.append({**row,'packet':p,'date':stamp.astimezone(JST).date().isoformat(),
                     'close_at':p['source']['close_at']})
    return rows,models


def register_confirmation(output_dir, starts_at, target_hit_delta, now=None):
    now=now or clock()
    folder=Path(output_dir)/'company'
    if (starts_at.tzinfo is None or starts_at <= now or not 0 < target_hit_delta < 1
            or starts_at.astimezone(JST).time() != datetime.min.time()):
        raise ValueError('confirmation requires a future start and an explicit positive improvement target')
    starts_at=starts_at.astimezone(JST)
    trained=[m for m in checked(folder/MODELS) if m['code']==code_id() and m['training']['model'] is not None]
    if not trained:
        raise ValueError('fit_and_review_before_registering_confirmation')
    existing=list(checked(folder/TRIALS))
    if any(datetime.fromisoformat(t['ends_at']) > now for t in existing):
        raise ValueError('an earlier confirmation is still open')
    attempt=len(existing)+1
    trial={'registered_at':now.isoformat(),'starts_at':starts_at.isoformat(),
        'ends_at':(starts_at+timedelta(days=RULES['confirmation_days'])).isoformat(),
        'code':code_id(),'attempt':attempt,'family_alpha':RULES['lifetime_alpha']/(attempt*(attempt+1)),
        'target_hit_delta':target_hit_delta,'method':'fusion','source_code':code_provenance()['sha256'],
        'automatic_promotion':False,'ceo_integration':False}
    append(folder/TRIALS,trial)
    return trial


def confirmation(samples, trial, group, now):
    if trial['code']!=code_id() or trial['source_code']!=code_provenance()['sha256']:
        return {'status':'procedure_changed_requires_new_future_registration'}
    if now < datetime.fromisoformat(trial['ends_at']):
        return {'status':'future_confirmation_in_progress'}
    start,end=map(datetime.fromisoformat,(trial['starts_at'],trial['ends_at']))
    rows=[r for r in samples if start <= datetime.fromisoformat(r['snapshot_at']) < end]
    days=defaultdict(list)
    for row in rows:
        a,av=ticket_return(row['views'][group]['fusion'],row['outcome'])
        b,bv=ticket_return(row['views'][group]['risk'],row['outcome'])
        days[row['date']].append((int(a)-int(b),av-bv,len(row['views'][group]['risk'])*100))
    if len(rows)<RULES['minimum_confirmation_pairs'] or len(days)<RULES['minimum_confirmation_days']:
        return {'status':'insufficient_at_fixed_deadline','pairs':len(rows),'days':len(days)}
    calendar=[]
    for i in range(RULES['confirmation_days']):
        values=days[(start+timedelta(days=i)).date().isoformat()]
        calendar.append((len(values),sum(v[0] for v in values),sum(v[1] for v in values),sum(v[2] for v in values)))
    rng=random.Random(20261009);hit=[];roi=[]
    width=RULES['bootstrap_block_days']
    for _ in range(RULES['bootstrap_resamples']):
        draw=[]
        while len(draw)<len(calendar):
            i=rng.randrange(len(calendar)-width+1);draw.extend(calendar[i:i+width])
        draw=draw[:len(calendar)]
        count=sum(v[0] for v in draw);stake=sum(v[3] for v in draw)
        if count and stake:
            hit.append(sum(v[1] for v in draw)/count);roi.append(sum(v[2] for v in draw)/stake)
    alpha=trial['family_alpha']/4  # two ticket groups x hit/ROI, across all registered attempts
    if len(hit)*alpha/2 < 20:
        return {'status':'insufficient_bootstrap_tail_resolution','automatic_promotion':False}
    def interval(xs):
        xs.sort();return [xs[int(len(xs)*alpha/2)],xs[min(len(xs)-1,int(len(xs)*(1-alpha/2)))]]
    hc,rc=interval(hit),interval(roi)
    return {'status':'human_review_only','hit_difference_interval':hc,'return_difference_interval':rc,
        'target_hit_delta':trial['target_hit_delta'],'practical_target_supported':hc[0]>trial['target_hit_delta'],
        'ROI_noninferiority_supported':rc[0]>=0,'method':'approximate_7_day_moving_block_bootstrap',
        'minimum_count_is_not_a_power_guarantee':True,'automatic_promotion':False,'ceo_integration':False}


def confirmation_records(folder):
    trials=list(checked(folder/TRIALS))
    for i,trial in enumerate(trials,1):
        registered,start,end=map(datetime.fromisoformat,(trial['registered_at'],trial['starts_at'],trial['ends_at']))
        if (any(t.tzinfo is None for t in (registered,start,end)) or not registered < start
                or start.astimezone(JST).time()!=datetime.min.time()
                or end-start!=timedelta(days=RULES['confirmation_days']) or trial['attempt']!=i
                or trial['family_alpha']!=RULES['lifetime_alpha']/(i*(i+1))
                or not 0 < trial['target_hit_delta'] < 1 or trial['method']!='fusion'
                or trial['automatic_promotion'] is not False or trial['ceo_integration'] is not False):
            raise ValueError('invalid future confirmation registration')
    return trials


def score_summary(rows,method):
    metrics=[eq.probability_scores(distributions(r['forecast'])[method],r['outcome']['winning_buys'][0]) for r in rows]
    means={k:sum(m[k] for m in metrics)/len(metrics) for k in metrics[0]} if metrics else {}
    bins=defaultdict(lambda:[0,0.,0])
    for row in rows:
        p=distributions(row['forecast'])[method];top=max(p,key=p.get);prob=p[top]
        edge=next(x for x in (.01,.025,.05,.1,.2,.4,1.) if prob<=x)
        b=bins[str(edge)];b[0]+=1;b[1]+=prob;b[2]+=top==row['outcome']['winning_buys'][0]
    return {'races':len(rows),**means,'top_ticket_reliability':[
        {'upper':k,'races':v[0],'mean_prediction':v[1]/v[0],'observed_hit_rate':v[2]/v[0]} for k,v in sorted(bins.items())]}


def build_report(output_dir=ROOT/'outputs',now=None):
    now=now or clock();folder=Path(output_dir)/'company';folder.mkdir(parents=True,exist_ok=True)
    all_packets=packets(folder);rows,models=validated_rows(folder)
    forecast_snapshots=len(rows)
    sources=source_map(folder)
    latest_sources={}
    for source in sources.values():
        key=(source['code']['sha256'],source['race_id'])
        if key not in latest_sources or datetime.fromisoformat(source['snapshot_at']) > datetime.fromisoformat(latest_sources[key]['snapshot_at']):
            latest_sources[key]=source
    outcomes,conflicts=outcomes_asof(folder,now)
    if conflicts:
        raise ValueError('official outcomes conflict; fusion evaluation stopped')
    latest={}
    for r in rows:
        key=(r['cohort'],r['race_id'])
        if key not in latest or r['snapshot_at']>latest[key]['snapshot_at']:
            latest[key]=r
    latest_rows=list(latest.values())
    rows=[r for r in latest_rows if r['packet']['source_record']==latest_sources[
        (r['packet']['source']['code']['sha256'],r['race_id'])]['record_sha256']]
    # The final scheduled source snapshot is chosen BEFORE checking method
    # availability. A missing method is not silently replaced by its older pick.
    settled=[{**r,'outcome':outcomes[r['race_id']]['outcome']} for r in rows
             if r['race_id'] in outcomes and r['close_at']<now.timestamp()
             and datetime.fromisoformat(outcomes[r['race_id']]['observed_at']).timestamp() >= r['close_at']]
    comparisons=[];proper=[];cohorts=sorted({r['cohort'] for r in rows})
    for cohort in cohorts:
        available=[r for r in settled if r['cohort']==cohort]
        for method in METHODS:
            for group in GROUPS:
                paired=[r for r in available if r['views'][group]['risk'] and r['views'][group][method]
                        and r['outcome']['payout_complete']]
                if not paired:
                    continue
                errors=Counter()
                for r in paired:
                    actual=r['outcome']['winning_buys'];tickets=r['views'][group][method]
                    if len(actual)!=1:
                        errors['multiple_winners']+=1;continue
                    a,b,c=actual[0].split('-');buys=[t['buy'].split('-') for t in tickets]
                    reason=('hit' if [a,b,c] in buys else 'third_only_missed' if any(t[:2]==[a,b] for t in buys)
                            else 'second_given_first_missed' if any(t[0]==a for t in buys) else 'winner_missed')
                    errors[reason]+=1
                comparisons.append({'cohort':cohort,'method':method,'group':group,'status':'exploratory_not_adoption_evidence',
                    'race_ids':[r['race_id'] for r in paired], 'challenger':performance(paired,method,group),
                    'risk':performance(paired,'risk',group),'error_locations':dict(errors),
                    'forecast_coverage_mass_mean':sum(sum(t['prob'] for t in r['views'][group][method]) for r in paired)/len(paired)})
            for baseline in ('market','original'):
                if method==baseline:
                    continue
                paired=[r for r in available if len(r['outcome']['winning_buys'])==1
                        and distributions(r['forecast'])[method] and distributions(r['forecast'])[baseline]
                        and r['outcome']['winning_buys'][0] in distributions(r['forecast'])[method]]
                if paired:
                    proper.append({'cohort':cohort,'method':method,'baseline':baseline,
                        'scope':'all_forecasts_including_risk_abstentions_same_snapshot',
                        'race_ids':[r['race_id'] for r in paired],'challenger':score_summary(paired,method),
                        'reference':score_summary(paired,baseline)})
    missing=Counter();strata=Counter()
    for p in all_packets:
        s=p['source'];strata[(len(s['evidence']['inputs']['risk_department']),s['strata']['race_class'],
                            s['quote_quality']['complete_single_snapshot'])]+=1
        for d,opinion in p['auxiliary']['departments'].items():
            if not opinion['top3_cars']:missing[d]+=1
    trials=confirmation_records(folder);confirmed=[]
    for trial in trials:
        for group in GROUPS:
            paired=[r for r in settled if r['code']==trial['code']
                    and r['packet']['source']['code']['sha256']==trial['source_code'] and r['views'][group]['risk']
                    and r['views'][group]['fusion'] and r['outcome']['payout_complete']]
            confirmed.append({'trial':trial['record_sha256'],'group':group,**confirmation(paired,trial,group,now)})
    samples,paths,diagnostics=prior_packets(output_dir,now)
    current_models=[m for m in models.values() if m['code']==code_id()]
    latest_model=max(current_models,key=lambda m:m['created_at']) if current_models else None
    captured_sources={p['source_record'] for p in all_packets}
    report={'version':RULES['version'],'updated_at':now.isoformat(),'rules':RULES,'training_config':eq.CONFIG,
        'code':code_id(),'methods':LABELS,'automatic_promotion':False,'ceo_integration':False,'purchase_authorized':False,
        'prior_training_races':len(samples),'prior_training_days':len({s['date'] for s in samples}),
        'verified_path_annotations':len(paths),'latest_model':latest_model,
        'input_snapshots':len(all_packets),'forecast_snapshots':forecast_snapshots,
        'latest_race_cohort_forecasts':len(rows),'unique_races':len({r['race_id'] for r in rows}),
        'excluded_older_forecasts_after_missing_latest_capture':len(latest_rows)-len(rows),
        'canonical_source_snapshots':len(sources),'sources_without_fusion_input':sum(k not in captured_sources for k in sources),
        'collection_attempt_reasons':dict(Counter(r['reason'] for r in checked(folder/ATTEMPTS))),
        'missing_department_opinions':dict(missing),'snapshot_strata':[
            {'field_size':k[0],'race_class':k[1],'complete_market':k[2],'snapshots':v} for k,v in sorted(strata.items())],
        'quote_exclusions':dict(sum((Counter(p['source']['quote_quality']['rejected']) for p in all_packets),Counter())),
        'selection_reasons':{g:{m:dict(Counter(r['selection_reasons'][g][m] for r in rows)) for m in METHODS} for g in GROUPS},
        'risk_abstained_forecasts':{g:sum(r['views'][g]['risk'] is None for r in rows) for g in GROUPS},
        'comparisons':comparisons,'probability_comparisons':proper,'confirmations':confirmed,
        'confirmation_state':'not_registered_exploratory_only' if not trials else 'see_registered_trials',
        'component_removal_is_not_causal_attribution':True,'diagnostics':diagnostics}
    (folder/'annual_fusion_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    def pct(x):return '—' if x is None else f'{x*100:.1f}%'
    table=[]
    for r in comparisons:
        a,b=r['challenger'],r['risk']
        table.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (LABELS[r['method']],r['group'],a['races'],pct(a['hit_rate']),pct(b['hit_rate']),pct(a['return_rate']),pct(b['return_rate'])))+'</tr>')
    page='<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>統合方程式の比較研究</title><style>body{font:16px/1.8 system-ui;background:#f5f7fa;color:#172235;margin:0}main{max-width:1080px;margin:auto;padding:24px}section{background:white;padding:20px;margin:16px 0;border-radius:12px}table{border-collapse:collapse;min-width:760px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}a{color:#2459ae}</style><main><h1>部署と方程式を組み合わせて検証</h1><p><b>研究用。社長への組み込み・購入承認・的中率改善の確認はしていません。</b>既存７部署とリスク部の買い目を維持しています。</p>'
    page+=f'<section><h2>準備状況</h2><p>統合用の締切前入力：{len(all_packets)}件。前日までの学習可能記録：{len(samples)}レース・{len({s["date"] for s in samples})}日。確認済みの途中隊列：{len(paths)}件。</p><p>モデル学習・組み合わせ学習・確率補正に、重ならない期間を使用します。各期間50レース・7日以上が学習開始の条件です。的中率を保証する件数ではありません。</p></section>'
    page+='<section><h2>統合する判断</h2><p>部署別２・３着評価、順位別・条件付き市場支持、専門展開補正、初期の市場補正・先着関係・隊列頻度モデルに、１・２着の組み合わせを使う評価と条件付き隊列モデルを加えます。予想部・軍師・高配当戦略部の意見も、同じ処理回で保存したものだけを学習用の特徴として使います。</p><p>P(着順)＝確率補正〔Σ 学習した重み × 各式の着順確率〕。重みは別期間の予測実績から学習。同じ予測の複製は１枠にまとめ、似た誤りへの偏りを抑えます。未観測の隊列、未提出の判断は捏造しません。部品の除外比較は、原因の証明とは区別します。</p></section>'
    page+='<section><h2>検証の見直し</h2><p>未取得値と実績ゼロを区別し、ライン先頭・番手・競り合う先行選手と１・２着の組み合わせを使います。リスク部が見送るレースでも、確率の誤差を市場支持・元のモデルと同時点で比較します。的中失敗を１着・２着条件・３着条件へ分解し、対象外の理由も保存します。</p><p>買い目比較は研究用の確率順。本線・穴それぞれリスク部と同点数・同金額、１点100円・最大12点、穴100倍以上。価格不足・候補不足は水増ししません。未検証の期待値から購入承認は出しません。旧式の期待値条件による別実験は維持しています。</p></section>'
    page+='<section><h2>同条件の買い目比較</h2><p>式によって比較可能なレースが異なるため、行同士の数字を単純に順位付けしません。詳細では同一レースの確率誤差、対象外理由、最大払戻への依存を確認できます。</p><div class="scroll"><table><tr><th>式</th><th>区分</th><th>比較R</th><th>式の的中率</th><th>リスク的中率</th><th>式の回収率</th><th>リスク回収率</th></tr>'+(''.join(table) or '<tr><td colspan="7">事前記録と学習データを収集中です。改善は未確認です。</td></tr>')+'</table></div></section>'
    page+='<section><h2>社長に組み込む前に</h2><p>現在は候補を調べる段階です。採用確認は、目標改善幅と方式を将来の期間に事前登録してから行います。登録した56日間で比較500レース・28日未満なら証拠不足。登録回数に応じて誤判定の許容幅を分配し、方式を途中変更した試験は採用根拠にしません。自動採用はありません。</p></section><p><a href="annual_fusion_report.json">詳細・不足理由・重み・着順別誤差</a> ／ <a href="annual_equation_report.html">初期３式</a> ／ <a href="annual_position_v2_report.html">２・３着の個別改善実験</a> ／ <a href="operations.html">会社の運営</a></p></main></html>'
    (folder/'annual_fusion_report.html').write_text(page,encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['report','train','register-confirmation'])
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    parser.add_argument('--starts-at');parser.add_argument('--target-hit-delta',type=float)
    args=parser.parse_args()
    if args.command=='train':prepare_model(args.output_dir)
    if args.command=='register-confirmation':
        if not args.starts_at or args.target_hit_delta is None:parser.error('future --starts-at and --target-hit-delta required')
        register_confirmation(args.output_dir,datetime.fromisoformat(args.starts_at),args.target_hit_delta)
    report=build_report(args.output_dir)
    print(json.dumps({k:report[k] for k in ('prior_training_races','prior_training_days','input_snapshots','confirmation_state')},ensure_ascii=False))
