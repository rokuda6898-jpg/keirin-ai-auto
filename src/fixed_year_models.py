"""Offline reconstructions for the fixed-year study, never prospective records.

Department score arithmetic mirrors annual_knowledge.forecast_departments.
All profiles and fitted coefficients come from before the held-out year.
"""
import math
from functools import lru_cache
from itertools import permutations

import numpy as np
import pandas as pd

import fusion_equations as fusion
import equation_models as legacy
from department_experiment_v2 import frame_records
from department_ticket_v2 import distributions


def department_records(rows, profiles):
    fallback=np.array([[max(float(r.get(c) or 0),0) for c in ('p_win','p_second','p_third')] for r in rows])
    fallback/=fallback.sum(axis=0)
    result={}
    for department in fusion.DEPARTMENTS[:4]:
        values=[]
        for i,rider in enumerate(rows):
            baseline=fallback[i].copy();profile=profiles.get(str(rider['player_id']))
            if not profile:
                values.append(baseline);continue
            evaluation=profile.get('evaluation',profile)
            count=float(evaluation.get('effective_races',evaluation.get('races',0)))
            rates=(np.asarray(evaluation['rates'])*count+baseline*20)/(count+20)
            context=None
            if department=='pace_department':context=profile.get('recent90')
            if department=='line_department':
                position=fusion.finite(rider.get('line_position'))
                context=profile.get('line_positions',{}).get(str(int(position))) if position is not None else None
            if context and context.get('races',0):
                mass=float(context.get('effective_races',context['races']));weight=mass/(mass+20)
                rates=rates*(1-weight)+np.asarray(context['rates'])*weight
            if department=='pace_department':
                associations=[];weights=[]
                for event in profile.get('events',{}).values():
                    observed=float(event.get('observed',0));r=event.get('true_results',{})
                    n=float(r.get('effective_races',r.get('races',0)))
                    if observed>=10 and n>=5:
                        associations.append((np.asarray(r['rates'])*n+rates*20)/(n+20))
                        weights.append(float(event.get('effective_true',event.get('true',0)))/max(1.,float(event.get('effective_observed',observed))))
                if weights and sum(weights)>0:
                    strength=min(.25,sum(weights)/len(weights)*count/(count+40))
                    rates=rates*(1-strength)+np.average(associations,axis=0,weights=weights)*strength
            if department=='risk_department':
                entries=max(float(evaluation.get('effective_entries',evaluation.get('entries',0))),1)
                unplaced=float(evaluation.get('effective_unplaced',evaluation.get('unplaced_rows',0)))
                reliability=count/(count+40)*max(0,1-unplaced/entries)
                rates=rates*reliability+baseline*(1-reliability)
            values.append(np.maximum(np.nan_to_num(rates,nan=0,posinf=0,neginf=0),0))
        matrix=np.asarray(values)
        for j in range(3):
            mass=matrix[:,j].sum();matrix[:,j]=matrix[:,j]/mass if mass>0 else fallback[:,j]
        result[department]=[{**r,'p_win':float(v[0]),'score_first':float(v[0]*100),
            'score_second':float(v[1]*100),'score_third':float(v[2]*100),
            'department_score_second':float(v[1]*100),'department_score_third':float(v[2]*100)} for r,v in zip(rows,matrix)]
    return result


def department_inputs(race, profiles):
    return {d:pd.DataFrame(rows) for d,rows in department_records(frame_records(race),profiles).items()}


@lru_cache(maxsize=10)
def triple_indices(n):
    return np.array(list(permutations(range(n),3)),dtype=int)


def best_order(rows,fields=('p_win','p_second','p_third')):
    ix=triple_indices(len(rows))
    values=np.log(np.maximum([[r[f] for f in fields] for r in rows],1e-12))
    return [int(rows[i]['car_no']) for i in ix[np.argmax(values[ix,np.arange(3)].sum(axis=1))]]


def make_packet(race,profiles):
    """Recalculate opinions explicitly; never claim historical submissions."""
    if race['finish_pos'].notna().any():raise ValueError('outcomes must be stripped before prediction')
    raw=frame_records(race.sort_values('car_no'))
    inputs=department_records(raw,profiles)
    scenarios={};arms={};specialists=[]
    for d,rows in inputs.items():
        order=best_order(rows,('score_first','score_second','score_third'))
        scenarios[d]={'top3_cars':order,'availability':'retrospective_reconstruction'}
        specialists.append({'top3_cars':order,'forecast_available':True})
        packed=legacy_position_weights(rows)
        dist={'-'.join(map(str,t)):p for t,p in distributions(packed)[3].items()}
        if d=='risk_department':risk=dist
        else:arms[d+':baseline']={'distribution':dist}
    standard=best_order(raw)
    cars=[int(r['car_no']) for r in raw];triples=list(permutations(cars,3))
    votes=[{c:sum(s['top3_cars'][i]==c for s in specialists) for c in cars} for i in range(3)]
    weights={c:3-i for i,c in enumerate(standard)}
    strategist=max(triples,key=lambda t:(sum(votes[i][c]*(3-i) for i,c in enumerate(t)),
        sum(weights.get(c,0)*(3-i) for i,c in enumerate(t)),tuple(-c for c in t)))
    first=max((r for r in raw if int(r['car_no'])!=standard[0]),key=lambda r:(r['p_win'],-r['car_no']))['car_no']
    second=max((r for r in raw if r['car_no']!=first),key=lambda r:(r['p_second'],-r['car_no']))['car_no']
    third=max((r for r in raw if r['car_no'] not in (first,second)),key=lambda r:(r['p_third'],-r['car_no']))['car_no']
    scenarios['prediction_department']={'top3_cars':standard,'availability':'retrospective_reconstruction'}
    scenarios['strategist_department']={'top3_cars':list(strategist),'availability':'retrospective_reconstruction'}
    scenarios['high_payout_department']={'top3_cars':list(map(int,(first,second,third))),
        'availability':'unpriced_reconstructed_scenario_not_a_100plus_ticket'}
    return {'race_id':str(race.iloc[0].race_id),'date':str(race.iloc[0].date),
        'source':{'race_id':str(race.iloc[0].race_id),'quote_quality':{'complete_single_snapshot':False},
            'arms':arms,'evidence':{'inputs':inputs,'usable_quotes':{}}},
        'auxiliary':{'riders':raw,'departments':scenarios},'risk_distribution':risk}


def legacy_position_weights(rows):
    """Array form of current score_riders' unvalidated-specialist branch.

    Tested against that function; no change to live risk selection is made.
    No quoted market is available in this archive, so popularity is not used.
    """
    def values(name):
        return np.array([float(r[name]) if r.get(name) is not None else np.nan for r in rows])
    def relative(v,fallback):
        v=np.where(np.isfinite(v)&(v>=0),v,np.nan)
        good=np.isfinite(v);maximum=v[good].max() if good.any() else np.nan
        return np.where(good,v/maximum,fallback) if np.isfinite(maximum) and maximum>0 else fallback.copy()
    win=values('p_win');win/=win.sum();strength=relative(win,np.ones(len(rows)))
    result=[{'car_no':int(r['car_no']),'first':float(p*100)} for r,p in zip(rows,win)]
    for pos,name in ((2,'second'),(3,'third')):
        prior=(values('player_prior_place2_rate')-values('player_prior_win_rate') if pos==2 else
               values('player_prior_place3_rate')-values('player_prior_place2_rate'))
        exact=relative(values(f'place{pos}_rate'),strength)
        historical=relative(np.maximum(prior,0),exact)
        context=np.mean([relative(values(f'{prefix}_place{pos}_rate'),exact)
                         for prefix in ('track','weather','race_type','line_role')],axis=0)
        score=np.clip(100*(.45*exact+.25*historical+.25*context+.05*strength)+3*(values('line_position')==pos),.01,100)
        for r,v in zip(result,score):r[name]=float(v)
    return result


def pairwise_distribution(beta,rows):
    """Same price-independent pairwise equation, without inventing odds."""
    fs=legacy.rider_features(rows);cars=sorted(fs);log_q={}
    for a,b in permutations(cars,2):
        z=sum(w*(u-v) for w,u,v in zip(beta,fs[a],fs[b]))
        log_q[a,b]=-max(0,-z)-math.log1p(math.exp(-abs(z)))
    scores={}
    for a,b,c in permutations(cars,3):
        v=log_q[a,b]+log_q[a,c]+log_q[b,c]
        v+=sum(log_q[w,o] for o in cars if o not in (a,b,c) for w in (a,b,c))
        scores[f'{a}-{b}-{c}']=v
    return legacy.softmax(scores)


def components(model,packet):
    available=fusion.static_experts(packet)
    available['risk:current_logic']=packet['risk_distribution']
    if model is not None:
        ctx=fusion.feature_context(packet)
        fs=[fusion.joint_features(packet,t,ctx) for t in fusion.triples(packet)]
        available['joint:prefix_context']=dict(zip(fusion.keys(packet),fusion.softmax([
            fusion.dot(model['joint_beta'],f) for f in fs])))
        available['pairwise:pairwise_order']=pairwise_distribution(model['pair_beta'],
            packet['source']['evidence']['inputs']['risk_department'])
    return available


def prediction(model,packet):
    available=components(model,packet)
    combined=fusion.temperature(fusion.mix(model['pool'],available),model['temperature'])
    names={'original':'original:model_positions','risk':'risk:current_logic',
        'data':'positions:data_department','pace':'positions:pace_department','line':'positions:line_department',
        'data_legacy':'legacy_positions:data_department:baseline',
        'pace_legacy':'legacy_positions:pace_department:baseline',
        'line_legacy':'legacy_positions:line_department:baseline',
        'pairwise':'pairwise:pairwise_order','joint':'joint:prefix_context'}
    result={name:available[key] for name,key in names.items()}
    result['fusion']=combined
    for name,families in {'without_departments':('positions','legacy_positions'),
                          'without_pairwise':('pairwise',),'without_joint':('joint',)}.items():
        result[name]=fusion.temperature(fusion.mix(model['pool'],available,families),model['temperature'])
    return result


def fit_pairwise(designs,labels,weights):
    X=np.asarray(designs,dtype=float);y=np.asarray(labels,dtype=float);w=np.asarray(weights,dtype=float);w/=w.sum()
    beta=np.zeros(X.shape[1])
    for _ in range(legacy.CONFIG['iterations']):
        q=1/(1+np.exp(-np.clip(X@beta,-30,30)))
        beta-=legacy.CONFIG['learning_rate']*(X.T@((q-y)*w)+legacy.CONFIG['ridge']*beta)
    return beta.tolist()


def fit_joint_matrix(X,lengths,targets,iterations=None):
    """Full-batch race-equal version of fusion.fit_linear; disk-backed X allowed."""
    starts=np.r_[0,np.cumsum(lengths)[:-1]].astype(np.int64)
    target_rows=starts+np.asarray(targets)
    beta=np.zeros(X.shape[1],dtype=X.dtype)
    for _ in range(iterations or fusion.CONFIG['iterations']):
        logits=X@beta
        exp=np.exp(logits-np.repeat(np.maximum.reduceat(logits,starts),lengths))
        mass=np.add.reduceat(exp,starts)
        residual=exp/np.repeat(mass,lengths);residual[target_rows]-=1
        grad=X.T@residual/len(lengths)+fusion.CONFIG['ridge']*beta
        beta-=fusion.CONFIG['learning_rate']*grad
    return beta.astype(float).tolist()
