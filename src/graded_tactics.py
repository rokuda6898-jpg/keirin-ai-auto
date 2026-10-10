"""Pre-race tactical feature prototype. No invented scenario probabilities."""
import math
from race_features import verify_line_prediction, line_features


def tactical_inputs(data):
    entries=[e for e in data.get('entries',[]) if not e.get('absent')]
    cars=[int(e['number']) for e in entries]
    status=verify_line_prediction(data.get('linePrediction'),cars)
    race=data.get('race',{})
    result={'race_id':race.get('id'),'stage':race.get('raceType'),
            'advancement_text':race.get('advancementConditionText') or None,
            'advancement_status':'available' if race.get('advancementConditionText') else 'missing',
            'line_status':status,'source_scope':'provider_predicted_formation_not_actual_race',
            'forecast_available':False,'reason':'scenario_model_not_trained','riders':{},'leader_matchups':[]}
    if status!='verified':return result
    lines,n,_=line_features(data.get('linePrediction'))
    leaders=sorted(c for c,v in lines.items() if v['line_position']==1)
    for car in cars:
        own=lines[car]
        teammates=sorted([c for c in cars if lines[c]['line_id']==own['line_id']],key=lambda c:lines[c]['line_position'])
        result['riders'][str(car)]={**own,'number_of_lines':n,'line_leader':teammates[0],
            'ahead_in_line':teammates[:own['line_position']-1],
            'behind_in_line':teammates[own['line_position']:],
            'opposing_leaders':[c for c in leaders if lines[c]['line_id']!=own['line_id']]}
    # Directed alternatives: A moves first and B responds. No guessed weights.
    for a in leaders:
        for b in leaders:
            if a!=b:result['leader_matchups'].append({'first_mover':a,'responding_leader':b,
                'first_mover_line_size':lines[a]['line_size'],'responding_line_size':lines[b]['line_size']})
    return result


def conditional_tactical_features(inputs,car,prefix=()):
    """Features for later fitting, conditional on hypothesized earlier finishers."""
    if inputs['line_status']!='verified':return [math.nan]*8
    own=inputs['riders'][str(car)]
    picked=[inputs['riders'][str(c)] for c in prefix]
    return [own['line_position'],own['line_size'],own['number_of_lines'],len(own['opposing_leaders']),
            sum(r['line_id']==own['line_id'] for r in picked),
            int(own['line_leader'] in prefix),
            sum(c in prefix for c in own['opposing_leaders']),
            sum(c not in prefix for c in own['ahead_in_line'])]


def tactical_interaction_features(inputs,records,car,prefix=()):
    """Add line support and rival-line pressure using forecast inputs only."""
    if inputs.get('line_status')!='verified':return [math.nan]*10
    rows={int(r['car_no']):r for r in records}
    own=inputs['riders'][str(car)]
    teammates=[c for c in rows if c!=car and inputs['riders'][str(c)]['line_id']==own['line_id']]
    behind=[c for c in teammates if inputs['riders'][str(c)]['line_position']>own['line_position']]
    rivals={}
    for c,r in inputs['riders'].items():
        if r['line_id']==own['line_id']:continue
        rivals.setdefault(r['line_id'],[]).append(int(c))
    def score(c):
        try:
            v=float(rows[c].get('score'))
            return v if math.isfinite(v) else math.nan
        except (TypeError,ValueError):return math.nan
    def mean(values):
        values=[v for v in values if math.isfinite(v)]
        return sum(values)/len(values) if values else math.nan
    support=[score(c) for c in behind]
    own_lead=int(own['line_leader']); lead_score=score(own_lead)
    rival_totals=[sum(v for v in (score(c) for c in group) if math.isfinite(v)) for group in rivals.values()]
    rival_leaders=[int(group[0]) for group in rivals.values()]
    rival_scores=[score(c) for c in rival_leaders]
    try:own_attack=float(rows[own_lead].get('back_count'))
    except (TypeError,ValueError):own_attack=math.nan
    attacks=[]
    for c in rival_leaders:
        try:
            v=float(rows[c].get('back_count'))
            if math.isfinite(v):attacks.append(v)
        except (TypeError,ValueError):pass
    front_pressure=sum(v>=own_attack-1 for v in attacks) if math.isfinite(own_attack) else math.nan
    return [len(behind),mean(support),max(support,default=math.nan),
            lead_score-score(car) if math.isfinite(lead_score) and math.isfinite(score(car)) else math.nan,
            max(rival_totals,default=math.nan),
            max(rival_scores,default=math.nan)-lead_score if rival_scores and math.isfinite(lead_score) else math.nan,
            len(rival_leaders),front_pressure,own_attack,
            sum(inputs['riders'][str(c)]['line_id']==own['line_id'] for c in prefix)]
