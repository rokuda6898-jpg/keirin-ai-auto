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
