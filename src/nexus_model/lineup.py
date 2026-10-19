"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta


def normalize(data,meta):
    errors=[];race=data.get('race',{});schedule=data.get('schedule',{})
    rid=str(meta['race_id']);date=meta['date']
    if race.get('id')!=rid:errors.append('race_id_mismatch')
    if str(schedule.get('date'))!=date.replace('-',''):errors.append('date_mismatch')
    if str(schedule.get('cupId'))!=str(meta['cup_id']) or schedule.get('index')!=meta['index']:errors.append('meeting_mismatch')
    entries=data.get('entries',[]);players={str(p['id']):p for p in data.get('players',[])}
    roster={};absent=[]
    for e in entries:
        car=e.get('number');pid=str(e.get('playerId',''))
        if type(car)!=int or not 1<=car<=9 or not pid.isdigit() or e.get('raceId')!=rid:errors.append('invalid_entry');continue
        if car in roster or car in absent:errors.append('duplicate_entry');continue
        if e.get('absent') is True:absent.append(car);continue
        if e.get('absent') is not False:errors.append('unknown_absence_status');continue
        p=players.get(pid)
        if not p:errors.append('missing_player');continue
        # Birthday and term are retained for historical-age calculations. Current
        # profile age, current class and mutable profile ratings are not features.
        roster[car]=dict(player_id=pid,birthday=p.get('birthday'),gender=p.get('gender'),term=p.get('term'),
            entry_class=e.get('playerCurrentTermClass'),entry_group=e.get('playerCurrentTermGroup'))
    if len({r['player_id'] for r in roster.values()})!=len(roster):errors.append('duplicate_player')
    pred=data.get('linePrediction');lines=[]
    try:
        lines=[[list(group['numbers']) for group in line['entries']] for line in pred['lines']]
        if any(not line or any(not group for group in line) for line in lines):errors.append('empty_line_group')
    except (KeyError,TypeError):errors.append('provider_missing_lineup');lines=[]
    flat=[car for line in lines for group in line for car in group]
    if any(type(car)!=int or not 1<=car<=9 for car in flat):errors.append('invalid_lineup_car')
    if len(flat)!=len(set(flat)):errors.append('duplicate_lineup_car')
    if set(flat)-set(roster)-set(absent):errors.append('unregistered_lineup_car')
    missing=sorted(set(roster)-set(flat))
    if missing:errors.append('active_rider_missing_from_lineup')
    female=any(r['gender']=='女' for r in roster.values()) or meta['race_class'].startswith('L')
    advance='アドバンス' in meta['race_type'] or 'アドバンス' in meta['race_class']
    regime='cancelled' if meta['cancel'] else 'girls_individual' if female else 'advance_separate' if advance else 'mens_lines'
    clean_lines=[];affected=[];membership={};position={};contested=[]
    hard=bool(errors)
    for li,line in enumerate(lines):
        cars={car for group in line for car in group};withdrawn=sorted(cars&set(absent))
        if withdrawn:
            affected.append(dict(source_line_index=li,source_groups=line,withdrawn_cars=withdrawn,active_cars=sorted(cars&set(roster)),new_formation_unknown=True))
            # Retain source groups for audit only; do not pretend the active
            # remainder is the newly declared post-withdrawal lineup.
            continue
        if hard or regime!='mens_lines':continue
        clean_lines.append(line)
        for slot,group in enumerate(line):
            if len(group)>1:contested.append(list(group))
            for car in group:membership[car]=li;position[car]=slot
    roles={}
    for line in clean_lines:
        size=sum(map(len,line))
        for slot,group in enumerate(line):
            for car in group:
                roles[str(car)]='single' if size==1 else 'contested' if len(group)>1 else 'leader' if slot==0 else 'second' if slot==1 else 'rear'
    pairs=[]
    for a in sorted(roster):
        for b in sorted(roster):
            if a==b:continue
            known=a in membership and b in membership
            same=known and membership[a]==membership[b]
            pairs.append(dict(a=a,b=b,relationship_known=known,same_line=same if known else None,
                same_contested_slot=(same and position[a]==position[b]) if known else None,
                a_strictly_ahead=(same and position[a]<position[b]) if known else None))
    # Matched legacy control: flatten parallel slots, and lose all line features
    # when declared lineup and active roster do not coincide.
    legacy=[[car for group in line for car in group] for line in lines]
    legacy_valid=not hard and not affected and regime=='mens_lines'
    result=dict(race_id=rid,date=date,roster={str(k):v for k,v in sorted(roster.items())},regime=regime,
        source_groups=lines,provider_line_type=pred.get('lineType') if isinstance(pred,dict) else None,
        known_unchanged_groups=clean_lines,withdrawal_affected_groups=affected,absent_cars=sorted(absent),
        competition_groups=contested,role_by_car=roles,pairs=pairs,
        legacy_flattened_groups=legacy if legacy_valid else [],legacy_valid=legacy_valid,
        active_cars_without_verified_role=sorted(set(roster)-set(membership)),errors=sorted(set(errors)),
        historically_reconstructed=True,preclose_capture_certified=False,withdrawal_announcement_time_verified=False,
        actual_race_formation_observed=False)
    return result

