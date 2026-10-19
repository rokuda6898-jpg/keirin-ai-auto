"""Exact saved 53.85% composition and complete-formation handoff."""
import json,hashlib,itertools
from pathlib import Path
import numpy as np
import lightgbm as lgb
from . import base,team,role,v2,optimizer,pair

class Engine:
    def __init__(self,root):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text('utf8'));self.models={}
        for name,info in self.manifest['assets'].items():
            path=self.root/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha']:raise ValueError('model hash mismatch: '+name)
            self.models[name.removesuffix('.txt')]=lgb.Booster(model_file=str(path))
    def predict(self,d,i=0):
        r=d['rows'][i];cars=r['cars'];n=len(cars)
        raw=np.asarray(self.models['team_judge-1'].predict(team.class_row(d,i)[None],num_threads=2)[0])
        spec=self.manifest;pp=team.calibrate(raw,spec['team_selection']['final_temperature']);cap=team.cap(pp);cap=15 if cap==12 else cap
        cat='FAVORITE' if cap==6 else ['MAIN','MEDIUM','HIGH'][int(team.three(pp).argmax())]
        choice=spec['score_selection'][str(cap)]
        def chain(prefix,feat,query=None,oldthird=None):
            parts=[]
            for k in range(3):
                tm=base.templates(n)[k]
                if k==2 and oldthird is not None:parts.append(oldthird);continue
                mat=feat(d,i,tm[:,-1],tm[:,0] if k else None,tm[:,1] if k==2 else None)
                if query is not None:mat[:,v2.names(k).index('specialist_query_category')]=query
                values=np.maximum(self.models[prefix+str(k)].predict(mat,num_threads=2),1e-15).reshape(-1,n-k)
                parts.append((values/values.sum(1)[:,None]).ravel())
            return np.repeat(parts[0],(n-1)*(n-2))*np.repeat(parts[1],n-2)*parts[2],parts
        if choice in ('general_only_free','expert_only_free'):
            cal=spec['v2_calibration'];p=team.calibrate(self.models['v2_head-1'].predict(v2.judge_row(d,i)[None],num_threads=2)[0],cal['judge'])
            if choice=='general_only_free':q=team.calibrate(chain('v2_head',v2.features,0)[0],cal['experts'][0])
            else:q=sum(p[j]*team.calibrate(chain('v2_head',v2.features,j+1)[0],cal['experts'][j+1]) for j in range(3))
        else:
            align=spec['alignment'];temps=align['final_temperatures'];alpha=align['general_weight']
            soft=sum(team.three(pp)[j]*team.calibrate(chain('team_'+label,team.features)[0],temps[label]) for j,label in enumerate(['MAIN','MEDIUM','HIGH']))
            general,parts=chain('team_general',team.features)
            newer,_=chain('role_head',role.features,oldthird=parts[2])
            q=(1-alpha)*soft+alpha*team.calibrate(newer,temps['general'])
        assert np.isfinite(q).all() and abs(q.sum()-1)<1e-8
        plan=optimizer.select(cars,q,cap)
        if spec['formation_selection'][str(cap)]=='shape_only':plan=pair.select(cars,q,cap,plan)
        tickets=sorted(set(t for b in plan['blocks'] for t in itertools.product(b['first'],b['second'],b['third']) if len(set(t))==3))
        assert len(tickets)==cap and len(plan['blocks'])<=4 and set(tickets)==set(map(tuple,plan['tickets']))
        cube=np.zeros((n,n,n));cube[tuple(base.templates(n)[2].T)]=q
        order=sorted(range(n),key=lambda j:(-cube[j].sum(),-cube[:,j,:].sum(),cars[j]))
        marks={cars[j]:'◎○▲△☆'[k] if k<5 else '' for k,j in enumerate(order)}
        return dict(model_id=spec['id'],category=cat,count=cap,blocks=plan['blocks'],tickets=['-'.join(map(str,t)) for t in tickets],
            marks=marks,band_probabilities=dict(zip(['MAIN','MEDIUM','HIGH'],team.three(pp).tolist())),
            first_probabilities={c:float(cube[j].sum()) for j,c in enumerate(cars)},q=q.tolist())
