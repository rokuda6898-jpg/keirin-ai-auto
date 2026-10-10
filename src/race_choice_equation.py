"""Race-wise conditional choice equation for ordered podium positions."""
from dataclasses import dataclass
from itertools import permutations
import numpy as np
from scipy.optimize import minimize


@dataclass
class RaceChoiceModel:
    fills: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    coef: np.ndarray

    def transform(self,x):
        x=np.asarray(x,dtype=float)
        missing=~np.isfinite(x)
        x=np.where(missing,self.fills,x)
        return np.column_stack([(x-self.centers)/self.scales,missing.astype(float)])

    def predict_scores(self,x):
        return self.transform(x)@self.coef


def fit_choice(groups,features,prefix_length,stages):
    matrices=[];positives=[];group_ids=[];weights=[];cursor=0
    for _,race in groups:
        podium=[int(c) for c in __import__('graded_retrospective').podium(race) or ()]
        if len(podium)<3:continue
        prefix=tuple(podium[:prefix_length])
        records=race.drop(columns=['finish_pos','official_finish_pos','result_available'],errors='ignore').to_dict('records')
        cars=[int(r['car_no']) for r in records if int(r['car_no']) not in prefix]
        if podium[prefix_length] not in cars:continue
        x=np.asarray([features(records,c,prefix,stages,True) for c in cars],dtype=float)
        matrices.append(x);positives.append(cursor+cars.index(podium[prefix_length]))
        group_ids.extend([len(positives)-1]*len(cars));cursor+=len(cars)
        weights.append(2. if str(race.iloc[0].date)>='2024-10-10' else 1.)
    raw=np.vstack(matrices);positive=np.asarray(positives);gid=np.asarray(group_ids,dtype=int);gw=np.asarray(weights)
    fills=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in raw.T])
    missing=~np.isfinite(raw);raw=np.where(missing,fills,raw)
    centers=raw.mean(axis=0);scales=raw.std(axis=0);scales=np.where(scales>1e-8,scales,1.)
    x=np.column_stack([(raw-centers)/scales,missing.astype(float)])
    group_count=len(positive);denom=gw.sum()

    def objective(coef):
        z=x@coef; maxima=np.full(group_count,-np.inf);np.maximum.at(maxima,gid,z)
        expz=np.exp(np.clip(z-maxima[gid],-60,0));totals=np.bincount(gid,weights=expz,minlength=group_count)
        loss=np.sum(gw*(maxima+np.log(totals)-z[positive]))/denom
        probs=expz/totals[gid]*gw[gid]/denom;probs[positive]-=gw/denom
        l2=0.025*np.dot(coef,coef)
        grad=x.T@probs+0.05*coef
        return loss+l2,grad

    result=minimize(objective,np.zeros(x.shape[1]),method='L-BFGS-B',jac=True,
                    options={'maxiter':180,'ftol':1e-8,'gtol':1e-5,'maxls':20})
    if not np.isfinite(result.fun):raise ValueError('Race-choice optimization did not converge')
    return RaceChoiceModel(fills,centers,scales,result.x)


def distribution(records,models,features,stages):
    cars=sorted(int(r['car_no']) for r in records);by_pos=[]
    for position,model in enumerate(models):
        prefixes=[()] if position==0 else ([ (a,) for a in cars ] if position==1 else list(permutations(cars,2)))
        result={}
        for prefix in prefixes:
            remaining=[c for c in cars if c not in prefix]
            x=np.asarray([features(records,c,prefix,stages,True) for c in remaining],dtype=float)
            scores=model.predict_scores(x);scores-=scores.max();p=np.exp(scores);p/=p.sum()
            result[prefix]=dict(zip(remaining,p))
        by_pos.append(result)
    p1=by_pos[0][()]
    return {(a,b,c):p1[a]*by_pos[1][(a,)][b]*by_pos[2][(a,b)][c]
            for a,b,c in permutations(cars,3)}
