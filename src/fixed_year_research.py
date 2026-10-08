"""Fixed older-history training and last-calendar-year holdout research.

No production model, purchase, or prospective prediction ledger is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def boundaries(asof):
    end = pd.Timestamp(asof).normalize().tz_localize(None)
    return end - pd.DateOffset(years=1), end


def split_history(history, asof):
    start, end = boundaries(asof)
    dates = pd.to_datetime(history['date'], errors='coerce').dt.normalize()
    return history.loc[dates.lt(start)].copy(), history.loc[dates.ge(start) & dates.lt(end)].copy()


def inventory(history, path, asof):
    start, end = boundaries(asof)
    train, target = split_history(history, asof)
    def describe(frame):
        dates = pd.to_datetime(frame['date'], errors='coerce')
        return {'rows':len(frame), 'races':int(frame['race_id'].nunique()),
            'first':str(dates.min().date()) if len(frame) else None,
            'last':str(dates.max().date()) if len(frame) else None,
            'days':int(dates.nunique()),
            'months':frame.assign(month=dates.dt.strftime('%Y-%m')).groupby('month').race_id.nunique().to_dict()}
    fields = ('score','win_rate','place2_rate','place3_rate','back_count','style','line_id',
        'line_position','line_verification_status','feature_captured_at_jst','source_url',
        'odds_win','recent_avg_finish','wind_speed','race_class','finish_pos')
    return {'status':'inventory_only','asof_exclusive':str(end.date()),'holdout_start_inclusive':str(start.date()),
        'training_cutoff_exclusive':str(start.date()),'model_update_during_holdout':False,
        'purchase_authorized':False,'ceo_integration':False,'history_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'training':describe(train),'holdout':describe(target),'columns':list(history.columns),
        'availability':{c:{'present':c in history,'training_nonmissing':int(train[c].notna().sum()) if c in train else 0,
            'holdout_nonmissing':int(target[c].notna().sum()) if c in target else 0} for c in fields},
        'line_statuses':history['line_verification_status'].fillna('missing').astype(str).value_counts().to_dict()
                       if 'line_verification_status' in history else {},
        'invalid_dates':int(pd.to_datetime(history['date'],errors='coerce').isna().sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',type=Path,default=Path('data/raw/history.csv'))
    parser.add_argument('--asof',default='2026-10-09')
    parser.add_argument('--output-dir',type=Path,default=Path('research/fixed_year_20261009'))
    parser.add_argument('--stage',choices=('inventory','train','predict','assess'),default='inventory')
    args=parser.parse_args()
    if not args.history.is_file():raise SystemExit('Real historical archive missing; refusing synthetic replacement.')
    history=pd.read_csv(args.history,dtype={'race_id':str,'player_id':str},low_memory=False)
    report=inventory(history,args.history,args.asof)
    for name in ('history_metadata.json','history_odds.csv','history_trifecta_odds.csv'):
        p=args.history.with_name(name)
        report.setdefault('auxiliary_files',{})[name]={'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else 0}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage!='inventory':
        prior=json.loads((args.output_dir/'inventory.json').read_text(encoding='utf-8'))
        if (prior['history_sha256']!=report['history_sha256'] or prior['asof_exclusive']!=report['asof_exclusive']):
            raise ValueError('archive or fixed boundary changed after inventory')
        from fixed_year_study import train,predict,assess
        older,target=split_history(history,args.asof)
        if args.stage=='train':train(older,args.output_dir,report['training_cutoff_exclusive'])
        elif args.stage=='predict':predict(target,args.output_dir)
        else:assess(target,args.output_dir)
        return
    (args.output_dir/'inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FIXED_YEAR_INVENTORY '+json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
