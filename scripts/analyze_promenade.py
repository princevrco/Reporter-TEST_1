"""Merge old/new exports, compare Aug 15-Oct 20, and learn electrical BMS response."""
import argparse
import math
from pathlib import Path
import pandas as pd
from option_c.data import DataError,load,save
from option_c.ingestion import merge_meters,merge_bms
from option_c.response import discover,features,fit,predict
from option_c.batch import run
from option_c.overview import build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meter',type=Path,nargs='+',required=True,help='Oldest to newest CARMA ZIP/CSV/directory')
    p.add_argument('--bms',type=Path,nargs='+',required=True,help='Oldest to newest BMS history/update CSV')
    p.add_argument('--config',type=Path,default=Path('config/promenade-test.json'))
    p.add_argument('--response-config',type=Path,help='Reviewed role-to-column bindings; otherwise propose bindings from names')
    p.add_argument('--overlap-policy',choices=['error','prefer-last'],default='error')
    p.add_argument('--start',default='2026-08-15');p.add_argument('--end',default='2026-10-21',help='Exclusive local date')
    p.add_argument('--rate',type=float,default=.12);p.add_argument('--output',type=Path,default=Path('results/aug15-oct20'))
    p.add_argument('--forecast-start');p.add_argument('--forecast-end')
    p.add_argument('--future-bms',type=Path,help='Explicit future optimized operating trajectory, including temperatures/states and prior-day context')
    p.add_argument('--baseline-future-bms',type=Path,help='Explicit old-policy trajectory under comparable conditions; not merely substituted setpoints')
    p.add_argument('--total-meters',nargs='+');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    try:
        if a.output.exists():raise DataError('Choose a new --output directory; existing outputs are preserved')
        cfg=load(a.config)
        a.forecast_start=a.forecast_start or str(pd.Timestamp.now(tz=cfg['timezone']).date())
        a.forecast_end=a.forecast_end or str((pd.Timestamp(a.forecast_start)+pd.DateOffset(months=3)).date())
        if pd.Timestamp(a.end)<=pd.Timestamp(a.start):raise DataError('End must follow start')
        if pd.Timestamp(a.start).date()<pd.Timestamp(cfg['intervention_timestamp']).date():raise DataError('Reporting begins before configured intervention')
        if a.rate<0 or not math.isfinite(a.rate):raise DataError('Rate must be finite and nonnegative')
        meters,ma=merge_meters(a.meter,a.overlap_policy)
        bms,ba=merge_bms(a.bms,a.overlap_policy)
        out=a.output;out.mkdir(parents=True)
        meters.to_csv(out/'merged-meter.csv',index=False);bms.to_csv(out/'merged-bms.csv',index=False)
        save(out/'merge-audit.json',{'meter':ma,'bms':ba})
        rcfg=load(a.response_config) if a.response_config else discover(bms)
        save(out/'response-config.json',rcfg)
        prepared=out/'response';prepared.mkdir()
        feature_frame=None
        try:
            feature_frame,audit=features(bms,rcfg,cfg['timezone'])
            feature_frame.to_csv(prepared/'daily-features.csv',index=False);save(prepared/'feature-audit.json',audit)
        except DataError as exc:save(prepared/'feature-audit.json',{'status':'unavailable','reason':str(exc)})
        if a.prepare_only:
            print('Merged inputs and proposed mappings: '+str(out));return
        run(out/'merged-meter.csv',out/'merged-bms.csv',cfg,a.start,a.end,out,kind='normalized',
            progress=lambda e,i,n:print(f"[{i}/{n}] {e['meter_id']}: {e['status']}",flush=True))
        scenarios={}
        for name,path in (('optimized',a.future_bms),('old_policy',a.baseline_future_bms)):
            if path:
                raw,provenance=merge_bms([path]);ff,fa=features(raw,rcfg,cfg['timezone'])
                scenarios[name]=ff;save(prepared/(name+'-scenario-audit.json'),{'source':provenance,'features':fa,
                    'scope':'User-supplied operating scenario; trajectory consistency and comparable external conditions not automatically verified'})
        summary=load(out/'summary.json');response_rows=[]
        for entry in summary['meters']:
            meter=entry['meter_id'];folder=out/entry['folder']/'response';folder.mkdir(exist_ok=True)
            row={'meter_id':meter,'status':'unavailable'}
            try:
                if feature_frame is None:raise DataError('No usable BMS features; see feature-audit.json')
                # No data on/after reporting end may enter training or feature selection.
                hist=meters[(meters.meter_id==meter)&(meters.date<a.end)]
                ff=feature_frame[feature_frame.date<a.end]
                model,diagnostics=fit(hist,ff,rcfg,cfg['timezone'])
                save(folder/'model.json',model);pd.DataFrame(diagnostics).to_csv(folder/'fit-diagnostics.csv',index=False)
                pd.DataFrame(model['heldout_predictions']).to_csv(folder/'heldout.csv',index=False)
                row.update(status='fitted_requires_review',valid_days=model['valid_days'],
                    heldout_cvrmse_pct=model['holdout_metrics']['cvrmse_pct'],heldout_nmbe_pct=model['holdout_metrics']['nmbe_pct'],
                    ridge_alpha=model['alpha'],feature_count=len(model['features']),gates=';'.join(model['gates']))
                predictions={}
                for name,scenario in scenarios.items():
                    result=predict(model,scenario,rcfg)
                    if a.forecast_start:result=[r for r in result if r['date']>=a.forecast_start]
                    if a.forecast_end:result=[r for r in result if r['date']<a.forecast_end]
                    predictions[name]={r['date']:r for r in result};save(folder/(name+'-future.json'),result)
                if 'optimized' in predictions and 'old_policy' in predictions:
                    comparison=[]
                    for day in sorted(set(predictions['optimized'])|set(predictions['old_policy'])):
                        old=predictions['old_policy'].get(day,{});opt=predictions['optimized'].get(day,{})
                        good=all(r.get('bms_estimated_kwh') is not None and not r.get('outside_training_support',True) for r in (old,opt))
                        value=old['bms_estimated_kwh']-opt['bms_estimated_kwh'] if good else None
                        comparison.append({'date':day,'old_policy_kwh':old.get('bms_estimated_kwh'),'optimized_kwh':opt.get('bms_estimated_kwh'),
                            'projected_difference_kwh':value,'projected_difference_cad':value*a.rate if value is not None else None,
                            'scope':'conditional_scenario_difference_not_verified_savings','semantics_verified':rcfg.get('semantics_verified',False)})
                    pd.DataFrame(comparison).to_csv(folder/'scenario-comparison.csv',index=False)
            except (DataError,ValueError,KeyError) as exc:row['reason']=str(exc)
            save(folder/'status.json',row);response_rows.append(row)
        pd.DataFrame(response_rows).to_csv(out/'response-summary.csv',index=False)
        build(out,a.rate,a.forecast_start,a.forecast_end,a.total_meters)
        save(out/'analysis-assumptions.json',{'start':a.start,'end_exclusive':a.end,'full_control_start':cfg['intervention_timestamp'],
            'confirmation':'User confirmed all assets under optimized control from August 15, 2026',
            'baseline_end_exclusive':cfg.get('baseline_end_exclusive'),'rate_cad_per_kwh':a.rate,
            'bms_role':'Held-out estimates are validation evidence. Full-fit diagnostics are in-sample, not independent savings. Missing meter energy never backfilled.',
            'future_role':'Baseline forecast always separate. Optimized usage requires explicit future operating trajectory; old-policy scenario differences require both trajectories.',
            'scope':'Unadjusted engineering comparison; meter boundary, BMS semantics and non-control changes require review'})
        print('Open '+str(out/'index.html'));print('Response diagnostics: '+str(out/'response-summary.csv'))
    except (DataError,ValueError,KeyError,FileNotFoundError) as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
