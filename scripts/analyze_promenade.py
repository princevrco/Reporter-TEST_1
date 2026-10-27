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
from option_c.quality import assess
from option_c import setpoint_response
from option_c.reconciliation import audit as reconcile
from option_c.weather import read as read_weather, join as join_weather


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meter',type=Path,nargs='+',required=True,help='Oldest to newest meter inputs: CARMA exports, Alectra XLSX or explicit normalized/portal CSV')
    p.add_argument('--meter-kind',choices=['auto','carma','utility','normalized','portal'],default='auto',help='Auto detects Alectra XLSX; CSV defaults to CARMA')
    p.add_argument('--bms',type=Path,nargs='+',required=True,help='Oldest to newest BMS history/update CSV')
    p.add_argument('--config',type=Path,default=Path('config/promenade-test.json'))
    p.add_argument('--weather',type=Path,help='Daily local date,temperature_c (or outdoor_temp_c); Celsius; baseline/reporting and optional future dates')
    p.add_argument('--meter-ids',nargs='+',help='Optional subset; default all separate meters')
    p.add_argument('--response-config',type=Path,help='Reviewed role-to-column bindings; otherwise propose bindings from names')
    p.add_argument('--quality-reviews',type=Path,help='Reviewed meter-quality decisions; original readings are preserved')
    p.add_argument('--overlap-policy',choices=['error','prefer-last'],default='error')
    p.add_argument('--start',default='2026-08-15');p.add_argument('--end',default='2026-10-21',help='Exclusive local date')
    p.add_argument('--rate',type=float,default=.12);p.add_argument('--output',type=Path,default=Path('results/aug15-oct20'))
    p.add_argument('--forecast-start');p.add_argument('--forecast-end')
    p.add_argument('--future-bms',type=Path,help='Explicit future optimized operating trajectory, including temperatures/states and prior-day context')
    p.add_argument('--baseline-future-bms',type=Path,help='Explicit old-policy trajectory under comparable conditions; not merely substituted setpoints')
    p.add_argument('--reference-setpoints',type=Path,help='Explicit cooling/heating setpoints by fitted equipment group; conditional scenario only, not inferred old policy')
    p.add_argument('--total-meters',nargs='+');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    try:
        if a.output.exists():raise DataError('Choose a new --output directory; existing outputs are preserved')
        cfg=load(a.config)
        reviews=load(a.quality_reviews) if a.quality_reviews else []
        a.forecast_start=a.forecast_start or str(pd.Timestamp.now(tz=cfg['timezone']).date())
        a.forecast_end=a.forecast_end or str((pd.Timestamp(a.forecast_start)+pd.DateOffset(months=3)).date())
        if pd.Timestamp(a.end)<=pd.Timestamp(a.start):raise DataError('End must follow start')
        if pd.Timestamp(a.start).date()<pd.Timestamp(cfg['intervention_timestamp']).date():raise DataError('Reporting begins before configured intervention')
        if a.rate<0 or not math.isfinite(a.rate):raise DataError('Rate must be finite and nonnegative')
        meters,ma=merge_meters(a.meter,a.overlap_policy,a.meter_kind)
        bms,ba=merge_bms(a.bms,a.overlap_policy)
        if a.meter_ids:
            if len(set(a.meter_ids))!=len(a.meter_ids) or not set(a.meter_ids)<=set(meters.meter_id):raise DataError('Unknown or duplicate meter IDs')
            meters=meters[meters.meter_id.isin(a.meter_ids)].copy()
        bms=bms[pd.to_datetime(bms.timestamp,utc=True)<pd.Timestamp(a.end,tz=cfg['timezone']).tz_convert('UTC')].copy()
        weather=weather_audit=None
        if a.weather:
            weather,weather_audit=read_weather(a.weather,cfg['timezone'])
            # Check complete baseline/reporting dates before creating output artifacts.
            needed=meters[meters.date<a.end].copy();join_weather(needed,weather)
            join_weather(pd.DataFrame({'date':pd.date_range(a.start,a.end,inclusive='left').date.astype(str)}),weather)
        out=a.output;out.mkdir(parents=True)
        meters.to_csv(out/'merged-meter.csv',index=False);bms.to_csv(out/'merged-bms.csv',index=False)
        save(out/'merge-audit.json',{'meter':ma,'bms':ba})
        if cfg.get('meter_reconciliation'):
            reconciliation=reconcile(meters,cfg['meter_reconciliation'])
            pd.DataFrame(reconciliation).to_csv(out/'meter-reconciliation.csv',index=False)
            save(out/'meter-reconciliation.json',{'relationships':cfg['meter_reconciliation'],
                'inconsistent_days':sum(r['status']=='inconsistent' for r in reconciliation),
                'missing_days':sum(r['status']=='missing_readings' for r in reconciliation),
                'scope':'Proposed arithmetic identities only; physical boundary unverified. No automatic corrections.',
                'rows':reconciliation})
        rcfg=load(a.response_config) if a.response_config else discover(bms)
        if weather is not None:
            rcfg['weather_columns']=['outdoor_temperature_c']
            weather.to_csv(out/'external-weather.csv',index=False);save(out/'weather-audit.json',weather_audit)
        save(out/'response-config.json',rcfg)
        prepared=out/'response';prepared.mkdir()
        feature_frame=None
        try:
            feature_frame,audit=features(bms,rcfg,cfg['timezone'],weather)
            feature_frame.to_csv(prepared/'daily-features.csv',index=False);save(prepared/'feature-audit.json',audit)
            pd.DataFrame([{'point':k,**v} for k,v in audit['signal_audit'].items()]).to_csv(prepared/'point-coverage.csv',index=False)
        except DataError as exc:save(prepared/'feature-audit.json',{'status':'unavailable','reason':str(exc)})
        setpoint_frame=setpoint_meta=None
        if cfg.get('setpoint_response',{}).get('enabled',False):
            try:
                setpoint_frame,setpoint_meta,sp_audit=setpoint_response.prepare(bms,rcfg,cfg['setpoint_response'],cfg['timezone'],a.end,weather)
                setpoint_frame.to_csv(prepared/'setpoint-daily-features.csv',index=False)
                save(prepared/'setpoint-feature-definition.json',setpoint_meta)
                save(prepared/'setpoint-feature-audit.json',sp_audit)
            except DataError as exc:save(prepared/'setpoint-feature-audit.json',{'status':'unavailable','reason':str(exc)})
        if a.prepare_only:
            events=[]
            for meter,g in meters.groupby('meter_id'):
                _,found,_=assess(g,{**cfg,'meter_id':str(meter)},a.start,a.end,reviews);events.extend(found)
            save(out/'data-quality-events.json',events)
            pd.DataFrame(events).to_csv(out/'data-quality-events.csv',index=False)
            print('Merged inputs and proposed mappings: '+str(out));return
        run(out/'merged-meter.csv',out/'merged-bms.csv',cfg,a.start,a.end,out,kind='normalized',
            progress=lambda e,i,n:print(f"[{i}/{n}] {e['meter_id']}: {e['status']}",flush=True),quality_reviews=reviews,weather_path=a.weather)
        if (out/'quality-reviewed-meter.csv').exists():meters=pd.read_csv(out/'quality-reviewed-meter.csv',dtype={'meter_id':str})
        scenarios={};raw_scenarios={}
        for name,path in (('optimized',a.future_bms),('old_policy',a.baseline_future_bms)):
            if path:
                raw,provenance=merge_bms([path]);raw_scenarios[name]=raw
                try:
                    ff,fa=features(raw,rcfg,cfg['timezone'],weather);scenarios[name]=ff
                except DataError as exc:fa={'legacy_response_status':'unavailable','reason':str(exc)}
                save(prepared/(name+'-scenario-audit.json'),{'source':provenance,'features':fa,
                    'scope':'User-supplied operating scenario; trajectory consistency and comparable external conditions not automatically verified'})
        summary=load(out/'summary.json');response_rows=[];setpoint_rows=[]
        overrides=load(a.reference_setpoints) if a.reference_setpoints else None
        for entry in summary['meters']:
            meter=entry['meter_id'];folder=out/entry['folder']/'response';folder.mkdir(exist_ok=True)
            hist=meters[(meters.meter_id==meter)&(meters.date<a.end)]
            row={'meter_id':meter,'status':'unavailable'}
            try:
                if feature_frame is None:raise DataError('No usable BMS features; see feature-audit.json')
                # No data on/after reporting end may enter training or feature selection.
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
                        released=not any(g in model['gates'] for g in ('heldout_model_error_requires_review','meter_quality_review_required','bms_signal_semantics_unverified'))
                        comparison.append({'date':day,'old_policy_kwh':old.get('bms_estimated_kwh'),'optimized_kwh':opt.get('bms_estimated_kwh'),
                            'projected_difference_kwh':value if released else None,'projected_difference_cad':value*a.rate if value is not None and released else None,
                            'diagnostic_projected_difference_kwh':value,'release_status':'conditional_requires_review' if released else 'withheld_requires_review',
                            'scope':'conditional_scenario_difference_not_verified_savings','semantics_verified':rcfg.get('semantics_verified',False)})
                    pd.DataFrame(comparison).to_csv(folder/'scenario-comparison.csv',index=False)
            except (DataError,ValueError,KeyError) as exc:row['reason']=str(exc)
            save(folder/'status.json',row);response_rows.append(row)
            if cfg.get('setpoint_response',{}).get('enabled',False):
                sfolder=out/entry['folder']/'setpoint-response';sfolder.mkdir(exist_ok=True)
                sr={'meter_id':meter,'status':'unavailable'}
                try:
                    if setpoint_frame is None:raise DataError('No eligible fixed setpoint cohort; see response/setpoint-feature-audit.json')
                    model,sp_summary,daily,held=setpoint_response.fit(hist,setpoint_frame,setpoint_meta,a.start,a.end,a.rate)
                    save(sfolder/'model.json',model);save(sfolder/'consumption-summary.json',sp_summary)
                    pd.DataFrame(daily).to_csv(sfolder/'reporting-daily.csv',index=False)
                    pd.DataFrame(held).to_csv(sfolder/'heldout.csv',index=False)
                    sr.update({k:v for k,v in sp_summary.items() if k not in ('heldout_metrics','calendar_comparator_metrics','gates')})
                    sr.update(heldout_cvrmse_pct=sp_summary['heldout_metrics']['cvrmse_pct'],
                        heldout_nmbe_pct=sp_summary['heldout_metrics']['nmbe_pct'],gates=';'.join(sp_summary['gates']))
                    future_estimates={}
                    for name,raw in raw_scenarios.items():
                        try:
                            predicted=setpoint_response.predict(model,raw,a.forecast_start,a.forecast_end,weather=weather)
                            save(sfolder/(name+'-future.json'),predicted);future_estimates[name]=predicted
                        except (DataError,ValueError,KeyError) as exc:
                            save(sfolder/(name+'-future-status.json'),{'status':'unavailable','reason':str(exc)})
                    if 'optimized' in future_estimates and 'old_policy' in future_estimates:
                        reference={p['date']:p for p in future_estimates['old_policy']};scenario_rows=[]
                        for p in future_estimates['optimized']:
                            q=reference.get(p['date'],{})
                            good=all(x.get('conditional_setpoint_kwh') is not None and not x.get('outside_training_support',True) for x in (p,q))
                            value=q['conditional_setpoint_kwh']-p['conditional_setpoint_kwh'] if good else None
                            scenario_rows.append({'date':p['date'],'diagnostic_conditional_difference_kwh':value,
                                'diagnostic_conditional_difference_cad':value*a.rate if value is not None else None,
                                'savings_cad':None,'scope':'Supplied operating scenarios; comparability, thermal dynamics and causal attribution require review'})
                        pd.DataFrame(scenario_rows).to_csv(sfolder/'future-scenario-comparison.csv',index=False)
                    result_path=out/entry['folder']/'savings/result.json'
                    if result_path.exists():
                        result=load(result_path)
                        baselines={str(pd.Timestamp(p['timestamp']).tz_convert(cfg['timezone']).date()):p['baseline_kwh'] for p in result['periods']}
                        comparison=[]
                        for p in daily:
                            bp=baselines.get(p['date']);sp=p['heldout_setpoint_kwh']
                            value=bp-sp if bp is not None and sp is not None else None
                            comparison.append({**p,'baseline_kwh':bp,
                                'diagnostic_baseline_minus_heldout_setpoint_kwh':value,
                                'diagnostic_baseline_minus_heldout_setpoint_cad':value*a.rate if value is not None else None,
                                'scope':'Matched heldout days only. Model-to-model comparison, not measured or exact savings.'})
                        pd.DataFrame(comparison).to_csv(sfolder/'baseline-comparison.csv',index=False)
                    if overrides is not None:
                        actual=setpoint_response.predict(model,bms,a.start,a.end,weather=weather)
                        old=setpoint_response.predict(model,bms,a.start,a.end,overrides,weather)
                        comparison=[]
                        for x,y in zip(actual,old):
                            good=all(z['conditional_setpoint_kwh'] is not None and not z['outside_training_support'] for z in (x,y))
                            difference=y['conditional_setpoint_kwh']-x['conditional_setpoint_kwh'] if good else None
                            comparison.append({'date':x['date'],'actual_setpoints_kwh':x['conditional_setpoint_kwh'],
                                'reference_setpoints_kwh':y['conditional_setpoint_kwh'],
                                'diagnostic_conditional_difference_kwh':difference,
                                'diagnostic_conditional_difference_cad':difference*a.rate if difference is not None else None,
                                'savings_cad':None,'scope':'Conditional reconstruction with fixed observed temperatures; not a causal old-policy savings estimate'})
                        pd.DataFrame(comparison).to_csv(sfolder/'reference-setpoint-comparison.csv',index=False)
                except (DataError,ValueError,KeyError) as exc:sr['reason']=str(exc)
                save(sfolder/'status.json',sr);setpoint_rows.append(sr)
        pd.DataFrame(response_rows).to_csv(out/'response-summary.csv',index=False)
        if setpoint_rows:pd.DataFrame(setpoint_rows).to_csv(out/'setpoint-response-summary.csv',index=False)
        build(out,a.rate,a.forecast_start,a.forecast_end,a.total_meters,weather)
        save(out/'analysis-assumptions.json',{'start':a.start,'end_exclusive':a.end,'full_control_start':cfg['intervention_timestamp'],
            'confirmation':'User confirmed all assets under optimized control from August 15, 2026',
            'baseline_end_exclusive':cfg.get('baseline_end_exclusive'),'rate_cad_per_kwh':a.rate,
            'bms_role':'Held-out estimates are validation evidence. Full-fit diagnostics are in-sample, not independent savings. Missing meter energy never backfilled.',
            'future_role':'Baseline forecast always separate. Optimized usage requires explicit future operating trajectory; old-policy scenario differences require both trajectories.',
            'external_weather_supplied':weather is not None,
            'scope':'Engineering comparison; outdoor temperature adjusted when supplied, meter boundary/BMS semantics/non-control changes require review'})
        print('Open '+str(out/'index.html'));print('Response diagnostics: '+str(out/'response-summary.csv'))
    except (DataError,ValueError,KeyError,FileNotFoundError) as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
