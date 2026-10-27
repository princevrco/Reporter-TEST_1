"""Complete meter/weather/BMS/policy workflow with date-level BMS feature eligibility."""
import argparse,hashlib,json,html
from pathlib import Path
import numpy as np
import pandas as pd
from option_c.data import DataError,load,save,records,digest
from option_c.ingestion import merge_meters
from option_c.weather import read as read_weather,join as join_weather
from option_c import strict_history,date_history,power_model,thermal_model,policy_forecast,weather_scenario,observation_mask
from option_c.batch import run as baseline_run
from option_c.overview import build


def policy_template(groups,kind):
    return {'kind':kind,'reference':'REPLACE with documented policy source','reviewed':False,'assets':{
        a:{'weekdays':[],'occupied_start':None,'occupied_end':None,'occupied_cooling_c':None,'unoccupied_cooling_c':None,
           'occupied_heating_c':None,'unoccupied_heating_c':None,'control_margin_c':None} for a in groups}}


def future_features(weather,start,end,tz):
    grid=pd.date_range(pd.Timestamp(start,tz=tz),pd.Timestamp(end,tz=tz),freq='D',inclusive='left')
    f=pd.DataFrame({'date':grid.date.astype(str),'hours':[(t+pd.DateOffset(days=1)-t).total_seconds()/3600 for t in grid]})
    return join_weather(f,weather)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meter',type=Path,nargs='+',required=True);p.add_argument('--meter-kind',default='auto',choices=['auto','utility','carma','normalized','portal'])
    p.add_argument('--bms',type=Path,nargs='+',required=True);p.add_argument('--weather',type=Path,required=True)
    p.add_argument('--bms-observation-mask',type=Path,help='Verified native observation NPZ for one BMS CSV; otherwise auto-detect <CSV stem>.observations.npz')
    p.add_argument('--future-weather',type=Path);p.add_argument('--future-weather-basis',help='Reference for supplied future forecast/scenario')
    p.add_argument('--weather-scenario',choices=['climatology'],help='Explicit alternative: historical climatology scenario, not a forecast')
    p.add_argument('--config',type=Path,default=Path('config/promenade-test.json'));p.add_argument('--signal-config',type=Path)
    p.add_argument('--required-bms-columns',nargs='+',help='Explicit additional required signals for date eligibility; unrelated columns are ignored')
    p.add_argument('--cadence-minutes',type=int,default=15,choices=[5,10,15,20,30,60])
    p.add_argument('--old-policy',type=Path);p.add_argument('--optimized-policy',type=Path)
    p.add_argument('--initial-state',type=Path);p.add_argument('--historical-initial-state',type=Path)
    p.add_argument('--old-trajectory',type=Path);p.add_argument('--optimized-trajectory',type=Path)
    p.add_argument('--old-trajectory-evidence',type=Path);p.add_argument('--optimized-trajectory-evidence',type=Path)
    p.add_argument('--meter-ids',nargs='+');p.add_argument('--start',default='2026-08-15');p.add_argument('--end',default='2026-10-21',help='Exclusive local date')
    p.add_argument('--forecast-start',required=True);p.add_argument('--forecast-end',required=True,help='Exclusive local date')
    p.add_argument('--rate',type=float,default=.12);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.output.exists():raise DataError('Choose a new output directory')
        if not np.isfinite(a.rate) or a.rate<0:raise DataError('Invalid energy rate')
        if a.future_weather and a.weather_scenario:raise DataError('Choose supplied future weather or explicit climatology, not both')
        cfg=load(a.config);tz=cfg['timezone'];start=pd.Timestamp(a.start);end=pd.Timestamp(a.end)
        fs=pd.Timestamp(a.forecast_start);fe=pd.Timestamp(a.forecast_end)
        if any(len(s)!=10 for s in (a.start,a.end,a.forecast_start,a.forecast_end)) or end<=start or fe<=fs or fs<end:raise DataError('Use increasing local date windows; independent future forecasts must follow reporting end')
        if start.date()<pd.Timestamp(cfg['intervention_timestamp']).date():raise DataError('Historical savings must follow intervention')
        if (fe-fs).days>366:raise DataError('Forecast window limited to one year')
        meters,ma=merge_meters(a.meter,kind=a.meter_kind)
        if a.meter_ids:
            if len(set(a.meter_ids))!=len(a.meter_ids) or not set(a.meter_ids)<=set(meters.meter_id):raise DataError('Unknown/duplicate meter IDs')
            meters=meters[meters.meter_id.isin(a.meter_ids)].copy()
        weather,wa=read_weather(a.weather,tz)
        historical=meters[meters.date<a.end].copy();join_weather(historical,weather)
        sources=[];accepted=[];rejected=[];columns=None
        for path in a.bms:
            raw=pd.read_csv(path)
            if columns is None:columns=list(raw.columns)
            clean,bad,audit=date_history.clean(raw,a.required_bms_columns)
            audit.update(filename=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest());sources.append(audit)
            bad['source_file']=path.name;rejected.append(bad);accepted.append(clean)
        clean=pd.concat(accepted,ignore_index=True)
        if len(clean):
            for t,g in clean[clean.timestamp.duplicated(keep=False)].groupby('timestamp'):
                if (g.drop(columns='timestamp').nunique(dropna=True)>1).any():raise DataError('Conflicting BMS observations at '+str(t))
            clean=clean.groupby('timestamp',as_index=False).last().sort_values('timestamp').reset_index(drop=True)
            clean=clean[clean.timestamp<pd.Timestamp(a.end,tz=tz).tz_convert('UTC')].copy()
        native=None;mask_path=None
        if a.bms_observation_mask and len(a.bms)!=1:raise DataError('Native observation mask currently requires one combined BMS CSV')
        if len(a.bms)==1:
            candidate=a.bms_observation_mask or a.bms[0].with_suffix('.observations.npz')
            if a.bms_observation_mask or candidate.exists():
                native=observation_mask.read(candidate,a.bms[0]);mask_path=candidate
                if list(native.columns)!=[c for c in columns if c!='timestamp']:raise DataError('Observation mask columns do not match BMS CSV')
                original=pd.read_csv(a.bms[0],usecols=['timestamp'])
                if not pd.DatetimeIndex(pd.to_datetime(original.timestamp,utc=True,format='mixed')).equals(native.index):
                    raise DataError('Observation mask timestamps do not match BMS CSV')
        opc=load(a.signal_config) if a.signal_config else None
        clean,groups,asset_daily,coverage,signal_audit=date_history.prepare(clean,weather,tz,a.end,opc,a.cadence_minutes,a.required_bms_columns,observation_mask=native)
        eligible_dates=set(asset_daily.date)
        eligible_observations=clean[clean.timestamp.dt.tz_convert(tz).dt.date.astype(str).isin(eligible_dates)].copy()
        out=a.output;out.mkdir(parents=True)
        clean.to_csv(out/'bms-observations.csv',index=False);eligible_observations.to_csv(out/'eligible-bms-observations.csv',index=False);pd.concat(rejected,ignore_index=True).to_csv(out/'rejected-bms-rows.csv',index=False)
        meters.to_csv(out/'merged-meter.csv',index=False)
        save(out/'bms-date-audit.json',{'sources':sources,'accepted_before_signal_checks':sum(s['accepted_rows'] for s in sources),
            'usable_rows_before_reporting_end':len(clean),'daily_feature_audit':signal_audit,'date_policy':'Retain sparse observations. Exclude dates lacking required daily features; never fill missing dates.'})
        save(out/'input-audit.json',{'meter':ma,'weather':wa,'mapping':groups,'signal_config':opc,'native_observation_mask':str(mask_path) if mask_path else None,'bms_observations_sha256':hashlib.sha256((out/'bms-observations.csv').read_bytes()).hexdigest()})
        for kind in ('old_policy','optimized_policy'):save(out/(kind+'-request.json'),policy_template(groups,kind))
        save(out/'initial-state-request.json',{'timestamp':pd.Timestamp(a.forecast_start,tz=tz).isoformat(),'reference':'REPLACE with measured state or validated initialization source','temperatures_c':{asset:None for asset in groups}})
        cfg['allow_empty_bms_context']=True
        baseline_run(out/'merged-meter.csv',out/'bms-observations.csv',cfg,a.start,a.end,out,kind='normalized',weather_path=a.weather,
            progress=lambda e,i,n:print(f"Meter baseline {i}/{n}: {e['meter_id']} {e['status']}",flush=True))
        reviewed=pd.read_csv(out/'quality-reviewed-meter.csv',dtype={'meter_id':str}) if (out/'quality-reviewed-meter.csv').exists() else meters
        reviewed=join_weather(reviewed[reviewed.date<a.end],weather)
        fd=None;fw=None;weather_basis=None;forecast_block=None
        try:
            if a.future_weather:
                if not a.future_weather_basis:raise DataError('Declare --future-weather-basis for supplied future inputs')
                fw,fwa=read_weather(a.future_weather,tz);weather_basis=a.future_weather_basis;save(out/'future-weather-audit.json',fwa)
            elif a.weather_scenario:
                fw,fwa=weather_scenario.climatology(weather,a.forecast_start,a.forecast_end);weather_basis=fwa['kind'];save(out/'future-weather-audit.json',fwa)
            else:raise DataError('Supply --future-weather with its basis or explicitly choose --weather-scenario climatology')
            fd=future_features(fw,a.forecast_start,a.forecast_end,tz);fw.to_csv(out/'future-weather.csv',index=False)
        except (DataError,ValueError,KeyError) as exc:forecast_block=str(exc)
        combined=pd.concat([weather,fw],ignore_index=True).drop_duplicates('date',keep='last') if fw is not None else weather
        # Preserve the original overview, then create a distinct end-to-end dashboard.
        build(out,a.rate,a.forecast_start,a.forecast_end,weather=combined if fw is not None else weather)
        pd.DataFrame(coverage).to_csv(out/'bms-date-coverage.csv',index=False);asset_daily.to_csv(out/'asset-daily.csv',index=False)
        thermal=None;thermal_error=None
        try:
            thermal=thermal_model.fit(eligible_observations,groups,weather,tz,a.end,a.cadence_minutes);save(out/'thermal-model.json',thermal)
        except (DataError,ValueError,KeyError) as exc:thermal_error=str(exc)
        old=load(a.old_policy) if a.old_policy else None;opt=load(a.optimized_policy) if a.optimized_policy else None
        initial=None;initial_error=None
        if thermal is not None:
            try:
                initial,ia=policy_forecast.initial_state(eligible_observations,groups,thermal,weather,a.forecast_start,load(a.initial_state) if a.initial_state else None)
                save(out/'forecast-initial-state.json',{'temperatures_c':initial,'audit':ia})
            except (DataError,ValueError,KeyError) as exc:initial_error=str(exc)
        summary=[]
        for mid,g in reviewed.groupby('meter_id',sort=True):
            folder=out/('meter-'+hashlib.sha256(mid.encode()).hexdigest()[:16]);row={'meter_id':mid,'scope':'Separate meter; no automatic site aggregation','historical_actual_vs_baseline':'See savings/result.json; incomplete/quality gates retained'}
            local=pd.to_datetime(g.date);g=g.copy();g['hours']=[(pd.Timestamp(d,tz=tz)+pd.DateOffset(days=1)-pd.Timestamp(d,tz=tz)).total_seconds()/3600 for d in g.date]
            try:
                recent=g[g.date>=str(pd.Timestamp(cfg['intervention_timestamp']).date())]
                meter_model,diagnostic=power_model.fit(recent,'meter',tz,a.end,28);save(folder/'meter-weather-model.json',meter_model)
                if fd is None:raise DataError(forecast_block)
                pred=power_model.predict(meter_model,fd);value=sum(r['predicted_kwh'] for r in pred)
                save(folder/'meter-weather-forecast.json',{'forecast_kwh':value,'energy_cad':value*a.rate,'daily':pred,'model_id':meter_model['model_id'],
                    'status':'conditional_forecast_requires_review' if meter_model['validation_passed'] else 'diagnostic_forecast_validation_failed',
                    'gates':meter_model['gates']+(['future_conditions_outside_training_support'] if any(r['outside_training_support'] for r in pred) else [])+['future_weather_is_a_supplied_forecast_or_scenario','complete_projection_uncertainty_budget_required'],'weather_basis':weather_basis,'scope':'Recent controlled meter/weather pattern continuation, not an old-policy or setpoint forecast'})
                row.update(meter_weather_forecast_kwh=value,meter_weather_forecast_cad=value*a.rate,meter_weather_validation_passed=meter_model['validation_passed'],meter_weather_outside_support=any(r['outside_training_support'] for r in pred))
            except (DataError,ValueError,KeyError) as exc:row['meter_weather_forecast_blocker']=str(exc)
            electrical=None
            try:
                if asset_daily.empty:raise DataError('No eligible BMS dates with the required daily features; missing dates are not filled')
                f=g.drop(columns=['hours','temperature_c'],errors='ignore').merge(asset_daily,on='date',validate='one_to_one')
                electrical,reconstruction=power_model.fit(f,'bms',tz,a.end,40)
                electrical['mapping']=groups;electrical['model_id']=digest({k:v for k,v in electrical.items() if k!='model_id'})
                save(folder/'bms-power-model.json',electrical);pd.DataFrame(reconstruction).to_csv(folder/'bms-consumption-diagnostics.csv',index=False)
                report=[r for r in reconstruction if a.start<=r['date']<a.end]
                held=[r for r in report if r['heldout_kwh'] is not None]
                row.update(bms_reconstruction_days=len(report),bms_independent_test_days=len(held),bms_validation_passed=electrical['validation_passed'],
                    bms_holdout_cvrmse_pct=electrical['holdout_metrics']['cvrmse_pct'],bms_holdout_nmbe_pct=electrical['holdout_metrics']['nmbe_pct'],
                    full_period_bms_kwh=sum(r['reconstructed_kwh'] for r in report) if len(report)==(end-start).days else None,
                    heldout_meter_minus_bms_kwh=sum(r['meter_kwh']-r['heldout_kwh'] for r in held) if held else None)
            except (DataError,ValueError,KeyError) as exc:row['bms_model_blocker']=str(exc)
            forecasts={}
            for kind,policy,trajectory,evidence in [('old_policy',old,a.old_trajectory,a.old_trajectory_evidence),('optimized_policy',opt,a.optimized_trajectory,a.optimized_trajectory_evidence)]:
                try:
                    if policy is None:raise DataError('Provide documented '+kind+' schedule/setpoints; see generated request JSON')
                    policy_forecast.validate(policy,groups,kind)
                    if electrical is None:raise DataError(row.get('bms_model_blocker','No calibrated electrical model'))
                    if fd is None:raise DataError(forecast_block)
                    if thermal is None:raise DataError(thermal_error)
                    if initial is None:raise DataError(initial_error or 'Initial state unavailable')
                    # Save context hashes so Neural ODE/STGNN exporters can bind a trajectory correctly.
                    ph=policy_forecast.validate(policy,groups,kind)
                    ctx={'weather':{d:float(fw.set_index('date').temperature_c[d]) for d in fd.date},'initial':initial,'start':a.forecast_start,'end_exclusive':a.forecast_end,'weather_basis':weather_basis}
                    save(folder/(kind+'-trajectory-request.json'),{'policy_sha256':ph,'conditions_sha256':digest(ctx),'assets':list(groups),'cadence_minutes':a.cadence_minutes,'start':a.forecast_start,'end_exclusive':a.forecast_end,'initial_temperatures_c':initial,'required_columns':['timestamp','asset_id','predicted_zone_temperature_c']})
                    result,tf=policy_forecast.rollout(electrical,thermal,policy,fw,a.forecast_start,a.forecast_end,initial,weather_basis,a.rate,
                        pd.read_csv(trajectory) if trajectory else None,load(evidence) if evidence else None)
                    save(folder/(kind+'-forecast.json'),result);tf.to_csv(folder/(kind+'-temperature-trajectory.csv'),index=False);forecasts[kind]=result
                    row[kind+'_forecast_kwh']=result['forecast_kwh'];row[kind+'_forecast_cad']=result['forecast_energy_cad'];row[kind+'_forecast_status']=result['status']
                except (DataError,ValueError,KeyError) as exc:row[kind+'_forecast_blocker']=str(exc)
            if len(forecasts)==2:
                comparison=policy_forecast.compare(forecasts['old_policy'],forecasts['optimized_policy'],a.rate);save(folder/'projected-policy-difference.json',comparison)
                row.update(projected_difference_kwh=comparison['projected_difference_kwh'],projected_difference_cad=comparison['projected_difference_cad'],projected_difference_status=comparison['status'])
            # Independent historical old-policy replay needs models calibrated BEFORE reporting start.
            try:
                if old is None:raise DataError('Historical old-policy replay needs a documented old policy')
                policy_forecast.validate(old,groups,'old_policy')
                if asset_daily.empty:raise DataError('No eligible BMS dates for pre-reporting calibration')
                f=g.drop(columns=['hours','temperature_c'],errors='ignore').merge(asset_daily,on='date',validate='one_to_one')
                prior,_=power_model.fit(f,'bms',tz,a.start,40)
                tprior=thermal_model.fit(eligible_observations,groups,weather,tz,a.start,a.cadence_minutes)
                pre=eligible_observations[eligible_observations.timestamp<pd.Timestamp(a.start,tz=tz).tz_convert('UTC')]
                state,_=policy_forecast.initial_state(pre,groups,tprior,weather,a.start,load(a.historical_initial_state) if a.historical_initial_state else None)
                replay,tf=policy_forecast.rollout(prior,tprior,old,weather,a.start,a.end,state,'Historical daily weather on reporting dates',a.rate)
                actual=g.set_index('date').energy_kwh;paired=[(r,actual[r['date']]) for r in replay['daily'] if r['date'] in actual and np.isfinite(actual[r['date']])]
                value=sum(r['predicted_kwh']-v for r,v in paired) if paired else None
                save(folder/'historical-old-policy-replay.json',{'old_policy_model_kwh':replay['forecast_kwh'],'paired_days':len(paired),'expected_days':(end-start).days,
                    'paired_difference_kwh':value,'paired_difference_cad':value*a.rate if value is not None else None,'complete_difference_kwh':value if len(paired)==(end-start).days else None,
                    'scope':'Independent pre-reporting calibration; conditional old-policy simulation minus actual meter, not verified savings','gates':replay['gates']})
                row['historical_old_policy_replay_paired_days']=len(paired)
            except (DataError,ValueError,KeyError) as exc:row['historical_old_policy_replay_blocker']=str(exc)
            save(folder/'end-to-end-status.json',row);summary.append(row)
        result={'version':'0.9.2','start':a.start,'end_exclusive':a.end,'forecast_start':a.forecast_start,'forecast_end_exclusive':a.forecast_end,
                'rate_cad_per_kwh':a.rate,'synthetic':bool(cfg.get('synthetic',False)),'bms_source_rows_retained':len(clean),'eligible_bms_days':len(asset_daily),'bms_rows_on_eligible_dates':len(eligible_observations),'asset_count':len(groups),'weather_basis':weather_basis,
                'thermal_model_status':'validated' if thermal and thermal['all_assets_validated'] else 'unavailable_or_failed_validation',
                'thermal_blocker':thermal_error,'meters':summary,'verified_savings_cad':None,
                'scope':'Meter-model errors, historical avoided-energy estimates, meter-history forecasts and conditional future policy differences remain separate. No exact bill savings or automatic standards certification.'}
        save(out/'project-summary.json',result);pd.DataFrame(summary).to_csv(out/'project-meter-summary.csv',index=False)
        page='<meta charset="utf-8"><title>Promenade end-to-end energy workflow</title><style>body{font:15px system-ui;margin:30px;color:#163247}td,th{padding:9px;border-bottom:1px solid #ccd5dd;text-align:left}table{font-size:12px}pre{white-space:pre-wrap}.scroll{overflow:auto}</style><h1>Meter, weather and BMS policy forecasts</h1>'
        if result['synthetic']:page+='<h2>Synthetic demonstration â€” not Promenade evidence</h2>'
        page+='<p>'+html.escape(result['scope'])+'</p><p>BMS timestamp rows retained: <b>'+str(len(clean))+'</b>. Eligible BMS dates: '+str(len(asset_daily))+'. Thermal status: '+html.escape(result['thermal_model_status'])+'</p>'
        page+='<p><a href="index.html">Historical meter baseline dashboard</a> Â· <a href="project-meter-summary.csv">End-to-end CSV</a> Â· <a href="bms-date-audit.json">Date eligibility audit</a></p><div class="scroll">'+pd.DataFrame(summary).to_html(index=False,escape=True)+'</div>'
        (out/'project.html').write_text(page,encoding='utf-8')
        print('Open '+str(out/'project.html'));print('BMS rows retained: '+str(len(clean))+'; eligible dates: '+str(len(asset_daily)))
    except (DataError,ValueError,KeyError,FileNotFoundError) as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
