from pathlib import Path
import copy,json
import numpy as np
import pandas as pd
import pytest
from option_c.data import DataError,digest,records,load
from option_c import strict_history,power_model,thermal_model,policy_forecast,weather_scenario


def synthetic(days=100):
    rng=np.random.default_rng(15);cad=15;start='2025-01-01';grid=pd.date_range(start,periods=days*96,freq='15min',tz='UTC')
    day=np.arange(len(grid))//96;spday=rng.uniform(21.5,27,days);outday=rng.uniform(23,31,days)
    t=np.empty(len(grid));t[0]=24
    for i in range(len(t)-1):t[i+1]=t[i]+.25*(.04+.08*(outday[day[i]]-t[i])-.8*max(t[i]-spday[day[i]],0))
    raw=pd.DataFrame({'timestamp':grid,'A_ZoneTemp':t,'A_CoolSetpointU_AV':spday[day]})
    dates=pd.date_range(start,periods=days);weather=pd.DataFrame({'date':dates.date.astype(str),'temperature_c':outday})
    groups={'A':{'temperature':'A_ZoneTemp','cool_setpoint':'A_CoolSetpointU_AV'}}
    ff,_=strict_history.daily(raw,groups,weather,'UTC',15)
    meter=pd.DataFrame({'date':ff.date,'meter_id':'M','energy_kwh':24*20+5*ff.cool_degree_hours+1*ff.cool_proxy_hours,'quality':'good'})
    return raw,groups,weather,ff,meter,str((dates[-1]+pd.Timedelta(days=1)).date())


def policy(kind,sp):
    return {'kind':kind,'reference':'Synthetic policy test','reviewed':True,'assets':{'A':{'weekdays':list(range(7)),'occupied_start':'00:00','occupied_end':'00:00',
        'occupied_cooling_c':sp,'unoccupied_cooling_c':sp,'occupied_heating_c':'OFF','unoccupied_heating_c':'OFF','control_margin_c':0.}}}


def test_strict_missing_any_field_rejects_row_no_forward_fill():
    raw=pd.DataFrame({'timestamp':['2026-01-01T00:00:00+00:00','2026-01-01T00:15:00+00:00','2026-01-01T00:30:00+00:00'],
                      'A':[1,np.nan,2],'B':[0,1,' ']})
    kept,reject,audit=strict_history.clean(raw)
    assert len(kept)==1 and audit['rejected_rows']==2 and not audit['imputation']
    limited,_,_=strict_history.clean(raw,['A'])
    assert len(limited)==2  # Only an explicit opt-in can limit completeness scope.


def test_missing_slot_excludes_day_and_breaks_thermal_pair():
    raw,g,w,ff,m,end=synthetic(3);raw=raw.drop(index=40)
    clean,_,_=strict_history.clean(raw);daily,coverage=strict_history.daily(clean,g,w,'UTC')
    assert len(daily)==2 and not coverage[0]['eligible']
    assert coverage[0]['accepted_slot_hours']==23.75


def test_exact_local_day_coverage_handles_dst():
    grid=pd.date_range('2026-03-08','2026-03-09',freq='15min',tz='America/Toronto',inclusive='left')
    raw=pd.DataFrame({'timestamp':grid.tz_convert('UTC'),'T':24.,'S':23.});g={'A':{'temperature':'T','cool_setpoint':'S'}}
    w=pd.DataFrame({'date':['2026-03-08'],'temperature_c':[10.]})
    f,c=strict_history.daily(raw,g,w,'America/Toronto');assert f.hours.iloc[0]==23 and f.cool_degree_hours.iloc[0]==23


def test_power_heldout_targets_cannot_change_frozen_predictions():
    raw,g,w,ff,m,end=synthetic();f=m.merge(ff,on='date');model,rows=power_model.fit(f,'bms','UTC',end)
    altered=f.copy();altered.loc[altered.date>=model['holdout_start'],'energy_kwh']*=2
    second,rows2=power_model.fit(altered,'bms','UTC',end)
    assert second['alpha']==model['alpha']
    assert second['frozen_holdout_coefficients']==model['frozen_holdout_coefficients']
    assert [r['heldout_kwh'] for r in rows]==[r['heldout_kwh'] for r in rows2]
    assert model['holdout_metrics']['cvrmse_pct']<1.


def test_weather_thermal_policy_forecasts_and_same_condition_savings():
    raw,g,w,ff,m,end=synthetic();electrical,_=power_model.fit(m.merge(ff,on='date'),'bms','UTC',end)
    thermal=thermal_model.fit(raw,g,w,'UTC',end)
    assert thermal['all_assets_validated'],thermal
    assert thermal['assets']['A']['heldout_rollout_rmse_c']<.3
    finish=str((pd.Timestamp(end)+pd.Timedelta(days=3)).date());future=pd.DataFrame({'date':pd.date_range(end,finish,inclusive='left').date.astype(str),'temperature_c':28.})
    initial,ia=policy_forecast.initial_state(raw,g,thermal,w,end)
    old,_=policy_forecast.rollout(electrical,thermal,policy('old_policy',22.),future,end,finish,initial,'Synthetic common future weather')
    optimized,_=policy_forecast.rollout(electrical,thermal,policy('optimized_policy',26.),future,end,finish,initial,'Synthetic common future weather')
    result=policy_forecast.compare(old,optimized)
    assert old['forecast_kwh']>optimized['forecast_kwh']>0
    assert result['projected_difference_cad']==pytest.approx(result['projected_difference_kwh']*.12)
    assert result['verified_savings_cad'] is None
    changed=copy.deepcopy(optimized);changed['conditions_sha256']='different';changed['forecast_id']=digest({k:v for k,v in changed.items() if k!='forecast_id'})
    with pytest.raises(DataError,match='mismatch'):policy_forecast.compare(old,changed)
    with pytest.raises(DataError,match='supply explicit initial'):policy_forecast.initial_state(raw,g,thermal,w,str((pd.Timestamp(end)+pd.Timedelta(days=2)).date()))


def test_thermal_holdout_does_not_change_selected_frozen_coefficients():
    raw,g,w,ff,m,end=synthetic();model=thermal_model.fit(raw,g,w,'UTC',end)
    changed=raw.copy();cut=pd.Timestamp(end,tz='UTC')-pd.Timedelta(days=14)
    changed.loc[changed.timestamp>=cut,'A_ZoneTemp']+=.2
    other=thermal_model.fit(changed,g,w,'UTC',end)
    assert model['assets']['A']['alpha']==other['assets']['A']['alpha']
    assert model['assets']['A']['frozen_holdout_coefficients']==other['assets']['A']['frozen_holdout_coefficients']


def test_unfilled_policy_or_external_trajectory_is_not_accepted():
    with pytest.raises(DataError):policy_forecast.validate({'kind':'old_policy','reference':'REPLACE','assets':{}},['A'],'old_policy')
    p=policy('old_policy',None)
    with pytest.raises(DataError,match='Setpoints'):policy_forecast.validate(p,['A'],'old_policy')


def test_weather_climatology_uses_only_past_dates_and_is_labeled():
    dates=pd.date_range('2023-01-01','2026-12-31');w=pd.DataFrame({'date':dates.date.astype(str),'temperature_c':10.})
    w.loc[w.date>='2026-01-01','temperature_c']=100.
    scenario,audit=weather_scenario.climatology(w,'2026-01-01','2026-01-04')
    assert scenario.temperature_c.eq(10).all()
    assert 'not_weather_forecast' in audit['kind']


def test_external_trajectory_adapter_checks_binding_and_missing_rows():
    raw,g,w,ff,m,end=synthetic();power,_=power_model.fit(m.merge(ff,on='date'),'bms','UTC',end)
    thermal=thermal_model.fit(raw,g,w,'UTC',end);initial,_=policy_forecast.initial_state(raw,g,thermal,w,end)
    finish=str((pd.Timestamp(end)+pd.Timedelta(days=2)).date());weather=pd.DataFrame({'date':pd.date_range(end,finish,inclusive='left').date.astype(str),'temperature_c':28.})
    pol=policy('optimized_policy',25.);original,trajectory=policy_forecast.rollout(power,thermal,pol,weather,end,finish,initial,'Test scenario')
    evidence={'model_id':'synthetic-neural-ode','validation_reference':'synthetic fixture only','policy_sha256':original['policy_sha256'],'conditions_sha256':original['conditions_sha256']}
    result,_=policy_forecast.rollout(power,thermal,pol,weather,end,finish,initial,'Test scenario',external=trajectory,evidence=evidence)
    assert result['forecast_kwh']==pytest.approx(original['forecast_kwh'])
    with pytest.raises(DataError,match='complete and finite'):
        policy_forecast.rollout(power,thermal,pol,weather,end,finish,initial,'Test scenario',external=trajectory.iloc[1:],evidence=evidence)
    evidence['conditions_sha256']='wrong'
    with pytest.raises(DataError,match='mismatch'):
        policy_forecast.rollout(power,thermal,pol,weather,end,finish,initial,'Test scenario',external=trajectory,evidence=evidence)


def test_full_cli_executes_all_forecast_and_historical_paths(tmp_path):
    import subprocess,sys,os
    raw,g,w,ff,m,end=synthetic(240);root=Path(__file__).parents[1]
    mpath=tmp_path/'meter.csv';bpath=tmp_path/'bms.csv';wpath=tmp_path/'weather.csv';mpath.write_text(m.to_csv(index=False));raw.to_csv(bpath,index=False);w.to_csv(wpath,index=False)
    finish=str((pd.Timestamp(end)+pd.Timedelta(days=3)).date());fpath=tmp_path/'future.csv'
    pd.DataFrame({'date':pd.date_range(end,finish,inclusive='left').date.astype(str),'temperature_c':28.}).to_csv(fpath,index=False)
    old=tmp_path/'old.json';opt=tmp_path/'opt.json';old.write_text(json.dumps(policy('old_policy',22.)));opt.write_text(json.dumps(policy('optimized_policy',26.)))
    cfg=load(root/'config/promenade-test.json');cfg.update(timezone='UTC',intervention_timestamp='2025-07-01T00:00:00+00:00',baseline_end_exclusive='2025-07-01T00:00:00+00:00')
    cfg['baseline_options'].update(windows_days=[None],estimators=['ols'],annual_calendar_candidates=False,bootstrap_repeats=0)
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg));out=tmp_path/'out';env=os.environ.copy();env['PYTHONPATH']=str(root/'src')
    process=subprocess.run([sys.executable,str(root/'scripts/run_project.py'),'--meter',str(mpath),'--meter-kind','normalized','--bms',str(bpath),'--weather',str(wpath),
        '--future-weather',str(fpath),'--future-weather-basis','Synthetic supplied forecast scenario','--old-policy',str(old),'--optimized-policy',str(opt),
        '--config',str(config),'--start','2025-07-01','--end',end,'--forecast-start',end,'--forecast-end',finish,'--output',str(out)],env=env,cwd=root,capture_output=True,text=True)
    assert process.returncode==0,process.stdout+process.stderr
    result=load(out/'project-summary.json');row=result['meters'][0]
    assert result['bms_source_rows_retained']==len(raw)
    assert row.get('meter_weather_forecast_kwh',0)>0,row
    assert row.get('old_policy_forecast_kwh',0)>row.get('optimized_policy_forecast_kwh',0)>0,row
    assert row['historical_old_policy_replay_paired_days']>0,row
    assert result['verified_savings_cad'] is None
    assert (out/'project.html').exists()


def test_overnight_schedule_uses_previous_weekday_after_midnight():
    p=policy('old_policy',22.)['assets']['A'];p.update(weekdays=[0],occupied_start='22:00',occupied_end='06:00',unoccupied_cooling_c=30.)
    assert policy_forecast.schedule(pd.Timestamp('2026-10-26T23:00:00Z'),p)[0]==22.
    assert policy_forecast.schedule(pd.Timestamp('2026-10-27T03:00:00Z'),p)[0]==22.
    assert policy_forecast.schedule(pd.Timestamp('2026-10-27T12:00:00Z'),p)[0]==30.


def test_long_unobserved_gap_is_not_used_as_a_thermal_adjacent_observation():
    raw,g,w,ff,m,end=synthetic();a=thermal_model.fit(raw,g,w,'UTC',end)
    b=thermal_model.fit(raw.drop(index=list(range(40,44))),g,w,'UTC',end)
    assert b['assets']['A']['training_adjacent_pairs']==a['assets']['A']['training_adjacent_pairs']-5


def test_unrelated_sparse_column_does_not_discard_usable_dates(tmp_path):
    import subprocess,sys,os
    raw,g,w,ff,m,end=synthetic(240);raw['unused_sparse_signal']=np.nan
    mp=tmp_path/'meter.csv';bp=tmp_path/'bms.csv';wp=tmp_path/'weather.csv';m.to_csv(mp,index=False);raw.to_csv(bp,index=False);w.to_csv(wp,index=False)
    root=Path(__file__).parents[1];cfg=load(root/'config/promenade-test.json')
    cfg.update(timezone='UTC',intervention_timestamp='2025-07-01T00:00:00+00:00',baseline_end_exclusive='2025-07-01T00:00:00+00:00')
    cfg['baseline_options'].update(windows_days=[None],estimators=['ols'],annual_calendar_candidates=False,bootstrap_repeats=0)
    cp=tmp_path/'config.json';cp.write_text(json.dumps(cfg));out=tmp_path/'run';finish=str((pd.Timestamp(end)+pd.Timedelta(days=3)).date())
    env=os.environ.copy();env['PYTHONPATH']=str(root/'src')
    r=subprocess.run([sys.executable,str(root/'scripts/run_project.py'),'--meter',str(mp),'--meter-kind','normalized','--bms',str(bp),'--weather',str(wp),'--config',str(cp),'--start','2025-07-01','--end',end,'--forecast-start',end,'--forecast-end',finish,'--output',str(out)],cwd=root,env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    result=load(out/'project-summary.json');assert result['bms_source_rows_retained']==len(raw)
    assert result['eligible_bms_days']==240
    assert result['meters'][0]['bms_validation_passed']
    assert (out/'meter-consumption.csv').exists()
    assert list(out.glob('meter-*/bms-power-model.json'))
