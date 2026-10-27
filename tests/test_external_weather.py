import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from option_c.data import DataError,records,load,save
from option_c.weather import read,join
from option_c import resilient,setpoint_response,response
from option_c.batch import run
from option_c.overview import build


def config():
    cfg=load(Path(__file__).parents[1]/'config/promenade-test.json')
    cfg.update(timezone='UTC',meter_id='A',boundary_description='Synthetic test',intervention_timestamp='2025-08-15T00:00:00+00:00',baseline_end_exclusive='2025-07-24T00:00:00+00:00')
    cfg['baseline_options'].update(windows_days=[None],estimators=['ols','huber'],annual_calendar_candidates=False,bootstrap_repeats=0)
    return cfg


def meter_weather():
    dates=pd.date_range('2024-01-01','2025-10-20')
    i=np.arange(len(dates));temperature=12+15*np.sin(i*.13)
    energy=24*(30+5*np.maximum(temperature-18.3,0)+3*np.maximum(18.3-temperature,0)+2*(dates.dayofweek==1))
    meter=pd.DataFrame({'date':dates.date.astype(str),'meter_id':'A','energy_kwh':energy,'quality':'good'})
    weather=pd.DataFrame({'date':meter.date,'temperature_c':temperature})
    return meter,weather


@pytest.mark.parametrize('problem',['duplicate','missing_value','fahrenheit','invalid_date','timezone'])
def test_weather_input_rejects_invalid_values(tmp_path,problem):
    f=pd.DataFrame({'date':['2025-01-01','2025-01-02'],'outdoor_temp_c':[1.,2.]})
    if problem=='duplicate':f.loc[1,'date']=f.loc[0,'date']
    if problem=='missing_value':f.loc[1,'outdoor_temp_c']=np.nan
    if problem=='fahrenheit':f['unit']='F'
    if problem=='invalid_date':f.loc[0,'date']='2025-02-30'
    if problem=='timezone':f['timezone']='America/Toronto'
    path=tmp_path/'weather.csv';f.to_csv(path,index=False)
    with pytest.raises(DataError):read(path,'UTC')


def test_daily_join_preserves_order_and_requires_matching_dates(tmp_path):
    f=pd.DataFrame({'date':['2025-01-02','2025-01-01'],'energy_kwh':[0.,123.]})
    w=pd.DataFrame({'date':['2025-01-01','2025-01-02'],'temperature_c':[5.,10.]})
    joined=join(f,w)
    assert joined.temperature_c.tolist()==[10.,5.]
    assert joined.energy_kwh.tolist()==[0.,123.]
    with pytest.raises(DataError,match='Missing external weather'):join(f,w.iloc[:1])


def test_robust_baseline_learns_temperature_without_changing_meter_energy():
    m,w=meter_weather();c=config();h=m[m.date<'2025-07-24'];f=join(h,w)
    no_weather=resilient.fit(records(h),c)
    c['external_weather_enabled']=True
    model=resilient.fit(records(f),c)
    assert model['external_weather_selected']
    assert model['holdout_metrics']['cvrmse_pct']<.01
    assert model['holdout_metrics']['rmse_kwh']<no_weather['holdout_metrics']['rmse_kwh']*.01
    assert f.energy_kwh.tolist()==h.energy_kwh.tolist()
    changed=f.copy();cutoff=pd.Timestamp(model['holdout_start']).date().isoformat()
    changed.loc[changed.date>=cutoff,'energy_kwh']*=1.2
    modified=resilient.fit(records(changed),c)
    assert modified['family']==model['family']
    old={x['name']:x['cv_rmse_kwh'] for x in model['selection_candidates'] if x['status']=='evaluated'}
    new={x['name']:x['cv_rmse_kwh'] for x in modified['selection_candidates'] if x['status']=='evaluated'}
    assert old==new
    assert modified['holdout_metrics']['rmse_kwh']>model['holdout_metrics']['rmse_kwh']


def test_batch_weather_and_future_missing_weather_are_explicit(tmp_path):
    m,w=meter_weather();mp=tmp_path/'meter.csv';wp=tmp_path/'weather.csv';bp=tmp_path/'bms.csv'
    m.to_csv(mp,index=False);w.to_csv(wp,index=False)
    pd.DataFrame({'timestamp':['2025-08-15T00:00:00+00:00'],'A_ZoneTemp':[24.]}).to_csv(bp,index=False)
    out=tmp_path/'results'
    result=run(mp,bp,config(),'2025-08-15','2025-10-21',out,kind='normalized',weather_path=wp)
    model=load(out/result['meters'][0]['folder']/'baseline-model.json')
    assert model['external_weather_selected']
    assert result['meters'][0]['meter_consumption_observed_days']==67
    report=build(out,forecast_start='2025-10-21',forecast_end='2025-10-28')
    assert report['meters'][0]['forecast_kwh'] is None
    assert 'Missing external weather' in report['meters'][0]['forecast_status']
    future=pd.DataFrame({'date':pd.date_range('2025-10-21','2025-10-27').date.astype(str),'temperature_c':20.})
    report=build(out,forecast_start='2025-10-21',forecast_end='2025-10-28',weather=pd.concat([w,future]))
    assert report['meters'][0]['forecast_kwh']>0


def test_setpoint_and_broad_response_use_weather_and_require_it_for_prediction():
    rng=np.random.default_rng(120);n=100
    days=pd.date_range('2026-05-01',periods=n)
    outdoor=rng.uniform(10,30,n);sp=rng.uniform(20,24,n);temp=rng.uniform(25,29,n)
    t=pd.date_range('2026-05-01',periods=n*96,freq='15min',tz='UTC');idx=np.arange(len(t))//96
    raw=pd.DataFrame({'timestamp':t,'A_CoolSetpointU_AV':sp[idx],'A_ZoneTemp':temp[idx]})
    weather=pd.DataFrame({'date':days.date.astype(str),'temperature_c':outdoor})
    cfg=response.discover(raw);cfg['weather_columns']=['outdoor_temperature_c']
    daily,audit=response.features(raw,cfg,'UTC',weather)
    meter=pd.DataFrame({'date':days.date.astype(str),'energy_kwh':24*(20+4*(temp-sp)+3*outdoor),'quality':'good'})
    full,meta,_=setpoint_response.prepare(raw,cfg,{},'UTC','2026-08-09',weather)
    model,summary,_,_=setpoint_response.fit(meter,full,meta,'2026-06-01','2026-08-09')
    assert 'outdoor_temperature_c' in model['features']
    assert summary['heldout_metrics']['cvrmse_pct']<1.
    assert 'outdoor_temperature_c' in meta['definition']
    predictions=setpoint_response.predict(model,raw,'2026-06-01','2026-08-09',weather=weather)
    assert any(r['conditional_setpoint_kwh'] is not None for r in predictions)
    with pytest.raises(DataError,match='requires supplied daily external weather'):
        setpoint_response.predict(model,raw,'2026-06-01','2026-08-09')
    broad,_=response.fit(meter,daily,cfg,'UTC')
    assert 'outdoor_temperature_c' in broad['features']
    assert broad['holdout_metrics']['cvrmse_pct']<1.
    assert 'weather_not_modeled' not in broad['gates']
    # Withheld targets cannot select the Ridge model or change frozen heldout predictions.
    cutoff=meta['cohort_selected_before'];changed=meter.copy()
    changed.loc[changed.date>=cutoff,'energy_kwh']*=2
    altered,_,_,held=setpoint_response.fit(changed,full,meta,'2026-06-01','2026-08-09')
    assert altered['family']==model['family'] and altered['alpha']==model['alpha']
    assert altered['selection_candidates']==model['selection_candidates']


def test_analyze_cli_weather_is_connected_end_to_end(tmp_path):
    import subprocess,sys,os
    from test_carma_directory import chunk
    repo=Path(__file__).parents[1];source=tmp_path/'carma';chunk(source,'A','2025-01-01','2026-10-21')
    t=pd.date_range('2026-07-24','2026-10-21',freq='15min',tz='America/Toronto',inclusive='left')
    i=np.arange(len(t))//96
    raw=pd.DataFrame({'timestamp':t.tz_convert('UTC'),'A_ZoneTemp':25+np.sin(i*.4),'A_CoolSetpointU_AV':22+np.cos(i*.3)})
    b=tmp_path/'bms.csv';raw.to_csv(b,index=False)
    dates=pd.date_range('2025-01-01','2027-01-21');w=pd.DataFrame({'date':dates.date.astype(str),'outdoor_temp_c':12+10*np.sin(np.arange(len(dates))*.08)})
    wp=tmp_path/'weather.csv';w.to_csv(wp,index=False);out=tmp_path/'out'
    env=os.environ.copy();env['PYTHONPATH']=str(repo/'src')
    r=subprocess.run([sys.executable,str(repo/'scripts/analyze_promenade.py'),'--meter',str(source),'--bms',str(b),'--weather',str(wp),'--meter-ids','A','--output',str(out),'--forecast-start','2026-10-21','--forecast-end','2027-01-21'],cwd=repo,env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    entry=load(out/'summary.json')['meters'][0];base=out/entry['folder']
    assert load(base/'baseline-model.json')['external_weather_supplied']
    assert 'outdoor_temperature_c' in load(base/'response/model.json')['features']
    assert 'outdoor_temperature_c' in load(base/'setpoint-response/model.json')['features']
    assert (out/'weather-audit.json').exists()
