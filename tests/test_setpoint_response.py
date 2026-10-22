import copy
import numpy as np
import pandas as pd
import pytest
from option_c.data import DataError
from option_c.setpoint_response import fit,prepare,predict
from option_c.response import discover
from option_c.regression import uncertainty_samples


def sample():
    rng=np.random.default_rng(27);dates=pd.date_range('2026-06-01',periods=80)
    sp=rng.uniform(20,24,len(dates));temp=rng.uniform(25,28,len(dates));change=rng.uniform(0,.1,len(dates))
    ff=pd.DataFrame({'date':dates.date.astype(str),'hours':24.,'cool_setpoint':sp,'temperature':temp,
        'cooling_demand':temp-sp,'setpoint_change':change,'calendar_weekend':(dates.dayofweek>=5).astype(float),
        'calendar_sin_year':np.sin(dates.dayofyear*2*np.pi/365),
        'calendar_cos_year':np.cos(dates.dayofyear*2*np.pi/365)})
    m=pd.DataFrame({'date':ff.date,'energy_kwh':24*(10+5*(temp-sp)+10*change),'quality':'good'})
    meta={'settings':{},'cohort_selected_before':'2026-08-06','semantics_verified':True,'cohort':['G']}
    return m,ff,meta


def test_setpoint_model_predicts_consumption_and_measures_incremental_value():
    m,f,c=sample();model,summary,rows,held=fit(m,f,c,'2026-07-01','2026-08-20')
    assert summary['heldout_metrics']['cvrmse_pct']<2
    assert summary['setpoint_rmse_improvement_pct']>80
    assert model['family']=='setpoint_changes'
    assert len(held)==14 and all(p['setpoint_estimated_kwh']>0 for p in held)
    assert model['training_days']>=28
    assert any(r['heldout_setpoint_kwh'] is None for r in rows)


def test_heldout_targets_never_select_or_train_the_frozen_predictor():
    m,f,c=sample();a,_,_,ah=fit(m,f,c,'2026-07-01','2026-08-20')
    changed=m.copy();changed.loc[changed.date.ge(c['cohort_selected_before']),'energy_kwh']*=2
    b,_,_,bh=fit(changed,f,c,'2026-07-01','2026-08-20')
    assert a['family']==b['family'] and a['alpha']==b['alpha']
    assert a['selection_candidates']==b['selection_candidates']
    assert [r['setpoint_estimated_kwh'] for r in ah]==[r['setpoint_estimated_kwh'] for r in bh]
    assert b['holdout_metrics']['cvrmse_pct']>20


def test_missing_bms_days_never_become_full_consumption_or_zero():
    m,f,c=sample();f=f[~f.date.isin(['2026-07-04','2026-07-05'])]
    _,summary,rows,_=fit(m,f,c,'2026-07-01','2026-08-20')
    assert summary['full_period_setpoint_consumption_kwh'] is None
    missing=[r for r in rows if r['date']=='2026-07-04'][0]
    assert missing['diagnostic_fitted_setpoint_kwh'] is None and missing['measured_kwh']>0


def test_unsupported_short_history_and_no_setpoint_variation_are_explicit():
    m,f,c=sample()
    with pytest.raises(DataError,match='training days'):fit(m.iloc[-20:],f,c,'2026-07-01','2026-08-20')
    frozen=f.copy();frozen['cool_setpoint']=22.;frozen['cooling_demand']=0.;frozen['setpoint_change']=0.
    with pytest.raises(DataError,match='not identifiable'):fit(m,frozen,c,'2026-07-01','2026-08-20')


def test_cohort_is_fixed_from_past_availability_and_scenario_checks_support():
    rng=np.random.default_rng(5)
    t=pd.date_range('2026-06-01',periods=75*24*6,freq='10min',tz='UTC')
    day=np.arange(len(t))//144;sp=22+np.sin(day*.9);temp=26+np.cos(day*.7)
    raw=pd.DataFrame({'timestamp':t,'G_CoolSetpointU_AV':sp,'G_ZoneTemp':temp})
    # A group introduced only during holdout cannot enter the cohort.
    raw['L_CoolSetpointU_AV']=np.where(day>=61,22.,np.nan)
    raw['L_ZoneTemp']=np.where(day>=61,26.,np.nan)
    rc=discover(raw);f,meta,_=prepare(raw,rc,{},'UTC','2026-08-15')
    assert meta['cohort']==['G'] and 'L' in meta['excluded_groups']
    meter=pd.DataFrame({'date':f.date,'energy_kwh':24*(20+4*f.cooling_demand),'quality':'good'})
    model,_,_,_=fit(meter,f,meta,'2026-07-01','2026-08-15')
    p=predict(model,raw,'2026-07-01','2026-08-15',{'G':50})
    assert any(x['outside_training_support'] for x in p)
    with pytest.raises(DataError,match='outside fitted cohort'):predict(model,raw,'2026-07-01','2026-08-15',{'L':22})
    minimal=raw[['timestamp','G_CoolSetpointU_AV','G_ZoneTemp']]
    assert predict(model,minimal,'2026-07-01','2026-08-15')==predict(model,raw,'2026-07-01','2026-08-15')


def test_uncertainty_does_not_turn_holdout_bias_into_extra_savings():
    model={'bootstrap_coef':[[40.]]*40,'holdout_residual_kwh':[100.]*14,
           'uncertainty_residual_block_size':7,'uncertainty_residual_block_starts':list(range(8)),
           'coef_average_kw':[40.],'uncertainty_centering':'centered_bootstrap_and_holdout_residuals'}
    draws=uncertainty_samples(model,np.ones((10,1)),np.full(10,24.))
    assert np.allclose(draws,9600.)


def test_heating_setpoint_model_and_explicit_heating_override():
    t=pd.date_range('2026-06-01',periods=75*144,freq='10min',tz='UTC')
    day=np.arange(len(t))//144
    raw=pd.DataFrame({'timestamp':t,'G_CoolSetpointU_AV':30.,
        'G_HeatSetpointU_AV':21+np.sin(day*.9),'G_ZoneTemp':20+np.cos(day*.7)})
    f,c,_=prepare(raw,discover(raw),{},'UTC','2026-08-15')
    assert c['heating_cohort']==['G']
    meter=pd.DataFrame({'date':f.date,'energy_kwh':24*(20+4*f.heating_demand),'quality':'good'})
    model,summary,_,_=fit(meter,f,c,'2026-07-01','2026-08-15')
    assert 'heating_demand' in model['features']
    assert summary['heldout_metrics']['cvrmse_pct']<1
    actual=predict(model,raw,'2026-07-01','2026-08-15')
    increased=predict(model,raw,'2026-07-01','2026-08-15',{'G':{'heating':23.}})
    assert sum(x['conditional_setpoint_kwh'] for x in increased)>sum(x['conditional_setpoint_kwh'] for x in actual)
