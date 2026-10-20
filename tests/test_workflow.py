import copy
import numpy as np
import pandas as pd
import pytest
from option_c.data import DataError,frame,digest
from option_c.regression import fit,predict,verify
from option_c.workflow import forecast,savings,train_post
from option_c.conditions import scenario
from fixtures import config,baseline_rows,post_rows


@pytest.fixture(scope='module')
def model():return fit(baseline_rows(),config())


@pytest.fixture(scope='module')
def post():
    meter,bms=post_rows()
    return train_post(meter,bms,config())


def test_change_point_selection_and_heldout_accuracy(model):
    assert model['family'].startswith('heat_cool')
    assert model['holdout_metrics']['cvrmse_pct']<2
    assert model['valid_periods']==365
    assert 'synthetic_not_site_evidence' in model['gates']


def test_no_control_leakage_into_baseline():
    c=config();c['independent_variables']=['cooling_setpoint']
    with pytest.raises(DataError,match='control variables'):fit(baseline_rows(),c)


def test_post_intervention_baseline_rows_are_rejected():
    rows=baseline_rows();rows[-1]['date']='2026-01-02'
    with pytest.raises(DataError,match='post-intervention'):fit(rows,config())


def test_model_integrity(model):
    changed=copy.deepcopy(model);changed['coef_average_kw'][0]+=1
    with pytest.raises(DataError,match='checksum'):verify(changed)


def test_future_dates_require_conditions_and_preserve_missing_days(model):
    cond=scenario(config(),'2026-04-20','2026-04-27',20.)
    full=forecast(model,cond,'2026-04-20','2026-04-27')
    assert full['expected_periods']==7 and full['baseline_kwh']>0
    assert 'future_weather_is_an_assumed_scenario' in full['gates']
    cond.pop(3)
    partial=forecast(model,cond,'2026-04-20','2026-04-27')
    assert partial['baseline_kwh'] is None
    assert partial['predicted_periods']==6


def test_dst_daily_durations():
    c=config()
    f=frame([{'date':'2025-03-09','energy_kwh':230},{'date':'2025-11-02','energy_kwh':250}],c)
    assert list(f.hours)==[23.,25.]


def test_daily_cadence_does_not_accept_partial_day_boundaries(model):
    cond=scenario(config(),'2026-04-20','2026-04-27',20)
    with pytest.raises(DataError,match='midnight'):forecast(model,cond,'2026-04-20T12:00:00-04:00','2026-04-27')


def test_measured_savings_sign_and_zero(model):
    cond=scenario(config(),'2026-04-20','2026-04-27',20)
    f=frame(cond,config());pred,_,_,_=predict(model,f)
    for r,p in zip(cond,pred):r['energy_kwh']=float(p)
    zero=savings(model,cond,'2026-04-20','2026-04-27')
    assert zero['savings_kwh_complete_period']==pytest.approx(0)
    for r in cond:r['energy_kwh']+=100
    negative=savings(model,cond,'2026-04-20','2026-04-27')
    assert negative['savings_kwh_complete_period']==pytest.approx(-700)
    assert negative['bms_estimated_periods']==0
    assert negative['demand_savings_kw'] is None


def test_missing_meter_without_calibrated_post_model_is_unavailable(model):
    cond=scenario(config(),'2026-04-20','2026-04-27',20)
    r=savings(model,cond,'2026-04-20','2026-04-27')
    assert r['paired_periods']==0 and r['savings_kwh_complete_period'] is None
    assert r['unavailable_periods']==7
    assert r['savings_kwh_paired'] is None
    assert r['status']=='unavailable_no_paired_consumption'


def test_bms_assisted_post_estimates_and_measured_precedence(model,post):
    meter,bms=post_rows(7,'2026-04-20')
    first=meter[0]['energy_kwh']
    for r in meter[1:]:r['energy_kwh']=None
    out=savings(model,meter,'2026-04-20','2026-04-27',post,bms)
    assert out['status']=='bms_assisted_modeled_estimate_requires_review'
    assert out['observed_meter_periods']==1 and out['bms_estimated_periods']==6
    assert out['periods'][0]['reporting_kwh']==first
    assert out['savings_kwh_complete_period'] is not None


def test_bms_extrapolation_not_used_to_fill_meter_gap(model,post):
    meter,bms=post_rows(7,'2026-04-20')
    for r in meter:r['energy_kwh']=None
    bms[0]['bms_cool1']=2
    out=savings(model,meter,'2026-04-20','2026-04-27',post,bms)
    assert out['unavailable_periods']>=1 and out['savings_kwh_complete_period'] is None


def test_post_model_cannot_impute_its_training_period(model,post):
    meter,bms=post_rows(7,'2026-01-01')
    for r in meter:r['energy_kwh']=None
    out=savings(model,meter,'2026-01-01','2026-01-08',post,bms)
    assert out['bms_estimated_periods']==0


def test_unverified_bms_or_collinear_signals_do_not_fit():
    meter,bms=post_rows();c=config();c['review']['bms_semantics_verified']=False
    with pytest.raises(DataError,match='Verify BMS'):train_post(meter,bms,c)
    c=config()
    for r in bms:r['bms_cool2']=r['bms_cool1']
    with pytest.raises(DataError,match='identifiable'):train_post(meter,bms,c)


def test_boundary_mismatch_rejected(model,post):
    changed=copy.deepcopy(post);changed['config']['boundary_description']='Different boundary';changed['model_id']=digest({k:v for k,v in changed.items() if k!='model_id'})
    meter,bms=post_rows(7,'2026-04-20')
    with pytest.raises(DataError,match='boundary'):savings(model,meter,'2026-04-20','2026-04-27',changed,bms)


def test_documented_nonroutine_adjustment(model):
    cond=scenario(config(),'2026-04-20','2026-04-27',20)
    for r in cond:r['energy_kwh']=500
    initial=savings(model,cond,'2026-04-20','2026-04-27')
    adjustment={'timestamp':'2026-04-20','baseline_adjustment_kwh':100,'reference':'Reviewed tenant load change','reviewed':True}
    adjusted=savings(model,cond,'2026-04-20','2026-04-27',adjustments=[adjustment])
    assert adjusted['savings_kwh_complete_period']==pytest.approx(initial['savings_kwh_complete_period']+100)
    adjustment['reviewed']=False
    with pytest.raises(DataError,match='reviewed'):savings(model,cond,'2026-04-20','2026-04-27',adjustments=[adjustment])


def test_interval_average_kw_is_not_peak_demand():
    c=config();c.update(resolution='interval',interval_minutes=15)
    f=frame([{'timestamp':'2025-01-01T00:00:00Z','energy_kwh':5}],c)
    assert f.hours.iloc[0]==.25
    with pytest.raises(DataError,match='offset'):frame([{'timestamp':'2025-01-01T00:00:00','energy_kwh':5}],c)


def test_hourly_regression_covers_calendar_and_weather_without_daily_smearing():
    c=config();c.update(resolution='interval',interval_minutes=60)
    rows=[]
    for i,t in enumerate(pd.date_range('2025-01-01',periods=24*30,freq='1h',tz='America/Toronto')):
        temp=20+4*np.sin(i/(24*7))+np.sin(t.hour*2*np.pi/24)
        kw=30+.8*temp+5*np.sin(t.hour*2*np.pi/24)+2*(t.dayofweek>=5)
        rows.append({'timestamp':t.tz_convert('UTC').isoformat(),'temperature_c':temp,'energy_kwh':kw,'quality':'good'})
    m=fit(rows,c)
    assert m['holdout_metrics']['cvrmse_pct']<1
    assert any(t['kind']=='sin_hour' for t in m['terms'])
    assert 'baseline_less_than_one_year' in m['gates']
