import numpy as np
import pandas as pd
import pytest
from option_c.response import features,discover,fit,predict
from option_c.data import DataError,digest


def test_bounded_age_invalid_status_and_gap_are_not_zero_runtime():
    f=pd.DataFrame({'timestamp':pd.to_datetime(['2026-08-15T00:00:00Z','2026-08-15T00:15:00Z','2026-08-17T00:00:00Z']),
        'RTU_Cool1':[1,3,0],'RTU_ZoneTemp':[25,25,22],'RTU_CoolSetpointU_AV':[24,24,24]})
    c=discover(f);d,a=features(f,c,'UTC')
    assert a['signal_audit']['RTU_Cool1']['rejected_rows']==1
    assert a['signal_audit']['RTU_Cool1']['populated_bins']==3
    assert pd.isna(d.loc[d.date.eq('2026-08-16'),'bms_RTU::cool_stage1']).all()
    assert d.filter(like='lag1').isna().all().all()


def synthetic():
    rng=np.random.default_rng(78);days=pd.date_range('2026-06-01',periods=110)
    x=rng.uniform(.1,.9,len(days));z=rng.uniform(.1,.9,len(days))
    ff=pd.DataFrame({'date':days.date.astype(str),'hours':24.,'bms_on':x,'bms_cooling':z,'bms_duplicate':x})
    meter=pd.DataFrame({'date':ff.date,'energy_kwh':24*(10+20*x+30*z),'quality':'good'})
    return meter,ff,{'minimum_training_days':40,'minimum_holdout_days':8,'semantics_verified':True,'ridge_alphas':[.01,1,100]}


def test_ridge_handles_correlated_features_and_keeps_holdout_out_of_selection():
    m,f,c=synthetic();a,diag=fit(m,f,c,'UTC')
    assert a['holdout_metrics']['cvrmse_pct']<1
    assert len(a['heldout_predictions'])>=8 and diag[0]['scope'].startswith('in_sample')
    m2=m.copy();m2.loc[88:,'energy_kwh']*=2
    b,_=fit(m2,f,c,'UTC')
    assert a['alpha']==b['alpha']
    assert a['heldout_predictions']==[{**p,'measured_kwh':p['measured_kwh']/2} for p in b['heldout_predictions']]
    assert a['parameters']!=b['parameters']


def test_response_future_requires_later_date_and_checks_support_and_hash():
    m,f,c=synthetic();model,_=fit(m,f,c,'UTC')
    future=f.iloc[:2].copy();future['date']=['2026-09-19','2026-09-20'];future.loc[future.index[0],'bms_on']=100
    r=predict(model,future,c)
    assert len(r)==2 and r[0]['outside_training_support']
    assert predict(model,f,c)==[]
    bad={**model,'alpha':999}
    with pytest.raises(DataError,match='checksum'):predict(bad,future,c)
    with pytest.raises(DataError,match='configuration'):predict(model,future,{**c,'semantics_verified':False})


def test_insufficient_days_do_not_produce_energy_estimates():
    m,f,c=synthetic()
    with pytest.raises(DataError,match='training days'):fit(m.iloc[:25],f,c,'UTC')


def test_proposed_aliases_report_sampling_and_keep_semantics_unverified():
    f=pd.DataFrame({'timestamp':pd.date_range('2026-08-15',periods=4,freq='15min',tz='UTC'),
        'RTU_SupplyFan':[1,1,0,0],'RTU_CoolSetpointUsed':[24,24,25,25],'RTU_ZoneHeatSetpoint':[20,20,20,20]})
    cfg=discover(f)
    assert cfg['groups']['RTU']['fan']=='RTU_SupplyFan'
    assert cfg['groups']['RTU']['cool_setpoint']=='RTU_CoolSetpointUsed'
    assert cfg['semantics_verified'] is False
    _,audit=features(f,cfg,'UTC')
    assert audit['signal_audit']['RTU_SupplyFan']['median_sampling_minutes']==15
    assert audit['signal_audit']['RTU_SupplyFan']['maximum_gap_minutes']==15
