import copy
import numpy as np
import pandas as pd


def config():
    return {'schema':'option_c.config.v1','meter_id':'synthetic_meter','boundary_description':'Synthetic whole-facility import boundary',
            'timezone':'America/Toronto','resolution':'daily','interval_minutes':15,'intervention_timestamp':'2026-01-01T00:00:00-05:00',
            'calendar_features':True,'post_calendar_features':False,'independent_variables':[],'allow_meter_only_baseline':False,
            'minimum_baseline_periods':90,'minimum_post_calibration_periods':60,'change_points_c':[12,15,18.3,21,24],
            'validation_limits':{'cvrmse_pct':20.,'abs_nmbe_pct':5.},'synthetic':True,
            'review':{k:True for k in ('meter_boundary_verified','timestamps_verified','intervention_confirmed','independent_variables_verified','static_factors_reviewed','mv_plan_approved','bms_semantics_verified')},
            'bms_features':{'bms_fan':{'column':'fan','allowed_values':[0,1],'minimum_hour_coverage':.9},'bms_cool1':{'column':'cool1','allowed_values':[0,1],'minimum_hour_coverage':.9},'bms_cool2':{'column':'cool2','allowed_values':[0,1],'minimum_hour_coverage':.9}}}


def baseline_rows(days=365):
    rng=np.random.default_rng(100)
    rows=[]
    for i,t in enumerate(pd.date_range('2025-01-01',periods=days,freq='D',tz='America/Toronto')):
        temp=10+15*np.sin(2*np.pi*i/365)+rng.normal(0,1)
        hours=(t+pd.DateOffset(days=1)-t).total_seconds()/3600
        kw=40+1.8*max(18.3-temp,0)+2.4*max(temp-21,0)+3*(t.dayofweek>=5)
        rows.append({'date':t.date().isoformat(),'meter_id':'synthetic_meter','temperature_c':float(temp),'energy_kwh':float(kw*hours+rng.normal(0,2)),'quality':'good'})
    return rows


def post_rows(days=100,start='2026-01-01'):
    rng=np.random.default_rng(101);meter=[];features=[]
    for i,t in enumerate(pd.date_range(start,periods=days,freq='D',tz='America/Toronto')):
        temp=20+6*np.sin(i*.13)
        fan=float(rng.uniform(.3,.95));c1=float(rng.uniform(.05,fan));c2=float(rng.uniform(0,c1))
        hours=(t+pd.DateOffset(days=1)-t).total_seconds()/3600
        kw=30+.12*temp+5*fan+35*c1+15*c2
        meter.append({'timestamp':t.tz_convert('UTC').isoformat(),'meter_id':'synthetic_meter','temperature_c':temp,'energy_kwh':float(kw*hours+rng.normal(0,1)),'quality':'good'})
        features.append({'timestamp':t.tz_convert('UTC').isoformat(),'meter_id':'synthetic_meter','bms_fan':fan,'bms_cool1':c1,'bms_cool2':c2})
    return meter,features
