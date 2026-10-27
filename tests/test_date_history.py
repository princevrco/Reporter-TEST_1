import numpy as np
import pandas as pd
import pytest
from option_c import date_history


def source(days=40):
    grid=pd.date_range('2026-01-01',periods=days*96,freq='15min',tz='UTC')
    f=pd.DataFrame({'timestamp':grid,'A_ZoneTemp':24.,'A_CoolSetpointU_AV':22.,'unrelated':np.nan})
    dates=pd.date_range('2026-01-01',periods=days)
    w=pd.DataFrame({'date':dates.date.astype(str),'temperature_c':10.})
    return f,w,str((dates[-1]+pd.Timedelta(days=1)).date())


def test_missing_unrelated_values_retain_rows_and_every_eligible_date():
    f,w,end=source();clean,bad,audit=date_history.clean(f)
    assert len(clean)==len(f) and len(bad)==0
    clean,groups,daily,coverage,meta=date_history.prepare(clean,w,'UTC',end)
    assert len(daily)==40 and set(groups)=={'A'}
    assert coverage.eligible.all() and daily.cool_degree_hours.eq(48.).all()
    assert not meta['date_imputation']


def test_empty_required_date_excluded_and_never_zero_filled():
    f,w,end=source();day='2026-01-10';f.loc[f.timestamp.dt.date.astype(str).eq(day),'A_CoolSetpointU_AV']=np.nan
    clean,_,_=date_history.clean(f)
    _,g,daily,cov,meta=date_history.prepare(clean,w,'UTC',end)
    assert len(clean)==len(f)
    assert len(daily)==39 and day not in set(daily.date)
    assert not cov[cov.date.eq(day)].eligible.iloc[0]
    assert cov[cov.date.eq(day)].observed_required_hours.iloc[0]==0


def test_completely_absent_day_is_listed_and_excluded():
    f,w,end=source();f=f[~f.timestamp.dt.date.astype(str).eq('2026-01-10')]
    _,g,d,c,_=date_history.prepare(f,w,'UTC',end)
    assert '2026-01-10' not in set(d.date)
    assert not c[c.date.eq('2026-01-10')].eligible.iloc[0]


def test_heldout_only_asset_cannot_change_fixed_cohort():
    f,w,end=source();cutoff=pd.Timestamp(end,tz='UTC')-pd.Timedelta(days=14)
    f['L_ZoneTemp']=np.where(f.timestamp>=cutoff,24.,np.nan)
    f['L_CoolSetpointU_AV']=np.where(f.timestamp>=cutoff,22.,np.nan)
    _,g,d,c,meta=date_history.prepare(f,w,'UTC',end)
    assert set(g)=={'A'} and 'L' in meta['excluded_assets']


def test_blank_observation_stops_signal_without_previous_value_propagation():
    f,w,end=source();f.loc[20,'A_CoolSetpointU_AV']=np.nan
    _,g,d,c,_=date_history.prepare(f,w,'UTC',end)
    # The date remains usable at >90% coverage, but the missing interval is not filled.
    first=d.iloc[0]
    assert first.coverage_fraction==pytest.approx(23.75/24)
    assert first.cool_degree_hours==pytest.approx(47.5)


def test_absent_date_never_becomes_eligible_even_with_low_coverage_setting():
    f,w,end=source();f=f[~f.timestamp.dt.date.astype(str).eq('2026-01-10')]
    config={'groups':{'A':{'temperature':'A_ZoneTemp','cool_setpoint':'A_CoolSetpointU_AV'}},'minimum_daily_coverage':.001}
    _,g,d,c,_=date_history.prepare(f,w,'UTC',end,config)
    assert '2026-01-10' not in set(d.date)


def test_native_asynchronous_samples_preserve_coverage_and_empty_dates_excluded():
    f,w,end=source()
    t=f[['timestamp','A_ZoneTemp']].copy();s=f[['timestamp','A_CoolSetpointU_AV']].copy();s.timestamp+=pd.Timedelta(minutes=5)
    raw=t.merge(s,on='timestamp',how='outer').sort_values('timestamp').reset_index(drop=True)
    mask=raw.drop(columns='timestamp').notna();mask.index=pd.DatetimeIndex(raw.timestamp)
    _,_,d,_,a=date_history.prepare(raw,w,'UTC',end,observation_mask=mask)
    assert len(d)==40 and a['observation_semantics']=='native_mask'
    assert d.iloc[1:].cool_degree_hours.eq(48.).all()
    missing=raw.timestamp.dt.strftime('%Y-%m-%d').eq('2026-01-10')
    _,_,d,_,_=date_history.prepare(raw.loc[~missing],w,'UTC',end,observation_mask=mask.loc[~missing.to_numpy()])
    assert '2026-01-10' not in set(d.date)


def test_native_missing_resets_signal_and_alignment_cannot_extend_age():
    f,w,end=source();mask=f.drop(columns='timestamp').notna();mask.index=pd.DatetimeIndex(f.timestamp)
    f.loc[20,'A_CoolSetpointU_AV']=np.nan
    _,_,d,_,_=date_history.prepare(f,w,'UTC',end,observation_mask=mask)
    assert d.iloc[0].coverage_fraction==pytest.approx(23.75/24)
    mask.loc[mask.index>=f.timestamp.iloc[21],'A_CoolSetpointU_AV']=False
    f.loc[21:,'A_CoolSetpointU_AV']=np.nan
    _,_,d,_,_=date_history.prepare(f,w,'UTC',end,observation_mask=mask)
    assert d.empty


def test_cohort_preserves_maximum_earlier_days():
    f,w,end=source(70);f['B_ZoneTemp']=24.;f['B_CoolSetpointU_AV']=22.
    f.loc[f.timestamp<pd.Timestamp('2026-01-10',tz='UTC'),['B_ZoneTemp','B_CoolSetpointU_AV']]=np.nan
    _,g,d,_,a=date_history.prepare(f,w,'UTC',end)
    assert set(g)=={'A'} and len(d)==70 and a['cohort_target_earlier_days']==56
