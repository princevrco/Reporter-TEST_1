import copy
import numpy as np
import pandas as pd
import pytest
from fixtures import config
from option_c.data import DataError,records,digest
from option_c.quality import assess,withhold
from option_c.resilient import fit,sensitivity
from option_c.workflow import savings,forecast


def data():
    c=config();c.update(allow_meter_only_baseline=True)
    c['baseline_options']={'enabled':True,'windows_days':[None,180],
        'estimators':['ols','huber','weekday_median'],'bootstrap_repeats':0,
        'minimum_training_days':60,'holdout_days':56,'fold_days':28}
    c['data_quality']={'enabled':True}
    dates=pd.date_range('2025-01-01',periods=365)
    rng=np.random.default_rng(7)
    f=pd.DataFrame({'date':dates.date.astype(str),'meter_id':c['meter_id'],'quality':'good',
                    'energy_kwh':1000+50*(dates.dayofweek>=5)+rng.normal(0,1,365)})
    f.loc[[50,130],'energy_kwh']+=1_000_000
    return c,f


def test_spikes_are_flagged_not_removed_and_review_is_anchored():
    c,f=data();original=f.copy(deep=True)
    clean,events,stats=assess(f,c,'2026-01-01','2026-01-03')
    assert stats['unresolved_baseline_events']==2
    pd.testing.assert_series_equal(clean.energy_kwh,original.energy_kwh)
    ledger=[{'event_id':e['event_id'],'reviewed':True,'reviewer':'Engineer','reference':'CARMA ticket 1','decision':'exclude'} for e in events]
    clean,events,stats=assess(f,c,'2026-01-01','2026-01-03',ledger)
    assert clean.energy_kwh.isna().sum()==2 and stats['unresolved_baseline_events']==0
    pd.testing.assert_frame_equal(f,original)
    # Changing the source invalidates the old anchored decision.
    revised=f.copy();revised.loc[50,'energy_kwh']+=1
    clean,events,stats=assess(revised,c,'2026-01-01','2026-01-03',ledger)
    assert clean.energy_kwh.isna().sum()==1 and stats['unresolved_baseline_events']==1
    ledger[0]['reviewed']=False
    with pytest.raises(DataError,match='reviewed=true'):assess(f,c,'2026-01-01','2026-01-03',ledger)


def test_robust_selection_uses_common_holdout_and_resists_spikes():
    c,f=data();m=fit(records(f),c)
    assert not m['family'].startswith('ols')
    assert m['holdout_metrics']['cvrmse_pct']<1
    assert all(s['holdout']['n']==56 for s in m['selection_candidates'] if s['status']=='evaluated')
    p=forecast(m,[{'date':'2026-01-01','meter_id':c['meter_id']}],'2026-01-01','2026-01-02')
    assert 900<p['baseline_kwh']<1200
    altered=f.copy();altered.loc[altered.index[-56:],'energy_kwh']*=2
    changed=fit(records(altered),c)
    assert m['family']==changed['family']
    assert [s.get('cv_rmse_kwh') for s in m['selection_candidates']]==[s.get('cv_rmse_kwh') for s in changed['selection_candidates']]
    assert changed['holdout_metrics']['cvrmse_pct']>20


def test_failed_models_withhold_money_but_preserve_diagnostics():
    c,f=data();m=fit(records(f),c)
    m['gates'].append('baseline_quality_review_required');m['model_id']=digest({k:v for k,v in m.items() if k!='model_id'})
    rows=[{'date':'2026-01-01','meter_id':c['meter_id'],'energy_kwh':900,'quality':'good'}]
    r=savings(m,rows,'2026-01-01','2026-01-02',flat_rate={'rate_per_kwh':.12,'reference':'test'})
    assert r['savings_kwh_paired'] is None and r['diagnostic_savings_kwh_paired'] is not None
    assert r['variable_energy_cost_savings_paired']['amount'] is None
    assert r['periods'][0]['savings_kwh'] is None
    original=r['diagnostic_savings_kwh_paired'];withhold(r,['another_reason'])
    assert r['diagnostic_savings_kwh_paired']==original and len(r['release_reasons'])==2
    future=forecast(m,rows,'2026-01-01','2026-01-02')
    assert future['baseline_kwh'] is None and future['diagnostic_baseline_kwh']>0


def test_reviewed_correction_does_not_overwrite_raw():
    c,f=data();_,events,_=assess(f,c,'2026-01-01','2026-01-03')
    ledger=[{'event_id':events[0]['event_id'],'decision':'correct','corrected_kwh':1000,
        'reviewed':True,'reviewer':'Engineer','reference':'Corrected CARMA export'}]
    cleaned,_,_=assess(f,c,'2026-01-01','2026-01-03',ledger)
    corrected=cleaned[cleaned.reviewed_action.eq('correct')]
    assert corrected.energy_kwh.iloc[0]==1000 and corrected.original_energy_kwh.iloc[0]>1e6


def test_missing_day_breaks_zero_run():
    c,_=data();c['data_quality']['zero_run_days']=7
    f=pd.DataFrame({'date':['2025-01-01','2025-01-02','2025-01-04','2025-01-05','2025-01-06','2025-01-07','2025-01-08'],
        'meter_id':c['meter_id'],'energy_kwh':0.,'quality':'good'})
    _,events,_=assess(f,c,'2026-01-01','2026-01-03')
    assert events==[]


def test_batch_withholds_suspect_energy_and_accepts_documented_review(tmp_path):
    from option_c.batch import run
    from option_c.overview import build,total
    from option_c.data import load
    c,f=data()
    f=pd.concat([f,pd.DataFrame([{'date':'2026-01-01','meter_id':c['meter_id'],'energy_kwh':900.,'quality':'good'}])],ignore_index=True)
    mp=tmp_path/'meter.csv';bp=tmp_path/'bms.csv';f.to_csv(mp,index=False)
    pd.DataFrame({'timestamp':['2026-01-01T05:00:00Z']}).to_csv(bp,index=False)
    out=tmp_path/'unreviewed';summary=run(mp,bp,c,'2026-01-01','2026-01-02',out,kind='normalized')
    entry=summary['meters'][0]
    assert entry['status']=='comparison_withheld_requires_review'
    result=load(out/entry['folder']/'savings/result.json')
    assert result['savings_kwh_paired'] is None and result['reporting_kwh_paired']==900
    assert total({c['meter_id']:result},[c['meter_id']],.12)['savings_kwh'] is None
    events=load(out/'data-quality-events.json')
    assert len(events)==2 and load(out/'quality-review-summary.json')['unreviewed']==2
    overview=build(out,.12,'2026-01-02','2026-01-05')
    assert overview['meters'][0]['paired_savings_cad'] is None
    assert overview['meters'][0]['forecast_kwh'] is None
    assert overview['meters'][0]['diagnostic_forecast_kwh']>0
    ledger=[{'event_id':e['event_id'],'reviewed':True,'reviewer':'Engineer','reference':'Confirmed corrupt export','decision':'exclude'} for e in events]
    reviewed=tmp_path/'reviewed';second=run(mp,bp,c,'2026-01-01','2026-01-02',reviewed,kind='normalized',quality_reviews=ledger)
    assert second['meters'][0]['savings_kwh_paired'] is not None
    assert load(reviewed/'quality-review-summary.json')['unreviewed']==0
    analyzed=pd.read_csv(reviewed/'quality-reviewed-meter.csv')
    assert analyzed.energy_kwh.isna().sum()==2
    assert pd.read_csv(mp).energy_kwh.isna().sum()==0


def test_quality_flags_level_shift_and_real_zero_runs_without_deletion():
    c,f=data();f['energy_kwh']=1000.
    f.loc[200:,'energy_kwh']=3000.
    _,events,stats=assess(f,c,'2026-01-01','2026-01-03')
    assert any(e['code']=='baseline_level_shift' for e in events)
    f.loc[20:27,'energy_kwh']=0.
    clean,events,_=assess(f,c,'2026-01-01','2026-01-03')
    assert any(e['code']=='sustained_zero_run' and e['days']==8 for e in events)
    assert clean.energy_kwh.eq(0).sum()==8


def test_reporting_spikes_remain_measured_but_withhold_comparison(tmp_path):
    from option_c.batch import run
    from option_c.data import load
    c,f=data();f['energy_kwh']=1000.
    f=pd.concat([f,pd.DataFrame([{'date':'2026-01-01','meter_id':c['meter_id'],'energy_kwh':1e6,'quality':'good'}])],ignore_index=True)
    mp=tmp_path/'meter.csv';bp=tmp_path/'bms.csv';f.to_csv(mp,index=False)
    pd.DataFrame({'timestamp':['2026-01-01T05:00:00Z']}).to_csv(bp,index=False)
    out=tmp_path/'output';summary=run(mp,bp,c,'2026-01-01','2026-01-02',out,kind='normalized')
    result=load(out/summary['meters'][0]['folder']/'savings/result.json')
    assert result['reporting_kwh_paired']==1e6 and result['savings_kwh_paired'] is None
    assert 'reporting_quality_review_required' in result['release_reasons']
    assert result['diagnostic_savings_kwh_paired']<0


def test_annual_calendar_candidate_learns_seasonality_only_with_sufficient_past():
    c,f=data();c['intervention_timestamp']='2027-01-01T00:00:00-05:00'
    c['baseline_options']['annual_calendar_candidates']=True
    dates=pd.date_range('2025-01-01',periods=650)
    energy=1000+300*np.sin(2*np.pi*(dates.dayofyear-1)/365)+30*(dates.dayofweek>=5)
    f=pd.DataFrame({'date':dates.date.astype(str),'meter_id':c['meter_id'],'energy_kwh':energy,'quality':'good'})
    m=fit(records(f),c)
    assert m['family'].endswith('_annual') and m['holdout_metrics']['cvrmse_pct']<1
    assert any(t['kind']=='sin_year' for t in m['terms'])
    # A shortened archive cannot fit annual terms in its earliest validation fold.
    short=fit(records(f.iloc[-365:]),c)
    assert all(s['status']=='unavailable' for s in short['selection_candidates'] if s['name'].endswith('_annual'))
    totals=sensitivity(m,[{'date':'2027-01-01','meter_id':c['meter_id']}])
    assert all(x['baseline_kwh']>0 for x in totals['models'])


def test_consumption_remains_available_when_baseline_is_blocked(tmp_path):
    from option_c.batch import run
    from option_c.overview import build
    c,_=data();mp=tmp_path/'meter.csv';bp=tmp_path/'bms.csv'
    pd.DataFrame([{'date':'2026-01-01','meter_id':c['meter_id'],'energy_kwh':0.,'quality':'good'}]).to_csv(mp,index=False)
    pd.DataFrame({'timestamp':['2026-01-01T05:00:00Z']}).to_csv(bp,index=False)
    out=tmp_path/'output';s=run(mp,bp,c,'2026-01-01','2026-01-02',out,kind='normalized')
    assert s['blocked_meter_count']==1
    consumption=pd.read_csv(out/'meter-consumption.csv')
    assert consumption.meter_consumption_kwh_complete_period.iloc[0]==0
    overview=build(out,.12,'2026-01-02','2026-01-05')
    assert overview['meters'][0]['meter_consumption_kwh_complete_period']==0
    assert overview['meters'][0]['savings_kwh'] is None
