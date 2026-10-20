import json
import pandas as pd
import pytest
from fixtures import config
from option_c.data import DataError
from option_c.optimizer import load,prepare,summaries


def inputs(tmp_path,day='2026-04-01',minutes=30,zone_counts=None):
    c=config();c['optimizer']={'bin_minutes':minutes,'minimum_daily_coverage':.9,'zones':{}}
    start=pd.Timestamp(day,tz=c['timezone']);end=start+pd.DateOffset(days=1)
    times=pd.date_range(start,end,freq=f'{minutes}min',inclusive='left')
    zone_counts=zone_counts or {'Z1':2,'Z2':1,'Z3':3}
    opt=[];bms=pd.DataFrame({'timestamp':times.tz_convert('UTC').astype(str)})
    for zone,count in zone_counts.items():
        modes=[1 if 2<=i<2+count else 0 for i in range(len(times))]
        for t,mode in zip(times,modes):opt.append({'timestamp':t.isoformat(),'issued_at':t.isoformat(),'zone_id':zone,'achieved_mode':mode,'applied':True})
        bms[f'{zone}_T']=[18 if mode else 22 for mode in modes];bms[f'{zone}_H']=20;bms[f'{zone}_C']=24
        c['optimizer']['zones'][zone]={'equipment_id':zone,'meter_ids':[],
            'hysteresis_margin_c':.5,'bms_columns':{'temperature_c':f'{zone}_T','heating_setpoint_c':f'{zone}_H','cooling_setpoint_c':f'{zone}_C'}}
    op=tmp_path/'optimizer.csv';bp=tmp_path/'bms.csv'
    pd.DataFrame(opt).to_csv(op,index=False);bms.to_csv(bp,index=False)
    return c,op,bp


def test_duration_ratios_shares_and_censored_run_distribution(tmp_path):
    c,op,bp=inputs(tmp_path)
    daily,features,groups,audit=prepare([op],bp,c)
    s={r['zone_id']:r for r in summaries(daily,'2026-04-01','2026-04-02',run_stats=audit['zone_on_run_statistics'])}
    assert [s[z]['on_duration_ratio_to_one_hour'] for z in ('Z1','Z2','Z3')]==[1,.5,1.5]
    assert [s[z]['share_of_zone_on_time_pct'] for z in ('Z1','Z2','Z3')]==pytest.approx([100/3,100/6,50])
    assert all(r['coverage_fraction']==1 for r in s.values())
    assert daily.uncertain_hours.sum()==0
    assert audit['zone_on_run_statistics'][0]['mean_uncensored_run_hours']==1
    assert [s[z]['mean_on_run_ratio_to_one_hour'] for z in ('Z1','Z2','Z3')]==[1,.5,1.5]
    # Entire extra day without optimizer rows must count as missing, never OFF.
    missing=summaries(daily,'2026-04-01','2026-04-03')
    assert all(r['coverage_fraction']==.5 and r['unknown_or_missing_hours']==24 for r in missing)


def test_deadband_boundary_and_feedback_gaps_remain_uncertain(tmp_path):
    c,op,bp=inputs(tmp_path,zone_counts={'Z1':2})
    bms=pd.read_csv(bp);bms.loc[2,'Z1_T']=20;bms.to_csv(bp,index=False)
    daily,_,_,_=prepare([op],bp,c)
    r=daily.iloc[0]
    assert r.on_hours_estimated==1
    assert r.on_hours_lower_sensitivity==.5 and r.on_hours_upper_sensitivity==1
    # No measured point may be held over a long missing interval.
    bms=bms.iloc[:1];bms.to_csv(bp,index=False)
    daily,_,_,_=prepare([op],bp,c)
    assert daily.uncertain_hours.iloc[0]==23.5


def test_shared_equipment_uses_union_not_sum(tmp_path):
    c,op,bp=inputs(tmp_path,zone_counts={'Z1':2,'Z2':2})
    for spec in c['optimizer']['zones'].values():spec['equipment_id']='SHARED_RTU'
    _,features,groups,_=prepare([op],bp,c)
    assert len(groups)==1 and set(groups[0]['zone_ids'])=={'Z1','Z2'}
    assert features.heating_fraction.iloc[0]==pytest.approx(1/24)


def test_future_issue_is_rejected_and_latest_valid_forecast_wins(tmp_path):
    c,op,bp=inputs(tmp_path,zone_counts={'Z1':2})
    f=pd.read_csv(op);row=f.iloc[0].to_dict();row['achieved_mode']=-1
    row['issued_at']=(pd.Timestamp(row['timestamp'])+pd.Timedelta(minutes=5)).isoformat()
    pd.concat([f,pd.DataFrame([row])]).to_csv(op,index=False)
    loaded,audit=load([op],c)
    assert audit['future_issued_rows_rejected']==1
    assert loaded.iloc[0]['mode']=='off'
    old=f.iloc[2].to_dict();old['issued_at']=(pd.Timestamp(old['timestamp'])-pd.Timedelta(minutes=30)).isoformat();old['achieved_mode']=-1
    oldp=tmp_path/'old.csv';pd.DataFrame([old]).to_csv(oldp,index=False)
    loaded,audit=load([op,oldp],c)
    assert audit['overlapping_forecasts_resolved']==1
    assert loaded.loc[loaded.timestamp==pd.Timestamp(old['timestamp']),'mode'].iloc[0]=='heating'


def test_missing_issue_time_and_invalid_mode_cannot_create_on_hours(tmp_path):
    c,op,bp=inputs(tmp_path,zone_counts={'Z1':2})
    f=pd.read_csv(op).drop(columns='issued_at');f.loc[2,'achieved_mode']=99;f.to_csv(op,index=False)
    daily,features,_,audit=prepare([op],bp,c)
    assert audit['unknown_mode_rows']==1 and audit['missing_issue_time_rows']==48
    assert daily.eligible_mode_hours.iloc[0]==0
    assert pd.isna(features.heating_fraction.iloc[0])


def test_dst_day_has_23_hours(tmp_path):
    c,op,bp=inputs(tmp_path,day='2026-03-08',minutes=60,zone_counts={'Z1':1})
    daily,_,_,_=prepare([op],bp,c)
    assert daily.period_hours.iloc[0]==23 and daily.coverage_fraction.iloc[0]==1


def test_mapper_json_requires_issue_and_horizon_metadata(tmp_path):
    c,op,bp=inputs(tmp_path,zone_counts={'Z1':2})
    jp=tmp_path/'mapper.json'
    payload={'zones':{'Z1':{'achieved_modes':[-1,0,1]}},'metrics':{'state_timestamp':'2026-04-01T04:00:00Z','control_horizon':1,'cadence_seconds':1800}}
    jp.write_text(json.dumps(payload))
    with pytest.raises(DataError,match='issued_at'):load([jp],c)
    payload.update(issued_at='2026-04-01T04:00:00Z',horizon_start='2026-04-01T04:00:00Z')
    jp.write_text(json.dumps(payload));daily,_,_,_=prepare([jp],bp,c)
    assert daily.eligible_mode_hours.iloc[0]==.5
    assert daily.cooling_hours_estimated.iloc[0]==.5


def test_conflicting_same_run_modes_are_not_silently_accepted(tmp_path):
    c,op,bp=inputs(tmp_path,zone_counts={'Z1':2})
    f=pd.read_csv(op);row=f.iloc[0].to_dict();row['achieved_mode']=1
    pd.concat([f,pd.DataFrame([row])]).to_csv(op,index=False)
    with pytest.raises(DataError,match='Conflicting'):prepare([op],bp,c)
