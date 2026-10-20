import copy
import numpy as np
import pandas as pd
import pytest
from fixtures import config
from option_c.effects import analyze


def samples(shared=False):
    c=config();c['optimizer']={'minimum_training_days':60,'calendar_features':True}
    rng=np.random.default_rng(78);rows=[];features=[]
    groups=[{'equipment_id':e,'zone_ids':[z],'meter_ids':[],'physical_mapping_verified':False} for e,z in [('E1','Z1'),('E2','Z2')]]
    pattern=[]
    for i in range(100):
        h1,c1,h2,c2=rng.uniform(.05,.4,4)
        if i%4==0:h1=c1=0
        if i%5==0:h2=c2=0
        pattern.append((h1,c1,h2,c2))
    for i,t in enumerate(pd.date_range('2026-01-01',periods=120,freq='D',tz=c['timezone'])):
        h1,c1,h2,c2=pattern[i%100]
        if shared:h2,c2=h1,c1
        hours=(t+pd.DateOffset(days=1)-t).total_seconds()/3600
        kw=40+8*h1+12*c1+(0 if shared else 4*h2+6*c2)
        rows.append({'date':t.date().isoformat(),'meter_id':c['meter_id'],'energy_kwh':kw*hours,'quality':'good'})
        for e,h,co in [('E1',h1,c1),('E2',h2,c2)]:
            features.append({'date':t.date().isoformat(),'equipment_id':e,'heating_fraction':h,'cooling_fraction':co,'uncertain_fraction':0})
    return c,rows,pd.DataFrame(features),groups


def test_joint_association_percentages_and_holdout_no_reporting_leakage():
    c,rows,features,groups=samples()
    result=analyze(rows,features,groups,c,'2026-04-11','2026-05-01')
    assert result['status']=='modeled_associations_require_review'
    assert result['model']['holdout_metrics']['cvrmse_pct']<1e-8
    assert result['reporting_admitted_periods']==20
    matched=features[(features.date>='2026-04-11')&(features.equipment_id=='E1')]
    associated=float(((8*matched.heating_fraction+12*matched.cooling_fraction)*24).sum())
    assert result['groups'][0]['associated_kwh']==pytest.approx(associated)
    assert result['zone_effects'][0]['associated_pct_of_meter_kwh']==pytest.approx(100*associated/result['reporting_meter_kwh_on_matched_periods'])
    changed=copy.deepcopy(rows)
    for row in changed:
        if row['date']>='2026-04-11':row['energy_kwh']*=2
    newer=analyze(changed,features,groups,c,'2026-04-11','2026-05-01')
    assert newer['model']['model_id']==result['model']['model_id']
    assert newer['groups'][0]['associated_pct_of_meter_kwh']==pytest.approx(result['groups'][0]['associated_pct_of_meter_kwh']/2)


def test_co_moving_zones_get_group_effect_not_invented_individual_percent():
    c,rows,features,groups=samples(shared=True)
    result=analyze(rows,features,groups,c,'2026-04-11','2026-05-01')
    assert len(result['groups'])==1
    assert result['groups'][0]['associated_pct_of_meter_kwh'] is not None
    assert set(result['groups'][0]['zone_ids'])=={'Z1','Z2'}
    assert all(row['associated_pct_of_meter_kwh'] is None for row in result['zone_effects'])


def test_short_history_never_returns_an_effect_percentage():
    c,rows,features,groups=samples()
    result=analyze(rows,features,groups,c,'2026-02-01','2026-02-08')
    assert result['status']=='unavailable' and '60' in result['reason']
    assert all(r['associated_pct_of_meter_kwh'] is None for r in result['zone_effects'])


def test_operation_prediction_without_reporting_meter_never_invents_savings_or_percent():
    c,rows,features,groups=samples()
    rows=[r for r in rows if r['date']<'2026-04-11']
    result=analyze(rows,features,groups,c,'2026-04-11','2026-05-01')
    assert result['status']=='operation_prediction_only_requires_review'
    assert len(result['operation_predictions'])==20 and result['operation_model_kwh_available']>0
    assert result['reporting_meter_kwh_on_matched_periods'] is None
    assert all(r['associated_pct_of_meter_kwh'] is None for r in result['zone_effects'])


def test_shared_equipment_has_no_individual_zone_allocation():
    c,rows,features,groups=samples()
    groups[0]['zone_ids']=['Z1','Z1B']
    result=analyze(rows,features,groups,c,'2026-04-11','2026-05-01')
    assert result['groups'][0]['associated_kwh'] is not None
    assert result['zone_effects'][0]['associated_kwh'] is None


def test_unsupported_off_reference_is_unavailable_even_if_model_fits():
    c,rows,features,groups=samples()
    features.loc[features.equipment_id=='E1','heating_fraction']+=.1
    result=analyze(rows,features,groups,c,'2026-04-11','2026-05-01')
    assert result['status']=='modeled_associations_require_review'
    assert result['groups'][0]['associated_pct_of_meter_kwh'] is None
    assert result['groups'][0]['status']=='unavailable_no_supported_on_off_comparison'
