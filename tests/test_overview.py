import json
import pandas as pd
import pytest
from fixtures import config,baseline_rows
from option_c.batch import run
from option_c.overview import total,build


def test_total_uses_common_pairs_and_retains_negative_money():
    a={'periods':[{'timestamp':'a','savings_kwh':10},{'timestamp':'b','savings_kwh':-20}]}
    b={'periods':[{'timestamp':'a','savings_kwh':2},{'timestamp':'b','savings_kwh':None}]}
    assert total({'x':a},[],.12)['savings_kwh'] is None
    assert total({'x':a},['x'],.12)['savings_cad']==-1.2
    r=total({'x':a,'y':b},['x','y'],.12)
    assert r['savings_kwh'] is None and r['paired_savings_kwh']==12
    assert r['paired_savings_cad']==pytest.approx(1.44)
    assert total({'x':a},['missing'],.12)['savings_cad'] is None


def test_dashboard_forecasts_without_future_measured_energy(tmp_path):
    c=config();mp=tmp_path/'meter.csv';bp=tmp_path/'bms.csv'
    pd.DataFrame(baseline_rows()+[{'date':'2026-01-01','meter_id':c['meter_id'],'energy_kwh':1000,'quality':'good'}]).to_csv(mp,index=False)
    pd.DataFrame({'timestamp':['2026-01-01T05:00:00Z']}).to_csv(bp,index=False)
    out=tmp_path/'results';run(mp,bp,c,'2026-01-01','2026-01-02',out,kind='normalized')
    r=build(out,.12,'2026-01-02','2026-01-05',[c['meter_id']])
    assert r['meters'][0]['forecast_kwh']>0
    assert r['meters'][0]['savings_cad']==pytest.approx(r['meters'][0]['savings_kwh']*.12)
    assert r['site_total']['forecast_kwh']==r['meters'][0]['forecast_kwh']
    assert (out/'index.html').exists() and 'future baseline forecast' in (out/'index.html').read_text()
    with pytest.raises(ValueError):build(out,-1)
