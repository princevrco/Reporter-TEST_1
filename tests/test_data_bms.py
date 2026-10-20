import copy,json
import numpy as np
import pandas as pd
import pytest
from fixtures import config
from option_c.data import DataError,frame
from option_c.bms import aggregate
from option_c.importers import prepare
from option_c.reporting import write


def test_duplicate_or_negative_energy_is_not_silently_reconciled():
    c=config();r={'date':'2025-01-01','energy_kwh':50}
    with pytest.raises(DataError,match='Duplicate'):frame([r,r],c)
    r['energy_kwh']=-1
    with pytest.raises(DataError,match='Negative'):frame([r],c)


def test_bms_overlap_invalid_states_and_missing_hours(tmp_path):
    c=config();c['bms_features']={'bms_fan':{'column':'fan','allowed_values':[0,1],'minimum_hour_coverage':.9}}
    times=pd.date_range('2026-04-20',periods=24,freq='1h',tz='America/Toronto')
    f=pd.DataFrame({'timestamp':times.tz_convert('UTC').astype(str),'fan':np.ones(24)})
    p=tmp_path/'history.csv';u=tmp_path/'update.csv';f.to_csv(p,index=False)
    update=f.copy();update.loc[0,'fan']=896;update.to_csv(u,index=False)
    out,audit=aggregate([p,u],c)
    assert out.bms_fan.iloc[0]==pytest.approx(1)
    assert audit['conflicting_overlap_cells']==1
    assert audit['point_audit']['bms_fan']['rejected_nonmissing_samples']==1
    update.loc[:3,'fan']=.5;update.to_csv(u,index=False)
    out,audit=aggregate([p,u],c)
    assert pd.isna(out.bms_fan.iloc[0])


def test_bms_import_requires_confirmed_point_semantics(tmp_path):
    c=config();c['review']['bms_semantics_verified']=False
    with pytest.raises(DataError,match='verified'):aggregate([],c)


def test_portal_source_quality_preserved_and_meters_separate(tmp_path):
    f=pd.DataFrame([{'date_portal':'2025-01-01','meter_code':'A','consumption_kwh':100,'reconciliation_review':False},
                    {'date_portal':'2025-01-02','meter_code':'A','consumption_kwh':120,'reconciliation_review':True},
                    {'date_portal':'2025-01-01','meter_code':'B','consumption_kwh':900,'reconciliation_review':False}])
    p=tmp_path/'meter.csv';f.to_csv(p,index=False);out=prepare(p,'A')
    assert len(out)==2 and out.raw_energy_kwh.sum()==220
    assert out.energy_kwh.notna().sum()==1


def test_report_escapes_script_content_and_has_offline_chart(tmp_path):
    r={'schema':'option_c.forecast.v1','status':'review','scope':'</script><script>bad()</script>','start':'2026-01-01T05:00:00Z','end_exclusive':'2026-01-02T05:00:00Z','expected_periods':1,'periods':[{'timestamp':'2026-01-01T05:00:00Z','baseline_kwh':100}]}
    write(r,tmp_path)
    text=(tmp_path/'report.html').read_text(encoding='utf-8')
    assert '<script>bad()' not in text
    assert 'svg id="chart"' in text and 'type="date"' in text
    assert json.loads((tmp_path/'result.json').read_text())['scope']==r['scope']


def test_optimizer_report_preserves_embedded_data_and_escapes_labels(tmp_path):
    zone='<script>bad()</script> r.baseline_kwh,r.reporting_kwh'
    r={'schema':'option_c.savings.v1','status':'review','scope':'test','start':'2026-01-01T05:00:00Z',
       'end_exclusive':'2026-01-02T05:00:00Z','expected_periods':1,'periods':[],
       'zone_meter_effects':{'scope':'Statistical associations','status':'unavailable','zone_effects':[{'zone_id':zone}]}}
    write(r,tmp_path);html=(tmp_path/'report.html').read_text()
    assert '<script>bad()' not in html and '&lt;script&gt;bad()' in html
    data,_=json.JSONDecoder().raw_decode(html.split('<script>const R=',1)[1])
    assert data==r
