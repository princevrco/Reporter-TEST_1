import json
import pandas as pd
import pytest
from fixtures import config,baseline_rows
from option_c.batch import run,bms_context
from option_c.importers import prepare
from option_c.data import DataError


def test_all_meter_import_preserves_identifiers_and_flags(tmp_path):
    p=tmp_path/'portal.csv'
    pd.DataFrame([{'date_portal':'2025-01-01','meter_code':'001','consumption_kwh':12},
                  {'date_portal':'2025-01-01','meter_code':'002','consumption_kwh':24,'reconciliation_review':True}]).to_csv(p,index=False)
    f=prepare(p)
    assert f.meter_id.tolist()==['001','002']
    assert pd.isna(f.energy_kwh.iloc[1])


def test_two_file_batch_separates_meters_and_never_infers_missing_energy(tmp_path):
    c=config();c['review']['bms_semantics_verified']=False
    history=baseline_rows()
    rows=[]
    for meter in ('001','002'):
        rows += [{**r,'meter_id':meter} for r in history]
        rows += [{'date':'2026-01-01','meter_id':meter,'energy_kwh':900,'quality':'good'}]
    # This meter is discovered but lacks enough historical readings. Other meters still run.
    rows.append({'date':'2025-12-01','meter_id':'003','energy_kwh':100,'quality':'good'})
    # A corrupt meter with duplicate dates must not abort unrelated meter reports.
    rows.extend([{'date':'2025-12-01','meter_id':'004','energy_kwh':100,'quality':'good'}]*2)
    mp=tmp_path/'meter.csv';pd.DataFrame(rows).to_csv(mp,index=False)
    bp=tmp_path/'bms.csv'
    pd.DataFrame({'timestamp':['2026-01-01T05:00:00Z','2026-01-02T05:00:00Z'],'fan':[1,1]}).to_csv(bp,index=False)
    out=tmp_path/'results'
    summary=run(mp,bp,c,'2026-01-01','2026-01-03',out,kind='normalized')
    assert summary['meter_count']==4 and summary['blocked_meter_count']==2
    assert summary['site_total_kwh'] is None
    assert summary['meters'][2]['status']=='blocked'
    assert 'Duplicate' in summary['meters'][3]['reason']
    for entry in summary['meters'][:2]:
        result=json.loads((out/entry['folder']/'savings/result.json').read_text())
        assert result['observed_meter_periods']==1 and result['unavailable_periods']==1
        assert result['savings_kwh_complete_period'] is None
        assert result['bms_estimated_periods']==0
        assert result['periods'][1]['reporting_source']=='unavailable'
        assert result['periods'][1]['bms_samples']==1
        model=json.loads((out/entry['folder']/'baseline-model.json').read_text())
        assert 'weather_not_modeled' in model['gates']
        assert model['config']['review']['meter_boundary_verified'] is False
        assert all(t['kind'] in ('constant','weekday') for t in model['terms'])
    # No reporting readings: still produce a forecast but never report zero savings.
    future=run(mp,bp,c,'2026-01-03','2026-01-05',tmp_path/'future',kind='normalized')
    for entry in future['meters'][:2]:
        assert entry['forecast_kwh'] is not None
        assert entry['savings_kwh_paired'] is None
        assert entry['status']=='unavailable_no_paired_consumption'


def test_bms_context_rejects_ambiguous_time(tmp_path):
    p=tmp_path/'bms.csv'
    pd.DataFrame({'timestamp':['2026-01-01 12:00:00'],'fan':[1]}).to_csv(p,index=False)
    with pytest.raises(DataError,match='offsets'):bms_context(p,'America/Toronto')


def test_all_meter_workbook_discovers_separate_boundaries(tmp_path):
    pytest.importorskip('openpyxl')
    p=tmp_path/'utility.xlsx'
    with pd.ExcelWriter(p,engine='openpyxl') as writer:
        pd.DataFrame({'Date':['2025-01-01'],'A (kWh)':[10],'B (kWh)':[20]}).to_excel(writer,sheet_name='Daily kWh',index=False)
        pd.DataFrame({'Date':['2025-01-02'],'B (kWh)':[30]}).to_excel(writer,sheet_name='Recent daily',index=False)
    f=prepare(p,kind='utility')
    assert set(f.meter_id)=={'A','B'} and len(f)==3
