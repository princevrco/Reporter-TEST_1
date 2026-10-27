import copy
from pathlib import Path
import pandas as pd
import pytest
from option_c.data import DataError,load
from option_c.utility import read
from option_c.ingestion import merge_meters
from option_c.quality import assess


def fixture(tmp_path,monkeypatch):
    source=tmp_path/'compiled.xlsx';source.write_bytes(b'fixture placeholder; read_excel is mocked')
    sheets={'Daily kWh':pd.DataFrame([
        ['Report title',None,None,None,None,None,None],
        ['Date','PWST104274 kWh','PWST152089 kWh','PWST104274 review','PWST152089 review','PWST104274 record','PWST152089 record'],
        [pd.Timestamp('2026-10-01'),499.8,20000.,'Low reading; completeness unverified',None,'A1','B1'],
        [pd.Timestamp('2026-10-02'),24000.,None,'Revised reading; earlier values retained','No reading available','A2',None]]),
        'Source readings':pd.DataFrame([
        ['Record ID','Meter','Date','kWh','Use','Source filename','Source type'],
        ['A1','PWST104274',pd.Timestamp('2026-10-01'),499.8,'Selected','new.xlsx','Native'],
        ['B1','PWST152089',pd.Timestamp('2026-10-01'),20000.,'Selected','new.xlsx','Native'],
        ['A2','PWST104274',pd.Timestamp('2026-10-02'),24000.,'Selected','new.xlsx','Native'],
        ['A0','PWST104274',pd.Timestamp('2026-10-02'),500.,'Not selected','old.xlsx','Native']])}
    monkeypatch.setattr(pd,'read_excel',lambda *a,**k:copy.deepcopy(sheets))
    return source,sheets


def test_compiled_workbook_uses_selected_versions_not_sum_and_preserves_flags(tmp_path,monkeypatch):
    path,sheets=fixture(tmp_path,monkeypatch);f=read(path)
    a=f[f.meter_id=='PWST104274'];b=f[f.meter_id=='PWST152089']
    assert a.energy_kwh.sum()==24499.8
    assert a.source_review_required.all() and not b.source_review_required.any()
    assert b.energy_kwh.notna().sum()==1 and b.quality.tolist()==['good','missing']
    assert f.attrs['source_audit']['unselected_source_records']==1
    assert f.attrs['source_audit']['selected_source_records']==3
    cfg=load(Path(__file__).parents[1]/'config/promenade-test.json');cfg['meter_id']='PWST104274'
    checked,events,stats=assess(a,cfg,'2026-08-15','2026-10-21')
    assert checked.energy_kwh.sum()==24499.8
    assert sum(e['code']=='source_reading_review' for e in events)==2
    assert stats['unresolved_reporting_events']==2
    from option_c.data import canonical
    canonical(stats)  # No-baseline quality audits must remain valid finite JSON.


@pytest.mark.parametrize('problem',['value','selection','meter','record','orphan_selected'])
def test_selected_source_lineage_mismatches_are_rejected(tmp_path,monkeypatch,problem):
    path,sheets=fixture(tmp_path,monkeypatch)
    if problem=='value':sheets['Source readings'].iloc[1,3]=600.
    if problem=='selection':sheets['Source readings'].iloc[1,4]='Not selected'
    if problem=='meter':sheets['Source readings'].iloc[1,1]='OTHER'
    if problem=='record':sheets['Daily kWh'].iloc[2,5]='MISSING'
    if problem=='orphan_selected':sheets['Source readings'].iloc[4,4]='Selected'
    with pytest.raises(DataError):read(path)


def test_auto_ingestion_accepts_utility_and_does_not_duplicate_energy(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch);f,audit=merge_meters([path,path],kind='auto')
    assert len(f)==4 and audit['duplicate_rows_collapsed']==4
    assert f.energy_kwh.sum()==44499.8


def test_legacy_per_meter_quality_note_does_not_exclude_other_meter(tmp_path,monkeypatch):
    path=tmp_path/'old.xlsx';path.write_bytes(b'test')
    sheets={'Daily kWh':pd.DataFrame([['Date','PWST104274 (kWh)','PWST152089 (kWh)'],[pd.Timestamp('2026-08-09'),20000.,22000.]]),
        'Recent daily':pd.DataFrame([['Date','PWST104274 (kWh)','PWST152089 (kWh)','Availability / quality note'],[pd.Timestamp('2026-08-10'),546.,24686.2,'PWST104274: incomplete day; PWST152089: reading available']])}
    monkeypatch.setattr(pd,'read_excel',lambda *a,**k:copy.deepcopy(sheets));f=read(path)
    assert f[(f.meter_id=='PWST104274')&(f.date=='2026-08-10')].energy_kwh.isna().all()
    assert f[(f.meter_id=='PWST152089')&(f.date=='2026-08-10')].energy_kwh.iloc[0]==24686.2
