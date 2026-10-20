import csv
import pandas as pd
import pytest
from option_c.data import DataError
from option_c.importers import prepare,discover
from option_c.batch import run
from fixtures import config


def export(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        csv.writer(stream).writerows(rows)


def test_carma_all_meters_footer_missing_zero_and_virtual_identity(tmp_path):
    p=tmp_path/'raw.csv'
    export(p,[['ReadingDate','E004M11 - AC-unit-5J - Promenade Mall','vm_total_load - Total load - Promenade Mall'],
              ['8/1/2025 12:00:00 AM','0','100'],['8/2/2025 12:00:00 AM','','-1'],
              ['Date Exported: 10/16/2026 12:28:14 PM']])
    with pytest.raises(DataError,match='units'):prepare(p,kind='carma')
    f=prepare(p,kind='carma',carma_daily_kwh=True)
    assert len(f)==4 and set(discover(p,'carma',True))=={'E004M11','vm_total_load'}
    zero=f[(f.meter_id=='E004M11')&(f.date=='2025-08-01')].iloc[0]
    assert zero.energy_kwh==0 and zero.quality=='good'
    assert f.energy_kwh.notna().sum()==2
    assert f[f.is_virtual_meter].meter_id.unique().tolist()==['vm_total_load']
    assert f.source_exported_at_local.unique().tolist()==['2026-10-16T12:28:14']
    assert len(prepare(p,'E004M11',kind='carma',carma_daily_kwh=True))==2


@pytest.mark.parametrize('header,data,error',[
    (['ReadingDate','A - Meter','A - Duplicate'],[['8/1/2025 12:00:00 AM','1','2']],'identifiers'),
    (['ReadingDate','A - Meter'],[['8/1/2025 01:00:00 AM','1']],'midnight'),
    (['ReadingDate','A - Meter'],[['8/1/2025 12:00:00 AM','1','2']],'width'),
    (['ReadingDate','A - Meter'],[['8/1/2025 12:00:00 AM','1'],['8/1/2025 12:00:00 AM','2']],'Duplicate')])
def test_carma_rejects_ambiguous_layouts(tmp_path,header,data,error):
    p=tmp_path/'bad.csv';export(p,[header]+data)
    with pytest.raises(DataError,match=error):prepare(p,kind='carma',carma_daily_kwh=True)


def test_carma_quarantines_incomplete_export_day_and_one_day_cannot_train(tmp_path):
    p=tmp_path/'one-day.csv'
    export(p,[['ReadingDate','A - Meter'],['8/1/2025 12:00:00 AM','25'],['Date Exported: 8/1/2025 12:00:00 PM']])
    f=prepare(p,kind='carma',carma_daily_kwh=True)
    assert pd.isna(f.energy_kwh.iloc[0]) and 'incomplete' in f.source_flags.iloc[0]
    bms=tmp_path/'bms.csv';export(bms,[['timestamp','fan'],['2026-01-01T05:00:00Z',1]])
    result=run(p,bms,config(),'2026-01-01','2026-01-02',tmp_path/'out','carma',True)
    assert result['status']=='all_meters_blocked'
    assert result['meters'][0]['meter_id']=='A'
    assert 'Too few' in result['meters'][0]['reason']
