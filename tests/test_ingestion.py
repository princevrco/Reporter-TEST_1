import json
import hashlib
import pandas as pd
import pytest
from zipfile import ZipFile
from test_carma_directory import chunk
from option_c.ingestion import merge_meters,merge_bms
from option_c.data import DataError


def test_meter_overlap_is_not_added_and_revisions_are_explicit(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new'
    chunk(old,'A','2025-01-01','2025-01-04');p=chunk(new,'A','2025-01-03','2025-01-05')
    archive=tmp_path/'old.zip'
    with ZipFile(archive,'w') as z:
        for source in old.rglob('*'):
            if source.is_file():z.write(source,source.relative_to(old))
    f,a=merge_meters([archive,new]);assert len(f)==4 and f.energy_kwh.sum()==40
    assert a['duplicate_rows_collapsed']==1
    p.write_text(p.read_text().replace(',10',',20'))
    qpath=p.with_suffix('.quality.json');q=json.loads(qpath.read_text());q['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();qpath.write_text(json.dumps(q))
    with pytest.raises(DataError,match='revised'):merge_meters([archive,new])
    f,a=merge_meters([archive,new],'prefer-last')
    assert len(a['revised_readings'])==1 and f.energy_kwh.sum()==60


def test_bms_overlap_reconciles_cells_without_interpolation(tmp_path):
    old=tmp_path/'old.csv';new=tmp_path/'new.csv'
    pd.DataFrame({'timestamp':['2026-08-15T00:00:00Z'],'A':[1],'B':[2]}).to_csv(old,index=False)
    pd.DataFrame({'timestamp':['2026-08-15T00:00:00Z'],'A':[3],'B':[None]}).to_csv(new,index=False)
    with pytest.raises(DataError,match='conflicting'):merge_bms([old,new])
    f,a=merge_bms([old,new],'prefer-last')
    assert f.A.iloc[0]==3 and f.B.iloc[0]==2 and len(f)==1
    assert a['duplicate_timestamp_rows']==1


def test_merge_rejects_unsafe_zip_and_accepts_mixed_iso_precision(tmp_path):
    archive=tmp_path/'unsafe.zip'
    with ZipFile(archive,'w') as z:z.writestr('../outside_daily_kWh.csv','bad')
    with pytest.raises(DataError,match='Unsafe'):merge_meters([archive])
    path=tmp_path/'mixed.csv'
    pd.DataFrame({'timestamp':['2026-08-15T00:00:00.123456Z','2026-08-15T00:15:00Z'],'A':[1,0]}).to_csv(path,index=False)
    f,_=merge_bms([path]);assert len(f)==2
