import csv
import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest
from option_c.carma import source
from option_c.importers import prepare
from option_c.data import DataError


def chunk(root,meter,start,end):
    p=root/meter/f'{meter}_{start}_{end}_daily_kWh.csv';p.parent.mkdir(parents=True,exist_ok=True)
    days=pd.date_range(start,end,freq='D',inclusive='left')
    with p.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['ReadingDate',meter+' - Synthetic'])
        for t in days:writer.writerow([t.strftime('%m/%d/%Y %I:%M:%S %p'),10])
        writer.writerow(['Date Exported: 10/16/2026 12:00:00 PM'])
    q={'meter':meter,'mode':'Consumption','unit':'kWh','resolution':'daily','requested_start':start+'T00:00:00',
       'requested_end':end+'T00:00:00','rows':len(days),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'warnings':[]}
    p.with_suffix('.quality.json').write_text(json.dumps(q))
    return p


def test_directory_combines_chunks_and_excludes_duplicate_batch_responses(tmp_path):
    chunk(tmp_path,'A','2025-01-01','2025-01-03');chunk(tmp_path,'A','2025-01-03','2025-01-05')
    chunk(tmp_path,'B','2025-01-01','2025-01-05')
    chunk(tmp_path/'_batches','A','2025-01-01','2025-01-03')
    chunk(tmp_path/'demand_kw','A','2025-01-01','2025-01-03')
    f,audit=source(tmp_path,True)
    assert len(f)==8 and set(f.meter_id)=={'A','B'} and audit['file_count']==3
    imported=prepare(tmp_path,kind='carma',carma_daily_kwh=True)
    assert imported.attrs['source_audit']['sha256']==audit['sha256']


def test_directory_rejects_checksum_tampering(tmp_path):
    p=chunk(tmp_path,'A','2025-01-01','2025-01-03');p.write_text(p.read_text().replace(',10',',99'))
    with pytest.raises(DataError,match='checksum'):source(tmp_path,True)


def test_directory_rejects_overlap_and_demand_metadata(tmp_path):
    chunk(tmp_path,'A','2025-01-01','2025-01-03');p=chunk(tmp_path,'A','2025-01-02','2025-01-04')
    with pytest.raises(DataError,match='Overlapping'):source(tmp_path,True)
    qpath=p.with_suffix('.quality.json');q=json.loads(qpath.read_text());q['unit']='kW';qpath.write_text(json.dumps(q))
    with pytest.raises(DataError,match='Consumption/kWh/daily'):source(tmp_path,True)


def test_directory_requires_quality_json_and_actual_date_grid(tmp_path):
    p=chunk(tmp_path,'A','2025-01-01','2025-01-03');qpath=p.with_suffix('.quality.json')
    q=json.loads(qpath.read_text());q['requested_end']='2025-01-04T00:00:00';qpath.write_text(json.dumps(q))
    with pytest.raises(DataError,match='daily grid'):source(tmp_path,True)
    qpath.unlink()
    with pytest.raises(DataError,match='quality.json'):source(tmp_path,True)
