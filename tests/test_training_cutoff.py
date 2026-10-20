import json
import pandas as pd
import pytest
from fixtures import config, baseline_rows
from option_c.batch import run
from option_c.data import DataError


def test_earlier_baseline_boundary_excludes_uncertain_control_history(tmp_path):
    cfg=config();cfg['baseline_end_exclusive']='2025-07-01T00:00:00-04:00'
    rows=baseline_rows()
    for row in rows:
        if row['date']>='2025-07-01':row['energy_kwh']=1e9
    meter=tmp_path/'meter.csv';pd.DataFrame(rows).to_csv(meter,index=False)
    bms=tmp_path/'bms.csv';pd.DataFrame({'timestamp':['2026-01-01T05:00:00Z']}).to_csv(bms,index=False)
    out=tmp_path/'result'
    summary=run(meter,bms,cfg,'2026-01-01','2026-01-02',out,kind='normalized')
    entry=summary['meters'][0]
    model=json.loads((out/entry['folder']/'baseline-model.json').read_text())
    assert model['valid_periods']==181
    assert entry['forecast_kwh']<100000
    cfg['baseline_end_exclusive']='2026-02-01T00:00:00-05:00'
    with pytest.raises(DataError,match='Baseline end'):
        run(meter,bms,cfg,'2026-01-01','2026-01-02',tmp_path/'bad',kind='normalized')
