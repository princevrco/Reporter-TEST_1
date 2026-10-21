import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from test_carma_directory import chunk


def test_combined_old_new_data_completes_requested_window(tmp_path):
    repo=Path(__file__).parents[1]
    old=tmp_path/'meter-old';new=tmp_path/'meter-new'
    chunk(old,'A','2025-01-01','2026-10-16')
    p=chunk(new,'A','2026-10-10','2026-10-21')
    p.write_text(p.read_text().replace('10/16/2026 12:00:00 PM','10/21/2026 12:00:00 PM'))
    qpath=p.with_suffix('.quality.json');q=json.loads(qpath.read_text());q['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();qpath.write_text(json.dumps(q))
    def bms(path,start,end):
        t=pd.date_range(start,end,freq='15min',tz='America/Toronto',inclusive='left')
        day=np.array([x.date().toordinal() for x in t]);hour=t.hour.to_numpy()
        pd.DataFrame({'timestamp':t.tz_convert('UTC'),'RTU_Cool1':(hour<(12+day%6)).astype(int),
            'RTU_ZoneTemp':24+np.sin(day*.2),'RTU_CoolSetpointU_AV':23+np.cos(day*.3)}).to_csv(path,index=False)
    a=tmp_path/'bms-old.csv';b=tmp_path/'bms-new.csv'
    bms(a,'2026-07-24','2026-10-14');bms(b,'2026-10-10','2026-10-21')
    out=tmp_path/'results';env=os.environ.copy();env['PYTHONPATH']=str(repo/'src')
    process=subprocess.run([sys.executable,str(repo/'scripts/analyze_promenade.py'),
        '--meter',str(old),str(new),'--bms',str(a),str(b),'--output',str(out),
        '--forecast-start','2026-10-21','--forecast-end','2027-01-21'],cwd=repo,env=env,capture_output=True,text=True)
    assert process.returncode==0,process.stdout+process.stderr
    summary=json.loads((out/'summary.json').read_text())
    result=json.loads((out/summary['meters'][0]['folder']/'savings/result.json').read_text())
    assert result['expected_periods']==67 and result['observed_meter_periods']==67
    assert result['reporting_kwh_paired']==670
    # Constant daily kWh crosses a 23-hour training day; the average-kW baseline
    # handles real day lengths and may differ slightly from a constant-energy fit.
    assert abs(result['savings_kwh_complete_period'])<.1
    response=pd.read_csv(out/'response-summary.csv')
    assert response.status.iloc[0]=='fitted_requires_review'
    assert (out/'index.html').exists()
    assert json.loads((out/'overview.json').read_text())['meters'][0]['future_optimized_kwh'] is None
    assert json.loads((out/'merge-audit.json').read_text())['meter']['duplicate_rows_collapsed']==6
