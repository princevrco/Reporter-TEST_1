import json
import numpy as np
import pandas as pd
from fixtures import config,baseline_rows
from option_c.batch import run


def test_optimizer_model_learns_each_meter_and_preserves_measured_savings(tmp_path):
    c=config();c['optimizer']={'bin_minutes':60,'minimum_training_days':60,'calendar_features':True}
    rng=np.random.default_rng(57);meter_rows=[];optimizer_rows=[];bms_rows=[]
    for meter in ('A','B'):meter_rows.extend({**r,'meter_id':meter} for r in baseline_rows())
    pattern=[rng.integers(0,7,4).tolist() for _ in range(100)]
    for i,day in enumerate(pd.date_range('2026-01-01',periods=120,freq='D',tz=c['timezone'])):
        times=pd.date_range(day,day+pd.DateOffset(days=1),freq='1h',inclusive='left')
        h1,c1,h2,c2=pattern[i%100]
        if i%4==0:h1=c1=0
        if i%5==0:h2=c2=0
        for j,t in enumerate(times):
            bms_rows.append({'timestamp':t.isoformat(),'fan':1})
            for zone,h,co in [('Z1',h1,c1),('Z2',h2,c2)]:
                mode=1 if j<h else -1 if j<h+co else 0
                optimizer_rows.append({'timestamp':t.isoformat(),'issued_at':t.isoformat(),'zone_id':zone,'achieved_mode':mode})
        hours=len(times);activity=8*h1+12*c1+4*h2+6*c2
        for meter,scale in [('A',1),('B',2)]:
            meter_rows.append({'date':day.date().isoformat(),'meter_id':meter,'energy_kwh':40*hours+scale*activity,'quality':'good'})
    mp=tmp_path/'meters.csv';op=tmp_path/'optimizer.csv';bp=tmp_path/'bms.csv'
    pd.DataFrame(meter_rows).to_csv(mp,index=False);pd.DataFrame(optimizer_rows).to_csv(op,index=False);pd.DataFrame(bms_rows).to_csv(bp,index=False)
    out=tmp_path/'results'
    summary=run(mp,bp,c,'2026-04-11','2026-05-01',out,'normalized',optimizer_paths=[op])
    assert summary['blocked_meter_count']==0 and summary['optimizer_supplied']
    assert len(summary['zone_duration_summary'])==2
    effects=pd.read_csv(out/'zone-meter-effects.csv')
    assert set(effects.meter_id)=={'A','B'} and set(effects.zone_id)=={'Z1','Z2'}
    a=effects[(effects.meter_id=='A')&(effects.zone_id=='Z1')].iloc[0]
    b=effects[(effects.meter_id=='B')&(effects.zone_id=='Z1')].iloc[0]
    assert a.associated_pct_of_meter_kwh>0 and b.associated_pct_of_meter_kwh>a.associated_pct_of_meter_kwh
    for entry in summary['meters']:
        folder=out/entry['folder'];result=json.loads((folder/'savings/result.json').read_text())
        assert result['bms_estimated_periods']==0
        assert all(row['reporting_source']=='meter' for row in result['periods'])
        assert result['zone_meter_effects']['status']=='modeled_associations_require_review'
        assert (folder/'operation-response-model.json').is_file()
        html=(folder/'savings/report.html').read_text()
        assert 'Optimizer modes and meter associations' in html
        assert 'Operation model kWh' in html and "['operation_model_kwh','#8451a1']" in html
    # Electricity baseline artifacts contain no optimizer mode terms.
    for entry in summary['meters']:
        baseline=json.loads((out/entry['folder']/'baseline-model.json').read_text())
        assert all(t['kind'] in ('constant','weekday') for t in baseline['terms'])
