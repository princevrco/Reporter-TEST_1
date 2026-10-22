import pandas as pd
import pytest
from option_c.reconciliation import audit
from option_c.data import DataError
from option_c.quality import assess


def test_missing_and_inconsistent_are_not_repaired():
    frame=pd.DataFrame([{'date':day,'meter_id':meter,'energy_kwh':value}
        for day,values in [('2026-08-15',[10,6,4]),('2026-08-16',[0,6,0]),('2026-08-17',[10,6,None])]
        for meter,value in zip(['total','a','b'],values)])
    original=frame.copy(deep=True)
    result=audit(frame,[{'parent':'total','children':['a','b']}])
    assert [r['status'] for r in result]==['consistent','inconsistent','missing_readings']
    assert result[1]['difference_kwh']==-6
    assert result[2]['children_sum_kwh'] is None
    pd.testing.assert_frame_equal(frame,original)
    assert not any(r['repair_applied'] for r in result)
    with pytest.raises(DataError):audit(pd.concat([frame,frame]),[])


def test_isolated_virtual_zero_requires_review_and_preserves_source():
    frame=pd.DataFrame({'date':['2026-07-01','2026-07-02','2026-07-03','2026-08-15','2026-08-16','2026-08-17'],
        'energy_kwh':[10,0,12,10,0,12],'is_virtual_meter':[True]*6})
    cfg={'meter_id':'total','timezone':'America/Toronto','intervention_timestamp':'2026-08-15T00:00:00-04:00',
         'baseline_end_exclusive':'2026-07-24T00:00:00-04:00'}
    revised,events,stats=assess(frame,cfg,'2026-08-15','2026-08-18')
    zeros=[e for e in events if e['code']=='isolated_virtual_zero']
    assert {e['phase'] for e in zeros}=={'baseline','reporting'}
    assert stats['unresolved_reporting_events']==1
    assert revised.energy_kwh.tolist()==frame.energy_kwh.tolist()
    frame['is_virtual_meter']=False
    assert not any(e['code']=='isolated_virtual_zero' for e in assess(frame,cfg,'2026-08-15','2026-08-18')[1])
