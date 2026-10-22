import pandas as pd
import pytest
from option_c.control_forecast import fit_predict
from option_c.data import DataError


def test_controlled_forecast_uses_weekday_regression_and_chronological_holdout():
    rows=[{'timestamp':t.isoformat(),'reporting_kwh':24*(10+t.dayofweek)}
          for t in pd.date_range('2026-08-15',periods=67,tz='America/Toronto')]
    model,daily=fit_predict(rows,'America/Toronto','2026-10-21','2026-10-28')
    assert model['forecast_kwh']==pytest.approx(24*sum(range(10,17)))
    assert model['holdout_metrics']['cvrmse_pct']==pytest.approx(0)
    assert len(daily)==7
    # Changing held-out targets changes evaluation, not the frozen prior prediction.
    revised=[dict(p) for p in rows]
    for p in revised[-14:]:p['reporting_kwh']*=2
    other,_=fit_predict(revised,'America/Toronto','2026-10-21','2026-10-28')
    assert other['holdout_metrics']['cvrmse_pct']>20
    assert other['selected_candidate']==model['selected_candidate']
    with pytest.raises(DataError):fit_predict(rows[:41],'America/Toronto','2026-10-21','2026-10-28')
    with pytest.raises(DataError):fit_predict(rows,'America/Toronto','2026-10-20','2026-10-28')
