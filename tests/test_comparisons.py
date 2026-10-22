import pytest
from option_c.comparisons import compare
from option_c.data import DataError


def test_matching_dates_and_separate_consumption_interpretations():
    periods=[{'timestamp':'2026-08-15T04:00:00Z','baseline_kwh':100,'reporting_kwh':90},
             {'timestamp':'2026-08-16T04:00:00Z','baseline_kwh':200,'reporting_kwh':220}]
    sp=[{'date':'2026-08-16','diagnostic_fitted_setpoint_kwh':210,'heldout_setpoint_kwh':205}]
    result,daily=compare(periods,sp,'2026-08-15','2026-08-17','America/Toronto',.12)
    assert result['baseline_meter_difference_kwh']==-10
    assert result['baseline_meter_difference_cad']==pytest.approx(-1.2)
    assert result['baseline_meter_complete_difference_kwh']==-10
    assert result['baseline_optimizer_reconstruction_difference_kwh']==-10
    assert result['meter_optimizer_reconstruction_difference_kwh']==10
    assert result['baseline_optimizer_heldout_difference_kwh']==-5
    assert result['baseline_optimizer_reconstruction_paired_days']==1
    assert result['baseline_optimizer_reconstruction_complete_difference_kwh'] is None
    assert daily[0]['optimizer_reconstructed_kwh'] is None


def test_zero_is_retained_but_missing_is_not_zero():
    periods=[{'timestamp':'2026-08-15T04:00:00Z','baseline_kwh':0,'reporting_kwh':0}]
    result,_=compare(periods,[],'2026-08-15','2026-08-17','America/Toronto',.12)
    assert result['baseline_meter_difference_kwh']==0
    assert result['baseline_meter_complete_difference_kwh'] is None
    assert result['meter_optimizer_reconstruction_difference_kwh'] is None
    with pytest.raises(DataError):compare(periods*2,[],'2026-08-15','2026-08-17','America/Toronto',.12)


def test_invalid_energy_cannot_be_reported_as_difference():
    with pytest.raises(DataError):
        compare([{'timestamp':'2026-08-15T04:00:00Z','baseline_kwh':-1}],[],
                '2026-08-15','2026-08-16','America/Toronto',.12)
