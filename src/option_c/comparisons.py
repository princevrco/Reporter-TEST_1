"""Signed energy comparisons on identical dates, independent of release gates."""
import math
import pandas as pd
from .data import DataError


def compare(periods, setpoints, start, end, timezone, rate):
    if not math.isfinite(rate) or rate<0:raise DataError('Invalid comparison rate')
    def number(value):
        if value is None or pd.isna(value):return None
        value=float(value)
        if not math.isfinite(value) or value<0:raise DataError('Consumption must be finite and nonnegative')
        return value
    past={}
    for row in periods:
        day=str(pd.Timestamp(row['timestamp']).tz_convert(timezone).date())
        if day in past:raise DataError('Duplicate comparison date')
        past[day]=row
    sp={}
    for row in setpoints:
        if row['date'] in sp:raise DataError('Duplicate setpoint comparison date')
        sp[row['date']]=row
    daily=[]
    for day in pd.date_range(start,end,inclusive='left'):
        date=str(day.date());p=past.get(date,{});s=sp.get(date,{})
        baseline=number(p.get('baseline_kwh'));measured=number(p.get('reporting_kwh'))
        fitted=number(s.get('diagnostic_fitted_setpoint_kwh'));held=number(s.get('heldout_setpoint_kwh'))
        row={'date':date,'baseline_kwh':baseline,'measured_kwh':measured,
             'optimizer_reconstructed_kwh':fitted,'optimizer_heldout_kwh':held}
        for name,left,right in [('baseline_meter',baseline,measured),
            ('baseline_optimizer_reconstruction',baseline,fitted),
            ('meter_optimizer_reconstruction',measured,fitted),
            ('baseline_optimizer_heldout',baseline,held),('meter_optimizer_heldout',measured,held)]:
            value=left-right if left is not None and right is not None else None
            row[name+'_difference_kwh']=value
            row[name+'_difference_cad']=value*rate if value is not None else None
        daily.append(row)
    result={'comparison_expected_days':len(daily)}
    for name,left,right in [('baseline_meter','baseline_kwh','measured_kwh'),
        ('baseline_optimizer_reconstruction','baseline_kwh','optimizer_reconstructed_kwh'),
        ('meter_optimizer_reconstruction','measured_kwh','optimizer_reconstructed_kwh'),
        ('baseline_optimizer_heldout','baseline_kwh','optimizer_heldout_kwh'),
        ('meter_optimizer_heldout','measured_kwh','optimizer_heldout_kwh')]:
        paired=[r for r in daily if r[name+'_difference_kwh'] is not None]
        value=sum(r[name+'_difference_kwh'] for r in paired) if paired else None
        result.update({name+'_paired_days':len(paired),name+'_left_kwh':sum(r[left] for r in paired) if paired else None,
            name+'_right_kwh':sum(r[right] for r in paired) if paired else None,
            name+'_difference_kwh':value,name+'_difference_cad':value*rate if value is not None else None,
            name+'_complete_difference_kwh':value if paired and len(paired)==len(daily) else None,
            name+'_complete_difference_cad':value*rate if paired and len(paired)==len(daily) else None})
    result['comparison_interpretation']='Baseline minus measured is estimated avoided energy. Baseline minus optimizer is a model-to-model estimate. Meter minus optimizer is prediction residual, not savings. Reconstruction is fitted history; heldout values are separate validation evidence. All amounts use matching dates only.'
    return result,daily
