"""Explicit historical climatology scenario; never presented as a weather forecast."""
import numpy as np
import pandas as pd
from .data import DataError,digest,records


def climatology(history,start,end):
    past=history[history.date<start].copy()
    dates=pd.to_datetime(past.date);phase=np.array([pd.Timestamp(year=2000,month=t.month,day=t.day).dayofyear for t in dates])
    grid=pd.date_range(start,end,inclusive='left');rows=[]
    for t in grid:
        target=pd.Timestamp(year=2000,month=t.month,day=t.day).dayofyear
        distance=abs(phase-target);mask=np.minimum(distance,366-distance)<=7
        years=set(dates[mask].dt.year)
        if len(years)<2:raise DataError('Climatology needs at least two historical years near '+str(t.date()))
        rows.append({'date':str(t.date()),'temperature_c':float(past.loc[mask,'temperature_c'].mean())})
    if not rows:raise DataError('Empty weather scenario window')
    return pd.DataFrame(rows),{'kind':'historical_climatology_scenario_not_weather_forecast','method':'Mean of historical observations within +/-7 calendar days, at least two years; only source dates before forecast start',
        'start':start,'end_exclusive':end,'historical_sha256':digest(records(past)),'historical_end':past.date.max()}
