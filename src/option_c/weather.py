"""Explicit daily external weather: no interpolation, imputation or network access."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .data import DataError


def read(path, timezone):
    path=Path(path);f=pd.read_csv(path,dtype={'date':str})
    if 'date' not in f:raise DataError('Weather requires ISO local date column')
    names=[c for c in ('temperature_c','outdoor_temp_c') if c in f]
    if len(names)!=1:raise DataError('Weather requires exactly one temperature_c or outdoor_temp_c column in Celsius')
    if f.empty or f.date.isna().any() or not f.date.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
        raise DataError('Weather dates must be nonempty ISO local YYYY-MM-DD dates')
    try:
        parsed=pd.to_datetime(f.date,format='%Y-%m-%d',errors='raise')
        if not parsed.dt.strftime('%Y-%m-%d').eq(f.date).all():raise ValueError()
    except ValueError as exc:raise DataError('Invalid weather calendar date') from exc
    if f.date.duplicated().any():raise DataError('Duplicate daily weather dates')
    if 'timezone' in f and not f.timezone.eq(timezone).all():raise DataError('Weather timezone differs from site timezone')
    if 'unit' in f and not f.unit.isin(['C','°C','celsius']).all():raise DataError('Weather units must be Celsius')
    values=pd.to_numeric(f[names[0]],errors='coerce')
    if not (np.isfinite(values)&values.between(-70,60)).all():raise DataError('Missing, nonfinite or out-of-range Celsius weather; no silent fallback')
    result=pd.DataFrame({'date':f.date,'temperature_c':values}).sort_values('date').reset_index(drop=True)
    audit={'source_filename':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'rows':len(result),'first_date':result.date.min(),'last_date':result.date.max(),
           'timezone':timezone,'unit':'C','aggregation':'User-supplied local daily mean outdoor temperature',
           'scope':'Date timezone and temperature source must be confirmed; no external source automatically verified',
           'missing_policy':'No interpolation, gap filling or calendar-only fallback when weather is requested'}
    return result,audit


def join(frame, weather, column='temperature_c'):
    if 'date' not in frame:raise DataError('Daily weather join requires date')
    dates=frame.date.astype(str)
    missing=sorted(set(dates)-set(weather.date))
    if missing:raise DataError('Missing external weather for '+str(len(missing))+' dates; first: '+', '.join(missing[:5]))
    values=weather.set_index('date').temperature_c
    result=frame.copy();result[column]=dates.map(values).to_numpy(float)
    return result


def features(frame,weather,required=False):
    if weather is None:
        if required:raise DataError('This response model requires supplied daily external weather')
        return frame
    return join(frame,weather,'outdoor_temperature_c')
