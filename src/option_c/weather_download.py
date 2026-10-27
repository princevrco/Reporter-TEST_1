"""Fetch and preserve explicit Open-Meteo historical daily temperature inputs."""
from datetime import date,datetime,timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from urllib.error import URLError
from zoneinfo import ZoneInfo
import hashlib,json,math
import pandas as pd
from .data import DataError,save,load
from .weather import read

ENDPOINT='https://archive-api.open-meteo.com/v1/archive'


def parameters(start,end,latitude=43.83128,longitude=-79.5159,timezone_name='America/Toronto',model='ecmwf_ifs'):
    try:
        a,b=date.fromisoformat(start),date.fromisoformat(end)
        if a.isoformat()!=start or b.isoformat()!=end or b<a or (b-a).days>7305:raise ValueError()
        ZoneInfo(timezone_name)
    except (ValueError,KeyError) as exc:raise DataError('Use increasing ISO dates (end inclusive), <=20 years, and a valid IANA timezone') from exc
    if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90<=latitude<=90 or not -180<=longitude<=180:
        raise DataError('Invalid weather latitude/longitude')
    if model not in ('ecmwf_ifs','era5','era5_land'):raise DataError('Unsupported historical weather model')
    return {'latitude':latitude,'longitude':longitude,'start_date':start,'end_date':end,'timezone':timezone_name,
            'models':model,'daily':'temperature_2m_mean','temperature_unit':'celsius'}


def normalize(payload,params):
    if payload.get('error'):raise DataError('Open-Meteo: '+str(payload.get('reason','API error')))
    if payload.get('timezone')!=params['timezone']:raise DataError('Weather API timezone does not match request')
    if payload.get('daily_units',{}).get('temperature_2m_mean') not in ('°C','C'):raise DataError('Weather API did not return Celsius')
    daily=payload.get('daily',{});dates=daily.get('time',[]);values=daily.get('temperature_2m_mean',[])
    expected=pd.date_range(params['start_date'],params['end_date']).date.astype(str).tolist()
    if dates!=expected or len(values)!=len(expected):raise DataError('Weather API dates are incomplete, duplicated or out of order')
    numeric=pd.to_numeric(pd.Series(values),errors='coerce')
    missing=[d for d,v in zip(dates,numeric) if pd.isna(v)]
    if missing:raise DataError('Historical weather is not yet available for '+', '.join(missing[:5])+'. Choose an earlier end date or a different explicit model; no gap filling.')
    if not numeric.between(-70,60).all():raise DataError('Weather API temperatures are nonfinite or outside supported Celsius range')
    return pd.DataFrame({'date':dates,'temperature_c':numeric})


def download(path,params):
    path=Path(path);audit_path=path.with_suffix('.provenance.json');raw_path=path.with_suffix('.source.json')
    if path.exists() or audit_path.exists() or raw_path.exists():
        if not all(p.exists() for p in (path,audit_path,raw_path)):raise DataError('Partial weather download exists; choose a new output filename')
        audit=load(audit_path)
        if audit.get('parameters')!=params:raise DataError('Weather cache belongs to different dates/location/model; choose a new filename')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=audit.get('csv_sha256') or hashlib.sha256(raw_path.read_bytes()).hexdigest()!=audit.get('raw_sha256'):
            raise DataError('Weather cache checksum mismatch')
        frame=normalize(load(raw_path),params);saved,_=read(path,params['timezone'])
        if saved.date.tolist()!=frame.date.tolist() or (saved.temperature_c-frame.temperature_c).abs().max()>1e-10:
            raise DataError('Weather cache values do not match saved API response')
        return audit,True
    url=ENDPOINT+'?'+urlencode(params)
    try:
        with urlopen(Request(url,headers={'User-Agent':'promenade-option-c-weather/0.6.2.post2'}),timeout=60) as response:
            raw=response.read(20_000_001)
        if len(raw)>20_000_000:raise DataError('Weather response exceeds size limit')
        payload=json.loads(raw)
    except (URLError,TimeoutError,json.JSONDecodeError) as exc:raise DataError('Weather download failed: '+str(exc)) from exc
    frame=normalize(payload,params)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Validate all remote values before writing a model input. Never overwrite an existing dataset.
    raw_path.write_bytes(raw);frame.to_csv(path,index=False)
    audit={'provider':'Open-Meteo historical API','source_url':url,'parameters':params,'rows':len(frame),
           'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
           'grid_latitude':payload.get('latitude'),'grid_longitude':payload.get('longitude'),
           'raw_sha256':hashlib.sha256(raw).hexdigest(),'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'documentation':'https://open-meteo.com/en/docs/historical-weather-api',
           'scope':'Historical gridded weather estimate, not on-site sensor readings or a future forecast. Provider usage terms apply.'}
    save(audit_path,audit)
    return audit,False
