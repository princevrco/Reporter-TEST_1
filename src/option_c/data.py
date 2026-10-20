from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd


class DataError(ValueError): pass


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def save(path,value):
    import os,tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        stream.write(canonical(value));stream.flush();os.fsync(stream.fileno());name=stream.name
    os.replace(name,path)


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def stamp(value,tz,allow_date=False):
    t=pd.Timestamp(value)
    if t.tzinfo is None:
        if not allow_date or len(str(value))!=10: raise DataError('Timestamp requires explicit offset; daily dates use the declared site timezone')
        t=t.tz_localize(tz)
    return t.tz_convert('UTC')


def step_end(t,cfg):
    if cfg['resolution']=='daily':return (t.tz_convert(cfg['timezone'])+pd.DateOffset(days=1)).tz_convert('UTC')
    if cfg['resolution']=='interval':return t+pd.Timedelta(minutes=cfg['interval_minutes'])
    raise DataError('Use daily or fixed interval resolution')


def frame(rows,cfg,energy=False):
    if not rows:raise DataError('No rows supplied')
    if cfg.get('resolution') not in ('daily','interval'):raise DataError('Unsupported resolution')
    if cfg['resolution']=='interval' and cfg.get('interval_minutes') not in (15,30,60):raise DataError('Intervals must be 15, 30 or 60 minutes')
    f=pd.DataFrame(rows).copy()
    key='timestamp' if 'timestamp' in f else 'date' if 'date' in f else None
    if not key:raise DataError('CSV requires date or timestamp')
    f['timestamp']=[stamp(x,cfg['timezone'],cfg['resolution']=='daily') for x in f[key]]
    if f.timestamp.duplicated().any():raise DataError('Duplicate periods; reconcile sources explicitly')
    f=f.sort_values('timestamp').reset_index(drop=True)
    for t in f.timestamp:
        if cfg['resolution']=='daily':
            if t.tz_convert(cfg['timezone']).time()!=pd.Timestamp('00:00').time():raise DataError('Daily rows must start at local midnight')
        elif t.value % pd.Timedelta(minutes=cfg['interval_minutes']).value:raise DataError('Interval starts must align to UTC boundaries')
    if 'meter_id' in f and (f.meter_id.astype(str)!=cfg['meter_id']).any():raise DataError('Meter boundary mismatch; analyze meters separately')
    f['period_end']=[step_end(t,cfg) for t in f.timestamp]
    f['hours']=(f.period_end-f.timestamp).dt.total_seconds()/3600
    f['energy_kwh']=pd.to_numeric(f.get('energy_kwh',pd.Series(np.nan,index=f.index)),errors='coerce')
    if (f.energy_kwh.dropna()<0).any():raise DataError('Negative import energy; counters/resets and exports require reconciliation')
    if energy and f.energy_kwh.notna().sum()<4:raise DataError('Too few valid meter observations')
    for name in ['temperature_c']+cfg.get('independent_variables',[]):
        f[name]=pd.to_numeric(f.get(name,pd.Series(np.nan,index=f.index)),errors='coerce')
    bad=f.temperature_c.notna() & ~f.temperature_c.between(-60,60)
    if bad.any():raise DataError('Outdoor temperature outside supported range')
    f['quality']=f.get('quality','good')
    f.loc[f.quality!='good','energy_kwh']=np.nan
    return f


def records(f):
    columns=[c for c in f if c not in ('period_end','hours')]
    return [{k:(v.isoformat() if isinstance(v,pd.Timestamp) else None if pd.isna(v) else v.item() if isinstance(v,np.generic) else v) for k,v in r.items()} for r in f[columns].to_dict('records')]


def window(f,cfg,start,end):
    a,b=stamp(start,cfg['timezone'],True),stamp(end,cfg['timezone'],True)
    if b<=a:raise DataError('End must be after start; end is exclusive')
    if cfg['resolution']=='daily':
        for t in (a,b):
            if t.tz_convert(cfg['timezone']).time()!=pd.Timestamp('00:00').time():raise DataError('Daily reporting boundaries must be local midnight')
        grid=pd.date_range(a.tz_convert(cfg['timezone']),b.tz_convert(cfg['timezone']),freq='D',inclusive='left').tz_convert('UTC')
    else:
        delta=pd.Timedelta(minutes=cfg['interval_minutes'])
        if a.value%delta.value or b.value%delta.value:raise DataError('Requested boundaries must align to intervals')
        grid=pd.date_range(a,b,freq=delta,inclusive='left')
    if len(grid)>40000:raise DataError('Requested window exceeds 40000 periods')
    result=f.set_index('timestamp').reindex(grid).reset_index(names='timestamp')
    result['period_end']=[step_end(t,cfg) for t in result.timestamp]
    result['hours']=(result.period_end-result.timestamp).dt.total_seconds()/3600
    return result,a,b


def check_cfg(cfg):
    if cfg.get('schema')!='option_c.config.v1':raise DataError('Unsupported config schema')
    if not cfg.get('meter_id') or not cfg.get('boundary_description'):raise DataError('Declare meter and measurement boundary')
    pd.Timestamp('2026-01-01',tz=cfg['timezone'])
    for name in cfg.get('independent_variables',[]):
        if any(x in name.lower() for x in ('setpoint','bms','fan','cool','heat','valve','runtime','stage')):
            raise DataError('Optimized control variables cannot enter the Option C baseline')
    stamp(cfg['intervention_timestamp'],cfg['timezone'])
    return cfg
