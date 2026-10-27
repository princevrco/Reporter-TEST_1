"""Complete-row BMS handling. No forward fill, interpolation or missing-as-OFF."""
import numpy as np
import pandas as pd
from .data import DataError,digest,records
from .response import discover


def clean(raw, required_columns=None):
    if 'timestamp' not in raw:raise DataError('BMS history requires timestamp')
    if raw.columns.duplicated().any():raise DataError('Duplicate BMS column names')
    required=list(raw.columns) if required_columns is None else ['timestamp']+list(required_columns)
    required=list(dict.fromkeys(required))
    if not set(required)<=set(raw):raise DataError('Required BMS columns missing from file')
    present=raw[required].notna() & raw[required].astype(str).apply(lambda c:c.str.strip().ne(''))
    present &= ~raw[required].isin([np.inf,-np.inf])
    valid=present.all(axis=1)
    parsed=[];bad_time=[]
    for i,t in enumerate(raw.timestamp):
        try:
            v=pd.Timestamp(t)
            if pd.isna(v) or v.tzinfo is None:raise ValueError()
            parsed.append(v.tz_convert('UTC'));bad_time.append(False)
        except (ValueError,TypeError):parsed.append(pd.NaT);bad_time.append(True)
    valid &= ~np.asarray(bad_time)
    kept=raw.loc[valid].copy();kept['timestamp']=pd.to_datetime(np.asarray(parsed,dtype=object)[valid.to_numpy()],utc=True)
    if kept.timestamp.duplicated().any():raise DataError('Duplicate complete BMS timestamps; reconcile explicitly')
    kept=kept.sort_values('timestamp').reset_index(drop=True)
    rejected=pd.DataFrame({'source_row':np.flatnonzero(~valid.to_numpy())+2,'reason':np.where(np.asarray(bad_time)[~valid.to_numpy()],'invalid_or_missing_timestamp','missing_or_empty_required_value')})
    audit={'input_rows':len(raw),'accepted_rows':len(kept),'rejected_rows':len(rejected),
           'required_columns':required,'missing_by_column':{c:int((~present[c]).sum()) for c in required},
           'rule':'Any empty required field rejects the entire original row. Default requires every source column.',
           'imputation':False,'accepted_sha256':digest(records(kept)) if len(kept) else None}
    return kept,rejected,audit


def mapping(raw,config=None):
    groups=(config or {}).get('groups') if config is not None else None
    if groups is None:groups=discover(raw)['groups']
    groups={a:{k:v for k,v in g.items() if k in ('temperature','cool_setpoint','heat_setpoint')}
            for a,g in groups.items() if {'temperature','cool_setpoint'}<=set(g)}
    if not groups:raise DataError('No assets with mapped temperature and cooling setpoint')
    used=[]
    for a,g in groups.items():
        for col in g.values():
            if col not in raw:raise DataError('Mapped BMS signal missing: '+col)
            if col in used:raise DataError('Shared BMS point assigned twice; consolidate shared equipment: '+col)
            used.append(col)
    return groups


def valid_signals(raw,groups):
    f=raw.copy();good=pd.Series(True,index=f.index)
    for g in groups.values():
        for c in g.values():
            f[c]=pd.to_numeric(f[c],errors='coerce');good &= np.isfinite(f[c])&f[c].between(-20,60)
    return f[good].copy(),{'rows_before_signal_validation':len(f),'invalid_signal_rows_rejected':int((~good).sum())}


def daily(raw,groups,weather,timezone,cadence_minutes=15,minimum_coverage=1.):
    if cadence_minutes not in (5,10,15,20,30,60) or not 0<minimum_coverage<=1:raise DataError('Invalid cadence or daily coverage')
    if raw.empty:return pd.DataFrame(),[]
    f=raw.copy();f['timestamp']=pd.to_datetime(f.timestamp,utc=True,format='mixed')
    step=pd.Timedelta(minutes=cadence_minutes)
    if (f.timestamp.astype('int64')%step.value!=0).any():raise DataError('BMS timestamps must align to the declared cadence; no resampling is performed')
    local=f.timestamp.dt.tz_convert(timezone);f['date']=local.dt.date.astype(str)
    # Grid-aligned slots divide local days for supported site timezones/cadences.
    if any((t+step).tz_convert(timezone).date()!=t.tz_convert(timezone).date() and (t+step).tz_convert(timezone).time()!=pd.Timestamp('00:00').time() for t in f.timestamp):
        raise DataError('BMS slots cross local midnight off-grid')
    w=weather.set_index('date').temperature_c;out=[];coverage=[]
    for day,g in f.groupby('date',sort=True):
        left=pd.Timestamp(day,tz=timezone);right=left+pd.DateOffset(days=1);hours=(right.tz_convert('UTC')-left.tz_convert('UTC')).total_seconds()/3600
        known=len(g)*cadence_minutes/60
        if known>hours+1e-8:raise DataError('Duplicate or overlapping BMS slots')
        fraction=known/hours;eligible=fraction>=minimum_coverage-1e-10
        coverage.append({'date':day,'accepted_slot_hours':known,'day_hours':hours,'coverage_fraction':fraction,'eligible':eligible})
        if not eligible:continue
        if day not in w or not np.isfinite(w[day]):raise DataError('Weather missing for BMS day '+day)
        cool=heat=con=hon=0.
        for spec in groups.values():
            t=g[spec['temperature']].to_numpy(float);sp=g[spec['cool_setpoint']].to_numpy(float)
            c=np.maximum(t-sp,0);h=np.maximum(g[spec['heat_setpoint']].to_numpy(float)-t,0) if 'heat_setpoint' in spec else np.zeros(len(g))
            cool+=c.sum()*cadence_minutes/60;heat+=h.sum()*cadence_minutes/60
            con+=(c>0).sum()*cadence_minutes/60;hon+=(h>0).sum()*cadence_minutes/60
        out.append({'date':day,'hours':hours,'temperature_c':float(w[day]),'cool_degree_hours':cool,
                    'heat_degree_hours':heat,'cool_proxy_hours':con,'heat_proxy_hours':hon,
                    'asset_count':len(groups),'coverage_fraction':fraction})
    return pd.DataFrame(out),coverage
