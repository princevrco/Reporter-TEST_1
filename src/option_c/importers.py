"""Adapters for the previously supplied Promenade daily exports; never sum boundaries."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import DataError


def yes(v):return str(v).strip().lower() in ('true','1','yes')


def discover(path,kind,carma_daily_kwh=False):
    if kind=='carma':
        from .carma import source
        return sorted(source(path,carma_daily_kwh)[0].meter_id.unique().tolist())
    if kind=='portal':
        f=pd.read_csv(path,dtype={'meter_code':str})
        if 'meter_code' not in f:raise DataError('Portal CSV requires meter_code')
        return sorted(f.meter_code.dropna().unique().tolist())
    if kind=='utility':
        from .utility import read
        return sorted(read(path).meter_id.unique().tolist())
    if kind=='normalized':
        f=pd.read_csv(path,dtype={'meter_id':str})
        if not {'date','meter_id','energy_kwh'}<=set(f):raise DataError('Normalized CSV requires date, meter_id, energy_kwh')
        return sorted(f.meter_id.dropna().unique().tolist())
    raise DataError('Use portal, utility, normalized or carma adapter')


def prepare(path,meter_id=None,weather=None,kind='portal',carma_daily_kwh=False):
    if kind=='utility':
        from .utility import read
        result=read(path)
        audit=result.attrs.get('source_audit')
        if meter_id is not None:result=result[result.meter_id==meter_id].copy()
        if result.empty:raise DataError('Selected meter not present in utility workbook')
        result.attrs['source_audit']=audit
        if weather:
            w=pd.read_csv(weather)
            if 'outdoor_temp_c' in w:w=w.rename(columns={'outdoor_temp_c':'temperature_c'})
            from .weather import join
            result=join(result,w)
            result.attrs['source_audit']=audit
        return result
    if meter_id is None and kind!='carma':
        ids=discover(path,kind,carma_daily_kwh)
        if not ids:raise DataError('No meters found')
        return pd.DataFrame([r for m in ids for r in prepare(path,m,weather,kind,carma_daily_kwh).to_dict('records')])
    path=Path(path);rows=[];source_audit=None
    if kind=='carma':
        from .carma import source
        f,source_audit=source(path,carma_daily_kwh)
        rows=(f if meter_id is None else f[f.meter_id==meter_id]).to_dict('records')
    elif kind=='portal':
        f=pd.read_csv(path,dtype={'meter_code':str})
        if not {'date_portal','meter_code','consumption_kwh'}<=set(f):raise DataError('Portal CSV schema mismatch')
        f=f[f.meter_code.astype(str)==meter_id]
        for r in f.to_dict('records'):
            value=pd.to_numeric(r['consumption_kwh'],errors='coerce');flags=[]
            if not np.isfinite(value) or value<0:flags.append('invalid_consumption')
            for k in ('reconciliation_review','reported_kwh_with_zero_demand'):
                if yes(r.get(k)):flags.append(k)
            flag=r.get('consumption_quality_flags')
            if isinstance(flag,str) and flag.strip():flags.append(flag)
            rows.append({'date':pd.Timestamp(r['date_portal']).date().isoformat(),'meter_id':meter_id,
                         'energy_kwh':float(value) if np.isfinite(value) and not flags else None,
                         'raw_energy_kwh':float(value) if np.isfinite(value) else None,'quality':'excluded' if flags else 'good','source_flags':';'.join(flags)})
    elif kind=='normalized':
        f=pd.read_csv(path,dtype={'meter_id':str})
        if not {'date','meter_id','energy_kwh'}<=set(f):raise DataError('Normalized CSV requires date, meter_id, energy_kwh')
        rows=f[f.meter_id==meter_id].to_dict('records')
    else:raise DataError('Use portal, utility, normalized or carma adapter')
    if not rows:raise DataError('Selected meter not present in source')
    f=pd.DataFrame(rows).sort_values('date')
    if f.duplicated(['date','meter_id']).any():raise DataError('Duplicate selected meter dates')
    if weather:
        w=pd.read_csv(weather,dtype={'date':str})
        if w.date.duplicated().any():raise DataError('Duplicate weather days')
        if 'outdoor_temp_c' in w:w=w.rename(columns={'outdoor_temp_c':'temperature_c'})
        if not {'date','temperature_c'}<=set(w):raise DataError('Weather CSV requires date and temperature_c')
        f=f.merge(w[['date','temperature_c']],on='date',how='left',validate='many_to_one')
    if source_audit:f.attrs['source_audit']=source_audit
    return f
