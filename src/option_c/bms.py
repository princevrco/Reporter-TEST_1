from __future__ import annotations
import numpy as np
import pandas as pd
from .data import DataError,digest


def aggregate(paths,cfg):
    """Validated sample means, never fabricated equipment runtime or electrical kW."""
    if cfg.get('review',{}).get('bms_semantics_verified') is not True:
        raise DataError('BMS point meanings must be verified before aggregation')
    specs=cfg.get('bms_features',{})
    if not specs:raise DataError('Declare BMS columns and their meanings')
    parts=[]
    for p in paths:
        f=pd.read_csv(p)
        if 'timestamp' not in f:raise DataError('BMS CSV needs timestamp')
        for x in f.timestamp:
            if pd.Timestamp(x).tzinfo is None:raise DataError('BMS timestamps need explicit offsets')
        f['timestamp']=pd.to_datetime(f.timestamp,utc=True)
        parts.append(f)
    f=pd.concat(parts,ignore_index=True).sort_values('timestamp',kind='stable')
    # Overlapping history/update exports: last non-null cell wins, with conflicts audited.
    conflicts=int(sum((g.drop(columns='timestamp').nunique(dropna=True)>1).sum() for _,g in f.groupby('timestamp')))
    f=f.groupby('timestamp',sort=True).last()
    local=f.index.tz_convert(cfg['timezone']);output={};audit={}
    for name,spec in specs.items():
        col=spec['column']
        if col not in f:raise DataError(f'BMS mapped column missing: {col}')
        raw=pd.to_numeric(f[col],errors='coerce')
        good=raw.notna() & np.isfinite(raw)
        if 'allowed_values' in spec:good &= raw.isin(spec['allowed_values'])
        if 'valid_range' in spec:good &= raw.between(*spec['valid_range'])
        cleaned=(raw.where(good)*spec.get('scale',1)).copy();cleaned.index=local
        resolution=cfg['resolution'];freq='1h' if resolution=='daily' else f"{cfg['interval_minutes']}min"
        grouped=cleaned.resample(freq)
        samples=grouped.count();means=grouped.mean().where(samples>=spec.get('minimum_samples_per_bin',1))
        if resolution=='daily':
            valid=means.resample('D').count();mean=means.resample('D').mean()
            expected=pd.Series([(d+pd.DateOffset(days=1)-d).total_seconds()/3600 for d in mean.index],index=mean.index)
            coverage=valid/expected
            result=mean.where(coverage>=spec.get('minimum_hour_coverage',.9))
        else:
            coverage=means.notna().astype(float);result=means
        output[name]=result
        audit[name]={'source_column':col,'valid_samples':int(good.sum()),'rejected_nonmissing_samples':int((raw.notna() & ~good).sum()),
                     'periods_with_feature':int(result.notna().sum()),'aggregation':'Equal-weight mean of populated time-bin sample means; NOT measured interval runtime or energy',
                     'minimum_coverage':spec.get('minimum_hour_coverage',.9)}
    result=pd.DataFrame(output)
    result.index=result.index.tz_convert('UTC');result.index.name='timestamp'
    result=result.reset_index();result['meter_id']=cfg['meter_id']
    return result,{'point_audit':audit,'conflicting_overlap_cells':conflicts,'overlap_policy':'Latest supplied non-null value; conflicts reported for reconciliation','feature_definition_sha256':digest(specs),'source_files':list(map(str,paths))}


def join_features(periods,bms_rows,cfg):
    f=pd.DataFrame(bms_rows).copy()
    if 'timestamp' not in f:raise DataError('BMS feature data needs timestamp')
    f['timestamp']=pd.to_datetime(f.timestamp,utc=True)
    if f.timestamp.duplicated().any():raise DataError('Duplicate BMS feature periods')
    if 'meter_id' in f and (f.meter_id!=cfg['meter_id']).any():raise DataError('BMS feature boundary mismatch')
    columns=list(cfg.get('bms_features',{}))
    if any(c not in f for c in columns):raise DataError('BMS features do not match config')
    return periods.merge(f[['timestamp']+columns],on='timestamp',how='left',validate='one_to_one')
