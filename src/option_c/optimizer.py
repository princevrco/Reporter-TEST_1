"""Uncertain optimizer thermal modes, time-bin duration estimates and equipment groups."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .data import DataError,digest,records,stamp


MODES={'-1':'cooling','0':'off','1':'heating'}


def run_statistics(g,minutes):
    """Contiguous observed ON slots. Gaps censor runs rather than implying OFF."""
    delta=pd.Timedelta(minutes=minutes);runs=[];start=None;last=None;left=True;previous=None;previous_off=False
    def close(right):
        nonlocal start,last
        if start is not None:
            runs.append({'start':start.isoformat(),'end_exclusive':(last+delta).isoformat(),
                         'duration_hours':(last+delta-start).total_seconds()/3600,
                         'left_censored':left,'right_censored':right})
        start=last=None
    for t,row in g.iterrows():
        contiguous=previous is not None and t-previous==delta
        if not contiguous:close(True)
        if bool(row.eligible) and bool(row.on):
            if start is None:start=t;left=not(contiguous and previous_off)
            last=t
        else:close(not (contiguous and bool(row.eligible)))
        previous=t;previous_off=bool(row.eligible) and not bool(row.on)
    close(True)
    completed=[r['duration_hours'] for r in runs if not r['left_censored'] and not r['right_censored']]
    durations=[r['duration_hours'] for r in runs]
    return {'observed_on_runs':len(runs),'uncensored_on_runs':len(completed),
            'runs':runs,
            'mean_uncensored_run_hours':float(np.mean(completed)) if completed else None,
            'median_uncensored_run_hours':float(np.median(completed)) if completed else None,
            'observed_run_duration_histogram':{str(v):durations.count(v) for v in sorted(set(durations))},
            'note':'Input-window thermal-mode runs; gaps and window edges censor durations. Not measured equipment runtime.'}


def _number(value):
    try:return float(value)
    except (ValueError,TypeError):return np.nan


def _boolean(value):
    if value is True or str(value).lower() in ('true','1','1.0'):return True
    if value is False or str(value).lower() in ('false','0','0.0'):return False
    return None


def _json_rows(payload):
    if isinstance(payload,list):return payload
    if 'records' in payload:return payload['records']
    if 'zones' not in payload:raise DataError('Optimizer JSON needs records or Mapper zones with explicit issued_at and horizon_start')
    meta=payload.get('metrics',{})
    if not payload.get('issued_at') or not payload.get('horizon_start'):
        raise DataError('Mapper trajectories need explicit issued_at and horizon_start; state_timestamp is not an issue time')
    seconds=_number(payload.get('cadence_seconds',meta.get('cadence_seconds')))
    if not np.isfinite(seconds) or seconds<=0:raise DataError('Mapper JSON requires positive cadence_seconds')
    start=stamp(payload['horizon_start'],'UTC');rows=[]
    head=meta.get('control_horizon',0);extrapolated=meta.get('steps_extrapolated',[])
    for zone,item in payload['zones'].items():
        for i,mode in enumerate(item.get('achieved_modes',[])):
            rows.append({'timestamp':(start+pd.Timedelta(seconds=i*seconds)).isoformat(),
                         'issued_at':payload['issued_at'],'zone_id':str(zone),'achieved_mode':mode,
                         'within_control_horizon':i<int(head),'outside_support':i in extrapolated,
                         'source_cadence_minutes':seconds/60,
                         'applied':None,'source_kind':'optimizer_prediction'})
    return rows


def load(paths,cfg):
    settings=cfg.get('optimizer',{});parts=[];lineage=[]
    if isinstance(paths,(str,Path)):paths=[paths]
    for p in paths:
        p=Path(p)
        if p.suffix.lower()=='.csv':f=pd.read_csv(p,dtype={'zone_id':str})
        elif p.suffix.lower()=='.json':f=pd.DataFrame(_json_rows(json.loads(p.read_text(encoding='utf-8-sig'))))
        else:raise DataError('Optimizer inputs must be CSV or JSON files')
        f=f.rename(columns={v:k for k,v in settings.get('columns',{}).items()})
        if not {'timestamp','zone_id','achieved_mode'}<=set(f):raise DataError('Optimizer rows require timestamp, zone_id, achieved_mode')
        if f.zone_id.isna().any():raise DataError('Optimizer zone_id is missing')
        f.zone_id=f.zone_id.astype(str)
        f['timestamp']=[stamp(t,cfg['timezone']) for t in f.timestamp]
        f['issued_at']=[stamp(t,cfg['timezone']) if pd.notna(t) and str(t).strip() else pd.NaT
                        for t in f.get('issued_at',pd.Series(None,index=f.index,dtype=object))]
        f['issued_at']=pd.to_datetime(f.issued_at,utc=True)
        parts.append(f);lineage.append({'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    if not parts:raise DataError('No optimizer files supplied')
    f=pd.concat(parts,ignore_index=True);future=f.issued_at.notna()&(f.issued_at>f.timestamp)
    rejected=int(future.sum());f=f[~future].copy()
    if f.empty:raise DataError('No optimizer rows issued at or before their valid time')
    f['mode_number']=pd.to_numeric(f.achieved_mode,errors='coerce')
    mapping=settings.get('mode_map',MODES)
    if any(v not in ('cooling','heating','off') for v in mapping.values()):raise DataError('Optimizer mode_map values must be cooling, heating or off')
    f['mode']=f.mode_number.map(lambda v:mapping.get(str(int(v)),'unknown') if np.isfinite(v) and v==int(v) else 'unknown')
    f['_issued_sort']=f.issued_at.fillna(pd.Timestamp('1970-01-01',tz='UTC'))
    conflicts=0
    for _,g in f.groupby(['zone_id','timestamp','_issued_sort']):
        if g.mode_number.nunique(dropna=False)>1:conflicts+=1
    if conflicts:raise DataError('Conflicting optimizer modes for the same zone, valid time and issue time')
    f=f.sort_values(['timestamp','_issued_sort'],kind='stable')
    overlap=len(f)-len(f.drop_duplicates(['timestamp','zone_id']))
    f=f.drop_duplicates(['timestamp','zone_id'],keep='last').drop(columns='_issued_sort')
    return f,{'files':lineage,'sha256':digest(lineage),'overlapping_forecasts_resolved':overlap,
              'future_issued_rows_rejected':rejected,'overlap_policy':'Latest issue at or before valid time, per zone',
              'mode_map':mapping,'unknown_mode_rows':int((f['mode']=='unknown').sum())}


def prepare(paths,bms_path,cfg):
    settings=cfg.get('optimizer',{});f,audit=load(paths,cfg)
    minutes=settings.get('bin_minutes',15)
    if minutes not in (5,10,15,20,30,60):raise DataError('Optimizer bin_minutes must divide one hour')
    bin_delta=pd.Timedelta(minutes=minutes);f['bin']=f.timestamp.dt.floor(f'{minutes}min')
    if 'source_cadence_minutes' in f and (f.source_cadence_minutes.dropna()!=minutes).any():
        raise DataError('Set optimizer.bin_minutes to match the Mapper cadence_seconds; durations cannot be guessed')
    max_lead=float(settings.get('maximum_lead_minutes',60));age=float(settings.get('bms_max_age_minutes',5))
    if not np.isfinite([max_lead,age]).all() or min(max_lead,age)<0:raise DataError('Invalid optimizer lead/BMS age limits')
    # A slot is an estimated mode observation, not proof that a command was applied.
    lead=(f.timestamp-f.issued_at).dt.total_seconds()/60
    f['eligible']=(f['mode']!='unknown')&lead.between(0,max_lead)
    if 'within_control_horizon' in f:f['eligible'] &= f.within_control_horizon.map(_boolean)!=False
    if 'outside_support' in f:f['eligible'] &= f.outside_support.map(_boolean)!=True
    applied=f.get('applied',pd.Series(None,index=f.index,dtype=object)).map(_boolean)
    f['eligible'] &= applied!=False
    f['applied_confirmed']=applied==True
    raw=pd.read_csv(bms_path)
    if 'timestamp' not in raw:raise DataError('BMS CSV needs timestamp')
    raw['timestamp']=[stamp(t,cfg['timezone']) for t in raw.timestamp]
    raw=raw.sort_values('timestamp',kind='stable').groupby('timestamp',as_index=False).last()
    zones=settings.get('zones',{});daily=[];bin_frames={};groups={};run_stats=[]
    for zone,g in f.groupby('zone_id',sort=True):
        spec=zones.get(zone,{});equipment=str(spec.get('equipment_id') or zone)
        meters=spec.get('meter_ids',[])
        if not isinstance(meters,list) or any(not isinstance(m,str) for m in meters):raise DataError('Optimizer meter_ids must be a list of meter identifiers')
        group=groups.setdefault(equipment,{'equipment_id':equipment,'zone_ids':[],'meter_ids':meters,
                                         'physical_mapping_verified':bool(spec.get('mapping_verified',False))})
        if set(group['meter_ids'])!=set(meters):raise DataError('Zones sharing equipment must declare the same meter_ids')
        group['physical_mapping_verified'] &= bool(spec.get('mapping_verified',False));group['zone_ids'].append(zone)
        g=g.sort_values('timestamp').copy()
        for name in ('temperature_c','heating_setpoint_c','cooling_setpoint_c'):
            column=spec.get('bms_columns',{}).get(name)
            g[name]=np.nan
            if column:
                if column not in raw:raise DataError('Mapped BMS measurement column missing: '+column)
                measured=raw[['timestamp',column]].copy();measured[column]=pd.to_numeric(measured[column],errors='coerce')
                measured=measured[np.isfinite(measured[column])]
                joined=pd.merge_asof(g[['timestamp']],measured,on='timestamp',direction='backward',tolerance=pd.Timedelta(minutes=age))
                g[name]=joined[column].to_numpy()
        margin=_number(spec.get('hysteresis_margin_c'))
        consistent=pd.Series(False,index=g.index)
        if np.isfinite(margin) and margin>=0:
            t,h,c=g.temperature_c,g.heating_setpoint_c,g.cooling_setpoint_c
            valid=(h<=c)&t.between(-60,70)&h.between(-60,70)&c.between(-60,70)
            consistent=valid&(((g['mode']=='heating')&(t<h-margin))|
                              ((g['mode']=='cooling')&(t>c+margin))|
                              ((g['mode']=='off')&(t>h+margin)&(t<c-margin)))
        # Threshold consistency is a sensitivity screen, not a probability or equipment feedback.
        g['uncertain']=~(consistent&g.applied_confirmed)
        g=g.sort_values(['bin','timestamp']).drop_duplicates('bin',keep='last').set_index('bin')
        g['heating']=(g['mode']=='heating').astype(float);g['cooling']=(g['mode']=='cooling').astype(float)
        g['on']=(g['mode'].isin(['heating','cooling'])).astype(float)
        bin_frames[zone]=g
        run_stats.append({'zone_id':zone,**run_statistics(g,minutes)})
        for day,d in g.groupby(g.index.tz_convert(cfg['timezone']).date):
            start=pd.Timestamp(day,tz=cfg['timezone']);hours=(start+pd.DateOffset(days=1)-start).total_seconds()/3600
            eligible=d[d.eligible];known_hours=len(eligible)*minutes/60;on=float(eligible.on.sum()*minutes/60)
            uncertain=float(eligible.uncertain.sum()*minutes/60)
            lower=float(eligible.loc[~eligible.uncertain,'on'].sum()*minutes/60)
            upper=lower+uncertain
            daily.append({'date':str(day),'zone_id':zone,'equipment_id':equipment,'period_hours':hours,
                          'eligible_mode_hours':known_hours,'coverage_fraction':known_hours/hours,
                          'on_hours_estimated':on,'off_hours_estimated':known_hours-on,
                          'heating_hours_estimated':float(eligible.heating.sum()*minutes/60),
                          'cooling_hours_estimated':float(eligible.cooling.sum()*minutes/60),
                          'uncertain_hours':uncertain,'on_hours_lower_sensitivity':lower,'on_hours_upper_sensitivity':upper,
                          'unknown_or_missing_hours':max(0,hours-known_hours),
                          'on_duration_ratio_to_one_hour':on,
                          'interpretation':'Estimated thermal-mode slot duration; OFF does not establish zero fan/electrical load'})
    zone_daily=pd.DataFrame(daily)
    if not zone_daily.empty:
        total=zone_daily.groupby('date').on_hours_estimated.transform('sum')
        zone_daily['share_of_zone_on_time_pct']=np.where(total>0,zone_daily.on_hours_estimated/total*100,np.nan)
    features=[];coverage=float(settings.get('minimum_daily_coverage',.9))
    if not 0<coverage<=1:raise DataError('minimum_daily_coverage must be in (0,1]')
    for equipment,spec in groups.items():
        joined=pd.concat({z:bin_frames[z][['eligible','heating','cooling','uncertain']] for z in spec['zone_ids']},axis=1)
        eligible=joined.xs('eligible',level=1,axis=1).fillna(False).all(axis=1)
        heat=joined.xs('heating',level=1,axis=1).max(axis=1);cool=joined.xs('cooling',level=1,axis=1).max(axis=1)
        uncertain=joined.xs('uncertain',level=1,axis=1).fillna(True).any(axis=1)|(heat.gt(0)&cool.gt(0))
        for day,idx in pd.Series(range(len(joined)),index=joined.index).groupby(joined.index.tz_convert(cfg['timezone']).date):
            indices=idx.to_numpy();ok=eligible.iloc[indices];positions=indices[ok.to_numpy()]
            start=pd.Timestamp(day,tz=cfg['timezone']);hours=(start+pd.DateOffset(days=1)-start).total_seconds()/3600
            expected=hours*60/minutes;fraction=len(positions)/expected;available=fraction>=coverage
            features.append({'date':str(day),'equipment_id':equipment,'coverage_fraction':fraction,
                             'heating_fraction':float(heat.iloc[positions].mean()) if available else None,
                             'cooling_fraction':float(cool.iloc[positions].mean()) if available else None,
                             'uncertain_fraction':float(uncertain.iloc[positions].mean()) if available else None,
                             'source':'Optimizer proxy; shared-zone modes combined by union, never summed'})
    audit.update(bin_minutes=minutes,maximum_lead_minutes=max_lead,bms_max_age_minutes=age,
                 rows_offset_from_bin_start=int((f.timestamp!=f.bin).sum()),
                 zone_count=len(bin_frames),equipment_group_count=len(groups),
                 missing_issue_time_rows=int(f.issued_at.isna().sum()),
                 zone_on_run_statistics=run_stats,
                 note='Mode signs confirmed by user. No observed-temperature inference from predicted_state. No hold beyond occupied bins. Sensitivity bounds are not confidence intervals.')
    return zone_daily,pd.DataFrame(features),list(groups.values()),audit


def summaries(zone_daily,start,end,timezone='America/Toronto',run_stats=None):
    """Duration shares are descriptive; they never stand in for meter impact percentages."""
    f=zone_daily[(zone_daily.date>=start)&(zone_daily.date<end)].copy()
    if f.empty:return []
    a=pd.Timestamp(start,tz=timezone);b=pd.Timestamp(end,tz=timezone)
    expected_hours=(b.tz_convert('UTC')-a.tz_convert('UTC')).total_seconds()/3600
    expected_days=len(pd.date_range(a,b,freq='D',inclusive='left'))
    output=[]
    for zone,g in f.groupby('zone_id'):
        on=float(g.on_hours_estimated.sum());covered=float(g.eligible_mode_hours.sum());hours=expected_hours
        output.append({'zone_id':zone,'equipment_id':g.equipment_id.iloc[0],
                       'on_hours_estimated':on,'off_hours_estimated':covered-on,
                       'on_duration_ratio_to_one_hour':on,'mean_daily_on_hours_estimated':on/expected_days,
                       'unknown_or_missing_hours':max(0,hours-covered),
                       'uncertain_hours':float(g.uncertain_hours.sum()),'coverage_fraction':covered/hours,
                       'on_hours_lower_sensitivity':float(g.on_hours_lower_sensitivity.sum()),
                       'on_hours_upper_sensitivity':float(g.on_hours_upper_sensitivity.sum())})
    total=sum(r['on_hours_estimated'] for r in output)
    for row in output:row['share_of_zone_on_time_pct']=100*row['on_hours_estimated']/total if total>0 else None
    stats={r['zone_id']:r for r in (run_stats or [])}
    for row in output:
        complete=[];observed=[];censored=0
        for run in stats.get(row['zone_id'],{}).get('runs',[]):
            left=pd.Timestamp(run['start']);right=pd.Timestamp(run['end_exclusive'])
            if right<=a or left>=b:continue
            duration=(min(right,b)-max(left,a)).total_seconds()/3600;observed.append(duration)
            if run['left_censored'] or run['right_censored'] or left<a or right>b:censored+=1
            else:complete.append(duration)
        row.update(uncensored_on_runs=len(complete),censored_on_runs=censored,
                   mean_uncensored_on_run_hours=float(np.mean(complete)) if complete else None,
                   mean_on_run_ratio_to_one_hour=float(np.mean(complete)) if complete else None,
                   observed_run_duration_histogram={str(v):observed.count(v) for v in sorted(set(observed))})
    return output
