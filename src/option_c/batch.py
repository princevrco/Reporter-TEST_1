"""All-meter daily workflow. Meter readings determine consumption; BMS is context."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pandas as pd
from .data import DataError,records,save,stamp,step_end
from .importers import prepare,discover
from .regression import fit
from .workflow import forecast,savings
from .reporting import write
from .optimizer import prepare as prepare_optimizer,summaries as optimizer_summaries
from .effects import analyze as analyze_effects


def bms_context(path,timezone):
    path=Path(path);f=pd.read_csv(path)
    if 'timestamp' not in f or f.empty:raise DataError('Live BMS CSV needs timestamp and data rows')
    if any(pd.isna(t) or pd.Timestamp(t).tzinfo is None for t in f.timestamp):
        raise DataError('Live BMS timestamps require explicit offsets')
    f['timestamp']=pd.to_datetime(f.timestamp,utc=True,format='mixed')
    unique=f.drop_duplicates('timestamp').copy()
    unique['date']=unique.timestamp.dt.tz_convert(timezone).dt.date.astype(str)
    context={day:{'bms_samples':len(g),'bms_hours_with_samples':int(g.timestamp.dt.floor('h').nunique())}
             for day,g in unique.groupby('date')}
    audit={'source_filename':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'rows':len(f),'unique_timestamps':len(unique),'duplicate_timestamp_rows':len(f)-len(unique),
           'columns':list(f.columns),'first_timestamp':f.timestamp.min().isoformat(),
           'last_timestamp':f.timestamp.max().isoformat(),
           'role':'Timestamp/sample coverage only; presence is not asset operation, validated point semantics, runtime or kWh'}
    return context,audit


def run(meter_path,bms_path,cfg,start,end,output,kind='portal',carma_daily_kwh=False,optimizer_paths=None,progress=None):
    if cfg.get('resolution')!='daily':raise DataError('Raw all-meter workflow currently supports daily kWh only')
    if cfg.get('independent_variables'):raise DataError('Two-file mode does not accept separate independent variables; use the full workflow')
    a,b=stamp(start,cfg['timezone'],True),stamp(end,cfg['timezone'],True)
    cutoff=stamp(cfg['intervention_timestamp'],cfg['timezone'])
    baseline_end=stamp(cfg.get('baseline_end_exclusive',cfg['intervention_timestamp']),cfg['timezone'])
    if baseline_end>cutoff:raise DataError('Baseline end cannot be after intervention')
    if b<=a or a<cutoff:raise DataError('Choose a reporting window after intervention, with end after start')
    if (b-a).total_seconds()/86400>40000:raise DataError('Requested window exceeds 40000 days')
    if any(t.tz_convert(cfg['timezone']).time()!=pd.Timestamp('00:00').time() for t in (a,b)):
        raise DataError('Daily boundaries must be local midnight')
    context,audit=bms_context(bms_path,cfg['timezone'])
    carma=prepare(meter_path,kind=kind,carma_daily_kwh=carma_daily_kwh) if kind=='carma' else None
    normalized=pd.read_csv(meter_path,dtype={'meter_id':str}) if kind=='normalized' else None
    if normalized is not None and not {'date','meter_id','energy_kwh'}<=set(normalized):raise DataError('Normalized meter data needs date, meter_id, energy_kwh')
    meters=sorted(carma.meter_id.unique()) if carma is not None else sorted(normalized.meter_id.unique()) if normalized is not None else discover(meter_path,kind)
    if not meters:raise DataError('No meters found in source')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    save(out/'bms-audit.json',audit)
    source_audit=carma.attrs.get('source_audit') if carma is not None else None
    if source_audit:save(out/'meter-import-audit.json',source_audit)
    zone_daily=features=groups=optimizer_audit=None;profiles=[];zone_effect_rows=[]
    first=a.tz_convert(cfg['timezone']).date().isoformat();last=b.tz_convert(cfg['timezone']).date().isoformat()
    if optimizer_paths:
        zone_daily,features,groups,optimizer_audit=prepare_optimizer(optimizer_paths,bms_path,cfg)
        zone_daily.to_csv(out/'zone-daily.csv',index=False);features.to_csv(out/'equipment-daily.csv',index=False)
        save(out/'optimizer-audit.json',optimizer_audit);save(out/'equipment-groups.json',groups)
        profiles=optimizer_summaries(zone_daily,first,last,cfg['timezone'],optimizer_audit['zone_on_run_statistics'])
        run_rows=[{'zone_id':zone['zone_id'],**run} for zone in optimizer_audit['zone_on_run_statistics'] for run in zone['runs']]
        pd.DataFrame(run_rows).to_csv(out/'zone-on-runs.csv',index=False)
        save(out/'zone-duration-summary.json',{'zones':profiles,'scope':'Estimated thermal-mode durations; duration shares are not meter energy shares'})
    summary={'schema':'option_c.all_meters.v1','selection':'all available meters, analyzed separately',
             'meter_source_sha256':source_audit['sha256'] if source_audit else hashlib.sha256(Path(meter_path).read_bytes()).hexdigest(),
             'source_kind':kind,'carma_daily_kwh_operator_confirmed':carma_daily_kwh if kind=='carma' else None,
             'start':start,'end_exclusive':end,'meters':[],
             'baseline_end_exclusive':baseline_end.isoformat(),
             'intervention_timestamp':cfg['intervention_timestamp'],
             'intervention_confirmed':cfg.get('review',{}).get('intervention_confirmed',False),
             'optimizer_supplied':bool(optimizer_paths),'zone_duration_summary':profiles,
             'site_total_kwh':None,'site_total_note':'Not summed: main meters and submeters may overlap; a verified boundary map is required',
             'method':'Historical meter calendar regression; no separate equipment calibration or weather file. BMS is coverage context only.',
             'limitations':['Weather and other non-routine changes are not adjusted in this two-file mode',
                            'Differences cannot be attributed solely to BMS controls',
                            'Missing reporting meter readings remain unavailable; no BMS-to-kWh conversion']}
    if optimizer_paths:
        summary['method']='Frozen historical meter baseline plus a separate optimizer-to-meter response model trained on ordinary synchronized history'
        summary['limitations']=['Weather and other unmeasured changes are not adjusted in this workflow',
            'Optimizer modes and threshold consistency do not prove physical equipment operation',
            'Zone-meter percentages are statistical associations; physical mappings and causal effects require review',
            'Operation predictions remain separate from actual reporting consumption and measured savings']
    for meter in meters:
        meter=str(meter);folder=out/('meter-'+hashlib.sha256(meter.encode()).hexdigest()[:16])
        folder.mkdir(exist_ok=True)
        entry={'meter_id':meter,'folder':folder.name}
        try:
            part=carma[carma.meter_id==meter].copy() if carma is not None else normalized[normalized.meter_id==meter].copy() if normalized is not None else prepare(meter_path,meter,kind=kind)
            if part.duplicated(['date','meter_id']).any():raise DataError('Duplicate selected meter dates')
            if 'meter_description' in part and 'is_virtual_meter' in part:
                entry.update(meter_description=part.meter_description.iloc[0],is_virtual_meter=bool(part.is_virtual_meter.iloc[0]))
            c=deepcopy(cfg);c['meter_id']=meter;c['allow_meter_only_baseline']=True
            c['boundary_description']=f'Raw source meter {meter}; physical boundary requires verification'
            c.setdefault('review',{})['meter_boundary_verified']=False
            # Deliberately use calendar-only regression consistently in two-file mode.
            part=part.drop(columns=['temperature_c'],errors='ignore').copy()
            times=[stamp(x,c['timezone'],True) for x in part.date]
            history=part.loc[[step_end(t,c)<=baseline_end for t in times]]
            model=fit(records(history),c);save(folder/'baseline-model.json',model)
            grid=pd.date_range(a.tz_convert(c['timezone']),b.tz_convert(c['timezone']),freq='D',inclusive='left')
            periods=pd.DataFrame({'date':grid.date.astype(str),'meter_id':meter})
            periods=periods.merge(part,on=['date','meter_id'],how='left',validate='one_to_one')
            rows=records(periods)
            projection=forecast(model,rows,start,end)
            write(projection,folder/'forecast')
            result=savings(model,rows,start,end)
            if optimizer_paths:
                effects=analyze_effects(records(part),features,groups,c,first,last)
                save(folder/'zone-meter-effects.json',effects)
                if effects.get('model'):save(folder/'operation-response-model.json',effects['model'])
                result['zone_meter_effects']={k:v for k,v in effects.items() if k not in ('model','operation_predictions')}
                selected_zones={z for group in effects['groups'] for z in group['zone_ids']}
                result['optimizer_duration_summary']=[p for p in profiles if p['zone_id'] in selected_zones]
                zone_effect_rows.extend(effects['zone_effects'])
                predictions={p['timestamp']:p['operation_model_kwh'] for p in effects.get('operation_predictions',[])}
                for row in result['periods']:row['operation_model_kwh']=predictions.get(row['timestamp'])
                entry['optimizer_response_model_status']=effects['status']
                result['gates'].append('optimizer_modes_are_uncertain_proxies_not_equipment_feedback')
            for row in result['periods']:
                day=pd.Timestamp(row['timestamp']).tz_convert(c['timezone']).date().isoformat()
                row.update(context.get(day,{'bms_samples':0,'bms_hours_with_samples':0}))
            result['gates']=sorted(set(result['gates']+['calendar_only_unadjusted_comparison']+
                ([] if optimizer_paths else ['bms_context_only_no_energy_inference'])))
            write(result,folder/'savings')
            entry.update(status=result['status'],valid_baseline_periods=model['valid_periods'],
                         holdout_metrics=model['holdout_metrics'],gates=result['gates'],
                         forecast_kwh=projection['baseline_kwh'],
                         observed_meter_periods=result['observed_meter_periods'],
                         unavailable_periods=result['unavailable_periods'],
                         savings_kwh_complete_period=result['savings_kwh_complete_period'],
                         savings_kwh_paired=result['savings_kwh_paired'])
        except (DataError,ValueError,KeyError) as exc:
            entry.update(status='blocked',reason=str(exc))
            if optimizer_paths:
                for group in groups:
                    if not group['meter_ids'] or meter in group['meter_ids']:
                        for zone in group['zone_ids']:
                            zone_effect_rows.append({'zone_id':zone,'meter_id':meter,'group_id':group['equipment_id'],
                                'status':'meter_analysis_blocked','associated_kwh':None,'associated_pct_of_meter_kwh':None,
                                'group_associated_pct_of_meter_kwh':None,'note':str(exc)})
        summary['meters'].append(entry)
        if progress:progress(entry,len(summary['meters']),len(meters))
    summary['meter_count']=len(summary['meters'])
    summary['blocked_meter_count']=sum(m['status']=='blocked' for m in summary['meters'])
    summary['status']='all_meters_blocked' if summary['blocked_meter_count']==summary['meter_count'] else 'completed_requires_review'
    if optimizer_paths:
        pd.DataFrame(zone_effect_rows).to_csv(out/'zone-meter-effects.csv',index=False)
        save(out/'zone-meter-effects.json',{'effects':zone_effect_rows,
             'scope':'Per-meter statistical associations. Do not sum across overlapping meter boundaries or treat duration shares as energy savings.'})
    save(out/'summary.json',summary)
    return summary
