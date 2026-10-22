"""Reviewable energy anomalies. Flags never silently remove or correct a reading."""
import numpy as np
import pandas as pd
from .data import DataError,digest,stamp


def assess(part,cfg,start,end,reviews=None):
    f=part.copy().sort_values('date').reset_index(drop=True)
    if f.duplicated('date').any():raise DataError('Duplicate meter dates before quality screening')
    meter=str(cfg['meter_id']);options=cfg.get('data_quality',{})
    cutoff=stamp(cfg.get('baseline_end_exclusive',cfg['intervention_timestamp']),cfg['timezone']).tz_convert(cfg['timezone']).date().isoformat()
    baseline=f[(f.date<cutoff)&f.energy_kwh.notna()]
    values=baseline.energy_kwh.to_numpy(float)
    if not len(values):return f,[],{'unresolved_baseline_events':0,'unresolved_reporting_events':0}
    median=float(np.median(values));positive=values[values>0]
    reference=median if median>0 else float(np.median(positive)) if len(positive) else 0.
    scale=float(1.4826*np.median(np.abs(values-median)))
    threshold=max(1.,reference*float(options.get('spike_factor',8)),median+float(options.get('spike_mad',12))*scale)
    events=[]
    def add(code,indexes,phase,detail):
        if not len(indexes):return
        rows=f.loc[indexes];dates=rows.date.tolist()
        anchor={'meter_id':meter,'code':code,'dates':dates,
                'raw_kwh':[None if pd.isna(v) else float(v) for v in rows.energy_kwh]}
        events.append({'event_id':digest(anchor),'meter_id':meter,'code':code,'phase':phase,
            'start_date':dates[0],'end_exclusive':str((pd.Timestamp(dates[-1])+pd.DateOffset(days=1)).date()),
            'days':len(dates),'energy_kwh':float(rows.energy_kwh.sum()),'detail':detail,
            'review_status':'unreviewed','decision':None,'reference':None,'reviewer':None,'_indexes':list(indexes)})
    scope=((f.date<cutoff)|((f.date>=start)&(f.date<end)))
    for i in f.index[scope&f.energy_kwh.gt(threshold)]:
        add('extreme_daily_spike',[i],'baseline' if f.loc[i,'date']<cutoff else 'reporting',
            f'Baseline median={median:.6g} kWh; engineering screen threshold={threshold:.6g} kWh. Not proof of invalid data.')
    zeros=f.energy_kwh.eq(0)&scope
    # Missing calendar dates break a run. Optimization may explain reporting zeros; review rather than reject them.
    segments=[];current=[]
    for i in f.index:
        if zeros.iloc[i] and (not current or pd.Timestamp(f.loc[i,'date'])-pd.Timestamp(f.loc[current[-1],'date'])==pd.Timedelta(days=1)):
            current.append(i)
        else:
            if current:segments.append(current)
            current=[i] if zeros.iloc[i] else []
    if current:segments.append(current)
    for group in segments:
        for phase,indices in (('baseline',[i for i in group if f.loc[i,'date']<cutoff]),
                              ('reporting',[i for i in group if start<=f.loc[i,'date']<end])):
            if len(indices)>=int(options.get('zero_run_days',7)):add('sustained_zero_run',indices,phase,'Could be true shutdown or missing telemetry represented as zero; verify meter source.')
    # Detect a sustained historical level change only within the untreated baseline.
    window=int(options.get('shift_window_days',28));ratio=float(options.get('shift_ratio',1.75))
    clean=baseline.set_index(pd.to_datetime(baseline.date)).energy_kwh
    if len(clean)>=2*window:
        grid=clean.reindex(pd.date_range(clean.index.min(),clean.index.max(),freq='D'))
        candidates=[]
        for pos in range(window,len(grid)-window+1,7):
            left=grid.iloc[pos-window:pos];right=grid.iloc[pos:pos+window]
            if left.notna().sum()<window*.9 or right.notna().sum()<window*.9:continue
            a,b=float(left.median()),float(right.median())
            if min(a,b)<=0:continue
            local_scale=float(1.4826*np.median(np.abs(left.dropna()-a)))
            strength=max(a,b)/min(a,b)
            if strength>=ratio and abs(a-b)>max(1.,6*local_scale):candidates.append((strength,pos,a,b))
        if candidates:
            strength,pos,a,b=max(candidates)
            dates=set(grid.index[pos-window:pos+window].date.astype(str))
            add('baseline_level_shift',f.index[f.date.isin(dates)],'baseline',
                f'Adjacent {window}-day medians changed from {a:.6g} to {b:.6g} kWh; ratio={strength:.3g}. Investigate boundary, occupancy, source or operating changes.')
    decisions={}
    for item in reviews or []:
        key=item.get('event_id')
        if not key or key in decisions:raise DataError('Review ledger event IDs must be present and unique')
        if item.get('reviewed') is not True or not item.get('reviewer') or not item.get('reference'):raise DataError('Review decisions require reviewed=true, reviewer and documented reference')
        if item.get('decision') not in ('accept','exclude','correct'):raise DataError('Review decision must be accept, exclude or correct')
        decisions[key]=item
    changes={}
    f['quality_review_required']=False
    for event in events:
        item=decisions.get(event['event_id'])
        if not item:f.loc[event['_indexes'],'quality_review_required']=True
        if item:
            event.update(review_status='reviewed',decision=item['decision'],reference=item['reference'],reviewer=item['reviewer'])
            if item['decision']=='correct':
                value=float(item.get('corrected_kwh',float('nan')))
                if event['days']!=1 or not np.isfinite(value) or value<0:raise DataError('Correction requires a single-day event and finite nonnegative corrected_kwh')
            for i in event['_indexes']:
                if item['decision'] in ('exclude','correct'):
                    change=(item['decision'],value if item['decision']=='correct' else None)
                    if i in changes and changes[i]!=change:raise DataError('Conflicting reviewed actions for the same reading')
                    changes[i]=change
    for i,(decision,value) in changes.items():
        if 'original_energy_kwh' not in f:f['original_energy_kwh']=f.energy_kwh
        f.loc[i,'energy_kwh']=np.nan if decision=='exclude' else value
        f.loc[i,'quality']='excluded' if decision=='exclude' else 'good'
        f.loc[i,'reviewed_action']=decision
    stats={'unresolved_baseline_events':sum(e['phase']=='baseline' and e['review_status']=='unreviewed' for e in events),
           'unresolved_reporting_events':sum(e['phase']=='reporting' and e['review_status']=='unreviewed' for e in events),
           'reviewed_events':sum(e['review_status']=='reviewed' for e in events),'screen_reference_median_kwh':median,
           'screen_spike_threshold_kwh':threshold,'note':'Engineering screens are not proof of error; no unreviewed deletion, replacement or winsorization.'}
    return f,[{k:v for k,v in e.items() if k!='_indexes'} for e in events],stats


def withhold(result,reasons):
    """Keep investigative calculations, but never release failed models as savings."""
    if not reasons:return result
    result['release_status']='withheld_requires_review';result['release_reasons']=sorted(set(result.get('release_reasons',[])+list(reasons)))
    result['status']='comparison_withheld_requires_review'
    for key in ('savings_kwh_paired','savings_kwh_complete_period','savings_pct_paired','aggregate_model_band_95_savings_kwh'):
        if key in result:
            if 'diagnostic_'+key not in result:result['diagnostic_'+key]=result[key]
            result[key]=None
    for row in result.get('periods',[]):
        if 'savings_kwh' in row:
            if 'diagnostic_difference_kwh' not in row:row['diagnostic_difference_kwh']=row['savings_kwh']
            row['savings_kwh']=None
    result['gates']=sorted(set(result.get('gates',[])+list(reasons)))
    result['scope']='Unvalidated energy comparison. Diagnostic differences are not released savings; resolve data/model findings before interpretation.'
    return result
