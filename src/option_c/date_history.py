"""Sparse BMS observations retained; eligibility decided on complete daily features."""
import numpy as np
import pandas as pd
from .data import DataError,digest,records
from .strict_history import mapping

COLUMNS=['date','hours','temperature_c','cool_degree_hours','heat_degree_hours','cool_proxy_hours','heat_proxy_hours','asset_count','coverage_fraction']


def clean(raw,required_columns=None):
    if 'timestamp' not in raw or raw.columns.duplicated().any():raise DataError('BMS requires unique column names and timestamp')
    f=raw.copy();missing=f.isna()|f.astype(str).apply(lambda c:c.str.strip().eq(''))
    f=f.mask(missing);parsed=[];good=[]
    for t in f.timestamp:
        try:
            v=pd.Timestamp(t)
            if pd.isna(v) or v.tzinfo is None:raise ValueError()
            parsed.append(v.tz_convert('UTC'));good.append(True)
        except (ValueError,TypeError):parsed.append(pd.NaT);good.append(False)
    keep=np.array(good,dtype=bool);out=f.loc[keep].copy();out['timestamp']=pd.to_datetime(np.array(parsed,dtype=object)[keep],utc=True)
    bad=pd.DataFrame({'source_row':np.flatnonzero(~keep)+2,'reason':'invalid_or_missing_timestamp'})
    audit={'input_rows':len(f),'accepted_rows':len(out),'rejected_timestamp_rows':len(bad),
        'rows_with_any_blank_retained':int((missing.loc[keep].any(axis=1)).sum()),
        'rule':'Blank signal cells do not reject a source row; required daily feature coverage determines date eligibility','date_imputation':False}
    return out.sort_values('timestamp').reset_index(drop=True),bad,audit


def prepare(raw,weather,timezone,end,config=None,cadence_minutes=15,required_columns=None,observation_mask=None):
    settings=config or {};minimum=float(settings.get('minimum_daily_coverage',.9));age=float(settings.get('maximum_age_minutes',30))
    if not 0<minimum<=1 or not 0<age<=60:raise DataError('Invalid daily coverage or bounded observation age')
    groups=mapping(raw,config)
    f=raw.sort_values('timestamp').copy()
    invalid={}
    cols={c for g in groups.values() for c in g.values()}
    if required_columns:
        if not set(required_columns)<=set(f):raise DataError('Required BMS signal not found')
        cols|=set(required_columns)
    for c in cols:
        values=pd.to_numeric(f[c],errors='coerce');valid=np.isfinite(values)
        if c in {v for g in groups.values() for v in g.values()}:valid &= values.between(-20,60)
        invalid[c]=int((f[c].notna()&~valid).sum());f[c]=values.where(valid)
    if f.empty:
        return f,{},pd.DataFrame(columns=COLUMNS),pd.DataFrame(columns=['date','eligible','reason']),{'cohort':[],'excluded_assets':list(groups),'reason':'No valid BMS timestamp rows'}
    f['timestamp']=pd.to_datetime(f.timestamp,utc=True,format='mixed')
    if f.timestamp.duplicated().any():raise DataError('Merge duplicate BMS observations before date preparation')
    finish=pd.Timestamp(end,tz=timezone).tz_convert('UTC');f=f[f.timestamp<finish].copy()
    if f.empty:return f,{},pd.DataFrame(columns=COLUMNS),pd.DataFrame(columns=['date','eligible','reason']),{'cohort':[],'excluded_assets':list(groups),'reason':'No observations before reporting end'}
    first=f.timestamp.min().tz_convert(timezone).normalize();edges=pd.date_range(first,pd.Timestamp(end,tz=timezone),freq='D').tz_convert('UTC')
    expected=np.diff(edges.asi8)/3.6e12;dates=edges[:-1].tz_convert(timezone).date.astype(str)
    times=f.timestamp.array.asi8
    native=None
    if observation_mask is not None:
        if not isinstance(observation_mask,pd.DataFrame) or not observation_mask.index.is_unique or not set(cols)<=set(observation_mask.columns):
            raise DataError('Observation mask must cover required columns with unique timestamp index')
        native=observation_mask.reindex(pd.DatetimeIndex(f.timestamp))
        if native[list(cols)].isna().any().any():raise DataError('Observation mask does not cover every source timestamp')
    expiry=times+int(age*60e9)
    expiry[-1]=min(expiry[-1],times[-1]+int(cadence_minutes*60e9))
    boundaries=np.unique(np.concatenate([times,expiry,edges.asi8]));boundaries=boundaries[(boundaries>=edges.asi8[0])&(boundaries<=edges.asi8[-1])]
    left=boundaries[:-1];duration=np.diff(boundaries)/3.6e12;pos=np.searchsorted(times,left,side='right')-1;safe=np.maximum(pos,0)
    known=(pos>=0)&(left<expiry[safe]);day=np.searchsorted(edges.asi8,left,side='right')-1
    values={};known_by_column={}
    for c in cols:
        if native is None:
            # Conservative legacy behavior: every row is an observation for each signal.
            selected=np.ones(len(f),bool)
        else:selected=native[c].to_numpy(bool)
        positions=np.flatnonzero(selected)
        if not len(positions):
            values[c]=np.full(len(left),np.nan);known_by_column[c]=np.zeros(len(left),bool);continue
        observed_times=times[positions];k=np.searchsorted(observed_times,left,side='right')-1;safe_k=np.maximum(k,0)
        source_pos=positions[safe_k];values[c]=f[c].to_numpy(float)[source_pos]
        known_by_column[c]=(k>=0)&(left<observed_times[safe_k]+int(age*60e9))
        # The last source observation has only the declared sampling duration.
        known_by_column[c] &= (source_pos!=len(f)-1)|(left<times[-1]+int(cadence_minutes*60e9))
    valids={} 
    for asset,g in groups.items():
        mask=np.ones(len(left),bool)
        for c in g.values():mask &= known_by_column[c]&np.isfinite(values[c])
        if required_columns:
            for c in required_columns:mask &= known_by_column[c]&np.isfinite(values[c])
        valids[asset]=mask
    def hours(mask):return np.bincount(day,weights=duration*mask,minlength=len(dates))
    def available(mask):return hours(mask)/expected>=minimum-1e-9
    cutoff=str((pd.Timestamp(end)-pd.DateOffset(days=14)).date());past=dates<cutoff
    counts={a:int((available(v)&past).sum()) for a,v in valids.items()}
    explicit=config is not None and 'groups' in config
    chosen=[];joint=np.ones(len(left),bool);target=max(counts.values(),default=0)
    for asset in sorted(groups,key=lambda a:(-counts[a],a)):
        trial=joint&valids[asset]
        if explicit or (target>0 and int((available(trial)&past).sum())>=target):chosen.append(asset);joint=trial
    if not chosen:joint[:]=False
    known_hours=hours(joint);eligible=available(joint) if chosen else np.zeros(len(dates),bool)
    required_today=list({c for a in chosen for c in groups[a].values()}|set(required_columns or []))
    observed_dates=set()
    if required_today:
        observed_dates=None
        for c in required_today:
            present=np.isfinite(f[c].to_numpy(float))
            if native is not None:present &= native[c].to_numpy(bool)
            signal_dates=set(f.loc[present,'timestamp'].dt.tz_convert(timezone).dt.date.astype(str))
            observed_dates=signal_dates if observed_dates is None else observed_dates&signal_dates
        observed_dates=observed_dates or set()
    eligible &= np.isin(dates,list(observed_dates))
    w=weather.set_index('date').temperature_c;result=[];coverage=[]
    cool=np.zeros(len(left));heat=np.zeros(len(left));con=np.zeros(len(left));hon=np.zeros(len(left))
    for asset in chosen:
        g=groups[asset];t=values[g['temperature']];c=np.maximum(t-values[g['cool_setpoint']],0)
        h=np.maximum(values[g['heat_setpoint']]-t,0) if 'heat_setpoint' in g else np.zeros(len(left))
        cool+=np.where(joint,c,0);heat+=np.where(joint,h,0);con+=np.where(joint,c>0,0);hon+=np.where(joint,h>0,0)
    totals={k:np.bincount(day,weights=duration*v,minlength=len(dates)) for k,v in [('cool_degree_hours',cool),('heat_degree_hours',heat),('cool_proxy_hours',con),('heat_proxy_hours',hon)]}
    for i,date in enumerate(dates):
        good=bool(eligible[i]);reason='' if good else 'missing_required_daily_features_or_insufficient_subdaily_coverage'
        coverage.append({'date':date,'eligible':good,'reason':reason,'observed_required_hours':float(known_hours[i]),'day_hours':float(expected[i]),'coverage_fraction':float(known_hours[i]/expected[i]),'minimum_required_fraction':minimum})
        if good:
            if date not in w or not np.isfinite(w[date]):raise DataError('External weather missing for eligible BMS date '+date)
            result.append({'date':date,'hours':float(expected[i]),'temperature_c':float(w[date]),**{k:float(v[i]) for k,v in totals.items()},'asset_count':len(chosen),'coverage_fraction':float(known_hours[i]/expected[i])})
    selected={a:groups[a] for a in chosen}
    return f,selected,pd.DataFrame(result,columns=COLUMNS),pd.DataFrame(coverage),{
        'cohort':chosen,'cohort_selected_before':cutoff,'proposed_assets':len(groups),'excluded_assets':sorted(set(groups)-set(chosen)),
        'past_covered_days_by_asset':counts,'invalid_observations_by_signal':invalid,'minimum_daily_coverage':minimum,'maximum_observation_age_minutes':age,
        'eligible_dates':int(eligible.sum()),'excluded_dates':int((~eligible).sum()),'date_imputation':False,
        'observation_semantics':'native_mask' if native is not None else 'every_row_is_an_observation',
        'cohort_target_earlier_days':target,
        'method':'Fixed cohort selected on earlier availability only without reducing best available earlier coverage. Native observations apply until the next observation of that signal or bounded expiry. Explicit native blanks invalidate immediately. Alignment-only blanks are ignored only when a verified native mask is supplied. Empty dates never synthesized.'}
