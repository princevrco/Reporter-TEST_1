"""Daily BMS electrical response with bounded signal age and temporal Ridge validation."""
import numpy as np
import pandas as pd
from .data import DataError,digest,records
from .regression import metrics


SUFFIXES={'cool_stage1':'_Cool1','cool_stage2':'_Cool2','heat_stage1':'_Heat1',
          'heat_stage2':'_Heat2','fan':'_FanEnable','valve':'_CoolingValve',
          'temperature':'_ZoneTemp','cool_setpoint':'_CoolSetpointU_AV','heat_setpoint':'_HeatSetpointU_AV'}


def discover(frame):
    groups={}
    aliases={'fan':('_FanEnable','_SupplyFan'),'cool_setpoint':('_CoolSetpointU_AV','_CoolSetpointUsed','_ZoneCoolSetpoint'),
             'heat_setpoint':('_HeatSetpointU_AV','_HeatSetpointUsed','_ZoneHeatSetpoint')}
    for col in frame:
        for role,suffix in SUFFIXES.items():
            for accepted in aliases.get(role,(suffix,)):
                if col.endswith(accepted):groups.setdefault(col[:-len(accepted)],{}).setdefault(role,col)
    return {'schema':'option_c.response_config.v1','semantics_verified':False,'groups':groups,
            'bin_minutes':15,'maximum_age_minutes':15,'minimum_daily_coverage':.9,
            'minimum_training_days':40,'minimum_holdout_days':8,'maximum_features':24,
            'ridge_alphas':[.01,.1,1,10,100,1000],
            'note':'Proposed column bindings from names only; inspect encodings, shared equipment and load representativeness. Features are uncertain operation proxies until verified.'}


def features(frame,cfg,timezone):
    groups=cfg.get('groups',{})
    if not groups:raise DataError('No mapped BMS groups; supply response config or recognizable BMS columns')
    minutes=int(cfg.get('bin_minutes',15));age=float(cfg.get('maximum_age_minutes',15));coverage=float(cfg.get('minimum_daily_coverage',.9))
    if minutes not in (5,10,15,20,30,60) or not 0<age<=60 or not 0<coverage<=1:raise DataError('Invalid BMS bin, age or coverage setting')
    f=frame.copy();f['timestamp']=pd.to_datetime(f.timestamp,utc=True,format='mixed')
    if f.timestamp.duplicated().any():raise DataError('Merge duplicate BMS timestamps first')
    f=f.sort_values('timestamp');start=f.timestamp.min().tz_convert(timezone).normalize();end=f.timestamp.max().tz_convert(timezone).normalize()+pd.DateOffset(days=1)
    grid=pd.date_range(start,end,freq=f'{minutes}min',inclusive='left').tz_convert('UTC')
    if len(grid)>2_000_000:raise DataError('BMS grid too large')
    signals={};audit={};bindings=[]
    for group,mapping in groups.items():
        for role,col in mapping.items():
            if role not in SUFFIXES:raise DataError('Unsupported BMS role: '+role)
            if col not in f:raise DataError('Mapped BMS column missing: '+col)
            if (role,col) in bindings:raise DataError('Same BMS signal mapped to multiple equipment groups; consolidate shared equipment')
            bindings.append((role,col));raw=pd.to_numeric(f[col],errors='coerce');good=np.isfinite(raw)
            if role in ('temperature','cool_setpoint','heat_setpoint'):good &= raw.between(-20,60)
            elif role=='valve':good &= raw.between(0,100)
            else:good &= raw.isin([0,1])
            # Preserve explicit invalid observations as NaN so they stop propagation of an earlier valid state.
            src=pd.DataFrame({'timestamp':f.timestamp,'value':raw.where(good)}).loc[f[col].notna()]
            if src.empty:sample=pd.Series(np.nan,index=grid)
            else:
                aligned=pd.merge_asof(pd.DataFrame({'timestamp':grid}),src,on='timestamp',direction='backward',tolerance=pd.Timedelta(minutes=age))
                sample=pd.Series(aligned.value.to_numpy(),index=grid)
            signals[(group,role)]=sample
            observed=f.loc[f[col].notna(),'timestamp'];spacing=observed.diff().dt.total_seconds().div(60).dropna()
            audit[col]={'observed_rows':int(f[col].notna().sum()),'rejected_rows':int((f[col].notna()&~good).sum()),'populated_bins':int(sample.notna().sum()),
                'median_sampling_minutes':float(spacing.median()) if len(spacing) else None,
                'p95_sampling_minutes':float(spacing.quantile(.95)) if len(spacing) else None,
                'maximum_gap_minutes':float(spacing.max()) if len(spacing) else None}
    signals_frame={group+'::'+role:series for (group,role),series in signals.items()}
    for role in SUFFIXES:
        cols=[s for (group,r),s in signals.items() if r==role]
        if cols:
            block=pd.concat(cols,axis=1)
            # Sum/average only when all configured contributors are known, never silently treat a missing asset as OFF.
            signals_frame[role]=block.mean(axis=1).where(block.notna().all(axis=1))
    for role in ('cooling_demand','heating_demand','setpoint_change','heat_setpoint_change'):
        cols=[]
        for group,mapping in groups.items():
            sp=signals.get((group,'heat_setpoint' if role in ('heating_demand','heat_setpoint_change') else 'cool_setpoint'));temp=signals.get((group,'temperature'))
            if sp is not None:
                if role in ('setpoint_change','heat_setpoint_change'):
                    value=sp.diff().abs().where(sp.notna()&sp.shift().notna());cols.append(value);signals_frame[group+'::'+role]=value
                elif temp is not None:
                    value=(sp-temp).clip(lower=0) if role=='heating_demand' else (temp-sp).clip(lower=0)
                    cols.append(value);signals_frame[group+'::'+role]=value
        if cols:
            block=pd.concat(cols,axis=1);signals_frame[role]=block.mean(axis=1).where(block.notna().all(axis=1))
    if not signals_frame:raise DataError('No usable mapped BMS signals')
    bins=pd.DataFrame(signals_frame);bins.index=bins.index.tz_convert(timezone)
    expected=pd.Series(1.,index=bins.index).resample('D').sum()
    daily=bins.resample('D').mean().where(bins.resample('D').count().div(expected,axis=0)>=coverage)
    daily=daily.add_prefix('bms_')
    daily=pd.concat([daily,daily.shift(1).add_suffix('_lag1')],axis=1)
    daily['hours']=[(t+pd.DateOffset(days=1)-t).total_seconds()/3600 for t in daily.index]
    daily['date']=daily.index.date.astype(str)
    for day in range(1,7):daily['calendar_weekday_'+str(day)]=(daily.index.dayofweek==day).astype(float)
    return daily.reset_index(drop=True),{'signal_audit':audit,'config_sha256':digest(cfg),
        'semantics_verified':cfg.get('semantics_verified',False),'equipment_groups':list(groups),
        'method':'Past-only sample-and-hold at bounded age, daily equal-bin signal means; one-day lags require prior calendar day. Not measured runtime or electrical energy.',
        'coverage_rule':'All mapped contributors to each feature required; no missing asset assumed OFF',
        'maximum_age_minutes':age,'bin_minutes':minutes,'minimum_daily_coverage':coverage}


def solve(x,y,alpha):
    # Every fold learns scaling and feature selection from its own past data only.
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-10]=1
    z=(x-mean)/scale;center=float(y.mean())
    coef=np.linalg.solve(z.T@z+alpha*np.eye(x.shape[1]),z.T@(y-center))
    return {'mean':mean.tolist(),'scale':scale.tolist(),'coef':coef.tolist(),'intercept':center}


def estimate(params,x):
    return ((x-np.array(params['mean']))/np.array(params['scale']))@np.array(params['coef'])+params['intercept']


def fit(meter,feature_frame,cfg,timezone):
    f=meter.merge(feature_frame,on='date',validate='one_to_one').sort_values('date').reset_index(drop=True)
    cols=[c for c in f if c.startswith('bms_')]
    f=f.loc[pd.to_numeric(f.energy_kwh,errors='coerce').notna()].copy()
    if 'quality' in f:f=f[f.quality=='good'].copy()
    if not cols:raise DataError('No BMS feature columns')
    f=f[f[cols].notna().any(axis=1)].copy()
    minimum=max(28,int(cfg.get('minimum_training_days',40)));holdout=max(8,int(cfg.get('minimum_holdout_days',8)))
    if int(len(f)*.8)<minimum or len(f)-int(len(f)*.8)<holdout:
        raise DataError(f'BMS response needs >= {minimum} eligible training days and {holdout} later holdout days; only {len(f)} days have any coverage-screened BMS features')
    # Choose feature coverage on the first training portion only, never on held-out targets.
    past=f.iloc[:max(28,int(cfg.get('minimum_training_days',40)))]
    candidates=[c for c in cols if past[c].notna().mean()>=.6 and past[c].std()>1e-8]
    candidates=sorted(candidates,key=lambda c:(-past[c].notna().mean(),c))
    selected=[];admitted=pd.Series(True,index=past.index)
    for col in candidates:
        trial=admitted&past[col].notna()
        if trial.mean()>=.6:selected.append(col);admitted=trial
        if len(selected)>=int(cfg.get('maximum_features',24)):break
    candidates=selected
    if not candidates:raise DataError('No varying, sufficiently covered BMS features in training history')
    f=f[np.isfinite(f[candidates].to_numpy(float)).all(axis=1)].reset_index(drop=True)
    split=int(len(f)*.8)
    if split<minimum or len(f)-split<holdout:raise DataError(f'BMS response needs >= {minimum} eligible training days and {holdout} later holdout days; found {split} and {len(f)-split}')
    # Recompute feature variation from the actual outer training period. Drop constants.
    candidates=[c for c in candidates if f.iloc[:split][c].std()>1e-8]
    if not candidates:raise DataError('No BMS variation in the actual training period')
    candidates += [c for c in f if c.startswith('calendar_weekday_') and f.iloc[:split][c].std()>1e-8]
    x=f[candidates].to_numpy(float);hours=f.hours.to_numpy(float);y=f.energy_kwh.to_numpy(float)/hours
    folds=[]
    for fraction in (.5,.65,.8):
        cut=int(split*fraction);stop=min(split,cut+max(7,int(split*.15)))
        if cut>=20 and stop>cut:folds.append((cut,stop))
    if len(folds)<2:raise DataError('Insufficient chronological folds for penalty selection')
    scores=[]
    for alpha in cfg.get('ridge_alphas',[.01,.1,1,10,100,1000]):
        alpha=float(alpha)
        if not np.isfinite(alpha) or alpha<=0:raise DataError('Ridge penalties must be positive and finite')
        losses=[]
        for cut,stop in folds:
            prediction=estimate(solve(x[:cut],y[:cut],alpha),x[cut:stop])*hours[cut:stop]
            losses.append(float(np.mean((prediction-f.energy_kwh.to_numpy()[cut:stop])**2)))
        scores.append((float(np.mean(losses)),alpha))
    alpha=min(scores)[1];held=estimate(solve(x[:split],y[:split],alpha),x[split:])*hours[split:]
    if not np.isfinite(held).all() or (held<0).any():raise DataError('Unphysical held-out BMS energy prediction')
    validation=metrics(f.energy_kwh.to_numpy()[split:],held);gates=['electrical_boundary_unverified','statistical_response_not_causal_savings','weather_not_modeled','selected_bms_features_do_not_prove_all_asset_effects',
        'heldout_consumption_uses_observed_bms_not_a_free_running_operating_forecast']
    if cfg.get('semantics_verified') is not True:gates.append('bms_signal_semantics_unverified')
    if 'quality_review_required' in f and f.quality_review_required.isin([True,1,'True','true','1']).any():gates.append('meter_quality_review_required')
    if validation['cvrmse_pct']>20 or abs(validation['nmbe_pct'])>5:gates.append('heldout_model_error_requires_review')
    artifact={'schema':'option_c.ridge_response.v1','features':candidates,'parameters':solve(x,y,alpha),
        'alpha':alpha,'feature_min':x.min(axis=0).tolist(),'feature_max':x.max(axis=0).tolist(),
        'fit_start':f.date.iloc[0],'fit_end_exclusive':str((pd.Timestamp(f.date.iloc[-1])+pd.DateOffset(days=1)).date()),
        'valid_days':len(f),'holdout_metrics':validation,'timezone':timezone,'gates':gates,
        'config_sha256':digest(cfg),'input_sha256':digest(records(f)),
        'validation_scope':'Last 20 percent held out; inner expanding folds tune Ridge penalty; final artifact refits all eligible days',
        'heldout_predictions':[{'date':str(day),'measured_kwh':float(actual),'bms_estimated_kwh':float(pred)}
             for day,actual,pred in zip(f.date.iloc[split:],f.energy_kwh.iloc[split:],held)]}
    artifact['model_id']=digest(artifact)
    fitted=estimate(artifact['parameters'],x)*hours
    diagnostics=[{'date':str(day),'bms_estimated_kwh':float(pred) if pred>=0 else None,'measured_kwh':float(actual),
        'scope':'in_sample_diagnostic_not_independent_consumption_or_savings'} for day,pred,actual in zip(f.date,fitted,f.energy_kwh)]
    return artifact,diagnostics


def predict(model,feature_frame,cfg):
    if digest({k:v for k,v in model.items() if k!='model_id'})!=model.get('model_id'):raise DataError('BMS response model checksum mismatch')
    if digest(cfg)!=model['config_sha256']:raise DataError('BMS response configuration changed')
    f=feature_frame.copy();f=f[f.date>=model['fit_end_exclusive']]
    x=f[model['features']].to_numpy(float);p=estimate(model['parameters'],x)*f.hours.to_numpy(float)
    outside=np.any((x<np.array(model['feature_min']))|(x>np.array(model['feature_max'])),axis=1)
    return [{'date':day,'bms_estimated_kwh':float(v) if np.isfinite(v) and v>=0 else None,
             'outside_training_support':bool(o),'scope':'conditional_operating_scenario_not_realized_savings'} for day,v,o in zip(f.date,p,outside)]
