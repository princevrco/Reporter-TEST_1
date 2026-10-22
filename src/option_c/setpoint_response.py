"""Small conditional electricity models with explicit setpoint predictors.

Electrical coefficients are learned from existing daily meters, not assumed ratings.
Unobserved BMS days remain unavailable. Models describe observed conditions; changing
setpoints while holding zone temperatures fixed is not a causal policy simulation.
"""
import numpy as np
import pandas as pd
from .data import DataError, digest, records
from .regression import metrics
from .response import features as bms_features, solve, estimate


def prepare(raw, response_config, settings, timezone, reporting_end):
    """Choose a fixed asset cohort from past BMS availability, without meter targets."""
    settings=dict(settings)
    minimum=max(28,int(settings.get('minimum_training_days',28)))
    holdout_days=max(14,int(settings.get('holdout_calendar_days',14)))
    cutoff=str((pd.Timestamp(reporting_end)-pd.DateOffset(days=holdout_days)).date())
    cfg={**response_config,'maximum_age_minutes':float(settings.get('maximum_age_minutes',30))}
    daily,audit=bms_features(raw,cfg,timezone)
    daily=daily[daily.date<reporting_end].copy()
    past=daily[daily.date<cutoff]
    groups=[]
    for group,mapping in cfg['groups'].items():
        required=['bms_'+group+'::'+role for role in ('cool_setpoint','temperature','cooling_demand','setpoint_change')]
        if not all(col in past for col in required):continue
        count=int(past[required].notna().all(axis=1).sum())
        if count>=minimum:groups.append((count,group,required))
    groups.sort(key=lambda x:(-x[0],x[1]))
    admitted=pd.Series(True,index=past.index);cohort=[]
    for count,group,required in groups:
        trial=admitted & past[required].notna().all(axis=1)
        if int(trial.sum())>=minimum:
            cohort.append(group);admitted=trial
    if not cohort:raise DataError('No fixed setpoint/temperature cohort has 28 covered training days before the holdout')
    frame,definition=aggregate(daily,cohort)
    heating=[]
    for group in cohort:
        cols=['bms_'+group+'::'+role for role in ('heat_setpoint','heating_demand','heat_setpoint_change')]
        if not all(c in daily for c in cols):continue
        trial=admitted & past[cols].notna().all(axis=1)
        if trial.sum()>=minimum:heating.append(group);admitted=trial
    if heating:
        for role in ('heat_setpoint','heating_demand','heat_setpoint_change'):
            cols=['bms_'+group+'::'+role for group in heating]
            frame[role]=daily[cols].mean(axis=1).where(daily[cols].notna().all(axis=1))
            definition[role]={'columns':cols,'aggregation':'equal-weight fixed heating cohort; every contributor required'}
    meta={'schema':'option_c.setpoint_features.v1','cohort':cohort,'definition':definition,
          'heating_cohort':heating,
          'proposed_groups':len(cfg['groups']),'excluded_groups':sorted(set(cfg['groups'])-set(cohort)),
          'cohort_selected_before':cutoff,'config':cfg,'settings':settings,
          'semantics_verified':cfg.get('semantics_verified',False),'timezone':timezone,
          'scope':'Fixed cohort only. Group membership is not a physical zone-to-meter electrical map.',
          'input_sha256':digest(records(daily))}
    return frame,meta,audit


def aggregate(daily,cohort):
    result=daily[['date','hours']].copy()
    definition={}
    for role in ('cool_setpoint','temperature','cooling_demand','setpoint_change'):
        cols=['bms_'+group+'::'+role for group in cohort]
        if not all(c in daily for c in cols):raise DataError('Setpoint cohort signal missing: '+role)
        block=daily[cols]
        # The contributors never change from day to day. No missing asset is OFF.
        result[role]=block.mean(axis=1).where(block.notna().all(axis=1))
        definition[role]={'columns':cols,'aggregation':'equal-weight fixed-cohort mean; every contributor required'}
    local=pd.to_datetime(result.date)
    result['calendar_weekend']=(local.dt.dayofweek>=5).astype(float)
    length=np.where(local.dt.is_leap_year,366.,365.)
    phase=2*np.pi*(local.dt.dayofyear.to_numpy()-1)/length
    result['calendar_sin_year']=np.sin(phase)
    result['calendar_cos_year']=np.cos(phase)
    return result,definition


def _fit_parameters(frame,cols,alpha):
    x=frame[cols].to_numpy(float);y=frame.energy_kwh.to_numpy(float)/frame.hours.to_numpy(float)
    if len(frame)<max(12,4*(len(cols)+1)):
        raise DataError('Insufficient independent days for the selected low-dimensional response')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise DataError('Incomplete setpoint calibration inputs')
    try:return solve(x,y,alpha)
    except np.linalg.LinAlgError as exc:raise DataError('Unidentifiable setpoint calibration matrix') from exc


def _predict(frame,cols,parameters):
    return estimate(parameters,frame[cols].to_numpy(float))*frame.hours.to_numpy(float)


def fit(meter,feature_frame,meta,reporting_start,reporting_end,rate=.12):
    settings=meta['settings'];minimum=max(28,int(settings.get('minimum_training_days',28)))
    holdout=max(8,int(settings.get('minimum_holdout_days',8)))
    if rate<0 or not np.isfinite(rate):raise DataError('Rate must be finite and nonnegative')
    f=meter.merge(feature_frame,on='date',validate='one_to_one').sort_values('date')
    if 'quality' in f:f=f[f.quality.eq('good')]
    physical=['cool_setpoint','temperature','cooling_demand','setpoint_change']
    physical += [c for c in ('heat_setpoint','heating_demand','heat_setpoint_change') if c in f]
    f=f[f[physical].notna().all(axis=1)&pd.to_numeric(f.energy_kwh,errors='coerce').notna()]
    f=f[f.date<reporting_end].reset_index(drop=True)
    cutoff=meta['cohort_selected_before']
    training=f[f.date<cutoff];test=f[f.date>=cutoff]
    if len(training)<minimum or len(test)<holdout:
        raise DataError(f'Setpoint model needs {minimum} covered training days and {holdout} later test days; found {len(training)} and {len(test)}')
    calendar=[c for c in ('calendar_weekend',) if training[c].std()>1e-8]
    bundles={'setpoint_levels':['cool_setpoint','temperature'],
             'thermal_demand':['cooling_demand','temperature'],
             'setpoint_changes':['cool_setpoint','setpoint_change','temperature']}
    if 'heat_setpoint' in f:
        bundles.update(heating_levels=['heat_setpoint','temperature'],
                       heating_demand=['heating_demand','temperature'],
                       heating_changes=['heat_setpoint','heat_setpoint_change','temperature'],
                       combined_demand=['cooling_demand','heating_demand','temperature'])
    options=[]
    # Three chronological validation blocks before the final calendar holdout.
    # Feature variation is assessed on the first prefix, never test observations.
    prefix=training.iloc[:max(16,len(training)//2)]
    for name,cols in bundles.items():
        cols=[c for c in cols if prefix[c].std()>1e-8]
        if not any(c in cols for c in ('cool_setpoint','cooling_demand','setpoint_change','heat_setpoint','heating_demand','heat_setpoint_change')):continue
        options.append((name,cols+calendar))
    if not options:raise DataError('Setpoint effects are not identifiable: no varying setpoint/demand signals in past data')
    folds=[]
    for fraction in (.60,.75,.90):
        cut=int(len(training)*fraction);stop=min(len(training),cut+max(3,int(len(training)*.10)))
        if cut>=20 and stop>cut:folds.append((cut,stop))
    if len(folds)<2:raise DataError('Insufficient past validation blocks for the setpoint model')
    # At most three physical features plus weekend: every inner fit must have
    # four observations per coefficient, including the intercept.
    scores=[]
    alphas=[float(a) for a in settings.get('ridge_alphas',[.1,1,10,100])]
    if not alphas or any(not np.isfinite(a) or a<=0 for a in alphas):raise DataError('Invalid setpoint Ridge penalties')
    for name,cols in options:
        for alpha in alphas:
            errors=[]
            try:
                for cut,stop in folds:
                    past=training.iloc[:cut];valid=training.iloc[cut:stop]
                    params=_fit_parameters(past,cols,alpha);p=_predict(valid,cols,params)
                    if not np.isfinite(p).all() or (p<0).any():raise DataError('Unphysical inner response prediction')
                    errors.extend((valid.energy_kwh.to_numpy()-p).tolist())
                scores.append({'name':name,'features':cols,'alpha':alpha,'cv_rmse_kwh':float(np.sqrt(np.mean(np.square(errors))))})
            except DataError:continue
    if not scores:raise DataError('No setpoint candidate has enough past data for its parameter count')
    chosen=min(scores,key=lambda x:(x['cv_rmse_kwh'],len(x['features']),-x['alpha'],x['name']))
    cols=chosen['features'];frozen=_fit_parameters(training,cols,chosen['alpha'])
    p=_predict(test,cols,frozen);validation=metrics(test.energy_kwh,p)
    comparator=[]
    for alpha in alphas:
        errors=[]
        for cut,stop in folds:
            a=training.iloc[:cut];b=training.iloc[cut:stop]
            pred=_predict(b,calendar,_fit_parameters(a,calendar,alpha))
            errors.extend((b.energy_kwh.to_numpy()-pred).tolist())
        comparator.append((float(np.mean(np.square(errors))),alpha))
    ca=min(comparator)[1];cp=_predict(test,calendar,_fit_parameters(training,calendar,ca))
    calendar_metrics=metrics(test.energy_kwh,cp)
    improvement=(1-validation['rmse_kwh']/calendar_metrics['rmse_kwh'])*100 if calendar_metrics['rmse_kwh']>1e-8 else None
    low=training[cols].min().to_numpy();high=training[cols].max().to_numpy()
    outside=np.any((test[cols].to_numpy()<low-1e-8)|(test[cols].to_numpy()>high+1e-8),axis=1)
    gates=['electrical_boundary_unverified','conditional_temperature_not_a_thermal_policy_simulator',
           'fixed_asset_cohort_not_all_asset_causal_effect','historical_development_backtest_not_prospective_validation']
    if not meta['semantics_verified']:gates.append('bms_signal_semantics_unverified')
    if (p<0).any() or not np.isfinite(p).all():gates.append('unphysical_response_prediction')
    if validation['cvrmse_pct']>20 or abs(validation['nmbe_pct'])>5:gates.append('heldout_model_error_requires_review')
    if improvement is None or improvement<5:gates.append('setpoint_incremental_value_not_demonstrated')
    if outside.any():gates.append('heldout_bms_outside_training_support')
    if 'quality_review_required' in f and f.quality_review_required.isin([True,1,'True','true','1']).any():gates.append('meter_quality_review_required')
    final=_fit_parameters(f,cols,chosen['alpha']);fitted=_predict(f,cols,final)
    model={'schema':'option_c.setpoint_response.v1','features':cols,'parameters':final,
           'alpha':chosen['alpha'],'family':chosen['name'],'selection_candidates':scores,
           'feature_min':f[cols].min().tolist(),'feature_max':f[cols].max().tolist(),
           'training_days':len(training),'holdout_days':len(test),'valid_days':len(f),
           'holdout_start':cutoff,'fit_end_exclusive':str((pd.Timestamp(f.date.max())+pd.DateOffset(days=1)).date()),
           'holdout_metrics':validation,'calendar_comparator_metrics':calendar_metrics,
           'setpoint_rmse_improvement_pct':improvement,'feature_definition':meta,
           'gates':gates,'input_sha256':digest(records(f)),
           'validation_scope':'Fixed last calendar holdout; candidate and alpha selected only by earlier rolling folds. Scaling is learned per fold. Small model requires >=4 past days per coefficient and >=28 outer training days.'}
    model['model_id']=digest(model)
    heldout=[{'date':d,'measured_kwh':float(y),'setpoint_estimated_kwh':float(v) if np.isfinite(v) and v>=0 else None,
              'calendar_estimated_kwh':float(c),'outside_training_support':bool(o),'scope':'heldout_conditional_consumption_not_savings'}
             for d,y,v,c,o in zip(test.date,test.energy_kwh,p,cp,outside)]
    fit_map={d:float(v) if np.isfinite(v) and v>=0 else None for d,v in zip(f.date,fitted)}
    held_map={row['date']:row for row in heldout}
    meter_map=meter.set_index('date').energy_kwh.to_dict()
    rows=[]
    for day in pd.date_range(reporting_start,reporting_end,inclusive='left'):
        date=str(day.date());h=held_map.get(date,{})
        rows.append({'date':date,'measured_kwh':meter_map.get(date),
                     'diagnostic_fitted_setpoint_kwh':fit_map.get(date),
                     'heldout_setpoint_kwh':h.get('setpoint_estimated_kwh'),
                     'heldout_outside_support':h.get('outside_training_support'),
                     'scope':'Fitted values are reconstruction, not independently measured consumption or savings. Missing BMS periods are not filled.'})
    covered=[x for x in rows if x['diagnostic_fitted_setpoint_kwh'] is not None]
    eligible_held=[x for x in rows if x['heldout_setpoint_kwh'] is not None]
    total=sum(x['diagnostic_fitted_setpoint_kwh'] for x in covered) if covered else None
    summary={'status':'fitted_requires_review','family':model['family'],'training_days':len(training),'holdout_days':len(test),
             'reporting_expected_days':len(rows),'reporting_reconstruction_days':len(covered),
             'diagnostic_reconstructed_kwh_available_days':total,
             'diagnostic_reconstructed_energy_cost_cad':total*rate if total is not None else None,
             'full_period_setpoint_consumption_kwh':total if len(covered)==len(rows) else None,
             'independent_reporting_days':len(eligible_held),'heldout_setpoint_kwh':sum(x['heldout_setpoint_kwh'] for x in eligible_held) if eligible_held else None,
             'heldout_metrics':validation,'calendar_comparator_metrics':calendar_metrics,
             'setpoint_rmse_improvement_pct':improvement,'gates':gates,
             'scope':'Conditional consumption estimates, not exact savings. No historical setpoint policy or complete thermal counterfactual is inferred.'}
    return model,summary,rows,heldout


def predict(model,raw_bms,start,end,overrides=None):
    """Estimate a supplied operating trajectory, optionally overriding setpoints.

    Overrides are explicit {group: cooling_c} or {group: {cooling: c, heating: c}}. Unmeasured source
    rows remain missing. Estimates outside training support stay diagnostic only.
    """
    if digest({k:v for k,v in model.items() if k!='model_id'})!=model['model_id']:raise DataError('Setpoint model checksum mismatch')
    meta=model['feature_definition'];raw=raw_bms.copy();cfg=dict(meta['config'])
    # A future trajectory needs only the fitted cohort's inputs, not unrelated
    # historical BACnet points that the fitted model never uses.
    needs_heat=any(c in model['features'] for c in ('heat_setpoint','heating_demand','heat_setpoint_change'))
    heating=set(meta.get('heating_cohort',[])) if needs_heat else set()
    cfg['groups']={g:{role:col for role,col in cfg['groups'][g].items()
        if role in ('cool_setpoint','temperature') or (role=='heat_setpoint' and g in heating)}
        for g in meta['cohort']}
    for group,value in (overrides or {}).items():
        if group not in meta['cohort']:raise DataError('Override group outside fitted cohort: '+group)
        modes=value if isinstance(value,dict) else {'cooling':value}
        if not modes or set(modes)-{'cooling','heating'}:raise DataError('Override modes must be cooling or heating')
        for mode,number in modes.items():
            number=float(number)
            if not np.isfinite(number) or not -20<=number<=60:raise DataError('Invalid setpoint override')
            role='cool_setpoint' if mode=='cooling' else 'heat_setpoint'
            if role not in cfg['groups'][group]:raise DataError('No mapped '+mode+' setpoint for '+group)
            column=cfg['groups'][group][role]
            if column not in raw:raise DataError('Override input signal missing: '+column)
            raw.loc[raw[column].notna(),column]=number
    daily,_=bms_features(raw,cfg,meta['timezone']);f,_=aggregate(daily,meta['cohort'])
    for role in ('heat_setpoint','heating_demand','heat_setpoint_change'):
        if role in model['features'] and role in meta['definition']:
            cols=meta['definition'][role]['columns']
            f[role]=daily[cols].mean(axis=1).where(daily[cols].notna().all(axis=1))
    f=f[(f.date>=start)&(f.date<end)]
    cols=model['features'];x=f[cols].to_numpy(float)
    p=estimate(model['parameters'],x)*f.hours.to_numpy(float)
    low=np.array(model['feature_min']);high=np.array(model['feature_max'])
    outside=np.any((x<low-1e-8)|(x>high+1e-8),axis=1)
    return [{'date':d,'conditional_setpoint_kwh':float(v) if np.isfinite(v) and v>=0 else None,
             'outside_training_support':bool(o),
             'scope':'Conditional on supplied zone temperatures. A setpoint override is not simulated thermal evolution or verified savings.'}
            for d,v,o in zip(f.date,p,outside)]
