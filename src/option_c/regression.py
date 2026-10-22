from __future__ import annotations
import numpy as np
import pandas as pd
from .data import DataError,check_cfg,digest,frame,stamp,records


def metrics(y,p,parameters=0):
    y,p=np.asarray(y,float),np.asarray(p,float);n=len(y)
    if n<=parameters or y.mean()<=0:raise DataError('Insufficient metric degrees of freedom or zero mean energy')
    error=y-p;tss=float(np.sum((y-y.mean())**2))
    return {'n':n,'parameters':parameters,'rmse_kwh':float(np.sqrt(np.mean(error**2))),
            'cvrmse_pct':float(100*np.sqrt(np.sum(error**2)/(n-parameters))/y.mean()),
            'nmbe_pct':float(100*error.sum()/((n-parameters)*y.mean())),
            'r2':float(1-np.sum(error**2)/tss) if tss>1e-12 else None}


def matrix(f,terms,cfg):
    local=f.timestamp.dt.tz_convert(cfg['timezone']);temperature=f.temperature_c.to_numpy(float)
    values=[]
    for term in terms:
        kind=term['kind']
        if kind=='constant':v=np.ones(len(f))
        elif kind=='temperature':v=temperature
        elif kind=='hdd':v=np.maximum(term['base_c']-temperature,0)
        elif kind=='cdd':v=np.maximum(temperature-term['base_c'],0)
        elif kind=='weekday':v=(local.dt.dayofweek.to_numpy()==term['day']).astype(float)
        elif kind in ('sin_year','cos_year'):
            # Site-local annual phase with the actual calendar-year length.
            doy=local.dt.dayofyear.to_numpy()-1
            length=np.where(local.dt.is_leap_year.to_numpy(),366.,365.)
            phase=2*np.pi*term.get('harmonic',1)*doy/length
            v=np.sin(phase) if kind=='sin_year' else np.cos(phase)
        elif kind=='sin_hour':v=np.sin(2*np.pi*term['harmonic']*(local.dt.hour.to_numpy()+local.dt.minute.to_numpy()/60)/24)
        elif kind=='cos_hour':v=np.cos(2*np.pi*term['harmonic']*(local.dt.hour.to_numpy()+local.dt.minute.to_numpy()/60)/24)
        elif kind in ('independent','bms'):v=pd.to_numeric(f[term['column']],errors='coerce').to_numpy(float)
        else:raise DataError('Unknown regression term')
        values.append(v)
    return np.column_stack(values)


def calendar(cfg):
    terms=[{'kind':'weekday','day':i} for i in range(1,7)] if cfg.get('calendar_features',True) else []
    if cfg['resolution']=='interval' and cfg.get('calendar_features',True):
        for h in (1,2):terms += [{'kind':'sin_hour','harmonic':h},{'kind':'cos_hour','harmonic':h}]
    return terms+[{'kind':'independent','column':c} for c in cfg.get('independent_variables',[])]


def candidates(cfg,weather):
    extra=calendar(cfg);base=[{'kind':'constant'}]
    yield 'constant',base+extra
    if not weather:return
    yield 'linear_temperature',base+[{'kind':'temperature'}]+extra
    points=cfg.get('change_points_c',[12,15,18.3,21,24])
    for t in points:
        yield f'heating_{t}',base+[{'kind':'hdd','base_c':t}]+extra
        yield f'cooling_{t}',base+[{'kind':'cdd','base_c':t}]+extra
    for h in points:
        for c in points:
            if h<=c:yield f'heat_cool_{h}_{c}',base+[{'kind':'hdd','base_c':h},{'kind':'cdd','base_c':c}]+extra


def ols(x,y,physical_terms=None):
    active=np.std(x,axis=0)>1e-10;active[0]=True
    xa=x[:,active];scale=np.maximum(np.linalg.norm(xa,axis=0),1e-12);z=xa/scale
    if len(y)<=z.shape[1]*4 or np.linalg.matrix_rank(z)!=z.shape[1] or np.linalg.cond(z)>10000:
        raise DataError('Regression effects are not identifiable; insufficient independent variation or collinearity')
    small=np.linalg.lstsq(z,y,rcond=None)[0]/scale
    coef=np.zeros(x.shape[1]);coef[active]=small
    if physical_terms and any(coef[i]<-1e-8 for i,t in enumerate(physical_terms) if t['kind'] in ('hdd','cdd')):
        raise DataError('Negative heating/cooling weather slope rejected')
    return coef,active


def fit(rows,cfg,kind='baseline',bms_columns=None):
    check_cfg(cfg);f=frame(rows,cfg,energy=True)
    cutoff=stamp(cfg['intervention_timestamp'],cfg['timezone'])
    if kind=='baseline' and (f.period_end>cutoff).any():raise DataError('Baseline contains post-intervention periods')
    if kind=='bms_post' and (f.timestamp<cutoff).any():raise DataError('BMS post model needs post-intervention calibration')
    if kind=='bms_post' and cfg.get('review',{}).get('bms_semantics_verified') is not True:raise DataError('Verify BMS point meanings and aggregation before fitting its electrical model')
    f=f[f.energy_kwh.notna()].copy()
    weather=f.temperature_c.notna().all()
    if not weather and not cfg.get('allow_meter_only_baseline',False):raise DataError('Weather is missing; supply matched weather or explicitly allow a meter-only baseline')
    required=list(cfg.get('independent_variables',[]))+(list(bms_columns or []) if kind=='bms_post' else [])
    for col in required:
        if col not in f:raise DataError(f'Missing regressor: {col}')
        f[col]=pd.to_numeric(f[col],errors='coerce')
    if required:f=f[np.isfinite(f[required].to_numpy(float)).all(axis=1)].copy()
    f=f.reset_index(drop=True)
    if len(f):
        span=(f.period_end.max()-f.timestamp.min()).total_seconds()
        expected=int(round(span/(60*cfg['interval_minutes']))) if cfg['resolution']=='interval' else len(pd.date_range(f.timestamp.min().tz_convert(cfg['timezone']),f.period_end.max().tz_convert(cfg['timezone']),freq='D',inclusive='left'))
    else:expected=0
    minimum=cfg.get('minimum_baseline_periods',90) if kind=='baseline' else cfg.get('minimum_post_calibration_periods',30)
    if len(f)<minimum:raise DataError(f'Need at least {minimum} valid {kind} calibration periods; found {len(f)}')
    split=int(len(f)*.8);training=f.iloc[:split];holdout=f.iloc[split:]
    if len(holdout)<8:raise DataError('Need at least eight withheld periods')
    options=list(candidates(cfg,weather)) if kind=='baseline' else [('bms_electrical_linear',[{'kind':'constant'}]+([{'kind':'temperature'}] if weather else [])+[{'kind':'bms','column':c} for c in required if c in (bms_columns or [])]+calendar({**cfg,'calendar_features':cfg.get('post_calendar_features',False)}))]
    scores=[]
    for name,terms in options:
        x=matrix(training,terms,cfg);y=(training.energy_kwh/training.hours).to_numpy(float)
        losses=[]
        try:
            for fraction in (.5,.65,.8):
                cut=int(len(training)*fraction);stop=min(len(training),cut+max(8,int(len(training)*.15)))
                coef,_=ols(x[:cut],y[:cut],terms if kind=='baseline' else None)
                predicted=x[cut:stop]@coef*training.hours.to_numpy()[cut:stop]
                losses.append(float(np.mean((training.energy_kwh.to_numpy()[cut:stop]-predicted)**2)))
            coef,active=ols(x,y,terms if kind=='baseline' else None)
            pred=matrix(holdout,terms,cfg)@coef*holdout.hours.to_numpy()
            if not np.isfinite(pred).all() or (pred<0).any():raise DataError('Unphysical held-out prediction')
            validation=metrics(holdout.energy_kwh,pred)
            scores.append({'family':name,'terms':terms,'cv_rmse_kwh':float(np.sqrt(np.mean(losses))),'holdout':validation,'holdout_residual_kwh':(holdout.energy_kwh.to_numpy()-pred).tolist()})
        except DataError:continue
    if not scores:raise DataError('No identifiable regression model; collect independent operations/weather and enough observations')
    best=min(scores,key=lambda s:s['cv_rmse_kwh'])
    # Selection used training-only folds, never the outer holdout. Final fit uses all baseline/calibration data.
    x=matrix(f,best['terms'],cfg);y=(f.energy_kwh/f.hours).to_numpy(float)
    coef,active=ols(x,y,best['terms'] if kind=='baseline' else None)
    gates=[];review=cfg.get('review',{})
    for key in ('meter_boundary_verified','timestamps_verified','intervention_confirmed','independent_variables_verified','static_factors_reviewed','mv_plan_approved'):
        if review.get(key) is not True:gates.append(key+'_required')
    if not weather:gates.append('weather_not_modeled')
    coverage_days=(f.period_end.max()-f.timestamp.min()).total_seconds()/86400
    if kind=='baseline' and coverage_days<365:gates.append('baseline_less_than_one_year')
    if expected and len(f)/expected<.9:gates.append('calibration_calendar_coverage_below_90_percent')
    validation=best['holdout'];limits=cfg.get('validation_limits',{'cvrmse_pct':20,'abs_nmbe_pct':5})
    if validation['cvrmse_pct']>limits['cvrmse_pct'] or abs(validation['nmbe_pct'])>limits['abs_nmbe_pct']:gates.append('heldout_model_error_requires_review')
    if cfg.get('synthetic'):gates.append('synthetic_not_site_evidence')
    # Moving contiguous blocks; joint predictors/targets keep weather/operation correlation.
    block=7 if cfg['resolution']=='daily' else max(1,int(24*60/cfg['interval_minutes']))
    rng=np.random.default_rng(199712);ensemble=[]
    def contiguous_starts(part,length):
        if len(part)<length:return []
        if length==1:return list(range(len(part)))
        adjacent=(part.period_end.iloc[:-1].astype('int64').to_numpy()==part.timestamp.iloc[1:].astype('int64').to_numpy()).astype(int)
        return np.flatnonzero(np.convolve(adjacent,np.ones(length-1,dtype=int),mode='valid')==length-1).tolist()
    starts_available=contiguous_starts(f,block)
    for _ in range(64):
        if not starts_available:break
        starts=rng.choice(starts_available,size=(len(f)+block-1)//block)
        idx=np.concatenate([np.arange(s,min(s+block,len(f))) for s in starts])[:len(f)]
        try:
            c,_=ols(x[idx],y[idx],best['terms'] if kind=='baseline' else None);ensemble.append(c.tolist())
        except DataError:continue
    artifact={'schema':'option_c.model.v1','kind':kind,'family':best['family'],'terms':best['terms'],'coef_average_kw':coef.tolist(),
              'active_features':active.tolist(),'feature_min':x.min(axis=0).tolist(),'feature_max':x.max(axis=0).tolist(),
              'bootstrap_coef':ensemble,'holdout_residual_kwh':best['holdout_residual_kwh'],'holdout_metrics':validation,
              'uncertainty_residual_block_size':block,'uncertainty_residual_block_starts':contiguous_starts(holdout,block),
              'in_sample_metrics':metrics(f.energy_kwh,x@coef*f.hours.to_numpy(),int(active.sum())),
              'selection_candidates':[{'family':s['family'],'cv_rmse_kwh':s['cv_rmse_kwh']} for s in scores],
              'validation_scope':'Chronological last 20%, evaluated before final all-calibration refit; model selection on inner training folds only',
              'fit_start':f.timestamp.min().isoformat(),'fit_end_exclusive':f.period_end.max().isoformat(),'valid_periods':len(f),
              'coverage_days':coverage_days,'config':cfg,'bms_columns':list(bms_columns or []),'gates':gates,'input_sha256':digest(records(f)),
              'expected_calendar_periods':expected,'valid_calendar_fraction':float(len(f)/expected) if expected else 0.,
              'status':'fitted_requires_review','source_protocol':'IPMVP December 1997, printed pp.59-62; eqs.4.3.1-4 and 4.3.7-8'}
    artifact['model_id']=digest(artifact)
    return artifact


def verify(model):
    if model.get('schema')!='option_c.model.v1' or digest({k:v for k,v in model.items() if k!='model_id'})!=model.get('model_id'):
        raise DataError('Model checksum/schema mismatch')


def predict(model,f):
    verify(model);x=matrix(f,model['terms'],model['config'])
    good=np.isfinite(x).all(axis=1)
    pred=x@np.array(model['coef_average_kw'])*f.hours.to_numpy()
    good &= np.isfinite(pred)&(pred>=0)
    lo,hi=np.array(model['feature_min']),np.array(model['feature_max'])
    outside=np.any((x<lo-1e-8)|(x>hi+1e-8),axis=1)&good
    # Post estimates are unavailable outside calibration support; baseline forecasts remain labeled extrapolation.
    if model['kind']=='bms_post':good &= ~outside
    pred[~good]=np.nan
    return pred,good,outside,x


def uncertainty_samples(model,x,hours,repeats=256):
    coefs=model.get('bootstrap_coef',[]);res=np.array(model.get('holdout_residual_kwh',[]))
    if len(coefs)<30 or len(res)<8:return None
    rng=np.random.default_rng(199713);block=model.get('uncertainty_residual_block_size',7);starts_available=model.get('uncertainty_residual_block_starts',[]);samples=[]
    if not starts_available:return None
    centered=model.get('uncertainty_centering')=='centered_bootstrap_and_holdout_residuals'
    coefficient_center=np.mean(coefs,axis=0) if centered else None
    if centered:
        # Bias is a separate validation failure, not extra claimed avoided energy.
        # Center over admissible blocks, including their edge sampling frequency.
        block_values=np.concatenate([res[s:s+block] for s in starts_available])
        res=res-float(block_values.mean())
    for _ in range(repeats):
        starts=rng.choice(starts_available,size=(len(x)+block-1)//block)
        noise=np.concatenate([res[s:s+block] for s in starts])[:len(x)]
        c=np.array(coefs[rng.integers(len(coefs))])
        if centered:c=c-coefficient_center+np.array(model['coef_average_kw'])
        samples.append(float(np.sum((x@c)*hours+noise)))
    return np.array(samples)
