"""Calendar baselines with robust alternatives and identical chronological validation."""
import numpy as np
import pandas as pd
from .data import DataError,check_cfg,digest,frame,records,stamp
from .regression import calendar,matrix,metrics,ols


def coefficients(x,y,estimator):
    if estimator=='ols':return ols(x,y)
    active=np.std(x,axis=0)>1e-10;active[0]=True
    xa=x[:,active]
    norm=np.maximum(np.linalg.norm(xa,axis=0),1e-12);z=xa/norm
    if len(y)<=4*xa.shape[1] or np.linalg.matrix_rank(z)!=xa.shape[1] or np.linalg.cond(z)>10000:raise DataError('Insufficient independent baseline observations')
    if estimator=='weekday_median':
        small=np.zeros(xa.shape[1]);reference=np.all(xa[:,1:]==0,axis=1)
        small[0]=np.median(y[reference]) if reference.any() else np.median(y)
        for j in range(1,xa.shape[1]):small[j]=np.median(y[xa[:,j]==1])-small[0]
    elif estimator=='huber':
        small=np.zeros(xa.shape[1]);small[0]=np.median(y)
        for _ in range(60):
            residual=y-xa@small;scale=max(float(1.4826*np.median(np.abs(residual-np.median(residual)))),1e-8,float(np.median(np.abs(y)))*1e-6)
            weights=np.minimum(1.,1.35*scale/np.maximum(np.abs(residual),1e-15));root=np.sqrt(weights)
            updated=np.linalg.lstsq(z*root[:,None],y*root,rcond=None)[0]/norm
            if np.max(np.abs(updated-small))<1e-7*(1+np.max(np.abs(small))):small=updated;break
            small=updated
    else:raise DataError('Unknown baseline estimator')
    coef=np.zeros(x.shape[1]);coef[active]=small
    return coef,active


def fit(rows,cfg):
    check_cfg(cfg)
    if cfg['resolution']!='daily' or cfg.get('independent_variables'):raise DataError('Resilient batch baseline supports daily calendar models only; use advanced CLI for other regressors')
    f=frame(rows,cfg,energy=True);cutoff=stamp(cfg['intervention_timestamp'],cfg['timezone'])
    if (f.period_end>cutoff).any():raise DataError('Baseline includes controlled periods')
    f=f[f.energy_kwh.notna()].reset_index(drop=True)
    opt=cfg.get('baseline_options',{});minimum=int(opt.get('minimum_training_days',60));holdout_days=int(opt.get('holdout_days',56));fold_days=int(opt.get('fold_days',28))
    if minimum<35 or holdout_days<14 or fold_days<7:raise DataError('Baseline validation settings below supported floors')
    if len(f)<cfg.get('minimum_baseline_periods',90):raise DataError('Insufficient valid baseline periods')
    def before(t,days):return (t.tz_convert(cfg['timezone'])-pd.DateOffset(days=days)).tz_convert('UTC')
    end=f.period_end.max();outer=before(end,holdout_days)
    train=f[f.timestamp<outer];test=f[f.timestamp>=outer]
    if len(train)<minimum or len(test)<holdout_days*.8:raise DataError('Insufficient training or shared held-out baseline coverage')
    terms=[{'kind':'constant'}]+calendar(cfg)
    windows=opt.get('windows_days',[None,365,180]);estimators=opt.get('estimators',['ols','huber','weekday_median'])
    if not windows or not estimators or any(w is not None and (not isinstance(w,int) or w<90) for w in windows):raise DataError('Baseline windows must be null or at least 90 days')
    origins=[]
    for i in (3,2,1):
        a=before(outer,i*fold_days);b=before(outer,(i-1)*fold_days)
        if len(train[train.timestamp<a])>=minimum and len(train[(train.timestamp>=a)&(train.timestamp<b)])>=fold_days*.8:origins.append((a,b))
    if len(origins)<2:raise DataError('Need at least two complete chronological inner validation folds')
    def prior(data,origin,days):
        g=data[data.timestamp<origin]
        if days is not None:g=g[g.timestamp>=before(origin,days)]
        if len(g)<minimum:raise DataError('Insufficient lookback training observations')
        return g
    def solve(data,estimator,model_terms):
        if any(t['kind']=='sin_year' for t in model_terms):
            days=len(pd.date_range(data.timestamp.min().tz_convert(cfg['timezone']),data.period_end.max().tz_convert(cfg['timezone']),freq='D',inclusive='left'))
            if days<365 or len(data)/days<.9:raise DataError('Annual calendar terms need a full year with 90% coverage before every validation origin')
        return coefficients(matrix(data,model_terms,cfg),(data.energy_kwh/data.hours).to_numpy(float),estimator)
    def predict(data,coef,model_terms):
        p=matrix(data,model_terms,cfg)@coef*data.hours.to_numpy()
        if not np.isfinite(p).all() or (p<0).any():raise DataError('Unphysical candidate baseline prediction')
        return p
    scores=[];finals={}
    specs=[(d,e,False) for d in windows for e in estimators]
    if opt.get('annual_calendar_candidates',False):
        specs += [(d,e,True) for d in windows for e in estimators if e!='weekday_median' and (d is None or d>=365)]
    for days,estimator,annual in specs:
        model_terms=terms+([{'kind':'sin_year','harmonic':1},{'kind':'cos_year','harmonic':1}] if annual else [])
        name=estimator+'_'+('all' if days is None else str(days))+('_annual' if annual else '')
        entry={'name':name,'estimator':estimator,'window_days':days,'status':'unavailable'}
        try:
            losses=[]
            for a,b in origins:
                past=prior(train,a,days);valid=train[(train.timestamp>=a)&(train.timestamp<b)]
                coef,_=solve(past,estimator,model_terms);p=predict(valid,coef,model_terms)
                losses.append(float(np.mean((valid.energy_kwh.to_numpy()-p)**2)))
            coef,_=solve(prior(train,outer,days),estimator,model_terms)
            p=matrix(test,model_terms,cfg)@coef*test.hours.to_numpy()
            if not np.isfinite(p).all():raise DataError('Non-finite candidate prediction')
            v=metrics(test.energy_kwh,p)
            limits=cfg.get('validation_limits',{'cvrmse_pct':20,'abs_nmbe_pct':5})
            final=prior(f,end,days);final_coef,active=solve(final,estimator,model_terms)
            finals[name]=(final,final_coef,active,model_terms)
            entry.update(status='evaluated',terms=model_terms,cv_rmse_kwh=float(np.sqrt(np.mean(losses))),holdout=v,
                validation_passed=bool((p>=0).all()) and v['cvrmse_pct']<=limits['cvrmse_pct'] and abs(v['nmbe_pct'])<=limits['abs_nmbe_pct'],
                holdout_residual_kwh=(test.energy_kwh.to_numpy()-p).tolist(),final_coef=final_coef.tolist())
        except (DataError,np.linalg.LinAlgError) as exc:entry['reason']=str(exc)
        scores.append(entry)
    available=[s for s in scores if s['status']=='evaluated']
    if not available:raise DataError('No baseline candidate passed chronological fitting checks')
    # Selection is fixed before consulting the outer validation errors.
    best=min(available,key=lambda s:(s['cv_rmse_kwh'],s['name']))
    final,coef,active,terms=finals[best['name']];gates=['weather_not_modeled']
    for key in ('meter_boundary_verified','timestamps_verified','intervention_confirmed','independent_variables_verified','static_factors_reviewed','mv_plan_approved'):
        if cfg.get('review',{}).get(key) is not True:gates.append(key+'_required')
    if not best['validation_passed']:gates.append('heldout_model_error_requires_review')
    span=(end-final.timestamp.min()).total_seconds()/86400
    if span<365:gates.append('baseline_less_than_one_year')
    expected=len(pd.date_range(final.timestamp.min().tz_convert(cfg['timezone']),end.tz_convert(cfg['timezone']),freq='D',inclusive='left'))
    if len(final)/expected<.9:gates.append('calibration_calendar_coverage_below_90_percent')
    if cfg.get('synthetic'):gates.append('synthetic_not_site_evidence')
    def starts(g,length):
        if len(g)<length:return []
        t=g.timestamp.astype('int64').to_numpy();p=g.period_end.astype('int64').to_numpy()
        return [i for i in range(len(g)-length+1) if np.all(p[i:i+length-1]==t[i+1:i+length])]
    block=7;positions=starts(final,block);ensemble=[];rng=np.random.default_rng(199712)
    x=matrix(final,terms,cfg);y=(final.energy_kwh/final.hours).to_numpy(float)
    for _ in range(int(opt.get('bootstrap_repeats',64))):
        if not positions:break
        chosen=rng.choice(positions,size=(len(final)+block-1)//block)
        idx=np.concatenate([np.arange(i,i+block) for i in chosen])[:len(final)]
        try:ensemble.append(coefficients(x[idx],y[idx],best['estimator'])[0].tolist())
        except (DataError,np.linalg.LinAlgError):continue
    artifact={'schema':'option_c.model.v1','kind':'baseline','family':best['name'],'terms':terms,'coef_average_kw':coef.tolist(),
        'active_features':active.tolist(),'feature_min':x.min(axis=0).tolist(),'feature_max':x.max(axis=0).tolist(),
        'bootstrap_coef':ensemble,'holdout_metrics':best['holdout'],'holdout_residual_kwh':best['holdout_residual_kwh'],
        'uncertainty_residual_block_size':block,'uncertainty_residual_block_starts':starts(test,block),
        'in_sample_metrics':metrics(final.energy_kwh,x@coef*final.hours.to_numpy(),int(active.sum())),
        'selection_candidates':scores,'fit_start':final.timestamp.min().isoformat(),'fit_end_exclusive':end.isoformat(),
        'valid_periods':len(final),'coverage_days':span,'expected_calendar_periods':expected,'valid_calendar_fraction':len(final)/expected,
        'config':cfg,'bms_columns':[],'gates':gates,'input_sha256':digest(records(final)),'status':'fitted_requires_review',
        'validation_scope':'Same final held-out calendar dates for every estimator/lookback; rolling inner folds alone select the model. Final refit uses the chosen lookback.',
        'holdout_start':outer.isoformat(),'holdout_end_exclusive':end.isoformat(),
        'source_protocol':'Engineering implementation inspired by IPMVP Option C; not automatic protocol certification',
        'release_checks_enabled':True}
    artifact['model_id']=digest(artifact)
    return artifact


def sensitivity(model,rows):
    cfg=model['config'];f=frame(rows,cfg);output=[]
    for candidate in model['selection_candidates']:
        if candidate['status']!='evaluated':continue
        x=matrix(f,candidate.get('terms',model['terms']),cfg)
        p=x@np.array(candidate['final_coef'])*f.hours.to_numpy()
        output.append({'name':candidate['name'],'validation_passed':candidate['validation_passed'],
            'baseline_kwh':float(p.sum()) if np.isfinite(p).all() and (p>=0).all() else None,
            'cv_rmse_kwh':candidate['cv_rmse_kwh'],'holdout_cvrmse_pct':candidate['holdout']['cvrmse_pct']})
    valid=[r['baseline_kwh'] for r in output if r['validation_passed'] and r['baseline_kwh'] is not None]
    selected=next(r['baseline_kwh'] for r in output if r['name']==model['family'])
    spread=(max(valid)-min(valid))/selected*100 if len(valid)>1 and selected and selected>0 else None
    return {'models':output,'passing_model_spread_pct':spread,'threshold_pct':cfg.get('baseline_options',{}).get('sensitivity_limit_pct',10),
            'note':'Alternative-model totals use identical reporting dates. Selection never uses reported savings. Failed models remain diagnostic only.'}
