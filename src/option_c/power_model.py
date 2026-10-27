"""Weather-aware daily meter/electrical models with chronological validation."""
import numpy as np
import pandas as pd
from .data import DataError,digest,records
from .regression import metrics


def design(f,kind):
    t=f.temperature_c.to_numpy(float);h=f.hours.to_numpy(float);weekend=pd.to_datetime(f.date).dt.dayofweek.to_numpy()>=5
    cols={'weekday_hours':h*(~weekend),'weekend_hours':h*weekend,
          'outdoor_heating_degree_hours':np.maximum(18.3-t,0)*h,
          'outdoor_cooling_degree_hours':np.maximum(t-18.3,0)*h}
    if kind=='bms':
        for c in ('cool_degree_hours','heat_degree_hours','cool_proxy_hours','heat_proxy_hours'):cols[c]=f[c].to_numpy(float)
    return np.column_stack(list(cols.values())),list(cols)


def solve(x,y,alpha):
    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-10);z=x/scale
    gram=z.T@z+alpha*np.eye(x.shape[1]);rhs=z.T@y;c=np.zeros(x.shape[1])
    for _ in range(4000):
        prev=c.copy()
        for j in range(len(c)):c[j]=max(0.,(rhs[j]-gram[j]@c+gram[j,j]*c[j])/gram[j,j])
        if np.max(abs(c-prev))<1e-8*(1+np.max(abs(c))):break
    return c/scale


def fit(frame,kind,timezone,end,minimum_days=40,holdout_days=14):
    f=frame[frame.date<end].sort_values('date').copy()
    f=f[np.isfinite(pd.to_numeric(f.energy_kwh,errors='coerce'))]
    if 'quality' in f:f=f[f.quality.eq('good')]
    cutoff=str((pd.Timestamp(end)-pd.DateOffset(days=holdout_days)).date());train=f[f.date<cutoff];test=f[f.date>=cutoff]
    minimum=max(minimum_days,40 if kind=='bms' else 28)
    if len(train)<minimum or len(test)<8:raise DataError(f'{kind} energy model needs {minimum} training days and 8 later test days; found {len(train)} and {len(test)}')
    x,names=design(f,kind);active=np.max(abs(x),axis=0)>1e-10
    tr,names=design(train,kind);te,_=design(test,kind)
    if not np.isfinite(x).all():raise DataError('Incomplete power predictors; rows are not imputed')
    # Select only on earlier expanding folds, using the same fixed feature family.
    specs=[]
    for alpha in (.0001,.01,.1,1.,10.):
        losses=[]
        for frac in (.6,.75,.9):
            k=int(len(train)*frac);stop=min(len(train),k+max(3,int(len(train)*.1)))
            if k<20 or stop<=k:continue
            c=solve(tr[:k],train.energy_kwh.to_numpy()[:k],alpha)
            losses.extend((tr[k:stop]@c-train.energy_kwh.to_numpy()[k:stop]).tolist())
        if losses:specs.append((float(np.mean(np.square(losses))),alpha))
    if not specs:raise DataError('Insufficient earlier validation folds')
    alpha=min(specs)[1];prior=solve(tr,train.energy_kwh.to_numpy(),alpha);held=te@prior
    val=metrics(test.energy_kwh,held);passed=val['cvrmse_pct']<=20 and abs(val['nmbe_pct'])<=5
    final=solve(x,f.energy_kwh.to_numpy(),alpha);gates=['meter_boundary_unverified','historical_development_backtest_not_prospective_validation','unmeasured_load_changes_not_adjusted']
    if 'coverage_fraction' in f and f.coverage_fraction.lt(1.-1e-8).any():gates.append('partial_subdaily_feature_coverage_requires_review')
    if kind=='bms':gates+=['thermal_calls_are_proxies_not_verified_feedback','pooled_asset_response_not_individual_power_calibration']
    if not passed:gates.append('heldout_power_validation_failed')
    if 'quality_review_required' in f and f.quality_review_required.astype(str).str.lower().isin(['true','1']).any():gates.append('source_meter_quality_requires_review')
    model={'schema':'option_c.power.v1','kind':kind,'timezone':timezone,'coefficients':final.tolist(),
           'frozen_holdout_coefficients':prior.tolist(),'features':names,'alpha':alpha,
           'training_days':len(train),'holdout_days':len(test),'holdout_start':cutoff,'fit_end_exclusive':end,
           'holdout_metrics':val,'validation_passed':passed,'gates':gates,
           'feature_min':tr.min(axis=0).tolist(),'feature_max':tr.max(axis=0).tolist(),
           'holdout_residual_kwh':(test.energy_kwh.to_numpy()-held).tolist(),'input_sha256':digest(records(f)),
           'coefficient_scope':'Empirical nonnegative pooled associations; not independently identified equipment kW.'}
    model['model_id']=digest(model)
    withheld={d:float(v) for d,v in zip(test.date,held)}
    reconstructed=[{'date':d,'meter_kwh':float(y),'reconstructed_kwh':float(v),'heldout_kwh':withheld.get(d),
                    'meter_minus_reconstruction_kwh':float(y-v)} for d,y,v in zip(f.date,f.energy_kwh,x@final)]
    return model,reconstructed


def verify(model):
    if digest({k:v for k,v in model.items() if k!='model_id'})!=model.get('model_id'):raise DataError('Power model checksum mismatch')


def predict(model,frame):
    verify(model);x,names=design(frame,model['kind'])
    if names!=model['features'] or not np.isfinite(x).all():raise DataError('Missing or incompatible power predictors')
    values=x@np.array(model['coefficients']);outside=((x<np.array(model['feature_min'])-1e-8)|(x>np.array(model['feature_max'])+1e-8)).any(axis=1)
    return [{'date':str(d),'predicted_kwh':float(v),'outside_training_support':bool(o)} for d,v,o in zip(frame.date,values,outside)]
