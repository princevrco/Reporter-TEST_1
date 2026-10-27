"""Weather-forced empirical thermal dynamics, validated by free-running day rollouts."""
import numpy as np
import pandas as pd
from .data import DataError,digest,records


def features(t,outdoor,cool,heat,hour):
    t=np.asarray(t,float);outdoor=np.asarray(outdoor,float)
    cooling=np.maximum(t-np.asarray(cool,float),0);heating=np.maximum(np.asarray(heat,float)-t,0)
    return np.column_stack([np.ones(len(t)),outdoor-t,-cooling,heating,np.sin(2*np.pi*np.asarray(hour)/24),np.cos(2*np.pi*np.asarray(hour)/24)])


def solve(x,y,alpha):
    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-8);gram=x.T@x+alpha*np.diag(scale*scale);rhs=x.T@y
    lower=np.array([-5.,0.,0.,0.,-5.,-5.]);upper=np.array([5.,1.,5.,5.,5.,5.]);c=np.zeros(6)
    for _ in range(4000):
        old=c.copy()
        for j in range(6):c[j]=np.clip((rhs[j]-gram[j]@c+gram[j,j]*c[j])/gram[j,j],lower[j],upper[j])
        if np.max(abs(c-old))<1e-9:break
    return c


def step(item,t,outdoor,cool,heat,hour,dt):
    c=t if cool is None else cool;h=t if heat is None else heat
    result=t+dt*float(features([t],[outdoor],[c],[h],[hour])[0]@np.asarray(item['coefficients']))
    if not np.isfinite(result) or not -20<=result<=60:raise DataError('Unphysical thermal rollout; no clipping or persistence fallback')
    return result


def rollout_errors(f,coef,timezone,cadence):
    errors=[];naive=[];days=0
    for day,g in f.groupby('date',sort=True):
        left=pd.Timestamp(day,tz=timezone).tz_convert('UTC');right=(pd.Timestamp(day,tz=timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
        # No unobserved gaps are bridged. Allow irregular sampling with valid adjacent pairs.
        if len(g)<2 or g.timestamp.iloc[0]>left+pd.Timedelta(minutes=30) or g.next_timestamp.iloc[-1]<right-pd.Timedelta(minutes=30):continue
        if not np.array_equal(g.next_timestamp.iloc[:-1].astype('int64').to_numpy(),g.timestamp.iloc[1:].astype('int64').to_numpy()):continue
        if g.dt_hours.sum()/(right-left).total_seconds()*3600<.9:continue
        state=float(g.temperature.iloc[0]);initial=state;block=[]
        for r in g.itertuples():
            state=step({'coefficients':coef},state,r.outdoor,r.cool,r.heat,r.hour,r.dt_hours)
            block.append(state-r.next_temperature);naive.append(initial-r.next_temperature)
        errors.extend(block);days+=1
    return errors,naive,days


def fit(raw,groups,weather,timezone,end,cadence=15,maximum_rmse_c=1.5):
    if raw.empty:raise DataError('No complete BMS rows for thermal model')
    f=raw.sort_values('timestamp').copy();f['timestamp']=pd.to_datetime(f.timestamp,utc=True,format='mixed')
    limit=pd.Timestamp(end,tz=timezone).tz_convert('UTC');cutoff=(pd.Timestamp(end,tz=timezone)-pd.DateOffset(days=14)).tz_convert('UTC')
    f=f[f.timestamp<limit].copy();local=f.timestamp.dt.tz_convert(timezone);dates=local.dt.date.astype(str)
    w=weather.set_index('date').temperature_c
    if not set(dates)<=set(w.index):raise DataError('Historical weather missing for thermal training')
    result={'schema':'option_c.weather_thermal.v1','timezone':timezone,'cadence_minutes':cadence,'fit_end_exclusive':end,
            'assets':{},'blocked_assets':{},'mapping':groups,'input_sha256':digest(records(f)), 'weather_sha256':digest(records(weather[weather.date<end])),
            'gates':['empirical_weather_thermal_surrogate_not_verified_causal_dynamics','unverified_actuator_feedback_and_equipment_deadbands','daily_mean_outdoor_temperature_used_as_intraday_forcing'],
            'validation_basis':'Project screening: >=2 complete daily rollouts, <=1.5 C RMSE by default and no worse than persistence; not automatic standards certification.'}
    for asset,spec in groups.items():
        try:
            t=f[spec['temperature']].to_numpy(float);cool=f[spec['cool_setpoint']].to_numpy(float)
            heat=f[spec['heat_setpoint']].to_numpy(float) if 'heat_setpoint' in spec else t.copy()
            a=pd.DataFrame({'timestamp':f.timestamp,'date':dates,'temperature':t,'cool':cool,'heat':heat,
                            'outdoor':dates.map(w).to_numpy(),'hour':local.dt.hour+local.dt.minute/60,
                            'next_temperature':pd.Series(t,index=f.index).shift(-1),'next_timestamp':f.timestamp.shift(-1)})
            a['dt_hours']=(a.next_timestamp-a.timestamp).dt.total_seconds()/3600
            a=a[a.dt_hours.gt(0)&a.dt_hours.le(.5)&a.next_timestamp.lt(limit)].copy()
            required=['temperature','cool','heat','outdoor','next_temperature']
            a=a[np.isfinite(a[required].to_numpy(float)).all(axis=1)].copy()
            train=a[a.next_timestamp<cutoff];test=a[a.timestamp>=cutoff]
            if train.dt_hours.sum()<7*24:raise DataError('Thermal model needs at least seven days of earlier adjacent complete pairs')
            if test.dt_hours.sum()<2*24:raise DataError('Thermal model needs later adjacent pairs for rollout validation')
            x=features(train.temperature,train.outdoor,train.cool,train.heat,train.hour);y=(train.next_temperature-train.temperature).to_numpy()/train.dt_hours.to_numpy()
            active=np.std(x,axis=0)>1e-8;active[0]=True;z=x[:,active]/np.maximum(np.linalg.norm(x[:,active],axis=0),1e-8)
            if not np.isfinite(x).all():raise DataError('Missing thermal pairs remain excluded')
            if not active[2] or np.linalg.matrix_rank(z)!=z.shape[1] or np.linalg.cond(z)>10000:raise DataError('Thermal cooling response not independently identifiable')
            split=int(len(train)*.75);choices=[]
            for alpha in (.0001,.01,.1,1.):
                c=solve(x[:split],y[:split],alpha)
                errs,_,days=rollout_errors(train.iloc[split:],c,timezone,cadence)
                if days:choices.append((float(np.mean(np.square(errs))),alpha))
            if not choices:raise DataError('No earlier complete thermal rollout validation days')
            alpha=min(choices)[1];prior=solve(x,y,alpha);errs,naive,days=rollout_errors(test,prior,timezone,cadence)
            rmse=float(np.sqrt(np.mean(np.square(errs)))) if errs else None
            persistence=float(np.sqrt(np.mean(np.square(naive)))) if naive else None
            passed=days>=2 and rmse is not None and rmse<=maximum_rmse_c and rmse<=persistence
            xa=features(a.temperature,a.outdoor,a.cool,a.heat,a.hour);ya=(a.next_temperature-a.temperature).to_numpy()/a.dt_hours.to_numpy()
            final=solve(xa,ya,alpha)
            result['assets'][asset]={'coefficients':final.tolist(),'frozen_holdout_coefficients':prior.tolist(),'alpha':alpha,
                'training_adjacent_pairs':len(train),'test_adjacent_pairs':len(test),'heldout_rollout_rmse_c':rmse,'persistence_rmse_c':persistence,'heldout_rollout_days':days,'validation_passed':passed,
                'temperature_range_c':[float(train.temperature.min()),float(train.temperature.max())],
                'cooling_setpoint_range_c':[float(train.cool.min()),float(train.cool.max())],
                'heating_setpoint_range_c':[float(train.heat.min()),float(train.heat.max())],
                'outdoor_range_c':[float(train.outdoor.min()),float(train.outdoor.max())]}
        except (DataError,ValueError,np.linalg.LinAlgError) as exc:result['blocked_assets'][asset]=str(exc)
    result['all_assets_validated']=set(result['assets'])==set(groups) and all(a['validation_passed'] for a in result['assets'].values())
    result['model_id']=digest(result)
    return result


def verify(model):
    if digest({k:v for k,v in model.items() if k!='model_id'})!=model.get('model_id'):raise DataError('Thermal model checksum mismatch')
