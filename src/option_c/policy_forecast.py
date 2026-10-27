"""Same-weather old/optimized policy rollouts and separately labeled projected savings."""
import numpy as np
import pandas as pd
from .data import DataError,digest,records
from . import thermal_model,power_model
from .strict_history import daily


def validate(policy,assets,kind):
    if policy.get('kind')!=kind or not isinstance(policy.get('reference'),str) or not policy['reference'].strip() or policy['reference'].startswith('REPLACE'):
        raise DataError('Policy needs matching kind and an explicit non-placeholder reference')
    if set(policy.get('assets',{}))!=set(assets):raise DataError('Policy must define exactly the fitted asset cohort')
    for asset,p in policy['assets'].items():
        required={'weekdays','occupied_start','occupied_end','occupied_cooling_c','unoccupied_cooling_c','occupied_heating_c','unoccupied_heating_c','control_margin_c'}
        if not required<=set(p):raise DataError('Policy schedule/setpoints/margin missing for '+asset)
        if not isinstance(p['weekdays'],list) or not p['weekdays'] or any(type(d) is not int or not 0<=d<=6 for d in p['weekdays']):raise DataError('Policy weekdays must use Monday=0 through Sunday=6')
        for c in ('occupied_start','occupied_end'):
            try:
                hour,minute=map(int,p[c].split(':'))
                if not 0<=hour<24 or not 0<=minute<60:raise ValueError()
            except (ValueError,AttributeError):raise DataError('Policy hours must be HH:MM')
        for c in ('occupied_cooling_c','unoccupied_cooling_c','occupied_heating_c','unoccupied_heating_c'):
            v=p[c]
            if v!='OFF' and (isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) or not -20<=v<=60):raise DataError('Setpoints must be numeric Celsius or explicit OFF; null is not a policy')
        if isinstance(p['control_margin_c'],bool) or not isinstance(p['control_margin_c'],(int,float)) or not np.isfinite(p['control_margin_c']) or not 0<=p['control_margin_c']<=5:raise DataError('Explicit control_margin_c must be in [0,5]')
        for prefix in ('occupied','unoccupied'):
            h=p[prefix+'_heating_c'];c=p[prefix+'_cooling_c']
            if h!='OFF' and c!='OFF' and h>c:raise DataError('Heating setpoint exceeds cooling setpoint')
    return digest(policy)


def schedule(t,p):
    now=t.hour*60+t.minute;start=sum(v*k for v,k in zip(map(int,p['occupied_start'].split(':')),(60,1)));end=sum(v*k for v,k in zip(map(int,p['occupied_end'].split(':')),(60,1)))
    weekday=t.dayofweek
    if start==end:occupied=weekday in p['weekdays']
    elif start<end:occupied=weekday in p['weekdays'] and start<=now<end
    else:
        if now<end:weekday=(t-pd.DateOffset(days=1)).dayofweek
        occupied=weekday in p['weekdays'] and (now>=start or now<end)
    prefix='occupied' if occupied else 'unoccupied';c=p[prefix+'_cooling_c'];h=p[prefix+'_heating_c'];margin=p['control_margin_c']
    return (None if c=='OFF' else float(c)+margin,None if h=='OFF' else float(h)-margin)


def initial_state(raw,groups,thermal,weather,start,provided=None):
    tz=thermal['timezone'];a=pd.Timestamp(start,tz=tz).tz_convert('UTC')
    if provided is not None:
        if not provided.get('reference') or provided['reference'].startswith('REPLACE') or not provided.get('timestamp') or pd.Timestamp(provided['timestamp']).tzinfo is None or pd.Timestamp(provided['timestamp']).tz_convert('UTC')!=a:raise DataError('Initial state requires timestamp exactly at forecast start and reference')
        states=provided.get('temperatures_c',{})
        if set(states)!=set(groups) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not -20<=v<=60 for v in states.values()):raise DataError('Initial temperatures must cover all assets without missing values')
        return states,{'kind':'user_supplied','reference':provided['reference'],'timestamp':a.isoformat()}
    if raw.empty:raise DataError('No complete BMS initial state; supply an explicit forecast-start state')
    last=raw.sort_values('timestamp').iloc[-1];t=pd.Timestamp(last.timestamp);cad=thermal['cadence_minutes']
    delta=(a-t).total_seconds()/3600
    if not 0<delta<=.5:raise DataError('BMS does not reach the interval immediately before forecast start; supply explicit initial-state JSON, do not hold stale temperatures')
    day=str(t.tz_convert(tz).date());w=weather.set_index('date').temperature_c
    if day not in w:raise DataError('Weather missing at thermal initialization')
    states={}
    for asset,g in groups.items():
        item=thermal['assets'].get(asset)
        if not item or not item['validation_passed']:raise DataError('Cannot initialize unvalidated thermal asset '+asset)
        hour=t.tz_convert(tz).hour+t.tz_convert(tz).minute/60
        states[asset]=thermal_model.step(item,float(last[g['temperature']]),float(w[day]),float(last[g['cool_setpoint']]),float(last[g['heat_setpoint']]) if 'heat_setpoint' in g else None,hour,delta)
    return states,{'kind':'one_step_model_projection_from_last_complete_observation','timestamp':a.isoformat(),'source_timestamp':t.isoformat()}


def rollout(power,thermal,policy,weather,start,end,initial,weather_basis,rate=.12,external=None,evidence=None):
    power_model.verify(power);thermal_model.verify(thermal)
    if not np.isfinite(rate) or rate<0:raise DataError('Invalid energy rate')
    groups=thermal['mapping'];kind=policy.get('kind');policy_hash=validate(policy,groups,kind)
    if kind not in ('old_policy','optimized_policy'):raise DataError('Unknown scenario policy kind')
    if not isinstance(weather_basis,str) or not weather_basis.strip():raise DataError('Declare future weather forecast/scenario basis')
    tz=power['timezone'];a=pd.Timestamp(start,tz=tz);b=pd.Timestamp(end,tz=tz)
    if a!=a.normalize() or b!=b.normalize() or b<=a:raise DataError('Policy dates must be increasing local dates')
    if a.date()<pd.Timestamp(power['fit_end_exclusive']).date() or a.date()<pd.Timestamp(thermal['fit_end_exclusive']).date():raise DataError('Independent policy forecast must follow electrical and thermal calibration')
    if set(initial)!=set(groups):raise DataError('Initial state asset cohort mismatch')
    days=pd.date_range(a,b,freq='D',inclusive='left').date.astype(str).tolist();w=weather.set_index('date').temperature_c
    if not set(days)<=set(w.index):raise DataError('Future policy weather is missing requested dates')
    cad=thermal['cadence_minutes'];grid=pd.date_range(a,b,freq=f'{cad}min',inclusive='left').tz_convert('UTC')
    context={'weather':{d:float(w[d]) for d in days},'initial':initial,'start':start,'end_exclusive':end,'weather_basis':weather_basis}
    conditions_id=digest(context);temperatures=None
    if external is not None:
        evidence=evidence or {}
        for key,value in [('policy_sha256',policy_hash),('conditions_sha256',conditions_id)]:
            if evidence.get(key)!=value:raise DataError('External trajectory '+key+' mismatch')
        if not evidence.get('model_id') or not evidence.get('validation_reference'):raise DataError('External Neural ODE/STGNN trajectory needs model_id and validation_reference')
        if not {'timestamp','asset_id','predicted_zone_temperature_c'}<=set(external):raise DataError('External trajectory requires timestamp,asset_id,predicted_zone_temperature_c')
        q=external.copy()
        if any(pd.Timestamp(t).tzinfo is None for t in q.timestamp):raise DataError('External trajectory timestamps need explicit offsets')
        q['timestamp']=pd.to_datetime(q.timestamp,utc=True,format='mixed')
        if q.duplicated(['timestamp','asset_id']).any() or set(q.asset_id)!=set(groups):raise DataError('Duplicate/mismatched external trajectory assets')
        temperatures={}
        for asset,g in q.groupby('asset_id'):
            g=g.sort_values('timestamp');v=pd.to_numeric(g.predicted_zone_temperature_c,errors='coerce')
            if g.timestamp.tolist()!=list(grid) or not np.isfinite(v).all() or not v.between(-20,60).all():raise DataError('External trajectory must be complete and finite; no missing rows or interpolation')
            if abs(v.iloc[0]-initial[asset])>1e-6:raise DataError('External trajectory initial state differs from common scenario initial state')
            temperatures[asset]=v.to_numpy(float)
    elif not thermal['all_assets_validated']:raise DataError('Internal thermal rollout validation failed; use validated external trajectories or more complete training data')
    state={k:float(v) for k,v in initial.items()};rows=[];trajectory=[];outside=set()
    for i,t in enumerate(grid):
        local=t.tz_convert(tz);day=str(local.date());r={'timestamp':t}
        for asset,g in groups.items():
            item=thermal['assets'].get(asset);c,h=schedule(local,policy['assets'][asset]);value=float(temperatures[asset][i]) if temperatures is not None else state[asset]
            r[g['temperature']]=value
            # OFF contributes zero thermal demand, not zero electrical background load.
            r[g['cool_setpoint']]=value if c is None else c
            if 'heat_setpoint' in g:r[g['heat_setpoint']]=value if h is None else h
            elif h is not None:raise DataError('Heating policy supplied without calibrated heating signal: '+asset)
            if item:
                if not item['temperature_range_c'][0]-1<=value<=item['temperature_range_c'][1]+1:outside.add(asset)
                if c is not None and not item['cooling_setpoint_range_c'][0]<=c<=item['cooling_setpoint_range_c'][1]:outside.add(asset)
            trajectory.append({'timestamp':t.isoformat(),'asset_id':asset,'predicted_zone_temperature_c':value,
                'cooling_threshold_c':c,'heating_threshold_c':h,'thermal_cooling_proxy':bool(c is not None and value>c),'thermal_heating_proxy':bool(h is not None and value<h)})
            if temperatures is None:state[asset]=thermal_model.step(item,value,float(w[day]),c,h,local.hour+local.minute/60,cad/60)
        rows.append(r)
    frame,coverage=daily(pd.DataFrame(rows),groups,weather,tz,cad,1.)
    estimates=power_model.predict(power,frame)
    if len(estimates)!=len(days):raise DataError('Incomplete simulated forecast days')
    value=sum(r['predicted_kwh'] for r in estimates)
    gates=list(power['gates'])+list(thermal['gates'])+['conditional_policy_projection_not_verified_causal_savings','future_weather_is_supplied_forecast_or_scenario','policy_control_margin_is_an_explicit_assumption','complete_projection_uncertainty_budget_required']
    if not policy.get('reviewed'):gates.append('policy_assumptions_unverified')
    if outside or any(r['outside_training_support'] for r in estimates):gates.append('scenario_outside_training_support')
    if len(days)>1:gates.append('forecast_horizon_exceeds_validated_one_day_thermal_rollouts')
    if external is not None:gates.append('external_temperature_model_validation_evidence_requires_review')
    result={'schema':'option_c.policy_forecast.v1','scenario_kind':kind,'start':start,'end_exclusive':end,'days':len(days),
        'electrical_model_id':power['model_id'],'thermal_model_id':thermal['model_id'],'policy_sha256':policy_hash,'conditions_sha256':conditions_id,
        'forecast_kwh':value,'forecast_energy_cad':value*rate,'rate_cad_per_kwh':rate,'gates':sorted(set(gates)),
        'status':'diagnostic_projection_validation_failed' if not power['validation_passed'] else 'conditional_projection_requires_review','uncertainty_status':'Not quantified for the joint thermal/electrical/policy counterfactual; no verified savings claim','initial_states_c':initial,'weather_basis':weather_basis,
        'temperature_source':'external_model' if external is not None else 'weather_forced_empirical_thermal_model','daily':estimates,
        'scope':'Model projection; flat energy rate excludes demand charges, taxes and fixed charges.'}
    result['forecast_id']=digest(result)
    return result,pd.DataFrame(trajectory)


def compare(old,optimized,rate=.12):
    if old.get('scenario_kind')!='old_policy' or optimized.get('scenario_kind')!='optimized_policy':raise DataError('Compare old_policy against optimized_policy')
    for f in (old,optimized):
        if digest({k:v for k,v in f.items() if k!='forecast_id'})!=f.get('forecast_id'):raise DataError('Policy forecast checksum mismatch')
    for k in ('electrical_model_id','thermal_model_id','conditions_sha256','start','end_exclusive','days'):
        if old.get(k)!=optimized.get(k):raise DataError('Policy comparison mismatch: '+k)
    if not np.isfinite(rate) or rate<0 or old['rate_cad_per_kwh']!=rate or optimized['rate_cad_per_kwh']!=rate:raise DataError('Policy comparison rate mismatch')
    difference=old['forecast_kwh']-optimized['forecast_kwh']
    return {'old_policy_kwh':old['forecast_kwh'],'optimized_policy_kwh':optimized['forecast_kwh'],
        'projected_difference_kwh':difference,'projected_difference_cad':difference*rate,
        'verified_savings_cad':None,'status':'diagnostic_only' if not (old['status']==optimized['status']=='conditional_projection_requires_review') else 'conditional_projection_requires_review','gates':sorted(set(old['gates']+optimized['gates'])),
        'scope':'Same-weather, same-initial-state policy projection, not exact or verified utility-bill savings'}
