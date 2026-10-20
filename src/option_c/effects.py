"""Joint zone/equipment-to-meter association regression from ordinary historical data.

This is separate from the frozen Option C baseline and never substitutes for measured
consumption. Concurrent zones can be unidentifiable; duration shares are not energy shares.
"""
import numpy as np
import pandas as pd
from .data import DataError,digest,frame,stamp,window
from .regression import ols,metrics,calendar,matrix


KINDS=('heating_fraction','cooling_fraction','uncertain_fraction')


def analyze(meter_rows,features,groups,cfg,start,end):
    meter=cfg['meter_id'];settings=cfg.get('optimizer',{})
    selected=[g for g in groups if not g['meter_ids'] or meter in g['meter_ids']]
    result={'schema':'option_c.zone_meter_effects.v1','meter_id':meter,'status':'unavailable',
            'scope':'Conditional statistical associations, not causal effects, energy disaggregation or verified zone savings',
            'training_scope':'Ordinary synchronized post-intervention history strictly before reporting start; no separate test dataset',
            'start':start,'end_exclusive':end,'groups':[],'zone_effects':[],
            'gates':['weather_not_adjusted','optimizer_modes_are_uncertain_proxies',
                     'unmeasured_other_loads_may_confound_associations','complete_uncertainty_budget_required']}
    for g in selected:
        result['groups'].append({'group_id':g['equipment_id'],'zone_ids':g['zone_ids'],
             'physical_mapping_verified':g['physical_mapping_verified'],'status':'unavailable',
             'associated_kwh':None,'associated_pct_of_meter_kwh':None})
    def unavailable(reason):
        result['status']='unavailable';result['reason']=reason
        for group in result['groups']:
            group['status']='unavailable';group.setdefault('reason',reason)
        return finish(result)
    if not selected:return unavailable('No optimizer groups selected for this meter')
    if any(not g['physical_mapping_verified'] for g in selected):result['gates'].append('zone_to_meter_physical_mapping_not_verified')
    f=frame(meter_rows,cfg,energy=True)
    a,b=stamp(start,cfg['timezone'],True),stamp(end,cfg['timezone'],True)
    future,_,_=window(f,cfg,start,end)
    f=pd.concat([f[f.period_end<=a],future],ignore_index=True)
    f['date']=f.timestamp.dt.tz_convert(cfg['timezone']).dt.date.astype(str)
    equipment={g['equipment_id']:g for g in selected};column_groups={}
    for i,key in enumerate(equipment):
        part=features[features.equipment_id==key][['date']+list(KINDS)].copy()
        names={name:f'g{i}_{name}' for name in KINDS};part=part.rename(columns=names)
        f=f.merge(part,on='date',how='left',validate='one_to_one');column_groups[key]=list(names.values())
    cutoff=stamp(cfg['intervention_timestamp'],cfg['timezone'])
    history=f[(f.timestamp>=cutoff)&(f.period_end<=a)&(f.quality=='good')&f.energy_kwh.notna()].copy()
    minimum=int(settings.get('minimum_training_days',60))
    if minimum<40:raise DataError('Optimizer electrical model requires at least 40 training days, including withheld validation')
    supported={key:cols for key,cols in column_groups.items() if history[cols].notna().all(axis=1).sum()>=minimum}
    excluded=set(column_groups)-set(supported)
    if excluded:result['gates'].append('some_zone_groups_excluded_for_insufficient_overlap')
    if not supported:return unavailable(f'Need at least {minimum} synchronized post-intervention daily readings before reporting start')
    columns=[col for cols in supported.values() for col in cols]
    history=history.dropna(subset=columns).sort_values('timestamp').reset_index(drop=True)
    if len(history)<minimum:return unavailable(f'Only {len(history)} complete joint training days; require {minimum}')
    split=int(len(history)*.8);inner=history.iloc[:split];holdout=history.iloc[split:]
    # Collapse exactly coincident group histories using training predictors only.
    clusters=[]
    for key,cols in supported.items():
        found=next((c for c in clusters if np.allclose(inner[cols].to_numpy(),inner[supported[c[0]]].to_numpy(),rtol=0,atol=1e-10)),None)
        if found is None:clusters.append([key])
        else:found.append(key)
    terms=[{'kind':'constant'}];cluster_columns={}
    for cluster in clusters:
        key=cluster[0];cluster_columns[key]=supported[key]
        terms += [{'kind':'bms','column':col} for col in supported[key]]
    terms+=calendar({**cfg,'independent_variables':[],'calendar_features':bool(settings.get('calendar_features',True))})
    def same_patterns(part):
        for cluster in clusters:
            for key in cluster[1:]:
                if not np.allclose(part[supported[key]].to_numpy(),part[supported[cluster[0]]].to_numpy(),rtol=0,atol=1e-10):return False
        return True
    try:
        if len(holdout)<8:raise DataError('Need eight independent withheld daily readings')
        if not same_patterns(holdout):raise DataError('Co-moving groups diverge in withheld data; individual effects cannot be identified')
        x=matrix(inner,terms,cfg);y=(inner.energy_kwh/inner.hours).to_numpy(float)
        c,active=ols(x,y)
        if not any(active[i] for i,t in enumerate(terms) if t.get('column','').endswith(('heating_fraction','cooling_fraction'))):
            raise DataError('No independent zone mode variation; electrical effects cannot be identified')
        hp=matrix(holdout,terms,cfg)@c*holdout.hours.to_numpy()
        if not np.isfinite(hp).all() or (hp<0).any():raise DataError('Unphysical held-out operation prediction')
        validation=metrics(holdout.energy_kwh,hp)
        result['holdout_metrics']=validation
        limits=cfg.get('validation_limits',{'cvrmse_pct':20,'abs_nmbe_pct':5})
        if validation['cvrmse_pct']>limits['cvrmse_pct'] or abs(validation['nmbe_pct'])>limits['abs_nmbe_pct']:
            raise DataError('Electrical association model failed held-out error thresholds')
        x=matrix(history,terms,cfg);c,active=ols(x,(history.energy_kwh/history.hours).to_numpy(float))
    except DataError as exc:return unavailable(str(exc))
    artifact={'schema':'option_c.operation_response.v1','meter_id':meter,'terms':terms,
              'coef_average_kw':c.tolist(),'active_features':active.tolist(),'holdout_metrics':validation,
              'feature_min':x.min(axis=0).tolist(),'feature_max':x.max(axis=0).tolist(),
              'fit_start':history.timestamp.min().isoformat(),'fit_end_exclusive':history.period_end.max().isoformat(),
              'training_periods':len(history),'clusters':clusters,'optimizer_settings':settings,
              'scope':result['scope'],'training_sha256':digest(history[['date','energy_kwh']+columns].to_dict('records'))}
    artifact['model_id']=digest(artifact)
    report=f[(f.timestamp>=a)&(f.timestamp<b)].dropna(subset=columns).copy()
    result.update(status='modeled_associations_require_review',training_periods=len(history),
                  model=artifact,reporting_matched_periods=len(report))
    if report.empty:return unavailable('No synchronized reporting meter/optimizer periods')
    if not same_patterns(report):return unavailable('Co-moving groups diverge in reporting; do not split their shared coefficient')
    rx=matrix(report,terms,cfg);outside=np.any((rx<x.min(axis=0)-1e-8)|(rx>x.max(axis=0)+1e-8),axis=1)
    if outside.any():result['gates'].append('operation_reporting_outside_training_support')
    predicted=rx@c*report.hours.to_numpy();admitted=(~outside)&np.isfinite(predicted)&(predicted>=0)
    report=report.loc[admitted].copy();rx=rx[admitted];predicted=predicted[admitted]
    result['reporting_admitted_periods']=len(report)
    if report.empty:return unavailable('Reporting operation lies outside validated model support')
    paired=(report.quality=='good')&report.energy_kwh.notna()
    denominator=float(report.loc[paired,'energy_kwh'].sum()) if paired.any() else None
    result['reporting_meter_kwh_on_matched_periods']=denominator
    result['operation_model_kwh_on_matched_periods']=float(predicted[paired.to_numpy()].sum()) if paired.any() else None
    result['operation_model_kwh_available']=float(predicted.sum())
    result['reporting_paired_periods']=int(paired.sum())
    if any(report[cols[2]].gt(0).any() for cols in supported.values()):result['gates'].append('deadband_or_application_uncertainty_present')
    result['operation_predictions']=[{'timestamp':t.isoformat(),'operation_model_kwh':float(p)} for t,p in zip(report.timestamp,predicted)]
    if not paired.any():
        result['status']='operation_prediction_only_requires_review'
        result['reason']='Operation prediction available, but no reporting meter readings for association percentages or measured savings'
        for group in result['groups']:group['reason']=result['reason']
        return finish(result)
    report=report.loc[paired].copy();rx=rx[paired.to_numpy()]
    grouped=[]
    for cluster in clusters:
        key=cluster[0];cols=supported[key];members=[equipment[k] for k in cluster]
        zone_ids=[zone for member in members for zone in member['zone_ids']]
        mode_indices=[i for i,t in enumerate(terms) if t.get('column') in cols[:2]]
        off_days=int(((history[cols[0]]==0)&(history[cols[1]]==0)).sum())
        on_days=int(((history[cols[0]]>0)|(history[cols[1]]>0)).sum())
        identifiable=any(active[i] for i in mode_indices) and off_days>=int(settings.get('minimum_off_days',5)) and on_days>=5
        contribution=float(np.sum((rx[:,mode_indices]@c[mode_indices])*report.hours.to_numpy())) if identifiable else None
        grouped.append({'group_id':key,'member_equipment':cluster,'zone_ids':zone_ids,
                        'physical_mapping_verified':all(m['physical_mapping_verified'] for m in members),
                        'status':'associated_requires_review' if identifiable else 'unavailable_no_supported_on_off_comparison',
                        'associated_kwh':contribution,'associated_pct_of_meter_kwh':100*contribution/denominator if contribution is not None and denominator>0 else None,
                        'supported_off_days':off_days,'supported_on_days':on_days,
                        'mean_uncertain_fraction':float(report[cols[2]].mean()),
                        'individual_zone_effect_identifiable':len(zone_ids)==1 and identifiable,
                        'note':'Signed additive heating/cooling association against supported OFF reference, holding other fitted terms fixed; not causal savings'})
    for group in result['groups']:
        if group['group_id'] in excluded:
            group['reason']='Insufficient synchronized training days for this equipment group';grouped.append(group)
    result['groups']=grouped
    return finish(result)


def finish(result):
    for group in result['groups']:
        for zone in group['zone_ids']:
            unique=group.get('individual_zone_effect_identifiable',False)
            result['zone_effects'].append({'zone_id':zone,'meter_id':result['meter_id'],
                 'group_id':group['group_id'],'status':group['status'] if unique else 'individual_effect_unavailable',
                 'associated_kwh':group['associated_kwh'] if unique else None,
                 'associated_pct_of_meter_kwh':group['associated_pct_of_meter_kwh'] if unique else None,
                 'group_associated_pct_of_meter_kwh':group['associated_pct_of_meter_kwh'],
                 'note':group.get('reason') or ('Shared equipment/co-moving zones are reported as a group' if not unique else 'Statistical association, not causal savings')})
    return result
