from __future__ import annotations
import numpy as np
import pandas as pd
from .data import DataError,digest,frame,records,stamp,window
from .regression import fit,predict,verify,uncertainty_samples
from .bms import join_features


def train_post(meter_rows,bms_rows,cfg):
    f=join_features(frame(meter_rows,cfg,energy=True),bms_rows,cfg)
    result=fit(records(f),cfg,'bms_post',list(cfg.get('bms_features',{})))
    result['bms_feature_definition_sha256']=digest(cfg['bms_features'])
    result['model_id']=digest({k:v for k,v in result.items() if k!='model_id'})
    return result


def forecast(model,condition_rows,start,end):
    verify(model)
    if model['kind']!='baseline':raise DataError('Forecast requires a baseline model')
    cfg=model['config'];f,a,b=window(frame(condition_rows,cfg),cfg,start,end)
    if a<stamp(model['fit_end_exclusive'],cfg['timezone']):raise DataError('Forecast window must follow the frozen baseline period')
    pred,good,outside,x=predict(model,f)
    gates=list(model['gates'])
    if 'conditions_source' in f and f.conditions_source.fillna('').str.contains('assumed|scenario').any():gates.append('future_weather_is_an_assumed_scenario')
    if outside.any():gates.append('weather_or_operating_condition_extrapolation')
    if not good.all():gates.append('forecast_conditions_incomplete_or_unphysical_prediction')
    band=uncertainty_samples(model,x[good],f.hours.to_numpy()[good]) if good.any() else None
    return {'schema':'option_c.forecast.v1','status':'conditional_forecast_requires_review','timezone':cfg['timezone'],'start':a.isoformat(),'end_exclusive':b.isoformat(),
            'expected_periods':len(f),'predicted_periods':int(good.sum()),'baseline_kwh':float(np.nansum(pred)) if good.all() else None,
            'available_baseline_kwh':float(np.nansum(pred)) if good.any() else None,'aggregate_model_band_95_kwh':np.quantile(band,[.025,.975]).tolist() if band is not None else None,
            'gates':gates+['complete_uncertainty_budget_required'],'model_id':model['model_id'],
            'scope':'Conditional no-intervention energy forecast using supplied future weather/independent conditions; not savings or peak demand',
            'periods':[{'timestamp':t.isoformat(),'baseline_kwh':float(p) if v else None,'average_baseline_kw':float(p/h) if v else None,'outside_support':bool(o)} for t,p,v,o,h in zip(f.timestamp,pred,good,outside,f.hours)]}


def savings(baseline,reporting_rows,start,end,post_model=None,bms_rows=None,adjustments=None,flat_rate=None):
    verify(baseline)
    if baseline['kind']!='baseline':raise DataError('Savings require a frozen baseline model')
    cfg=baseline['config'];f,a,b=window(frame(reporting_rows,cfg),cfg,start,end)
    cutoff=stamp(cfg['intervention_timestamp'],cfg['timezone'])
    if a<cutoff or a<stamp(baseline['fit_end_exclusive'],cfg['timezone']):raise DataError('Reporting must follow baseline and intervention')
    bp,bgood,boutside,bx=predict(baseline,f)
    measured=f.energy_kwh.to_numpy(float);observed=np.isfinite(measured)
    measured[f.quality.fillna('missing').to_numpy()!='good']=np.nan;observed=np.isfinite(measured)
    used=measured.copy();estimated=np.zeros(len(f),bool);post_samples=None
    gates=list(baseline['gates']);post_outside=np.zeros(len(f),bool)
    if post_model:
        verify(post_model)
        if post_model['kind']!='bms_post':raise DataError('Post artifact must be a calibrated BMS electrical model')
        pc=post_model['config']
        for k in ('meter_id','boundary_description','timezone','resolution','interval_minutes','intervention_timestamp'):
            if pc.get(k)!=cfg.get(k):raise DataError('Baseline/post model boundary or cadence mismatch')
        if post_model.get('bms_feature_definition_sha256')!=digest(pc['bms_features']):raise DataError('BMS feature definitions changed')
        if bms_rows is None:raise DataError('BMS reporting features required with post model')
        pf=join_features(f,bms_rows,pc)
        pp,pgood,post_outside,px=predict(post_model,pf)
        after=pf.timestamp>=stamp(post_model['fit_end_exclusive'],pc['timezone'])
        # No calibration-period backfill. Never replace an observed meter reading with prediction.
        estimated=(~observed)&pgood&after.to_numpy()&bgood
        used[estimated]=pp[estimated]
        if estimated.any():
            gates+=post_model['gates']+['post_consumption_contains_bms_model_estimates']
            post_samples=uncertainty_samples(post_model,px[estimated],pf.hours.to_numpy()[estimated])
        if post_outside.any():gates.append('bms_outside_electrical_calibration_support')
    adjustment=np.zeros(len(f));adjustment_refs=[]
    for item in adjustments or []:
        if not item.get('reference') or item.get('reviewed') is not True:raise DataError('Non-routine adjustments need documented reviewed references')
        t=stamp(item['timestamp'],cfg['timezone'],True);idx=np.flatnonzero(f.timestamp==t)
        if len(idx)!=1:raise DataError('Adjustment outside requested period')
        amount=float(item['baseline_adjustment_kwh'])
        if not np.isfinite(amount):raise DataError('Non-finite adjustment')
        adjustment[idx[0]]+=amount;adjustment_refs.append(item['reference'])
    adjusted=bp+adjustment
    good=bgood&np.isfinite(used)&np.isfinite(adjusted)&(adjusted>=0)
    difference=adjusted-used
    if boutside.any():gates.append('baseline_condition_extrapolation')
    if not good.all():gates.append('reporting_period_incomplete')
    paired=float(np.sum(difference[good])) if good.any() else None;full=paired if good.all() else None
    confidence=None
    bs=uncertainty_samples(baseline,bx[good],f.hours.to_numpy()[good]) if good.any() else None
    if bs is not None:
        totals=bs+adjustment[good].sum()-np.sum(used[good])
        if estimated.any():
            if post_samples is None:totals=None
            else:totals-=post_samples-np.sum(used[estimated])
        if totals is not None:confidence=np.quantile(totals,[.025,.975]).tolist()
    if confidence is None:gates.append('model_uncertainty_unavailable')
    elif confidence[0]<=0<=confidence[1]:gates.append('modeled_savings_not_resolved_from_zero')
    gates.append('complete_uncertainty_budget_required')
    if paired is not None and abs(paired/int(good.sum()))<=baseline['holdout_metrics']['rmse_kwh']:gates.append('mean_period_savings_not_above_baseline_holdout_rmse')
    result={'schema':'option_c.savings.v1','status':'unavailable_no_paired_consumption' if not good.any() else 'meter_based_option_c_estimate_requires_review' if not estimated.any() else 'bms_assisted_modeled_estimate_requires_review',
            'timezone':cfg['timezone'],'start':a.isoformat(),'end_exclusive':b.isoformat(),'expected_periods':len(f),'paired_periods':int(good.sum()),
            'observed_meter_periods':int((observed&good).sum()),'bms_estimated_periods':int((estimated&good).sum()),'unavailable_periods':int((~good).sum()),
            'adjusted_baseline_kwh_paired':float(np.sum(adjusted[good])) if good.any() else None,'reporting_kwh_paired':float(np.sum(used[good])) if good.any() else None,
            'savings_kwh_paired':paired,'savings_kwh_complete_period':full,'savings_pct_paired':float(100*paired/np.sum(adjusted[good])) if adjusted[good].sum()>0 else None,
            'aggregate_model_band_95_savings_kwh':confidence,'band_scope':'Paired periods only. Approximate joint predictor/target coefficient resampling and contiguous holdout residual blocks; excludes structural, measurement, adjustment and causal uncertainty. Separate baseline/post samples assume independent calibration errors.',
            'gates':sorted(set(gates)),'baseline_model_id':baseline['model_id'],'post_model_id':post_model.get('model_id') if post_model else None,
            'adjustment_references':adjustment_refs,'scope':'Whole declared meter boundary avoided energy; cannot attribute savings uniquely to BMS controls without reviewing other changes',
            'demand_savings_kw':None,'demand_note':'kWh/period gives average kW, not billing peak demand; a separate demand model is required',
            'periods':[{'timestamp':t.isoformat(),'baseline_kwh':float(adjusted[i]) if bgood[i] and np.isfinite(adjusted[i]) else None,'unadjusted_baseline_kwh':float(bp[i]) if bgood[i] else None,'baseline_adjustment_kwh':float(adjustment[i]),
                        'reporting_kwh':float(used[i]) if np.isfinite(used[i]) else None,'reporting_source':'meter' if observed[i] else 'bms_model' if estimated[i] else 'unavailable',
                        'savings_kwh':float(difference[i]) if good[i] else None,'baseline_outside_support':bool(boutside[i])} for i,t in enumerate(f.timestamp)]}
    if flat_rate:
        rate=float(flat_rate['rate_per_kwh'])
        if not np.isfinite(rate) or rate<0 or not flat_rate.get('reference'):raise DataError('Rate must be finite, nonnegative and documented')
        result['variable_energy_cost_savings_paired']={'amount':paired*rate if paired is not None else None,'currency':flat_rate.get('currency','CAD'),'rate_reference':flat_rate['reference'],'scope':'Same flat energy rate only; excludes demand charges, taxes, fixed charges, ratchets and Global Adjustment'}
    return result
