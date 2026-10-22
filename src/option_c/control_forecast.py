"""Weekday regression forecast conditional on continuation of recent controls."""
import numpy as np
import pandas as pd
from .data import DataError,stamp,digest
from .regression import metrics


def fit_predict(periods, timezone, start, end):
    rows=[]
    for p in periods:
        v=p.get('reporting_kwh')
        if v is None:continue
        if not np.isfinite(v) or v<0:raise DataError('Invalid controlled-period meter energy')
        t=pd.Timestamp(p['timestamp']).tz_convert(timezone)
        rows.append({'date':str(t.date()),'weekday':t.dayofweek,'kwh':float(v),
                     'hours':((t+pd.DateOffset(days=1)).tz_convert('UTC')-t.tz_convert('UTC')).total_seconds()/3600})
    f=pd.DataFrame(rows)
    if len(f)<42:raise DataError('Continued-control forecast requires at least 42 observed reporting days')
    f=f.sort_values('date').reset_index(drop=True)
    if f.date.duplicated().any():raise DataError('Duplicate controlled-period meter day')
    cutoff=str((pd.Timestamp(f.date.max())-pd.DateOffset(days=13)).date())
    train=f[f.date<cutoff];test=f[f.date>=cutoff]
    if len(train)<28 or len(test)<12:raise DataError('Need 28 training days and >=12 days in last 14 calendar-day holdout')
    def coefficients(g,trend):
        if any((g.weekday==i).sum()<4 for i in range(7)):raise DataError('Each weekday needs four controlled-period training observations')
        origin=g.date.min();x=design(g,origin,trend)
        if len(g)<4*x.shape[1] or np.linalg.matrix_rank(x)!=x.shape[1]:raise DataError('Insufficient regression observations')
        return np.linalg.lstsq(x,(g.kwh/g.hours).to_numpy(),rcond=None)[0],origin
    def design(g,origin,trend):
        x=np.eye(7)[g.weekday.to_numpy()]
        if trend:x=np.column_stack([x,(pd.to_datetime(g.date)-pd.Timestamp(origin)).dt.days.to_numpy()/30.])
        return x
    def window(g,days):
        return g if days is None else g[g.date>=str((pd.Timestamp(g.date.max())-pd.DateOffset(days=days-1)).date())]
    candidates=[]
    origins=[str((pd.Timestamp(cutoff)-pd.DateOffset(days=n)).date()) for n in (14,7)]
    for days in (None,35,42):
        for trend in (False,True):
            losses=[]
            try:
                for origin in origins:
                    past=window(train[train.date<origin],days)
                    valid=train[(train.date>=origin)&(train.date<str((pd.Timestamp(origin)+pd.DateOffset(days=7)).date()))]
                    if len(valid)<6:raise DataError('Incomplete control forecast inner fold')
                    coef,base=coefficients(past,trend)
                    pred=design(valid,base,trend)@coef*valid.hours.to_numpy()
                    losses.append(float(np.mean((valid.kwh.to_numpy()-pred)**2)))
                candidates.append({'window_days':days,'linear_trend':trend,'inner_rmse_kwh':float(np.sqrt(np.mean(losses)))})
            except DataError:continue
    # Short histories retain the simple weekday model; selection never sees the outer targets.
    chosen=min(candidates,key=lambda c:(c['inner_rmse_kwh'],c['linear_trend'])) if candidates else {'window_days':None,'linear_trend':False}
    prior,origin=coefficients(window(train,chosen['window_days']),chosen['linear_trend'])
    prediction=design(test,origin,chosen['linear_trend'])@prior*test.hours.to_numpy()
    validation=metrics(test.kwh,prediction)
    coef,origin=coefficients(window(f,chosen['window_days']),chosen['linear_trend']);grid=pd.date_range(stamp(start,timezone,True).tz_convert(timezone),
        stamp(end,timezone,True).tz_convert(timezone),freq='D',inclusive='left')
    if not len(grid):raise DataError('Empty continued-control forecast window')
    if str(grid[0].date())<=f.date.max():raise DataError('Continued-control forecast must start after its last training day')
    daily=[]
    for t in grid:
        hours=((t+pd.DateOffset(days=1)).tz_convert('UTC')-t.tz_convert('UTC')).total_seconds()/3600
        future=pd.DataFrame({'date':[str(t.date())],'weekday':[t.dayofweek]})
        value=float((design(future,origin,chosen['linear_trend'])@coef)[0]*hours)
        daily.append({'date':str(t.date()),'continued_control_kwh':value if value>=0 else None})
    gates=['recent_control_policy_assumed_unchanged','seasonal_and_weather_changes_not_modeled',
           'historical_development_backtest_not_prospective_validation','meter_boundary_requires_review']
    if (prediction<0).any() or validation['cvrmse_pct']>20 or abs(validation['nmbe_pct'])>5:gates.append('heldout_model_error_requires_review')
    if any(p['continued_control_kwh'] is None for p in daily):gates.append('unphysical_forecast_requires_review')
    if chosen['linear_trend']:gates.append('linear_trend_extrapolated_beyond_observed_dates')
    model={'family':'controlled_period_weekday_ols','timezone':timezone,'coefficients_average_kw':coef.tolist(),
        'regression_origin':origin,'selected_candidate':chosen,'selection_candidates':candidates,
        'terms':['weekday_'+str(i) for i in range(7)]+(['linear_time_per_30_days'] if chosen['linear_trend'] else []),
        'training_days':len(window(f,chosen['window_days'])),'available_reporting_days':len(f),
        'fit_start':window(f,chosen['window_days']).date.min(),'fit_end':f.date.max(),
        'holdout_metrics':validation,'holdout_start':cutoff,'forecast_start':start,'forecast_end_exclusive':end,
        'forecast_kwh':sum(p['continued_control_kwh'] for p in daily) if all(p['continued_control_kwh'] is not None for p in daily) else None,'gates':gates,
        'status':'conditional_continued_control_forecast_requires_review',
        'scope':'OLS with weekday indicators and optional linear trend/lookback chosen by earlier rolling validation, before separate outer holdout. Assumes recent policy and load pattern/trend continue. Does not infer future optimizer setpoints or seasonal response.',
        'input_sha256':digest(rows)}
    model['model_id']=digest(model)
    return model,daily
