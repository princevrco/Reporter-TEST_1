"""Visible all-meter forecast and flat-rate comparison summary."""
from pathlib import Path
from html import escape
import json
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
from .data import DataError,load,save,stamp
from .workflow import forecast
from .reporting import write
from .comparisons import compare
from .control_forecast import fit_predict as continued_forecast
from .weather import read as read_weather, join as join_weather


def total(results,selected,rate):
    if not selected:return {'status':'select_non_overlapping_meters','meter_ids':[], 'savings_kwh':None,'savings_cad':None}
    if len(selected)!=len(set(selected)):raise DataError('Duplicate total meter IDs')
    if any(m not in results for m in selected):
        return {'status':'selected_meter_unavailable','meter_ids':selected,'savings_kwh':None,'savings_cad':None}
    if any(results[m].get('release_status')=='withheld_requires_review' for m in selected):
        return {'status':'selected_model_or_data_review_required','meter_ids':selected,'savings_kwh':None,'savings_cad':None,
            'note':'At least one selected meter has withheld results. Do not sum diagnostic differences.'}
    maps=[{p['timestamp']:p for p in results[m]['periods']} for m in selected]
    timestamps=set(maps[0])
    if any(set(m)!=timestamps for m in maps):raise DataError('Selected meters have different reporting grids')
    paired=[t for t in sorted(timestamps) if all(m[t].get('savings_kwh') is not None for m in maps)]
    value=sum(m[t]['savings_kwh'] for m in maps for t in paired) if paired else None
    return {'status':'operator_selected_boundary_requires_review','meter_ids':selected,
            'paired_periods':len(paired),'expected_periods':len(timestamps),
            'savings_kwh':value if len(paired)==len(timestamps) else None,
            'savings_cad':value*rate if value is not None and len(paired)==len(timestamps) else None,
            'paired_savings_kwh':value,'paired_savings_cad':value*rate if value is not None else None,
            'note':'Common observed/eligible periods only; operator must verify these boundaries do not overlap. Signed differences, not verified causal savings.'}


def build(folder,rate=.12,forecast_start=None,forecast_end=None,total_meters=None,weather=None):
    folder=Path(folder);summary=load(folder/'summary.json');results={};rows=[];views=[]
    if not np.isfinite(rate) or rate<0:raise DataError('Rate must be finite and nonnegative')
    first=next((e for e in summary['meters'] if (folder/e['folder']/'baseline-model.json').exists()),None)
    if first is None and not summary.get('timezone'):raise DataError('No fitted baseline models; inspect summary.json blocked reasons')
    timezone=load(folder/first['folder']/'baseline-model.json')['config']['timezone'] if first else summary['timezone']
    start=forecast_start or str(pd.Timestamp.now(tz=timezone).date())
    end=forecast_end or str((pd.Timestamp(start)+pd.DateOffset(months=3)).date())
    a,b=stamp(start,timezone,True),stamp(end,timezone,True)
    if b<=a or (b-a).total_seconds()>366*86400:raise DataError('Choose a positive forecast period of at most one year')
    if weather is None and (folder/'forecast-weather.csv').exists():weather,_=read_weather(folder/'forecast-weather.csv',timezone)
    if weather is None and (folder/'external-weather.csv').exists():weather,_=read_weather(folder/'external-weather.csv',timezone)
    forecasts=[];daily_forecasts=[];comparison_daily=[];controlled_daily=[]
    for entry in summary['meters']:
        meter=entry['meter_id'];base=folder/entry['folder'];row={'meter_id':meter,'status':entry['status'],
            'is_virtual_meter':entry.get('is_virtual_meter',False),'meter_description':entry.get('meter_description',''),
            'future_optimized_kwh':None,'future_savings_kwh':None,'future_savings_cad':None}
        row.update({k:v for k,v in entry.items() if k.startswith('meter_consumption_')})
        row['baseline_holdout_metrics']=entry.get('holdout_metrics')
        row['meter_consumption_coverage']=str(entry.get('meter_consumption_observed_days','?'))+'/'+str(entry.get('meter_consumption_expected_days','?'))
        result_path=base/'savings/result.json'
        if result_path.exists():
            r=load(result_path);results[meter]=r
            kwh=r.get('savings_kwh_complete_period');paired=r.get('savings_kwh_paired')
            row.update(baseline_kwh=r.get('adjusted_baseline_kwh_paired'),measured_kwh=r.get('reporting_kwh_paired'),
                       savings_kwh=kwh,savings_cad=kwh*rate if kwh is not None else None,
                       paired_savings_kwh=paired,paired_savings_cad=paired*rate if paired is not None else None,
                       rate_cad_per_kwh=rate,unavailable_periods=r.get('unavailable_periods'),
                       validation='Held-out error failed' if 'heldout_model_error_requires_review' in r.get('gates',[]) else 'Numerical screen passed; site review pending',
                       gates=';'.join(r.get('gates',[])))
            row.update(release_status=r.get('release_status','provisional_requires_review'),release_reasons=';'.join(r.get('release_reasons',[])),
                diagnostic_difference_kwh=r.get('diagnostic_savings_kwh_paired'),
                diagnostic_difference_cad=r['diagnostic_savings_kwh_paired']*rate if r.get('diagnostic_savings_kwh_paired') is not None else None,
                selected_baseline=entry.get('selected_baseline'))
            row['diagnostic_model_band_95_difference_kwh']=r.get('diagnostic_aggregate_model_band_95_savings_kwh',r.get('aggregate_model_band_95_savings_kwh'))
            row['model_band_scope']=r.get('band_scope')
            if r.get('release_status')=='withheld_requires_review':row['validation']='Withheld: '+row['release_reasons']
            views.append({'meter_id':meter,'reporting_periods':r['periods'],'gates':r.get('gates',[])})
        else:row.update(reason=entry.get('reason'),savings_kwh=None,savings_cad=None)
        if (base/'baseline-model.json').exists():
            try:
                model=load(base/'baseline-model.json')
                grid=pd.date_range(a.tz_convert(timezone),b.tz_convert(timezone),freq='D',inclusive='left')
                conditions=[{'date':t.date().isoformat(),'meter_id':meter} for t in grid]
                if model['config'].get('external_weather_enabled'):
                    if weather is None:raise DataError('Future baseline needs explicitly supplied daily weather')
                    conditions=join_weather(pd.DataFrame(conditions),weather).to_dict('records')
                f=forecast(model,conditions,start,end);write(f,base/'future-forecast')
                row.update(forecast_kwh=f['baseline_kwh'],forecast_baseline_cad=f['baseline_kwh']*rate if f['baseline_kwh'] is not None else None,forecast_status=f['status'],diagnostic_forecast_kwh=f.get('diagnostic_baseline_kwh'))
                forecasts.append({'meter_id':meter,'baseline_kwh':f['baseline_kwh']})
                daily_forecasts.extend({'meter_id':meter,**period} for period in f['periods'])
                if views and views[-1]['meter_id']==meter:views[-1]['forecast_periods']=f['periods']
            except (DataError,ValueError,KeyError) as exc:row.update(forecast_kwh=None,forecast_status=str(exc))
        response_path=base/'response/model.json'
        if response_path.exists():
            response=load(response_path)
            row['bms_response_validation']=response['holdout_metrics']
            row['bms_response_gates']=response['gates']
            view=next((v for v in views if v['meter_id']==meter),None)
            if view is None:
                view={'meter_id':meter,'reporting_periods':[],'gates':response['gates']};views.append(view)
            view['gates']=sorted(set(view['gates']+['BMS response: '+g for g in response['gates']]))
            view['response_periods']=[{'timestamp':stamp(p['date'],timezone,True).isoformat(),'bms_estimated_kwh':p['bms_estimated_kwh']}
                for p in response['heldout_predictions'] if summary['start']<=p['date']<summary['end_exclusive']]
            future_path=base/'response/optimized-future.json'
            if future_path.exists():
                future=load(future_path)
                if any(p['outside_training_support'] for p in future):view['gates'].append('Optimized scenario exceeds BMS training support')
                view['optimized_periods']=[{'timestamp':stamp(p['date'],timezone,True).isoformat(),'optimized_kwh':p['bms_estimated_kwh']} for p in future if start<=p['date']<end]
                eligible={p['date']:p for p in future if start<=p['date']<end}
                expected=pd.date_range(a.tz_convert(timezone),b.tz_convert(timezone),freq='D',inclusive='left').date.astype(str)
                if all(d in eligible and eligible[d]['bms_estimated_kwh'] is not None and not eligible[d]['outside_training_support'] for d in expected):
                    value=sum(eligible[d]['bms_estimated_kwh'] for d in expected)
                    if any(g in response['gates'] for g in ('heldout_model_error_requires_review','meter_quality_review_required','bms_signal_semantics_unverified')):
                        row['diagnostic_future_optimized_kwh']=value
                    else:row['future_optimized_kwh']=value
        spath=base/'setpoint-response/consumption-summary.json'
        if spath.exists():
            sp=load(spath)
            row.update(setpoint_response_status=sp['status'],setpoint_model_family=sp['family'],
                setpoint_holdout_cvrmse_pct=sp['heldout_metrics']['cvrmse_pct'],
                setpoint_holdout_nmbe_pct=sp['heldout_metrics']['nmbe_pct'],
                setpoint_rmse_improvement_pct=sp['setpoint_rmse_improvement_pct'],
                setpoint_reconstruction_days=sp['reporting_reconstruction_days'],
                diagnostic_setpoint_reconstruction_kwh=sp['diagnostic_reconstructed_kwh_available_days'],
                setpoint_consumption_kwh_complete_period=sp['full_period_setpoint_consumption_kwh'],
                setpoint_heldout_days=sp['independent_reporting_days'],setpoint_heldout_kwh=sp['heldout_setpoint_kwh'],
                setpoint_response_gates=sp['gates'])
            view=next((v for v in views if v['meter_id']==meter),None)
            if view is None:
                view={'meter_id':meter,'reporting_periods':[],'gates':[]};views.append(view)
            held=pd.read_csv(base/'setpoint-response/heldout.csv')
            view['setpoint_periods']=[{'timestamp':stamp(p['date'],timezone,True).isoformat(),
                'setpoint_estimated_kwh':p['setpoint_estimated_kwh'] if pd.notna(p['setpoint_estimated_kwh']) else None}
                for p in held.to_dict('records') if summary['start']<=p['date']<summary['end_exclusive']]
            view['gates']+=['Setpoint response: '+g for g in sp['gates']]
            scenario_path=base/'setpoint-response/optimized-future.json'
            if scenario_path.exists():
                scenario=load(scenario_path)
                if any(p['outside_training_support'] for p in scenario):view['gates'].append('Setpoint future scenario exceeds training support; plotted values are diagnostic')
                predictions={p['date']:p for p in scenario}
                expected=pd.date_range(a.tz_convert(timezone),b.tz_convert(timezone),freq='D',inclusive='left').date.astype(str)
                good=all(d in predictions and predictions[d]['conditional_setpoint_kwh'] is not None
                    and not predictions[d]['outside_training_support'] for d in expected)
                if good:
                    value=sum(predictions[d]['conditional_setpoint_kwh'] for d in expected)
                    row['diagnostic_setpoint_future_optimized_kwh']=value
                    blocked={'bms_signal_semantics_unverified','heldout_model_error_requires_review',
                             'meter_quality_review_required','setpoint_incremental_value_not_demonstrated',
                             'heldout_bms_outside_training_support','unphysical_response_prediction'}
                    if not blocked.intersection(sp['gates']):row['future_optimized_kwh']=value
                view['optimized_periods']=[{'timestamp':stamp(p['date'],timezone,True).isoformat(),
                    'optimized_kwh':p['conditional_setpoint_kwh']} for p in scenario if start<=p['date']<end]
        periods=results.get(meter,{}).get('periods',[])
        sp_daily=base/'setpoint-response/reporting-daily.csv'
        sp_rows=pd.read_csv(sp_daily).to_dict('records') if sp_daily.exists() else []
        if not periods and sp_rows:
            periods=[{'timestamp':stamp(p['date'],timezone,True).isoformat(),
                      'reporting_kwh':p.get('measured_kwh'),'baseline_kwh':None} for p in sp_rows]
        comparisons,days=compare(periods,sp_rows,summary['start'],summary['end_exclusive'],timezone,rate)
        row.update(comparisons)
        row['difference_kwh']=comparisons['baseline_meter_difference_kwh']
        row['difference_cad']=comparisons['baseline_meter_difference_cad']
        row['estimated_savings_kwh']=row['difference_kwh']
        row['estimated_savings_cad']=row['difference_cad']
        row['difference_status']=row.get('release_status','no_fitted_baseline')
        row['calculated_forecast_kwh']=row.get('forecast_kwh') if row.get('forecast_kwh') is not None else row.get('diagnostic_forecast_kwh')
        row['calculated_forecast_cad']=row['calculated_forecast_kwh']*rate if row['calculated_forecast_kwh'] is not None else None
        try:
            controlled,cdays=continued_forecast(periods,timezone,start,end)
            row.update(continued_control_forecast_kwh=controlled['forecast_kwh'],
                continued_control_forecast_cad=controlled['forecast_kwh']*rate if controlled['forecast_kwh'] is not None else None,
                continued_control_forecast_status=controlled['status'],
                continued_control_forecast_holdout=controlled['holdout_metrics'],
                continued_control_forecast_gates=controlled['gates'])
            save(base/'continued-control-forecast.json',controlled)
            controlled_daily.extend({'meter_id':meter,**p} for p in cdays)
            view=next((v for v in views if v['meter_id']==meter),None)
            if view is not None:view['continued_control_periods']=[{'timestamp':stamp(p['date'],timezone,True).isoformat(),
                'continued_control_kwh':p['continued_control_kwh']} for p in cdays]
        except DataError as exc:row.update(continued_control_forecast_kwh=None,continued_control_forecast_cad=None,
            continued_control_forecast_status=str(exc))
        comparison_daily.extend({'meter_id':meter,**p} for p in days)
        rows.append(row)
    selected=total_meters or [];totals=total(results,selected,rate)
    fm={f['meter_id']:f['baseline_kwh'] for f in forecasts}
    totals['forecast_kwh']=sum(fm[m] for m in selected) if selected and all(fm.get(m) is not None for m in selected) else None
    candidates=[r for r in rows if r['is_virtual_meter']]
    report={'rate_cad_per_kwh':rate,'currency':'CAD','start':summary['start'],'end_exclusive':summary['end_exclusive'],
            'forecast_start':start,'forecast_end_exclusive':end,'site_total':totals,'meters':rows,'unverified_boundary_candidates':candidates,
            'scope':('Weather/calendar candidate baseline' if summary.get('external_weather_supplied') else 'Calendar-only baseline')+' comparison requiring site review. CAD is kWh difference times the user-assumed flat rate; excludes demand, taxes and fixed charges. Baseline forecasts describe no-intervention energy. Any optimized line is conditional on a supplied operating scenario; it is not future realized savings.'}
    save(folder/'site-total.json',totals)
    def money(value):
        return float((Decimal(str(value))*Decimal(str(rate))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)) if value is not None else None
    financial=[]
    for row in rows:
        row['measured_energy_cost_cad_complete_period']=money(row.get('meter_consumption_kwh_complete_period'))
        row['diagnostic_setpoint_reconstruction_cost_cad']=money(row.get('diagnostic_setpoint_reconstruction_kwh'))
        row['setpoint_heldout_energy_cost_cad']=money(row.get('setpoint_heldout_kwh'))
        band=row.get('diagnostic_model_band_95_difference_kwh')
        row['diagnostic_model_band_95_difference_cad']=[money(x) for x in band] if band is not None else None
        financial.append({k:row.get(k) for k in tuple(k for k in row if k.startswith(('baseline_meter_','baseline_optimizer_','meter_optimizer_','calculated_forecast_','continued_control_forecast_','estimated_savings_'))) + ('difference_kwh','difference_cad','comparison_interpretation','comparison_expected_days','meter_id','is_virtual_meter','meter_consumption_kwh_complete_period',
            'measured_energy_cost_cad_complete_period','baseline_kwh','savings_kwh','savings_cad',
            'diagnostic_difference_kwh','diagnostic_difference_cad','release_status','release_reasons',
            'diagnostic_model_band_95_difference_kwh','diagnostic_model_band_95_difference_cad','model_band_scope',
            'baseline_holdout_metrics','setpoint_holdout_cvrmse_pct','setpoint_holdout_nmbe_pct','setpoint_rmse_improvement_pct',
            'setpoint_response_status','setpoint_reconstruction_days','diagnostic_setpoint_reconstruction_kwh',
            'diagnostic_setpoint_reconstruction_cost_cad','setpoint_consumption_kwh_complete_period',
            'setpoint_heldout_days','setpoint_heldout_kwh','setpoint_heldout_energy_cost_cad')})
    save(folder/'financial-summary.json',{'rate_cad_per_kwh':rate,'currency':'CAD','start':summary['start'],
        'end_exclusive':summary['end_exclusive'],'site_total':totals,'exact_savings_cad':None,
        'exact_savings_note':'Savings require an estimated counterfactual; exact causal savings are not observable from these two input files.',
        'scope':'Flat-rate variable energy amounts only, excluding demand charges, tax and fixed charges. Diagnostic differences are not released savings. Do not add overlapping meters.',
        'meters':financial})
    # Include measured energy amounts added above in the main table export as well.
    save(folder/'overview.json',report);pd.DataFrame(rows).to_csv(folder/'meter-summary.csv',index=False)
    pd.DataFrame(daily_forecasts).to_csv(folder/'forecast-daily.csv',index=False)
    pd.DataFrame(comparison_daily).to_csv(folder/'consumption-comparison-daily.csv',index=False)
    pd.DataFrame(controlled_daily).to_csv(folder/'continued-control-forecast-daily.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k.startswith(('baseline_meter_','baseline_optimizer_','meter_optimizer_','calculated_forecast_','continued_control_forecast_','estimated_savings_')) or k in ('meter_id','difference_kwh','difference_cad','difference_status','release_reasons','forecast_status','reason','comparison_interpretation','comparison_expected_days')} for r in rows]).to_csv(folder/'consumption-comparison-summary.csv',index=False)
    pd.DataFrame(candidates).to_csv(folder/'boundary-candidates.csv',index=False)
    data=json.dumps({'summary':report,'views':views},allow_nan=False).replace('<','\\u003c')
    table=''.join('<tr>'+''.join('<td>'+escape('Unavailable' if row.get(k) is None else f'{row[k]:,.2f}' if isinstance(row[k],float) else str(row[k]))+'</td>' for k in ('meter_id','meter_consumption_kwh_available','meter_consumption_coverage','baseline_kwh','measured_kwh','difference_kwh','difference_cad','calculated_forecast_kwh','continued_control_forecast_kwh','validation','reason'))+'</tr>' for row in rows)
    html='''<!doctype html><meta charset="utf-8"><title>Promenade energy overview</title><style>body{font:16px system-ui;margin:30px;color:#173247}section{padding:20px;background:#eff5f8;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ccd5dc;text-align:left}svg{width:100%;max-width:1100px}select{padding:10px}strong{font-size:21px}.scroll{overflow:auto}</style><h1>Promenade forecast and savings comparison</h1><p>Unverified engineering comparison. Negative savings mean consumption exceeded the predicted baseline.</p><section id="totals"></section><section id="candidates"></section><section><label>Meter <select id="meter"></select></label><p id="dates"></p><p id="numbers"></p><p>Teal: reporting-period baseline. Blue: measured consumption. Orange: future baseline forecast.</p><svg id="chart" viewBox="0 0 1000 340" aria-label="Daily energy forecast and measured consumption"></svg><p id="quality"></p></section><p id="scope"></p><div class="scroll"><table><thead><tr><th>Meter</th><th>Source measured kWh (available days)</th><th>Meter days available/expected</th><th>Paired baseline kWh</th><th>Paired measured kWh</th><th>Complete savings kWh</th><th>Complete savings CAD</th><th>Future forecast kWh</th><th>Validation</th><th>Blocked reason</th></tr></thead><tbody>'''+table+'''</tbody></table></div><script>
const D='''+data+''';const S=D.summary;const fmt=n=>n==null?'Unavailable':Number(n).toLocaleString(undefined,{maximumFractionDigits:2});const el=id=>document.getElementById(id);const total=S.site_total;
el('totals').textContent='Site total: '+fmt(total.savings_kwh)+' kWh | CAD $'+fmt(total.savings_cad)+' at $'+S.rate_cad_per_kwh+'/kWh. '+(total.meter_ids.length?'Selected meters: '+total.meter_ids.join(', '):'Choose a verified non-overlapping meter boundary with --total-meters. All-meter sums would double-count overlapping loads.');
el('scope').textContent=S.scope;for(const v of D.views){const o=document.createElement('option');o.value=v.meter_id;o.textContent=v.meter_id;el('meter').append(o)}
el('candidates').textContent='Existing virtual totals (unverified candidates; never added together): '+S.unverified_boundary_candidates.map(m=>m.meter_id+': comparison '+fmt(m.savings_kwh)+' kWh / CAD $'+fmt(m.savings_cad)+'; future baseline '+fmt(m.forecast_kwh)+' kWh').join(' | ');
function draw(){const v=D.views.find(x=>x.meter_id===el('meter').value);if(!v)return;const m=S.meters.find(x=>x.meter_id===v.meter_id);el('dates').textContent='Comparison: '+S.start+' to '+S.end_exclusive+' exclusive. Forecast: '+S.forecast_start+' to '+S.forecast_end_exclusive+' exclusive.';el('numbers').textContent='Source consumption: '+fmt(m.meter_consumption_kwh_available)+' kWh across '+m.meter_consumption_coverage+' days. Complete-period difference: '+fmt(m.savings_kwh)+' kWh | CAD $'+fmt(m.savings_cad)+' | Future baseline: '+fmt(m.forecast_kwh)+' kWh';el('quality').textContent=m.validation+' | '+v.gates.join('; ');
const past=v.reporting_periods||[],future=v.forecast_periods||[];const series=[[past,'baseline_kwh','#087d80'],[past,'reporting_kwh','#3265ad'],[future,'baseline_kwh','#d57b16']];const times=[...past,...future].map(r=>Date.parse(r.timestamp));if(!times.length)return;const lo=Math.min(...times),hi=Math.max(...times);const values=series.flatMap(([rs,k])=>rs.map(r=>r[k]).filter(x=>x!=null));const max=Math.max(1,...values)*1.1;const svg=el('chart');svg.innerHTML='';const make=(tag,attrs,text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,x]of Object.entries(attrs))n.setAttribute(k,x);if(text!=null)n.textContent=text;svg.append(n);return n};for(let i=0;i<5;i++){const y=280-i*60;make('line',{x1:75,x2:970,y1:y,y2:y,stroke:'#ccd5dc'});make('text',{x:5,y:y+5,'font-size':12},fmt(max*i/4))}make('text',{x:75,y:20,'font-size':13},'Daily kWh');for(const[rs,k,color]of series){let points=[];const flush=()=>{if(points.length)make('polyline',{points:points.join(' '),fill:'none',stroke:color,'stroke-width':2});points=[]};for(const r of rs){if(r[k]==null){flush();continue}const x=75+(Date.parse(r.timestamp)-lo)/Math.max(1,hi-lo)*895,y=280-r[k]/max*240;points.push(x+','+y);const dot=make('circle',{cx:x,cy:y,r:3,fill:color});const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=r.timestamp+' '+fmt(r[k])+' kWh';dot.append(title)}flush()}make('text',{x:75,y:315,'font-size':12},new Date(lo).toISOString().slice(0,10));make('text',{x:970,y:315,'font-size':12,'text-anchor':'end'},new Date(hi).toISOString().slice(0,10))}el('meter').addEventListener('change',draw);draw();</script>'''
    html=html.replace('Orange: future baseline forecast.','Orange: future baseline forecast. Purple: held-out BMS consumption estimate. Red: user-supplied optimized operating scenario. Fitted-history diagnostics are excluded from this chart.')
    html=html.replace('Unverified engineering comparison. Negative savings mean consumption exceeded the predicted baseline.',
        'Engineering comparison requiring review. Results with unresolved data, validation or sensitivity findings are withheld. Lines from withheld models are diagnostic only. Diagnostic differences are not verified savings. Negative differences mean measured energy exceeded the baseline.')
    html=html.replace('Complete savings kWh','Matched difference kWh').replace('Complete savings CAD','Matched difference CAD')
    html=html.replace('Results with unresolved data, validation or sensitivity findings are withheld.', 'Calculated differences remain visible even when validation fails; review status identifies unreliable models.')
    html=html.replace("fmt(m.savings_kwh)+' kWh | CAD $'+fmt(m.savings_cad)+' | Future baseline: '+fmt(m.forecast_kwh)","fmt(m.difference_kwh)+' kWh | CAD $'+fmt(m.difference_cad)+' | Future baseline: '+fmt(m.calculated_forecast_kwh)")
    html=html.replace('Complete-period difference: ','Matched baseline minus meter: ')
    html=html.replace('<th>Future forecast kWh</th>','<th>Future baseline kWh</th><th>Continued recent-control forecast kWh</th>')
    html=html.replace("el('quality').textContent=m.validation", "el('numbers').textContent+=' | Continued recent-control forecast: '+fmt(m.continued_control_forecast_kwh)+' kWh / CAD $'+fmt(m.continued_control_forecast_cad)+' (assumes recent policy and loads continue; seasonal changes not modeled).';el('quality').textContent=m.validation")
    html=html.replace("comparison '+fmt(m.savings_kwh)+' kWh / CAD $'+fmt(m.savings_cad)+'; future baseline '+fmt(m.forecast_kwh)","calculated difference '+fmt(m.difference_kwh)+' kWh / CAD $'+fmt(m.difference_cad)+' ('+m.difference_status+'); future baseline '+fmt(m.calculated_forecast_kwh)")
    html=html.replace("[future,'baseline_kwh','#d57b16']];const times=[...past,...future]", "[future,'baseline_kwh','#d57b16'],[v.response_periods||[],'bms_estimated_kwh','#8451a1'],[v.optimized_periods||[],'optimized_kwh','#b43f42']];const times=[...past,...future,...(v.response_periods||[]),...(v.optimized_periods||[])]")
    html=html.replace("[v.optimized_periods||[],'optimized_kwh','#b43f42']];", "[v.optimized_periods||[],'optimized_kwh','#b43f42'],[v.setpoint_periods||[],'setpoint_estimated_kwh','#6a40bc']];")
    html=html.replace("...(v.optimized_periods||[])]", "...(v.optimized_periods||[]),...(v.setpoint_periods||[])]")
    html=html.replace("el('quality').textContent=m.validation", "el('numbers').textContent+=(m.setpoint_response_status?' | Setpoint model: '+fmt(m.diagnostic_setpoint_reconstruction_kwh)+' kWh reconstructed on '+m.setpoint_reconstruction_days+' reporting days; '+m.setpoint_heldout_days+' held-out days; test error '+fmt(m.setpoint_holdout_cvrmse_pct)+'%, bias '+fmt(m.setpoint_holdout_nmbe_pct)+'%. Partial reconstruction is not full-period consumption.':'');el('quality').textContent=m.validation")
    html=html.replace('Purple: held-out BMS consumption estimate.', 'Purple: held-out BMS/setpoint consumption estimates; excludes fitted reconstruction.')
    html=html.replace("el('meter').addEventListener('change',draw);draw();", "const initial=S.meters.find(m=>m.meter_id.toLowerCase()==='vm_total_load');if(initial)el('meter').value=initial.meter_id;el('meter').addEventListener('change',draw);draw();")
    html=html.replace("el('quality').textContent=m.validation", "if(m.diagnostic_difference_cad!=null)el('numbers').textContent+=' | Withheld diagnostic difference: '+fmt(m.diagnostic_difference_kwh)+' kWh / CAD $'+fmt(m.diagnostic_difference_cad)+'; not released savings.';if(m.diagnostic_model_band_95_difference_cad)el('numbers').textContent+=' | Approximate model band: CAD $'+m.diagnostic_model_band_95_difference_cad.map(fmt).join(' to $')+'. A band crossing zero does not establish positive savings.';el('quality').textContent=m.validation")
    audit_link=' | <a href="meter-reconciliation.csv">Virtual-meter reconciliation</a>' if (folder/'meter-reconciliation.csv').exists() else ''
    html=html.replace('<p id="scope"></p>', '<p id="scope"></p><p><a href="financial-summary.json">CAD and energy summary</a> | <a href="setpoint-response-summary.csv">Setpoint-model validation</a> | <a href="meter-summary.csv">All-meter results</a>'+audit_link+'</p>')
    html=html.replace('<p id="scope"></p>','<p id="scope"></p><p><a href="consumption-comparison-summary.csv">All calculated consumption differences</a> | <a href="consumption-comparison-daily.csv">Daily matched calculations</a></p>')
    html=html.replace("el('quality').textContent=m.validation", "el('numbers').textContent+=' | Baseline minus optimizer reconstruction: '+fmt(m.baseline_optimizer_reconstruction_difference_kwh)+' kWh / CAD $'+fmt(m.baseline_optimizer_reconstruction_difference_cad)+' on '+m.baseline_optimizer_reconstruction_paired_days+' days. Meter minus optimizer: '+fmt(m.meter_optimizer_reconstruction_difference_kwh)+' kWh (prediction residual, not savings).';el('quality').textContent=m.validation")
    html=html.replace("[v.setpoint_periods||[],'setpoint_estimated_kwh','#6a40bc']];", "[v.setpoint_periods||[],'setpoint_estimated_kwh','#6a40bc'],[v.continued_control_periods||[],'continued_control_kwh','#21834b']];")
    html=html.replace("...(v.setpoint_periods||[])]", "...(v.setpoint_periods||[]),...(v.continued_control_periods||[])]")
    html=html.replace('Orange: future baseline forecast.', 'Orange: future baseline forecast. Green: forecast assuming recent controlled operation continues.')
    (folder/'index.html').write_text(html,encoding='utf-8')
    return report
