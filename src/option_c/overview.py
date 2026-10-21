"""Visible all-meter forecast and flat-rate comparison summary."""
from pathlib import Path
from html import escape
import json
import numpy as np
import pandas as pd
from .data import DataError,load,save,stamp
from .workflow import forecast
from .reporting import write


def total(results,selected,rate):
    if not selected:return {'status':'select_non_overlapping_meters','meter_ids':[], 'savings_kwh':None,'savings_cad':None}
    if len(selected)!=len(set(selected)):raise DataError('Duplicate total meter IDs')
    if any(m not in results for m in selected):
        return {'status':'selected_meter_unavailable','meter_ids':selected,'savings_kwh':None,'savings_cad':None}
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


def build(folder,rate=.12,forecast_start=None,forecast_end=None,total_meters=None):
    folder=Path(folder);summary=load(folder/'summary.json');results={};rows=[];views=[]
    if not np.isfinite(rate) or rate<0:raise DataError('Rate must be finite and nonnegative')
    first=next((e for e in summary['meters'] if (folder/e['folder']/'baseline-model.json').exists()),None)
    if first is None:raise DataError('No fitted baseline models; inspect summary.json blocked reasons')
    timezone=load(folder/first['folder']/'baseline-model.json')['config']['timezone']
    start=forecast_start or str(pd.Timestamp.now(tz=timezone).date())
    end=forecast_end or str((pd.Timestamp(start)+pd.DateOffset(months=3)).date())
    a,b=stamp(start,timezone,True),stamp(end,timezone,True)
    if b<=a or (b-a).total_seconds()>366*86400:raise DataError('Choose a positive forecast period of at most one year')
    forecasts=[];daily_forecasts=[]
    for entry in summary['meters']:
        meter=entry['meter_id'];base=folder/entry['folder'];row={'meter_id':meter,'status':entry['status'],
            'is_virtual_meter':entry.get('is_virtual_meter',False),'meter_description':entry.get('meter_description',''),
            'future_optimized_kwh':None,'future_savings_kwh':None,'future_savings_cad':None}
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
            views.append({'meter_id':meter,'reporting_periods':r['periods'],'gates':r.get('gates',[])})
        else:row.update(reason=entry.get('reason'),savings_kwh=None,savings_cad=None)
        if (base/'baseline-model.json').exists():
            try:
                model=load(base/'baseline-model.json')
                grid=pd.date_range(a.tz_convert(timezone),b.tz_convert(timezone),freq='D',inclusive='left')
                conditions=[{'date':t.date().isoformat(),'meter_id':meter} for t in grid]
                f=forecast(model,conditions,start,end);write(f,base/'future-forecast')
                row.update(forecast_kwh=f['baseline_kwh'],forecast_baseline_cad=f['baseline_kwh']*rate if f['baseline_kwh'] is not None else None,forecast_status=f['status'])
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
                    row['future_optimized_kwh']=sum(eligible[d]['bms_estimated_kwh'] for d in expected)
        rows.append(row)
    selected=total_meters or [];totals=total(results,selected,rate)
    fm={f['meter_id']:f['baseline_kwh'] for f in forecasts}
    totals['forecast_kwh']=sum(fm[m] for m in selected) if selected and all(fm.get(m) is not None for m in selected) else None
    candidates=[r for r in rows if r['is_virtual_meter']]
    report={'rate_cad_per_kwh':rate,'currency':'CAD','start':summary['start'],'end_exclusive':summary['end_exclusive'],
            'forecast_start':start,'forecast_end_exclusive':end,'site_total':totals,'meters':rows,'unverified_boundary_candidates':candidates,
            'scope':'Calendar-only baseline comparison requiring site review. CAD is kWh difference times the user-assumed flat rate; excludes demand, taxes and fixed charges. Baseline forecasts describe no-intervention energy. Any optimized line is conditional on a supplied operating scenario; it is not future realized savings.'}
    save(folder/'overview.json',report);pd.DataFrame(rows).to_csv(folder/'meter-summary.csv',index=False)
    save(folder/'site-total.json',totals)
    pd.DataFrame(daily_forecasts).to_csv(folder/'forecast-daily.csv',index=False)
    pd.DataFrame(candidates).to_csv(folder/'boundary-candidates.csv',index=False)
    data=json.dumps({'summary':report,'views':views},allow_nan=False).replace('<','\\u003c')
    table=''.join('<tr>'+''.join('<td>'+escape('Unavailable' if row.get(k) is None else f'{row[k]:,.2f}' if isinstance(row[k],float) else str(row[k]))+'</td>' for k in ('meter_id','baseline_kwh','measured_kwh','savings_kwh','savings_cad','forecast_kwh','validation','reason'))+'</tr>' for row in rows)
    html='''<!doctype html><meta charset="utf-8"><title>Promenade energy overview</title><style>body{font:16px system-ui;margin:30px;color:#173247}section{padding:20px;background:#eff5f8;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ccd5dc;text-align:left}svg{width:100%;max-width:1100px}select{padding:10px}strong{font-size:21px}.scroll{overflow:auto}</style><h1>Promenade forecast and savings comparison</h1><p>Unverified engineering comparison. Negative savings mean consumption exceeded the predicted baseline.</p><section id="totals"></section><section id="candidates"></section><section><label>Meter <select id="meter"></select></label><p id="dates"></p><p id="numbers"></p><p>Teal: reporting-period baseline. Blue: measured consumption. Orange: future baseline forecast.</p><svg id="chart" viewBox="0 0 1000 340" aria-label="Daily energy forecast and measured consumption"></svg><p id="quality"></p></section><p id="scope"></p><div class="scroll"><table><thead><tr><th>Meter</th><th>Paired baseline kWh</th><th>Paired measured kWh</th><th>Complete savings kWh</th><th>Complete savings CAD</th><th>Future forecast kWh</th><th>Validation</th><th>Blocked reason</th></tr></thead><tbody>'''+table+'''</tbody></table></div><script>
const D='''+data+''';const S=D.summary;const fmt=n=>n==null?'Unavailable':Number(n).toLocaleString(undefined,{maximumFractionDigits:2});const el=id=>document.getElementById(id);const total=S.site_total;
el('totals').textContent='Site total: '+fmt(total.savings_kwh)+' kWh | CAD $'+fmt(total.savings_cad)+' at $'+S.rate_cad_per_kwh+'/kWh. '+(total.meter_ids.length?'Selected meters: '+total.meter_ids.join(', '):'Choose a verified non-overlapping meter boundary with --total-meters. All-meter sums would double-count overlapping loads.');
el('scope').textContent=S.scope;for(const v of D.views){const o=document.createElement('option');o.value=v.meter_id;o.textContent=v.meter_id;el('meter').append(o)}
el('candidates').textContent='Existing virtual totals (unverified candidates; never added together): '+S.unverified_boundary_candidates.map(m=>m.meter_id+': comparison '+fmt(m.savings_kwh)+' kWh / CAD $'+fmt(m.savings_cad)+'; future baseline '+fmt(m.forecast_kwh)+' kWh').join(' | ');
function draw(){const v=D.views.find(x=>x.meter_id===el('meter').value);if(!v)return;const m=S.meters.find(x=>x.meter_id===v.meter_id);el('dates').textContent='Comparison: '+S.start+' to '+S.end_exclusive+' exclusive. Forecast: '+S.forecast_start+' to '+S.forecast_end_exclusive+' exclusive.';el('numbers').textContent='Complete-period difference: '+fmt(m.savings_kwh)+' kWh | CAD $'+fmt(m.savings_cad)+' | Future baseline: '+fmt(m.forecast_kwh)+' kWh';el('quality').textContent=m.validation+' | '+v.gates.join('; ');
const past=v.reporting_periods||[],future=v.forecast_periods||[];const series=[[past,'baseline_kwh','#087d80'],[past,'reporting_kwh','#3265ad'],[future,'baseline_kwh','#d57b16']];const times=[...past,...future].map(r=>Date.parse(r.timestamp));if(!times.length)return;const lo=Math.min(...times),hi=Math.max(...times);const values=series.flatMap(([rs,k])=>rs.map(r=>r[k]).filter(x=>x!=null));const max=Math.max(1,...values)*1.1;const svg=el('chart');svg.innerHTML='';const make=(tag,attrs,text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,x]of Object.entries(attrs))n.setAttribute(k,x);if(text!=null)n.textContent=text;svg.append(n);return n};for(let i=0;i<5;i++){const y=280-i*60;make('line',{x1:75,x2:970,y1:y,y2:y,stroke:'#ccd5dc'});make('text',{x:5,y:y+5,'font-size':12},fmt(max*i/4))}make('text',{x:75,y:20,'font-size':13},'Daily kWh');for(const[rs,k,color]of series){let points=[];const flush=()=>{if(points.length)make('polyline',{points:points.join(' '),fill:'none',stroke:color,'stroke-width':2});points=[]};for(const r of rs){if(r[k]==null){flush();continue}const x=75+(Date.parse(r.timestamp)-lo)/Math.max(1,hi-lo)*895,y=280-r[k]/max*240;points.push(x+','+y);const dot=make('circle',{cx:x,cy:y,r:3,fill:color});const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=r.timestamp+' '+fmt(r[k])+' kWh';dot.append(title)}flush()}make('text',{x:75,y:315,'font-size':12},new Date(lo).toISOString().slice(0,10));make('text',{x:970,y:315,'font-size':12,'text-anchor':'end'},new Date(hi).toISOString().slice(0,10))}el('meter').addEventListener('change',draw);draw();</script>'''
    html=html.replace('Orange: future baseline forecast.','Orange: future baseline forecast. Purple: held-out BMS consumption estimate. Red: user-supplied optimized operating scenario. Fitted-history diagnostics are excluded from this chart.')
    html=html.replace("[future,'baseline_kwh','#d57b16']];const times=[...past,...future]", "[future,'baseline_kwh','#d57b16'],[v.response_periods||[],'bms_estimated_kwh','#8451a1'],[v.optimized_periods||[],'optimized_kwh','#b43f42']];const times=[...past,...future,...(v.response_periods||[]),...(v.optimized_periods||[])]")
    (folder/'index.html').write_text(html,encoding='utf-8')
    return report
