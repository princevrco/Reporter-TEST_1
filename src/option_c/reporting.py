from html import escape
import json
from pathlib import Path
import pandas as pd
from .data import save


def write(result,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    save(folder/'result.json',result)
    pd.DataFrame(result.get('periods',[])).to_csv(folder/'periods.csv',index=False)
    data=json.dumps(result,allow_nan=False).replace('<','\\u003c')
    title='Energy forecast' if result['schema']=='option_c.forecast.v1' else 'Energy savings analysis'
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Option C report</title><style>
body{font:15px system-ui,sans-serif;background:#f2f5f7;color:#172936;margin:0}main{max-width:1100px;margin:auto;padding:28px}h1{margin:0 0 8px;font-size:28px}p{line-height:1.55}.card{background:white;border:1px solid #dbe3e9;border-radius:10px;padding:20px;margin:16px 0}.stats{display:flex;gap:20px;flex-wrap:wrap}.stats>div{flex:1;min-width:180px}.value{font-size:26px;font-weight:650;margin-top:5px}.muted{color:#536575}.gates{background:#fff4dd}label{display:inline-block;margin:0 12px 12px 0}input,button{font:inherit;padding:8px;border:1px solid #acbac5;border-radius:4px}button{background:#183e54;color:white;cursor:pointer}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:8px;border-bottom:1px solid #e4eaee}.scroll{overflow:auto}#tip{min-height:24px;color:#334f62}.legend span{margin-right:18px}.dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:5px}.footer{font-size:12px;color:#5b6e7b}
</style><main><h1>''' +escape(title)+'''</h1><p id="scope"></p><div class="card gates"><strong id="status"></strong><p id="gates"></p></div>
<div class="card"><label>View from <input id="from" type="date"></label><label>View until, exclusive <input id="until" type="date"></label><button id="filter">Update view</button><p class="muted">Dates filter this existing report. Rerun the command to predict a different period or recalculate its uncertainty.</p><div class="stats" id="stats"></div></div>
<div class="card"><div class="legend"><span><i class="dot" style="background:#087d80"></i>Baseline</span><span><i class="dot" style="background:#3265ad"></i>Meter</span><span><i class="dot" style="background:#d57b16"></i>BMS estimate</span></div><svg id="chart" viewBox="0 0 1000 330" role="img" aria-label="Energy consumption by period"></svg><div id="tip"></div></div>
<div class="card"><strong>Full analysis window</strong><p id="full"></p><p id="uncertainty" class="muted"></p></div><div class="card scroll"><table><thead><tr><th>Period</th><th>Baseline kWh</th><th>Reporting kWh</th><th>Source</th><th>Savings kWh</th></tr></thead><tbody id="table"></tbody></table></div><p class="footer">Average power and energy are different quantities. This report does not establish billing peak demand, full invoice savings or standards certification. Keep site reports private.</p></main>
<script>const R='''+data+''';const $=id=>document.getElementById(id);const fmt=x=>x==null?'Unavailable':Number(x).toLocaleString(undefined,{maximumFractionDigits:2});const tz=R.timezone||'America/Toronto';const day=t=>new Intl.DateTimeFormat('en-CA',{timeZone:tz,year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(t));const rows=R.periods||[];
$('scope').textContent=R.scope;$('status').textContent=R.status;$('gates').textContent=(R.gates||[]).join('; ');$('from').value=day(R.start);$('until').value=day(R.end_exclusive);$('full').textContent='Window: '+R.start+' to '+R.end_exclusive+'; expected periods: '+R.expected_periods+'; complete-period savings: '+fmt(R.savings_kwh_complete_period)+' kWh.';$('uncertainty').textContent='Full-window model band: '+JSON.stringify(R.aggregate_model_band_95_savings_kwh||R.aggregate_model_band_95_kwh||null)+'. '+(R.band_scope||'Approximate model/residual uncertainty; excludes structural, measurement and causal uncertainty.');
function update(){const a=$('from').value,b=$('until').value,s=rows.filter(r=>day(r.timestamp)>=a&&day(r.timestamp)<b);const paired=s.filter(r=>r.savings_kwh!=null);const sum=(rs,k)=>rs.reduce((v,r)=>v+(r[k]||0),0);$('stats').innerHTML='';for(const [label,value]of [['Available baseline kWh',fmt(sum(s,'baseline_kwh'))],['Paired savings kWh',paired.length?fmt(sum(paired,'savings_kwh')):'Unavailable'],['Selected / paired periods',s.length+' / '+paired.length]]){const d=document.createElement('div');const n=document.createElement('div');n.textContent=label;const v=document.createElement('div');v.className='value';v.textContent=value;d.append(n,v);$('stats').append(d)}
const vals=s.flatMap(r=>[r.baseline_kwh,r.reporting_kwh]).filter(v=>v!=null);const ymax=Math.max(1,...vals)*1.08;const svg=$('chart');svg.innerHTML='';const NS='http://www.w3.org/2000/svg';const node=(tag,attrs,text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!=null)n.textContent=text;svg.append(n);return n};for(let i=0;i<=4;i++){const y=280-i*60;node('line',{x1:70,x2:970,y1:y,y2:y,stroke:'#e1e7eb'});node('text',{x:62,y:y+4,'text-anchor':'end','font-size':12,fill:'#526776'},fmt(i*ymax/4))}node('text',{x:70,y:20,'font-size':12},'kWh per period');
const points=s.length>800?s.filter((_,i)=>i%Math.ceil(s.length/800)===0):s;for(const[k,color]of [['baseline_kwh','#087d80'],['reporting_kwh','#3265ad']]){let part=[];const draw=()=>{if(part.length)node('polyline',{points:part.join(' '),fill:'none',stroke:color,'stroke-width':2});part=[]};points.forEach((r,i)=>{const x=70+i*900/Math.max(1,points.length-1),v=r[k];if(v==null){draw();return}const y=280-v/ymax*240;part.push(x+','+y);if(k==='reporting_kwh'&&r.reporting_source==='bms_model')node('circle',{cx:x,cy:y,r:3,fill:'#d57b16'});const c=node('circle',{cx:x,cy:y,r:6,fill:'transparent'});c.addEventListener('mouseover',()=>{$('tip').textContent=day(r.timestamp)+' | Baseline '+fmt(r.baseline_kwh)+' kWh | Reporting '+fmt(r.reporting_kwh)+' kWh | '+(r.reporting_source||'forecast')})});draw()}if(points.length){node('text',{x:70,y:310,'font-size':12},day(points[0].timestamp));node('text',{x:970,y:310,'font-size':12,'text-anchor':'end'},day(points[points.length-1].timestamp))}
$('table').innerHTML='';for(const r of s.slice(0,200)){const tr=document.createElement('tr');for(const v of [r.timestamp,fmt(r.baseline_kwh),fmt(r.reporting_kwh),r.reporting_source||'forecast',fmt(r.savings_kwh)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}$('table').append(tr)}$('tip').textContent=s.length?'Hover over a point. Table shows up to 200 rows; CSV/JSON contain all periods.':'No periods in this view.'} $('filter').addEventListener('click',update);update();</script></html>'''
    if 'zone_meter_effects' in result:
        effects=result['zone_meter_effects'];profiles={p['zone_id']:p for p in result.get('optimizer_duration_summary',[])}
        html=html.replace('const R='+data+';','const R=__OPTION_C_REPORT_DATA__;')
        html=html.replace('r.baseline_kwh,r.reporting_kwh','r.baseline_kwh,r.reporting_kwh,r.operation_model_kwh')
        html=html.replace("['reporting_kwh','#3265ad']]","['reporting_kwh','#3265ad'],['operation_model_kwh','#8451a1']]")
        html=html.replace('BMS estimate</span></div>','BMS estimate</span><span><i class="dot" style="background:#8451a1"></i>Operation model (estimate)</span></div>')
        html=html.replace('<th>Source</th>','<th>Operation model kWh</th><th>Source</th>')
        html=html.replace('fmt(r.reporting_kwh),r.reporting_source','fmt(r.reporting_kwh),fmt(r.operation_model_kwh),r.reporting_source')
        extra='<div class="card scroll"><h2>Optimizer modes and meter associations</h2>'
        extra+='<p>Full analysis window. Duration shares describe estimated thermal ON time; they are not energy shares. Mode 0 does not establish zero fan or electrical load. Hysteresis screens are sensitivity checks, not probabilities.</p>'
        extra+='<p>'+escape(effects.get('scope',''))+' Status: '+escape(effects.get('status',''))+'. '+escape(effects.get('reason',''))+'</p>'
        extra+='<p>Review findings: '+escape('; '.join(effects.get('gates',[])))+'</p>'
        extra+='<table><thead><tr><th>Zone</th><th>Estimated ON hours</th><th>Uncertain hours</th><th>ON-hour sensitivity range</th><th>Mean complete ON run (h)</th><th>ON-time share %</th><th>Coverage %</th><th>Individual meter association %</th><th>Group association %</th></tr></thead><tbody>'
        for effect in effects.get('zone_effects',[]):
            profile=profiles.get(effect['zone_id'],{})
            bounds=(f"{profile['on_hours_lower_sensitivity']:.2f} to {profile['on_hours_upper_sensitivity']:.2f}"
                    if 'on_hours_lower_sensitivity' in profile else None)
            values=[effect['zone_id'],profile.get('on_hours_estimated'),profile.get('uncertain_hours'),bounds,profile.get('mean_uncensored_on_run_hours'),profile.get('share_of_zone_on_time_pct'),
                    100*profile['coverage_fraction'] if 'coverage_fraction' in profile else None,
                    effect.get('associated_pct_of_meter_kwh'),effect.get('group_associated_pct_of_meter_kwh')]
            extra+='<tr>'+''.join('<td>'+escape('Unavailable' if v is None else f'{v:.2f}' if isinstance(v,(float,int)) else str(v))+'</td>' for v in values)+'</tr>'
        extra+='</tbody></table><p class="muted">Shared equipment and co-moving zones may only have group associations. Missing values are unavailable, not zero. These percentages do not establish zone-level savings. Date filters above affect the energy view only.</p></div>'
        html=html.replace('<p class="footer">',extra+'<p class="footer">')
        html=html.replace('const R=__OPTION_C_REPORT_DATA__;','const R='+data+';')
    if result.get('release_status')=='withheld_requires_review':
        html=html.replace('Energy savings analysis','Energy comparison — withheld')
        html=html.replace('<p class="footer">','<div class="card gates"><strong>Results withheld pending review</strong><p>Unresolved data or model findings prevent releasing savings. Diagnostic differences remain in JSON/CSV for investigation and must not be quoted as achieved savings.</p></div><p class="footer">')
    (folder/'report.html').write_text(html,encoding='utf-8')
