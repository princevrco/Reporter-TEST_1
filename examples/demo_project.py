"""Generate a fully synthetic, complete-row end-to-end demonstration."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from option_c.strict_history import daily


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('results/synthetic-project'))
    a=parser.parse_args()
    if a.output.exists():parser.error('Choose a new output directory')
    inputs=a.output/'inputs';inputs.mkdir(parents=True);rng=np.random.default_rng(15);n=240
    times=pd.date_range('2025-01-01',periods=n*96,freq='15min',tz='UTC');idx=np.arange(len(times))//96
    setpoints=rng.uniform(21.5,27,n);outdoor=rng.uniform(23,31,n);temp=np.empty(len(times));temp[0]=24
    for i in range(len(temp)-1):temp[i+1]=temp[i]+.25*(.04+.08*(outdoor[idx[i]]-temp[i])-.8*max(temp[i]-setpoints[idx[i]],0))
    raw=pd.DataFrame({'timestamp':times,'A_ZoneTemp':temp,'A_CoolSetpointU_AV':setpoints[idx]})
    dates=pd.date_range('2025-01-01',periods=n);weather=pd.DataFrame({'date':dates.date.astype(str),'temperature_c':outdoor})
    groups={'A':{'temperature':'A_ZoneTemp','cool_setpoint':'A_CoolSetpointU_AV'}}
    ff,_=daily(raw,groups,weather,'UTC');meter=pd.DataFrame({'date':ff.date,'meter_id':'SYNTHETIC_METER','energy_kwh':480+5*ff.cool_degree_hours+ff.cool_proxy_hours,'quality':'good'})
    end=str((dates[-1]+pd.Timedelta(days=1)).date());finish=str((pd.Timestamp(end)+pd.Timedelta(days=3)).date())
    raw.to_csv(inputs/'bms.csv',index=False);weather.to_csv(inputs/'weather.csv',index=False);meter.to_csv(inputs/'meter.csv',index=False)
    pd.DataFrame({'date':pd.date_range(end,finish,inclusive='left').date.astype(str),'temperature_c':28.}).to_csv(inputs/'future.csv',index=False)
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'config/promenade-test.json').read_text(encoding='utf-8'))
    cfg.update(timezone='UTC',synthetic=True,intervention_timestamp='2025-07-01T00:00:00+00:00',baseline_end_exclusive='2025-07-01T00:00:00+00:00')
    cfg['baseline_options'].update(windows_days=[None],estimators=['ols'],annual_calendar_candidates=False,bootstrap_repeats=0)
    (inputs/'config.json').write_text(json.dumps(cfg),encoding='utf-8')
    for kind,sp in [('old_policy',22.),('optimized_policy',26.)]:
        policy={'kind':kind,'reference':'Synthetic illustrative policy only','reviewed':True,'assets':{'A':{'weekdays':list(range(7)),'occupied_start':'00:00','occupied_end':'00:00',
                'occupied_cooling_c':sp,'unoccupied_cooling_c':sp,'occupied_heating_c':'OFF','unoccupied_heating_c':'OFF','control_margin_c':0.}}}
        (inputs/(kind+'.json')).write_text(json.dumps(policy),encoding='utf-8')
    command=[sys.executable,'-m','option_c.project_cli','--meter',str(inputs/'meter.csv'),'--meter-kind','normalized','--bms',str(inputs/'bms.csv'),
             '--weather',str(inputs/'weather.csv'),'--future-weather',str(inputs/'future.csv'),'--future-weather-basis','Synthetic constant-weather scenario',
             '--config',str(inputs/'config.json'),'--old-policy',str(inputs/'old_policy.json'),'--optimized-policy',str(inputs/'optimized_policy.json'),
             '--start','2025-07-01','--end',end,'--forecast-start',end,'--forecast-end',finish,'--output',str(a.output/'run')]
    subprocess.run(command,check=True)
    print('SYNTHETIC ONLY: '+str(a.output/'run/project.html'))


if __name__=='__main__':main()
