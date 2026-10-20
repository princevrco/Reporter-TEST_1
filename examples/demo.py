"""Synthetic standalone demonstration, not Promenade data or savings."""
import argparse,copy
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'tests'))
from fixtures import config,baseline_rows,post_rows
from option_c.data import save,frame
from option_c.regression import fit,predict
from option_c.workflow import forecast,savings,train_post
from option_c.reporting import write
from option_c.conditions import scenario
from option_c.batch import run as run_all

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--optimizer',action='store_true');args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
c=config();c['site_id']='promenade'
base_rows=baseline_rows();model=fit(base_rows,c);save(args.output/'baseline.json',model);save(args.output/'config.json',c)
pd.DataFrame(base_rows).to_csv(args.output/'baseline.csv',index=False)
cond=scenario(c,'2026-04-20','2026-04-27',20)
pd.DataFrame(cond).to_csv(args.output/'conditions.csv',index=False)
write(forecast(model,cond,'2026-04-20','2026-04-27'),args.output/'forecast')
f=frame(cond,c);pred,_,_,_=predict(model,f)
report=copy.deepcopy(cond)
for r,p in zip(report,pred):r['energy_kwh']=float(p-100)
pd.DataFrame(report).to_csv(args.output/'reporting.csv',index=False)
measured=savings(model,report,'2026-04-20','2026-04-27')
write(measured,args.output/'measured-savings')
cal,features=post_rows();post=train_post(cal,features,c);save(args.output/'post-model.json',post)
meter,bms=post_rows(7,'2026-04-20')
for r in meter[1:]:r['energy_kwh']=None
assisted=savings(model,meter,'2026-04-20','2026-04-27',post,bms)
write(assisted,args.output/'bms-assisted')
print({'synthetic':True,'measured_demo_savings_kwh':measured['savings_kwh_complete_period'],'bms_estimated_periods':assisted['bms_estimated_periods'],'output':str(args.output)})
if args.optimizer:
    oc=copy.deepcopy(c);oc['optimizer']={'bin_minutes':60,'minimum_training_days':60,'calendar_features':True}
    rng=np.random.default_rng(57);meter_rows=[];optimizer_rows=[];bms_rows=[]
    for mid in ('demo_meter_A','demo_meter_B'):meter_rows.extend({**r,'meter_id':mid} for r in base_rows)
    pattern=[rng.integers(0,7,4).tolist() for _ in range(100)]
    for i,day in enumerate(pd.date_range('2026-01-01',periods=120,freq='D',tz=oc['timezone'])):
        times=pd.date_range(day,day+pd.DateOffset(days=1),freq='1h',inclusive='left')
        h1,c1,h2,c2=pattern[i%100]
        if i%4==0:h1=c1=0
        if i%5==0:h2=c2=0
        for j,t in enumerate(times):
            bms_rows.append({'timestamp':t.isoformat(),'fan':1})
            for zone,h,co in [('demo_zone_1',h1,c1),('demo_zone_2',h2,c2)]:
                mode=1 if j<h else -1 if j<h+co else 0
                optimizer_rows.append({'timestamp':t.isoformat(),'issued_at':t.isoformat(),'zone_id':zone,'achieved_mode':mode})
        for mid,scale in [('demo_meter_A',1),('demo_meter_B',2)]:
            meter_rows.append({'date':day.date().isoformat(),'meter_id':mid,'energy_kwh':40*len(times)+scale*(8*h1+12*c1+4*h2+6*c2),'quality':'good'})
    mp=args.output/'raw-meters.csv';op=args.output/'optimizer.csv';bp=args.output/'live-bms.csv'
    pd.DataFrame(meter_rows).to_csv(mp,index=False);pd.DataFrame(optimizer_rows).to_csv(op,index=False);pd.DataFrame(bms_rows).to_csv(bp,index=False)
    save(args.output/'optimizer-config.json',oc)
    result=run_all(mp,bp,oc,'2026-04-11','2026-05-01',args.output/'optimizer-results','normalized',optimizer_paths=[op])
    print({'synthetic':True,'optimizer_demo_meters':result['meter_count'],
           'blocked_meters':result['blocked_meter_count'],'output':str(args.output/'optimizer-results')})
