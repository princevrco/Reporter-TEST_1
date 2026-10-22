"""Reproducible local/Runpod Pod test; no CARMA connection or BACnet writes."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

from option_c.batch import run
from option_c.data import load, save


def extract_meter(archive, destination):
    """Extract only bounded daily meter CSV/quality files inside a fresh directory."""
    with ZipFile(archive) as z:
        members=[i for i in z.infolist() if not i.is_dir()]
        if len(members)>20000 or sum(i.file_size for i in members)>500_000_000:
            raise ValueError('Meter archive exceeds test limits')
        seen=set()
        for i in members:
            name=i.filename.replace('\\','/')
            target=(destination/name).resolve()
            if not target.is_relative_to(destination.resolve()) or target in seen:
                raise ValueError('Unsafe or duplicate archive path')
            seen.add(target)
            if not name.endswith(('_daily_kWh.csv','_daily_kWh.quality.json')):
                # The CARMA exporter names metadata as CSV-stem.quality.json.
                if not name.endswith('_daily_kWh.csv.quality.json'):
                    raise ValueError('Unexpected file in meter archive: '+name)
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(i) as source,target.open('wb') as dest:shutil.copyfileobj(source,dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meter',type=Path,default=Path('local/consumption_kwh.zip'))
    p.add_argument('--bms',type=Path,default=Path('local/live_bms_history.csv'))
    p.add_argument('--config',type=Path,default=Path('config/promenade-test.json'))
    p.add_argument('--start',default='2026-08-15')
    p.add_argument('--end',default='2026-10-21',help='Exclusive local date')
    p.add_argument('--output',type=Path,default=Path('results/promenade-test'))
    p.add_argument('--quality-reviews',type=Path,help='Documented meter-quality review ledger')
    p.add_argument('--rate',type=float,default=.12,help='Assumed CAD per kWh')
    p.add_argument('--forecast-start');p.add_argument('--forecast-end',help='Exclusive local date')
    p.add_argument('--total-meters',nargs='+',help='Non-overlapping meters selected for site total')
    args=p.parse_args()
    for file in (args.meter,args.bms,args.config):
        if not file.exists():p.error('Missing input: '+str(file))
    if args.output.exists():p.error('Output already exists; choose a new --output directory')
    cfg=load(args.config)
    print('Engineering comparison: confirmed August 15 control start; calendar-only baseline.',flush=True)
    print('All source meters analyzed separately. Zone effects unavailable. Flat-rate CAD comparison will be generated.',flush=True)
    with tempfile.TemporaryDirectory(prefix='promenade-meter-') as temp:
        meter=args.meter
        if meter.suffix.lower()=='.zip':
            meter=Path(temp);extract_meter(args.meter,meter)
            print('Meter archive extracted. Validating daily files and fitting every meter; this can take several minutes.',flush=True)
        summary=run(meter,args.bms,cfg,args.start,args.end,args.output,kind='carma',carma_daily_kwh=True,
                    quality_reviews=load(args.quality_reviews) if args.quality_reviews else None,
                    progress=lambda entry,index,total:print(f"[{index}/{total}] {entry['meter_id']}: {entry['status']}",flush=True))
    entries=summary['meters']
    import pandas as pd
    pd.DataFrame([{k:v for k,v in e.items() if k!='holdout_metrics'} |
                  {'gates':';'.join(e.get('gates',[])),
                   'holdout_cvrmse_pct':e.get('holdout_metrics',{}).get('cvrmse_pct'),
                   'holdout_nmbe_pct':e.get('holdout_metrics',{}).get('nmbe_pct')}
                  for e in entries]).to_csv(args.output/'meter-summary.csv',index=False)
    save(args.output/'test-assumptions.json',{
        'control_start_provisional':not cfg.get('review',{}).get('intervention_confirmed',False),
        'baseline_end_exclusive':cfg.get('baseline_end_exclusive'),
        'baseline_exclusion_reason':'Exclude July onward BMS/commissioning history pending verified control history',
        'zone_effects':'unavailable: no eligible October optimizer export supplied',
        'cost_savings':f'Engineering flat-rate energy comparison at CAD {args.rate}/kWh; excludes other bill components',
        'claim':'Engineering comparison only; not verified energy savings'})
    from option_c.overview import build
    build(args.output,args.rate,args.forecast_start,args.forecast_end,args.total_meters)
    print('Forecast and savings dashboard: '+str(args.output/'index.html'),flush=True)
    print(json.dumps({'meter_count':summary['meter_count'],'blocked_meter_count':summary['blocked_meter_count'],
                      'status':summary['status'],'output':str(args.output)},indent=2))


if __name__=='__main__':main()
