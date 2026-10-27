import argparse,json,hashlib
from pathlib import Path
import pandas as pd
from .data import DataError,load,save,records
from .regression import fit
from .workflow import forecast,savings,train_post
from .reporting import write
from .bms import aggregate
from .importers import prepare
from .conditions import scenario
from .batch import run as run_all
from .optimizer import prepare as optimizer_prepare


def csv(path):return pd.read_csv(path).to_dict('records')


def main():
    p=argparse.ArgumentParser(description='Standalone Option C regression and BMS-assisted reporting')
    commands=p.add_subparsers(dest='command',required=True)
    batch=commands.add_parser('run-all',help='Two-file daily workflow: all raw meters plus live BMS context, no separate calibration file')
    batch.add_argument('--config',type=Path,required=True);batch.add_argument('--meter',type=Path,required=True)
    batch.add_argument('--bms',type=Path,required=True);batch.add_argument('--kind',choices=['portal','utility','normalized','carma'],required=True)
    batch.add_argument('--carma-daily-kwh',action='store_true',help='Confirm CARMA export is daily Consumption kWh, not demand/cumulative/monthly data')
    batch.add_argument('--optimizer',type=Path,nargs='+',help='Optional optimizer CSV/JSON files with zone modes and valid/issue times')
    batch.add_argument('--weather',type=Path,help='External daily Celsius weather CSV')
    batch.add_argument('--quality-reviews',type=Path,help='Documented decisions keyed by data-quality event IDs')
    batch.add_argument('--start',required=True);batch.add_argument('--end',required=True);batch.add_argument('--output',type=Path,required=True)
    train=commands.add_parser('train');train.add_argument('--config',type=Path,required=True);train.add_argument('--meter',type=Path,required=True);train.add_argument('--output',type=Path,required=True)
    agg=commands.add_parser('prepare-bms');agg.add_argument('--config',type=Path,required=True);agg.add_argument('--bms',type=Path,nargs='+',required=True);agg.add_argument('--output',type=Path,required=True)
    op=commands.add_parser('prepare-optimizer');op.add_argument('--config',type=Path,required=True);op.add_argument('--bms',type=Path,required=True);op.add_argument('--optimizer',type=Path,nargs='+',required=True);op.add_argument('--output',type=Path,required=True)
    post=commands.add_parser('train-post');post.add_argument('--config',type=Path,required=True);post.add_argument('--meter',type=Path,required=True);post.add_argument('--bms-features',type=Path,required=True);post.add_argument('--output',type=Path,required=True)
    prep=commands.add_parser('prepare-meter');prep.add_argument('--input',type=Path,required=True);prep.add_argument('--kind',choices=['portal','utility','normalized','carma'],required=True);prep.add_argument('--meter-id',help='Default: all meters, kept separate');prep.add_argument('--weather',type=Path);prep.add_argument('--start');prep.add_argument('--end');prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--carma-daily-kwh',action='store_true',help='Confirm CARMA export is daily Consumption kWh')
    cond=commands.add_parser('conditions');cond.add_argument('--config',type=Path,required=True);cond.add_argument('--start',required=True);cond.add_argument('--end',required=True);cond.add_argument('--temperature-c',type=float,required=True);cond.add_argument('--covariates',type=Path);cond.add_argument('--output',type=Path,required=True)
    for name in ('forecast','savings'):
        sub=commands.add_parser(name);sub.add_argument('--model',type=Path,required=True);sub.add_argument('--data',type=Path,required=True);sub.add_argument('--start',required=True);sub.add_argument('--end',required=True);sub.add_argument('--output',type=Path,required=True)
        if name=='savings':
            sub.add_argument('--post-model',type=Path);sub.add_argument('--bms-features',type=Path);sub.add_argument('--adjustments',type=Path);sub.add_argument('--rate',type=Path)
    args=p.parse_args()
    try:
        if args.command=='run-all':
            result=run_all(args.meter,args.bms,load(args.config),args.start,args.end,args.output,args.kind,args.carma_daily_kwh,args.optimizer,
                quality_reviews=load(args.quality_reviews) if args.quality_reviews else None,weather_path=args.weather)
            print(json.dumps(result,indent=2))
        elif args.command=='prepare-optimizer':
            zones,features,groups,audit=optimizer_prepare(args.optimizer,args.bms,load(args.config))
            args.output.mkdir(parents=True,exist_ok=True)
            zones.to_csv(args.output/'zone-daily.csv',index=False);features.to_csv(args.output/'equipment-daily.csv',index=False)
            save(args.output/'equipment-groups.json',groups);save(args.output/'optimizer-audit.json',audit)
            print(json.dumps({'zone_count':audit['zone_count'],'equipment_group_count':audit['equipment_group_count'],
                              'unknown_mode_rows':audit['unknown_mode_rows'],'output':str(args.output)},indent=2))
        elif args.command=='train':
            result=fit(csv(args.meter),load(args.config));save(args.output,result)
            print(json.dumps({k:result[k] for k in ('status','model_id','family','valid_periods','holdout_metrics','gates')},indent=2))
        elif args.command=='train-post':
            result=train_post(csv(args.meter),csv(args.bms_features),load(args.config));save(args.output,result)
            print(json.dumps({k:result[k] for k in ('status','model_id','holdout_metrics','gates')},indent=2))
        elif args.command=='prepare-bms':
            f,audit=aggregate(args.bms,load(args.config));args.output.parent.mkdir(parents=True,exist_ok=True);f.to_csv(args.output,index=False);save(args.output.with_suffix('.audit.json'),audit);print(json.dumps(audit,indent=2))
        elif args.command=='conditions':
            rows=scenario(load(args.config),args.start,args.end,args.temperature_c,load(args.covariates) if args.covariates else None)
            args.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.output,index=False)
            print(json.dumps({'periods':len(rows),'scope':'User-assumed constant-weather scenario, not measured or fetched weather'}))
        elif args.command=='prepare-meter':
            f=prepare(args.input,args.meter_id,args.weather,args.kind,args.carma_daily_kwh)
            if args.start:f=f[f.date>=args.start]
            if args.end:f=f[f.date<args.end]
            args.output.parent.mkdir(parents=True,exist_ok=True);f.to_csv(args.output,index=False)
            save(args.output.with_suffix('.audit.json'),{'source_filename':args.input.name,'source_sha256':f.attrs['source_audit']['sha256'] if args.input.is_dir() else hashlib.sha256(args.input.read_bytes()).hexdigest(),'source_details':f.attrs.get('source_audit'),'weather_filename':args.weather.name if args.weather else None,'weather_sha256':hashlib.sha256(args.weather.read_bytes()).hexdigest() if args.weather else None,'meter_id':args.meter_id,'start':args.start,'end_exclusive':args.end,'valid_energy_rows':int(f.energy_kwh.notna().sum()),'excluded_energy_rows':int(f.energy_kwh.isna().sum()),'note':'No boundaries combined; source flags/raw values retained; confirm portal timezone independently'})
            print(json.dumps({'rows':len(f),'valid_energy_rows':int(f.energy_kwh.notna().sum()),'meter_id':args.meter_id,'output':str(args.output)}))
        else:
            model=load(args.model)
            if args.command=='forecast':result=forecast(model,csv(args.data),args.start,args.end)
            else:result=savings(model,csv(args.data),args.start,args.end,load(args.post_model) if args.post_model else None,csv(args.bms_features) if args.bms_features else None,load(args.adjustments) if args.adjustments else None,load(args.rate) if args.rate else None)
            write(result,args.output);print(json.dumps({k:v for k,v in result.items() if k!='periods'},indent=2))
    except (DataError,KeyError,ValueError) as exc:
        p.exit(2,json.dumps({'status':'blocked','reason':str(exc)})+'\n')


if __name__=='__main__':main()
