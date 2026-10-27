"""Merge successive exports without summing overlapping energy or hiding revisions."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import hashlib
import shutil
import pandas as pd
from .carma import source
from .data import DataError


def extract_meter(archive,destination):
    destination=Path(destination)
    with ZipFile(archive) as z:
        files=[i for i in z.infolist() if not i.is_dir()]
        if len(files)>20000 or sum(i.file_size for i in files)>500_000_000:
            raise DataError('Meter archive exceeds limits')
        seen=set()
        for info in files:
            name=info.filename.replace('\\','/');target=(destination/name).resolve()
            if not target.is_relative_to(destination.resolve()) or target in seen:
                raise DataError('Unsafe or duplicate meter ZIP path')
            if not name.endswith(('_daily_kWh.csv','_daily_kWh.quality.json','_daily_kWh.csv.quality.json')):
                raise DataError('Unexpected meter ZIP file: '+name)
            seen.add(target);target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)


def merge_meters(paths,policy='error',kind='carma'):
    if policy not in ('error','prefer-last'):raise DataError('Unknown overlap policy')
    parts=[];sources=[]
    for i,p in enumerate(map(Path,paths)):
        chosen=('utility' if p.suffix.lower() in ('.xlsx','.xls') else 'carma') if kind=='auto' else kind
        if chosen=='carma':
            with TemporaryDirectory(prefix='meter-input-') as temp:
                folder=p
                if p.suffix.lower()=='.zip':
                    folder=Path(temp);extract_meter(p,folder)
                f,a=source(folder,True)
        else:
            from .importers import prepare
            f=prepare(p,kind=chosen)
            a=f.attrs.get('source_audit',{'kind':chosen,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        sources.append({'input_order':i,'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else a['sha256'],'validation':a})
        f=f.copy();f['_input_order']=i;parts.append(f)
    if not parts:raise DataError('No meter exports supplied')
    all_rows=pd.concat(parts,ignore_index=True)
    conflicts=[];duplicates=0
    dup=all_rows[all_rows.duplicated(['meter_id','date'],keep=False)]
    for (meter,day),g in dup.groupby(['meter_id','date'],sort=False):
        duplicates+=len(g)-1
        values=g[['energy_kwh','quality']].drop_duplicates()
        if len(values)>1:conflicts.append({'meter_id':str(meter),'date':str(day),'input_orders':g._input_order.tolist(),
            'values':[None if pd.isna(v) else float(v) for v in g.energy_kwh]})
    if conflicts and policy=='error':raise DataError(f'{len(conflicts)} revised meter/date readings; inspect exports or explicitly use --overlap-policy prefer-last. First: {conflicts[0]}')
    merged=all_rows.sort_values('_input_order',kind='stable').drop_duplicates(['meter_id','date'],keep='last').drop(columns='_input_order')
    return merged.sort_values(['meter_id','date']),{'sources':sources,'overlap_policy':policy,
        'duplicate_rows_collapsed':duplicates,'revised_readings':conflicts,'row_count':len(merged),
        'note':'Inputs ordered oldest to newest. Overlapping dates are never summed. Prefer-last uses supplied order, not filesystem timestamps.'}


def merge_bms(paths,policy='error'):
    if policy not in ('error','prefer-last'):raise DataError('Unknown overlap policy')
    parts=[];sources=[]
    for p in map(Path,paths):
        f=pd.read_csv(p)
        if f.empty or 'timestamp' not in f:raise DataError('BMS input needs timestamp and observations')
        if any(pd.isna(t) or pd.Timestamp(t).tzinfo is None for t in f.timestamp):raise DataError('BMS timestamps require explicit offsets')
        f['timestamp']=pd.to_datetime(f.timestamp,utc=True,format='mixed');parts.append(f)
        sources.append({'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(f)})
    if not parts:raise DataError('No BMS inputs supplied')
    f=pd.concat(parts,ignore_index=True);conflicts=[]
    for t,g in f[f.timestamp.duplicated(keep=False)].groupby('timestamp',sort=False):
        bad=g.drop(columns='timestamp').nunique(dropna=True)
        for col in bad.index[bad>1]:conflicts.append({'timestamp':t.isoformat(),'column':str(col)})
    if conflicts and policy=='error':raise DataError(f'{len(conflicts)} conflicting BMS cells; inspect exports or explicitly use --overlap-policy prefer-last. First: {conflicts[0]}')
    merged=f.groupby('timestamp',sort=True).last().reset_index()
    return merged,{'sources':sources,'overlap_policy':policy,'conflicting_cells':conflicts,
        'duplicate_timestamp_rows':len(f)-f.timestamp.nunique(),'note':'Last non-null cell in supplied file order; no temporal interpolation or gap filling.'}
