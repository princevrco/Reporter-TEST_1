"""CARMA Plus ReadingDate wide CSV, explicitly declared daily consumption kWh."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from .data import DataError


def source(path,daily_kwh=False):
    """Read a raw CSV or the exporter's verified per-meter daily files, never batches."""
    path=Path(path)
    if path.is_file():
        f=read(path,daily_kwh)
        return f,{'kind':'carma_csv','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'files':[{'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]}
    if not path.is_dir():raise DataError('CARMA source must be an existing CSV or export directory')
    files=sorted(p for p in path.rglob('*_daily_kWh.csv')
                 if '_batches' not in p.relative_to(path).parts and 'demand_kw' not in p.relative_to(path).parts)
    if not files:raise DataError('No per-meter *_daily_kWh.csv files found; select the consumption export directory')
    parts=[];lineage=[]
    for p in files:
        if p.is_symlink() or not p.resolve().is_relative_to(path.resolve()):raise DataError('CARMA source cannot escape its directory')
        qpath=p.with_suffix('.quality.json')
        if not qpath.is_file() or qpath.is_symlink():raise DataError('Each CARMA directory CSV requires its .quality.json file')
        q=json.loads(qpath.read_text(encoding='utf-8-sig'));sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if q.get('sha256')!=sha:raise DataError('CARMA CSV checksum mismatch: '+p.name)
        if (q.get('mode'),q.get('unit'),q.get('resolution'))!=('Consumption','kWh','daily'):
            raise DataError('CARMA quality metadata must confirm Consumption/kWh/daily: '+p.name)
        f=read(p,daily_kwh)
        if set(f.meter_id)!={str(q.get('meter'))}:raise DataError('CARMA CSV/quality meter identity mismatch')
        try:
            a=pd.Timestamp(q['requested_start']);b=pd.Timestamp(q['requested_end'])
            if a.tzinfo or b.tzinfo or a.time()!=datetime.min.time() or b.time()!=datetime.min.time() or b<=a:
                raise ValueError('bounds')
            expected=pd.date_range(a,b,freq='D',inclusive='left').date.astype(str).tolist()
        except (KeyError,ValueError,TypeError) as exc:raise DataError('Invalid CARMA quality date bounds') from exc
        if sorted(f.date.tolist())!=expected or q.get('rows')!=len(f):raise DataError('CARMA file does not match its declared daily grid')
        f['source_file']=p.relative_to(path).as_posix();parts.append(f)
        lineage.append({'path':p.relative_to(path).as_posix(),'sha256':sha,
                        'quality_sha256':hashlib.sha256(qpath.read_bytes()).hexdigest(),'meter_id':str(q['meter']),
                        'warnings':q.get('warnings',[]),'start':q['requested_start'],'end_exclusive':q['requested_end']})
    result=pd.concat(parts,ignore_index=True)
    if result.duplicated(['meter_id','date']).any():raise DataError('Overlapping CARMA meter/date files; remove duplicate inputs explicitly')
    from .data import digest
    return result.sort_values(['meter_id','date']),{'kind':'carma_export_directory','sha256':digest(lineage),
             'files':lineage,'file_count':len(files),'meter_count':int(result.meter_id.nunique()),
             'note':'Only per-meter daily files loaded; _batches and demand files excluded. Missing exports are not fabricated.'}


def read(path,daily_kwh=False):
    if not daily_kwh:
        raise DataError('CARMA raw CSV does not declare units or period length. Confirm Consumption, Daily and kWh with --carma-daily-kwh; never use demand or cumulative readings')
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        rows=[r for r in csv.reader(stream) if any(x.strip() for x in r)]
    if not rows or rows[0][0].strip()!='ReadingDate' or len(rows[0])<2:
        raise DataError('CARMA CSV requires ReadingDate followed by meter columns')
    header=rows[0];meters=[]
    for label in header[1:]:
        code,separator,description=label.partition(' - ')
        if not separator or not code.strip():raise DataError('CARMA meter header requires identifier - description')
        meters.append((code.strip(),label))
    if len({m[0] for m in meters})!=len(meters):raise DataError('Duplicate CARMA meter identifiers')
    footer=None;data=[]
    for r in rows[1:]:
        if len(r)==1 and r[0].startswith('Date Exported:'):
            if footer is not None:raise DataError('Duplicate CARMA export footer')
            try:footer=datetime.strptime(r[0].split(':',1)[1].strip(),'%m/%d/%Y %I:%M:%S %p')
            except ValueError as exc:raise DataError('Invalid CARMA export date') from exc
        else:
            if footer is not None:raise DataError('CARMA data found after export footer')
            if len(r)!=len(header):raise DataError('CARMA data width differs from header; reconcile truncated/misaligned rows')
            data.append(r)
    result=[];dates=set()
    for row in data:
        try:t=datetime.strptime(row[0].strip(),'%m/%d/%Y %I:%M:%S %p')
        except ValueError as exc:raise DataError('CARMA ReadingDate must use month/day/year and AM/PM time') from exc
        if t.time()!=datetime.min.time():raise DataError('CARMA daily periods must begin at midnight; do not import interval demand')
        date=t.date().isoformat()
        if date in dates:raise DataError('Duplicate CARMA daily dates')
        dates.add(date)
        for (code,label),text in zip(meters,row[1:]):
            value=pd.to_numeric(text,errors='coerce');flags=[]
            if not np.isfinite(value) or value<0:flags.append('invalid_or_missing_consumption')
            if footer and t.date()>=footer.date():flags.append('export_day_or_future_incomplete')
            result.append({'date':date,'meter_id':code,'meter_description':label,
                           'is_virtual_meter':code.lower().startswith('vm_'),
                           'energy_kwh':float(value) if np.isfinite(value) and not flags else None,
                           'raw_energy_kwh':float(value) if np.isfinite(value) else None,
                           'raw_source_value':text,'quality':'excluded' if flags else 'good',
                           'source_flags':';'.join(flags),'source_unit':'kWh',
                           'source_period':'daily_operator_confirmed',
                           'source_exported_at_local':footer.isoformat() if footer else None})
    if not result:raise DataError('CARMA CSV has no daily readings')
    return pd.DataFrame(result)
