"""Alectra daily workbook adapter: selected values, separate meters, source lineage."""
from pathlib import Path
import re,hashlib
import numpy as np
import pandas as pd
from .data import DataError


def table(raw, required):
    for i in range(min(40,len(raw))):
        vals=[str(x).strip() if pd.notna(x) else '' for x in raw.iloc[i].tolist()]
        if required<=set(vals):
            result=raw.iloc[i+1:].copy();result.columns=vals
            return result
    raise DataError('Workbook table header missing: '+', '.join(sorted(required)))


def meter_column(column):
    match=re.fullmatch(r'(.+?)\s+(?:\(kWh\)|kWh)',str(column).strip())
    return match.group(1).strip() if match else None


def old_note(note,meter):
    # Legacy combined-note rows can mention multiple meters. Do not mark another
    # meter incomplete merely because its neighbour has an incomplete reading.
    if not isinstance(note,str):return ''
    parts=[s.strip() for s in note.split(';')]
    return '; '.join(s for s in parts if meter in s or not re.search(r'PWST\d+',s))


def read(path):
    path=Path(path);sheets=pd.read_excel(path,sheet_name=None,header=None,engine='openpyxl')
    if 'Daily kWh' not in sheets:raise DataError('Alectra workbook needs Daily kWh sheet')
    selected={};source_ids={};source_audit={}
    if 'Source readings' in sheets:
        t=table(sheets['Source readings'],{'Record ID','Meter','Date','kWh','Use'})
        for row in t.to_dict('records'):
            if pd.isna(row.get('Record ID')):continue
            key=str(row['Record ID']).strip()
            if key in source_ids:raise DataError('Duplicate Alectra source record ID')
            source_ids[key]=row
        source_audit={'source_records':len(source_ids),
            'selected_source_records':sum(str(r['Use']).strip()=='Selected' for r in source_ids.values()),
            'unselected_source_records':sum(str(r['Use']).strip()!='Selected' for r in source_ids.values())}
    ids_used=set();rows=[]
    for sheet_name in ('Daily kWh','Recent daily'):
        if sheet_name not in sheets:continue
        f=table(sheets[sheet_name],{'Date'})
        cols=[(c,meter_column(c)) for c in f.columns if meter_column(c)]
        if not cols:raise DataError('Alectra worksheet has no meter kWh columns')
        if len({m for _,m in cols})!=len(cols):raise DataError('Duplicate Alectra meter columns')
        for row in f.to_dict('records'):
            dt=row.get('Date')
            if pd.isna(dt):continue
            try:date=pd.Timestamp(dt)
            except (ValueError,TypeError):continue
            if pd.isna(date):continue
            if date.tzinfo is not None or date!=date.normalize():raise DataError('Utility daily dates must be local dates at midnight')
            day=str(date.date())
            for col,meter in cols:
                original=row.get(col);value=pd.to_numeric(original,errors='coerce')
                missing=pd.isna(original)
                if not missing and (not np.isfinite(value) or value<0):raise DataError('Invalid Alectra kWh: '+meter+' '+day)
                note=row.get(meter+' review')
                if not isinstance(note,str):note=old_note(row.get('Availability / quality note',''),meter)
                note=note.strip() if isinstance(note,str) else ''
                record=row.get(meter+' record');record=str(record).strip() if pd.notna(record) else None
                source_filename=source_type=None
                if source_ids and not missing:
                    if record is None or record not in source_ids:raise DataError('Alectra selected value lacks a valid source record ID')
                    sr=source_ids[record]
                    if str(sr['Use']).strip()!='Selected' or str(sr['Meter']).strip()!=meter or str(pd.Timestamp(sr['Date']).date())!=day:
                        raise DataError('Alectra source record selection/date/meter mismatch')
                    sv=pd.to_numeric(sr['kWh'],errors='coerce')
                    if not np.isfinite(sv) or abs(float(sv)-float(value))>1e-7:raise DataError('Alectra daily value differs from selected source reading')
                    ids_used.add(record);source_filename=sr.get('Source filename');source_type=sr.get('Source type')
                # New compiled workbook: low/revised are review flags only. Keep values.
                # Legacy snapshot: explicit incomplete-day observations are not full daily energy.
                incomplete=bool(note) and ('incomplete' in note.lower()) and not source_ids
                review=bool(note) and not missing
                usable=not missing and not incomplete
                item={'date':day,'meter_id':meter,'meter_description':'Alectra '+meter+'; physical boundary requires review',
                    'is_virtual_meter':False,'energy_kwh':float(value) if usable else None,
                    'raw_energy_kwh':float(value) if not missing else None,'quality':'good' if usable else 'missing' if missing else 'excluded',
                    'source_flags':note,'source_review_required':review,'source_sheet':sheet_name,
                    'source_record_id':record,'source_filename':source_filename,'source_type':source_type}
                key=(meter,day)
                if key in selected:
                    prev=selected[key]
                    a,b=prev['raw_energy_kwh'],item['raw_energy_kwh']
                    if a is not None and b is not None and a!=b:raise DataError('Conflicting selected utility readings: '+meter+' '+day)
                    if b is None and a is not None:continue
                selected[key]=item
    if not selected:raise DataError('No daily utility readings')
    if source_ids:
        declared={key for key,r in source_ids.items() if str(r['Use']).strip()=='Selected'}
        if ids_used!=declared:raise DataError('Selected source records and Daily kWh table do not reconcile')
    result=pd.DataFrame(selected.values()).sort_values(['meter_id','date']).reset_index(drop=True)
    result.attrs['source_audit']={'kind':'alectra_daily_workbook','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'filename':path.name,'meters':sorted(result.meter_id.unique()),'rows':len(result),'selection':'Daily kWh selected values; Source readings checked by ID when present',
        'missing_values':'Unavailable, never zero','review_policy':'Retain flagged/revised compiled-workbook values for review; no inferred corrections',**source_audit}
    return result
