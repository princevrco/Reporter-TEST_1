"""Investigate named virtual totals without inferring electrical boundaries."""
import pandas as pd
import math
from .data import DataError


def audit(rows, relationships, tolerance_kwh=0.01):
    """Relationships are proposed identities, never permission to repair a meter."""
    if not math.isfinite(tolerance_kwh) or tolerance_kwh < 0:raise DataError('Reconciliation tolerance must be finite and nonnegative')
    if rows.duplicated(['date','meter_id']).any():raise DataError('Duplicate reconciliation meter/date')
    table=rows.pivot(index='date',columns='meter_id',values='energy_kwh')
    output=[]
    for relation in relationships:
        parent=relation['parent'];children=relation['children']
        if not children or len(set(children))!=len(children) or parent in children:
            raise DataError('Reconciliation children must be unique and exclude parent')
        selected=table.reindex(columns=[parent]+children)
        for day,values in selected.iterrows():
            complete=values.notna().all() and all(math.isfinite(v) and v>=0 for v in values)
            total=float(values[children].sum()) if complete else None
            residual=float(values[parent]-total) if complete else None
            output.append({'date':day,'parent':parent,'children':';'.join(children),
                'parent_kwh':None if pd.isna(values[parent]) else float(values[parent]),
                'children_sum_kwh':total,'difference_kwh':residual,
                'status':'missing_readings' if not complete else 'consistent' if abs(residual)<=tolerance_kwh else 'inconsistent',
                'boundary_verified':False,'repair_applied':False})
    return output
