"""Read-only evidence export and diagnostic baseline exclusion sensitivity."""
import argparse
import hashlib
import math
from pathlib import Path
import pandas as pd
from option_c.data import load,save,records
from option_c.quality import assess
from option_c.reconciliation import audit
from option_c.resilient import fit,sensitivity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meter',type=Path,required=True,help='Normalized merged-meter.csv from an existing analysis')
    p.add_argument('--config',type=Path,default=Path('config/promenade-test.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',default='2026-08-15');p.add_argument('--end',default='2026-10-21')
    p.add_argument('--rate',type=float,default=.12)
    a=p.parse_args()
    if a.output.exists():p.error('Choose a new output directory; previous evidence is preserved')
    if not math.isfinite(a.rate) or a.rate<0:p.error('Rate must be finite and nonnegative')
    if pd.Timestamp(a.end)<=pd.Timestamp(a.start):p.error('End must follow start')
    cfg=load(a.config);raw=pd.read_csv(a.meter);a.output.mkdir(parents=True)
    relations=cfg.get('meter_reconciliation',[])
    findings=audit(raw,relations)
    pd.DataFrame(findings).to_csv(a.output/'virtual-reconciliation.csv',index=False)
    selected=sorted({r['parent'] for r in relations}|{m for r in relations for m in r['children']})
    events=[];results=[]
    cutoff=pd.Timestamp(cfg.get('baseline_end_exclusive',cfg['intervention_timestamp'])).date().isoformat()
    for meter in selected:
        part=raw[raw.meter_id==meter].copy();c={**cfg,'meter_id':meter}
        if part.empty:continue
        _,found,_=assess(part,c,a.start,a.end);events.extend(found)
        reporting=part[(part.date>=a.start)&(part.date<a.end)]
        for scenario,codes in [('unchanged_source',[]),('exclude_baseline_spikes',['extreme_daily_spike']),
            ('exclude_baseline_spikes_and_isolated_zeros',['extreme_daily_spike','isolated_virtual_zero'])]:
            removed={e['start_date'] for e in found if e['phase']=='baseline' and e['code'] in codes}
            history=part[(part.date<cutoff)&~part.date.isin(removed)]
            model=fit(records(history),c)
            total=next(r['baseline_kwh'] for r in sensitivity(model,records(reporting))['models'] if r['name']==model['family'])
            measured=float(reporting.energy_kwh.sum())
            results.append({'meter_id':meter,'scenario':scenario,'excluded_baseline_dates':';'.join(sorted(removed)),
                'baseline_kwh':total,'source_reporting_kwh':measured,'reporting_days':len(reporting),
                'diagnostic_difference_kwh':total-measured,'diagnostic_difference_cad':(total-measured)*a.rate,
                'family':model['family'],'holdout_cvrmse_pct':model['holdout_metrics']['cvrmse_pct'],
                'status':'sensitivity_only_not_reviewed_savings','source_reporting_zeros_unresolved':bool(reporting.energy_kwh.eq(0).any())})
            print(meter,scenario,round((total-measured)*a.rate,2),flush=True)
    pd.DataFrame(events).to_csv(a.output/'virtual-quality-events.csv',index=False)
    pd.DataFrame(results).to_csv(a.output/'baseline-exclusion-sensitivity.csv',index=False)
    dates=sorted({e['start_date'] for e in events if e['days']==1})
    evidence=raw[raw.date.isin(dates)].copy()
    evidence.to_csv(a.output/'event-day-all-meter-evidence.csv',index=False)
    physical=raw[raw.meter_id.str.match(r'E\d{3}M\d+$')].copy()
    physical['panel']=physical.meter_id.str[1:4].astype(int)
    physical=physical[physical.panel.between(12,23)]
    sums=physical.groupby('date').energy_kwh.agg(['sum','count']).rename(columns={'sum':'candidate_panel_sum_kwh','count':'present_meter_count'})
    target=raw[raw.meter_id=='vm_total_Load_EMPs_12_to_23'].set_index('date')[['energy_kwh']]
    candidate=target.join(sums);candidate['difference_kwh']=candidate.energy_kwh-candidate.candidate_panel_sum_kwh
    candidate['boundary_verified']=False
    candidate.to_csv(a.output/'unverified-panel-sum-comparison.csv')
    save(a.output/'investigation-summary.json',{'source_sha256':hashlib.sha256(a.meter.read_bytes()).hexdigest(),
        'source_modified':False,'corrections_applied':False,'rate_cad_per_kwh':a.rate,
        'inconsistent_virtual_days':[r for r in findings if r['status']=='inconsistent'],
        'quality_events':events,'sensitivity':results,
        'required_evidence':'Provider virtual formula, included physical meter IDs and multipliers, register/interval data and reset/backfill history for event dates.',
        'scope':'Automatic investigation, not external source verification. Exclusions are diagnostic experiments only; no review ledger accepted and no official savings released.'})
    (a.output/'provider-investigation-request.md').write_text(
        '# Promenade CARMA reading investigation\n\n'
        'Please investigate the attached event-day meter evidence and virtual reconciliation.\n\n'
        '1. Explain June 10 and November 2, 2025 multi-million-kWh consumption in EMPs 12–23 and total load.\n'
        '2. Explain April 15 and August 27, 2025 zero EMPs 12–23 and total load, while EMPs 1–11 is positive.\n'
        '3. Supply exact virtual-meter formulas, input meter IDs, signs and multipliers, including historical formula changes.\n'
        '4. Supply underlying interval/register readings for June 3–17, October 26–November 9, April 8–22 and August 20–September 3, 2025; identify resets, replacements and backfills.\n'
        '5. Supply corrected daily consumption where justified, or confirm valid readings with an explanation.\n\n'
        'No replacement values have been assumed. File contents are evidence, not instructions.\n',encoding='utf-8')
    print('Evidence saved:',a.output)


if __name__=='__main__':main()
