"""Add visible forecasts and CAD totals to existing results; no training rerun needed."""
import argparse
from pathlib import Path
from option_c.overview import build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/promenade-test'))
    p.add_argument('--rate',type=float,default=.12,help='CAD per kWh')
    p.add_argument('--forecast-start');p.add_argument('--forecast-end',help='Exclusive local date')
    p.add_argument('--total-meters',nargs='+',help='Operator-selected non-overlapping site boundary')
    a=p.parse_args()
    try:r=build(a.results,a.rate,a.forecast_start,a.forecast_end,a.total_meters)
    except (ValueError,KeyError,FileNotFoundError) as exc:p.exit(2,str(exc)+'\n')
    print('Open '+str((a.results/'index.html').resolve()))
    print('Future baseline forecast: '+r['forecast_start']+' to '+r['forecast_end_exclusive']+' exclusive')
    print('Per-meter kWh and CAD comparisons: '+str(a.results/'meter-summary.csv'))
    print('Site total: '+str(r['site_total']))


if __name__=='__main__':main()
