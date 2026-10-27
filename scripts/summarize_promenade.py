"""Add visible forecasts and CAD totals to existing results; no training rerun needed."""
import argparse
from pathlib import Path
from option_c.overview import build
from option_c.weather import read as read_weather
from option_c.data import load,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/promenade-test'))
    p.add_argument('--weather',type=Path,help='Explicit daily weather for future baseline dates; saved separately from historical inputs')
    p.add_argument('--rate',type=float,default=.12,help='CAD per kWh')
    p.add_argument('--forecast-start');p.add_argument('--forecast-end',help='Exclusive local date')
    p.add_argument('--total-meters',nargs='+',help='Operator-selected non-overlapping site boundary')
    a=p.parse_args()
    try:
        weather=None
        if a.weather:
            weather,audit=read_weather(a.weather,load(a.results/'summary.json')['timezone'])
            save(a.results/'forecast-weather-audit.json',audit)
            weather.to_csv(a.results/'forecast-weather.csv',index=False)
        r=build(a.results,a.rate,a.forecast_start,a.forecast_end,a.total_meters,weather)
    except (ValueError,KeyError,FileNotFoundError) as exc:p.exit(2,str(exc)+'\n')
    print('Open '+str((a.results/'index.html').resolve()))
    print('Future baseline forecast: '+r['forecast_start']+' to '+r['forecast_end_exclusive']+' exclusive')
    print('Per-meter kWh and CAD comparisons: '+str(a.results/'meter-summary.csv'))
    print('Site total: '+str(r['site_total']))


if __name__=='__main__':main()
