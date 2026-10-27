"""Download daily outdoor temperatures directly from Open-Meteo."""
import argparse
from pathlib import Path
from option_c.weather_download import parameters,download
from option_c.data import DataError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',required=True,help='Inclusive local YYYY-MM-DD')
    p.add_argument('--end',required=True,help='Inclusive local YYYY-MM-DD; historical data only')
    p.add_argument('--latitude',type=float,default=43.83128)
    p.add_argument('--longitude',type=float,default=-79.5159)
    p.add_argument('--timezone',default='America/Toronto')
    p.add_argument('--model',choices=['ecmwf_ifs','era5','era5_land'],default='ecmwf_ifs')
    p.add_argument('--output',type=Path,default=Path('local/weather_daily.csv'))
    a=p.parse_args()
    try:
        params=parameters(a.start,a.end,a.latitude,a.longitude,a.timezone,a.model)
        audit,cached=download(a.output,params)
    except (DataError,FileNotFoundError,ValueError) as exc:p.exit(2,str(exc)+'\n')
    print(('Verified cached' if cached else 'Downloaded')+' daily weather: '+str(a.output))
    print(str(audit['rows'])+' days; '+a.timezone+'; Celsius; model '+a.model)
    print('API --end is inclusive. Analysis --end remains exclusive.')


if __name__=='__main__':main()
