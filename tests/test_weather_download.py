import copy,json
from io import BytesIO
import pandas as pd
import pytest
from option_c.data import DataError
from option_c import weather_download as wd


def response():
    return {'timezone':'America/Toronto','daily_units':{'temperature_2m_mean':'°C'},
            'daily':{'time':['2026-10-01','2026-10-02'],'temperature_2m_mean':[12.,13.]},'latitude':43.8,'longitude':-79.5}


def test_download_normalizes_and_reuses_verified_cache(tmp_path,monkeypatch):
    calls=[]
    def fetch(request,timeout):
        calls.append(request.full_url);assert timeout==60
        return BytesIO(json.dumps(response()).encode())
    monkeypatch.setattr(wd,'urlopen',fetch)
    params=wd.parameters('2026-10-01','2026-10-02');path=tmp_path/'weather.csv'
    a,cached=wd.download(path,params);assert not cached and a['rows']==2
    b,cached=wd.download(path,params);assert cached and len(calls)==1 and b==a
    assert pd.read_csv(path).temperature_c.tolist()==[12.,13.]
    path.write_text('tampered')
    with pytest.raises(DataError,match='checksum mismatch'):wd.download(path,params)


@pytest.mark.parametrize('problem',['null','dates','units','timezone'])
def test_api_invalid_or_incomplete_response_is_rejected(problem):
    payload=response();params=wd.parameters('2026-10-01','2026-10-02')
    if problem=='null':payload['daily']['temperature_2m_mean'][1]=None
    if problem=='dates':payload['daily']['time'][1]='2026-10-03'
    if problem=='units':payload['daily_units']['temperature_2m_mean']='F'
    if problem=='timezone':payload['timezone']='UTC'
    with pytest.raises(DataError):wd.normalize(payload,params)


def test_cache_cannot_be_reused_for_different_query(tmp_path,monkeypatch):
    monkeypatch.setattr(wd,'urlopen',lambda request,timeout:BytesIO(json.dumps(response()).encode()))
    path=tmp_path/'weather.csv';params=wd.parameters('2026-10-01','2026-10-02');wd.download(path,params)
    params=wd.parameters('2026-10-01','2026-10-02',model='era5')
    with pytest.raises(DataError,match='different dates/location/model'):wd.download(path,params)


def test_download_with_nulls_writes_no_model_input(tmp_path,monkeypatch):
    bad=response();bad['daily']['temperature_2m_mean'][1]=None
    monkeypatch.setattr(wd,'urlopen',lambda request,timeout:BytesIO(json.dumps(bad).encode()))
    path=tmp_path/'weather.csv'
    with pytest.raises(DataError):wd.download(path,wd.parameters('2026-10-01','2026-10-02'))
    assert not path.exists() and not path.with_suffix('.source.json').exists()
