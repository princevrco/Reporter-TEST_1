import importlib.util
from pathlib import Path
from zipfile import ZipFile
import pytest

spec=importlib.util.spec_from_file_location('promenade_test_runner',Path(__file__).parents[1]/'scripts/test_promenade.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.mark.parametrize('name',['../outside_daily_kWh.csv','unrelated.json'])
def test_meter_zip_rejects_unsafe_or_unexpected_files(tmp_path,name):
    archive=tmp_path/'bad.zip'
    with ZipFile(archive,'w') as z:z.writestr(name,'bad')
    with pytest.raises(ValueError):runner.extract_meter(archive,tmp_path/'unpacked')
    assert not (tmp_path/'outside_daily_kWh.csv').exists()


def test_meter_zip_keeps_daily_csv_and_quality_companions(tmp_path):
    archive=tmp_path/'good.zip'
    with ZipFile(archive,'w') as z:
        z.writestr('M1/2025/M1_daily_kWh.csv','ReadingDate,M1\n')
        z.writestr('M1/2025/M1_daily_kWh.quality.json','{}')
    runner.extract_meter(archive,tmp_path/'unpacked')
    assert len(list((tmp_path/'unpacked').rglob('*.*')))==2
