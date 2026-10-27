from pathlib import Path
import json
import pytest
from option_c.data import DataError
from option_c import project_job


def test_project_job_uses_scoped_files_and_installed_module(tmp_path,monkeypatch):
    inputs=tmp_path/'inputs';inputs.mkdir()
    for n in ('meter.xlsx','bms.csv','weather.csv','config.json'):(inputs/n).write_text('test')
    payload={'files':{'meter':['meter.xlsx'],'bms':['bms.csv'],'weather':'weather.csv','config':'config.json'},
             'start':'2026-08-15','end':'2026-10-21','forecast_start':'2026-10-26','forecast_end':'2027-01-26','weather_scenario':'climatology'}
    def fake_run(cmd,**kwargs):
        assert cmd[1:3]==['-m','option_c.project_cli']
        assert str(inputs/'meter.xlsx') in cmd
        out=Path(cmd[cmd.index('--output')+1]);out.mkdir(parents=True);(out/'project-summary.json').write_text(json.dumps({'strict_bms_rows_used':0,'verified_savings_cad':None}))
        return type('Result',(),{'returncode':0,'stderr':''})()
    monkeypatch.setattr(project_job.subprocess,'run',fake_run)
    result=project_job.execute(payload,tmp_path,'request1');assert result['summary']['verified_savings_cad'] is None


def test_project_job_rejects_input_traversal(tmp_path):
    with pytest.raises(DataError,match='simple site input filenames'):
        project_job.execute({'files':{'meter':['../other.xlsx'],'bms':['bms.csv'],'weather':'weather.csv','config':'config.json'}},tmp_path,'request2')
