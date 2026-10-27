"""Site-scoped mounted-file Runpod entry for the complete project workflow."""
from pathlib import Path
import re,subprocess,sys
from .data import DataError,load


def execute(payload,base,request):
    inputs=(Path(base)/'inputs').resolve();output=Path(base)/'project-results'/request
    if not output.resolve().is_relative_to(Path(base).resolve()):raise DataError('Project output escaped site store')
    files=payload.get('files',{})
    def path(name):
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,160}',name) or '..' in name:raise DataError('Project inputs must use simple site input filenames')
        candidate=(inputs/name).resolve()
        if not candidate.is_relative_to(inputs) or not candidate.is_file():raise DataError('Project input missing or outside authorized site inputs')
        return str(candidate)
    if not isinstance(files,dict):raise DataError('Project files must be an object')
    cmd=[sys.executable,'-m','option_c.project_cli']
    for key in ('meter','bms'):
        names=files.get(key)
        if not isinstance(names,list) or not names:raise DataError('Project files require nonempty '+key+' list')
        cmd+=['--'+key]+[path(n) for n in names]
    for key in ('weather','config'):
        if key not in files:raise DataError('Project files require '+key)
        input_path=path(files[key])
        if key=='config' and payload.get('site_id') and load(input_path).get('site_id')!=payload['site_id']:raise DataError('Configuration site must match authorized project site')
        cmd+=['--'+key,input_path]
    for key in ('bms_observation_mask','future_weather','old_policy','optimized_policy','initial_state','historical_initial_state','signal_config','old_trajectory','optimized_trajectory','old_trajectory_evidence','optimized_trajectory_evidence'):
        if key in files:cmd+=['--'+key.replace('_','-'),path(files[key])]
    for key in ('start','end','forecast_start','forecast_end'):
        if key not in payload or not isinstance(payload[key],str):raise DataError('Project request requires '+key)
        cmd+=['--'+key.replace('_','-'),payload[key]]
    for key in ('weather_scenario','future_weather_basis','meter_kind','rate','cadence_minutes'):
        if key in payload:cmd+=['--'+key.replace('_','-'),str(payload[key])]
    for key in ('meter_ids','required_bms_columns'):
        if key in payload:
            values=payload[key]
            if not isinstance(values,list) or not values or any(not isinstance(v,str) or v.startswith('-') for v in values):raise DataError('Invalid project '+key)
            cmd+=['--'+key.replace('_','-')]+values
    cmd+=['--output',str(output)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    if result.returncode:raise DataError('Project job failed: '+result.stderr[-4000:])
    return {'status':'completed_requires_review','summary':load(output/'project-summary.json'),
            'artifact_directory':str(output),'artifact_files':[str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()],
            'scope':'Private mounted artifacts; inspect output gates. No public artifact URLs generated.'}
