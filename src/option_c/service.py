"""Signed, site-scoped Runpod jobs. Single worker; no arbitrary paths or downloads."""
import hashlib,hmac,os,re
from pathlib import Path
from .data import DataError,canonical,digest,load,save
from .regression import fit
from .workflow import forecast,savings,train_post


def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value):raise DataError('Invalid site/request/model identifier')
    return value


def execute(payload,root):
    site=identifier(payload['site_id']);request=identifier(payload['request_id'])
    base=Path(root).resolve()/site;job=base/'jobs'/(request+'.json');sha=digest(payload)
    if job.exists():
        old=load(job)
        if old['request_sha256']!=sha:raise DataError('Request identifier reused with different content')
        return old['result']
    action=payload['action']
    if action=='run_project':
        from .project_job import execute as execute_project
        result=execute_project(payload,base,request)
    elif action in ('train','train_post'):
        cfg=payload['config']
        if cfg.get('site_id')!=site:raise DataError('Configuration site must match authorized job site')
        result=fit(payload['rows'],cfg) if action=='train' else train_post(payload['rows'],payload['bms_rows'],cfg)
        save(base/'models'/(result['model_id']+'.json'),result)
    elif action in ('forecast','savings'):
        baseline=load(base/'models'/(identifier(payload['model_id'])+'.json'))
        if action=='forecast':result=forecast(baseline,payload['rows'],payload['start'],payload['end'])
        else:
            post=load(base/'models'/(identifier(payload['post_model_id'])+'.json')) if payload.get('post_model_id') else None
            result=savings(baseline,payload['rows'],payload['start'],payload['end'],post,payload.get('bms_rows'),payload.get('adjustments'),payload.get('flat_rate'))
    else:raise DataError('Actions: run_project, train, train_post, forecast, savings')
    save(base/'inputs'/(request+'.json'),payload)
    save(job,{'request_sha256':sha,'result':result})
    return result


def authenticated_handler(job):
    key=os.environ.get('OPTION_C_JOB_HMAC_KEY','');root=os.environ.get('OPTION_C_STORE_ROOT')
    if len(key)<32 or not root:raise RuntimeError('Configure a >=32 character HMAC key and persistent store root')
    envelope=job['input'];payload=envelope['payload'];raw=canonical(payload)
    if len(raw)>8_000_000:raise DataError('Inline payload exceeds 8 MB; use private Pod files for larger history')
    expected=hmac.new(key.encode(),raw,hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected,envelope.get('signature','')):raise PermissionError('Invalid job signature')
    return execute(payload,root)
