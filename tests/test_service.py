import hashlib,hmac
import pytest
from fixtures import config,baseline_rows
from option_c.data import DataError,canonical
from option_c.service import execute,authenticated_handler


def test_signed_training_idempotency_and_site_isolation(tmp_path,monkeypatch):
    key='secret-for-test-only-'*3
    monkeypatch.setenv('OPTION_C_JOB_HMAC_KEY',key);monkeypatch.setenv('OPTION_C_STORE_ROOT',str(tmp_path))
    c=config();c['site_id']='promenade'
    p={'site_id':'promenade','request_id':'baseline_1','action':'train','config':c,'rows':baseline_rows()}
    sig=hmac.new(key.encode(),canonical(p),hashlib.sha256).hexdigest()
    result=authenticated_handler({'input':{'payload':p,'signature':sig}})
    assert execute(p,tmp_path)==result
    p['config']['boundary_description']='changed'
    with pytest.raises(DataError,match='reused'):execute(p,tmp_path)
    with pytest.raises(PermissionError):authenticated_handler({'input':{'payload':p,'signature':sig}})
    with pytest.raises(FileNotFoundError):execute({'site_id':'other','request_id':'x','action':'forecast','model_id':result['model_id'],'rows':[],'start':'2026-01-01','end':'2026-01-02'},tmp_path)
    with pytest.raises(DataError,match='identifier'):execute({'site_id':'../other','request_id':'x'},tmp_path)
