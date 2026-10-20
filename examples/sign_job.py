"""Run only in the trusted backend; never publish actual secrets or signed site jobs."""
import hashlib,hmac,json,os,sys
from pathlib import Path
from option_c.data import canonical

payload=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
key=os.environ['OPTION_C_JOB_HMAC_KEY']
signature=hmac.new(key.encode(),canonical(payload),hashlib.sha256).hexdigest()
Path(sys.argv[2]).write_text(json.dumps({'input':{'payload':payload,'signature':signature}}),encoding='utf-8')
