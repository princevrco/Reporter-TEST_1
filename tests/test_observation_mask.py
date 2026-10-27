import hashlib
import numpy as np
import pytest
from option_c import observation_mask
from option_c.data import DataError


def test_mask_bound_to_csv_and_unique_timestamps(tmp_path):
    csv=tmp_path/'bms.csv';csv.write_text('timestamp,A\n2026-01-01T00:00:00Z,1\n');mask=tmp_path/'bms.observations.npz'
    def write(times):
        np.savez_compressed(mask,mask=np.array([[128]],dtype=np.uint8),rows=len(times),columns=np.array(['A']),timestamps=np.array(times),csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest())
    write(['2026-01-01T00:00:00Z']);assert observation_mask.read(mask,csv).iloc[0,0]
    csv.write_text(csv.read_text()+'\n')
    with pytest.raises(DataError,match='checksum'):observation_mask.read(mask,csv)
    write(['2026-01-01T00:00:00Z','2026-01-01T00:00:00Z'])
    with pytest.raises(DataError,match='duplicate'):observation_mask.read(mask,csv)
