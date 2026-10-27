"""Validate native observation provenance for sparse Mapper wide CSV exports."""
import hashlib
import numpy as np
import pandas as pd
from .data import DataError


def read(path,csv_path):
    try:
        with np.load(path,allow_pickle=False) as z:
            if str(z['csv_sha256'].item())!=hashlib.sha256(csv_path.read_bytes()).hexdigest():
                raise DataError('Observation mask CSV checksum mismatch')
            timestamps=pd.to_datetime(z['timestamps'],utc=True,format='mixed')
            columns=z['columns'].tolist();rows=int(z['rows'].item());packed=z['mask']
            if rows!=len(timestamps) or not timestamps.is_unique or len(set(columns))!=len(columns):
                raise DataError('Invalid observation mask dimensions or duplicate keys')
            if packed.dtype!=np.uint8 or packed.shape!=((rows+7)//8,len(columns)):
                raise DataError('Invalid packed observation mask shape')
            return pd.DataFrame(np.unpackbits(packed,axis=0,count=rows).astype(bool),index=timestamps,columns=columns)
    except (OSError,KeyError,ValueError) as exc:
        raise DataError('Cannot read native observation mask: '+str(exc)) from exc
