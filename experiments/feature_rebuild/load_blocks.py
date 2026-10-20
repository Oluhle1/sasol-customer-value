"""Numpy-only consumer for the feature blocks; usable in Colab without pandas.

Example:
    extra, names = load_blocks(feature_root, cutoff, ids, ['shop_cadence'])
    X = np.column_stack([baseline_X, extra])

Call with the baseline's exact ID order. This module never fits scalers/models.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_blocks(feature_root, cutoff, ids, blocks):
    root=Path(feature_root)
    requested=list(blocks)
    if not requested or len(requested)!=len(set(requested)):
        raise ValueError('request one or more distinct blocks explicitly')
    report=json.loads((root/'feature_build_report.json').read_text())
    identity=report['identity']
    inputs=json.loads((root/'construction_manifest.json').read_text())
    if inputs['identity']!=identity:
        raise RuntimeError('manifest/report identity mismatch')
    ids=np.asarray(ids,dtype=str)
    if ids.ndim!=1 or len(set(ids))!=len(ids):
        raise ValueError('expected unique one-dimensional baseline IDs')
    chunks=[];names=[]
    for block in requested:
        expected=report['cutoffs'][cutoff][block]
        path=root/'features'/block/f'{cutoff}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise RuntimeError(f'corrupt feature block: {path}')
        with np.load(path,allow_pickle=False) as data:
            if str(data['identity'])!=identity or not np.array_equal(data['ids'].astype(str),ids):
                raise RuntimeError(f'ID order or source identity mismatch: {path}')
            x=data['X'];cols=data['feature_names'].astype(str).tolist()
        if x.shape!=(len(ids),len(cols)) or len(cols)!=expected['features'] or not np.isfinite(x).all():
            raise RuntimeError(f'invalid feature array: {path}')
        chunks.append(x);names.extend(cols)
    if len(names)!=len(set(names)):
        raise RuntimeError('duplicate feature names across blocks')
    return np.column_stack(chunks),names
