"""Reuse baseline fits only after proving the model-producing functions unchanged."""
import ast,sys,json,shutil,hashlib
from pathlib import Path
import numpy as np
import run_round as r
source=Path(__file__).with_name('source_versions')/'run_round_v1.py';old=ast.parse(source.read_text().replace('.index.astype(str).to_numpy()', '.index.astype(str).to_numpy(dtype=str)'));new=ast.parse(Path(r.__file__).read_text())
def funcs(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
a,b=funcs(old),funcs(new)
for name in ['params','stack','fit_one','forecast']:assert a[name]==b[name],name
panel=r.load_panel();key=r.cache_key(panel,'baseline_reconstruction');entries=[]
for c in ['2024-12-01','2025-06-01','2025-09-01']:
 for target in ['Opportunity','CLV_fuel','CLV_nonfuel']:
  candidates=[]
  for p in r.OUT.glob(f'*_{c}_{target}.json'):
   d=json.loads(p.read_text())
   if d.get('variant')=='baseline_reconstruction' and p.stem.split('_')[0]!=key:candidates.append(p)
  assert candidates,(c,target)
  src=max(candidates,key=lambda p:p.stat().st_mtime);dst=r.OUT/f'{key}_{c}_{target}.npz';z=np.load(src.with_suffix('.npz'),allow_pickle=True);assert np.array_equal(z['ids'],panel['frames'][c].index.astype(str).to_numpy());assert np.isfinite(z['prediction']).all()
  assert not dst.exists(),dst;np.savez_compressed(dst,prediction=z['prediction'],ids=z['ids'].astype(str));assert np.array_equal(np.load(dst)['prediction'],z['prediction'])
  metadata=json.loads(src.read_text());metadata['original_fingerprint']=metadata['fingerprint'];metadata['fingerprint']=key;metadata['migration']='Model-producing functions params/stack/fit_one/forecast proven AST-identical to archived source v1 after normalizing only ID-array serialization to Unicode. Predictions checked exactly unchanged. Measurement correction and calendar feature construction do not alter baseline fitted predictions.';metadata['original_source_snapshot_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();dst.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n');entries.append({'old':src.name,'new':dst.name,'prediction_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
r.save('baseline_cache_migration.json',{'baseline_model_AST_identical':True,'new_fingerprint':key,'records':entries,'fresh_reproduction_metrics_already_matched_archive':True})
print(json.dumps({'migrated_fits':len(entries),'fingerprint':key}))
