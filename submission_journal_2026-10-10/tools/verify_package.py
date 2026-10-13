"""Verify archived hashes and submission lineage without retraining."""
from pathlib import Path
import csv, hashlib, json, math
root=Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'package_manifest.json').read_text())
for entry in manifest:
 p=root/entry['path']; assert digest(p)==entry['sha256'], str(p)
ledger=json.loads((root/'submission_ledger.json').read_text())
config=json.loads((root/'source_snapshot/configs/label_config.json').read_text())
def all_strings(obj):
 if isinstance(obj,str): yield obj
 elif isinstance(obj,dict):
  for k,v in obj.items(): yield k; yield from all_strings(v)
 elif isinstance(obj,list):
  for v in obj: yield from all_strings(v)
labels=set(all_strings(config))
def read(p):
 with p.open(newline='') as f:
  reader=csv.DictReader(f); rows=list(reader); fields=reader.fieldnames
 assert fields==['ID','CLV_fuel','CLV_nonfuel','Opportunity'], fields
 assert len(rows)==5488 and len({r['ID'] for r in rows})==5488
 for r in rows:
  for key in ('CLV_fuel','CLV_nonfuel'): assert math.isfinite(float(r[key])) and float(r[key])>=0
  assert r['Opportunity'] in labels, r['Opportunity']
 return rows
for entry in ledger:
 p=root/entry['csv']; b=root/entry['baseline']
 assert digest(p)==entry['sha256']; assert digest(b)==entry['baseline_sha256']
 rows,base=read(p),read(b)
 assert [r['ID'] for r in rows]==[r['ID'] for r in base]
 for key in ('CLV_fuel','CLV_nonfuel','Opportunity'):
  if key in entry['changed_targets']: continue
  for a,z in zip(rows,base):
   assert a[key]==z[key] if key=='Opportunity' else math.isclose(float(a[key]),float(z[key]),rel_tol=1e-12,abs_tol=1e-12)
 print(f"PASS {p.name}: score {entry['public_score']}, hash, IDs, labels, values, preserved targets")
print(f'PASS {len(manifest)} package file hashes')
