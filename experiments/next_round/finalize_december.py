"""Apply the December measurement amendment without refitting or deleting raw evidence."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def finalize(repo,dest):
 out=repo/'outputs/next_round';amend=out/'december_scoring_amendment.json'
 if not amend.exists():return {}
 classes=json.loads((repo/'configs/label_config.json').read_text())['opportunity_labels'];panel=__import__('pickle').loads((repo/'outputs/oct10/category_dynamics_panel.pkl').read_bytes());frame=panel['frames']['2024-12-01'];ids=frame.index.astype(str).to_numpy();y=panel['labels']['2024-12-01'].reindex(frame.index).Opportunity.to_numpy()
 def pred(variant):
  paths=[]
  for p in out.glob('*_2024-12-01_Opportunity.json'):
   metadata=json.loads(p.read_text())
   if metadata['variant']==variant:paths.append(p)
  assert paths,variant
  chosen=max(paths,key=lambda p:p.stat().st_mtime);z=np.load(chosen.with_suffix('.npz'),allow_pickle=True);assert np.array_equal(z['ids'],ids)
  return np.asarray(classes)[z['prediction'].argmax(axis=1)],chosen.name
 b,source=pred('baseline_reconstruction');bm=float(f1_score(y,b,average='weighted'));result={'December_baseline_untuned_F1':bm,'source':source,'amendment':json.loads(amend.read_text()),'no_model_refit':True}
 if (out/'baseline_reproduction.csv').exists():
  table=pd.read_csv(out/'baseline_reproduction.csv');mask=table.origin.eq('2024-12-01')&table.target.eq('Opportunity');table.loc[mask,'fresh_metric']=bm;table['December_Opportunity_rule']=np.where(mask,'Uniform argmax; later March-tuned multipliers excluded.','');table.to_csv(dest/'baseline_reproduction_corrected.csv',index=False)
 if (out/'calendar_decision.json').exists():
  c,csource=pred('calendar_exposure_v1');cm=float(f1_score(y,c,average='weighted'));table=pd.read_csv(out/'calendar_comparison.csv');mask=table.origin.eq('2024-12-01')&table.target.eq('Opportunity');table.loc[mask,['baseline_metric','candidate_metric','gain']]=[bm,cm,cm-bm];table.to_csv(dest/'calendar_comparison_corrected.csv',index=False)
  result.update({'December_calendar_untuned_F1':cm,'December_calendar_gain':cm-bm,'calendar_source':csource})
  decision=json.loads((out/'calendar_decision.json').read_text());accepted=[]
  for target in ['Opportunity','CLV_fuel','CLV_nonfuel']:
   t=table[table.target.eq(target)];screen=t[t.origin.isin(['2025-06-01','2025-09-01'])].gain.to_numpy();six=t[t.origin.isin(['2025-04-01','2025-05-01','2025-06-01','2025-07-01','2025-08-01','2025-09-01'])].gain.to_numpy();dec=t[t.origin.eq('2024-12-01')].gain.to_numpy()
   if len(screen)==2 and min(screen)>0 and screen.mean()>=.001 and len(six)==6 and sum(six>0)>=5 and min(six)>=-.002 and six.mean()>=.001 and len(dec)==1 and dec[0]>=0:accepted.append(target)
  old=decision['accepted_targets'];decision['accepted_targets']=accepted;decision['rows']=table.astype(object).where(pd.notna(table),None).to_dict(orient='records');decision['measurement_amendment']='December uniform multipliers declared before candidate fitting; raw evidence preserved.';decision['raw_decision_accepted_targets']=old;(dest/'calendar_decision_corrected.json').write_text(json.dumps(decision,indent=2)+'\n')
  pair=pd.DataFrame({'ID':ids,'origin':'2024-12-01','true':y,'baseline':b,'candidate':c});pair.to_csv(dest/'calendar_exposure_v1_2024-12-01_Opportunity_paired_corrected.csv',index=False)
  sys.path.insert(0,str(repo/'experiments/next_round'));import run_round as r
  pairs=[pair]+[pd.read_csv(p,dtype={'ID':str}) for p in out.glob('calendar_exposure_v1_*_Opportunity_paired.csv') if '2024-12-01' not in p.name]
  (dest/'calendar_Opportunity_bootstrap_corrected.json').write_text(json.dumps(r.bootstrap(pd.concat(pairs),'Opportunity'),indent=2)+'\n')
  result['accepted_targets_corrected']=accepted;result['raw_acceptance_changed']=accepted!=old
  # Do not silently issue or invalidate a submission on changed measurement.
  if accepted!=old:result['requires_export_review']='Reconcile final export with corrected guard before any manual submission.'
 (dest/'december_scoring_correction.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__': print(json.dumps(finalize(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
