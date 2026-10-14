"""Produce human-readable results and final artifact integrity checks."""
import sys,json,hashlib,pickle
from pathlib import Path
import numpy as np
import pandas as pd
REPO=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[2]
OUT=REPO/'outputs/next_round'; DEST=Path(sys.argv[2]) if len(sys.argv)>2 else OUT;DEST.mkdir(exist_ok=True)
best=REPO/'submissions/sasol_oct10_small_fuel_values.csv';expected='11a4da31d86ab4738c2366b9c06f8439124991fc1e138bef685dbc38e61eb79a';assert hashlib.sha256(best.read_bytes()).hexdigest()==expected
sys.path.insert(0,str(Path(__file__).parent))
from finalize_december import finalize
correction=finalize(REPO,DEST)
panel=pickle.loads((REPO/'outputs/oct10/category_dynamics_panel.pkl').read_bytes())
raw=pd.read_csv(OUT/'population_comparison.csv');raw['mean_difference']=raw.final_mean-raw.historical_mean;raw['standardized_shift_defined']=raw.feature.map(lambda k:any(np.std(panel['frames'][c][k].to_numpy())>1e-10 for c in [*panel['monthly_cutoffs'],'2025-12-01']))
# Global origin-level prices are constant within both compared populations; SMD is undefined.
for i,row in raw.iterrows():
 a=panel['frames'][row.historical_origin][row.feature].to_numpy();b=panel['frames']['2025-12-01'][row.feature].to_numpy();sd=np.sqrt((np.var(a)+np.var(b))/2)
 if sd<=1e-10:raw.loc[i,'smd']=np.nan;raw.loc[i,'standardized_shift_defined']=False
raw.to_csv(DEST/'population_comparison_corrected.csv',index=False)
counts=[];final=panel['frames']['2025-12-01']
for c in ['2024-12-01','2025-06-01','2025-09-01']:
 h=panel['frames'][c];common=h.index.intersection(final.index);counts.append({'origin':c,'historical_n':len(h),'final_n':len(final),'matched_n':len(common),'final_newly_eligible':len(final.index.difference(h.index)),'historical_not_final':len(h.index.difference(final.index)),'historical_short_history_90d':int(h.age_days.lt(90).sum()),'final_short_history_90d':int(final.age_days.lt(90).sum()),'missing_historical_feature_values':int(h.isna().sum().sum()),'missing_final_feature_values':int(final.isna().sum().sum())})
pd.DataFrame(counts).to_csv(DEST/'eligibility_history_audit.csv',index=False)
matched=[]
for c in ['2024-12-01','2025-06-01','2025-09-01']:
 h=panel['frames'][c];common=h.index.intersection(final.index)
 for col in ['age_days','recency_days','q1_fuel_l','q1_shop_r','n_cat_ever','n_cat_q1','fuel_price_q1','price_gap']:
  a=h.loc[common,col].to_numpy();b=final.loc[common,col].to_numpy();matched.append({'origin':c,'feature':col,'matched_customers':len(common),'historical_mean':float(a.mean()),'final_mean':float(b.mean()),'paired_mean_change':float((b-a).mean())})
pd.DataFrame(matched).to_csv(DEST/'matched_population_changes.csv',index=False)
cohorts=[]
from sklearn.metrics import f1_score
for paired_path in OUT.glob('calendar_exposure_v1_*_paired.csv'):
 parts=paired_path.name.removeprefix('calendar_exposure_v1_').removesuffix('_paired.csv').split('_',1);origin,target=parts
 corrected=DEST/(paired_path.stem+'_corrected.csv');data=pd.read_csv(corrected if corrected.exists() else paired_path,dtype={'ID':str});features=panel['frames'][origin].reindex(data.ID)
 masks=[('short_history_lt90d',features.age_days.lt(90).to_numpy()),('longer_history_ge90d',features.age_days.ge(90).to_numpy())]
 for lo,hi in [(0,14),(14,30),(30,90),(90,1e9)]:masks.append((f'fuel_recency_{lo}_{hi}',features.fuel_recency.ge(lo).to_numpy()&features.fuel_recency.lt(hi).to_numpy()))
 masks.append(('past_proxy_near_growth_threshold',(features.fuel_growth_gap.abs().le(.1)&features.cat_fuel_ever.gt(0)).to_numpy()))
 if target=='Opportunity':
  for label in ['Existing-category growth: Fuel','Stable','Inactivity']:masks.append((f'true_{label}',data.true.eq(label).to_numpy()))
 for name,mask in masks:
  if not mask.any():continue
  sub=data.loc[mask];y=sub.true.to_numpy();a=sub.baseline.to_numpy();b=sub.candidate.to_numpy()
  if target=='Opportunity':
   bm=float(f1_score(y,a,average='weighted',zero_division=0));cm=float(f1_score(y,b,average='weighted',zero_division=0));gain=cm-bm
  else:
   bm=float(np.sqrt(np.mean((y-a)**2)));cm=float(np.sqrt(np.mean((y-b)**2)));gain=bm-cm
  cohorts.append({'origin':origin,'target':target,'cohort':name,'rows':len(sub),'baseline_metric':bm,'candidate_metric':cm,'gain':gain,'baseline_error_rate':float(np.mean(y!=a)) if target=='Opportunity' else None,'candidate_error_rate':float(np.mean(y!=b)) if target=='Opportunity' else None,'interpretation':'Overlapping diagnostic cohorts on reused dates; not additional acceptance tests.'})
pd.DataFrame(cohorts).to_csv(DEST/'calendar_cohort_comparison.csv',index=False)
base=pd.read_csv(best,dtype={'ID':str});official=pd.read_csv(REPO/'data/raw/SampleSubmission.csv',dtype={'ID':str});labels=json.loads((REPO/'configs/label_config.json').read_text())['opportunity_labels'];exports=[]
for p in OUT.glob('*_submission.csv'):
 df=pd.read_csv(p,dtype={'ID':str});assert list(df.columns)==list(official.columns);assert df.ID.tolist()==official.ID.tolist();assert len(df)==5488 and df.ID.is_unique and not df.isna().any().any();assert set(df.Opportunity)<=set(labels)
 for key in ['CLV_fuel','CLV_nonfuel']:assert np.isfinite(df[key]).all() and df[key].ge(0).all()
 audit=json.loads(p.with_suffix('.audit.json').read_text());changed=audit['targets']
 for col in ['CLV_fuel','CLV_nonfuel','Opportunity']:
  if col not in changed:assert df[col].equals(base[col]) if col=='Opportunity' else np.allclose(df[col],base[col],rtol=1e-12,atol=1e-12)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==audit['output_sha256'];exports.append({'file':p.name,'changed_targets':changed,'checks':'PASS hash, official IDs/order/columns, labels, finite values, untouched target preservation'})
(DEST/'final_integrity.json').write_text(json.dumps({'baseline_sha256':expected,'baseline_unchanged':True,'exports':exports},indent=2)+'\n')
def md_table(df):
 def fmt(v):
  if pd.isna(v):return '—'
  if isinstance(v,(float,np.floating)):return f'{v:.8f}'
  return str(v).replace('|',r'\|').replace('\n',' ')
 return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
parts=['# October 14 experiment results','Confirmed public baseline remains **0.309164264**. New local metrics are not a measured public score. User-reported top10 reference is **0.317136582**; target gap **0.007972318**.','## Audit findings','The three large classes comprise 81.6% of September rows. Fuel growth has 1,435 examples and F1 0.4803745; its weighted F1 deficit contribution is 0.142356, larger than Stable 0.114368 and Inactivity 0.067620. Only 1 of 1,436 eligible fuel-growth events lost to another category in September. Threshold/purchase uncertainty is better supported than category competition as a fuel-error explanation.','Global price_index and price_change are constant within origins. Their original raw SMD values are numerically meaningless; use population_comparison_corrected.csv, which sets constant-population SMD to missing and reports absolute mean changes. Other comparisons show changing tenure and price gap; matched_population_changes.csv separates customer evolution from population composition.','December 2024 is a reused matched-season diagnostic. It had only one eligible training origin and no nested early-stop fold; 40 boosting rounds were fixed before results. Its lower metrics must not be interpreted as an unbiased seasonal effect because training history also differs.']
parts.extend(['## Prior-year coverage', 'All December 2024 prior-year fuel and shop features are zero because that history was not released; December 2025 has coverage. Zero padding must not be interpreted as observed customer inactivity. The matched-season backtest also differs in training-history length. See prior_year_coverage.csv.'])
for name,title in [('baseline_reproduction.csv','Baseline reproduction'),('calendar_comparison.csv','Calendar exposure'),('rule_comparison.csv','Fuel-rule OOF correction')]:
 corrected=DEST/name.replace('.csv','_corrected.csv');p=corrected if corrected.exists() else OUT/name
 if p.exists():
  table=pd.read_csv(p)
  cols=[c for c in ['origin','target','fresh_metric','archived_metric','metric_difference','baseline_metric','candidate_metric','matching_control_metric','gain','rows'] if c in table.columns]
  parts.extend(['## '+title,md_table(table[cols]),f'Full precision and alignment fields: `{p.name}`.'])
for name,title in [('calendar_decision.json','Calendar decision'),('rule_decision.json','Rule decision')]:
 corrected=DEST/name.replace('.json','_corrected.json');p=corrected if corrected.exists() else OUT/name
 if p.exists():parts.extend(['## '+title,'```json\n'+p.read_text()+'\n```'])
parts.extend(['## December measurement correction',json.dumps(correction,indent=2)])
parts.extend(['## Statistical and reproducibility limits','Per-customer paired rows and customer-cluster bootstrap intervals are archived alongside comparisons. Intervals condition on reused historical dates, repeated customers and overlapping horizons; they do not remove repeated-selection bias or measure future seasonal uncertainty. Exact library versions and source/config/panel fingerprints accompany caches. Inputs and canonical labels matched the recorded hashes. Original archives and best submission were preserved.','## Cohort diagnostics','See calendar_cohort_comparison.csv for recency, short-history, past-threshold-proxy and true-class comparisons. Cohorts overlap and are diagnostic rather than new model-selection tests.','## Final integrity',json.dumps(exports,indent=2),'No public-score improvement is established by this round. Rejected variants are retained with their original gates and evidence. No candidate is exported unless it meets its preregistered rule.'])
(DEST/'RESULTS.md').write_text('\n\n'.join(parts)+'\n')
print(json.dumps({'exports':exports,'baseline_unchanged':True,'results':str(DEST/'RESULTS.md')}))
