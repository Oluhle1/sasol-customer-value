"""October 14: audit, paired calendar experiment, and temporal OOF fuel rules."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse, csv, gc, hashlib, importlib.metadata, json, math, pickle, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'experiments/oct10'))
import feature_pipeline as fp
OLD=REPO/'outputs/oct10'; OUT=REPO/'outputs/next_round'; OUT.mkdir(exist_ok=True)
EXPECTED='11a4da31d86ab4738c2366b9c06f8439124991fc1e138bef685dbc38e61eb79a'
DATES=['2025-06-01','2025-09-01']; EXTRA=['2025-04-01','2025-05-01','2025-07-01','2025-08-01']; DEC='2024-12-01'
START=time.monotonic(); BUDGET=3600

def log(s): print(s,flush=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')
def read(name): return json.loads((OUT/name).read_text())
def budget():
 if time.monotonic()-START>BUDGET: raise RuntimeError('Declared one-hour run budget reached; resume only after reviewing results.')
def load_panel(): return pickle.loads((OLD/'category_dynamics_panel.pkl').read_bytes())

def cache_key(panel,variant):
 return hashlib.sha256(json.dumps({'panel':sha(OLD/'category_dynamics_panel.pkl'),'config':sha(REPO/'configs/label_config.json'),'source':sha(__file__),'rule_source':sha(Path(__file__).with_name('rule_experiment.py')),'helpers':{p.name:sha(p) for p in [REPO/'experiments/oct10/feature_pipeline.py',REPO/'src/reference/label_rules.py']},'variant':variant,'features':panel['feature_cols'],'versions':{m:importlib.metadata.version(m) for m in ['numpy','pandas','lightgbm','scikit-learn']},'threads':1},sort_keys=True).encode()).hexdigest()[:16]
def params(target,seed):
 if target=='Opportunity': return dict(n_estimators=1200,learning_rate=.04,num_leaves=47,min_child_samples=35,subsample=.85,subsample_freq=1,colsample_bytree=.70,reg_lambda=1.2,random_state=seed,n_jobs=1,verbosity=-1)
 return dict(n_estimators=1000,learning_rate=.04,num_leaves=15,min_child_samples=100 if target=='CLV_fuel' else 50,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_lambda=1.2,random_state=seed,n_jobs=1,verbosity=-1)

def stack(panel,cutoff):
 latest=(pd.Timestamp(cutoff)-pd.DateOffset(months=3)).strftime('%Y-%m-%d'); limit=(pd.Timestamp(latest)-pd.DateOffset(months=3)).strftime('%Y-%m-%d')
 cuts=[c for c in panel['monthly_cutoffs'] if c<=latest]; assert cuts
 xs=[];ys=[];ws=[];cs=[]
 for c in cuts:
  f=panel['frames'][c]; assert pd.Timestamp(c)+pd.DateOffset(months=3)<=pd.Timestamp(cutoff)
  xs.append(f[panel['feature_cols']]); ys.append(panel['labels'][c].reindex(f.index)); cs.extend([c]*len(f))
  ws.extend([.70+.05*((pd.Timestamp(c).year-2024)*12+pd.Timestamp(c).month-9)]*len(f))
 x=pd.concat(xs); y=pd.concat(ys); cs=np.asarray(cs); w=np.asarray(ws)
 tr=cs<=limit; va=cs==latest
 assert all(pd.Timestamp(str(c))+pd.DateOffset(months=3)<=pd.Timestamp(latest) for c in set(cs[tr]))
 return x,y,w,tr,va,latest

def fit_one(x,y,w,tr,va,pred,target,seed,kind='direct'):
 p=params(target,seed); binary=kind=='purchase'; clf=target=='Opportunity' or binary
 if kind=='conditional': p['min_child_samples']=50
 if binary:p['min_child_samples']=50
 cls=LGBMClassifier if clf else LGBMRegressor
 if tr.any() and va.any():
  inner=cls(**p); inner.fit(x.loc[tr],y[tr],sample_weight=w[tr],eval_set=[(x.loc[va],y[va])],callbacks=[early_stopping(60,verbose=False)])
  p['n_estimators']=max(inner.best_iteration_,40); del inner
 else: p['n_estimators']=40 # fixed before December outcomes are inspected
 model=cls(**p);model.fit(x,y,sample_weight=w)
 if binary:
  value=model.predict_proba(pred)[:,list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(pred))
 elif target=='Opportunity':
  value=np.zeros((len(pred),17)); pp=model.predict_proba(pred)
  for k,label in enumerate(model.classes_):value[:,CLASSES.index(label)]=pp[:,k]
 else:value=model.predict(pred)
 return value,p['n_estimators']

CLASSES=json.loads((REPO/'configs/label_config.json').read_text())['opportunity_labels']
MULT=np.asarray(json.loads((OLD/'category_dynamics_decision.json').read_text())['multipliers'])

def forecast(panel,cutoff,target,variant):
 budget(); key=cache_key(panel,variant);path=OUT/f'{key}_{cutoff}_{target}.npz'
 if path.exists():
  z=np.load(path); assert np.array_equal(z['ids'],panel['frames'][cutoff].index.astype(str).to_numpy());return z['prediction']
 x,labels,w,tr,va,latest=stack(panel,cutoff); pred=panel['frames'][cutoff][panel['feature_cols']]; metadata=[]
 if target=='Opportunity':
  outputs=[]
  for seed in [7,23]:
   z,n=fit_one(x,labels.Opportunity.to_numpy(),w,tr,va,pred,target,seed);outputs.append(z);metadata.append({'seed':seed,'rounds':n})
  result=np.mean(outputs,axis=0)
 else:
  y=labels[target].to_numpy();direct=[];hurdles=[];res=[];pos=y>0
  for seed in ([7,21] if target=='CLV_fuel' else [11,29]):
   z,n=fit_one(x,y,w,tr,va,pred,target,seed);direct.append(z)
   q,nc=fit_one(x,pos.astype(int),w,tr,va,pred,target,seed,'purchase')
   v,nv=fit_one(x.loc[pos],y[pos],w[pos],tr[pos],va[pos],pred,target,seed,'conditional');hurdles.append(q*np.maximum(v,0))
   m={'seed':seed,'direct_rounds':n,'purchase_rounds':nc,'positive_value_rounds':nv}
   if target=='CLV_fuel':
    anchor=x.fuel_z_q1.to_numpy();delta,nr=fit_one(x,y-anchor,w,tr,va,pred,target,seed,'residual');res.append(delta);m['residual_rounds']=nr
   metadata.append(m)
  d=np.maximum(np.mean(direct,axis=0),0);h=np.mean(hurdles,axis=0)
  result=(.4*d+.35*h+.25*np.maximum(pred.fuel_z_q1.to_numpy()+np.mean(res,axis=0),0)) if target=='CLV_fuel' else .5*d+.5*h
 np.savez_compressed(path,prediction=result,ids=panel['frames'][cutoff].index.astype(str).to_numpy())
 save(path.stem+'.json',{'variant':variant,'fingerprint':key,'target':target,'origin':cutoff,'latest_training_origin':latest,'train_rows':len(x),'threads':1,'models':metadata,'nested_selection':bool(tr.any()),'fallback_rounds':40 if not tr.any() else None})
 log(f'{variant} {cutoff} {target}: rows={len(x)} {metadata}')
 del x,labels;gc.collect();return result

def archived(panel,c,target):
 if target=='Opportunity':return np.load(OLD/f'{c}_category_dynamics.npy')
 d=np.load(OLD/f'{c}_{target}_category_values.npy');h=np.load(OLD/f'{c}_{target}_two_part.npy')
 if target=='CLV_fuel':return .4*d+.35*h+.25*np.load(OLD/f'{c}_fuel_residual.npy')
 return .5*d+.5*h

def labels_from(p):return np.asarray(CLASSES)[np.argmax(p*MULT,axis=1)]
def metric(y,p,target):return float(f1_score(y,labels_from(p),average='weighted')) if target=='Opportunity' else float(np.sqrt(np.mean((y-p)**2)))
def delta(y,b,c,target):return metric(y,c,target)-metric(y,b,target) if target=='Opportunity' else metric(y,b,target)-metric(y,c,target)
def bootstrap(frame,target,n=200):
 ids=frame.ID.unique();rng=np.random.default_rng(14102026);lookup={i:k for k,i in enumerate(ids)}; ix=np.array([lookup[i] for i in frame.ID]);vals=[]
 y=frame.true.to_numpy();b=frame.baseline.to_numpy();c=frame.candidate.to_numpy()
 for _ in range(n):
  counts=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids));w=counts[ix]
  if target=='Opportunity': vals.append(f1_score(y,c,average='weighted',sample_weight=w,zero_division=0)-f1_score(y,b,average='weighted',sample_weight=w,zero_division=0))
  else:vals.append(np.sqrt(np.average((y-b)**2,weights=w))-np.sqrt(np.average((y-c)**2,weights=w)))
 return {'replicates':n,'customer_clusters':len(ids),'gain_ci95':np.quantile(vals,[.025,.975]).tolist(),'interpretation':'Customer variation conditional on reused dates; excludes future seasonal uncertainty and repeated-selection bias.'}

def verify_inputs():
 state=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True);save('initial_git_status.json',{'status':state})
 manifest=json.loads((REPO/'submission_journal_2026-10-10/input_data_manifest.json').read_text())
 for row in manifest:assert sha(REPO/row['path'])==row['sha256'],row['path']
 assert sha(REPO/'submissions/sasol_oct10_small_fuel_values.csv')==EXPECTED
 pub=Path('/home/oluhle/sasol-github-update/submission_journal_2026-10-10')
 subprocess.run([sys.executable,str(pub/'tools/verify_package.py')],check=True)
 compared=0
 for p in (REPO/'submission_journal_2026-10-10').rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:
   counterpart=pub/p.relative_to(REPO/'submission_journal_2026-10-10')
   # Completed publication corrected relative links in the evidence narrative.
   if counterpart.exists() and p.name!='EXPERIMENT_RESULTS.md':assert sha(p)==sha(counterpart),str(p);compared+=1
 save('environment.json',{'python':sys.version,'executable':sys.executable,'versions':{m:importlib.metadata.version(m) for m in ['numpy','pandas','scipy','scikit-learn','lightgbm','catboost']},'threads':1,'baseline_sha256':EXPECTED,'inputs_verified':manifest,'local_archive_files_compared':compared,'local_verifier_missing':'Completed published archive verifier run instead; realized local artifacts compared with published copies.','cpu':os.cpu_count(),'memory':subprocess.check_output(['free','-h'],text=True),'disk':subprocess.check_output(['df','-h',str(REPO)],text=True),'rules_url':'https://zindi.world/competitions/sasol-customer-value-recruitment-challenge','evaluation':'Separate transformed fuel/nonfuel RMSE and weighted F1; final weights .3/.3/.4, normalization denominator unavailable.','external_data':'No holidays or other external datasets.'})
 log('Input fingerprints and realized archive verified.')

def error_audit(panel):
 classes=[];cohorts=[];margins=[]
 focus=['Existing-category growth: Fuel','Stable','Inactivity']
 for c in DATES:
  f=panel['frames'][c];y=panel['labels'][c].reindex(f.index).Opportunity.to_numpy();p=archived(panel,c,'Opportunity');pred=labels_from(p)
  report=classification_report(y,pred,labels=CLASSES,output_dict=True,zero_division=0)
  for label in CLASSES:
   row=report[label];classes.append({'origin':c,'label':label,**row,'weighted_f1_contribution':row['support']/len(y)*row['f1-score'],'weighted_f1_deficit':row['support']/len(y)*(1-row['f1-score'])})
  pd.DataFrame(confusion_matrix(y,pred,labels=CLASSES),index=CLASSES,columns=CLASSES).to_csv(OUT/f'{c}_confusion.csv')
  q=p*MULT;q=q/q.sum(axis=1,keepdims=True);s=np.sort(q,axis=1);margin=s[:,-1]-s[:,-2]
  rec=f['fuel_recency'] if 'fuel_recency' in f else f[[k for k in f.columns if 'fuel' in k and 'recency' in k][0]]
  for lo,hi in [(0,14),(14,30),(30,90),(90,1e9)]:
   mask=rec.ge(lo)&rec.lt(hi)
   cohorts.append({'origin':c,'cohort':f'fuel_recency_{lo}_{hi}','n':int(mask.sum()),'f1':float(f1_score(y[mask],pred[mask],average='weighted',zero_division=0)) if mask.any() else None,'error_rate':float(np.mean(y[mask]!=pred[mask])) if mask.any() else None})
  margins.append({'origin':c,'rows':len(y),'majority_three_share':float(np.isin(y,focus).mean()),'fuel_growth_support':int((y==focus[0]).sum()),'median_adjusted_probability_margin':float(np.median(margin)),'errors_low_margin_005':int(((y!=pred)&(margin<.05)).sum())})
  pd.DataFrame({'ID':f.index,'true':y,'pred':pred,'adjusted_margin':margin}).to_csv(OUT/f'{c}_baseline_opportunity_rows.csv',index=False)
 pd.DataFrame(classes).to_csv(OUT/'class_error_contributions.csv',index=False);pd.DataFrame(cohorts).to_csv(OUT/'recency_cohorts.csv',index=False);save('error_summary.json',margins)
 return margins

def population(panel):
 f=panel['frames'][fp.TEST_CUTOFF];cols=[k for k in f.columns if any(s in k for s in ['age_days','recency','q1_fuel','q1_shop','n_cat','price','growth_gap','dyn_fuel','n_tx','year'])];rows=[]
 for c in [DEC]+DATES:
  h=panel['frames'][c]
  for k in cols:
   a=h[k].to_numpy();b=f[k].to_numpy();sd=np.sqrt((a.std()**2+b.std()**2)/2)
   rows.append({'historical_origin':c,'feature':k,'historical_mean':float(a.mean()),'final_mean':float(b.mean()),'smd':float((b.mean()-a.mean())/sd) if sd else 0,'historical_zero_fraction':float((a==0).mean()),'final_zero_fraction':float((b==0).mean()),'historical_rows':len(h),'final_rows':len(f)})
 pd.DataFrame(rows).to_csv(OUT/'population_comparison.csv',index=False)
 return sorted([r for r in rows if r['historical_origin']=='2025-09-01'],key=lambda r:abs(r['smd']),reverse=True)[:12]

def rule_targets(panel):
 cfg=json.loads((REPO/'configs/label_config.json').read_text());d=fp.lr.load_transactions(REPO/'data/raw/train.csv')
 clean=d[~d.ItemName.isin(cfg['adjustment_items'])];included=[k for k,v in cfg['category_mapping'].items() if v['included']]
 fuel=clean[clean.ItemCategoryLevel2.eq('Fuel')];qual=clean[clean.ItemCategoryLevel2.isin(included)&clean._quantity.gt(0)&clean._amount.gt(0)]
 targets={};rows=[]
 for c in panel['monthly_cutoffs']:
  t=pd.Timestamp(c);end=t+pd.DateOffset(months=3);ids=panel['frames'][c].index
  past=fuel[(fuel._time>=t-pd.DateOffset(months=3))&(fuel._time<t)].groupby('ID')._amount.sum().reindex(ids,fill_value=0).to_numpy()
  fs=fuel[(fuel._time>=t)&(fuel._time<end)].groupby('ID')._amount.sum().reindex(ids,fill_value=0).to_numpy()
  fq=qual[qual.ItemCategoryLevel2.eq('Fuel')&(qual._time>=t)&(qual._time<end)].groupby('ID').TransactionId.nunique().reindex(ids,fill_value=0).to_numpy()
  ever=qual[qual.ItemCategoryLevel2.eq('Fuel')&(qual._time<t)].groupby('ID').size().reindex(ids,fill_value=0).to_numpy()>0
  purchase=fq>=cfg['growth_min_purchases'];cross=(fs-past>=int(cfg['growth_min_rand']*fp.lr.SCALE))&(fs>np.maximum(past,0)*(1+cfg['growth_min_relative']))
  joint=ever&purchase&cross
  targets[c]=pd.DataFrame({'purchase':purchase.astype(int),'fuel_growth_event':joint.astype(int)},index=ids)
  if c in DATES:
   y=panel['labels'][c].reindex(ids).Opportunity.to_numpy();pred=labels_from(archived(panel,c,'Opportunity'));slack=np.minimum((fs-past)/fp.lr.SCALE-cfg['growth_min_rand'],(fs-np.maximum(past,0)*(1+cfg['growth_min_relative']))/fp.lr.SCALE)
   for mask,name in [(abs(slack)<=100,'within_100_rand_of_threshold'),(joint,'fuel_growth_event_eligible'),(joint&(y!='Existing-category growth: Fuel'),'fuel_event_lost_category_competition')]:
    rows.append({'origin':c,'cohort':name,'n':int(mask.sum()),'error_rate':float(np.mean(y[mask]!=pred[mask])) if mask.any() else None})
 pd.DataFrame(rows).to_csv(OUT/'threshold_category_competition_audit.csv',index=False)
 del d,clean,fuel,qual;gc.collect();return targets,rows

def audit(panel):
 save('audit_proposal.json',{'phase':1,'hypothesis':'December seasonal exposure or final-population drift may contribute to plateau; not established.','baseline':'0.309164264','tuning_dates':'none; December backtest uses fixed 40 rounds without nested selection','evaluation_dates':[DEC]+DATES,'baseline_reproduction':'Fresh single-thread fits on matching June/September rows; compare cached historical vectors and metric deltas.','budget_seconds':BUDGET,'December_prior_use':'Phase 04 F1 and target calibration used December; reused seasonal check, not fresh.','seasonality_existing':'prior-year customer volumes/spend, elapsed origin_month, actual forecast horizon in existing cadence features'})
 summary=error_audit(panel);top=population(panel);targets,thresholds=rule_targets(panel);rows=[]
 for c in [DEC]+DATES:
  for target in ['Opportunity','CLV_fuel','CLV_nonfuel']:
   p=forecast(panel,c,target,'baseline_reconstruction');y=panel['labels'][c].reindex(panel['frames'][c].index)[target].to_numpy();m=metric(y,p,target)
   if c in DATES:
    old=archived(panel,c,target);oldm=metric(y,old,target);difference=float(np.max(abs(old-p)))
   else:oldm=None;difference=None
   row={'origin':c,'target':target,'fresh_metric':m,'archived_metric':oldm,'metric_difference':m-oldm if oldm is not None else None,'max_prediction_difference':difference,'same_validation_ids':True,'rows':len(y),'December_fixed_rounds':40 if c==DEC else None};rows.append(row);log(row)
 pd.DataFrame(rows).to_csv(OUT/'baseline_reproduction.csv',index=False)
 season=[k for k in panel['feature_cols'] if any(s in k for s in ['yago','year','origin_month','ses','bgn'])]
 save('audit_decision.json',{'test_first':'One calendar-exposure variant, keeping capacity/seeds/decision weights/rows fixed.','rule_candidate_supported':True,'support':'Observed fuel-growth confusion, threshold cohorts and category-competition event counts; see CSVs. This is a diagnostic hypothesis, not proof of improvement.','pooled_seasonal_variant':'Deferred: only one observed December outcome season and sparse prior-year coverage; no additional variant search this round.','existing_seasonality_features':season,'error_summary':summary,'top_population_shifts':top,'threshold_cohorts':thresholds})
 (OUT/'audit.md').write_text('# December forecast and error audit\n\nBaseline: 0.309164264; exact CSV and input hashes verified. December 2024 was already used in Phase 04 F1 and label calibration. Only September 2024 is eligible for its training; nested early stopping is infeasible, so all components use 40 rounds fixed before results. December is a reused matched-season diagnostic, not an untouched holdout.\n\n## Population shift\n\n'+pd.DataFrame(top).to_csv(index=False)+'\n## Large-class errors\n\n'+json.dumps(summary,indent=2)+'\n\n## Threshold and category competition\n\n'+pd.DataFrame(thresholds).to_csv(index=False)+'\n## Reproduction\n\n'+pd.DataFrame(rows).to_csv(index=False)+'\n## First decision\n\nTest one explicit calendar-exposure variant first. Existing seasonality features include prior-year outcomes and elapsed origin_month; seasonality is not absent. Tree thresholds on elapsed month cannot extrapolate a trend. Defer pooled seasonal fitting because only one historical December season is observed. Then test fuel-rule event probabilities via temporally out-of-fold inputs, preserving all class competition; this differs from the rejected three-group specialist.\n\nMetrics remain separate: official component weights are known but leaderboard RMSE normalization constants are unavailable. No external holidays, realized future prices or promotions enter features. See population_comparison.csv, confusion CSVs, class_error_contributions.csv, recency_cohorts.csv, and baseline_reproduction.csv.\n')
 return targets

def calendar_panel(panel):
 out=dict(panel);out['frames']={};extra_names=None
 for c,f in panel['frames'].items():
  t=pd.Timestamp(c);end=t+pd.DateOffset(months=3);days=pd.date_range(t,end-pd.Timedelta(days=1),freq='D');angle=2*np.pi*(t.month-1)/12;angles=2*np.pi*(days.month-1)/12;pastdays=(t-(t-pd.DateOffset(months=3))).days
  x=pd.DataFrame({'cal_sin':np.sin(angle),'cal_cos':np.cos(angle),'cal_window_sin':np.sin(angles).mean(),'cal_window_cos':np.cos(angles).mean(),'cal_window_days':len(days),'cal_weekend_fraction':np.mean(days.dayofweek>=5),'cal_december_fraction':np.mean(days.month==12),'cal_exposure_ratio':len(days)/pastdays},index=f.index)
  # Customer-dependent exposure interactions can vary even in the one-origin December backtest.
  x['cal_fuel_exposure_z']=np.log1p(np.maximum(f.q1_fuel_l,0)*len(days)/pastdays)/panel['s_fuel'];x['cal_shop_exposure_z']=np.log1p(np.maximum(f.q1_shop_r,0)*len(days)/pastdays)/panel['s_shop']
  out['frames'][c]=pd.concat([f,x],axis=1);extra_names=list(x.columns)
 out['feature_cols']=panel['feature_cols']+extra_names;return out

def comparison(panel,target,c,b,p,variant):
 y=panel['labels'][c].reindex(panel['frames'][c].index)[target].to_numpy();row={'variant':variant,'origin':c,'target':target,'baseline_metric':metric(y,b,target),'candidate_metric':metric(y,p,target),'gain':delta(y,b,p,target),'rows':len(y),'missing_rows':0}
 paired=pd.DataFrame({'ID':panel['frames'][c].index,'origin':c,'true':y,'baseline':labels_from(b) if target=='Opportunity' else b,'candidate':labels_from(p) if target=='Opportunity' else p});paired.to_csv(OUT/f'{variant}_{c}_{target}_paired.csv',index=False)
 return row,paired

def gate(rows,target):
 screen=[r['gain'] for r in rows if r['target']==target and r['origin'] in DATES];six=[r['gain'] for r in rows if r['target']==target and r['origin'] in DATES+EXTRA];dec=[r['gain'] for r in rows if r['target']==target and r['origin']==DEC]
 return bool(len(screen)==2 and min(screen)>0 and np.mean(screen)>=.001 and len(six)==6 and sum(g>0 for g in six)>=5 and min(six)>=-.002 and np.mean(six)>=.001 and (not dec or dec[0]>=0))

def calendar(panel):
 variant='calendar_exposure_v1';save('calendar_proposal.json',{'phase':2,'hypothesis':'Cyclic month and known calendar exposure may improve December-transfer behavior.','variant_count':1,'features':['month sin/cos','forecast-day weighted sin/cos','actual calendar-window length','weekend and December fractions','window/prior-quarter day ratio','fuel/shop exposure interactions'],'comparator':'Fresh baseline reproduction on same IDs plus recorded archive metric as a second guard','seeds':{'Opportunity':[7,23],'fuel':[7,21],'nonfuel':[11,29]},'threads':1,'rows_capacity_decision_weights':'Held fixed. Inner round selection is the unchanged nested procedure.','tuning_dates':'latest eligible inner origin, purged; December fixed40','evaluation_dates':[DEC]+DATES+EXTRA,'gate':'Both June and September improve over better of reconstructed/archived metric, mean >=.001; at least5/6 improve, mean>=.001, worst>=-.002. December cannot deteriorate. Bootstrap diagnostic, not extra acceptance search.','budget_seconds':BUDGET})
 cp=calendar_panel(panel);rows=[];accepted=[]
 for target in ['Opportunity','CLV_fuel','CLV_nonfuel']:
  paired=[]
  for c in [DEC]+DATES:
   b=forecast(panel,c,target,'baseline_reconstruction');p=forecast(cp,c,target,variant);row,pair=comparison(panel,target,c,b,p,variant)
   if c in DATES:
    y=panel['labels'][c].reindex(panel['frames'][c].index)[target].to_numpy();row['gain_vs_archived']=delta(y,archived(panel,c,target),p,target);row['gain']=min(row['gain'],row['gain_vs_archived'])
   rows.append(row);paired.append(pair);log(row);pd.DataFrame(rows).to_csv(OUT/'calendar_comparison.csv',index=False)
  screen=[r['gain'] for r in rows if r['target']==target and r['origin'] in DATES]
  if min(screen)>0 and np.mean(screen)>=.001:
   for c in EXTRA:
    b=forecast(panel,c,target,'baseline_reconstruction');p=forecast(cp,c,target,variant);row,pair=comparison(panel,target,c,b,p,variant);rows.append(row);paired.append(pair);log(row)
  save(f'calendar_{target}_bootstrap.json',bootstrap(pd.concat(paired),target))
  if gate(rows,target):accepted.append(target)
  pd.DataFrame(rows).to_csv(OUT/'calendar_comparison.csv',index=False)
 save('calendar_decision.json',{'accepted_targets':accepted,'rows':rows,'gate':'Unchanged predeclared calendar protocol; no automatic promotion for a December-only win.','public_score':None})
 if accepted:export(panel,cp,accepted,variant)
 del cp;gc.collect()

def export(panel,candidate,targets,variant):
 best=REPO/'submissions/sasol_oct10_small_fuel_values.csv';assert sha(best)==EXPECTED;df=pd.read_csv(best,dtype={'ID':str});base=df.copy();ids=panel['frames'][fp.TEST_CUTOFF].index.astype(str)
 for target in targets:
  p=forecast(candidate,fp.TEST_CUTOFF,target,variant);p=labels_from(p) if target=='Opportunity' else p
  values=pd.Series(p,index=ids);assert set(df.ID)==set(ids);df[target]=df.ID.map(values)
 assert len(df)==5488 and df.ID.is_unique and not df.isna().any().any();assert set(df.Opportunity)<=set(CLASSES)
 for target in ['CLV_fuel','CLV_nonfuel']:assert np.isfinite(df[target]).all() and (df[target]>=0).all()
 path=OUT/f'{variant}_submission.csv';df.to_csv(path,index=False,float_format='%.17g');check=pd.read_csv(path,dtype={'ID':str})
 for target in ['Opportunity','CLV_fuel','CLV_nonfuel']:
  if target not in targets:assert np.array_equal(check[target],base[target]) if target=='Opportunity' else np.allclose(check[target],base[target],rtol=1e-12,atol=1e-12)
 save(path.stem+'.audit.json',{'baseline_sha256':EXPECTED,'output_sha256':sha(path),'targets':targets,'rows':len(df),'public_score':None,'status':'Locally eligible; manual public evaluation still required.'})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['audit','calendar','rules','all'],default='all');args=ap.parse_args()
 verify_inputs();panel=load_panel();assert len(panel['feature_cols'])==305
 if args.phase in ['audit','all']:targets=audit(panel)
 else:targets=None
 if args.phase in ['calendar','all']:calendar(panel)
 if args.phase in ['rules','all']:
  from rule_experiment import run
  run(sys.modules[__name__],panel,targets)
 assert sha(REPO/'submissions/sasol_oct10_small_fuel_values.csv')==EXPECTED
 save('run_summary.json',{'phase':args.phase,'elapsed_seconds':time.monotonic()-START,'baseline_unchanged':True,'public_score':.309164264,'new_public_score':None})
 log('Round finished; confirmed public baseline preserved.')
if __name__=='__main__':main()
