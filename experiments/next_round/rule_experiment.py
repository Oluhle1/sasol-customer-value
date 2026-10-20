"""Unconditional fuel-rule signals and a regularized 17-class temporal OOF correction."""
import gc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

def run(r,panel,targets=None):
 if not r.read('audit_decision.json')['rule_candidate_supported']:
  r.save('rule_decision.json',{'accepted':False,'reason':'Audit did not support threshold/category-competition hypothesis.'});return
 r.save('rule_proposal.json',{'phase':3,'hypothesis':'Predicting an exact unconditional fuel growth event and qualifying fuel purchases adds threshold information beyond baseline class probabilities.','difference_from_rejected_specialist':'No macro-group routing, summed three-group probabilities or arbitrary blend. Two rule-derived binary targets generate temporally out-of-fold signals; downstream correction includes all configured labels and full baseline class probabilities.','target_definitions':{'purchase':'Unconditional probability of at least configured growth_min_purchases future qualifying Fuel baskets.','fuel_growth_event':'Unconditional joint probability: qualifying historical Fuel purchase, qualifying future purchase count, future signed fuel spending satisfies exact >=R20 increment and strict >25% growth relative to clipped prior spend. Does not imply winning against other category events.'},'dependence':'Signals are never multiplied as independent events. Joint growth probability is set to zero when known historical Fuel eligibility is false, before and after calibration; this is an exact rule constraint, not an independence assumption.','upstream_features':'Accepted 305 past-only features, unchanged label generator.','upstream_capacity':'15 leaves, min child50, 40 fixed boosting rounds, learning_rate .04, two seeds7/23, one thread; no early-stop or calibration on outer outcomes.','downstream':'Standardized log baseline17 probabilities plus log known growth threshold and observed fuel eligibility; candidate adds two calibrated rule logits. Multinomial logistic C=.05, max_iter500; matched no-rule logistic control. Original decision multipliers preserved. Both control and candidate set final Fuel-growth probability to zero for known historical Fuel-ineligible customers, then renormalize; all other configured labels still compete.','calibration':'For each downstream training origin, Platt C1 uses only OOF outcomes ending by that origin; identity fallback when no earlier OOF outcomes exist. At evaluation it uses all eligible earlier OOF records. No same-row auxiliary outcomes calibrate its own downstream input.','OOF_origins':'December2024 through latest eligible monthly origin; for outer T all training outcomes end<=T. Earlier insufficient nested history uses preregistered40 rounds.','evaluation_dates':['2025-06-01','2025-09-01','2025-04-01','2025-05-01','2025-07-01','2025-08-01'],'December_limitation':'No earlier OOF records with a complete outcome window are available for fitting correction before December2024; no dishonest December correction backtest. Calendar/baseline December remains available.','gate':'Both June/September improve weighted F1 over better of realized/reproduced baseline and matching logistic control, mean >=.001. Then at least5/6 positive gains versus same controls, mean>=.001 and worst>=-.002.','budget_seconds':r.BUDGET,'quantiles':'Deferred: binary exact event target suffices for this minimal round; no conditional-quantile search.'})
 if targets is None:targets,_=r.rule_targets(panel)
 key=r.cache_key(panel,'fuel_rule_oof_v1');origins=[c for c in panel['monthly_cutoffs'] if c>='2024-12-01']
 allrows=[];comparisons=[];paired=[]
 def upstream(c):
  r.budget();path=r.OUT/f'{key}_{c}_rule_binary.npz'
  if path.exists():
   z=np.load(path);assert np.array_equal(z['ids'],panel['frames'][c].index.astype(str).to_numpy(dtype=str));return z['prediction']
  x,_,w,tr,va,latest=r.stack(panel,c);cuts=[t for t in panel['monthly_cutoffs'] if t<=latest];y=pd.concat([targets[t].reindex(panel['frames'][t].index) for t in cuts]);pred=panel['frames'][c][panel['feature_cols']];output=[]
  # Fixed rounds; binary helper falls back to 40 when tr/va omitted.
  for name in ['purchase','fuel_growth_event']:
   pp=[]
   for seed in [7,23]:
    p,_=r.fit_one(x,y[name].to_numpy(),w,np.zeros(len(x),bool),np.zeros(len(x),bool),pred,'CLV_fuel',seed,'purchase');pp.append(p)
   output.append(np.mean(pp,axis=0))
  result=np.column_stack(output);result[panel['frames'][c].cat_fuel_ever.le(0).to_numpy(),1]=0;np.savez_compressed(path,prediction=result,ids=panel['frames'][c].index.astype(str).to_numpy(dtype=str));r.log(f'fuel rules OOF {c}: latest training {latest}, n={len(x)}')
  del x,y;gc.collect();return result
 def data(c):
  base=r.forecast(panel,c,'Opportunity','baseline_reconstruction');binary=upstream(c);f=panel['frames'][c]
  threshold=np.maximum(f.q1_fuel_r.to_numpy()+20,np.maximum(f.q1_fuel_r.to_numpy(),0)*1.25)
  known=np.column_stack([np.log1p(np.maximum(threshold,0)),f['cat_fuel_ever'].to_numpy()])
  x=np.column_stack([np.log(np.clip(base,1e-6,1)),known]);y=panel['labels'][c].reindex(f.index).Opportunity.to_numpy()
  return x,binary,y,f.index.to_numpy(dtype=str),base
 stored={}
 def get(c):
  if c not in stored:stored[c]=data(c)
  return stored[c]
 def score_origin(c):
  latest=(pd.Timestamp(c)-pd.DateOffset(months=3)).strftime('%Y-%m-%d');cuts=[v for v in origins if v<=latest];assert cuts
  records=[get(v) for v in cuts];x=np.vstack([v[0] for v in records]);binary=np.vstack([v[1] for v in records]);y=np.concatenate([v[2] for v in records]);w=np.concatenate([np.repeat(.70+.05*((pd.Timestamp(v).year-2024)*12+pd.Timestamp(v).month-9),len(z[2])) for v,z in zip(cuts,records)])
  assert all(pd.Timestamp(v)+pd.DateOffset(months=3)<=pd.Timestamp(c) for v in cuts)
  xeval,beval,yeval,ids,base=get(c);caltrain=[];caleval=[]
  for k,name in enumerate(['purchase','fuel_growth_event']):
   true=np.concatenate([targets[v][name].reindex(panel['frames'][v].index).to_numpy() for v in cuts]);logit=lambda p:np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
   if len(np.unique(true))==2:
    calibrator=LogisticRegression(C=1,max_iter=300,random_state=1410);calibrator.fit(logit(binary[:,k]).reshape(-1,1),true,sample_weight=w);ct=calibrator.predict_proba(logit(binary[:,k]).reshape(-1,1))[:,1];ce=calibrator.predict_proba(logit(beval[:,k]).reshape(-1,1))[:,1]
   else:ct=binary[:,k];ce=beval[:,k]
   # Cross-fit training calibration in time as well as the upstream models.
   causal=[]
   for origin,record in zip(cuts,records):
    prior=[v for v in cuts if pd.Timestamp(v)+pd.DateOffset(months=3)<=pd.Timestamp(origin)]
    if prior:
     raw_prior=np.concatenate([get(v)[1][:,k] for v in prior]);target_prior=np.concatenate([targets[v][name].reindex(panel['frames'][v].index).to_numpy() for v in prior])
     if len(np.unique(target_prior))==2:
      causal_cal=LogisticRegression(C=1,max_iter=300,random_state=1410).fit(logit(raw_prior).reshape(-1,1),target_prior)
      pp=causal_cal.predict_proba(logit(record[1][:,k]).reshape(-1,1))[:,1]
     else:pp=record[1][:,k]
    else:pp=record[1][:,k]
    causal.append(logit(pp))
   if name=='fuel_growth_event':
    causal=[np.where(record[0][:,-1]>0,v,logit(np.zeros(len(v)))) for record,v in zip(records,causal)]
    ce=np.where(xeval[:,-1]>0,ce,0)
   caltrain.append(np.concatenate(causal));caleval.append(logit(ce))
   actual=targets[c][name].reindex(ids).to_numpy();allrows.append({'origin':c,'event':name,'train_oof_origins':cuts,'train_rows':len(true),'raw_brier':float(brier_score_loss(actual,beval[:,k])),'calibrated_brier':float(brier_score_loss(actual,ce)),'auc':float(roc_auc_score(actual,ce)) if len(np.unique(actual))==2 else None})
   bins=pd.cut(ce,bins=np.linspace(0,1,11),include_lowest=True)
   pd.DataFrame({'bin':bins,'probability':ce,'outcome':actual}).groupby('bin',observed=False).agg(n=('outcome','size'),mean_probability=('probability','mean'),event_rate=('outcome','mean')).to_csv(r.OUT/f'rule_{c}_{name}_calibration.csv')
  def fit(a,b):
   scaler=StandardScaler();a=scaler.fit_transform(a);b=scaler.transform(b);model=LogisticRegression(C=.05,max_iter=500,random_state=1410);model.fit(a,y,sample_weight=w);p=np.zeros((len(b),17));small=model.predict_proba(b)
   for k,label in enumerate(model.classes_):p[:,r.CLASSES.index(label)]=small[:,k]
   p[xeval[:,-1]<=0,r.CLASSES.index('Existing-category growth: Fuel')]=0;p/=p.sum(axis=1,keepdims=True);return p,list(model.classes_)
  control,cc=fit(x,xeval);candidate,seen=fit(np.column_stack([x,*caltrain]),np.column_stack([xeval,*caleval]))
  row,pair=r.comparison(panel,'Opportunity',c,base,candidate,'fuel_rule_correction_v1');row['gain_vs_matching_logistic_control']=r.delta(yeval,control,candidate,'Opportunity');row['matching_control_metric']=r.metric(yeval,control,'Opportunity');row['unobserved_training_classes']=[v for v in r.CLASSES if v not in seen];row['gain']=min(row['gain'],row['gain_vs_matching_logistic_control'])
  if c in r.DATES:row['gain_vs_archived']=r.delta(yeval,r.archived(panel,c,'Opportunity'),candidate,'Opportunity');row['gain']=min(row['gain'],row['gain_vs_archived'])
  comparisons.append(row);paired.append(pair);r.log(row);pd.DataFrame(comparisons).to_csv(r.OUT/'rule_comparison.csv',index=False);r.save('rule_probability_metrics.json',allrows)
  np.savez_compressed(r.OUT/f'{key}_{c}_rule_correction.npz',ids=ids,prediction=candidate,control=control)
  r.save(f'{key}_{c}_rule_alignment.json',{'rows':len(ids),'oof_origins':cuts,'latest_training_origin':latest,'missing_eval_ids':0,'in_sample_upstream_training_predictions':False})
 for c in r.DATES:score_origin(c)
 screen=[v['gain'] for v in comparisons]
 if min(screen)>0 and np.mean(screen)>=.001:
  for c in r.EXTRA:score_origin(c)
 accepted=r.gate(comparisons,'Opportunity');r.save('rule_bootstrap.json',r.bootstrap(pd.concat(paired),'Opportunity'));r.save('rule_decision.json',{'accepted':accepted,'comparisons':comparisons,'public_score':None,'export':'Not exported if screen fails. An eligible candidate requires separate final OOF refit and export audit.'})
 # A passing correction still needs a final refit before release; expose that explicitly.
 if accepted:
  # Same causal routine at final origin, with no labels consulted; keep a separate candidate-only artifact.
  latest=(pd.Timestamp(r.fp.TEST_CUTOFF)-pd.DateOffset(months=3)).strftime('%Y-%m-%d');cuts=[v for v in origins if v<=latest];records=[get(v) for v in cuts]
  x=np.vstack([v[0] for v in records]);binary=np.vstack([v[1] for v in records]);y=np.concatenate([v[2] for v in records]);weights=np.concatenate([np.repeat(.70+.05*((pd.Timestamp(v).year-2024)*12+pd.Timestamp(v).month-9),len(z[2])) for v,z in zip(cuts,records)])
  f=panel['frames'][r.fp.TEST_CUTOFF];base=r.forecast(panel,r.fp.TEST_CUTOFF,'Opportunity','baseline_reconstruction');beval=upstream(r.fp.TEST_CUTOFF);threshold=np.maximum(f.q1_fuel_r.to_numpy()+20,np.maximum(f.q1_fuel_r.to_numpy(),0)*1.25);xeval=np.column_stack([np.log(np.clip(base,1e-6,1)),np.log1p(np.maximum(threshold,0)),f.cat_fuel_ever.to_numpy()]);ct=[];ce=[]
  for k,name in enumerate(['purchase','fuel_growth_event']):
   true=np.concatenate([targets[v][name].reindex(panel['frames'][v].index).to_numpy() for v in cuts]);logit=lambda p:np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)));cal=LogisticRegression(C=1,max_iter=300,random_state=1410).fit(logit(binary[:,k]).reshape(-1,1),true,sample_weight=weights);causal=[]
   for origin,record in zip(cuts,records):
    prior=[v for v in cuts if pd.Timestamp(v)+pd.DateOffset(months=3)<=pd.Timestamp(origin)]
    if prior:
     raw_prior=np.concatenate([get(v)[1][:,k] for v in prior]);target_prior=np.concatenate([targets[v][name].reindex(panel['frames'][v].index).to_numpy() for v in prior])
     if len(np.unique(target_prior))==2:
      prior_cal=LogisticRegression(C=1,max_iter=300,random_state=1410).fit(logit(raw_prior).reshape(-1,1),target_prior);pp=prior_cal.predict_proba(logit(record[1][:,k]).reshape(-1,1))[:,1]
     else:pp=record[1][:,k]
    else:pp=record[1][:,k]
    causal.append(logit(pp))
   eval_cal=cal.predict_proba(logit(beval[:,k]).reshape(-1,1))[:,1]
   if name=='fuel_growth_event':
    causal=[np.where(record[0][:,-1]>0,v,logit(np.zeros(len(v)))) for record,v in zip(records,causal)]
    eval_cal=np.where(xeval[:,-1]>0,eval_cal,0)
   ct.append(np.concatenate(causal));ce.append(logit(eval_cal))
  scaler=StandardScaler();xx=scaler.fit_transform(np.column_stack([x,*ct]));xe=scaler.transform(np.column_stack([xeval,*ce]));model=LogisticRegression(C=.05,max_iter=500,random_state=1410).fit(xx,y,sample_weight=weights);pp=np.zeros((len(f),17));small=model.predict_proba(xe)
  for k,label in enumerate(model.classes_):pp[:,r.CLASSES.index(label)]=small[:,k]
  pp[f.cat_fuel_ever.le(0).to_numpy(),r.CLASSES.index('Existing-category growth: Fuel')]=0;pp/=pp.sum(axis=1,keepdims=True)
  best=r.REPO/'submissions/sasol_oct10_small_fuel_values.csv';assert r.sha(best)==r.EXPECTED;df=pd.read_csv(best,dtype={'ID':str});df.Opportunity=df.ID.map(pd.Series(r.labels_from(pp),index=f.index));assert not df.isna().any().any();path=r.OUT/'fuel_rule_correction_v1_submission.csv';df.to_csv(path,index=False,float_format='%.17g');r.save(path.stem+'.audit.json',{'baseline_sha256':r.EXPECTED,'output_sha256':r.sha(path),'rows':len(df),'targets':['Opportunity'],'public_score':None})
