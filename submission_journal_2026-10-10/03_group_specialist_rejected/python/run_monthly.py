"""Add monthly training origins; purge outcomes crossing any validation origin."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sasol-mpl")
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,early_stopping
import feature_pipeline as fp
from run_experiment import OUT,score,tune,BEST_RECORDED

CACHE=OUT/"monthly_panel.pkl"

def build():
    panel=pickle.loads((OUT/"panel.pkl").read_bytes())
    cfg,included,shop,classes=fp.load_config()
    d=fp.lr.load_transactions(fp.DOWNLOADS/"train.csv")
    pack=fp.prepare_tables(d,included,shop,set(cfg["adjustment_items"]))
    cutoffs=[t.strftime("%Y-%m-%d") for t in pd.date_range("2024-09-01","2025-09-01",freq="MS")]
    needed=sorted(set(cutoffs+[fp.prev_quarter(c) for c in cutoffs]))
    labels=dict(panel["labels"])
    for c in needed:
        if c not in labels:
            labels[c]=fp.lr.transform(fp.lr.raw_labels(d,c,cfg),cfg).set_index("ID")
            labels[c].index=labels[c].index.astype(str)
    for c in cutoffs:
        if c not in panel["frames"]:
            frame,_=fp.build_features(pack,c,labels[c].index,labels[fp.prev_quarter(c)].Opportunity,included,panel["s_fuel"],panel["s_shop"],classes)
            assert list(frame.columns)==panel["feature_cols"]
            panel["frames"][c]=frame
            print(f"monthly features {c} rows={len(frame)}",flush=True)
    panel["labels"]={c:labels[c] for c in cutoffs}
    panel["monthly_cutoffs"]=cutoffs
    CACHE.write_bytes(pickle.dumps(panel))
    return panel

def predict(panel,cutoff,threads=1,return_models=False,weight_decay_months=None,model_params=None):
    latest=(pd.Timestamp(cutoff)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    inner_limit=(pd.Timestamp(latest)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    cuts=[c for c in panel["monthly_cutoffs"] if c<=latest]
    blocks=[]
    for c in cuts:
        frame=panel["frames"][c].copy()
        frame["Opportunity"]=panel["labels"][c].reindex(frame.index).Opportunity
        frame["cutoff"]=c
        frame["w"]=0.70+0.05*((pd.Timestamp(c).year-2024)*12+pd.Timestamp(c).month-9)
        if weight_decay_months is not None:
            assert weight_decay_months>0
            age=(pd.Timestamp(latest).year-pd.Timestamp(c).year)*12+pd.Timestamp(latest).month-pd.Timestamp(c).month
            frame["w"]*=2.0**(-age/weight_decay_months)
        blocks.append(frame)
    train=pd.concat(blocks)
    features=panel["feature_cols"]; classes=panel["classes"]
    tr=train.cutoff.le(inner_limit).to_numpy(); va=train.cutoff.eq(latest).to_numpy()
    assert (pd.to_datetime(train.loc[tr,"cutoff"])+pd.DateOffset(months=3)<=pd.Timestamp(latest)).all()
    p=np.zeros((len(panel["frames"][cutoff]),len(classes)))
    models=[]
    for seed in [7,23]:
        params=dict(n_estimators=1200,learning_rate=0.04,num_leaves=47,min_child_samples=35,
                    subsample=0.85,subsample_freq=1,colsample_bytree=0.70,reg_lambda=1.2,
                    random_state=seed,n_jobs=threads,verbosity=-1)
        if threads>1:params.update(deterministic=True,force_col_wise=True)
        if model_params:
            assert set(model_params)<= {"num_leaves","min_child_samples","reg_lambda"}
            params.update(model_params)
        inner=LGBMClassifier(**params)
        inner.fit(train.loc[tr,features],train.loc[tr,"Opportunity"],sample_weight=train.loc[tr,"w"],
                  eval_set=[(train.loc[va,features],train.loc[va,"Opportunity"])],callbacks=[early_stopping(60,verbose=False)])
        params["n_estimators"]=max(inner.best_iteration_,40)
        full=LGBMClassifier(**params)
        full.fit(train[features],train.Opportunity,sample_weight=train.w)
        p+=fp.align_proba(full,panel["frames"][cutoff],classes)/2
        if return_models:models.append(full)
        print(f"monthly {cutoff} seed={seed} rounds={params['n_estimators']} train={len(train)}",flush=True)
    return (p,models) if return_models else p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--build-only",action="store_true"); ap.add_argument("--evaluate-only",action="store_true"); args=ap.parse_args()
    panel=pickle.loads(CACHE.read_bytes()) if CACHE.exists() else build()
    if args.build_only:return
    classes=panel["classes"]; rows=[]; gains=[]
    fixed=np.array([4.55 if c=="Existing-category growth: Food & beverage" else 1.17 if c=="Existing-category growth: Fuel" else 1.3 if c.startswith("Existing-category growth") else 2.5 if c.startswith("New category adoption") else 1 for c in classes])
    for cutoff in ["2025-03-01","2025-06-01","2025-09-01"]:
        path=OUT/f"{cutoff}_monthly.npy"
        p=np.load(path) if path.exists() else predict(panel,cutoff)
        np.save(path,p)
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).Opportunity.to_numpy()
        if cutoff=="2025-03-01":mult=tune(y,p,classes)
        f1,pred=score(y,p,classes,mult)
        control_path=OUT/f"{cutoff}_lgb.npy"
        base=score(y,np.load(control_path),classes,fixed)[0] if control_path.exists() else BEST_RECORDED[cutoff]
        rows.append(dict(cutoff=cutoff,control_f1=base,monthly_f1=f1,gain=f1-base))
        pd.DataFrame(rows).to_csv(OUT/"monthly_comparison.csv",index=False)
        pd.DataFrame({"ID":panel["frames"][cutoff].index,"true":y,"pred":pred}).to_csv(OUT/f"{cutoff}_monthly_pred.csv",index=False)
        if cutoff!="2025-03-01":gains.append(f1-max(base,BEST_RECORDED[cutoff]))
        print(rows[-1],flush=True)
    accepted=min(gains)>0 and np.mean(gains)>=0.003
    result=dict(accepted=bool(accepted),gains=gains,multipliers=mult.tolist(),classes=classes)
    (OUT/"monthly_decision.json").write_text(json.dumps(result,indent=2))
    if accepted and not args.evaluate_only:
        p=predict(panel,fp.TEST_CUTOFF)
        np.save(OUT/f"{fp.TEST_CUTOFF}_monthly.npy",p)
        _,pred=score(np.repeat("Stable",len(p)),p,classes,mult)
        pd.DataFrame({"ID":panel["frames"][fp.TEST_CUTOFF].index,"Opportunity":pred}).to_csv(OUT/"monthly_opportunity.csv",index=False)
    print(result,flush=True)

if __name__=="__main__":main()
