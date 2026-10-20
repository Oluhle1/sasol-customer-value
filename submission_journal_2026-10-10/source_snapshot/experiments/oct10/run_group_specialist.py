"""Blend a three-group specialist with the confirmed category-dynamics classifier."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,early_stopping
import feature_pipeline as fp
from run_category_dynamics import CACHE
from run_experiment import OUT,score

FUEL="Existing-category growth: Fuel"


def groups(labels):
    return np.where(np.asarray(labels)==FUEL,0,np.where(np.asarray(labels)=="Inactivity",1,2))


def predict_group(panel,cutoff):
    latest=(pd.Timestamp(cutoff)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    inner_limit=(pd.Timestamp(latest)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks=[]
    for c in panel["monthly_cutoffs"]:
        if c>latest:continue
        b=panel["frames"][c].copy()
        b["target"]=groups(panel["labels"][c].reindex(b.index).Opportunity)
        b["cutoff"]=c
        b["weight"]=0.70+0.05*((pd.Timestamp(c).year-2024)*12+pd.Timestamp(c).month-9)
        blocks.append(b)
    train=pd.concat(blocks);features=panel["feature_cols"]
    tr=train.cutoff.le(inner_limit).to_numpy();va=train.cutoff.eq(latest).to_numpy()
    assert tr.any() and va.any()
    assert (pd.to_datetime(train.loc[tr,"cutoff"])+pd.DateOffset(months=3)<=pd.Timestamp(latest)).all()
    p=np.zeros((len(panel["frames"][cutoff]),3))
    metadata=[]
    for seed in [7,23]:
        params=dict(n_estimators=1000,learning_rate=0.04,num_leaves=31,min_child_samples=50,
                    subsample=0.85,subsample_freq=1,colsample_bytree=0.70,reg_lambda=2.0,
                    random_state=seed,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1)
        inner=LGBMClassifier(**params)
        inner.fit(train.loc[tr,features],train.loc[tr,"target"],sample_weight=train.loc[tr,"weight"],
                  eval_set=[(train.loc[va,features],train.loc[va,"target"])],callbacks=[early_stopping(60,verbose=False)])
        params["n_estimators"]=max(inner.best_iteration_,40)
        full=LGBMClassifier(**params)
        full.fit(train[features],train.target,sample_weight=train.weight)
        assert np.array_equal(full.classes_,[0,1,2])
        p+=full.predict_proba(panel["frames"][cutoff][features])/2
        metadata.append(dict(seed=seed,rounds=params["n_estimators"],train_rows=len(train),latest_training_origin=latest,
                             inner_training_outcomes_end_by=latest))
        print(f"group specialist {cutoff} seed={seed} rounds={params['n_estimators']} rows={len(train)}",flush=True)
    (OUT/f"{cutoff}_group_training.json").write_text(json.dumps(metadata,indent=2))
    return p


def blend(base,group,classes):
    fuel=classes.index(FUEL);inactive=classes.index("Inactivity")
    rest=[i for i in range(len(classes)) if i not in [fuel,inactive]]
    old_groups=np.column_stack([base[:,fuel],base[:,inactive],base[:,rest].sum(axis=1)])
    combined=(old_groups+group)/2
    p=base.copy()
    p[:,fuel]=combined[:,0];p[:,inactive]=combined[:,1]
    p[:,rest]=base[:,rest]*np.divide(combined[:,2],old_groups[:,2],
                                     out=np.zeros(len(base)),where=old_groups[:,2]>0)[:,None]
    assert np.isfinite(p).all() and (p>=0).all()
    np.testing.assert_allclose(p.sum(axis=1),1,rtol=1e-6,atol=1e-6)
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental",action="store_true",help="Fit final predictions when both validation periods improve but the mean-gain threshold fails.")
    args=parser.parse_args()
    panel=pickle.loads(CACHE.read_bytes())
    baseline=json.loads((OUT/"confirmed_category_baseline.json").read_text())
    decisions=json.loads((OUT/"category_dynamics_decision.json").read_text())
    classes=panel["classes"];mult=np.asarray(decisions["multipliers"])
    rows=[]
    for cutoff in ["2025-06-01","2025-09-01"]:
        path=OUT/f"{cutoff}_group_specialist.npy"
        gp=np.load(path) if path.exists() else predict_group(panel,cutoff)
        np.save(path,gp)
        base=np.load(OUT/f"{cutoff}_category_dynamics.npy")
        p=blend(base,gp,classes)
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).Opportunity.to_numpy()
        b,_=score(y,base,classes,mult);f1,pred=score(y,p,classes,mult)
        np.testing.assert_allclose(b,baseline["opportunity_validation_f1"][cutoff],rtol=1e-12,atol=1e-12)
        row=dict(cutoff=cutoff,category_baseline_f1=b,weighted_f1=f1,gain=f1-b)
        rows.append(row);print(row,flush=True)
        pd.DataFrame(rows).to_csv(OUT/"group_specialist_comparison.csv",index=False)
        pd.DataFrame({"ID":panel["frames"][cutoff].index,"true":y,"pred":pred}).to_csv(OUT/f"{cutoff}_group_specialist_pred.csv",index=False)
    gains=[r["gain"] for r in rows]
    accepted=min(gains)>0 and np.mean(gains)>=0.001
    result=dict(accepted=bool(accepted),gains=gains,classes=classes,multipliers=mult.tolist(),
                group_blend_weight=0.5,baseline_public_score=baseline["public_score"],
                experimental_export=bool(args.experimental and not accepted),
                rule="Both June and September improve over category baseline; mean F1 gain >=0.001.")
    (OUT/"group_specialist_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if accepted or (args.experimental and min(gains)>0):
        cutoff=fp.TEST_CUTOFF;path=OUT/f"{cutoff}_group_specialist.npy"
        gp=np.load(path) if path.exists() else predict_group(panel,cutoff)
        np.save(path,gp)
        p=blend(np.load(OUT/f"{cutoff}_category_dynamics.npy"),gp,classes)
        pred=np.asarray(classes)[np.argmax(p*mult,axis=1)]
        pd.DataFrame({"ID":panel["frames"][cutoff].index,"Opportunity":pred}).to_csv(OUT/"group_specialist_opportunity.csv",index=False)


if __name__=="__main__":main()
