"""Restore the baseline's four walk-forward value features on monthly origins."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,early_stopping
import feature_pipeline as fp
from run_experiment import OUT,BEST_RECORDED,score,tune
from run_monthly import predict

CACHE=OUT/"monthly_stacked_panel.pkl"

def build():
    panel=pickle.loads((OUT/"monthly_panel.pkl").read_bytes())
    features=list(panel["feature_cols"])
    stacks={}
    for cutoff in panel["monthly_cutoffs"]+[fp.TEST_CUTOFF]:
        latest=(pd.Timestamp(cutoff)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
        cuts=[c for c in panel["monthly_cutoffs"] if c<=latest]
        pair=[]
        for target,seed,zname in [("CLV_fuel",7,"fuel_z_q1"),("CLV_nonfuel",11,"shop_z_q1")]:
            if not cuts:
                p=panel["frames"][cutoff][zname].to_numpy()
            else:
                blocks=[]
                for c in cuts:
                    b=panel["frames"][c][features].copy()
                    b["target"]=panel["labels"][c].reindex(b.index)[target]
                    b["cutoff"]=c
                    b["w"]=0.70+0.05*((pd.Timestamp(c).year-2024)*12+pd.Timestamp(c).month-9)
                    blocks.append(b)
                train=pd.concat(blocks)
                inner_cutoff=cuts[-1]
                inner_limit=(pd.Timestamp(inner_cutoff)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
                tr=train.cutoff.le(inner_limit).to_numpy(); va=train.cutoff.eq(inner_cutoff).to_numpy()
                params=dict(n_estimators=40,learning_rate=0.04,num_leaves=47,min_child_samples=35,
                    subsample=0.85,subsample_freq=1,colsample_bytree=0.70,reg_lambda=1.2,
                    random_state=seed,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1)
                if tr.any():
                    params["n_estimators"]=900
                    inner=LGBMRegressor(**params)
                    inner.fit(train.loc[tr,features],train.loc[tr,"target"],sample_weight=train.loc[tr,"w"],
                        eval_set=[(train.loc[va,features],train.loc[va,"target"])],callbacks=[early_stopping(60,verbose=False)])
                    params["n_estimators"]=max(inner.best_iteration_,40)
                full=LGBMRegressor(**params)
                full.fit(train[features],train.target,sample_weight=train.w)
                p=np.maximum(full.predict(panel["frames"][cutoff][features]),0)
            pair.append(p)
        stacks[cutoff]=pair
        print(f"monthly stack {cutoff} past origins={len(cuts)}",flush=True)
    for cutoff,pair in stacks.items():
        frame=panel["frames"][cutoff].copy()
        frame["stack_fuel_z"]=pair[0]; frame["stack_shop_z"]=pair[1]
        frame["stack_fuel_dz"]=pair[0]-frame.fuel_z_q1
        frame["stack_shop_dz"]=pair[1]-frame.shop_z_q1
        panel["frames"][cutoff]=frame
    panel["feature_cols"]=features+["stack_fuel_z","stack_shop_z","stack_fuel_dz","stack_shop_dz"]
    CACHE.write_bytes(pickle.dumps(panel))
    return panel

def main():
    panel=pickle.loads(CACHE.read_bytes()) if CACHE.exists() else build()
    classes=panel["classes"]; rows=[]; gains=[]
    for cutoff in ["2025-03-01","2025-06-01","2025-09-01"]:
        path=OUT/f"{cutoff}_monthly_stacked.npy"
        p=np.load(path) if path.exists() else predict(panel,cutoff,threads=2)
        np.save(path,p)
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).Opportunity.to_numpy()
        if cutoff=="2025-03-01":mult=tune(y,p,classes)
        f1,pred=score(y,p,classes,mult)
        base=BEST_RECORDED.get(cutoff,float("nan"))
        rows.append(dict(cutoff=cutoff,baseline_f1=base,monthly_stacked_f1=f1,gain=f1-base))
        pd.DataFrame(rows).to_csv(OUT/"monthly_stacked_comparison.csv",index=False)
        pd.DataFrame({"ID":panel["frames"][cutoff].index,"true":y,"pred":pred}).to_csv(OUT/f"{cutoff}_monthly_stacked_pred.csv",index=False)
        if cutoff in BEST_RECORDED:
            reference=base
            compare=pd.read_csv(OUT/"comparison.csv")
            control=compare[compare.cutoff.eq(cutoff)&compare.model.eq("control")]
            if len(control):reference=max(reference,float(control.weighted_f1.iloc[0]))
            gains.append(f1-reference)
        print(rows[-1],flush=True)
    accepted=min(gains)>0 and np.mean(gains)>=0.003
    result=dict(accepted=bool(accepted),gains=gains,classes=classes,multipliers=mult.tolist(),features=len(panel["feature_cols"]),rule="Both June and September improve over the larger available control or recorded baseline; mean gain >=0.003")
    (OUT/"monthly_stacked_decision.json").write_text(json.dumps(result,indent=2))
    if accepted:
        path=OUT/f"{fp.TEST_CUTOFF}_monthly_stacked.npy"
        p=np.load(path) if path.exists() else predict(panel,fp.TEST_CUTOFF,threads=2)
        np.save(path,p)
        pred=np.asarray(classes)[np.argmax(p*mult,axis=1)]
        pd.DataFrame({"ID":panel["frames"][fp.TEST_CUTOFF].index,"Opportunity":pred}).to_csv(OUT/"monthly_stacked_opportunity.csv",index=False)
    print(result,flush=True)

if __name__=="__main__":main()
