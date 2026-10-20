"""Test fixed low-capacity CLV regressors using purged monthly training origins."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,early_stopping
import feature_pipeline as fp
from run_experiment import OUT

def predict(panel,cutoff,target,threads=1):
    latest=(pd.Timestamp(cutoff)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    inner_limit=(pd.Timestamp(latest)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    cuts=[c for c in panel["monthly_cutoffs"] if c<=latest]
    blocks=[]
    for c in cuts:
        block=panel["frames"][c].copy()
        block["target"]=panel["labels"][c].reindex(block.index)[target]
        block["cutoff"]=c
        block["w"]=0.70+0.05*((pd.Timestamp(c).year-2024)*12+pd.Timestamp(c).month-9)
        blocks.append(block)
    train=pd.concat(blocks)
    tr=train.cutoff.le(inner_limit).to_numpy(); va=train.cutoff.eq(latest).to_numpy()
    features=panel["feature_cols"]; result=[]
    for seed in ([7,21] if target=="CLV_fuel" else [11,29]):
        params=dict(n_estimators=1000,learning_rate=0.04,num_leaves=15,
            min_child_samples=100 if target=="CLV_fuel" else 50,
            colsample_bytree=0.9,subsample=0.9,subsample_freq=1,
            reg_lambda=1.2,random_state=seed,n_jobs=threads,verbosity=-1)
        if threads>1:params.update(deterministic=True,force_col_wise=True)
        inner=LGBMRegressor(**params)
        inner.fit(train.loc[tr,features],train.loc[tr,"target"],sample_weight=train.loc[tr,"w"],
            eval_set=[(train.loc[va,features],train.loc[va,"target"])],callbacks=[early_stopping(60,verbose=False)])
        params["n_estimators"]=max(inner.best_iteration_,40)
        full=LGBMRegressor(**params)
        full.fit(train[features],train.target,sample_weight=train.w)
        result.append(full.predict(panel["frames"][cutoff][features]))
        print(f"monthly values {cutoff} {target} seed={seed} rounds={params['n_estimators']}",flush=True)
    return np.maximum(np.mean(result,axis=0),0)

def main():
    panel=pickle.loads((OUT/"monthly_panel.pkl").read_bytes())
    recorded=json.loads((OUT/"confirmed_baseline.json").read_text())["historical_report"]
    rows=[]; gains={target:[] for target in ["CLV_fuel","CLV_nonfuel"]}
    for cutoff,key in [("2025-06-01","june"),("2025-09-01","september")]:
        for target in gains:
            path=OUT/f"{cutoff}_{target}_monthly_values.npy"
            p=np.load(path) if path.exists() else predict(panel,cutoff,target)
            np.save(path,p)
            y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index)[target].to_numpy()
            error=fp.rmse(y,p)
            baseline=recorded[key]["rmse_fuel" if target=="CLV_fuel" else "rmse_nonfuel"]
            gains[target].append(baseline-error)
            rows.append(dict(cutoff=cutoff,target=target,baseline_rmse=baseline,monthly_rmse=error,gain=baseline-error))
            pd.DataFrame(rows).to_csv(OUT/"monthly_value_comparison.csv",index=False)
            print(rows[-1],flush=True)
    accepted=[target for target,gg in gains.items() if min(gg)>0 and np.mean(gg)>=0.001]
    result=dict(accepted=accepted,gains=gains,rule="Both folds improve over the confirmed baseline's recorded RMSE; mean reduction >=0.001")
    (OUT/"monthly_value_decision.json").write_text(json.dumps(result,indent=2))
    final=pd.DataFrame(index=panel["frames"][fp.TEST_CUTOFF].index)
    for target in accepted:final[target]=predict(panel,fp.TEST_CUTOFF,target)
    if accepted:final.to_csv(OUT/"monthly_values.csv",index_label="ID",float_format="%.17g")
    print(result,flush=True)

if __name__=="__main__":main()
