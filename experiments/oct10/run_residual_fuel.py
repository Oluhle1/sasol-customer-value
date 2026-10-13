"""Test a fixed residual/persistence component for monthly fuel CLV forecasts."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,early_stopping
import feature_pipeline as fp
from run_experiment import OUT
from run_two_part_values import forecast as direct_hurdle


def residual(panel,cutoff,metadata_prefix="fuel_residual"):
    latest=(pd.Timestamp(cutoff)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit=(pd.Timestamp(latest)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks=[]
    for c in panel["monthly_cutoffs"]:
        if c>latest:continue
        b=panel["frames"][c].copy()
        b["delta"]=panel["labels"][c].reindex(b.index).CLV_fuel-b.fuel_z_q1
        b["cutoff"]=c
        b["w"]=0.70+0.05*((pd.Timestamp(c).year-2024)*12+pd.Timestamp(c).month-9)
        blocks.append(b)
    train=pd.concat(blocks);features=panel["feature_cols"]
    tr=train.cutoff.le(limit).to_numpy();va=train.cutoff.eq(latest).to_numpy()
    assert (pd.to_datetime(train.loc[tr,"cutoff"])+pd.DateOffset(months=3)<=pd.Timestamp(latest)).all()
    predictions=[];meta=[]
    for seed in [7,21]:
        params=dict(n_estimators=1000,learning_rate=0.04,num_leaves=15,min_child_samples=100,
                    colsample_bytree=0.9,subsample=0.9,subsample_freq=1,reg_lambda=1.2,
                    random_state=seed,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1)
        inner=LGBMRegressor(**params)
        inner.fit(train.loc[tr,features],train.loc[tr,"delta"],sample_weight=train.loc[tr,"w"],
                  eval_set=[(train.loc[va,features],train.loc[va,"delta"])],callbacks=[early_stopping(60,verbose=False)])
        params["n_estimators"]=max(inner.best_iteration_,40)
        full=LGBMRegressor(**params)
        full.fit(train[features],train.delta,sample_weight=train.w)
        predictions.append(full.predict(panel["frames"][cutoff][features]))
        meta.append(dict(seed=seed,rounds=params["n_estimators"],train_rows=len(train),latest_training_origin=latest))
        print(f"fuel residual {cutoff} seed={seed} rounds={params['n_estimators']}",flush=True)
    (OUT/f"{cutoff}_{metadata_prefix}_training.json").write_text(json.dumps(meta,indent=2))
    return np.maximum(panel["frames"][cutoff].fuel_z_q1.to_numpy()+np.mean(predictions,axis=0),0)


def forecast(panel,cutoff):
    path=OUT/f"{cutoff}_fuel_residual.npy"
    p=np.load(path) if path.exists() else residual(panel,cutoff)
    assert np.isfinite(p).all() and (p>=0).all()
    np.save(path,p)
    control,_=direct_hurdle(panel,cutoff,"CLV_fuel")
    return 0.75*control+0.25*p,control


def main():
    panel=pickle.loads((OUT/"category_dynamics_panel.pkl").read_bytes())
    recorded=json.loads((OUT/"confirmed_baseline.json").read_text())["historical_report"]
    rows=[];screen_gains=[]
    for cutoff,key in [("2025-06-01","june"),("2025-09-01","september")]:
        p,cp=forecast(panel,cutoff)
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).CLV_fuel.to_numpy()
        error=fp.rmse(y,p);ce=fp.rmse(y,cp);reference=recorded[key]["rmse_fuel"]
        screen_gains.append(min(ce,reference)-error)
        row=dict(cutoff=cutoff,target="CLV_fuel",control_rmse=ce,candidate_rmse=error,
                 recorded_baseline_rmse=reference,gain_vs_control=ce-error,gain_vs_recorded=reference-error)
        rows.append(row);print(row,flush=True)
        pd.DataFrame(rows).to_csv(OUT/"residual_value_comparison.csv",index=False)
    passed=min(screen_gains)>0 and np.mean(screen_gains)>=0.001
    accepted=[]
    if passed:
        for cutoff in ["2025-04-01","2025-05-01","2025-07-01","2025-08-01"]:
            p,cp=forecast(panel,cutoff)
            y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).CLV_fuel.to_numpy()
            error=fp.rmse(y,p);ce=fp.rmse(y,cp)
            row=dict(cutoff=cutoff,target="CLV_fuel",control_rmse=ce,candidate_rmse=error,
                     recorded_baseline_rmse=np.nan,gain_vs_control=ce-error,gain_vs_recorded=np.nan)
            rows.append(row);print(row,flush=True)
            pd.DataFrame(rows).to_csv(OUT/"residual_value_comparison.csv",index=False)
        gg=[r["gain_vs_control"] for r in rows]
        if sum(g>0 for g in gg)>=5 and min(gg)>=-0.002 and np.mean(gg)>=0.001:accepted=["CLV_fuel"]
    result=dict(accepted=accepted,screen_gains=screen_gains,features=len(panel["feature_cols"]),residual_weight=0.25,
                rule="Beat the better of recorded fuel baseline and two-part control on both June and September, mean gain >=0.001; improve at least 5 of 6 origins against two-part control, mean gain >=0.001, no loss >0.002.",
                extra_control="Monthly 50/50 direct/hurdle fuel model with same 305 features; not original submission's exact validation predictions.")
    (OUT/"residual_value_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if accepted:
        p,_=forecast(panel,fp.TEST_CUTOFF)
        pd.DataFrame({"ID":panel["frames"][fp.TEST_CUTOFF].index,"CLV_fuel":p}).to_csv(OUT/"residual_values.csv",index=False,float_format="%.17g")


if __name__=="__main__":main()
