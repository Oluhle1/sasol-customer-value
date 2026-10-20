"""Test a fixed CatBoost diversity blend against the confirmed nonfuel recipe."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import feature_pipeline as fp
from run_experiment import OUT
from run_two_part_values import forecast as direct_hurdle


def cat_predict(panel,cutoff,target="CLV_nonfuel",metadata_prefix="cat_shop"):
    seeds=[11,29] if target=="CLV_nonfuel" else [7,21]
    latest=(pd.Timestamp(cutoff)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit=(pd.Timestamp(latest)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks=[]
    for c in panel["monthly_cutoffs"]:
        if c>latest:continue
        b=panel["frames"][c].copy()
        b["target"]=panel["labels"][c].reindex(b.index)[target]
        b["cutoff"]=c
        b["w"]=0.70+0.05*((pd.Timestamp(c).year-2024)*12+pd.Timestamp(c).month-9)
        blocks.append(b)
    train=pd.concat(blocks);features=panel["feature_cols"]
    tr=train.cutoff.le(limit).to_numpy();va=train.cutoff.eq(latest).to_numpy()
    assert (pd.to_datetime(train.loc[tr,"cutoff"])+pd.DateOffset(months=3)<=pd.Timestamp(latest)).all()
    X=train[features].replace([np.inf,-np.inf],np.nan).fillna(0)
    test=panel["frames"][cutoff][features].replace([np.inf,-np.inf],np.nan).fillna(0)
    outputs=[];meta=[]
    for seed in seeds:
        params=dict(iterations=1000,learning_rate=0.04,depth=6,l2_leaf_reg=4.0,
                    loss_function="RMSE",random_seed=seed,verbose=False,
                    allow_writing_files=False,thread_count=2)
        inner=CatBoostRegressor(**params)
        inner.fit(X.loc[tr],train.loc[tr,"target"],sample_weight=train.loc[tr,"w"],
                  eval_set=(X.loc[va],train.loc[va,"target"]),early_stopping_rounds=60,use_best_model=True)
        rounds=max(inner.get_best_iteration()+1,40)
        full=CatBoostRegressor(**dict(params,iterations=rounds))
        full.fit(X,train.target,sample_weight=train.w)
        outputs.append(np.maximum(full.predict(test),0))
        meta.append(dict(seed=seed,rounds=rounds,train_rows=len(train),latest_training_origin=latest))
        print(f"CatBoost {target} {cutoff} seed={seed} rounds={rounds}",flush=True)
    (OUT/f"{cutoff}_{metadata_prefix}_training.json").write_text(json.dumps(meta,indent=2))
    return np.mean(outputs,axis=0)


def forecast(panel,cutoff):
    path=OUT/f"{cutoff}_cat_shop.npy"
    p=np.load(path) if path.exists() else cat_predict(panel,cutoff)
    assert np.isfinite(p).all() and (p>=0).all()
    np.save(path,p)
    cp,_=direct_hurdle(panel,cutoff,"CLV_nonfuel")
    return 0.75*cp+0.25*p,cp


def main():
    panel=pickle.loads((OUT/"category_dynamics_panel.pkl").read_bytes())
    recorded=json.loads((OUT/"confirmed_small_fuel_baseline.json").read_text())["nonfuel_validation_rmse"]
    rows=[];screen_gains=[]
    for cutoff,key in [("2025-06-01","june"),("2025-09-01","september")]:
        p,cp=forecast(panel,cutoff)
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).CLV_nonfuel.to_numpy()
        error=fp.rmse(y,p);ce=fp.rmse(y,cp);reference=recorded[cutoff]
        screen_gains.append(min(ce,reference)-error)
        row=dict(cutoff=cutoff,target="CLV_nonfuel",control_rmse=ce,candidate_rmse=error,
                 recorded_baseline_rmse=reference,gain_vs_control=ce-error,gain_vs_recorded=reference-error)
        rows.append(row);print(row,flush=True)
        pd.DataFrame(rows).to_csv(OUT/"cat_shop_value_comparison.csv",index=False)
    passed=min(screen_gains)>0 and np.mean(screen_gains)>=0.001
    accepted=[]
    if passed:
        for cutoff in ["2025-04-01","2025-05-01","2025-07-01","2025-08-01"]:
            p,cp=forecast(panel,cutoff)
            y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).CLV_nonfuel.to_numpy()
            error=fp.rmse(y,p);ce=fp.rmse(y,cp)
            row=dict(cutoff=cutoff,target="CLV_nonfuel",control_rmse=ce,candidate_rmse=error,
                     recorded_baseline_rmse=np.nan,gain_vs_control=ce-error,gain_vs_recorded=np.nan)
            rows.append(row);print(row,flush=True)
            pd.DataFrame(rows).to_csv(OUT/"cat_shop_value_comparison.csv",index=False)
        gg=[r["gain_vs_control"] for r in rows]
        if sum(g>0 for g in gg)>=5 and min(gg)>=-0.002 and np.mean(gg)>=0.001:accepted=["CLV_nonfuel"]
    result=dict(accepted=accepted,screen_gains=screen_gains,features=len(panel["feature_cols"]),catboost_weight=0.25,
                rule="Beat the better of confirmed nonfuel baseline and two-part control on both June and September, mean gain >=0.001; improve at least 5 of 6 origins against two-part control, mean gain >=0.001, no loss >0.002.",
                extra_control="Monthly 50/50 direct/hurdle nonfuel model with same 305 features; the current publicly confirmed nonfuel recipe.")
    (OUT/"cat_shop_value_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if accepted:
        p,_=forecast(panel,fp.TEST_CUTOFF)
        pd.DataFrame({"ID":panel["frames"][fp.TEST_CUTOFF].index,"CLV_nonfuel":p}).to_csv(OUT/"cat_shop_values.csv",index=False,float_format="%.17g")


if __name__=="__main__":main()
