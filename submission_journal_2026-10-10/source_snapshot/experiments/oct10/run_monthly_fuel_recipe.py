"""Apply the original fuel blend and model capacity to enriched monthly snapshots."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,LGBMRegressor,early_stopping
import feature_pipeline as fp
from run_experiment import OUT
from run_category_values import cached_predict
from run_two_part_values import hurdle as small_hurdle
from run_residual_fuel import residual as small_residual


def recipe(panel,cutoff):
    latest=(pd.Timestamp(cutoff)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit=(pd.Timestamp(latest)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks=[]
    for c in panel["monthly_cutoffs"]:
        if c>latest:continue
        b=panel["frames"][c].copy()
        b["target"]=panel["labels"][c].reindex(b.index).CLV_fuel
        b["delta"]=b.target-b.fuel_z_q1
        b["cutoff"]=c
        b["w"]=0.70+0.05*((pd.Timestamp(c).year-2024)*12+pd.Timestamp(c).month-9)
        blocks.append(b)
    train=pd.concat(blocks);features=panel["feature_cols"]
    tr=train.cutoff.le(limit).to_numpy();va=train.cutoff.eq(latest).to_numpy();pos=train.target.gt(0).to_numpy()
    assert (pd.to_datetime(train.loc[tr,"cutoff"])+pd.DateOffset(months=3)<=pd.Timestamp(latest)).all()
    X=panel["frames"][cutoff][features];parts={};meta=[]
    for kind,seeds in [("direct",[7,21]),("residual",[7]),("conditional",[7])]:
        outputs=[]
        label="delta" if kind=="residual" else "target"
        inner_mask=tr&pos if kind=="conditional" else tr
        valid_mask=va&pos if kind=="conditional" else va
        full_mask=pos if kind=="conditional" else np.ones(len(train),dtype=bool)
        for seed in seeds:
            params=dict(n_estimators=1400 if kind=="direct" else 1000,learning_rate=0.04,
                        num_leaves=47,min_child_samples=25 if kind=="conditional" else 35,
                        subsample=0.85,subsample_freq=1,colsample_bytree=0.70,reg_lambda=1.2,
                        random_state=seed,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1)
            inner=LGBMRegressor(**params)
            inner.fit(train.loc[inner_mask,features],train.loc[inner_mask,label],sample_weight=train.loc[inner_mask,"w"],
                      eval_set=[(train.loc[valid_mask,features],train.loc[valid_mask,label])],callbacks=[early_stopping(60,verbose=False)])
            params["n_estimators"]=max(inner.best_iteration_,30 if kind=="conditional" else 40)
            full=LGBMRegressor(**params)
            full.fit(train.loc[full_mask,features],train.loc[full_mask,label],sample_weight=train.loc[full_mask,"w"])
            outputs.append(full.predict(X))
            meta.append(dict(part=kind,seed=seed,rounds=params["n_estimators"],train_rows=int(full_mask.sum())))
            print(f"fuel recipe {cutoff} {kind} seed={seed} rounds={params['n_estimators']}",flush=True)
        value=np.mean(outputs,axis=0)
        if kind=="residual":value+=panel["frames"][cutoff].fuel_z_q1.to_numpy()
        parts[kind]=np.maximum(value,0)
    params=dict(n_estimators=1000,learning_rate=0.05,num_leaves=31,min_child_samples=40,
                subsample=0.85,subsample_freq=1,colsample_bytree=0.70,reg_lambda=1.0,
                random_state=7,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1)
    inner=LGBMClassifier(**params)
    inner.fit(train.loc[tr,features],pos[tr].astype(int),sample_weight=train.loc[tr,"w"],
              eval_set=[(train.loc[va,features],pos[va].astype(int))],callbacks=[early_stopping(50,verbose=False)])
    params["n_estimators"]=max(inner.best_iteration_,30)
    full=LGBMClassifier(**params)
    full.fit(train[features],pos.astype(int),sample_weight=train.w)
    assert np.array_equal(full.classes_,[0,1])
    parts["hurdle"]=full.predict_proba(X)[:,1]*parts["conditional"]
    meta.append(dict(part="purchase_probability",seed=7,rounds=params["n_estimators"],train_rows=len(train)))
    print(f"fuel recipe {cutoff} purchase rounds={params['n_estimators']}",flush=True)
    (OUT/f"{cutoff}_fuel_recipe_training.json").write_text(json.dumps(meta,indent=2))
    return 0.4*parts["direct"]+0.35*parts["hurdle"]+0.25*parts["residual"]


def forecast(panel,cutoff):
    path=OUT/f"{cutoff}_fuel_recipe.npy"
    p=np.load(path) if path.exists() else recipe(panel,cutoff)
    assert np.isfinite(p).all() and (p>=0).all()
    np.save(path,p)
    direct=cached_predict(panel,cutoff,"CLV_fuel","category_values")
    hp=OUT/f"{cutoff}_CLV_fuel_two_part.npy"
    h=np.load(hp) if hp.exists() else small_hurdle(panel,cutoff,"CLV_fuel")
    np.save(hp,h)
    rp=OUT/f"{cutoff}_fuel_residual.npy"
    r=np.load(rp) if rp.exists() else small_residual(panel,cutoff)
    np.save(rp,r)
    return p,0.4*direct+0.35*h+0.25*r


def main():
    panel=pickle.loads((OUT/"category_dynamics_panel.pkl").read_bytes())
    recorded=json.loads((OUT/"confirmed_baseline.json").read_text())["historical_report"]
    rows=[];gains=[];accepted=[]
    for cutoff,key in [("2025-06-01","june"),("2025-09-01","september")]:
        p,cp=forecast(panel,cutoff)
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).CLV_fuel.to_numpy()
        error=fp.rmse(y,p);ce=fp.rmse(y,cp);reference=recorded[key]["rmse_fuel"]
        gains.append(min(ce,reference)-error)
        row=dict(cutoff=cutoff,target="CLV_fuel",control_rmse=ce,candidate_rmse=error,
                 recorded_baseline_rmse=reference,gain_vs_control=ce-error,gain_vs_recorded=reference-error)
        rows.append(row);print(row,flush=True)
        pd.DataFrame(rows).to_csv(OUT/"fuel_recipe_value_comparison.csv",index=False)
    passed=min(gains)>0 and np.mean(gains)>=0.001
    if passed:
        for cutoff in ["2025-04-01","2025-05-01","2025-07-01","2025-08-01"]:
            p,cp=forecast(panel,cutoff)
            y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).CLV_fuel.to_numpy()
            error=fp.rmse(y,p);ce=fp.rmse(y,cp)
            row=dict(cutoff=cutoff,target="CLV_fuel",control_rmse=ce,candidate_rmse=error,
                     recorded_baseline_rmse=np.nan,gain_vs_control=ce-error,gain_vs_recorded=np.nan)
            rows.append(row);print(row,flush=True)
            pd.DataFrame(rows).to_csv(OUT/"fuel_recipe_value_comparison.csv",index=False)
        gg=[r["gain_vs_control"] for r in rows]
        if sum(g>0 for g in gg)>=5 and min(gg)>=-0.002 and np.mean(gg)>=0.001:accepted=["CLV_fuel"]
    result=dict(accepted=accepted,screen_gains=gains,features=len(panel["feature_cols"]),
                blend=dict(direct=0.4,hurdle=0.35,residual=0.25),
                rule="Beat the better of recorded fuel baseline and smaller-model control on both June and September, mean gain >=0.001; improve at least 5 of 6 origins against control, mean gain >=0.001, no loss >0.002.",
                extra_control="Same fixed fuel blend with low-capacity enriched monthly models; not original submission's exact validation predictions.")
    (OUT/"fuel_recipe_value_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if accepted:
        cutoff=fp.TEST_CUTOFF;path=OUT/f"{cutoff}_fuel_recipe.npy"
        p=np.load(path) if path.exists() else recipe(panel,cutoff)
        np.save(path,p)
        pd.DataFrame({"ID":panel["frames"][cutoff].index,"CLV_fuel":p}).to_csv(OUT/"fuel_recipe_values.csv",index=False,float_format="%.17g")


if __name__=="__main__":main()
