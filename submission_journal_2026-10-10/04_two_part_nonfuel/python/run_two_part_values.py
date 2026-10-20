"""Monthly two-part CLV forecast, with a fixed direct/hurdle blend and six-origin gate."""
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


def hurdle(panel,cutoff,target,metadata_prefix="two_part"):
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
    tr=train.cutoff.le(limit).to_numpy();va=train.cutoff.eq(latest).to_numpy();pos=train.target.gt(0).to_numpy()
    assert (pd.to_datetime(train.loc[tr,"cutoff"])+pd.DateOffset(months=3)<=pd.Timestamp(latest)).all()
    outputs=[];metadata=[]
    for seed in ([7,21] if target=="CLV_fuel" else [11,29]):
        params=dict(n_estimators=1000,learning_rate=0.04,num_leaves=15,min_child_samples=50,
                    subsample=0.90,subsample_freq=1,colsample_bytree=0.90,reg_lambda=1.2,
                    random_state=seed,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1)
        clf=LGBMClassifier(**params)
        clf.fit(train.loc[tr,features],pos[tr].astype(int),sample_weight=train.loc[tr,"w"],
                eval_set=[(train.loc[va,features],pos[va].astype(int))],callbacks=[early_stopping(60,verbose=False)])
        nclass=max(clf.best_iteration_,40)
        clf=LGBMClassifier(**dict(params,n_estimators=nclass))
        clf.fit(train[features],pos.astype(int),sample_weight=train.w)
        assert np.array_equal(clf.classes_,[0,1])
        probability=clf.predict_proba(panel["frames"][cutoff][features])[:,1]
        reg=LGBMRegressor(**params)
        reg.fit(train.loc[tr&pos,features],train.loc[tr&pos,"target"],sample_weight=train.loc[tr&pos,"w"],
                eval_set=[(train.loc[va&pos,features],train.loc[va&pos,"target"])],callbacks=[early_stopping(60,verbose=False)])
        nreg=max(reg.best_iteration_,40)
        reg=LGBMRegressor(**dict(params,n_estimators=nreg))
        reg.fit(train.loc[pos,features],train.loc[pos,"target"],sample_weight=train.loc[pos,"w"])
        conditional=np.maximum(reg.predict(panel["frames"][cutoff][features]),0)
        outputs.append(probability*conditional)
        metadata.append(dict(seed=seed,purchase_rounds=nclass,conditional_rounds=nreg,train_rows=len(train),latest_training_origin=latest))
        print(f"two-part {cutoff} {target} seed={seed} purchase={nclass} value={nreg}",flush=True)
    (OUT/f"{cutoff}_{target}_{metadata_prefix}_training.json").write_text(json.dumps(metadata,indent=2))
    return np.mean(outputs,axis=0)


def forecast(panel,cutoff,target):
    path=OUT/f"{cutoff}_{target}_two_part.npy"
    h=np.load(path) if path.exists() else hurdle(panel,cutoff,target)
    assert np.isfinite(h).all() and (h>=0).all()
    np.save(path,h)
    direct=cached_predict(panel,cutoff,target,"category_values")
    return (direct+h)/2,direct


def main():
    panel=pickle.loads((OUT/"category_dynamics_panel.pkl").read_bytes())
    recorded=json.loads((OUT/"confirmed_baseline.json").read_text())["historical_report"]
    rows=[];screen={};accepted=[]
    for target in ["CLV_fuel","CLV_nonfuel"]:
        gains=[]
        for cutoff,key in [("2025-06-01","june"),("2025-09-01","september")]:
            p,cp=forecast(panel,cutoff,target)
            y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index)[target].to_numpy()
            error=fp.rmse(y,p);ce=fp.rmse(y,cp)
            reference=recorded[key]["rmse_fuel" if target=="CLV_fuel" else "rmse_nonfuel"]
            gains.append(min(ce,reference)-error)
            row=dict(cutoff=cutoff,target=target,control_rmse=ce,candidate_rmse=error,
                     recorded_baseline_rmse=reference,gain_vs_control=ce-error,gain_vs_recorded=reference-error)
            rows.append(row);print(row,flush=True)
            pd.DataFrame(rows).to_csv(OUT/"two_part_value_comparison.csv",index=False)
        screen[target]=dict(gains=gains,passed=bool(min(gains)>0 and np.mean(gains)>=0.001))
    for target in screen:
        if not screen[target]["passed"]:continue
        for cutoff in ["2025-04-01","2025-05-01","2025-07-01","2025-08-01"]:
            p,cp=forecast(panel,cutoff,target)
            y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index)[target].to_numpy()
            error=fp.rmse(y,p);ce=fp.rmse(y,cp)
            row=dict(cutoff=cutoff,target=target,control_rmse=ce,candidate_rmse=error,
                     recorded_baseline_rmse=np.nan,gain_vs_control=ce-error,gain_vs_recorded=np.nan)
            rows.append(row);print(row,flush=True)
            pd.DataFrame(rows).to_csv(OUT/"two_part_value_comparison.csv",index=False)
        gg=[r["gain_vs_control"] for r in rows if r["target"]==target]
        if sum(g>0 for g in gg)>=5 and min(gg)>=-0.002 and np.mean(gg)>=0.001:accepted.append(target)
    result=dict(accepted=accepted,screen=screen,features=len(panel["feature_cols"]),blend_weight=0.5,
                rule="Beat the better of recorded baseline and direct control RMSE on June and September with mean gain >=0.001; then beat direct control in at least 5 of 6 origins, mean gain >=0.001, no loss >0.002.",
                extra_control="Monthly direct low-capacity regressors with same 305 features; not the historical best submission recipe.")
    (OUT/"two_part_value_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if accepted:
        final=pd.DataFrame(index=panel["frames"][fp.TEST_CUTOFF].index)
        for target in accepted:final[target]=forecast(panel,fp.TEST_CUTOFF,target)[0]
        final.to_csv(OUT/"two_part_values.csv",index_label="ID",float_format="%.17g")


if __name__=="__main__":main()
