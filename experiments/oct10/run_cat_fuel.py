"""Test fixed CatBoost diversity against the confirmed small monthly fuel blend."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
import feature_pipeline as fp
from run_experiment import OUT
from run_cat_shop import cat_predict
from run_fuel_cadence import forecast as current_fuel


def forecast(panel,cutoff):
    path=OUT/f"{cutoff}_cat_fuel.npy"
    p=np.load(path) if path.exists() else cat_predict(panel,cutoff,"CLV_fuel","cat_fuel")
    assert np.isfinite(p).all() and (p>=0).all()
    np.save(path,p)
    cp=current_fuel(panel,cutoff,False)
    return 0.75*cp+0.25*p,cp


def main():
    panel=pickle.loads((OUT/"category_dynamics_panel.pkl").read_bytes())
    recorded=json.loads((OUT/"confirmed_small_fuel_baseline.json").read_text())["fuel_validation_rmse"]
    rows=[];screen_gains=[]
    for cutoff,key in [("2025-06-01","june"),("2025-09-01","september")]:
        p,cp=forecast(panel,cutoff)
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).CLV_fuel.to_numpy()
        error=fp.rmse(y,p);ce=fp.rmse(y,cp);reference=recorded[cutoff]
        screen_gains.append(min(ce,reference)-error)
        row=dict(cutoff=cutoff,target="CLV_fuel",control_rmse=ce,candidate_rmse=error,
                 recorded_baseline_rmse=reference,gain_vs_control=ce-error,gain_vs_recorded=reference-error)
        rows.append(row);print(row,flush=True)
        pd.DataFrame(rows).to_csv(OUT/"cat_fuel_value_comparison.csv",index=False)
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
            pd.DataFrame(rows).to_csv(OUT/"cat_fuel_value_comparison.csv",index=False)
        gg=[r["gain_vs_control"] for r in rows]
        if sum(g>0 for g in gg)>=5 and min(gg)>=-0.002 and np.mean(gg)>=0.001:accepted=["CLV_fuel"]
    result=dict(accepted=accepted,screen_gains=screen_gains,features=len(panel["feature_cols"]),catboost_weight=0.25,
                rule="Beat the better of confirmed fuel baseline and current fuel control on both June and September, mean gain >=0.001; improve at least 5 of 6 origins against current fuel control, mean gain >=0.001, no loss >0.002.",
                extra_control="Monthly 40/35/25 direct/hurdle/residual fuel model with same 305 features; the current publicly confirmed fuel recipe.")
    (OUT/"cat_fuel_value_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if accepted:
        p,_=forecast(panel,fp.TEST_CUTOFF)
        pd.DataFrame({"ID":panel["frames"][fp.TEST_CUTOFF].index,"CLV_fuel":p}).to_csv(OUT/"cat_fuel_values.csv",index=False,float_format="%.17g")


if __name__=="__main__":main()
