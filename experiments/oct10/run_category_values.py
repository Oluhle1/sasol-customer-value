"""Test category dynamics for CLV; screen two known folds then four extra origins."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
import feature_pipeline as fp
from run_experiment import OUT
from run_monthly_values import predict


def cached_predict(panel,cutoff,target,kind):
    path=OUT/f"{cutoff}_{target}_{kind}.npy"
    p=np.load(path) if path.exists() else predict(panel,cutoff,target,threads=2)
    assert p.shape==(len(panel["frames"][cutoff]),) and np.isfinite(p).all() and (p>=0).all()
    np.save(path,p)
    return p


def main():
    enriched=pickle.loads((OUT/"category_dynamics_panel.pkl").read_bytes())
    control=pickle.loads((OUT/"monthly_panel.pkl").read_bytes())
    record=json.loads((OUT/"confirmed_baseline.json").read_text())["historical_report"]
    rows=[];screen={};accepted=[]
    for target in ["CLV_fuel","CLV_nonfuel"]:
        gains=[]
        for cutoff,key in [("2025-06-01","june"),("2025-09-01","september")]:
            p=cached_predict(enriched,cutoff,target,"category_values")
            cp=cached_predict(control,cutoff,target,"monthly_values")
            y=enriched["labels"][cutoff].reindex(enriched["frames"][cutoff].index)[target].to_numpy()
            error=fp.rmse(y,p);ce=fp.rmse(y,cp)
            baseline=record[key]["rmse_fuel" if target=="CLV_fuel" else "rmse_nonfuel"]
            gains.append(min(ce,baseline)-error)
            row=dict(cutoff=cutoff,target=target,control_rmse=ce,category_rmse=error,
                     recorded_baseline_rmse=baseline,gain_vs_control=ce-error,gain_vs_recorded=baseline-error)
            rows.append(row);print(row,flush=True)
            pd.DataFrame(rows).to_csv(OUT/"category_value_comparison.csv",index=False)
        screen[target]=dict(gains=gains,passed=bool(min(gains)>0 and np.mean(gains)>=0.001))
    for target in screen:
        if not screen[target]["passed"]:continue
        for cutoff in ["2025-04-01","2025-05-01","2025-07-01","2025-08-01"]:
            p=cached_predict(enriched,cutoff,target,"category_values")
            cp=cached_predict(control,cutoff,target,"monthly_values")
            y=enriched["labels"][cutoff].reindex(enriched["frames"][cutoff].index)[target].to_numpy()
            error=fp.rmse(y,p);ce=fp.rmse(y,cp)
            row=dict(cutoff=cutoff,target=target,control_rmse=ce,category_rmse=error,
                     recorded_baseline_rmse=np.nan,gain_vs_control=ce-error,gain_vs_recorded=np.nan)
            rows.append(row);print(row,flush=True)
            pd.DataFrame(rows).to_csv(OUT/"category_value_comparison.csv",index=False)
        gg=[r["gain_vs_control"] for r in rows if r["target"]==target]
        if sum(g>0 for g in gg)>=5 and min(gg)>=-0.002 and np.mean(gg)>=0.001:accepted.append(target)
    result=dict(accepted=accepted,screen=screen,features=len(enriched["feature_cols"]),
                rule="Beat the better of recorded baseline and feature-control RMSE on June and September with mean gain >=0.001; then beat the feature control in at least 5 of 6 origins, mean gain >=0.001, no loss >0.002.",
                extra_control="Same monthly low-capacity regressors with original 197 features; not the historical best submission recipe.")
    (OUT/"category_value_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if accepted:
        final=pd.DataFrame(index=enriched["frames"][fp.TEST_CUTOFF].index)
        for target in accepted:final[target]=cached_predict(enriched,fp.TEST_CUTOFF,target,"category_values")
        final.to_csv(OUT/"category_values.csv",index_label="ID",float_format="%.17g")


if __name__=="__main__":main()
