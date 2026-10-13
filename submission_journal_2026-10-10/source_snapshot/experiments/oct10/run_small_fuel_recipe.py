"""Compare a small monthly fuel blend with the reconstructed original quarterly recipe."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
import feature_pipeline as fp
from run_experiment import OUT
from run_fuel_cadence import forecast as small_forecast


def original_recipe(panel,cutoff):
    path=OUT/f"{cutoff}_original_fuel_recipe.npy"
    if path.exists():return np.load(path)
    latest=(pd.Timestamp(cutoff)-pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    cuts=[c for c in fp.TRAIN_CUTOFFS if c<=latest]
    assert len(cuts)>=2
    train=fp.stack_train(panel,cuts);pred=panel["frames"][cutoff]
    inner=cuts[-1]
    assert all(pd.Timestamp(c)+pd.DateOffset(months=3)<=pd.Timestamp(inner) for c in cuts[:-1])
    direct,nd,_,_=fp.lgb_reg_predict(train,pred,"CLV_fuel",(7,21),1400,False)
    h=fp.lgb_hurdle_predict(train,pred,"CLV_fuel",7,1000,False)
    delta_train=train.copy();delta_train.CLV_fuel-=delta_train.fuel_z_q1
    delta,nr,_,_=fp.lgb_reg_predict(delta_train,pred,"CLV_fuel",(7,),1000,False,clip=False)
    residual=np.maximum(pred.fuel_z_q1.to_numpy()+delta,0)
    p=0.4*direct+0.35*h+0.25*residual
    assert np.isfinite(p).all() and (p>=0).all()
    np.save(path,p)
    print(f"original fuel reconstruction {cutoff}: direct={nd} residual={nr} origins={cuts}",flush=True)
    return p


def main():
    small=pickle.loads((OUT/"category_dynamics_panel.pkl").read_bytes())
    original=pickle.loads((OUT/"monthly_panel.pkl").read_bytes())
    recorded=json.loads((OUT/"confirmed_baseline.json").read_text())["historical_report"]
    rows=[];gains=[];accepted=[]
    for cutoff,key in [("2025-06-01","june"),("2025-09-01","september"),
                       ("2025-04-01",None),("2025-05-01",None),("2025-07-01",None),("2025-08-01",None)]:
        p=small_forecast(small,cutoff,False)
        cp=original_recipe(original,cutoff)
        y=small["labels"][cutoff].reindex(small["frames"][cutoff].index).CLV_fuel.to_numpy()
        error=fp.rmse(y,p);ce=fp.rmse(y,cp)
        reference=recorded[key]["rmse_fuel"] if key else np.nan
        if key:gains.append(min(ce,reference)-error)
        row=dict(cutoff=cutoff,target="CLV_fuel",control_rmse=ce,candidate_rmse=error,
                 recorded_baseline_rmse=reference,gain_vs_control=ce-error,gain_vs_recorded=reference-error)
        rows.append(row);print(row,flush=True)
        pd.DataFrame(rows).to_csv(OUT/"small_fuel_value_comparison.csv",index=False)
        if len(gains)==2 and key=="september" and (min(gains)<=0 or np.mean(gains)<0.001):break
    gg=[r["gain_vs_control"] for r in rows]
    if len(rows)==6 and min(gains)>0 and np.mean(gains)>=0.001 and sum(g>0 for g in gg)>=5 and min(gg)>=-0.002 and np.mean(gg)>=0.001:
        accepted=["CLV_fuel"]
    result=dict(accepted=accepted,screen_gains=gains,features=len(small["feature_cols"]),
                rule="Beat the better of reconstructed original fuel recipe and recorded baseline on June/September, mean gain >=0.001; improve at least 5 of 6 origins versus reconstruction, mean gain >=0.001, no loss >0.002.",
                extra_control="Original quarterly 197-feature 47-leaf direct/hurdle/residual recipe with recorded fixed 40/35/25 weights, reconstructed in this environment. Original cached feature pickle requires unavailable pyarrow; exact byte-level training reproduction is not claimed.")
    (OUT/"small_fuel_value_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if accepted:
        p=small_forecast(small,fp.TEST_CUTOFF,False)
        pd.DataFrame({"ID":small["frames"][fp.TEST_CUTOFF].index,"CLV_fuel":p}).to_csv(OUT/"small_fuel_values.csv",index=False,float_format="%.17g")


if __name__=="__main__":main()
