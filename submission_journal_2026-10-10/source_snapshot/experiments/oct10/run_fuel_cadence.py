"""Test recent fuel purchase cadence and fill size in the low-capacity fuel recipe."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
import feature_pipeline as fp
from run_experiment import OUT
from run_category_values import cached_predict
from run_two_part_values import hurdle
from run_residual_fuel import residual

CACHE=OUT/"fuel_cadence_panel.pkl"


def build():
    panel=pickle.loads((OUT/"category_dynamics_panel.pkl").read_bytes())
    cfg,included,shop,classes=fp.load_config()
    d=fp.lr.load_transactions(fp.DOWNLOADS/"train.csv")
    pack=fp.prepare_tables(d,included,shop,set(cfg["adjustment_items"]))
    events=pack["fuel_tx"].loc[pack["fuel_tx"].qty.gt(0)&pack["fuel_tx"].amt.gt(0)].copy()
    extra_names=None
    for cutoff,frame in panel["frames"].items():
        t=pd.Timestamp(cutoff)
        h=events.loc[events.time.lt(t)].sort_values(["ID","time"]).copy()
        h["gap"]=h.groupby("ID").time.diff().dt.total_seconds()/86400
        recent=h.groupby("ID",sort=False).tail(3)
        gap=recent.groupby("ID").gap.agg(["mean","median","std"]).reindex(frame.index).fillna(0)
        qty=recent.groupby("ID").qty.agg(["mean","std"]).reindex(frame.index).fillna(0)
        last=h.drop_duplicates("ID",keep="last").set_index("ID").reindex(frame.index)
        count=h.groupby("ID").size().reindex(frame.index,fill_value=0).to_numpy()
        recency=(t-last.time).dt.total_seconds().div(86400).fillna(999).to_numpy()
        gm=gap["mean"].to_numpy();qm=qty["mean"].to_numpy()
        last_qty=last.qty.fillna(0).to_numpy()
        horizon=(t+pd.DateOffset(months=3)-t).days
        alive=np.exp(-np.maximum(recency-gm,0)/np.maximum(gm,1))
        projected=np.where(count>=2,horizon/np.maximum(gm,1)*alive*qm,0)
        extra=pd.DataFrame({
            "recent_fuel_gap_mean":gm,"recent_fuel_gap_median":gap["median"].to_numpy(),
            "recent_fuel_gap_cv":gap["std"].to_numpy()/np.maximum(gm,1),
            "recent_fuel_gap_change":np.log1p(gm)-np.log1p(frame.fuel_gap_mean.to_numpy()),
            "recent_fuel_fill_mean":np.log1p(qm),
            "recent_fuel_fill_cv":qty["std"].to_numpy()/np.maximum(qm,1),
            "recent_fuel_last_fill":np.log1p(last_qty),
            "recent_fuel_last_fill_ratio":last_qty/np.maximum(qm,1),
            "recent_fuel_overdue":np.log1p(recency/np.maximum(gm,1)),
            "recent_fuel_projected_z":np.log1p(projected)/panel["s_fuel"],
            "recent_fuel_positive_visits":np.log1p(count),
        },index=frame.index)
        assert np.isfinite(extra.to_numpy()).all()
        if extra_names is None:extra_names=list(extra.columns)
        assert list(extra.columns)==extra_names
        panel["frames"][cutoff]=pd.concat([frame,extra],axis=1)
        print(f"fuel cadence {cutoff}: {len(extra.columns)} features",flush=True)
    panel["feature_cols"]+=extra_names
    CACHE.write_bytes(pickle.dumps(panel))
    return panel


def forecast(panel,cutoff,enriched):
    direct=cached_predict(panel,cutoff,"CLV_fuel","fuel_cadence_direct" if enriched else "category_values")
    hp=OUT/(f"{cutoff}_fuel_cadence_hurdle.npy" if enriched else f"{cutoff}_CLV_fuel_two_part.npy")
    h=np.load(hp) if hp.exists() else hurdle(panel,cutoff,"CLV_fuel",metadata_prefix="fuel_cadence_hurdle" if enriched else "two_part")
    np.save(hp,h)
    rp=OUT/(f"{cutoff}_fuel_cadence_residual.npy" if enriched else f"{cutoff}_fuel_residual.npy")
    r=np.load(rp) if rp.exists() else residual(panel,cutoff,metadata_prefix="fuel_cadence_residual" if enriched else "fuel_residual")
    np.save(rp,r)
    p=0.4*direct+0.35*h+0.25*r
    assert np.isfinite(p).all() and (p>=0).all()
    return p


def main():
    panel=pickle.loads(CACHE.read_bytes()) if CACHE.exists() else build()
    control=pickle.loads((OUT/"category_dynamics_panel.pkl").read_bytes())
    recorded=json.loads((OUT/"confirmed_baseline.json").read_text())["historical_report"]
    rows=[];gains=[];accepted=[]
    def evaluate(cutoff,key=None):
        p=forecast(panel,cutoff,True);cp=forecast(control,cutoff,False)
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).CLV_fuel.to_numpy()
        error=fp.rmse(y,p);ce=fp.rmse(y,cp)
        reference=recorded[key]["rmse_fuel"] if key else np.nan
        row=dict(cutoff=cutoff,target="CLV_fuel",control_rmse=ce,candidate_rmse=error,
                 recorded_baseline_rmse=reference,gain_vs_control=ce-error,gain_vs_recorded=reference-error)
        rows.append(row);print(row,flush=True)
        pd.DataFrame(rows).to_csv(OUT/"fuel_cadence_value_comparison.csv",index=False)
        if key:gains.append(min(ce,reference)-error)
    for cutoff,key in [("2025-06-01","june"),("2025-09-01","september")]:evaluate(cutoff,key)
    if min(gains)>0 and np.mean(gains)>=0.001:
        for cutoff in ["2025-04-01","2025-05-01","2025-07-01","2025-08-01"]:evaluate(cutoff)
        gg=[r["gain_vs_control"] for r in rows]
        if sum(g>0 for g in gg)>=5 and min(gg)>=-0.002 and np.mean(gg)>=0.001:accepted=["CLV_fuel"]
    result=dict(accepted=accepted,screen_gains=gains,features=len(panel["feature_cols"]),
                rule="Beat the better of recorded fuel baseline and control on both June and September, mean gain >=0.001; improve at least 5 of 6 origins against control, mean gain >=0.001, no loss >0.002.",
                extra_control="Same low-capacity 40/35/25 direct/hurdle/residual blend, with the original 305 features.")
    (OUT/"fuel_cadence_value_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if accepted:
        p=forecast(panel,fp.TEST_CUTOFF,True)
        pd.DataFrame({"ID":panel["frames"][fp.TEST_CUTOFF].index,"CLV_fuel":p}).to_csv(OUT/"fuel_cadence_values.csv",index=False,float_format="%.17g")


if __name__=="__main__":main()
