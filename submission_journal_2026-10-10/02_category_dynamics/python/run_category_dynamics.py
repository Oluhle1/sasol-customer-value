"""Add past-only category recency and momentum to the confirmed monthly model."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
import feature_pipeline as fp
from run_monthly import CACHE as BASE_CACHE,predict
from run_experiment import OUT,score

CACHE=OUT/"category_dynamics_panel.pkl"


def build(base_cache=BASE_CACHE,cache=CACHE):
    panel=pickle.loads(base_cache.read_bytes())
    cfg,included,shop,classes=fp.load_config()
    d=fp.lr.load_transactions(fp.DOWNLOADS/"train.csv")
    pack=fp.prepare_tables(d,included,shop,set(cfg["adjustment_items"]))
    qual=d.loc[~d.ItemName.isin(cfg["adjustment_items"]) & d.ItemCategoryLevel2.isin(included)
               & d._quantity.gt(0) & d._amount.gt(0),["ID","ItemCategoryLevel2","_time"]]
    features=None
    for cutoff,frame in panel["frames"].items():
        t=pd.Timestamp(cutoff);ids=frame.index
        previous=[t.year*12+t.month-k for k in [4,5,6]]
        recent=[t.year*12+t.month-k for k in [1,2,3]]
        q2=fp.matrix_from(pack["cat_spend"],previous,ids,included,"spend","sum")
        q1=fp.matrix_from(pack["cat_spend"],recent,ids,included,"spend","sum")
        counts=fp.matrix_from(pack["cat_tx"],recent,ids,included,"ntx","sum")
        h=qual.loc[qual._time.lt(t)]
        last=h.groupby(["ID","ItemCategoryLevel2"])._time.max().unstack().reindex(index=ids,columns=included)
        extra={}
        for category in included:
            key=fp.slug(category)
            extra[f"dyn_{key}_recency"]=(t-last[category]).dt.total_seconds().div(86400).fillna(999).to_numpy()
            earlier=np.log1p(q2[category].clip(lower=0).to_numpy())
            now=np.log1p(q1[category].clip(lower=0).to_numpy())
            extra[f"dyn_{key}_q2"]=earlier
            extra[f"dyn_{key}_change"]=now-earlier
            extra[f"dyn_{key}_purchases"]=np.log1p(counts[category].clip(lower=0).to_numpy())
        for kind in ["tx","fuel_tx","shop_tx"]:
            events=pack[kind].loc[pack[kind].time.lt(t)]
            recent_counts=[]
            for start,end in [(0,14),(14,28),(28,56)]:
                sub=events.loc[events.time.ge(t-pd.Timedelta(days=end)) & events.time.lt(t-pd.Timedelta(days=start))]
                count=sub.groupby("ID").size().reindex(ids,fill_value=0).to_numpy()
                extra[f"dyn_{kind}_{start}_{end}_n"]=np.log1p(count)
                recent_counts.append(count)
            extra[f"dyn_{kind}_weekly_change"]=np.log1p(recent_counts[0])-np.log1p(recent_counts[1])
        extra=pd.DataFrame(extra,index=ids)
        assert np.isfinite(extra.to_numpy()).all()
        panel["frames"][cutoff]=pd.concat([frame,extra],axis=1)
        if features is None:features=list(extra.columns)
        assert list(extra.columns)==features
        print(f"category dynamics {cutoff}: {len(extra.columns)} new features",flush=True)
    panel["feature_cols"]+=features
    cache.write_bytes(pickle.dumps(panel))
    return panel


def main():
    panel=pickle.loads(CACHE.read_bytes()) if CACHE.exists() else build()
    baseline=json.loads((OUT/"monthly_decision.json").read_text())
    classes=panel["classes"];mult=np.asarray(baseline["multipliers"])
    rows=[];gains={"category":[],"blend":[]}
    for cutoff in ["2025-06-01","2025-09-01"]:
        path=OUT/f"{cutoff}_category_dynamics.npy"
        p=np.load(path) if path.exists() else predict(panel,cutoff,threads=2)
        np.save(path,p)
        bp=np.load(OUT/f"{cutoff}_monthly.npy")
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).Opportunity.to_numpy()
        base,_=score(y,bp,classes,mult)
        for name,proba in [("category",p),("blend",(p+bp)/2)]:
            f1,pred=score(y,proba,classes,mult)
            row=dict(cutoff=cutoff,model=name,monthly_baseline_f1=base,weighted_f1=f1,gain=f1-base)
            rows.append(row);gains[name].append(f1-base);print(row,flush=True)
            pd.DataFrame({"ID":panel["frames"][cutoff].index,"true":y,"pred":pred}).to_csv(OUT/f"{cutoff}_category_{name}_pred.csv",index=False)
        pd.DataFrame(rows).to_csv(OUT/"category_dynamics_comparison.csv",index=False)
    eligible=[name for name,gg in gains.items() if min(gg)>0 and np.mean(gg)>=0.001]
    chosen=max(eligible,key=lambda name:np.mean(gains[name])) if eligible else None
    result=dict(chosen=chosen,gains=gains,classes=classes,multipliers=mult.tolist(),
                features=len(panel["feature_cols"]),decision_weights="Unchanged from confirmed monthly model",
                rule="Improve both June and September over monthly baseline; mean F1 gain >=0.001.")
    (OUT/"category_dynamics_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if chosen:
        cutoff=fp.TEST_CUTOFF;path=OUT/f"{cutoff}_category_dynamics.npy"
        p=np.load(path) if path.exists() else predict(panel,cutoff,threads=2)
        np.save(path,p)
        if chosen=="blend":p=(p+np.load(OUT/f"{cutoff}_monthly.npy"))/2
        pred=np.asarray(classes)[np.argmax(p*mult,axis=1)]
        pd.DataFrame({"ID":panel["frames"][cutoff].index,"Opportunity":pred}).to_csv(OUT/"category_dynamics_opportunity.csv",index=False)


if __name__=="__main__":main()
