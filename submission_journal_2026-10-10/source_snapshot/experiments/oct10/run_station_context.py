"""Past-only station trends and leave-one-out prior outcomes for Opportunity."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import feature_pipeline as fp
from run_experiment import OUT,score
from run_monthly import predict

BASE_CACHE=OUT/"category_dynamics_panel.pkl"
CACHE=OUT/"station_context_panel.pkl"


def peer_priors(frame,site,classes):
    old=frame[["prev__"+fp.slug(c) for c in classes]].to_numpy()
    known=old.sum(axis=1)
    groups={"fuel_growth":["Existing-category growth: Fuel"],"inactivity":["Inactivity"],
            "stable":["Stable"],"food_growth":["Existing-category growth: Food & beverage"],
            "beverage_growth":["Existing-category growth: Beverages"],"other_growth":["Existing-category growth: Other"],
            "adoption":[c for c in classes if c.startswith("New category adoption")]}
    data=pd.DataFrame({"site":site.to_numpy(),"known":known},index=frame.index)
    for name,labels in groups.items():data[name]=old[:,[classes.index(c) for c in labels]].sum(axis=1)
    totals=data.groupby("site").transform("sum")
    n=totals.known.to_numpy()-known
    extra={"station_prior_peers":np.log1p(n)}
    for name in groups:
        own=data[name].to_numpy()
        global_prior=np.divide(own.sum()-own,known.sum()-known,
                               out=np.zeros(len(own)),where=known.sum()-known>0)
        extra[f"station_prior_{name}"]=(totals[name].to_numpy()-own+20*global_prior)/(n+20)
    return extra


def build():
    panel=pickle.loads(BASE_CACHE.read_bytes())
    cfg,included,shop,classes=fp.load_config()
    d=fp.lr.load_transactions(fp.DOWNLOADS/"train.csv")
    pack=fp.prepare_tables(d,included,shop,set(cfg["adjustment_items"]))
    location=pack["tx"].set_index(["ID","TransactionId"]).location
    for kind in ["fuel_tx","shop_tx"]:
        events=pack[kind].copy()
        events["location"]=location.reindex(pd.MultiIndex.from_frame(events[["ID","TransactionId"]])).to_numpy()
        assert events.location.notna().all()
        pack[kind]=events
    feature_names=None
    for cutoff,frame in panel["frames"].items():
        t=pd.Timestamp(cutoff)
        history=pack["tx"].loc[pack["tx"].time.lt(t)]
        counts=history.groupby(["ID","location"]).size().rename("n").reset_index()
        primary=counts.sort_values(["ID","n"],ascending=[True,False]).drop_duplicates("ID").set_index("ID").location.reindex(frame.index)
        assert primary.notna().all()
        extra=peer_priors(frame,primary,classes)
        for kind in ["tx","fuel_tx","shop_tx"]:
            events=pack[kind].loc[pack[kind].time.lt(t)]
            values=[]
            for start,end in [(0,30),(30,60),(0,90),(90,180)]:
                window=events.loc[events.time.ge(t-pd.Timedelta(days=end)) & events.time.lt(t-pd.Timedelta(days=start))]
                count=window.groupby("location").size()
                v=primary.map(count).fillna(0).to_numpy(dtype=float)
                values.append(v)
            extra[f"station_{kind}_recent_count"]=np.log1p(values[0])
            extra[f"station_{kind}_month_change"]=np.log1p(values[0])-np.log1p(values[1])
            extra[f"station_{kind}_quarter_change"]=np.log1p(values[2])-np.log1p(values[3])
            extra[f"station_{kind}_recent_share"]=values[0]/np.maximum(values[2],1)
        extra=pd.DataFrame(extra,index=frame.index)
        assert np.isfinite(extra.to_numpy()).all()
        if feature_names is None:feature_names=list(extra.columns)
        assert list(extra.columns)==feature_names
        panel["frames"][cutoff]=pd.concat([frame,extra],axis=1)
        print(f"station context {cutoff}: {len(extra.columns)} features",flush=True)
    panel["feature_cols"]+=feature_names
    CACHE.write_bytes(pickle.dumps(panel))
    return panel


def fitted(panel,cutoff,kind):
    path=OUT/f"{cutoff}_{kind}_models.pkl"
    if path.exists():models=pickle.loads(path.read_bytes())
    else:
        _,models=predict(panel,cutoff,threads=2,return_models=True,weight_decay_months=panel.get("weight_decay_months"),model_params=panel.get("model_params"))
        path.write_bytes(pickle.dumps(models))
    p=np.mean([fp.align_proba(m,panel["frames"][cutoff][panel["feature_cols"]],panel["classes"]) for m in models],axis=0)
    np.save(OUT/f"{cutoff}_{kind}.npy",p)
    return p,models


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=["station","recent","early","compact"],default="station")
    args=parser.parse_args()
    kind={"station":"station_context","recent":"recent_weighting","early":"early_history","compact":"compact_opportunity"}[args.variant]
    if args.variant=="station":panel=pickle.loads(CACHE.read_bytes()) if CACHE.exists() else build()
    elif args.variant=="recent":
        panel=pickle.loads(BASE_CACHE.read_bytes())
        panel["weight_decay_months"]=6.0
    elif args.variant=="compact":
        panel=pickle.loads(BASE_CACHE.read_bytes())
        panel["model_params"]={"num_leaves":31,"min_child_samples":60,"reg_lambda":2.4}
    else:
        from run_early_history import CACHE as EARLY_CACHE,build as build_early
        panel=pickle.loads(EARLY_CACHE.read_bytes()) if EARLY_CACHE.exists() else build_early()
    basepanel=pickle.loads(BASE_CACHE.read_bytes())
    baseline=json.loads((OUT/("confirmed_small_fuel_baseline.json" if args.variant=="compact" else "confirmed_two_part_baseline.json")).read_text())
    decision=json.loads((OUT/"category_dynamics_decision.json").read_text())
    classes=panel["classes"];mult=np.asarray(decision["multipliers"])
    rows=[];models_by_origin={}
    for cutoff in ["2025-06-01","2025-09-01"]:
        p,models=fitted(panel,cutoff,kind)
        models_by_origin[cutoff]=models
        bp=np.load(OUT/f"{cutoff}_category_dynamics.npy")
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).Opportunity.to_numpy()
        b,_=score(y,bp,classes,mult);f1,pred=score(y,p,classes,mult)
        np.testing.assert_allclose(b,baseline["opportunity_validation_f1"][cutoff],rtol=1e-12,atol=1e-12)
        row=dict(cutoff=cutoff,training_origin=cutoff,baseline_f1=b,weighted_f1=f1,gain=f1-b)
        rows.append(row);print(row,flush=True)
        pd.DataFrame(rows).to_csv(OUT/f"{kind}_comparison.csv",index=False)
        pd.DataFrame({"ID":panel["frames"][cutoff].index,"true":y,"pred":pred}).to_csv(OUT/f"{cutoff}_{kind}_pred.csv",index=False)
    main_gains=[r["gain"] for r in rows]
    screen=min(main_gains)>0 and np.mean(main_gains)>=0.001
    accepted=False
    if screen:
        for origin,cutoffs in [("2025-03-01",["2025-04-01","2025-05-01"]),("2025-06-01",["2025-07-01","2025-08-01"])]:
            if origin in models_by_origin:models=models_by_origin[origin]
            else:_,models=fitted(panel,origin,kind)
            bp,bmodels=fitted(basepanel,origin,"category_control")
            if origin=="2025-06-01":np.testing.assert_allclose(bp,np.load(OUT/f"{origin}_category_dynamics.npy"),rtol=1e-10,atol=1e-10)
            for cutoff in cutoffs:
                p=np.mean([fp.align_proba(m,panel["frames"][cutoff][panel["feature_cols"]],classes) for m in models],axis=0)
                bp=np.mean([fp.align_proba(m,basepanel["frames"][cutoff][basepanel["feature_cols"]],classes) for m in bmodels],axis=0)
                y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).Opportunity.to_numpy()
                b,_=score(y,bp,classes,mult);f1,pred=score(y,p,classes,mult)
                row=dict(cutoff=cutoff,training_origin=origin,baseline_f1=b,weighted_f1=f1,gain=f1-b)
                rows.append(row);print(row,flush=True)
                pd.DataFrame(rows).to_csv(OUT/f"{kind}_comparison.csv",index=False)
                pd.DataFrame({"ID":panel["frames"][cutoff].index,"true":y,"pred":pred}).to_csv(OUT/f"{cutoff}_{kind}_pred.csv",index=False)
        gg=[r["gain"] for r in rows]
        accepted=sum(g>0 for g in gg)>=5 and min(gg)>=-0.002 and np.mean(gg)>=0.001
    result=dict(accepted=bool(accepted),main_gains=main_gains,classes=classes,multipliers=mult.tolist(),
                features=len(panel["feature_cols"]),variant=kind,weight_decay_months=panel.get("weight_decay_months"),
                model_params=panel.get("model_params"),
                rule="Improve June and September with mean F1 gain >=0.001, then improve >=5 of 6 evaluated origins, mean gain >=0.001, no loss >0.002.",
                extra_validation="March-fitted models forecast April/May; June-fitted models forecast July/August. Both controls use the same frozen fitting origins. Feature cutoffs use available past data at each forecast date.")
    (OUT/f"{kind}_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if accepted:
        p,_=fitted(panel,fp.TEST_CUTOFF,kind)
        pred=np.asarray(classes)[np.argmax(p*mult,axis=1)]
        pd.DataFrame({"ID":panel["frames"][fp.TEST_CUTOFF].index,"Opportunity":pred}).to_csv(OUT/f"{kind}_opportunity.csv",index=False)


if __name__=="__main__":main()
