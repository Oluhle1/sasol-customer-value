"""Nested temporal comparison; March tunes decisions, June/September evaluate."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sasol-mpl")
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
import feature_pipeline as fp

OUT = fp.PROJECT / "outputs" / "oct10"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "panel.pkl"
BEST_RECORDED = {"2025-06-01":0.5007931647170651,
                 "2025-09-01":0.5130711211952799}

def score(y, p, classes, multipliers):
    pred = np.asarray(classes)[np.argmax(p * multipliers, axis=1)]
    return float(f1_score(y, pred, average="weighted", zero_division=0)), pred

def tune(y, p, classes):
    # Coordinate search only on March; no June/September outcomes used.
    mult = np.ones(len(classes))
    groups = [(["Existing-category growth: Fuel"], [0.9,1,1.15,1.3]),
              (["Existing-category growth: Food & beverage"], [1,2,3,4.5]),
              (["Existing-category growth: Beverages", "Existing-category growth: Other"], [1,1.3,1.6]),
              ([c for c in classes if c.startswith("New category adoption")], [1,1.5,2,2.5]),
              (["Stable"], [0.85,1,1.15])]
    for names, values in groups:
        idx = [classes.index(c) for c in names]
        choices = []
        for value in values:
            trial = mult.copy(); trial[idx] = value
            choices.append((score(y,p,classes,trial)[0], value))
        mult[idx] = max(choices)[1]
    return mult

def cat_proba(panel, cuts, cutoff):
    train = fp.stack_train(panel, cuts)
    features = panel["feature_cols"]
    classes = panel["classes"]
    mapping = {c:i for i,c in enumerate(classes)}
    y = train.Opportunity.map(mapping).to_numpy()
    latest = cuts[-1]; tr = train.cutoff.ne(latest).to_numpy()
    X = train[features].astype(float).fillna(0)
    params = dict(iterations=650, depth=6, learning_rate=0.06,
                  loss_function="MultiClass", l2_leaf_reg=5,
                  random_seed=20261010, thread_count=1,
                  allow_writing_files=False, verbose=False)
    inner = CatBoostClassifier(**params)
    inner.fit(X.loc[tr], y[tr], sample_weight=train.w.to_numpy()[tr],
              eval_set=(X.loc[~tr],y[~tr]), early_stopping_rounds=60)
    rounds = max(inner.best_iteration_ + 1, 50)
    params["iterations"] = rounds
    full = CatBoostClassifier(**params)
    full.fit(X,y,sample_weight=train.w.to_numpy())
    p = np.zeros((len(panel["frames"][cutoff]),len(classes)))
    p[:,full.classes_.astype(int)] = full.predict_proba(panel["frames"][cutoff][features].astype(float).fillna(0))
    print(f"cat {cutoff} rounds={rounds}",flush=True)
    return p,full

def main():
    if CACHE.exists():
        panel = pickle.loads(CACHE.read_bytes())
    else:
        panel = fp.build_panel()
        CACHE.write_bytes(pickle.dumps(panel))
    # Compare every cached label against the project's canonical tables.
    for cutoff in fp.TRAIN_CUTOFFS:
        canonical = pd.read_csv(fp.PROJECT / "data/processed/phase02_labels" / f"labels_transformed_{cutoff}.csv",dtype={"ID":str}).set_index("ID")
        labels = panel["labels"][cutoff].reindex(canonical.index)
        assert labels.Opportunity.equals(canonical.Opportunity)
        np.testing.assert_allclose(labels[["CLV_fuel","CLV_nonfuel"]],canonical[["CLV_fuel","CLV_nonfuel"]],rtol=1e-12,atol=1e-12)
    classes = panel["classes"]
    fixed = np.array([4.55 if c=="Existing-category growth: Food & beverage" else 1.17 if c=="Existing-category growth: Fuel" else 1.3 if c.startswith("Existing-category growth") else 2.5 if c.startswith("New category adoption") else 1 for c in classes])
    multipliers = {}; rows=[]; gains={"cat":[],"blend":[]}; preds={}
    for cutoff in ["2025-03-01","2025-06-01","2025-09-01"]:
        cuts=[c for c in fp.TRAIN_CUTOFFS if c<cutoff]
        paths=[OUT/f"{cutoff}_{name}.npy" for name in ("lgb","cat")]
        if all(p.exists() for p in paths):
            lp,cp=[np.load(p) for p in paths]
        else:
            lp,_=fp.opp_multiclass(fp.stack_train(panel,cuts),panel["frames"][cutoff],classes,[7,23],False)
            print(f"lgb {cutoff} complete",flush=True)
            cp,_=cat_proba(panel,cuts,cutoff)
            for path,proba in zip(paths,[lp,cp]): np.save(path,proba)
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).Opportunity.to_numpy()
        if cutoff=="2025-03-01":
            multipliers={name:tune(y,p,classes) for name,p in [("cat",cp),("blend",(cp+lp)/2)]}
        base=score(y,lp,classes,fixed)[0]
        for name,p,mult in [("control",lp,fixed),("cat",cp,multipliers["cat"]),("blend",(cp+lp)/2,multipliers["blend"])]:
            f1,pred=score(y,p,classes,mult)
            rows.append(dict(cutoff=cutoff,model=name,weighted_f1=f1,rows=len(y),gain=f1-base))
            pd.DataFrame({"ID":panel["frames"][cutoff].index,"true":y,"pred":pred}).to_csv(OUT/f"{cutoff}_{name}_pred.csv",index=False)
            if cutoff!="2025-03-01" and name!="control": gains[name].append(f1-max(base,BEST_RECORDED[cutoff]))
            print(rows[-1],flush=True)
        pd.DataFrame(rows).to_csv(OUT/"comparison.csv",index=False)
    accepted=[name for name,gg in gains.items() if min(gg)>0 and np.mean(gg)>=0.003]
    chosen=max(accepted,key=lambda name:np.mean(gains[name])) if accepted else None
    result={"chosen":chosen,"gains":gains,"classes":classes,"multipliers":{k:v.tolist() for k,v in multipliers.items()},"best_recorded_f1":BEST_RECORDED,"rule":"Both June and September improve over the larger of control and confirmed baseline's recorded F1; mean gain >= 0.003. March tunes multipliers."}
    (OUT/"decision.json").write_text(json.dumps(result,indent=2))
    if chosen:
        cp,model=cat_proba(panel,fp.TRAIN_CUTOFFS,fp.TEST_CUTOFF)
        model.save_model(str(OUT/"cat_final.cbm"))
        if chosen=="blend":
            lp,_=fp.opp_multiclass(fp.stack_train(panel,fp.TRAIN_CUTOFFS),panel["frames"][fp.TEST_CUTOFF],classes,[7,23],False)
            cp=(cp+lp)/2
        pred=np.asarray(classes)[np.argmax(cp*multipliers[chosen],axis=1)]
        pd.DataFrame({"ID":panel["frames"][fp.TEST_CUTOFF].index,"Opportunity":pred}).to_csv(OUT/"candidate_opportunity.csv",index=False)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
