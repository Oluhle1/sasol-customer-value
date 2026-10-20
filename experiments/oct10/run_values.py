"""Evaluate fixed regression recipes on June and September, then export gains."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
import feature_pipeline as fp

OUT=fp.PROJECT / "outputs/oct10"

def main():
    panel=pickle.loads((OUT/"panel.pkl").read_bytes())
    canonical=pd.read_csv(fp.PROJECT/"data/processed/phase04/supervised_historical.csv",dtype={"ID":str})
    manifest=pd.read_csv(fp.PROJECT/"data/processed/phase04/model_feature_manifest.csv")
    features=manifest.loc[manifest.selected_for_baseline.eq(1),"feature"].tolist()
    rows=[]; gains={t:[] for t in ["CLV_fuel","CLV_nonfuel"]}
    recorded=json.loads((OUT/"confirmed_baseline.json").read_text())["historical_report"]
    for cutoff in ["2025-06-01","2025-09-01"]:
        cuts=[c for c in fp.TRAIN_CUTOFFS if c<cutoff]
        tr=fp.stack_train(panel,cuts); pred=panel["frames"][cutoff]
        for target,alpha in [("CLV_fuel",0.3),("CLV_nonfuel",0.5)]:
            cache=OUT/f"{cutoff}_{target}_values.npy"
            if cache.exists():
                base,candidate=np.load(cache)
            else:
                # Control uses the frozen structural recipe; rounds selected on
                # the preceding quarter, never the outer evaluation quarter.
                earlier=canonical[canonical.cutoff.lt(cuts[-1])]
                inner=canonical[canonical.cutoff.eq(cuts[-1])]
                train=canonical[canonical.cutoff.lt(cutoff)]
                valid=canonical[canonical.cutoff.eq(cutoff)].set_index("ID").reindex(pred.index)
                params=dict(n_estimators=1200,learning_rate=0.03,num_leaves=15,
                    min_child_samples=100 if target=="CLV_fuel" else 50,
                    colsample_bytree=0.9,subsample=0.9,subsample_freq=1,
                    reg_lambda=1,random_state=20260926,n_jobs=1,verbosity=-1)
                inner_model=LGBMRegressor(**params)
                inner_model.fit(earlier[features],earlier[target],eval_set=[(inner[features],inner[target])],callbacks=[early_stopping(100,verbose=False)])
                params["n_estimators"]=max(inner_model.best_iteration_,40)
                control=LGBMRegressor(**params).fit(train[features],train[target])
                base=np.maximum(control.predict(valid[features]),0)
                direct,_,_,_=fp.lgb_reg_predict(tr,pred,target,[7,21],1200,False)
                hurdle=fp.lgb_hurdle_predict(tr,pred,target,7,1000,False)
                candidate=(1-alpha)*direct+alpha*hurdle
                np.save(cache,np.vstack([base,candidate]))
            y=panel["labels"][cutoff].reindex(pred.index)[target].to_numpy()
            rb=fp.rmse(y,base); rc=fp.rmse(y,candidate)
            key="june" if cutoff=="2025-06-01" else "september"
            known=recorded[key]["rmse_fuel" if target=="CLV_fuel" else "rmse_nonfuel"]
            gains[target].append(min(rb,known)-rc)
            rows.append(dict(cutoff=cutoff,target=target,control_rmse=rb,confirmed_baseline_rmse=known,candidate_rmse=rc,gain=min(rb,known)-rc))
            print(rows[-1],flush=True)
            pd.DataFrame(rows).to_csv(OUT/"value_comparison.csv",index=False)
    accepted=[t for t,gg in gains.items() if min(gg)>0 and np.mean(gg)>=0.001]
    result={"accepted":accepted,"gains":gains,"rule":"Fixed 30% fuel / 50% nonfuel hurdle; both folds improve over the lower of control and confirmed baseline's recorded RMSE, and mean gain >=0.001."}
    (OUT/"value_decision.json").write_text(json.dumps(result,indent=2))
    final=pd.DataFrame(index=panel["frames"][fp.TEST_CUTOFF].index)
    for target in accepted:
        tr=fp.stack_train(panel,fp.TRAIN_CUTOFFS); pred=panel["frames"][fp.TEST_CUTOFF]
        direct,_,_,_=fp.lgb_reg_predict(tr,pred,target,[7,21],1200,False)
        hurdle=fp.lgb_hurdle_predict(tr,pred,target,7,1000,False)
        alpha=0.3 if target=="CLV_fuel" else 0.5
        final[target]=(1-alpha)*direct+alpha*hurdle
    if accepted: final.to_csv(OUT/"candidate_values.csv",index_label="ID")
    print(result,flush=True)

if __name__=="__main__": main()
