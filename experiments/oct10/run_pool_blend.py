"""Fixed 50/50 quarterly/monthly probability blend; March alone tunes decisions."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import feature_pipeline as fp
from run_experiment import OUT,BEST_RECORDED,score,tune
from run_monthly import predict as predict_monthly

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--screen-only",action="store_true")
    args=parser.parse_args()
    panel=pickle.loads((OUT/"monthly_panel.pkl").read_bytes())
    quarterly=pickle.loads((OUT/"panel.pkl").read_bytes())
    classes=panel["classes"]; rows=[]; gains=[]
    fixed=np.array([4.55 if c=="Existing-category growth: Food & beverage" else 1.17 if c=="Existing-category growth: Fuel" else 1.3 if c.startswith("Existing-category growth") else 2.5 if c.startswith("New category adoption") else 1 for c in classes])
    cutoffs=["2025-03-01","2025-06-01"] if args.screen_only else ["2025-03-01","2025-06-01","2025-09-01"]
    for cutoff in cutoffs:
        path=OUT/f"{cutoff}_lgb.npy"
        if not path.exists():
            cuts=[c for c in fp.TRAIN_CUTOFFS if c<cutoff]
            lp,_=fp.opp_multiclass(fp.stack_train(quarterly,cuts),quarterly["frames"][cutoff],classes,[7,23],False)
            np.save(path,lp)
        else:lp=np.load(path)
        mp=np.load(OUT/f"{cutoff}_monthly.npy")
        p=(lp+mp)/2
        y=panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).Opportunity.to_numpy()
        if cutoff=="2025-03-01":mult=tune(y,p,classes)
        f1,pred=score(y,p,classes,mult)
        base=score(y,lp,classes,fixed)[0]
        rows.append(dict(cutoff=cutoff,quarterly_f1=base,pool_blend_f1=f1,gain=f1-base))
        pd.DataFrame(rows).to_csv(OUT/"pool_blend_comparison.csv",index=False)
        pd.DataFrame({"ID":panel["frames"][cutoff].index,"true":y,"pred":pred}).to_csv(OUT/f"{cutoff}_pool_blend_pred.csv",index=False)
        if cutoff!="2025-03-01":gains.append(f1-max(base,BEST_RECORDED[cutoff]))
        print(rows[-1],flush=True)
    if args.screen_only:return
    accepted=min(gains)>0 and np.mean(gains)>=0.003
    result=dict(accepted=bool(accepted),gains=gains,classes=classes,multipliers=mult.tolist(),blend=0.5)
    (OUT/"pool_blend_decision.json").write_text(json.dumps(result,indent=2))
    if accepted:
        lp_path=OUT/"2025-12-01_lgb.npy"
        mp_path=OUT/"2025-12-01_monthly.npy"
        if lp_path.exists():lp=np.load(lp_path)
        else:
            lp,_=fp.opp_multiclass(fp.stack_train(quarterly,fp.TRAIN_CUTOFFS),quarterly["frames"][fp.TEST_CUTOFF],classes,[7,23],False)
            np.save(lp_path,lp)
        if mp_path.exists():mp=np.load(mp_path)
        else:
            mp=predict_monthly(panel,fp.TEST_CUTOFF)
            np.save(mp_path,mp)
        pred=np.asarray(classes)[np.argmax((lp+mp)/2*mult,axis=1)]
        pd.DataFrame({"ID":panel["frames"][fp.TEST_CUTOFF].index,"Opportunity":pred}).to_csv(OUT/"pool_blend_opportunity.csv",index=False)
    print(result,flush=True)

if __name__=="__main__":main()
