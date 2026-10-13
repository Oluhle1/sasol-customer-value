"""Refine monthly class decisions on March only; compare against the new best."""
import json
import numpy as np
import pandas as pd
from run_experiment import OUT,score


def refine(y,p,classes,initial):
    # Small, fixed search; regularization discourages extreme changes to rare classes.
    initial=np.asarray(initial,dtype=float)
    mult=initial.copy()
    groups=[[classes.index(c)] for c in classes if not c.startswith("New category adoption")]
    groups.append([i for i,c in enumerate(classes) if c.startswith("New category adoption")])
    def objective(weights):
        penalty=0.002*np.mean(np.log(weights/initial)**2)
        return score(y,p,classes,weights)[0]-penalty
    for _ in range(2):
        for indices in groups:
            best=objective(mult); chosen=mult.copy()
            for factor in [0.7,0.85,1.0,1.15,1.3,1.6,2.0]:
                trial=mult.copy()
                trial[indices]=np.clip(initial[indices]*factor,0.5,5.0)
                value=objective(trial)
                if value>best+1e-8:best=value;chosen=trial
            mult=chosen
    return mult


def main():
    original=json.loads((OUT/"monthly_decision.json").read_text())
    classes=original["classes"]; initial=np.asarray(original["multipliers"])
    rows=[]
    for cutoff in ["2025-03-01","2025-06-01","2025-09-01"]:
        p=np.load(OUT/f"{cutoff}_monthly.npy")
        labels=pd.read_csv(OUT/f"{cutoff}_monthly_pred.csv",dtype={"ID":str})
        y=labels.true.to_numpy()
        if cutoff=="2025-03-01":mult=refine(y,p,classes,initial)
        base,_=score(y,p,classes,initial); f1,pred=score(y,p,classes,mult)
        row=dict(cutoff=cutoff,monthly_baseline_f1=base,refined_f1=f1,gain=f1-base)
        rows.append(row);print(row,flush=True)
        labels["pred"]=pred
        labels.to_csv(OUT/f"{cutoff}_decision_refined_pred.csv",index=False)
    gains=[r["gain"] for r in rows[1:]]
    result=dict(accepted=bool(min(gains)>0),classes=classes,multipliers=mult.tolist(),
                gains=gains,tuning_cutoff="2025-03-01",rule="Both June and September improve on the confirmed monthly baseline; March alone tunes weights.")
    pd.DataFrame(rows).to_csv(OUT/"decision_refinement_comparison.csv",index=False)
    (OUT/"decision_refinement_decision.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    if result["accepted"]:
        p=np.load(OUT/"2025-12-01_monthly.npy")
        final=pd.read_csv(OUT/"monthly_opportunity.csv",dtype={"ID":str})
        final.Opportunity=np.asarray(classes)[np.argmax(p*mult,axis=1)]
        final.to_csv(OUT/"decision_refined_opportunity.csv",index=False)


if __name__=="__main__":main()
