"""Fit the monthly Opportunity model for a user-requested leaderboard experiment."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
import feature_pipeline as fp
from run_monthly import CACHE, OUT, predict


def main():
    panel=pickle.loads(CACHE.read_bytes())
    decision=json.loads((OUT/"monthly_decision.json").read_text())
    assert decision["classes"]==panel["classes"]
    path=OUT/f"{fp.TEST_CUTOFF}_monthly.npy"
    p=np.load(path) if path.exists() else predict(panel,fp.TEST_CUTOFF,threads=2)
    assert p.shape==(len(panel["frames"][fp.TEST_CUTOFF]),len(panel["classes"]))
    assert np.isfinite(p).all() and (p>=0).all()
    np.testing.assert_allclose(p.sum(axis=1),1,rtol=1e-6,atol=1e-6)
    np.save(path,p)
    pred=np.asarray(panel["classes"])[np.argmax(p*np.asarray(decision["multipliers"]),axis=1)]
    pd.DataFrame({"ID":panel["frames"][fp.TEST_CUTOFF].index,"Opportunity":pred}).to_csv(OUT/"monthly_opportunity.csv",index=False)
    print(f"Saved monthly Opportunity predictions for {len(pred)} customers; validation acceptance is unchanged.",flush=True)


if __name__=="__main__":main()
