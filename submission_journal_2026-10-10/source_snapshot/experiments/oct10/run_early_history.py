"""Add eligible June/July/August 2024 snapshots without altering existing features."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import json
import pickle
import numpy as np
import pandas as pd
import feature_pipeline as fp
from run_experiment import OUT
from run_category_dynamics import build as enrich

CACHE=OUT/"early_history_panel.pkl"


def build():
    panel=pickle.loads((OUT/"monthly_panel.pkl").read_bytes())
    cfg,included,shop,classes=fp.load_config()
    d=fp.lr.load_transactions(fp.DOWNLOADS/"train.csv")
    pack=fp.prepare_tables(d,included,shop,set(cfg["adjustment_items"]))
    audits=[]
    for cutoff in ["2024-06-01","2024-07-01","2024-08-01"]:
        previous=fp.prev_quarter(cutoff)
        labels=fp.lr.transform(fp.lr.raw_labels(d,cutoff,cfg),cfg).set_index("ID")
        prior=fp.lr.transform(fp.lr.raw_labels(d,previous,cfg),cfg).set_index("ID")
        labels.index=labels.index.astype(str);prior.index=prior.index.astype(str)
        frame,_=fp.build_features(pack,cutoff,labels.index,prior.Opportunity,included,panel["s_fuel"],panel["s_shop"],classes)
        assert list(frame.columns)==panel["feature_cols"]
        assert frame.index.is_unique and set(frame.index)==set(labels.index)
        assert np.isfinite(frame.to_numpy()).all()
        panel["frames"][cutoff]=frame;panel["labels"][cutoff]=labels
        audits.append(dict(cutoff=cutoff,customers=len(frame),prior_label_origin=previous,
                           prior_outcomes_end=cutoff,target_end=(pd.Timestamp(cutoff)+pd.DateOffset(months=3)).strftime("%Y-%m-%d")))
        print(f"early history {cutoff}: {len(frame)} eligible customers",flush=True)
    panel["monthly_cutoffs"]=sorted(set(panel["monthly_cutoffs"]+[r["cutoff"] for r in audits]))
    basepath=OUT/"early_history_base_panel.pkl"
    basepath.write_bytes(pickle.dumps(panel))
    enhanced=enrich(base_cache=basepath,cache=CACHE)
    original=pickle.loads((OUT/"category_dynamics_panel.pkl").read_bytes())
    assert enhanced["feature_cols"]==original["feature_cols"]
    for cutoff,frame in original["frames"].items():
        pd.testing.assert_frame_equal(frame,enhanced["frames"][cutoff])
    final_rows=sum(len(enhanced["frames"][c]) for c in enhanced["monthly_cutoffs"])
    meta=dict(added_origins=audits,origins=len(enhanced["monthly_cutoffs"]),final_training_rows=final_rows,
              features=len(enhanced["feature_cols"]),existing_feature_tables_unchanged=True)
    (OUT/"early_history_build.json").write_text(json.dumps(meta,indent=2))
    print(meta,flush=True)
    return enhanced


if __name__=="__main__":build()
