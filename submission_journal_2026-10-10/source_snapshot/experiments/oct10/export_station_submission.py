"""Export station-context Opportunity only after its six-date validation gate."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/sasol-mpl")
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import feature_pipeline as fp
from run_experiment import OUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",choices=["station","recent","early","compact"],default="station")
    args=parser.parse_args()
    kind={"station":"station_context","recent":"recent_weighting","early":"early_history","compact":"compact_opportunity"}[args.candidate]
    decision=json.loads((OUT/f"{kind}_decision.json").read_text())
    if not decision["accepted"]:raise SystemExit("Station-context candidate did not pass validation.")
    baseline=json.loads((OUT/("confirmed_small_fuel_baseline.json" if args.candidate=="compact" else "confirmed_two_part_baseline.json")).read_text())
    report=pd.read_csv(OUT/f"{kind}_comparison.csv").set_index("cutoff")
    assert set(report.index)=={"2025-04-01","2025-05-01","2025-06-01","2025-07-01","2025-08-01","2025-09-01"} and report.index.is_unique
    known=[]
    for cutoff in ["2025-06-01","2025-09-01"]:
        assert report.loc[cutoff,"training_origin"]==cutoff
        np.testing.assert_allclose(report.loc[cutoff,"baseline_f1"],baseline["opportunity_validation_f1"][cutoff],rtol=1e-12,atol=1e-12)
        known.append(report.loc[cutoff,"weighted_f1"]-baseline["opportunity_validation_f1"][cutoff])
    assert min(known)>0 and np.mean(known)>=0.001
    for origin,cutoffs in [("2025-03-01",["2025-04-01","2025-05-01"]),("2025-06-01",["2025-07-01","2025-08-01"])]:
        for cutoff in cutoffs:assert report.loc[cutoff,"training_origin"]==origin
    gg=report.weighted_f1-report.baseline_f1
    assert (gg>0).sum()>=5 and gg.min()>=-0.002 and gg.mean()>=0.001
    basepath=OUT/("baseline_0309164264.csv" if args.candidate=="compact" else "baseline_0308109087.csv")
    assert hashlib.sha256(basepath.read_bytes()).hexdigest()==baseline["sha256"]
    sample=pd.read_csv(fp.PROJECT/"data/raw/SampleSubmission.csv",dtype={"ID":str})
    base=pd.read_csv(basepath,dtype={"ID":str})
    candidate=pd.read_csv(OUT/f"{kind}_opportunity.csv",dtype={"ID":str})
    for f in [base,candidate]:assert f.ID.is_unique and set(f.ID)==set(sample.ID)
    base=base.set_index("ID").reindex(sample.ID).reset_index()
    old=base.Opportunity.copy()
    base.Opportunity=candidate.set_index("ID").reindex(sample.ID).Opportunity.to_numpy()
    cfg=json.loads((fp.PROJECT/"configs/label_config.json").read_text())
    assert set(base.Opportunity)<=set(cfg["opportunity_labels"])
    assert list(base.columns)==list(sample.columns) and not base.isna().any().any()
    assert np.isfinite(base[["CLV_fuel","CLV_nonfuel"]].to_numpy()).all()
    path=fp.PROJECT/f"submissions/sasol_oct10_{kind}.csv"
    base.to_csv(path,index=False,float_format="%.17g")
    written=pd.read_csv(path,dtype={"ID":str})
    original=pd.read_csv(basepath,dtype={"ID":str}).set_index("ID").reindex(written.ID)
    assert written.ID.equals(sample.ID)
    np.testing.assert_allclose(written[["CLV_fuel","CLV_nonfuel"]],original[["CLV_fuel","CLV_nonfuel"]],rtol=1e-14,atol=1e-14)
    audit=dict(baseline_public_score=baseline["public_score"],baseline_sha256=baseline["sha256"],
               output=str(path),output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               rows=len(base),changed_opportunity_labels=int((old!=base.Opportunity).sum()),
               regression_columns_preserved=True,features=decision["features"],variant=kind,
               validation=report.reset_index().to_dict("records"),extra_validation=decision["extra_validation"],
               validation_gate_passed=True,public_score=None)
    path.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
