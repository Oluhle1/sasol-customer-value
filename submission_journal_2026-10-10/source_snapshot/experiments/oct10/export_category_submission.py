"""Export a validated Opportunity candidate against its confirmed public baseline."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from run_experiment import OUT
import feature_pipeline as fp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",choices=["category","group"],default="category")
    parser.add_argument("--experimental",action="store_true",help="Export a group experiment when both periods improve but the mean-gain threshold fails.")
    args=parser.parse_args()
    if args.experimental and args.candidate!="group":
        raise SystemExit("Experimental export is supported only for the group specialist.")
    name="category_dynamics" if args.candidate=="category" else "group_specialist"
    decision=json.loads((OUT/f"{name}_decision.json").read_text())
    chosen=decision.get("chosen") if args.candidate=="category" else "group_specialist" if decision["accepted"] or args.experimental else None
    if not chosen:
        raise SystemExit("No candidate improved both validation periods.")
    report=pd.read_csv(OUT/f"{name}_comparison.csv")
    rows=(report[report.model.eq(chosen)] if args.candidate=="category" else report).set_index("cutoff")
    baseline_name="confirmed_monthly_baseline.json" if args.candidate=="category" else "confirmed_category_baseline.json"
    baseline=json.loads((OUT/baseline_name).read_text())
    gains=[float(rows.loc[c,"weighted_f1"])-baseline["opportunity_validation_f1"][c]
           for c in ["2025-06-01","2025-09-01"]]
    assert min(gains)>0
    passed_gate=np.mean(gains)>=0.001
    assert passed_gate or args.experimental
    basepath=OUT/("baseline_0305007187.csv" if args.candidate=="category" else "baseline_0307534678.csv")
    assert hashlib.sha256(basepath.read_bytes()).hexdigest()==baseline["sha256"]
    sample=pd.read_csv(fp.PROJECT/"data/raw/SampleSubmission.csv",dtype={"ID":str})
    base=pd.read_csv(basepath,dtype={"ID":str})
    candidate=pd.read_csv(OUT/f"{name}_opportunity.csv",dtype={"ID":str})
    for frame in [base,candidate]:
        assert frame.ID.is_unique and set(frame.ID)==set(sample.ID)
    base=base.set_index("ID").reindex(sample.ID).reset_index()
    old=base.Opportunity.copy()
    base.Opportunity=candidate.set_index("ID").reindex(sample.ID).Opportunity.to_numpy()
    assert list(base.columns)==list(sample.columns) and not base.isna().any().any()
    cfg=json.loads((fp.PROJECT/"configs/label_config.json").read_text())
    assert set(base.Opportunity)<=set(cfg["opportunity_labels"])
    assert np.isfinite(base[["CLV_fuel","CLV_nonfuel"]].to_numpy()).all()
    assert (base[["CLV_fuel","CLV_nonfuel"]]>=0).all().all()
    path=fp.PROJECT/f"submissions/sasol_oct10_{name}.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    base.to_csv(path,index=False,float_format="%.17g")
    written=pd.read_csv(path,dtype={"ID":str})
    assert written.ID.equals(sample.ID)
    original=pd.read_csv(basepath,dtype={"ID":str}).set_index("ID").reindex(written.ID)
    np.testing.assert_allclose(written[["CLV_fuel","CLV_nonfuel"]],original[["CLV_fuel","CLV_nonfuel"]],rtol=1e-14,atol=1e-14)
    audit=dict(baseline_public_score=baseline["public_score"],baseline_sha256=baseline["sha256"],
               output=str(path),output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               rows=len(base),changed_opportunity_labels=int((old!=base.Opportunity).sum()),
               chosen_model=chosen,features=decision.get("features",305),
               june_september_f1_gains=gains,regression_columns_preserved=True,public_score=None,
               validation_gate_passed=bool(passed_gate),experimental_export=args.experimental)
    path.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
