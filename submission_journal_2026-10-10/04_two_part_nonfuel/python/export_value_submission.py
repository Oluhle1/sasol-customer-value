"""Apply six-origin-validated CLV changes to the best public submission."""
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
    parser.add_argument("--candidate",choices=["category","two-part","residual","fuel-recipe","fuel-cadence","small-fuel","shop-residual","cat-shop","cat-fuel"],default="category")
    args=parser.parse_args()
    tag={"category":"category","two-part":"two_part","residual":"residual","fuel-recipe":"fuel_recipe","fuel-cadence":"fuel_cadence","small-fuel":"small_fuel","shop-residual":"shop_residual","cat-shop":"cat_shop","cat-fuel":"cat_fuel"}[args.candidate]
    prediction_column="category_rmse" if args.candidate=="category" else "candidate_rmse"
    decision=json.loads((OUT/f"{tag}_value_decision.json").read_text())
    targets=decision["accepted"]
    if not targets:raise SystemExit("No CLV target passed the six-origin validation gate.")
    fuel_candidate=args.candidate in ["residual","fuel-recipe","fuel-cadence","small-fuel"]
    baseline_name="confirmed_two_part_baseline.json" if fuel_candidate else "confirmed_category_baseline.json"
    baseline=json.loads((OUT/baseline_name).read_text())
    basepath=OUT/("baseline_0308109087.csv" if fuel_candidate else "baseline_0307534678.csv")
    current_nonfuel=args.candidate in ["shop-residual","cat-shop"]
    current_fuel=args.candidate=="cat-fuel"
    if current_nonfuel or current_fuel:
        baseline=json.loads((OUT/"confirmed_small_fuel_baseline.json").read_text())
        basepath=OUT/"baseline_0309164264.csv"
        assert targets==(["CLV_fuel"] if current_fuel else ["CLV_nonfuel"])
    if fuel_candidate:assert targets==["CLV_fuel"]
    assert hashlib.sha256(basepath.read_bytes()).hexdigest()==baseline["sha256"]
    report=pd.read_csv(OUT/f"{tag}_value_comparison.csv")
    recorded=json.loads((OUT/"confirmed_baseline.json").read_text())["historical_report"]
    for target in targets:
        assert target in ["CLV_fuel","CLV_nonfuel"]
        rows=report[report.target.eq(target)].set_index("cutoff")
        assert set(rows.index)=={"2025-04-01","2025-05-01","2025-06-01","2025-07-01","2025-08-01","2025-09-01"} and rows.index.is_unique
        gg=rows.control_rmse-rows[prediction_column]
        assert (gg>0).sum()>=5 and gg.min()>=-0.002 and gg.mean()>=0.001
        known=[]
        for cutoff,key in [("2025-06-01","june"),("2025-09-01","september")]:
            reference=recorded[key]["rmse_fuel" if target=="CLV_fuel" else "rmse_nonfuel"]
            if current_nonfuel:
                reference=baseline["nonfuel_validation_rmse"][cutoff]
            if current_fuel:
                reference=baseline["fuel_validation_rmse"][cutoff]
            known.append(min(reference,rows.loc[cutoff,"control_rmse"])-rows.loc[cutoff,prediction_column])
        assert min(known)>0 and np.mean(known)>=0.001
    sample=pd.read_csv(fp.PROJECT/"data/raw/SampleSubmission.csv",dtype={"ID":str})
    original=pd.read_csv(basepath,dtype={"ID":str})
    new=pd.read_csv(OUT/f"{tag}_values.csv",dtype={"ID":str})
    for frame in [original,new]:assert frame.ID.is_unique and set(frame.ID)==set(sample.ID)
    base=original.set_index("ID").reindex(sample.ID).reset_index()
    new=new.set_index("ID").reindex(sample.ID)
    for target in targets:base[target]=new[target].to_numpy()
    assert list(base.columns)==list(sample.columns) and not base.isna().any().any()
    assert np.isfinite(base[["CLV_fuel","CLV_nonfuel"]].to_numpy()).all()
    assert (base[["CLV_fuel","CLV_nonfuel"]]>=0).all().all()
    path=fp.PROJECT/f"submissions/sasol_oct10_{tag}_values.csv"
    base.to_csv(path,index=False,float_format="%.17g")
    written=pd.read_csv(path,dtype={"ID":str})
    original=original.set_index("ID").reindex(written.ID)
    assert written.ID.equals(sample.ID) and np.array_equal(written.Opportunity.to_numpy(),original.Opportunity.to_numpy())
    for target in {"CLV_fuel","CLV_nonfuel"}-set(targets):
        np.testing.assert_allclose(written[target],original[target],rtol=1e-14,atol=1e-14)
    selected=report[report.target.isin(targets)].astype(object)
    evidence=selected.where(pd.notna(selected),None).to_dict("records")
    audit=dict(baseline_public_score=baseline["public_score"],baseline_sha256=baseline["sha256"],
               output=str(path),output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               rows=len(base),changed_targets=targets,model=tag,opportunity_preserved=True,
               validation=evidence,extra_control=decision["extra_control"],public_score=None)
    path.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()
