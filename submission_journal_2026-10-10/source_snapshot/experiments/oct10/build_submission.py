"""Apply a validated Opportunity candidate to an explicitly identified baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",required=True,type=Path)
    parser.add_argument("--candidate",choices=["classifier","monthly","pool"],default="classifier")
    parser.add_argument("--experimental",action="store_true",help="Export a monthly candidate for an explicitly requested leaderboard experiment even if the validation gate fails.")
    parser.add_argument("--output",type=Path,default=ROOT/"submissions/sasol_oct10_opportunity.csv")
    args=parser.parse_args()
    outdir=ROOT/"outputs/oct10"
    decision_name={"classifier":"decision.json","monthly":"monthly_decision.json","pool":"pool_blend_decision.json"}[args.candidate]
    decision=json.loads((outdir/decision_name).read_text())
    if args.experimental and args.candidate!="monthly":
        raise SystemExit("Experimental export is currently supported only for the monthly candidate.")
    chosen=args.candidate if args.candidate!="classifier" and (decision["accepted"] or args.experimental) else decision.get("chosen")
    if not chosen:
        raise SystemExit("No Opportunity candidate passed validation; preserve the baseline.")
    recorded=json.loads((outdir/"confirmed_baseline.json").read_text())
    comparison_name="monthly_comparison.csv" if chosen=="monthly" else "pool_blend_comparison.csv" if chosen=="pool" else "comparison.csv"
    comparisons=pd.read_csv(outdir/comparison_name)
    gains=[]; conservative_gains=[]
    for cutoff,key in [("2025-06-01","june"),("2025-09-01","september")]:
        subset=comparisons[comparisons.cutoff.eq(cutoff)]
        f1=float(subset.monthly_f1.iloc[0]) if chosen=="monthly" else float(subset.pool_blend_f1.iloc[0]) if chosen=="pool" else float(subset[subset.model.eq(chosen)].weighted_f1.iloc[0])
        gains.append(f1-recorded["historical_report"][key]["weighted_f1"])
        control_score=recorded["historical_report"][key]["weighted_f1"]
        control_file=outdir/"comparison.csv"
        if control_file.exists():
            control=pd.read_csv(control_file)
            row=control[control.cutoff.eq(cutoff)&control.model.eq("control")]
            if len(row):control_score=max(control_score,float(row.weighted_f1.iloc[0]))
        conservative_gains.append(f1-control_score)
    passed_gate=min(conservative_gains)>0 and np.mean(conservative_gains)>=0.003
    if not passed_gate and not args.experimental:
        raise SystemExit(f"Candidate does not improve the confirmed baseline on both folds: {gains}")
    base=pd.read_csv(args.baseline,dtype={"ID":str})
    candidate_name="monthly_opportunity.csv" if chosen=="monthly" else "pool_blend_opportunity.csv" if chosen=="pool" else "candidate_opportunity.csv"
    candidate=pd.read_csv(outdir/candidate_name,dtype={"ID":str})
    sample=pd.read_csv(ROOT/"data/raw/SampleSubmission.csv",dtype={"ID":str})
    for frame in [base,candidate]:
        assert frame.ID.is_unique and set(frame.ID)==set(sample.ID)
    base=base.set_index("ID").reindex(sample.ID).reset_index()
    old=base.Opportunity.copy()
    base.Opportunity=candidate.set_index("ID").reindex(sample.ID).Opportunity.to_numpy()
    cfg=json.loads((ROOT/"configs/label_config.json").read_text())
    assert set(base.Opportunity)<=set(cfg["opportunity_labels"])
    assert np.isfinite(base[["CLV_fuel","CLV_nonfuel"]].to_numpy()).all()
    assert (base[["CLV_fuel","CLV_nonfuel"]]>=0).all().all()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    base.to_csv(args.output,index=False,float_format="%.17g")
    check=pd.read_csv(args.output,dtype={"ID":str}).set_index("ID")
    original=pd.read_csv(args.baseline,dtype={"ID":str}).set_index("ID").reindex(check.index)
    np.testing.assert_allclose(check[["CLV_fuel","CLV_nonfuel"]],original[["CLV_fuel","CLV_nonfuel"]],rtol=1e-14,atol=1e-14)
    audit={"baseline":str(args.baseline.resolve()),"baseline_sha256":hashlib.sha256(args.baseline.read_bytes()).hexdigest(),"output":str(args.output.resolve()),"rows":len(base),"changed_opportunity_labels":int((old!=base.Opportunity).sum()),"chosen_model":chosen,"regression_columns_preserved":True,"june_september_f1_gains_over_recorded_best":gains,"gains_over_larger_available_control_or_recorded_best":conservative_gains}
    audit.update(experimental_export=args.experimental,validation_gate_passed=bool(passed_gate))
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=="__main__": main()
