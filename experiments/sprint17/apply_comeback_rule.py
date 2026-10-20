"""Sprint 17 E2. Score the pre-registered inactivity-to-fuel override.

The grid and the gates are read from outputs/sprint17/proposal_comeback_rule.json.
June and September are scored for one selected grid point, and only after the
pool screen passes.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint17"))
from fit_weekly_rank4 import DEST, OCT, load_p17, load_production_p4, sha256  # noqa: E402
from evaluate_saved import CLASSES, rank4_combine, weighted_f1  # noqa: E402

PROPOSAL = DEST / "proposal_comeback_rule.json"
FUEL = "Existing-category growth: Fuel"
FUEL_I = CLASSES.index(FUEL)
INACT = "Inactivity"


def load_proposal():
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    if proposal["id"] != "sprint17_comeback_rule":
        raise RuntimeError("unexpected proposal id")
    grid = proposal["grid"]
    if grid["fuel_recency_at_most"] != [30, 45, 60, 75, 90, 120]:
        raise RuntimeError("recency grid was edited")
    if grid["fuel_bgnbd_z_at_least"] != [0.0, 0.1, 0.5, 0.85, 1.08]:
        raise RuntimeError("cadence grid was edited")
    if grid["require_second_is_fuel"] != [False, True]:
        raise RuntimeError("second-class grid was edited")
    return proposal


def origin_cache(panel, cutoff, vR):
    frame = panel["frames"][cutoff]
    ids = frame.index.astype(str).to_numpy()
    y = panel["labels"][cutoff].reindex(frame.index).Opportunity
    if y.isna().any():
        raise RuntimeError(f"{cutoff} is missing labels")
    adjusted = rank4_combine(load_p17(cutoff, ids), load_production_p4(cutoff, ids), "full") * vR
    order = np.argsort(-adjusted, axis=1)
    pred_idx = order[:, 0].astype(int)
    second = np.asarray(CLASSES)[order[:, 1]]
    return {
        "y": y.to_numpy(),
        "pred_idx": pred_idx,
        "second_is_fuel": second == FUEL,
        "recency": frame["fuel_recency"].to_numpy(dtype=float),
        "bgnbd": frame["fuel_bgnbd_z"].to_numpy(dtype=float),
        "baseline_f1": weighted_f1(pred_idx, y.to_numpy()),
    }


def apply_rule(cache, recency_at_most, bgnbd_at_least, require_second):
    pred = cache["pred_idx"].copy()
    mask = (
        (np.asarray(CLASSES)[pred] == INACT)
        & (cache["recency"] <= recency_at_most)
        & (cache["bgnbd"] >= bgnbd_at_least)
    )
    if require_second:
        mask = mask & cache["second_is_fuel"]
    pred[mask] = FUEL_I
    return pred, int(mask.sum())


def gains_for(caches, cutoffs, recency_at_most, bgnbd_at_least, require_second):
    gains = {}
    flips = {}
    for cutoff in cutoffs:
        pred, n_flip = apply_rule(caches[cutoff], recency_at_most, bgnbd_at_least, require_second)
        f1 = weighted_f1(pred, caches[cutoff]["y"])
        gains[cutoff] = f1 - caches[cutoff]["baseline_f1"]
        flips[cutoff] = n_flip
    return gains, flips


def screen_passes(gains, proposal):
    values = list(gains.values())
    rule = proposal["screen"]
    return float(np.mean(values)) >= float(rule["mean_gain_at_least"]) and min(values) >= float(rule["min_origin_gain_at_least"])


def eval_passes(gains, proposal):
    values = list(gains.values())
    rule = proposal["export_gate"]
    return min(values) >= float(rule["each_fold_gain_at_least"]) and float(np.mean(values)) >= float(rule["mean_fold_gain_at_least"])


def export_december(panel, vR, chosen):
    source = ROOT / "submissions" / "sasol_sprint16_r4full_vR.csv"
    frame = panel["frames"]["2025-12-01"]
    ids = frame.index.astype(str).to_numpy()
    adjusted = rank4_combine(load_p17("2025-12-01", ids), load_production_p4("2025-12-01", ids), "full") * vR
    order = np.argsort(-adjusted, axis=1)
    cache = {
        "pred_idx": order[:, 0].astype(int),
        "second_is_fuel": np.asarray(CLASSES)[order[:, 1]] == FUEL,
        "recency": frame["fuel_recency"].to_numpy(dtype=float),
        "bgnbd": frame["fuel_bgnbd_z"].to_numpy(dtype=float),
    }
    pred_idx, n_flip = apply_rule(
        cache, chosen["fuel_recency_at_most"], chosen["fuel_bgnbd_z_at_least"], chosen["require_second_is_fuel"],
    )
    labels = np.asarray(CLASSES)[pred_idx]
    submitted = pd.read_csv(source, dtype={"ID": str})
    if list(submitted.columns) != ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]:
        raise RuntimeError("CLV source columns are not the submission schema")
    if submitted["ID"].duplicated().any() or len(submitted) != 5488:
        raise RuntimeError("CLV source is not the 5488-row submission")
    mapped = pd.Series(labels, index=ids).reindex(submitted["ID"])
    if mapped.isna().any() or not set(mapped) <= set(CLASSES):
        raise RuntimeError("December override does not cover the submission")
    out = submitted.copy()
    changed = int((out["Opportunity"].to_numpy() != mapped.to_numpy()).sum())
    out["Opportunity"] = mapped.to_numpy()
    if not np.array_equal(out[["CLV_fuel", "CLV_nonfuel"]].to_numpy(), submitted[["CLV_fuel", "CLV_nonfuel"]].to_numpy()):
        raise RuntimeError("export changed a CLV column")
    values = out[["CLV_fuel", "CLV_nonfuel"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("CLV columns are not finite and nonnegative")
    path = ROOT / "submissions" / "sasol_sprint17_comeback.csv"
    out.to_csv(path, index=False)
    audit = {
        "file": "submissions/sasol_sprint17_comeback.csv",
        "sha256": sha256(path),
        "source_sha256": sha256(source),
        "rows": 5488,
        "opportunity_changes_vs_source": changed,
        "december_rule_flips": n_flip,
        "rule": chosen,
    }
    (DEST / "comeback_export_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"EXPORT {path.name} changes {changed}", flush=True)
    return audit


def main():
    proposal = load_proposal()
    vR = np.asarray(json.loads((ROOT / "outputs" / "sprint16" / "pool_r4full_vector.json").read_text())["vR"], dtype=float)
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    caches = {cutoff: origin_cache(panel, cutoff, vR) for cutoff in proposal["pool"]}
    print("proposal", sha256(PROPOSAL)[:12], {c: round(caches[c]["baseline_f1"], 6) for c in proposal["pool"]}, flush=True)
    ranked = []
    for recency in proposal["grid"]["fuel_recency_at_most"]:
        for bgnbd in proposal["grid"]["fuel_bgnbd_z_at_least"]:
            for require_second in proposal["grid"]["require_second_is_fuel"]:
                gains, flips = gains_for(caches, proposal["pool"], recency, bgnbd, require_second)
                ranked.append({
                    "fuel_recency_at_most": recency,
                    "fuel_bgnbd_z_at_least": bgnbd,
                    "require_second_is_fuel": require_second,
                    "mean_gain": float(np.mean(list(gains.values()))),
                    "min_gain": float(min(gains.values())),
                    "gains": gains,
                    "flips": flips,
                    "screen_passed": screen_passes(gains, proposal),
                })
    ranked.sort(key=lambda row: (row["min_gain"], row["mean_gain"]), reverse=True)
    chosen = ranked[0]
    passed = bool(chosen["screen_passed"])
    decision = {
        "proposal_sha256": sha256(PROPOSAL),
        "screen_passed": passed,
        "selected_by": proposal["selection"],
        "selected": {k: chosen[k] for k in (
            "fuel_recency_at_most", "fuel_bgnbd_z_at_least", "require_second_is_fuel",
            "mean_gain", "min_gain", "gains", "flips", "screen_passed",
        )},
        "top5": [{k: row[k] for k in (
            "fuel_recency_at_most", "fuel_bgnbd_z_at_least", "require_second_is_fuel",
            "mean_gain", "min_gain", "flips",
        )} for row in ranked[:5]],
        "eval": None,
        "exported": False,
    }
    print(
        f"{'SCREEN_PASS' if passed else 'SCREEN_FAIL'} "
        f"T {chosen['fuel_recency_at_most']} B {chosen['fuel_bgnbd_z_at_least']} "
        f"second {chosen['require_second_is_fuel']} "
        f"mean {chosen['mean_gain']:+.6f} min {chosen['min_gain']:+.6f}",
        flush=True,
    )
    if passed:
        eval_caches = {cutoff: origin_cache(panel, cutoff, vR) for cutoff in proposal["eval"]}
        gains, flips = gains_for(
            eval_caches, proposal["eval"],
            chosen["fuel_recency_at_most"], chosen["fuel_bgnbd_z_at_least"], chosen["require_second_is_fuel"],
        )
        accepted = eval_passes(gains, proposal)
        decision["eval_passed"] = accepted
        decision["eval"] = {"gains": gains, "flips": flips, "mean_gain": float(np.mean(list(gains.values())))}
        print("EVAL", {k: round(v, 6) for k, v in gains.items()}, "pass" if accepted else "fail", flush=True)
        if accepted:
            decision["export"] = export_december(panel, vR, chosen)
            decision["exported"] = True
    (DEST / "comeback_rule_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
