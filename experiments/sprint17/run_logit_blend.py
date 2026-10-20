"""Sprint 17 E4 — log-space product blend, decision layer only, no fitting.

Pre-registration: experiments/sprint17/PROPOSAL_E4.md and
outputs/sprint17/proposal_logit_blend.json. Gates are read from the JSON and
never relaxed. Reuses the sprint16 evaluation harness (`evaluate_saved`) so the
baseline product and decode are byte-identical to the shipped r4full+vR.
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))

from evaluate_saved import (  # noqa: E402
    CLASSES, OCT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1,
)

OUT17 = ROOT / "outputs" / "sprint17"
PROPOSAL = OUT17 / "proposal_logit_blend.json"
PROPOSAL_SHA = hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()
POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
TEST = "2025-12-01"
EPS = 1e-12


def load_panel():
    return pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())


def geom(p, q, g):
    lp = (1.0 - g) * np.log(np.clip(p, EPS, None)) + g * np.log(np.clip(q, EPS, None))
    lp = lp - lp.max(axis=1, keepdims=True)
    w = np.exp(lp)
    return w / w.sum(axis=1, keepdims=True)


def arith(p, q, g):
    w = (1.0 - g) * p + g * q
    return w / np.clip(w.sum(axis=1, keepdims=True), EPS, None)


def main():
    started = time.time()
    proposal = json.loads(PROPOSAL.read_text())
    screen = proposal["screen_gate"]
    folds_gate = proposal["eval_gate"]
    vR = np.array(json.loads((OUT17.parent / "sprint16" / "pool_r4full_vector.json").read_text())["vR"], dtype=float)
    panel = load_panel()

    data = {}
    for cutoff in POOL + EVAL:
        ids, y = load_labels(cutoff)
        n = len(ids)
        p17 = load_proba("category_dynamics", cutoff, n)
        r4 = load_rank4(cutoff, n)
        if p17 is None or r4 is None:
            raise RuntimeError(f"missing cached arrays for {cutoff}")
        r4full = rank4_combine(p17, r4, "full")
        data[cutoff] = {"y": y, "catdyn": p17, "r4full": r4full}

    baseline = {}
    for cutoff, d in data.items():
        baseline[cutoff] = weighted_f1(np.argmax(d["r4full"] * vR, axis=1), d["y"])
    for cutoff, expected in proposal["baseline"]["pool_origin_f1"].items():
        if abs(baseline[cutoff] - expected) > 1e-9:
            raise RuntimeError(f"baseline mismatch at {cutoff}: {baseline[cutoff]} vs {expected}")

    results = {}
    for family, fn in (("arith", arith), ("geom", geom)):
        for g in proposal["grid"]["g_mix"]:
            scores = {c: weighted_f1(np.argmax(fn(data[c]["catdyn"], data[c]["r4full"], g) * vR, axis=1), data[c]["y"]) for c in POOL}
            gains = {c: scores[c] - baseline[c] for c in POOL}
            results[f"{family}_g{g}"] = {
                "family": family, "g": g,
                "scores": scores, "gains": gains,
                "mean_gain": float(np.mean(list(gains.values()))),
                "min_gain": float(np.min(list(gains.values()))),
            }

    best_key = max(
        results,
        key=lambda k: (
            results[k]["mean_gain"], results[k]["min_gain"],
            1 if results[k]["family"] == "geom" else 0, -abs(results[k]["g"] - 1.0),
        ),
    )
    best = results[best_key]
    passed_screen = best["mean_gain"] >= screen["mean_gain_min"] and best["min_gain"] >= screen["per_origin_min"]

    decision = {
        "proposal_sha256": PROPOSAL_SHA,
        "baseline": baseline,
        "grid_results": results,
        "selected": best_key,
        "selected_detail": best,
        "passed_screen": bool(passed_screen),
        "endpoints": {
            c: {
                "catdyn": weighted_f1(np.argmax(data[c]["catdyn"] * vR, axis=1), data[c]["y"]),
                "r4full": baseline[c],
            } for c in POOL + EVAL
        },
    }
    print(json.dumps({"selected": best_key, "mean": best["mean_gain"], "min": best["min_gain"], "passed_screen": bool(passed_screen)}, indent=1), flush=True)
    for c in POOL:
        print(f"  pool {c}: baseline {baseline[c]:.6f} selected {best['scores'][c]:.6f} gain {best['gains'][c]:+.6f}", flush=True)

    fold_gains = None
    if passed_screen:
        family_fn = geom if best["family"] == "geom" else arith
        g = best["g"]
        fold_scores = {}
        fold_gains = {}
        for c in EVAL:
            prod = family_fn(data[c]["catdyn"], data[c]["r4full"], g)
            f1 = weighted_f1(np.argmax(prod * vR, axis=1), data[c]["y"])
            fold_scores[c] = f1
            fold_gains[c] = f1 - baseline[c]
            print(f"  eval {c}: baseline {baseline[c]:.6f} selected {f1:.6f} gain {fold_gains[c]:+.6f}", flush=True)
        decision["fold_scores"] = fold_scores
        decision["fold_gains"] = fold_gains
        passed_folds = all(v >= folds_gate["per_origin_min"] for v in fold_gains.values()) and \
            float(np.mean(list(fold_gains.values()))) >= folds_gate["mean_gain_min"]
        decision["passed_folds"] = bool(passed_folds)

        early_path = OCT / "2025-06-01_early_history.npy"
        dec_diag = {}
        for c in EVAL:
            early = np.load(OCT / f"{c}_early_history.npy")
            e1 = 0.4 * data[c]["catdyn"] + 0.4 * early + 0.2 * data[c]["r4full"]
            e1 = e1 / e1.sum(axis=1, keepdims=True)
            e1_f1 = weighted_f1(np.argmax(e1 * vR, axis=1), data[c]["y"])
            mix_f1 = weighted_f1(np.argmax(family_fn(data[c]["catdyn"], e1, g) * vR, axis=1), data[c]["y"])
            dec_diag[c] = {"e1_blend": e1_f1, "e1_blend_mixed": mix_f1}
        decision["diagnostic_e1_blend"] = dec_diag

        if passed_folds:
            frame = panel["frames"][TEST]
            ids, _ = load_labels(TEST)
            p17 = load_proba("category_dynamics", TEST, len(ids))
            r4 = load_rank4(TEST, len(ids))
            r4full = rank4_combine(p17, r4, "full")
            prod = family_fn(p17, r4full, g)
            pred = np.asarray(CLASSES)[np.argmax(prod * vR, axis=1)]
            baseline_csv = pd.read_csv(ROOT / "submissions" / "sasol_sprint16_r4full_vR.csv", dtype={"ID": str})
            if list(baseline_csv["ID"].astype(str)) != list(frame.index.astype(str)):
                raise RuntimeError("December frame order does not match the source submission")
            out = baseline_csv.copy()
            if not np.array_equal(out["CLV_fuel"].to_numpy(), baseline_csv["CLV_fuel"].to_numpy()):
                raise RuntimeError("fuel column changed during export")
            out["Opportunity"] = pred
            path = ROOT / "submissions" / "sasol_sprint17_logitblend.csv"
            out.to_csv(path, index=False)
            changes = int((out["Opportunity"].to_numpy() != baseline_csv["Opportunity"].to_numpy()).sum())
            (OUT17 / "logitblend_audit.json").write_text(json.dumps({
                "source": "submissions/sasol_sprint16_r4full_vR.csv",
                "file": "submissions/sasol_sprint17_logitblend.csv",
                "selected": best_key,
                "changes": changes,
                "rows": int(len(out)),
            }, indent=2))
            decision["exported"] = {"file": path.name, "changes_vs_best": changes}
            print(f"  exported {path.name} with {changes} label changes", flush=True)
    decision["seconds"] = round(time.time() - started, 1)
    (OUT17 / "logitblend_decision.json").write_text(json.dumps(decision, indent=2))
    print("wrote", OUT17 / "logitblend_decision.json", flush=True)


if __name__ == "__main__":
    main()
