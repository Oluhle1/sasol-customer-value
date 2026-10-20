"""Choose how strongly to apply the four-class rule that scored 0.309411451.

The mix is p = (1 - alpha) * p17 + alpha * p_full. Alpha is chosen from
0.25, 0.50, 0.75 and 1.00 by the mean weighted F1 on April and May only.
A tie keeps the larger alpha. Alpha 1 is the current public file.

The chosen alpha is written down before June through September are scored.
An alpha below 1 is exported only when, against the alpha-1 labels, June
and September each improve, the June-September mean gain is at least
0.001, at least 5 of 6 April-September origins improve, and no origin
loses more than 0.002. Both CLV columns stay on the current public file.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sasol-mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "top10_round"))
import run_rank4 as r4  # noqa: E402
import run_rank5 as r5  # noqa: E402

DEST = ROOT / "outputs" / "top10_round" / "rank4_blend"
DEST.mkdir(parents=True, exist_ok=True)
PROGRESS = DEST / "progress.log"
ALPHAS = (0.25, 0.50, 0.75, 1.00)
SELECT = ["2025-04-01", "2025-05-01"]
LATER = ["2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
EVAL = SELECT + LATER


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with PROGRESS.open("a") as handle:
        handle.write(line + "\n")


def full_proba(panel, cutoff, classes):
    frame = panel["frames"][cutoff]
    p17 = r4.load_proba(cutoff, frame.index)
    p4 = r5.load_cached(r4.DEST / f"{cutoff}_rank4.npz", frame.index, 4)
    return p17, r4.combine(p17, p4, classes, "full")


def f1_at(panel, cutoff, classes, multipliers, alpha, cached):
    frame = panel["frames"][cutoff]
    y = panel["labels"][cutoff].reindex(frame.index).Opportunity.to_numpy()
    p17, full = cached[cutoff]
    proba = (1.0 - alpha) * p17 + alpha * full
    pred = r4.labels_from(proba, classes, multipliers)
    return r4.weighted_f1(y, pred), pred


def main():
    PROGRESS.write_text("")
    proposal = {
        "parent_public_score": 0.309411451,
        "alphas": list(ALPHAS),
        "selection": "Mean weighted F1 on April and May only. A tie keeps the larger alpha.",
        "gate": "If alpha is 1, keep the current file. Otherwise June and September must beat alpha 1, the June-September mean gain must be at least 0.001, at least 5 of 6 origins must improve, and no origin may lose more than 0.002.",
    }
    (DEST / "proposal.json").write_text(json.dumps(proposal, indent=2))
    classes, multipliers = r4.load_decision()
    log("loading panel")
    panel = __import__("pickle").loads((r4.OCT / "category_dynamics_panel.pkl").read_bytes())
    cached = {cutoff: full_proba(panel, cutoff, classes) for cutoff in SELECT}
    grid = []
    for alpha in ALPHAS:
        scores = [f1_at(panel, cutoff, classes, multipliers, alpha, cached)[0] for cutoff in SELECT]
        row = {"alpha": alpha, "april_f1": scores[0], "may_f1": scores[1], "mean_f1": float(np.mean(scores))}
        grid.append(row)
        log(json.dumps(row))
    chosen = max(grid, key=lambda row: (row["mean_f1"], row["alpha"]))
    (DEST / "selection.json").write_text(json.dumps({"grid": grid, "chosen_alpha": chosen["alpha"]}, indent=2))
    log(f"chosen alpha {chosen['alpha']} from April and May only")
    for cutoff in LATER:
        cached[cutoff] = full_proba(panel, cutoff, classes)
    rows = []
    for cutoff in EVAL:
        reference, _ = f1_at(panel, cutoff, classes, multipliers, 1.0, cached)
        candidate, pred = f1_at(panel, cutoff, classes, multipliers, chosen["alpha"], cached)
        base, _ = f1_at(panel, cutoff, classes, multipliers, 0.0, cached)
        rows.append({
            "cutoff": cutoff,
            "baseline_f1": base,
            "rank4_f1": reference,
            "candidate_f1": candidate,
            "gain": candidate - reference,
            "changes": int((pred != f1_at(panel, cutoff, classes, multipliers, 1.0, cached)[1]).sum()),
        })
        log(json.dumps(rows[-1]))
    gains = {row["cutoff"]: row["gain"] for row in rows}
    later = [gains[cutoff] for cutoff in LATER]
    accepted = bool(
        chosen["alpha"] < 1.0
        and min(gains["2025-06-01"], gains["2025-09-01"]) > 0
        and float(np.mean(later)) >= 0.001
        and sum(gain > 0 for gain in gains.values()) >= 5
        and min(gains.values()) >= -0.002
    )
    summary = {
        "grid": grid,
        "chosen_alpha": chosen["alpha"],
        "rows": rows,
        "accepted": accepted,
        "mean_later": float(np.mean(later)),
    }
    if accepted:
        cutoff = "2025-12-01"
        frame = panel["frames"][cutoff]
        p17, full = full_proba(panel, cutoff, classes)
        pred = r4.labels_from((1.0 - chosen["alpha"]) * p17 + chosen["alpha"] * full, classes, multipliers)
        parent = pd.read_csv(ROOT / "submissions" / "sasol_top10_rank4.csv", dtype={"ID": str})
        original = pd.read_csv(ROOT / "submissions" / "sasol_oct10_small_fuel_values.csv", dtype={"ID": str})
        out = parent.copy()
        out["Opportunity"] = pd.Series(pred, index=frame.index.astype(str)).loc[out.ID].to_numpy()
        if out.Opportunity.isna().any() or not set(out.Opportunity).issubset(classes):
            raise RuntimeError("invalid opportunity export")
        if np.max(np.abs(out.CLV_fuel - original.CLV_fuel)) > 1e-12 or np.max(np.abs(out.CLV_nonfuel - original.CLV_nonfuel)) > 1e-12:
            raise RuntimeError("CLV columns changed")
        output = ROOT / "submissions" / "sasol_top10_rank4_blend.csv"
        out.to_csv(output, index=False)
        summary["file"] = output.name
        summary["changes_vs_rank4"] = int((out.Opportunity.to_numpy() != parent.Opportunity.to_numpy()).sum())
        log(f"wrote {output.name} changes_vs_rank4 {summary['changes_vs_rank4']}")
    else:
        log("blend was not exported")
    (DEST / "decision.json").write_text(json.dumps(summary, indent=2))
    log("DONE: " + json.dumps({"alpha": chosen["alpha"], "accepted": accepted}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
