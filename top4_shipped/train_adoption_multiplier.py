"""June-gated adoption multiplier on the shipped 0.30225 rule.

Pre-registered grid is {1.5, 2.0, 2.5}. 2.5 is the control.
Every other multiplier stays frozen. September is scored once, after the choice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_growth_override as base  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
GRID = (1.5, 2.0, 2.5)
CONTROL = 2.5
PUBLISHED_SEPTEMBER_F1 = 0.5118
MIN_MEAN_GAIN = 0.005
BEST_FILE = base.BEST_FILE


def apply_rule(proba, classes, adoption: float) -> np.ndarray:
    adjusted = proba.copy()
    for i, name in enumerate(classes):
        label = str(name)
        if label.startswith("New category adoption"):
            factor = adoption
        else:
            factor = base.MULTIPLIER[label]
        adjusted[:, i] *= factor
    return np.asarray(classes, dtype=object)[adjusted.argmax(axis=1)]


def score_grid(proba, classes, y_true) -> dict[str, float]:
    scores = {}
    for adoption in GRID:
        pred = apply_rule(proba, classes, adoption)
        scores[f"{adoption:.1f}"] = base.weighted_f1(y_true, pred)
    return scores


def choose(scores: dict[str, float]) -> str:
    """Highest June weighted F1. Any tie, including a tie with 2.5, stays at 2.5."""
    best = max(scores.values())
    winners = [key for key, value in scores.items() if value == best]
    if len(winners) == 1:
        return winners[0]
    return f"{CONTROL:.1f}"


def fit_fold(customers, features, train_cutoffs, valid_cutoff):
    Xtr, ytr, wtr, Xva, yva = base.slice_fold(customers, features, train_cutoffs, valid_cutoff)
    print(f"  training seeds 7 and 23 for {valid_cutoff} rows={len(ytr):,}", flush=True)
    models = [base.fit_classifier(Xtr, ytr, wtr, Xva, yva, seed) for seed in (7, 23)]
    proba, classes = base.average_proba(models, Xva)
    return models, proba, classes, yva


def main():
    if base.ADOPTION_MULTIPLIER != 2.5:
        raise SystemExit("Control adoption multiplier in train_growth_override.py is not 2.5")
    expected = {
        "Inactivity": 1.0,
        "Stable": 1.0,
        "Existing-category growth: Fuel": 1.17,
        "Existing-category growth: Beverages": 1.3,
        "Existing-category growth: Food & beverage": 4.55,
        "Existing-category growth: Other": 1.3,
    }
    if base.MULTIPLIER != expected:
        raise SystemExit(f"Frozen multipliers changed: {base.MULTIPLIER}")

    ns = base.load_base()
    customers, test, features = base.build_customers(ns)
    print(f"rows={len(customers):,} features={len(features)} test={len(test):,}", flush=True)
    cutoffs = list(base.SNAPSHOT_WEIGHT)

    print("June fold only", flush=True)
    june_models, june_proba, june_classes, june_y = fit_fold(customers, features, cutoffs[:3], base.JUNE)
    june_scores = score_grid(june_proba, june_classes, june_y)
    frozen = choose(june_scores)
    june_choice = {
        "june_weighted_f1": june_scores,
        "frozen_adoption_multiplier": float(frozen),
        "control_adoption_multiplier": CONTROL,
        "september_not_used": True,
    }
    (OUT / "adoption_multiplier_june.json").write_text(json.dumps(june_choice, indent=2))
    print(json.dumps(june_choice, indent=2), flush=True)
    print(f"FROZEN {frozen}", flush=True)

    print("September fold, frozen multiplier only", flush=True)
    september_models, september_proba, september_classes, september_y = fit_fold(
        customers, features, cutoffs[:4], base.SEPTEMBER
    )
    september_control = base.weighted_f1(
        september_y, apply_rule(september_proba, september_classes, CONTROL)
    )
    september_frozen = base.weighted_f1(
        september_y, apply_rule(september_proba, september_classes, float(frozen))
    )

    june_control = june_scores[f"{CONTROL:.1f}"]
    june_chosen = june_scores[frozen]
    june_gain = june_chosen - june_control
    september_gain = september_frozen - september_control
    mean_gain = (june_gain + september_gain) / 2
    accepted = bool(
        june_chosen > june_control
        and september_frozen > PUBLISHED_SEPTEMBER_F1
        and mean_gain >= MIN_MEAN_GAIN
    )
    decision = {
        "rule_script": str(HERE / "train_growth_override.py"),
        "rule_file": str(BEST_FILE),
        "june_weighted_f1": june_scores,
        "frozen_adoption_multiplier": float(frozen),
        "september_weighted_f1": september_frozen,
        "september_control_weighted_f1": september_control,
        "published_september_f1": PUBLISHED_SEPTEMBER_F1,
        "june_gain": june_gain,
        "september_gain": september_gain,
        "mean_gain": mean_gain,
        "accepted": accepted,
    }
    (OUT / "adoption_multiplier_decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2), flush=True)
    if not accepted:
        print("Gate did not pass. No submission file was written.", flush=True)
        return

    print("refitting for the December book", flush=True)
    rounds = [int(model.best_iteration_ or 200) for model in september_models]
    X_all = customers[features].to_numpy()
    y_all = customers.Opportunity.to_numpy()
    w_all = customers.cutoff.map(base.SNAPSHOT_WEIGHT).to_numpy(dtype=float)
    final_models = []
    for seed, n_trees in zip((7, 23), rounds):
        model = LGBMClassifier(
            n_estimators=n_trees,
            learning_rate=0.04,
            num_leaves=47,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.70,
            reg_lambda=1.2,
            random_state=seed,
            verbosity=-1,
            n_jobs=4,
        )
        model.fit(X_all, y_all, sample_weight=w_all)
        final_models.append(model)
    proba, classes = base.average_proba(final_models, test[features].to_numpy())
    pred = apply_rule(proba, classes, float(frozen))
    source = pd.read_csv(BEST_FILE, dtype=str)
    out = source[["ID", "CLV_fuel", "CLV_nonfuel"]].copy()
    mapped = pd.DataFrame({"ID": test.ID.astype(str), "Opportunity": pred})
    out = out.merge(mapped, on="ID", how="left", validate="one_to_one")
    out = out[["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]]
    if out.Opportunity.isna().any() or not out.ID.equals(source.ID):
        raise AssertionError("Test predictions do not cover the 0.30225 file")
    if not out.CLV_fuel.equals(source.CLV_fuel) or not out.CLV_nonfuel.equals(source.CLV_nonfuel):
        raise AssertionError("Fuel or non-fuel values changed")
    changed = int((out.Opportunity.to_numpy() != source.Opportunity.to_numpy()).sum())
    destination = OUT / f"sasol_adoption_mult_{frozen}_submission.csv"
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite {destination}")
    out.to_csv(destination, index=False)
    note = OUT / "adoption_multiplier_note.md"
    note.write_text(
        "\n".join(
            [
                "# Adoption multiplier",
                "",
                f"Chosen on the June 2025 fold only: {frozen}.",
                "",
                f"June weighted F1: 1.5 = {june_scores['1.5']:.6f}, "
                f"2.0 = {june_scores['2.0']:.6f}, 2.5 = {june_scores['2.5']:.6f}.",
                f"September weighted F1 with {frozen}: {september_frozen:.6f}.",
                f"September control (2.5) on the same probabilities: {september_control:.6f}.",
                f"Opportunity labels changed versus the 0.302250514 file: {changed} of {len(out)}.",
                "Fuel and non-fuel values were copied from that file.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    decision["submission"] = str(destination)
    decision["opportunity_labels_changed"] = changed
    (OUT / "adoption_multiplier_decision.json").write_text(json.dumps(decision, indent=2))
    print(f"Wrote {destination} changed={changed}", flush=True)


if __name__ == "__main__":
    main()