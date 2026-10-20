"""
Opportunity model for the top-4 submission.

The rule in README.md is fixed before the outer-fold scores are read.
Fuel and non-fuel predictions are not trained here.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
MODEL_DIR = HERE / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CUTOFFS = [
    "2024-09-01",
    "2024-12-01",
    "2025-03-01",
    "2025-06-01",
    "2025-09-01",
]
FOLDS = {
    "F2": {"train": CUTOFFS[:2], "valid": CUTOFFS[2]},
    "F3": {"train": CUTOFFS[:3], "valid": CUTOFFS[3]},
    "F4": {"train": CUTOFFS[:4], "valid": CUTOFFS[4]},
}

# Official class ids from data/processed/phase04/opportunity_class_map.csv
FUEL_ID = 2
INACTIVITY_ID = 4
STABLE_ID = 16
OTHER_GROWTH_IDS = (0, 1, 3)
ADOPTION_IDS = tuple(range(5, 16))
HEAD_OF = {
    **{INACTIVITY_ID: 0, STABLE_ID: 1, FUEL_ID: 2},
    **{cid: 3 for cid in OTHER_GROWTH_IDS},
    **{cid: 4 for cid in ADOPTION_IDS},
}


def weighted_f1(y_true, y_pred):
    return float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )


def f1_eval(num_classes):
    def evaluate(predictions, dataset):
        labels = dataset.get_label().astype("int64")
        probabilities = np.asarray(predictions)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(num_classes, -1).T
        score = weighted_f1(labels, np.argmax(probabilities, axis=1))
        return "weighted_f1", score, True

    return evaluate


def class_weights(labels, num_classes):
    labels = np.asarray(labels, dtype="int64")
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise AssertionError(f"Training is missing classes {missing}")
    raw = np.sqrt(len(labels) / (num_classes * counts))
    sample = raw[labels]
    sample = sample / sample.mean()
    return sample


def parameters(num_classes):
    config = json.loads((ROOT / "configs" / "lightgbm_baseline.json").read_text())
    params = dict(config["common_parameters"])
    params.update(config["opportunity"])
    for key in ("class_weighting", "decision_rule", "validation_metric"):
        params.pop(key, None)
    seed = int(config["random_seed"])
    params.update(
        {
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "num_class": num_classes,
        }
    )
    return params, int(config["num_boost_round"]), int(config["early_stopping_rounds"])


def train_model(X_train, y_train, X_valid, y_valid, num_classes, rounds=None):
    params, max_rounds, early_rounds = parameters(num_classes)
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        weight=class_weights(y_train, num_classes),
        free_raw_data=False,
    )
    if rounds is None:
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_data,
            free_raw_data=False,
        )
        model = lgb.train(
            params,
            train_data,
            num_boost_round=max_rounds,
            valid_sets=[valid_data],
            valid_names=["validation"],
            feval=f1_eval(num_classes),
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=early_rounds,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        return model, int(model.best_iteration)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return model, rounds


def slice_xy(frame, features, mask):
    part = frame.loc[mask]
    X = part[features].to_numpy(dtype=np.float32)
    y = part["Opportunity_class_id"].to_numpy(dtype="int64")
    return X, y


def head_labels(class_ids):
    return np.array([HEAD_OF[int(cid)] for cid in class_ids], dtype="int64")


def route(head_prob, other_prob, adopt_prob):
    head = np.argmax(head_prob, axis=1)
    pred = np.empty(len(head), dtype="int64")
    pred[head == 0] = INACTIVITY_ID
    pred[head == 1] = STABLE_ID
    pred[head == 2] = FUEL_ID
    other = head == 3
    adopt = head == 4
    if other.any():
        local = np.argmax(other_prob[other], axis=1)
        pred[other] = np.asarray(OTHER_GROWTH_IDS, dtype="int64")[local]
    if adopt.any():
        local = np.argmax(adopt_prob[adopt], axis=1)
        pred[adopt] = np.asarray(ADOPTION_IDS, dtype="int64")[local]
    return pred


def fit_head_bundle(frame, features, train_cutoffs, valid_cutoff, rounds=None):
    train_mask = frame["cutoff"].isin(train_cutoffs)
    valid_mask = frame["cutoff"].eq(valid_cutoff)
    X_train, y_train = slice_xy(frame, features, train_mask)
    X_valid, y_valid = slice_xy(frame, features, valid_mask)
    head_train = head_labels(y_train)
    head_valid = head_labels(y_valid)

    fixed = rounds is not None
    head_rounds = None if not fixed else rounds["head"]
    other_rounds = None if not fixed else rounds["other"]
    adopt_rounds = None if not fixed else rounds["adopt"]

    head_model, head_used = train_model(
        X_train, head_train, X_valid, head_valid, 5, head_rounds
    )

    other_train = np.isin(y_train, OTHER_GROWTH_IDS)
    other_valid = np.isin(y_valid, OTHER_GROWTH_IDS)
    other_map = {cid: i for i, cid in enumerate(OTHER_GROWTH_IDS)}
    other_model, other_used = train_model(
        X_train[other_train],
        np.array([other_map[int(v)] for v in y_train[other_train]]),
        X_valid[other_valid],
        np.array([other_map[int(v)] for v in y_valid[other_valid]]),
        3,
        other_rounds,
    )

    adopt_train = np.isin(y_train, ADOPTION_IDS)
    adopt_valid = np.isin(y_valid, ADOPTION_IDS)
    adopt_map = {cid: i for i, cid in enumerate(ADOPTION_IDS)}
    adopt_model, adopt_used = train_model(
        X_train[adopt_train],
        np.array([adopt_map[int(v)] for v in y_train[adopt_train]]),
        X_valid[adopt_valid],
        np.array([adopt_map[int(v)] for v in y_valid[adopt_valid]]),
        11,
        adopt_rounds,
    )
    return {
        "head": head_model,
        "other": other_model,
        "adopt": adopt_model,
        "rounds": {"head": head_used, "other": other_used, "adopt": adopt_used},
        "X_valid": X_valid,
        "y_valid": y_valid,
    }


def predict_bundle(bundle, X):
    return route(
        bundle["head"].predict(X, num_iteration=bundle["rounds"]["head"]),
        bundle["other"].predict(X, num_iteration=bundle["rounds"]["other"]),
        bundle["adopt"].predict(X, num_iteration=bundle["rounds"]["adopt"]),
    )


def apply_shift(prob, stable_penalty, fuel_boost, inactivity_boost):
    adjusted = prob.copy()
    adjusted[:, STABLE_ID] -= stable_penalty
    adjusted[:, FUEL_ID] += fuel_boost
    adjusted[:, INACTIVITY_ID] += inactivity_boost
    return np.argmax(adjusted, axis=1)


def tune_shift(prob, y_true):
    """Grid is applied only to the previous labelled cutoff."""
    best = {
        "stable_penalty": 0.0,
        "fuel_boost": 0.0,
        "inactivity_boost": 0.0,
        "inner_f1": weighted_f1(y_true, np.argmax(prob, axis=1)),
    }
    for stable_penalty in (0.0, 0.05, 0.10, 0.15, 0.20):
        for fuel_boost in (0.0, 0.05, 0.10, 0.15):
            for inactivity_boost in (0.0, 0.05, 0.10):
                pred = apply_shift(
                    prob, stable_penalty, fuel_boost, inactivity_boost
                )
                score = weighted_f1(y_true, pred)
                if score > best["inner_f1"]:
                    best = {
                        "stable_penalty": stable_penalty,
                        "fuel_boost": fuel_boost,
                        "inactivity_boost": inactivity_boost,
                        "inner_f1": score,
                    }
    return best


def class_rows(fold_id, method, y_true, y_pred, names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(17)),
        zero_division=0,
    )
    rows = []
    for cid in range(17):
        rows.append(
            {
                "fold_id": fold_id,
                "method": method,
                "class_id": cid,
                "Opportunity": names[cid],
                "support": int(support[cid]),
                "precision": float(precision[cid]),
                "recall": float(recall[cid]),
                "f1": float(f1[cid]),
                "predicted": int(np.sum(y_pred == cid)),
            }
        )
    return rows


def passes(rows):
    improved = sum(row["change"] > 0 for row in rows)
    weighted_control = sum(row["n"] * row["control_f1"] for row in rows)
    weighted_candidate = sum(row["n"] * row["candidate_f1"] for row in rows)
    weight = sum(row["n"] for row in rows)
    f4 = next(row for row in rows if row["fold_id"] == "F4")
    return {
        "folds_improved": improved,
        "row_weighted_control_f1": weighted_control / weight,
        "row_weighted_candidate_f1": weighted_candidate / weight,
        "f4_not_worse": f4["change"] >= 0,
        "accepted": (
            weighted_candidate > weighted_control
            and f4["change"] >= 0
            and improved >= 2
        ),
    }


def load_frame():
    manifest = pd.read_csv(ROOT / "data" / "processed" / "phase04" / "model_feature_manifest.csv")
    features = manifest.loc[manifest["selected_for_baseline"].eq(1), "feature"].tolist()
    if len(features) != 276:
        raise AssertionError(f"Expected 276 features, found {len(features)}")
    usecols = ["snapshot_id", "ID", "cutoff", "Opportunity_class_id", *features]
    frame = pd.read_csv(
        ROOT / "data" / "processed" / "phase04" / "supervised_historical.csv",
        usecols=usecols,
    )
    frame["cutoff"] = frame["cutoff"].astype(str)
    frame = frame.sort_values(["cutoff", "snapshot_id"]).reset_index(drop=True)
    names = (
        pd.read_csv(ROOT / "data" / "processed" / "phase04" / "opportunity_class_map.csv")
        .sort_values("class_id")["Opportunity"]
        .tolist()
    )
    if len(names) != 17:
        raise AssertionError("Opportunity class map is not 17 classes")
    return frame, features, names


def evaluate():
    frame, features, names = load_frame()
    fold_rows = []
    metric_rows = []
    shift_rows = []
    print("Loaded", len(frame), "historical rows and", len(features), "features", flush=True)

    for fold_id, spec in FOLDS.items():
        print(f"\n=== {fold_id} valid {spec['valid']} ===", flush=True)
        train_mask = frame["cutoff"].isin(spec["train"])
        valid_mask = frame["cutoff"].eq(spec["valid"])
        X_train, y_train = slice_xy(frame, features, train_mask)
        X_valid, y_valid = slice_xy(frame, features, valid_mask)

        control, _ = train_model(X_train, y_train, X_valid, y_valid, 17)
        control_prob = control.predict(X_valid, num_iteration=control.best_iteration)
        control_pred = np.argmax(control_prob, axis=1)
        control_f1 = weighted_f1(y_valid, control_pred)
        print(f"control argmax F1 {control_f1:.6f}", flush=True)

        bundle = fit_head_bundle(frame, features, spec["train"], spec["valid"])
        head_pred = predict_bundle(bundle, bundle["X_valid"])
        head_f1 = weighted_f1(y_valid, head_pred)
        print(
            f"head route F1 {head_f1:.6f}  rounds {bundle['rounds']}",
            flush=True,
        )

        cal_cutoff = spec["train"][-1]
        cal_train = spec["train"][:-1]
        cal_mask = frame["cutoff"].eq(cal_cutoff)
        core_mask = frame["cutoff"].isin(cal_train)
        X_core, y_core = slice_xy(frame, features, core_mask)
        X_cal, y_cal = slice_xy(frame, features, cal_mask)
        cal_model, _ = train_model(X_core, y_core, X_cal, y_cal, 17)
        cal_prob = cal_model.predict(X_cal, num_iteration=cal_model.best_iteration)
        shift = tune_shift(cal_prob, y_cal)
        shift_pred = apply_shift(
            control_prob,
            shift["stable_penalty"],
            shift["fuel_boost"],
            shift["inactivity_boost"],
        )
        shift_f1 = weighted_f1(y_valid, shift_pred)
        print(
            f"shift F1 {shift_f1:.6f}  "
            f"stable-{shift['stable_penalty']} fuel+{shift['fuel_boost']} "
            f"inact+{shift['inactivity_boost']} inner {shift['inner_f1']:.6f}",
            flush=True,
        )

        fold_rows.append(
            {
                "fold_id": fold_id,
                "n": int(len(y_valid)),
                "control_f1": control_f1,
                "head_f1": head_f1,
                "shift_f1": shift_f1,
                "head_change": head_f1 - control_f1,
                "shift_change": shift_f1 - control_f1,
            }
        )
        shift_rows.append({"fold_id": fold_id, **shift})
        metric_rows.extend(class_rows(fold_id, "control", y_valid, control_pred, names))
        metric_rows.extend(class_rows(fold_id, "head", y_valid, head_pred, names))
        metric_rows.extend(class_rows(fold_id, "shift", y_valid, shift_pred, names))

    head_view = [
        {
            "fold_id": row["fold_id"],
            "n": row["n"],
            "control_f1": row["control_f1"],
            "candidate_f1": row["head_f1"],
            "change": row["head_change"],
        }
        for row in fold_rows
    ]
    shift_view = [
        {
            "fold_id": row["fold_id"],
            "n": row["n"],
            "control_f1": row["control_f1"],
            "candidate_f1": row["shift_f1"],
            "change": row["shift_change"],
        }
        for row in fold_rows
    ]
    head_decision = passes(head_view)
    shift_decision = passes(shift_view)
    if head_decision["accepted"]:
        chosen = "head"
    elif shift_decision["accepted"]:
        chosen = "shift"
    else:
        chosen = None

    decision = {
        "chosen": chosen,
        "head": head_decision,
        "shift": shift_decision,
        "folds": fold_rows,
        "shifts": shift_rows,
    }
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_comparison.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "class_metrics.csv", index=False)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))
    print("\nDecision", json.dumps({"chosen": chosen, "head": head_decision, "shift": shift_decision}, indent=2), flush=True)
    return frame, features, names, decision


def build_submission(frame, features, names, decision):
    if decision["chosen"] is None:
        print("No candidate passed. Submission file was not written.", flush=True)
        return None

    final_valid = CUTOFFS[-1]
    train_cutoffs = CUTOFFS[:-1]
    competition = pd.read_csv(
        ROOT / "data" / "processed" / "phase04" / "competition_features.csv"
    )
    X_comp = competition[features].to_numpy(dtype=np.float32)

    if decision["chosen"] == "head":
        selection = fit_head_bundle(frame, features, train_cutoffs, final_valid)
        full = fit_head_bundle(
            frame, features, CUTOFFS, final_valid, rounds=selection["rounds"]
        )
        pred = predict_bundle(full, X_comp)
        full["head"].save_model(str(MODEL_DIR / "final_head.txt"))
        full["other"].save_model(str(MODEL_DIR / "final_other_growth.txt"))
        full["adopt"].save_model(str(MODEL_DIR / "final_adoption.txt"))
        (OUT / "final_rounds.json").write_text(json.dumps(selection["rounds"], indent=2))
    else:
        X_train, y_train = slice_xy(frame, features, frame["cutoff"].isin(train_cutoffs))
        X_cal, y_cal = slice_xy(frame, features, frame["cutoff"].eq(final_valid))
        cal_model, rounds = train_model(X_train, y_train, X_cal, y_cal, 17)
        cal_prob = cal_model.predict(X_cal, num_iteration=rounds)
        shift = tune_shift(cal_prob, y_cal)
        X_all, y_all = slice_xy(frame, features, frame["cutoff"].isin(CUTOFFS))
        final, _ = train_model(X_all, y_all, X_cal, y_cal, 17, rounds=rounds)
        prob = final.predict(X_comp, num_iteration=rounds)
        pred = apply_shift(
            prob,
            shift["stable_penalty"],
            shift["fuel_boost"],
            shift["inactivity_boost"],
        )
        final.save_model(str(MODEL_DIR / "final_shift_17class.txt"))
        (OUT / "final_shift.json").write_text(json.dumps({"rounds": rounds, **shift}, indent=2))

    labels = [names[int(cid)] for cid in pred]
    opportunity = pd.DataFrame(
        {"ID": competition["ID"].astype(str), "Opportunity": labels}
    )
    if opportunity["Opportunity"].nunique() < 2:
        raise AssertionError("Opportunity predictions collapsed to one label")

    base = pd.read_csv(
        ROOT / "submissions" / "sasol_phase05_frozen_lightgbm_submission.csv",
        dtype=str,
    )
    sample = pd.read_csv(ROOT / "data" / "raw" / "SampleSubmission.csv", dtype=str)
    if not base["ID"].equals(sample["ID"]):
        raise AssertionError("Phase 05 submission ID order differs from SampleSubmission")
    merged = base[["ID", "CLV_fuel", "CLV_nonfuel"]].merge(
        opportunity, on="ID", how="left", validate="one_to_one"
    )
    if merged["Opportunity"].isna().any():
        raise AssertionError("Some submission IDs have no Opportunity prediction")
    allowed = set(names)
    unknown = set(merged["Opportunity"]) - allowed
    if unknown:
        raise AssertionError(f"Unexpected labels: {sorted(unknown)}")
    destination = HERE / "sasol_top4_opportunity_submission.csv"
    merged.to_csv(destination, index=False)
    counts = merged["Opportunity"].value_counts()
    counts.to_csv(OUT / "submission_label_counts.csv", header=["count"])
    print("Wrote", destination, flush=True)
    print(counts.to_string(), flush=True)
    return destination


def main():
    frame, features, names, decision = evaluate()
    build_submission(frame, features, names, decision)


if __name__ == "__main__":
    main()
