"""
Phase 04L: Opportunity class-weight intensity experiment.

Accepted weighting:

    (N / (K * n_c)) ** 0.50

Experimental weighting:

    (N / (K * n_c)) ** 0.75

Changes exactly one modelling assumption:
the exponent controlling class-weight intensity.

Everything else remains fixed relative to the accepted Phase 04D
Opportunity model.
"""

from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]

PHASE04_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
)

SUPERVISED_PATH = (
    PHASE04_DIR
    / "supervised_historical.csv"
)

FEATURE_MANIFEST_PATH = (
    PHASE04_DIR
    / "model_feature_manifest.csv"
)

CLASS_MAP_PATH = (
    PHASE04_DIR
    / "opportunity_class_map.csv"
)

FOLD_MEMBERSHIP_PATH = (
    PHASE04_DIR
    / "temporal_fold_membership.csv"
)

BASELINE_CONFIG_PATH = (
    ROOT
    / "configs"
    / "lightgbm_baseline.json"
)

ACCEPTED_METRICS_PATH = (
    PHASE04_DIR
    / "opportunity_weighting"
    / "fold_comparison.csv"
)

OUTPUT_DIR = (
    PHASE04_DIR
    / "opportunity_weight_intensity"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "phase04_opportunity_weight_intensity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ACCEPTED_EXPONENT = 0.50
EXPERIMENT_EXPONENT = 0.75


def weighted_f1_eval(
    num_classes,
):

    def evaluate(
        predictions,
        dataset,
    ):

        labels = (
            dataset
            .get_label()
            .astype("int64")
        )

        predictions = np.asarray(
            predictions
        )

        if predictions.ndim == 1:

            probabilities = (
                predictions
                .reshape(
                    num_classes,
                    -1,
                )
                .T
            )

        else:

            probabilities = predictions

        predicted = np.argmax(
            probabilities,
            axis=1,
        )

        score = f1_score(
            labels,
            predicted,
            average="weighted",
            zero_division=0,
        )

        return (
            "weighted_f1",
            float(score),
            True,
        )

    return evaluate


def make_parameters(
    config,
):

    parameters = dict(
        config[
            "common_parameters"
        ]
    )

    parameters.update(
        config[
            "opportunity"
        ]
    )

    # These are documentation fields rather
    # than LightGBM parameters.
    for key in [
        "class_weighting",
        "decision_rule",
        "validation_metric",
    ]:

        parameters.pop(
            key,
            None,
        )

    # Opportunity retains its accepted
    # Phase 04D tree structure.
    if int(
        parameters[
            "num_leaves"
        ]
    ) != 31:

        raise AssertionError(
            "Expected Opportunity "
            "num_leaves = 31."
        )

    if int(
        parameters[
            "min_data_in_leaf"
        ]
    ) != 50:

        raise AssertionError(
            "Expected Opportunity "
            "min_data_in_leaf = 50."
        )

    seed = int(
        config[
            "random_seed"
        ]
    )

    parameters.update(
        {
            "seed":
                seed,

            "feature_fraction_seed":
                seed,

            "bagging_seed":
                seed,

            "data_random_seed":
                seed,
        }
    )

    return parameters


def make_training_weights(
    labels,
    num_classes,
    exponent,
):
    """
    Class weighting rule:

        raw_w_c =
            (N / (K * n_c)) ** exponent

    Sample weights are normalized to
    mean exactly 1 within each fold.
    """

    labels = np.asarray(
        labels,
        dtype="int64",
    )

    counts = np.bincount(
        labels,
        minlength=num_classes,
    )

    if np.any(
        counts == 0
    ):

        raise AssertionError(
            "At least one Opportunity class "
            "is absent from training."
        )

    total = len(
        labels
    )

    raw_class_weights = (
        total
        /
        (
            num_classes
            * counts
        )
    ) ** float(
        exponent
    )

    raw_sample_weights = (
        raw_class_weights[
            labels
        ]
    )

    normalizer = (
        raw_sample_weights.mean()
    )

    class_weights = (
        raw_class_weights
        /
        normalizer
    )

    sample_weights = (
        class_weights[
            labels
        ]
    )

    if not np.isclose(
        sample_weights.mean(),
        1.0,
        atol=1e-12,
    ):

        raise AssertionError(
            "Sample weights do not "
            "normalize to mean 1."
        )

    return (
        sample_weights,
        counts,
        class_weights,
    )


def macro_label(
    label,
):

    if label == "Inactivity":
        return "Inactivity"

    if label == "Stable":
        return "Stable"

    if label.startswith(
        "Existing-category growth:"
    ):
        return "Growth"

    if label.startswith(
        "New category adoption:"
    ):
        return "Adoption"

    raise ValueError(
        f"Unknown Opportunity label: "
        f"{label}"
    )


def main():

    print(
        "Loading Phase 04L inputs..."
    )

    config = json.loads(
        BASELINE_CONFIG_PATH.read_text()
    )

    data = pd.read_csv(
        SUPERVISED_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    manifest = pd.read_csv(
        FEATURE_MANIFEST_PATH
    )

    class_map = pd.read_csv(
        CLASS_MAP_PATH
    ).sort_values(
        "class_id"
    )

    membership = pd.read_csv(
        FOLD_MEMBERSHIP_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    accepted_metrics = pd.read_csv(
        ACCEPTED_METRICS_PATH
    )

    selected_features = (
        manifest.loc[
            manifest[
                "selected_for_baseline"
            ].eq(1),
            "feature",
        ]
        .tolist()
    )

    if len(
        selected_features
    ) != 276:

        raise AssertionError(
            "Expected 276 predictors."
        )

    num_classes = len(
        class_map
    )

    if num_classes != 17:

        raise AssertionError(
            "Expected 17 Opportunity classes."
        )

    class_lookup = dict(
        zip(
            class_map[
                "class_id"
            ].astype(int),
            class_map[
                "Opportunity"
            ].astype(str),
        )
    )

    parameters = make_parameters(
        config
    )

    num_boost_round = int(
        config[
            "num_boost_round"
        ]
    )

    early_stopping_rounds = int(
        config[
            "early_stopping_rounds"
        ]
    )

    folds = sorted(
        membership[
            "fold_id"
        ].unique()
    )

    metric_rows = []
    class_metric_rows = []
    weight_rows = []
    prediction_frames = []
    macro_rows = []

    # ============================================================
    # TEMPORAL EXPERIMENT
    # ============================================================

    for fold_id in folds:

        print()
        print(
            "=" * 72
        )

        print(
            f"Training Opportunity "
            f"alpha={EXPERIMENT_EXPONENT:.2f}: "
            f"{fold_id}"
        )

        print(
            "=" * 72
        )

        fold = membership[
            membership[
                "fold_id"
            ].eq(
                fold_id
            )
        ]

        train_ids = set(
            fold.loc[
                fold[
                    "role"
                ].eq(
                    "train"
                ),
                "snapshot_id",
            ]
        )

        valid_ids = set(
            fold.loc[
                fold[
                    "role"
                ].eq(
                    "validation"
                ),
                "snapshot_id",
            ]
        )

        train = data[
            data[
                "snapshot_id"
            ].isin(
                train_ids
            )
        ].copy()

        valid = data[
            data[
                "snapshot_id"
            ].isin(
                valid_ids
            )
        ].copy()

        if len(
            train
        ) != len(
            train_ids
        ):

            raise AssertionError(
                f"{fold_id} train membership "
                f"mismatch."
            )

        if len(
            valid
        ) != len(
            valid_ids
        ):

            raise AssertionError(
                f"{fold_id} validation membership "
                f"mismatch."
            )

        X_train = train[
            selected_features
        ]

        X_valid = valid[
            selected_features
        ]

        y_train = train[
            "Opportunity_class_id"
        ].to_numpy(
            dtype="int64"
        )

        y_valid = valid[
            "Opportunity_class_id"
        ].to_numpy(
            dtype="int64"
        )

        (
            sample_weights,
            class_counts,
            class_weights,
        ) = make_training_weights(
            y_train,
            num_classes,
            EXPERIMENT_EXPONENT,
        )

        for class_id in range(
            num_classes
        ):

            weight_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "class_id":
                        class_id,

                    "Opportunity":
                        class_lookup[
                            class_id
                        ],

                    "training_count":
                        int(
                            class_counts[
                                class_id
                            ]
                        ),

                    "weight_exponent":
                        EXPERIMENT_EXPONENT,

                    "normalized_class_weight":
                        float(
                            class_weights[
                                class_id
                            ]
                        ),
                }
            )

        train_data = lgb.Dataset(
            X_train,
            label=
                y_train,
            weight=
                sample_weights,
            feature_name=
                selected_features,
            free_raw_data=False,
        )

        # Validation remains unweighted.
        valid_data = lgb.Dataset(
            X_valid,
            label=
                y_valid,
            reference=
                train_data,
            feature_name=
                selected_features,
            free_raw_data=False,
        )

        model = lgb.train(
            parameters,
            train_data,
            num_boost_round=
                num_boost_round,
            valid_sets=[
                valid_data,
            ],
            valid_names=[
                "validation",
            ],
            feval=weighted_f1_eval(
                num_classes
            ),
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=
                        early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(
                    period=0
                ),
            ],
        )

        probabilities = model.predict(
            X_valid,
            num_iteration=
                model.best_iteration,
        )

        prediction = np.argmax(
            probabilities,
            axis=1,
        ).astype(
            "int64"
        )

        weighted_f1 = f1_score(
            y_valid,
            prediction,
            average="weighted",
            zero_division=0,
        )

        macro_f1 = f1_score(
            y_valid,
            prediction,
            average="macro",
            zero_division=0,
        )

        accepted_row = (
            accepted_metrics[
                accepted_metrics[
                    "fold_id"
                ].eq(
                    fold_id
                )
            ]
            .iloc[0]
        )

        accepted_f1 = float(
            accepted_row[
                "weighted_model_f1"
            ]
        )

        metric_rows.append(
            {
                "fold_id":
                    fold_id,

                "train_rows":
                    len(
                        train
                    ),

                "validation_rows":
                    len(
                        valid
                    ),

                "accepted_weight_exponent":
                    ACCEPTED_EXPONENT,

                "experiment_weight_exponent":
                    EXPERIMENT_EXPONENT,

                "best_iteration":
                    model.best_iteration,

                "accepted_weighted_f1":
                    accepted_f1,

                "experiment_weighted_f1":
                    float(
                        weighted_f1
                    ),

                "weighted_f1_change":
                    float(
                        weighted_f1
                        -
                        accepted_f1
                    ),

                "experiment_macro_f1":
                    float(
                        macro_f1
                    ),

                "predicted_classes":
                    int(
                        np.unique(
                            prediction
                        ).size
                    ),
            }
        )

        # ========================================================
        # CLASS METRICS
        # ========================================================

        (
            precision,
            recall,
            f1_values,
            support,
        ) = precision_recall_fscore_support(
            y_valid,
            prediction,
            labels=np.arange(
                num_classes
            ),
            zero_division=0,
        )

        for class_id in range(
            num_classes
        ):

            class_metric_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "class_id":
                        class_id,

                    "Opportunity":
                        class_lookup[
                            class_id
                        ],

                    "macro":
                        macro_label(
                            class_lookup[
                                class_id
                            ]
                        ),

                    "support":
                        int(
                            support[
                                class_id
                            ]
                        ),

                    "predicted_count":
                        int(
                            (
                                prediction
                                ==
                                class_id
                            ).sum()
                        ),

                    "precision":
                        float(
                            precision[
                                class_id
                            ]
                        ),

                    "recall":
                        float(
                            recall[
                                class_id
                            ]
                        ),

                    "f1":
                        float(
                            f1_values[
                                class_id
                            ]
                        ),
                }
            )

        # ========================================================
        # MACRO-BEHAVIOUR DIAGNOSTICS
        # ========================================================

        true_labels = [
            class_lookup[
                int(value)
            ]
            for value in y_valid
        ]

        pred_labels = [
            class_lookup[
                int(value)
            ]
            for value in prediction
        ]

        true_macro = np.asarray(
            [
                macro_label(
                    value
                )
                for value in true_labels
            ]
        )

        pred_macro = np.asarray(
            [
                macro_label(
                    value
                )
                for value in pred_labels
            ]
        )

        macro_order = [
            "Adoption",
            "Growth",
            "Inactivity",
            "Stable",
        ]

        (
            macro_precision,
            macro_recall,
            macro_f1_values,
            macro_support,
        ) = precision_recall_fscore_support(
            true_macro,
            pred_macro,
            labels=
                macro_order,
            zero_division=0,
        )

        for index, macro in enumerate(
            macro_order
        ):

            macro_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "macro":
                        macro,

                    "support":
                        int(
                            macro_support[
                                index
                            ]
                        ),

                    "precision":
                        float(
                            macro_precision[
                                index
                            ]
                        ),

                    "recall":
                        float(
                            macro_recall[
                                index
                            ]
                        ),

                    "f1":
                        float(
                            macro_f1_values[
                                index
                            ]
                        ),
                }
            )

        prediction_frame = valid[
            [
                "snapshot_id",
                "ID",
                "cutoff",
                "Opportunity",
                "Opportunity_class_id",
            ]
        ].copy()

        prediction_frame[
            "fold_id"
        ] = fold_id

        prediction_frame[
            "pred_Opportunity_class_id"
        ] = prediction

        prediction_frame[
            "pred_Opportunity"
        ] = pred_labels

        prediction_frames.append(
            prediction_frame
        )

        model.save_model(
            str(
                MODEL_DIR
                /
                (
                    f"{fold_id}_"
                    f"opportunity_"
                    f"alpha075.txt"
                )
            )
        )

        print(
            f"{fold_id} | "
            f"accepted F1: "
            f"{accepted_f1:.6f} | "
            f"alpha075 F1: "
            f"{weighted_f1:.6f} | "
            f"change: "
            f"{weighted_f1-accepted_f1:+.6f} | "
            f"macro F1: "
            f"{macro_f1:.6f}"
        )

    # ============================================================
    # SUMMARISE
    # ============================================================

    metrics = pd.DataFrame(
        metric_rows
    )

    class_metrics = pd.DataFrame(
        class_metric_rows
    )

    weights = pd.DataFrame(
        weight_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    macro_metrics = pd.DataFrame(
        macro_rows
    )

    validation_weights = (
        metrics[
            "validation_rows"
        ]
        /
        metrics[
            "validation_rows"
        ].sum()
    )

    accepted_temporal_f1 = float(
        np.sum(
            metrics[
                "accepted_weighted_f1"
            ]
            *
            validation_weights
        )
    )

    experiment_temporal_f1 = float(
        np.sum(
            metrics[
                "experiment_weighted_f1"
            ]
            *
            validation_weights
        )
    )

    folds_improved = int(
        metrics[
            "experiment_weighted_f1"
        ].gt(
            metrics[
                "accepted_weighted_f1"
            ]
        ).sum()
    )

    f4 = metrics[
        metrics[
            "fold_id"
        ].eq(
            "F4"
        )
    ].iloc[0]

    criterion_temporal = (
        experiment_temporal_f1
        >
        accepted_temporal_f1
    )

    criterion_f4 = (
        float(
            f4[
                "experiment_weighted_f1"
            ]
        )
        >=
        float(
            f4[
                "accepted_weighted_f1"
            ]
        )
    )

    criterion_folds = (
        folds_improved >= 3
    )

    accepted_by_rule = (
        criterion_temporal
        and
        criterion_f4
        and
        criterion_folds
    )

    summary = pd.DataFrame(
        [
            {
                "accepted_weight_exponent":
                    ACCEPTED_EXPONENT,

                "experiment_weight_exponent":
                    EXPERIMENT_EXPONENT,

                "folds":
                    len(
                        metrics
                    ),

                "folds_improved":
                    folds_improved,

                "row_weighted_accepted_f1":
                    accepted_temporal_f1,

                "row_weighted_experiment_f1":
                    experiment_temporal_f1,

                "row_weighted_f1_change":
                    (
                        experiment_temporal_f1
                        -
                        accepted_temporal_f1
                    ),

                "f4_accepted_f1":
                    float(
                        f4[
                            "accepted_weighted_f1"
                        ]
                    ),

                "f4_experiment_f1":
                    float(
                        f4[
                            "experiment_weighted_f1"
                        ]
                    ),

                "f4_f1_change":
                    float(
                        f4[
                            "experiment_weighted_f1"
                        ]
                        -
                        f4[
                            "accepted_weighted_f1"
                        ]
                    ),

                "f4_macro_f1":
                    float(
                        f4[
                            "experiment_macro_f1"
                        ]
                    ),

                "f4_predicted_classes":
                    int(
                        f4[
                            "predicted_classes"
                        ]
                    ),

                "criterion_temporal_f1_improved":
                    int(
                        criterion_temporal
                    ),

                "criterion_f4_not_worse":
                    int(
                        criterion_f4
                    ),

                "criterion_three_folds_improved":
                    int(
                        criterion_folds
                    ),

                "accepted_by_preregistered_rule":
                    int(
                        accepted_by_rule
                    ),

                "validation_status":
                    "PASS",
            }
        ]
    )

    # ============================================================
    # SAVE
    # ============================================================

    metrics.to_csv(
        OUTPUT_DIR
        / "fold_comparison.csv",
        index=False,
    )

    class_metrics.to_csv(
        OUTPUT_DIR
        / "class_metrics.csv",
        index=False,
    )

    macro_metrics.to_csv(
        OUTPUT_DIR
        / "macro_metrics.csv",
        index=False,
    )

    weights.to_csv(
        OUTPUT_DIR
        / "training_class_weights.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "validation_predictions.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "weight_intensity_summary.csv",
        index=False,
    )

    # ============================================================
    # CONSOLE OUTPUT
    # ============================================================

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 04L OPPORTUNITY WEIGHT-INTENSITY EXPERIMENT COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "FOLD COMPARISON"
    )

    print()

    print(
        metrics.to_string(
            index=False
        )
    )

    print()

    print(
        "F4 CLASS METRICS"
    )

    print()

    print(
        class_metrics[
            class_metrics[
                "fold_id"
            ].eq(
                "F4"
            )
        ][
            [
                "Opportunity",
                "macro",
                "support",
                "predicted_count",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .sort_values(
            "support",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()

    print(
        "F4 MACRO METRICS"
    )

    print()

    print(
        macro_metrics[
            macro_metrics[
                "fold_id"
            ].eq(
                "F4"
            )
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "SUMMARY"
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Outputs written to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
