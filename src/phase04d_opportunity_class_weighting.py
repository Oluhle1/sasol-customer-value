"""
Phase 04D: Controlled Opportunity class-weighting experiment.

Changes exactly one modelling assumption relative to Phase 04B:

    training sample weights =
        square-root inverse Opportunity class frequency

Everything else remains fixed:

- same 276 predictors;
- same four rolling-origin folds;
- same LightGBM multiclass model;
- same tree parameters;
- same random seed;
- same weighted-F1 early stopping;
- same argmax decision rule.

Regression models are not retrained.

Weights are calculated from each fold's TRAINING population only.
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

BASELINE_METRICS_PATH = (
    PHASE04_DIR
    / "baseline"
    / "fold_metrics.csv"
)

BASELINE_CONFIG_PATH = (
    ROOT
    / "configs"
    / "lightgbm_baseline.json"
)

OUTPUT_DIR = (
    PHASE04_DIR
    / "opportunity_weighting"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "phase04_opportunity_weighting"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


def opportunity_parameters(
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

    for documentation_key in [
        "class_weighting",
        "decision_rule",
        "validation_metric",
    ]:

        parameters.pop(
            documentation_key,
            None,
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
):
    """
    Square-root inverse-frequency weighting.

    Raw class weight:
        sqrt(N / (K * n_c))

    Sample weights are then normalized so their mean is exactly 1.
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

    raw_class_weights = np.sqrt(
        total
        /
        (
            num_classes
            * counts
        )
    )

    sample_weights = (
        raw_class_weights[
            labels
        ]
    )

    sample_weights = (
        sample_weights
        /
        sample_weights.mean()
    )

    normalized_class_weights = (
        raw_class_weights
        /
        sample_weights.mean()
    )

    # Reconstruct class-level weights after
    # final sample normalization.
    normalizer = (
        raw_class_weights[
            labels
        ].mean()
    )

    normalized_class_weights = (
        raw_class_weights
        /
        normalizer
    )

    sample_weights = (
        normalized_class_weights[
            labels
        ]
    )

    if not np.isclose(
        sample_weights.mean(),
        1.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Training weights do not "
            "normalize to mean 1."
        )

    return (
        sample_weights,
        counts,
        normalized_class_weights,
    )


def main():

    print(
        "Loading Phase 04D inputs..."
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

    baseline_metrics = pd.read_csv(
        BASELINE_METRICS_PATH
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

    parameters = opportunity_parameters(
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

    metric_rows = []
    class_metric_rows = []
    weight_rows = []
    prediction_frames = []

    folds = sorted(
        membership[
            "fold_id"
        ].unique()
    )

    for fold_id in folds:

        print()
        print(
            "=" * 72
        )

        print(
            f"Training weighted Opportunity "
            f"model: {fold_id}"
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

                    "class_weight":
                        float(
                            class_weights[
                                class_id
                            ]
                        ),
                }
            )

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weights,
            feature_name=list(
                selected_features
            ),
            free_raw_data=False,
        )

        # Validation remains UNWEIGHTED because the
        # official evaluation population is unweighted.
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_data,
            feature_name=list(
                selected_features
            ),
            free_raw_data=False,
        )

        model = lgb.train(
            parameters,
            train_data,
            num_boost_round=
                num_boost_round,
            valid_sets=[
                valid_data
            ],
            valid_names=[
                "validation"
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

        baseline_row = (
            baseline_metrics[
                baseline_metrics[
                    "fold_id"
                ].eq(
                    fold_id
                )
            ]
            .iloc[0]
        )

        baseline_f1 = float(
            baseline_row[
                "opportunity_weighted_f1"
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

                "best_iteration":
                    model.best_iteration,

                "baseline_weighted_f1":
                    baseline_f1,

                "weighted_model_f1":
                    float(
                        weighted_f1
                    ),

                "absolute_f1_change":
                    float(
                        weighted_f1
                        -
                        baseline_f1
                    ),

                "macro_f1":
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

        pred_frame = valid[
            [
                "snapshot_id",
                "ID",
                "cutoff",
                "Opportunity",
                "Opportunity_class_id",
            ]
        ].copy()

        pred_frame[
            "fold_id"
        ] = fold_id

        pred_frame[
            "pred_Opportunity_class_id"
        ] = prediction

        pred_frame[
            "pred_Opportunity"
        ] = [
            class_lookup[
                int(value)
            ]
            for value in prediction
        ]

        prediction_frames.append(
            pred_frame
        )

        model.save_model(
            str(
                MODEL_DIR
                / f"{fold_id}_opportunity_weighted.txt"
            )
        )

        print(
            f"{fold_id} | "
            f"baseline F1: "
            f"{baseline_f1:.6f} | "
            f"weighted F1: "
            f"{weighted_f1:.6f} | "
            f"change: "
            f"{weighted_f1-baseline_f1:+.6f}"
        )

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

    # Row-weighted temporal summary.
    validation_weight = (
        metrics[
            "validation_rows"
        ]
        /
        metrics[
            "validation_rows"
        ].sum()
    )

    baseline_temporal_f1 = float(
        np.sum(
            metrics[
                "baseline_weighted_f1"
            ]
            *
            validation_weight
        )
    )

    weighted_temporal_f1 = float(
        np.sum(
            metrics[
                "weighted_model_f1"
            ]
            *
            validation_weight
        )
    )

    summary = pd.DataFrame(
        [
            {
                "folds":
                    len(
                        metrics
                    ),

                "baseline_row_weighted_f1":
                    baseline_temporal_f1,

                "weighted_model_row_weighted_f1":
                    weighted_temporal_f1,

                "absolute_change":
                    weighted_temporal_f1
                    -
                    baseline_temporal_f1,

                "f4_baseline_f1":
                    float(
                        metrics.loc[
                            metrics[
                                "fold_id"
                            ].eq(
                                "F4"
                            ),
                            "baseline_weighted_f1",
                        ].iloc[0]
                    ),

                "f4_weighted_f1":
                    float(
                        metrics.loc[
                            metrics[
                                "fold_id"
                            ].eq(
                                "F4"
                            ),
                            "weighted_model_f1",
                        ].iloc[0]
                    ),

                "validation_status":
                    "PASS",
            }
        ]
    )

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
        / "weighting_summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 04D OPPORTUNITY WEIGHTING EXPERIMENT COMPLETE"
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
