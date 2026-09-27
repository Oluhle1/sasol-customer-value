"""
Phase 04K: Regression minimum-leaf-support experiment.

Controlled change:

    min_data_in_leaf: 50 -> 100

Fixed accepted regression configuration:

    num_leaves = 15

Both Fuel and non-fuel are evaluated independently across the same
four expanding-window temporal folds.

No other modelling parameter changes.
"""

from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd


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

MANIFEST_PATH = (
    PHASE04_DIR
    / "model_feature_manifest.csv"
)

MEMBERSHIP_PATH = (
    PHASE04_DIR
    / "temporal_fold_membership.csv"
)

BASELINE_CONFIG_PATH = (
    ROOT
    / "configs"
    / "lightgbm_baseline.json"
)

ACCEPTED_FUEL_METRICS_PATH = (
    PHASE04_DIR
    / "fuel_lower_capacity"
    / "fold_comparison.csv"
)

ACCEPTED_NONFUEL_METRICS_PATH = (
    PHASE04_DIR
    / "nonfuel_lower_capacity"
    / "fold_comparison.csv"
)

OUTPUT_DIR = (
    PHASE04_DIR
    / "regression_leaf_support"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "phase04_regression_leaf_support"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def rmse(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype="float64",
    )

    y_pred = np.asarray(
        y_pred,
        dtype="float64",
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    y_true
                    -
                    y_pred
                )
                ** 2
            )
        )
    )


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
            "regression"
        ]
    )

    parameters.pop(
        "prediction_floor",
        None,
    )

    # Confirm original baseline values before override.
    if int(
        parameters[
            "num_leaves"
        ]
    ) != 31:

        raise AssertionError(
            "Expected original baseline "
            "num_leaves = 31."
        )

    if int(
        parameters[
            "min_data_in_leaf"
        ]
    ) != 50:

        raise AssertionError(
            "Expected original baseline "
            "min_data_in_leaf = 50."
        )

    # Accepted Phase 04I / 04J complexity.
    parameters[
        "num_leaves"
    ] = 15

    # ------------------------------------------------------------
    # PHASE 04K CONTROLLED CHANGE
    # ------------------------------------------------------------

    parameters[
        "min_data_in_leaf"
    ] = 100

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


def accepted_metrics_for_task(
    task,
):

    if task == "fuel":

        path = (
            ACCEPTED_FUEL_METRICS_PATH
        )

    elif task == "nonfuel":

        path = (
            ACCEPTED_NONFUEL_METRICS_PATH
        )

    else:

        raise ValueError(
            task
        )

    metrics = pd.read_csv(
        path
    )

    required = {
        "fold_id",
        "capacity_validation_rmse",
    }

    missing = (
        required
        - set(
            metrics.columns
        )
    )

    if missing:

        raise AssertionError(
            f"{path} missing "
            f"columns: {missing}"
        )

    return metrics


def main():

    print(
        "Loading Phase 04K inputs..."
    )

    config = json.loads(
        BASELINE_CONFIG_PATH.read_text()
    )

    data = pd.read_csv(
        SUPERVISED_PATH,
        dtype={
            "ID": str,
        },
        parse_dates=[
            "cutoff",
        ],
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    membership = pd.read_csv(
        MEMBERSHIP_PATH,
        dtype={
            "ID": str,
        },
        parse_dates=[
            "cutoff",
        ],
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
            "Expected 276 selected predictors."
        )

    parameters = make_parameters(
        config
    )

    if int(
        parameters[
            "num_leaves"
        ]
    ) != 15:

        raise AssertionError(
            "Phase 04K must use "
            "num_leaves = 15."
        )

    if int(
        parameters[
            "min_data_in_leaf"
        ]
    ) != 100:

        raise AssertionError(
            "Phase 04K must use "
            "min_data_in_leaf = 100."
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
    prediction_frames = []
    importance_rows = []

    tasks = [
        (
            "fuel",
            "CLV_fuel",
        ),
        (
            "nonfuel",
            "CLV_nonfuel",
        ),
    ]

    folds = sorted(
        membership[
            "fold_id"
        ].unique()
    )

    # ============================================================
    # TASK-SPECIFIC EXPERIMENTS
    # ============================================================

    for (
        task,
        target,
    ) in tasks:

        print()
        print(
            "#" * 76
        )

        print(
            f"TASK: {task.upper()}"
        )

        print(
            "#" * 76
        )

        accepted_metrics = (
            accepted_metrics_for_task(
                task
            )
        )

        for fold_id in folds:

            print()
            print(
                "=" * 72
            )

            print(
                f"Training {task} "
                f"leaf-support model: "
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
                    f"{task} {fold_id}: "
                    f"training membership mismatch."
                )

            if len(
                valid
            ) != len(
                valid_ids
            ):

                raise AssertionError(
                    f"{task} {fold_id}: "
                    f"validation membership mismatch."
                )

            X_train = train[
                selected_features
            ]

            X_valid = valid[
                selected_features
            ]

            y_train = (
                train[
                    target
                ]
                .to_numpy(
                    dtype="float64"
                )
            )

            y_valid = (
                valid[
                    target
                ]
                .to_numpy(
                    dtype="float64"
                )
            )

            train_data = lgb.Dataset(
                X_train,
                label=
                    y_train,
                feature_name=
                    selected_features,
                free_raw_data=False,
            )

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

            raw_valid_prediction = (
                model.predict(
                    X_valid,
                    num_iteration=
                        model.best_iteration,
                )
            )

            valid_prediction = np.maximum(
                raw_valid_prediction,
                0.0,
            )

            raw_train_prediction = (
                model.predict(
                    X_train,
                    num_iteration=
                        model.best_iteration,
                )
            )

            train_prediction = np.maximum(
                raw_train_prediction,
                0.0,
            )

            experiment_rmse = rmse(
                y_valid,
                valid_prediction,
            )

            train_rmse = rmse(
                y_train,
                train_prediction,
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

            accepted_rmse = float(
                accepted_row[
                    "capacity_validation_rmse"
                ]
            )

            metric_rows.append(
                {
                    "task":
                        task,

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

                    "num_leaves":
                        15,

                    "accepted_min_data_in_leaf":
                        50,

                    "experiment_min_data_in_leaf":
                        100,

                    "best_iteration":
                        model.best_iteration,

                    "train_rmse":
                        float(
                            train_rmse
                        ),

                    "experiment_validation_rmse":
                        float(
                            experiment_rmse
                        ),

                    "accepted_validation_rmse":
                        accepted_rmse,

                    "rmse_change":
                        float(
                            experiment_rmse
                            -
                            accepted_rmse
                        ),

                    "validation_to_train_rmse_ratio":
                        float(
                            experiment_rmse
                            /
                            train_rmse
                        ),

                    "raw_negative_predictions":
                        int(
                            (
                                raw_valid_prediction
                                < 0
                            ).sum()
                        ),
                }
            )

            prediction_frame = valid[
                [
                    "snapshot_id",
                    "ID",
                    "cutoff",
                    target,
                ]
            ].copy()

            prediction_frame[
                "task"
            ] = task

            prediction_frame[
                "fold_id"
            ] = fold_id

            prediction_frame[
                "experiment_prediction"
            ] = valid_prediction

            prediction_frames.append(
                prediction_frame
            )

            gain = model.feature_importance(
                importance_type="gain"
            )

            split = model.feature_importance(
                importance_type="split"
            )

            for (
                feature,
                gain_value,
                split_value,
            ) in zip(
                selected_features,
                gain,
                split,
            ):

                importance_rows.append(
                    {
                        "task":
                            task,

                        "fold_id":
                            fold_id,

                        "feature":
                            feature,

                        "gain":
                            float(
                                gain_value
                            ),

                        "split":
                            int(
                                split_value
                            ),
                    }
                )

            model.save_model(
                str(
                    MODEL_DIR
                    /
                    (
                        f"{fold_id}_"
                        f"{task}_"
                        f"leaves15_"
                        f"minleaf100.txt"
                    )
                )
            )

            print(
                f"{fold_id} | "
                f"accepted RMSE: "
                f"{accepted_rmse:.6f} | "
                f"minleaf100 RMSE: "
                f"{experiment_rmse:.6f} | "
                f"change: "
                f"{experiment_rmse-accepted_rmse:+.6f} | "
                f"train RMSE: "
                f"{train_rmse:.6f}"
            )

    # ============================================================
    # COMBINE RESULTS
    # ============================================================

    metrics = pd.DataFrame(
        metric_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    importance = pd.DataFrame(
        importance_rows
    )

    summary_rows = []

    for task, block in metrics.groupby(
        "task"
    ):

        validation_weights = (
            block[
                "validation_rows"
            ]
            /
            block[
                "validation_rows"
            ].sum()
        )

        accepted_weighted_rmse = float(
            np.sum(
                block[
                    "accepted_validation_rmse"
                ]
                *
                validation_weights
            )
        )

        experiment_weighted_rmse = float(
            np.sum(
                block[
                    "experiment_validation_rmse"
                ]
                *
                validation_weights
            )
        )

        folds_improved = int(
            block[
                "experiment_validation_rmse"
            ].lt(
                block[
                    "accepted_validation_rmse"
                ]
            ).sum()
        )

        f4 = block[
            block[
                "fold_id"
            ].eq(
                "F4"
            )
        ].iloc[0]

        weighted_improved = (
            experiment_weighted_rmse
            <
            accepted_weighted_rmse
        )

        f4_not_worse = (
            float(
                f4[
                    "experiment_validation_rmse"
                ]
            )
            <=
            float(
                f4[
                    "accepted_validation_rmse"
                ]
            )
        )

        at_least_three_folds = (
            folds_improved >= 3
        )

        accepted_by_rule = (
            weighted_improved
            and
            f4_not_worse
            and
            at_least_three_folds
        )

        summary_rows.append(
            {
                "task":
                    task,

                "num_leaves":
                    15,

                "accepted_min_data_in_leaf":
                    50,

                "experiment_min_data_in_leaf":
                    100,

                "folds":
                    len(
                        block
                    ),

                "folds_improved":
                    folds_improved,

                "row_weighted_accepted_rmse":
                    accepted_weighted_rmse,

                "row_weighted_experiment_rmse":
                    experiment_weighted_rmse,

                "row_weighted_rmse_change":
                    (
                        experiment_weighted_rmse
                        -
                        accepted_weighted_rmse
                    ),

                "f4_accepted_rmse":
                    float(
                        f4[
                            "accepted_validation_rmse"
                        ]
                    ),

                "f4_experiment_rmse":
                    float(
                        f4[
                            "experiment_validation_rmse"
                        ]
                    ),

                "f4_rmse_change":
                    float(
                        f4[
                            "experiment_validation_rmse"
                        ]
                        -
                        f4[
                            "accepted_validation_rmse"
                        ]
                    ),

                "f4_train_rmse":
                    float(
                        f4[
                            "train_rmse"
                        ]
                    ),

                "f4_validation_to_train_ratio":
                    float(
                        f4[
                            "validation_to_train_rmse_ratio"
                        ]
                    ),

                "criterion_weighted_rmse_improved":
                    int(
                        weighted_improved
                    ),

                "criterion_f4_not_worse":
                    int(
                        f4_not_worse
                    ),

                "criterion_three_folds_improved":
                    int(
                        at_least_three_folds
                    ),

                "accepted_by_preregistered_rule":
                    int(
                        accepted_by_rule
                    ),

                "validation_status":
                    "PASS",
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # ============================================================
    # SAVE
    # ============================================================

    metrics.to_csv(
        OUTPUT_DIR
        / "fold_comparison.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "validation_predictions.csv",
        index=False,
        float_format="%.10g",
    )

    importance.to_csv(
        OUTPUT_DIR
        / "feature_importance.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "task_summary.csv",
        index=False,
    )

    # ============================================================
    # CONSOLE CHECKPOINT
    # ============================================================

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 04K REGRESSION LEAF-SUPPORT EXPERIMENT COMPLETE"
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
        metrics[
            [
                "task",
                "fold_id",
                "train_rows",
                "validation_rows",
                "num_leaves",
                "accepted_min_data_in_leaf",
                "experiment_min_data_in_leaf",
                "best_iteration",
                "train_rmse",
                "accepted_validation_rmse",
                "experiment_validation_rmse",
                "rmse_change",
                "validation_to_train_rmse_ratio",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "TASK SUMMARY"
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
