"""
Phase 04I: Controlled direct non-fuel LightGBM capacity experiment.

Changes exactly one baseline hyperparameter:

    num_leaves: 31 -> 15

Everything else is fixed:
- direct CLV_nonfuel regression;
- same 276 predictors;
- same temporal folds;
- same learning rate;
- same min_data_in_leaf;
- same bagging and feature fractions;
- same regularization;
- same random seed;
- same RMSE early stopping.

Purpose:
Test whether the small latest-period train-validation gap indicates
useful additional nonlinear interaction capacity.
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
    / "nonfuel_lower_capacity"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "phase04_nonfuel_lower_capacity"
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

    # ------------------------------------------------------------
    # Phase 04I CONTROLLED CHANGE
    # ------------------------------------------------------------

    baseline_num_leaves = int(
        parameters[
            "num_leaves"
        ]
    )

    if baseline_num_leaves != 31:

        raise AssertionError(
            f"Expected baseline num_leaves=31, "
            f"found {baseline_num_leaves}."
        )

    parameters[
        "num_leaves"
    ] = 15

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


def main():

    print(
        "Loading Phase 04I inputs..."
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

    metric_rows = []
    prediction_frames = []
    importance_rows = []

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
            f"Training non-fuel capacity model: "
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

        X_train = train[
            selected_features
        ]

        X_valid = valid[
            selected_features
        ]

        y_train = (
            train[
                "CLV_nonfuel"
            ].to_numpy(
                dtype="float64"
            )
        )

        y_valid = (
            valid[
                "CLV_nonfuel"
            ].to_numpy(
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

        raw_prediction = model.predict(
            X_valid,
            num_iteration=
                model.best_iteration,
        )

        prediction = np.maximum(
            raw_prediction,
            0.0,
        )

        capacity_rmse = rmse(
            y_valid,
            prediction,
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

        baseline_rmse = float(
            baseline_row[
                "nonfuel_rmse"
            ]
        )

        # Training score at selected iteration.
        train_prediction = model.predict(
            X_train,
            num_iteration=
                model.best_iteration,
        )

        train_prediction = np.maximum(
            train_prediction,
            0.0,
        )

        train_rmse = rmse(
            y_train,
            train_prediction,
        )

        validation_to_train_ratio = (
            capacity_rmse
            /
            train_rmse
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

                "num_leaves":
                    15,

                "best_iteration":
                    model.best_iteration,

                "train_rmse":
                    float(
                        train_rmse
                    ),

                "capacity_validation_rmse":
                    float(
                        capacity_rmse
                    ),

                "validation_to_train_rmse_ratio":
                    float(
                        validation_to_train_ratio
                    ),

                "baseline_validation_rmse":
                    baseline_rmse,

                "rmse_change":
                    float(
                        capacity_rmse
                        -
                        baseline_rmse
                    ),

                "raw_negative_predictions":
                    int(
                        (
                            raw_prediction
                            < 0
                        ).sum()
                    ),
            }
        )

        pred_frame = valid[
            [
                "snapshot_id",
                "ID",
                "cutoff",
                "CLV_nonfuel",
            ]
        ].copy()

        pred_frame[
            "fold_id"
        ] = fold_id

        pred_frame[
            "capacity_prediction"
        ] = prediction

        prediction_frames.append(
            pred_frame
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
                / f"{fold_id}_nonfuel_leaves15.txt"
            )
        )

        print(
            f"{fold_id} | "
            f"baseline RMSE: "
            f"{baseline_rmse:.6f} | "
            f"leaves15 RMSE: "
            f"{capacity_rmse:.6f} | "
            f"change: "
            f"{capacity_rmse-baseline_rmse:+.6f} | "
            f"train RMSE: "
            f"{train_rmse:.6f}"
        )

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

    weights = (
        metrics[
            "validation_rows"
        ]
        /
        metrics[
            "validation_rows"
        ].sum()
    )

    baseline_weighted_rmse = float(
        np.sum(
            metrics[
                "baseline_validation_rmse"
            ]
            *
            weights
        )
    )

    capacity_weighted_rmse = float(
        np.sum(
            metrics[
                "capacity_validation_rmse"
            ]
            *
            weights
        )
    )

    pooled_capacity_rmse = rmse(
        predictions[
            "CLV_nonfuel"
        ],
        predictions[
            "capacity_prediction"
        ],
    )

    folds_improved = int(
        metrics[
            "capacity_validation_rmse"
        ].lt(
            metrics[
                "baseline_validation_rmse"
            ]
        ).sum()
    )

    summary = pd.DataFrame(
        [
            {
                "baseline_num_leaves":
                    31,

                "experiment_num_leaves":
                    15,

                "folds":
                    len(
                        metrics
                    ),

                "folds_improved":
                    folds_improved,

                "row_weighted_baseline_rmse":
                    baseline_weighted_rmse,

                "row_weighted_capacity_rmse":
                    capacity_weighted_rmse,

                "row_weighted_rmse_change":
                    (
                        capacity_weighted_rmse
                        -
                        baseline_weighted_rmse
                    ),

                "pooled_capacity_rmse":
                    pooled_capacity_rmse,

                "f4_baseline_rmse":
                    float(
                        metrics.loc[
                            metrics[
                                "fold_id"
                            ].eq(
                                "F4"
                            ),
                            "baseline_validation_rmse",
                        ].iloc[0]
                    ),

                "f4_capacity_rmse":
                    float(
                        metrics.loc[
                            metrics[
                                "fold_id"
                            ].eq(
                                "F4"
                            ),
                            "capacity_validation_rmse",
                        ].iloc[0]
                    ),

                "f4_train_rmse":
                    float(
                        metrics.loc[
                            metrics[
                                "fold_id"
                            ].eq(
                                "F4"
                            ),
                            "train_rmse",
                        ].iloc[0]
                    ),

                "f4_validation_to_train_ratio":
                    float(
                        metrics.loc[
                            metrics[
                                "fold_id"
                            ].eq(
                                "F4"
                            ),
                            "validation_to_train_rmse_ratio",
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
        / "capacity_summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "Phase 04I NON-FUEL CAPACITY EXPERIMENT COMPLETE"
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
