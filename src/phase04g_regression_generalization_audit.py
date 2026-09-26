"""
Phase 04G: Regression generalization-gap audit.

No model is trained.

The saved Phase 04B direct LightGBM regression models are applied to
their original training and validation populations.

Purpose:
- quantify train-versus-validation RMSE;
- determine whether current regression capacity appears strongly
  overfit, broadly balanced, or potentially underfit;
- inspect zero/positive error gaps;
- verify saved-model predictions reproduce Phase 04B OOF outputs.

This audit is performed before controlled LightGBM parameter tuning.
"""

from pathlib import Path

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

OOF_PATH = (
    PHASE04_DIR
    / "baseline"
    / "oof_predictions.csv"
)

ITERATION_PATH = (
    PHASE04_DIR
    / "baseline"
    / "best_iterations.csv"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "phase04_baseline"
)

OUTPUT_DIR = (
    PHASE04_DIR
    / "regression_generalization"
)

OUTPUT_DIR.mkdir(
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


def metric_block(
    y,
    prediction,
):

    y = np.asarray(
        y,
        dtype="float64",
    )

    prediction = np.asarray(
        prediction,
        dtype="float64",
    )

    zero = (
        y == 0
    )

    positive = (
        y > 0
    )

    residual = (
        prediction
        -
        y
    )

    return {
        "rows":
            len(
                y
            ),

        "target_mean":
            float(
                y.mean()
            ),

        "prediction_mean":
            float(
                prediction.mean()
            ),

        "rmse":
            rmse(
                y,
                prediction,
            ),

        "mean_residual":
            float(
                residual.mean()
            ),

        "zero_rate":
            float(
                zero.mean()
            ),

        "zero_rmse":
            (
                rmse(
                    y[
                        zero
                    ],
                    prediction[
                        zero
                    ],
                )
                if zero.any()
                else np.nan
            ),

        "positive_rmse":
            (
                rmse(
                    y[
                        positive
                    ],
                    prediction[
                        positive
                    ],
                )
                if positive.any()
                else np.nan
            ),
    }


def main():

    print(
        "Loading Phase 04 regression audit inputs..."
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

    oof = pd.read_csv(
        OOF_PATH,
        dtype={
            "ID": str,
        },
        parse_dates=[
            "cutoff",
        ],
    )

    iterations = pd.read_csv(
        ITERATION_PATH
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

    audit_rows = []

    # ============================================================
    # APPLY SAVED BASELINE MODELS
    # ============================================================

    for fold_id in sorted(
        membership[
            "fold_id"
        ].unique()
    ):

        fold_membership = (
            membership[
                membership[
                    "fold_id"
                ].eq(
                    fold_id
                )
            ]
        )

        train_ids = set(
            fold_membership.loc[
                fold_membership[
                    "role"
                ].eq(
                    "train"
                ),
                "snapshot_id",
            ]
        )

        valid_ids = set(
            fold_membership.loc[
                fold_membership[
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

        for (
            task,
            target,
            oof_prediction_column,
        ) in [
            (
                "fuel",
                "CLV_fuel",
                "pred_CLV_fuel",
            ),
            (
                "nonfuel",
                "CLV_nonfuel",
                "pred_CLV_nonfuel",
            ),
        ]:

            print(
                f"Auditing {fold_id} {task}..."
            )

            best_iteration = int(
                iterations.loc[
                    (
                        iterations[
                            "fold_id"
                        ].eq(
                            fold_id
                        )
                        &
                        iterations[
                            "task"
                        ].eq(
                            task
                        )
                    ),
                    "best_iteration",
                ].iloc[0]
            )

            model_path = (
                MODEL_DIR
                / f"{fold_id}_{task}.txt"
            )

            if not model_path.exists():

                raise FileNotFoundError(
                    model_path
                )

            model = lgb.Booster(
                model_file=str(
                    model_path
                )
            )

            train_prediction = (
                model.predict(
                    train[
                        selected_features
                    ],
                    num_iteration=
                        best_iteration,
                )
            )

            valid_prediction = (
                model.predict(
                    valid[
                        selected_features
                    ],
                    num_iteration=
                        best_iteration,
                )
            )

            # Match Phase 04B post-processing.
            train_prediction = (
                np.maximum(
                    train_prediction,
                    0.0,
                )
            )

            valid_prediction = (
                np.maximum(
                    valid_prediction,
                    0.0,
                )
            )

            # ----------------------------------------------------
            # REPRODUCIBILITY CHECK AGAINST STORED OOF
            # ----------------------------------------------------

            validation_check = (
                valid[
                    [
                        "snapshot_id",
                    ]
                ]
                .copy()
            )

            validation_check[
                "reconstructed_prediction"
            ] = (
                valid_prediction
            )

            validation_check = (
                validation_check
                .merge(
                    oof.loc[
                        oof[
                            "fold_id"
                        ].eq(
                            fold_id
                        ),
                        [
                            "snapshot_id",
                            oof_prediction_column,
                        ],
                    ],
                    on="snapshot_id",
                    how="left",
                    validate="one_to_one",
                )
            )

            if validation_check[
                oof_prediction_column
            ].isna().any():

                raise AssertionError(
                    f"{fold_id} {task}: "
                    f"stored OOF prediction missing."
                )

            max_oof_difference = float(
                np.max(
                    np.abs(
                        validation_check[
                            "reconstructed_prediction"
                        ].to_numpy(
                            dtype="float64"
                        )
                        -
                        validation_check[
                            oof_prediction_column
                        ].to_numpy(
                            dtype="float64"
                        )
                    )
                )
            )

            if max_oof_difference > 1e-6:

                raise AssertionError(
                    f"{fold_id} {task}: "
                    f"saved model does not reproduce "
                    f"OOF predictions. "
                    f"Max difference = "
                    f"{max_oof_difference}"
                )

            train_metrics = metric_block(
                train[
                    target
                ],
                train_prediction,
            )

            valid_metrics = metric_block(
                valid[
                    target
                ],
                valid_prediction,
            )

            train_rmse = (
                train_metrics[
                    "rmse"
                ]
            )

            valid_rmse = (
                valid_metrics[
                    "rmse"
                ]
            )

            audit_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "task":
                        task,

                    "best_iteration":
                        best_iteration,

                    "train_rows":
                        train_metrics[
                            "rows"
                        ],

                    "validation_rows":
                        valid_metrics[
                            "rows"
                        ],

                    "train_rmse":
                        train_rmse,

                    "validation_rmse":
                        valid_rmse,

                    "absolute_generalization_gap":
                        (
                            valid_rmse
                            -
                            train_rmse
                        ),

                    "validation_to_train_rmse_ratio":
                        (
                            valid_rmse
                            /
                            train_rmse
                            if train_rmse > 0
                            else np.nan
                        ),

                    "train_target_mean":
                        train_metrics[
                            "target_mean"
                        ],

                    "validation_target_mean":
                        valid_metrics[
                            "target_mean"
                        ],

                    "train_prediction_mean":
                        train_metrics[
                            "prediction_mean"
                        ],

                    "validation_prediction_mean":
                        valid_metrics[
                            "prediction_mean"
                        ],

                    "train_mean_residual":
                        train_metrics[
                            "mean_residual"
                        ],

                    "validation_mean_residual":
                        valid_metrics[
                            "mean_residual"
                        ],

                    "train_zero_rate":
                        train_metrics[
                            "zero_rate"
                        ],

                    "validation_zero_rate":
                        valid_metrics[
                            "zero_rate"
                        ],

                    "train_zero_rmse":
                        train_metrics[
                            "zero_rmse"
                        ],

                    "validation_zero_rmse":
                        valid_metrics[
                            "zero_rmse"
                        ],

                    "train_positive_rmse":
                        train_metrics[
                            "positive_rmse"
                        ],

                    "validation_positive_rmse":
                        valid_metrics[
                            "positive_rmse"
                        ],

                    "max_oof_reproduction_difference":
                        max_oof_difference,

                    "validation_status":
                        "PASS",
                }
            )

    audit = pd.DataFrame(
        audit_rows
    )

    # ============================================================
    # TASK SUMMARY
    # ============================================================

    summary_rows = []

    for task, block in audit.groupby(
        "task"
    ):

        summary_rows.append(
            {
                "task":
                    task,

                "mean_train_rmse":
                    float(
                        block[
                            "train_rmse"
                        ].mean()
                    ),

                "mean_validation_rmse":
                    float(
                        block[
                            "validation_rmse"
                        ].mean()
                    ),

                "mean_absolute_generalization_gap":
                    float(
                        block[
                            "absolute_generalization_gap"
                        ].mean()
                    ),

                "mean_validation_to_train_rmse_ratio":
                    float(
                        block[
                            "validation_to_train_rmse_ratio"
                        ].mean()
                    ),

                "latest_fold_gap":
                    float(
                        block.loc[
                            block[
                                "fold_id"
                            ].eq(
                                "F4"
                            ),
                            "absolute_generalization_gap",
                        ].iloc[0]
                    ),

                "latest_fold_ratio":
                    float(
                        block.loc[
                            block[
                                "fold_id"
                            ].eq(
                                "F4"
                            ),
                            "validation_to_train_rmse_ratio",
                        ].iloc[0]
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    audit.to_csv(
        OUTPUT_DIR
        / "regression_generalization_by_fold.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "regression_generalization_summary.csv",
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
        "PHASE 04G REGRESSION GENERALIZATION AUDIT COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "TRAIN VS VALIDATION"
    )

    print()

    print(
        audit[
            [
                "fold_id",
                "task",
                "best_iteration",
                "train_rows",
                "validation_rows",
                "train_rmse",
                "validation_rmse",
                "absolute_generalization_gap",
                "validation_to_train_rmse_ratio",
                "train_zero_rate",
                "validation_zero_rate",
                "train_zero_rmse",
                "validation_zero_rmse",
                "train_positive_rmse",
                "validation_positive_rmse",
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
