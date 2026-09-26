"""
Phase 04E: Two-part non-fuel expected-value experiment.

Hypothesis:
Direct squared-error regression shrinks a strongly zero-inflated
non-fuel target toward intermediate values.

Two-part representation:

    P(Y > 0 | X)
        *
    E[Y | Y > 0, X]

where Y is official transformed CLV_nonfuel.

Only the non-fuel modelling formulation changes.

Fixed:
- 276 Phase 03 predictors
- four expanding-window temporal folds
- LightGBM algorithm family
- common tree parameters
- deterministic random seed
- no hyperparameter search
"""

from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


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

FOLD_MEMBERSHIP_PATH = (
    PHASE04_DIR
    / "temporal_fold_membership.csv"
)

BASELINE_OOF_PATH = (
    PHASE04_DIR
    / "baseline"
    / "oof_predictions.csv"
)

BASELINE_CONFIG_PATH = (
    ROOT
    / "configs"
    / "lightgbm_baseline.json"
)

OUTPUT_DIR = (
    PHASE04_DIR
    / "nonfuel_twopart"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "phase04_nonfuel_twopart"
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


def make_common_parameters(
    config,
):

    parameters = dict(
        config[
            "common_parameters"
        ]
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


def main():

    print(
        "Loading Phase 04E inputs..."
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

    membership = pd.read_csv(
        FOLD_MEMBERSHIP_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    baseline_oof = pd.read_csv(
        BASELINE_OOF_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
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

    common = make_common_parameters(
        config
    )

    occurrence_parameters = dict(
        common
    )

    occurrence_parameters.update(
        {
            "objective":
                "binary",

            # Calibration matters because the probability
            # is multiplied directly by positive magnitude.
            "metric":
                "binary_logloss",
        }
    )

    magnitude_parameters = dict(
        common
    )

    magnitude_parameters.update(
        config[
            "regression"
        ]
    )

    magnitude_parameters.pop(
        "prediction_floor",
        None,
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

    # ============================================================
    # TEMPORAL EXPERIMENT
    # ============================================================

    for fold_id in folds:

        print()
        print(
            "=" * 72
        )

        print(
            f"Training non-fuel two-part model: "
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

        if (
            len(train)
            != len(train_ids)
        ):

            raise AssertionError(
                f"{fold_id} training "
                f"membership mismatch."
            )

        if (
            len(valid)
            != len(valid_ids)
        ):

            raise AssertionError(
                f"{fold_id} validation "
                f"membership mismatch."
            )

        X_train = train[
            selected_features
        ]

        X_valid = valid[
            selected_features
        ]

        y_train = (
            train[
                "CLV_nonfuel"
            ]
            .to_numpy(
                dtype="float64"
            )
        )

        y_valid = (
            valid[
                "CLV_nonfuel"
            ]
            .to_numpy(
                dtype="float64"
            )
        )

        occurrence_train = (
            y_train > 0
        ).astype(
            "int8"
        )

        occurrence_valid = (
            y_valid > 0
        ).astype(
            "int8"
        )

        # ========================================================
        # PART A: OCCURRENCE
        # ========================================================

        train_occurrence = lgb.Dataset(
            X_train,
            label=
                occurrence_train,
            feature_name=
                selected_features,
            free_raw_data=False,
        )

        valid_occurrence = lgb.Dataset(
            X_valid,
            label=
                occurrence_valid,
            reference=
                train_occurrence,
            feature_name=
                selected_features,
            free_raw_data=False,
        )

        occurrence_model = lgb.train(
            occurrence_parameters,
            train_occurrence,
            num_boost_round=
                num_boost_round,
            valid_sets=[
                valid_occurrence
            ],
            valid_names=[
                "validation"
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

        positive_probability = (
            occurrence_model.predict(
                X_valid,
                num_iteration=
                    occurrence_model.best_iteration,
            )
        )

        occurrence_auc = (
            roc_auc_score(
                occurrence_valid,
                positive_probability,
            )
        )

        # ========================================================
        # PART B: POSITIVE MAGNITUDE
        # ========================================================

        positive_train_mask = (
            y_train > 0
        )

        positive_valid_mask = (
            y_valid > 0
        )

        positive_train = (
            train.loc[
                positive_train_mask
            ]
        )

        positive_valid = (
            valid.loc[
                positive_valid_mask
            ]
        )

        X_train_positive = (
            positive_train[
                selected_features
            ]
        )

        X_valid_positive = (
            positive_valid[
                selected_features
            ]
        )

        y_train_positive = (
            positive_train[
                "CLV_nonfuel"
            ].to_numpy(
                dtype="float64"
            )
        )

        y_valid_positive = (
            positive_valid[
                "CLV_nonfuel"
            ].to_numpy(
                dtype="float64"
            )
        )

        train_magnitude = lgb.Dataset(
            X_train_positive,
            label=
                y_train_positive,
            feature_name=
                selected_features,
            free_raw_data=False,
        )

        valid_magnitude = lgb.Dataset(
            X_valid_positive,
            label=
                y_valid_positive,
            reference=
                train_magnitude,
            feature_name=
                selected_features,
            free_raw_data=False,
        )

        magnitude_model = lgb.train(
            magnitude_parameters,
            train_magnitude,
            num_boost_round=
                num_boost_round,
            valid_sets=[
                valid_magnitude
            ],
            valid_names=[
                "validation"
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

        # Magnitude estimate is required for ALL validation rows,
        # because it represents E[Y | Y > 0, X].
        magnitude_prediction_raw = (
            magnitude_model.predict(
                X_valid,
                num_iteration=
                    magnitude_model.best_iteration,
            )
        )

        magnitude_prediction = np.maximum(
            magnitude_prediction_raw,
            0.0,
        )

        # ========================================================
        # COMBINED EXPECTED VALUE
        # ========================================================

        twopart_prediction = (
            positive_probability
            *
            magnitude_prediction
        )

        twopart_rmse = rmse(
            y_valid,
            twopart_prediction,
        )

        positive_magnitude_rmse = (
            rmse(
                y_valid[
                    positive_valid_mask
                ],
                magnitude_prediction[
                    positive_valid_mask
                ],
            )
        )

        # Baseline predictions for the exact same validation rows.
        baseline_fold = (
            baseline_oof[
                baseline_oof[
                    "fold_id"
                ].eq(
                    fold_id
                )
            ][
                [
                    "snapshot_id",
                    "pred_CLV_nonfuel",
                ]
            ]
        )

        comparison = (
            valid[
                [
                    "snapshot_id"
                ]
            ]
            .merge(
                baseline_fold,
                on="snapshot_id",
                how="left",
                validate="one_to_one",
            )
        )

        if comparison[
            "pred_CLV_nonfuel"
        ].isna().any():

            raise AssertionError(
                f"{fold_id} baseline "
                f"predictions missing."
            )

        baseline_prediction = (
            comparison[
                "pred_CLV_nonfuel"
            ].to_numpy(
                dtype="float64"
            )
        )

        baseline_rmse = rmse(
            y_valid,
            baseline_prediction,
        )

        zero_mask = (
            y_valid == 0
        )

        positive_mask = (
            y_valid > 0
        )

        metric_rows.append(
            {
                "fold_id":
                    fold_id,

                "train_rows":
                    len(
                        train
                    ),

                "positive_train_rows":
                    int(
                        positive_train_mask.sum()
                    ),

                "validation_rows":
                    len(
                        valid
                    ),

                "positive_validation_rows":
                    int(
                        positive_valid_mask.sum()
                    ),

                "occurrence_best_iteration":
                    occurrence_model.best_iteration,

                "magnitude_best_iteration":
                    magnitude_model.best_iteration,

                "occurrence_auc":
                    float(
                        occurrence_auc
                    ),

                "positive_magnitude_rmse":
                    float(
                        positive_magnitude_rmse
                    ),

                "baseline_rmse":
                    float(
                        baseline_rmse
                    ),

                "twopart_rmse":
                    float(
                        twopart_rmse
                    ),

                "rmse_change":
                    float(
                        twopart_rmse
                        -
                        baseline_rmse
                    ),

                "baseline_zero_rmse":
                    rmse(
                        y_valid[
                            zero_mask
                        ],
                        baseline_prediction[
                            zero_mask
                        ],
                    ),

                "twopart_zero_rmse":
                    rmse(
                        y_valid[
                            zero_mask
                        ],
                        twopart_prediction[
                            zero_mask
                        ],
                    ),

                "baseline_positive_rmse":
                    rmse(
                        y_valid[
                            positive_mask
                        ],
                        baseline_prediction[
                            positive_mask
                        ],
                    ),

                "twopart_positive_rmse":
                    rmse(
                        y_valid[
                            positive_mask
                        ],
                        twopart_prediction[
                            positive_mask
                        ],
                    ),
            }
        )

        prediction_frame = valid[
            [
                "snapshot_id",
                "ID",
                "cutoff",
                "CLV_nonfuel",
            ]
        ].copy()

        prediction_frame[
            "fold_id"
        ] = fold_id

        prediction_frame[
            "baseline_prediction"
        ] = baseline_prediction

        prediction_frame[
            "positive_probability"
        ] = positive_probability

        prediction_frame[
            "conditional_positive_prediction"
        ] = magnitude_prediction

        prediction_frame[
            "twopart_prediction"
        ] = twopart_prediction

        prediction_frames.append(
            prediction_frame
        )

        # ========================================================
        # IMPORTANCE
        # ========================================================

        for component, model in [
            (
                "occurrence",
                occurrence_model,
            ),
            (
                "positive_magnitude",
                magnitude_model,
            ),
        ]:

            gain = (
                model.feature_importance(
                    importance_type="gain"
                )
            )

            split = (
                model.feature_importance(
                    importance_type="split"
                )
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

                        "component":
                            component,

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

        occurrence_model.save_model(
            str(
                MODEL_DIR
                /
                f"{fold_id}_nonfuel_occurrence.txt"
            )
        )

        magnitude_model.save_model(
            str(
                MODEL_DIR
                /
                f"{fold_id}_nonfuel_positive_magnitude.txt"
            )
        )

        print(
            f"{fold_id} | "
            f"baseline RMSE: "
            f"{baseline_rmse:.6f} | "
            f"two-part RMSE: "
            f"{twopart_rmse:.6f} | "
            f"change: "
            f"{twopart_rmse-baseline_rmse:+.6f} | "
            f"occurrence AUC: "
            f"{occurrence_auc:.6f}"
        )

    # ============================================================
    # SUMMARY
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

    pooled_baseline_rmse = rmse(
        predictions[
            "CLV_nonfuel"
        ],
        predictions[
            "baseline_prediction"
        ],
    )

    pooled_twopart_rmse = rmse(
        predictions[
            "CLV_nonfuel"
        ],
        predictions[
            "twopart_prediction"
        ],
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

    weighted_baseline_rmse = float(
        np.sum(
            metrics[
                "baseline_rmse"
            ]
            *
            validation_weights
        )
    )

    weighted_twopart_rmse = float(
        np.sum(
            metrics[
                "twopart_rmse"
            ]
            *
            validation_weights
        )
    )

    folds_improved = int(
        metrics[
            "twopart_rmse"
        ].lt(
            metrics[
                "baseline_rmse"
            ]
        ).sum()
    )

    summary = pd.DataFrame(
        [
            {
                "folds":
                    len(
                        metrics
                    ),

                "folds_improved":
                    folds_improved,

                "pooled_baseline_rmse":
                    pooled_baseline_rmse,

                "pooled_twopart_rmse":
                    pooled_twopart_rmse,

                "pooled_rmse_change":
                    (
                        pooled_twopart_rmse
                        -
                        pooled_baseline_rmse
                    ),

                "row_weighted_baseline_rmse":
                    weighted_baseline_rmse,

                "row_weighted_twopart_rmse":
                    weighted_twopart_rmse,

                "row_weighted_rmse_change":
                    (
                        weighted_twopart_rmse
                        -
                        weighted_baseline_rmse
                    ),

                "f4_baseline_rmse":
                    float(
                        metrics.loc[
                            metrics[
                                "fold_id"
                            ].eq(
                                "F4"
                            ),
                            "baseline_rmse",
                        ].iloc[0]
                    ),

                "f4_twopart_rmse":
                    float(
                        metrics.loc[
                            metrics[
                                "fold_id"
                            ].eq(
                                "F4"
                            ),
                            "twopart_rmse",
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
        / "component_feature_importance.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "twopart_summary.csv",
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
        "PHASE 04E NON-FUEL TWO-PART EXPERIMENT COMPLETE"
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
