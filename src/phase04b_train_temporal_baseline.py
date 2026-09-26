"""
Phase 04B: First LightGBM temporal baseline.

Trains exactly three fixed-methodology models:

1. LightGBM regression -> transformed CLV_fuel
2. LightGBM regression -> transformed CLV_nonfuel
3. LightGBM multiclass -> 17-class Opportunity

Validation uses the frozen Phase 04A expanding-window folds.

No hyperparameter search, ensembling, class reweighting or additional
feature engineering is performed.
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

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
)

SUPERVISED_PATH = (
    DATA_DIR
    / "supervised_historical.csv"
)

FEATURE_MANIFEST_PATH = (
    DATA_DIR
    / "model_feature_manifest.csv"
)

CLASS_MAP_PATH = (
    DATA_DIR
    / "opportunity_class_map.csv"
)

FOLD_MEMBERSHIP_PATH = (
    DATA_DIR
    / "temporal_fold_membership.csv"
)

BASELINE_CONFIG_PATH = (
    ROOT
    / "configs"
    / "lightgbm_baseline.json"
)

OUTPUT_DIR = (
    DATA_DIR
    / "baseline"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "phase04_baseline"
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
    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(
                        y_true,
                        dtype="float64",
                    )
                    -
                    np.asarray(
                        y_pred,
                        dtype="float64",
                    )
                )
                ** 2
            )
        )
    )


def weighted_f1_eval(
    num_classes,
):
    """
    LightGBM custom validation metric.

    Supports current two-dimensional multiclass predictions and
    historical flattened LightGBM prediction format.
    """

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

            probabilities = (
                predictions
            )

        predicted_class = (
            np.argmax(
                probabilities,
                axis=1,
            )
        )

        score = f1_score(
            labels,
            predicted_class,
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
    task,
):
    """
    Construct a fixed deterministic parameter set.
    """

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
            "seed": seed,
            "feature_fraction_seed":
                seed,
            "bagging_seed":
                seed,
            "data_random_seed":
                seed,
        }
    )

    if task in {
        "fuel",
        "nonfuel",
    }:

        parameters.update(
            config[
                "regression"
            ]
        )

        # This is a prediction post-processing
        # rule, not a LightGBM parameter.
        parameters.pop(
            "prediction_floor",
            None,
        )

    elif task == "opportunity":

        parameters.update(
            config[
                "opportunity"
            ]
        )

        # Documentation-only configuration.
        parameters.pop(
            "class_weighting",
            None,
        )

        parameters.pop(
            "decision_rule",
            None,
        )

        parameters.pop(
            "validation_metric",
            None,
        )

    else:

        raise ValueError(
            task
        )

    return parameters


def train_regression(
    X_train,
    y_train,
    X_valid,
    y_valid,
    parameters,
    num_boost_round,
    early_stopping_rounds,
):
    """
    Train one regression model using fixed baseline parameters.
    """

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=list(
            X_train.columns
        ),
        free_raw_data=False,
    )

    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=train_data,
        feature_name=list(
            X_train.columns
        ),
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

    return (
        model,
        raw_prediction,
        prediction,
    )


def train_classifier(
    X_train,
    y_train,
    X_valid,
    y_valid,
    parameters,
    num_boost_round,
    early_stopping_rounds,
    num_classes,
):
    """
    Train the fixed 17-class Opportunity baseline.
    """

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=list(
            X_train.columns
        ),
        free_raw_data=False,
    )

    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=train_data,
        feature_name=list(
            X_train.columns
        ),
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

    probability = model.predict(
        X_valid,
        num_iteration=
            model.best_iteration,
    )

    predicted_class = np.argmax(
        probability,
        axis=1,
    ).astype(
        "int64"
    )

    return (
        model,
        probability,
        predicted_class,
    )


def main():

    print(
        "Loading Phase 04B baseline configuration..."
    )

    config = json.loads(
        BASELINE_CONFIG_PATH.read_text()
    )

    print(
        "Loading supervised historical dataset..."
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
        FEATURE_MANIFEST_PATH
    )

    class_map = pd.read_csv(
        CLASS_MAP_PATH
    )

    fold_membership = pd.read_csv(
        FOLD_MEMBERSHIP_PATH,
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
            "Expected 276 baseline features."
        )

    num_classes = len(
        class_map
    )

    if num_classes != 17:

        raise AssertionError(
            "Expected 17 Opportunity classes."
        )

    class_name_lookup = dict(
        zip(
            class_map[
                "class_id"
            ].astype(int),
            class_map[
                "Opportunity"
            ].astype(str),
        )
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

    fuel_parameters = (
        make_parameters(
            config,
            "fuel",
        )
    )

    nonfuel_parameters = (
        make_parameters(
            config,
            "nonfuel",
        )
    )

    opportunity_parameters = (
        make_parameters(
            config,
            "opportunity",
        )
    )

    folds = sorted(
        fold_membership[
            "fold_id"
        ].unique()
    )

    metric_rows = []
    iteration_rows = []
    prediction_frames = []
    class_metric_rows = []
    importance_rows = []

    # ============================================================
    # TEMPORAL VALIDATION
    # ============================================================

    for fold_id in folds:

        print()
        print(
            "=" * 72
        )

        print(
            f"Training {fold_id}"
        )

        print(
            "=" * 72
        )

        fold = fold_membership[
            fold_membership[
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

        validation_ids = set(
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
                validation_ids
            )
        ].copy()

        if len(
            train
        ) != len(
            train_ids
        ):

            raise AssertionError(
                f"{fold_id} training "
                f"membership mismatch."
            )

        if len(
            valid
        ) != len(
            validation_ids
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

        # ========================================================
        # FUEL
        # ========================================================

        print(
            f"{fold_id}: training fuel regression..."
        )

        (
            fuel_model,
            fuel_raw_prediction,
            fuel_prediction,
        ) = train_regression(
            X_train,
            train[
                "CLV_fuel"
            ].to_numpy(),
            X_valid,
            valid[
                "CLV_fuel"
            ].to_numpy(),
            fuel_parameters,
            num_boost_round,
            early_stopping_rounds,
        )

        fuel_rmse = rmse(
            valid[
                "CLV_fuel"
            ],
            fuel_prediction,
        )

        raw_negative_fuel = int(
            (
                fuel_raw_prediction
                < 0
            ).sum()
        )

        fuel_model.save_model(
            str(
                MODEL_DIR
                / f"{fold_id}_fuel.txt"
            )
        )

        # ========================================================
        # NON-FUEL
        # ========================================================

        print(
            f"{fold_id}: training non-fuel regression..."
        )

        (
            nonfuel_model,
            nonfuel_raw_prediction,
            nonfuel_prediction,
        ) = train_regression(
            X_train,
            train[
                "CLV_nonfuel"
            ].to_numpy(),
            X_valid,
            valid[
                "CLV_nonfuel"
            ].to_numpy(),
            nonfuel_parameters,
            num_boost_round,
            early_stopping_rounds,
        )

        nonfuel_rmse = rmse(
            valid[
                "CLV_nonfuel"
            ],
            nonfuel_prediction,
        )

        raw_negative_nonfuel = int(
            (
                nonfuel_raw_prediction
                < 0
            ).sum()
        )

        nonfuel_model.save_model(
            str(
                MODEL_DIR
                / f"{fold_id}_nonfuel.txt"
            )
        )

        # ========================================================
        # OPPORTUNITY
        # ========================================================

        print(
            f"{fold_id}: training Opportunity classifier..."
        )

        (
            opportunity_model,
            opportunity_probability,
            opportunity_prediction,
        ) = train_classifier(
            X_train,
            train[
                "Opportunity_class_id"
            ].to_numpy(
                dtype="int64"
            ),
            X_valid,
            valid[
                "Opportunity_class_id"
            ].to_numpy(
                dtype="int64"
            ),
            opportunity_parameters,
            num_boost_round,
            early_stopping_rounds,
            num_classes,
        )

        opportunity_f1 = (
            f1_score(
                valid[
                    "Opportunity_class_id"
                ],
                opportunity_prediction,
                average="weighted",
                zero_division=0,
            )
        )

        opportunity_model.save_model(
            str(
                MODEL_DIR
                / f"{fold_id}_opportunity.txt"
            )
        )

        # ========================================================
        # FOLD METRICS
        # ========================================================

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

                "validation_cutoff":
                    valid[
                        "cutoff"
                    ].iloc[0]
                    .date()
                    .isoformat(),

                "fuel_rmse":
                    fuel_rmse,

                "nonfuel_rmse":
                    nonfuel_rmse,

                "opportunity_weighted_f1":
                    float(
                        opportunity_f1
                    ),

                "raw_negative_fuel_predictions":
                    raw_negative_fuel,

                "raw_negative_nonfuel_predictions":
                    raw_negative_nonfuel,
            }
        )

        iteration_rows.extend(
            [
                {
                    "fold_id":
                        fold_id,
                    "task":
                        "fuel",
                    "best_iteration":
                        fuel_model.best_iteration,
                },
                {
                    "fold_id":
                        fold_id,
                    "task":
                        "nonfuel",
                    "best_iteration":
                        nonfuel_model.best_iteration,
                },
                {
                    "fold_id":
                        fold_id,
                    "task":
                        "opportunity",
                    "best_iteration":
                        opportunity_model.best_iteration,
                },
            ]
        )

        # ========================================================
        # OOF PREDICTIONS
        # ========================================================

        prediction_frame = valid[
            [
                "snapshot_id",
                "ID",
                "cutoff",
                "CLV_fuel",
                "CLV_nonfuel",
                "Opportunity",
                "Opportunity_class_id",
            ]
        ].copy()

        prediction_frame[
            "fold_id"
        ] = fold_id

        prediction_frame[
            "pred_CLV_fuel"
        ] = fuel_prediction

        prediction_frame[
            "pred_CLV_nonfuel"
        ] = nonfuel_prediction

        prediction_frame[
            "pred_Opportunity_class_id"
        ] = opportunity_prediction

        prediction_frame[
            "pred_Opportunity"
        ] = [
            class_name_lookup[
                int(value)
            ]
            for value in
            opportunity_prediction
        ]

        prediction_frames.append(
            prediction_frame
        )

        # ========================================================
        # CLASS-LEVEL METRICS
        # ========================================================

        labels = np.arange(
            num_classes
        )

        (
            precision,
            recall,
            f1_values,
            support,
        ) = precision_recall_fscore_support(
            valid[
                "Opportunity_class_id"
            ],
            opportunity_prediction,
            labels=labels,
            zero_division=0,
        )

        for class_id in labels:

            class_metric_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "validation_cutoff":
                        valid[
                            "cutoff"
                        ].iloc[0]
                        .date()
                        .isoformat(),

                    "class_id":
                        int(
                            class_id
                        ),

                    "Opportunity":
                        class_name_lookup[
                            int(
                                class_id
                            )
                        ],

                    "support":
                        int(
                            support[
                                class_id
                            ]
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
        # FEATURE IMPORTANCE
        # ========================================================

        for task, model in [
            (
                "fuel",
                fuel_model,
            ),
            (
                "nonfuel",
                nonfuel_model,
            ),
            (
                "opportunity",
                opportunity_model,
            ),
        ]:

            gains = (
                model.feature_importance(
                    importance_type="gain"
                )
            )

            splits = (
                model.feature_importance(
                    importance_type="split"
                )
            )

            for feature, gain, split in zip(
                selected_features,
                gains,
                splits,
            ):

                importance_rows.append(
                    {
                        "fold_id":
                            fold_id,

                        "task":
                            task,

                        "feature":
                            feature,

                        "gain":
                            float(
                                gain
                            ),

                        "split":
                            int(
                                split
                            ),
                    }
                )

        print(
            f"{fold_id} results | "
            f"Fuel RMSE: "
            f"{fuel_rmse:.6f} | "
            f"Non-fuel RMSE: "
            f"{nonfuel_rmse:.6f} | "
            f"Opportunity weighted F1: "
            f"{opportunity_f1:.6f}"
        )

    # ============================================================
    # COMBINE RESULTS
    # ============================================================

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    best_iterations = pd.DataFrame(
        iteration_rows
    )

    oof = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    class_metrics = pd.DataFrame(
        class_metric_rows
    )

    importance = pd.DataFrame(
        importance_rows
    )

    # ============================================================
    # POOLED OOF METRICS
    # ============================================================

    pooled_fuel_rmse = rmse(
        oof[
            "CLV_fuel"
        ],
        oof[
            "pred_CLV_fuel"
        ],
    )

    pooled_nonfuel_rmse = rmse(
        oof[
            "CLV_nonfuel"
        ],
        oof[
            "pred_CLV_nonfuel"
        ],
    )

    pooled_opportunity_f1 = (
        f1_score(
            oof[
                "Opportunity_class_id"
            ],
            oof[
                "pred_Opportunity_class_id"
            ],
            average="weighted",
            zero_division=0,
        )
    )

    validation_weights = (
        fold_metrics[
            "validation_rows"
        ]
        /
        fold_metrics[
            "validation_rows"
        ].sum()
    )

    weighted_mean_fuel_rmse = float(
        np.sum(
            fold_metrics[
                "fuel_rmse"
            ]
            *
            validation_weights
        )
    )

    weighted_mean_nonfuel_rmse = float(
        np.sum(
            fold_metrics[
                "nonfuel_rmse"
            ]
            *
            validation_weights
        )
    )

    weighted_mean_opportunity_f1 = float(
        np.sum(
            fold_metrics[
                "opportunity_weighted_f1"
            ]
            *
            validation_weights
        )
    )

    summary = pd.DataFrame(
        [
            {
                "validated_rows":
                    len(
                        oof
                    ),

                "folds":
                    len(
                        fold_metrics
                    ),

                "selected_features":
                    len(
                        selected_features
                    ),

                "pooled_oof_fuel_rmse":
                    pooled_fuel_rmse,

                "pooled_oof_nonfuel_rmse":
                    pooled_nonfuel_rmse,

                "pooled_oof_opportunity_weighted_f1":
                    float(
                        pooled_opportunity_f1
                    ),

                "row_weighted_mean_fold_fuel_rmse":
                    weighted_mean_fuel_rmse,

                "row_weighted_mean_fold_nonfuel_rmse":
                    weighted_mean_nonfuel_rmse,

                "row_weighted_mean_fold_opportunity_f1":
                    weighted_mean_opportunity_f1,

                "validation_status":
                    "PASS",
            }
        ]
    )

    # ============================================================
    # SAVE
    # ============================================================

    fold_metrics.to_csv(
        OUTPUT_DIR
        / "fold_metrics.csv",
        index=False,
    )

    best_iterations.to_csv(
        OUTPUT_DIR
        / "best_iterations.csv",
        index=False,
    )

    oof.to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
        float_format="%.10g",
    )

    class_metrics.to_csv(
        OUTPUT_DIR
        / "opportunity_class_metrics.csv",
        index=False,
    )

    importance.to_csv(
        OUTPUT_DIR
        / "feature_importance_by_fold.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "baseline_summary.csv",
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
        "PHASE 04B LIGHTGBM TEMPORAL BASELINE COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "FOLD METRICS"
    )

    print()

    print(
        fold_metrics.to_string(
            index=False
        )
    )

    print()

    print(
        "BEST ITERATIONS"
    )

    print()

    print(
        best_iterations.to_string(
            index=False
        )
    )

    print()

    print(
        "BASELINE SUMMARY"
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
