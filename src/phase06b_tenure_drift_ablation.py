"""
Phase 06B: observed_tenure_days drift-feature ablation.

Exactly one feature is removed:

    observed_tenure_days

Control models:
- Fuel: accepted Phase 06A recency-weighted regression
- Non-fuel: accepted Phase 06A recency-weighted regression
- Opportunity: accepted Phase 04D alpha=0.50 classifier, no recency weighting

Validation:
    F2, F3, F4

No structural parameters are changed.
"""

from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]

PHASE04_DIR = ROOT / "data" / "processed" / "phase04"
PHASE06_DIR = ROOT / "data" / "processed" / "phase06"

SUPERVISED_PATH = PHASE04_DIR / "supervised_historical.csv"
FEATURE_MANIFEST_PATH = PHASE04_DIR / "model_feature_manifest.csv"
FOLD_MEMBERSHIP_PATH = PHASE04_DIR / "temporal_fold_membership.csv"
CLASS_MAP_PATH = PHASE04_DIR / "opportunity_class_map.csv"

BASELINE_CONFIG_PATH = ROOT / "configs" / "lightgbm_baseline.json"
FINAL_SPEC_PATH = ROOT / "configs" / "final_model_spec.json"

PHASE06A_SUMMARY_PATH = (
    PHASE06_DIR
    / "recency_weighting"
    / "fold_comparison.csv"
)

OPPORTUNITY_ACCEPTED_PATH = (
    PHASE04_DIR
    / "opportunity_weighting"
    / "fold_comparison.csv"
)

OUTPUT_DIR = PHASE06_DIR / "tenure_drift_ablation"
MODEL_DIR = ROOT / "models" / "phase06_tenure_drift_ablation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DROPPED_FEATURE = "observed_tenure_days"
HALF_LIFE_QUARTERS = 2.0


def rmse(y_true, y_pred):

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
                (y_true - y_pred) ** 2
            )
        )
    )


def weighted_f1_eval(num_classes):

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


def normalize_weights(weights):

    weights = np.asarray(
        weights,
        dtype="float64",
    )

    mean = weights.mean()

    if (
        not np.isfinite(mean)
        or mean <= 0
    ):

        raise AssertionError(
            "Invalid sample weights."
        )

    normalized = (
        weights
        /
        mean
    )

    if not np.isclose(
        normalized.mean(),
        1.0,
        atol=1e-12,
    ):

        raise AssertionError(
            "Weights do not normalize to mean one."
        )

    return normalized


def temporal_weights(cutoffs):

    dates = pd.to_datetime(
        cutoffs
    )

    origins = sorted(
        dates.unique()
    )

    origin_rank = {
        origin: index
        for index, origin
        in enumerate(origins)
    }

    newest_rank = (
        len(origins)
        - 1
    )

    quarter_lags = np.asarray(
        [
            newest_rank
            - origin_rank[
                value
            ]
            for value in dates
        ],
        dtype="float64",
    )

    raw_weights = (
        2.0
        **
        (
            -quarter_lags
            /
            HALF_LIFE_QUARTERS
        )
    )

    return normalize_weights(
        raw_weights
    )


def opportunity_class_component(
    labels,
    num_classes,
    exponent,
):

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
            "Opportunity training fold "
            "is missing at least one class."
        )

    n = len(
        labels
    )

    class_weights = (
        n
        /
        (
            num_classes
            * counts
        )
    ) ** float(
        exponent
    )

    return class_weights[
        labels
    ]


def common_parameters(
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


def regression_parameters(
    config,
    task_spec,
):

    parameters = common_parameters(
        config
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

    parameters[
        "num_leaves"
    ] = int(
        task_spec[
            "num_leaves"
        ]
    )

    parameters[
        "min_data_in_leaf"
    ] = int(
        task_spec[
            "min_data_in_leaf"
        ]
    )

    return parameters


def opportunity_parameters(
    config,
    task_spec,
):

    parameters = common_parameters(
        config
    )

    parameters.update(
        config[
            "opportunity"
        ]
    )

    for key in [
        "class_weighting",
        "decision_rule",
        "validation_metric",
    ]:

        parameters.pop(
            key,
            None,
        )

    parameters[
        "num_leaves"
    ] = int(
        task_spec[
            "num_leaves"
        ]
    )

    parameters[
        "min_data_in_leaf"
    ] = int(
        task_spec[
            "min_data_in_leaf"
        ]
    )

    parameters[
        "num_class"
    ] = int(
        task_spec[
            "num_class"
        ]
    )

    parameters[
        "metric"
    ] = "None"

    return parameters


def control_metric_lookup():

    phase06a = pd.read_csv(
        PHASE06A_SUMMARY_PATH
    )

    opportunity = pd.read_csv(
        OPPORTUNITY_ACCEPTED_PATH
    )

    lookup = {
        "fuel": {},
        "nonfuel": {},
        "opportunity": {},
    }

    for task in [
        "fuel",
        "nonfuel",
    ]:

        block = phase06a[
            phase06a[
                "task"
            ].eq(
                task
            )
        ]

        for _, row in block.iterrows():

            lookup[
                task
            ][
                row[
                    "fold_id"
                ]
            ] = float(
                row[
                    "experiment_metric"
                ]
            )

    for _, row in opportunity.iterrows():

        fold_id = row[
            "fold_id"
        ]

        if fold_id in [
            "F2",
            "F3",
            "F4",
        ]:

            lookup[
                "opportunity"
            ][
                fold_id
            ] = float(
                row[
                    "weighted_model_f1"
                ]
            )

    return lookup


def main():

    print(
        "Loading Phase 06B inputs..."
    )

    config = json.loads(
        BASELINE_CONFIG_PATH.read_text()
    )

    spec = json.loads(
        FINAL_SPEC_PATH.read_text()
    )

    if spec[
        "status"
    ] != "FROZEN":

        raise AssertionError(
            "Final Phase 04 specification "
            "is not frozen."
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

    class_map = pd.read_csv(
        CLASS_MAP_PATH
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
            "Expected 276 accepted predictors."
        )

    if DROPPED_FEATURE not in selected_features:

        raise AssertionError(
            f"{DROPPED_FEATURE} is not "
            "in the accepted feature set."
        )

    ablated_features = [
        feature
        for feature
        in selected_features
        if feature
        != DROPPED_FEATURE
    ]

    if len(
        ablated_features
    ) != 275:

        raise AssertionError(
            "Expected exactly 275 predictors "
            "after tenure ablation."
        )

    num_classes = len(
        class_map
    )

    if num_classes != 17:

        raise AssertionError(
            "Expected 17 Opportunity classes."
        )

    control_lookup = (
        control_metric_lookup()
    )

    folds = [
        "F2",
        "F3",
        "F4",
    ]

    metric_rows = []

    # ============================================================
    # TEMPORAL VALIDATION
    # ============================================================

    for fold_id in folds:

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
                f"{fold_id}: train membership mismatch."
            )

        if len(
            valid
        ) != len(
            valid_ids
        ):

            raise AssertionError(
                f"{fold_id}: validation membership mismatch."
            )

        X_train = train[
            ablated_features
        ]

        X_valid = valid[
            ablated_features
        ]

        recency_weight = temporal_weights(
            train[
                "cutoff"
            ]
        )

        # ========================================================
        # FUEL
        # ========================================================

        print()
        print(
            "=" * 72
        )

        print(
            f"Training Phase 06B fuel: "
            f"{fold_id}"
        )

        print(
            "=" * 72
        )

        fuel_spec = spec[
            "fuel"
        ]

        fuel_parameters = (
            regression_parameters(
                config,
                fuel_spec,
            )
        )

        y_train = train[
            "CLV_fuel"
        ].to_numpy(
            dtype="float64"
        )

        y_valid = valid[
            "CLV_fuel"
        ].to_numpy(
            dtype="float64"
        )

        train_dataset = lgb.Dataset(
            X_train,
            label=y_train,
            weight=recency_weight,
            feature_name=
                ablated_features,
            free_raw_data=False,
        )

        valid_dataset = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_dataset,
            feature_name=
                ablated_features,
            free_raw_data=False,
        )

        model = lgb.train(
            fuel_parameters,
            train_dataset,
            num_boost_round=int(
                config[
                    "num_boost_round"
                ]
            ),
            valid_sets=[
                valid_dataset,
            ],
            valid_names=[
                "validation",
            ],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=int(
                        config[
                            "early_stopping_rounds"
                        ]
                    ),
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
            float(
                fuel_spec[
                    "prediction_floor"
                ]
            ),
        )

        experiment_metric = rmse(
            y_valid,
            prediction,
        )

        control_metric = (
            control_lookup[
                "fuel"
            ][
                fold_id
            ]
        )

        metric_rows.append(
            {
                "task":
                    "fuel",

                "fold_id":
                    fold_id,

                "features":
                    275,

                "dropped_feature":
                    DROPPED_FEATURE,

                "best_iteration":
                    int(
                        model.best_iteration
                    ),

                "validation_rows":
                    len(valid),

                "control_metric":
                    control_metric,

                "ablation_metric":
                    experiment_metric,

                "metric_change":
                    (
                        experiment_metric
                        -
                        control_metric
                    ),

                "improved":
                    int(
                        experiment_metric
                        <
                        control_metric
                    ),
            }
        )

        model.save_model(
            str(
                MODEL_DIR
                /
                f"{fold_id}_fuel_no_tenure.txt"
            )
        )

        print(
            f"{fold_id} | control RMSE: "
            f"{control_metric:.6f} | "
            f"no-tenure RMSE: "
            f"{experiment_metric:.6f} | "
            f"change: "
            f"{experiment_metric-control_metric:+.6f}"
        )

        # ========================================================
        # NON-FUEL
        # ========================================================

        print()
        print(
            "=" * 72
        )

        print(
            f"Training Phase 06B nonfuel: "
            f"{fold_id}"
        )

        print(
            "=" * 72
        )

        nonfuel_spec = spec[
            "nonfuel"
        ]

        nonfuel_parameters = (
            regression_parameters(
                config,
                nonfuel_spec,
            )
        )

        y_train = train[
            "CLV_nonfuel"
        ].to_numpy(
            dtype="float64"
        )

        y_valid = valid[
            "CLV_nonfuel"
        ].to_numpy(
            dtype="float64"
        )

        train_dataset = lgb.Dataset(
            X_train,
            label=y_train,
            weight=recency_weight,
            feature_name=
                ablated_features,
            free_raw_data=False,
        )

        valid_dataset = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_dataset,
            feature_name=
                ablated_features,
            free_raw_data=False,
        )

        model = lgb.train(
            nonfuel_parameters,
            train_dataset,
            num_boost_round=int(
                config[
                    "num_boost_round"
                ]
            ),
            valid_sets=[
                valid_dataset,
            ],
            valid_names=[
                "validation",
            ],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=int(
                        config[
                            "early_stopping_rounds"
                        ]
                    ),
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
            float(
                nonfuel_spec[
                    "prediction_floor"
                ]
            ),
        )

        experiment_metric = rmse(
            y_valid,
            prediction,
        )

        control_metric = (
            control_lookup[
                "nonfuel"
            ][
                fold_id
            ]
        )

        metric_rows.append(
            {
                "task":
                    "nonfuel",

                "fold_id":
                    fold_id,

                "features":
                    275,

                "dropped_feature":
                    DROPPED_FEATURE,

                "best_iteration":
                    int(
                        model.best_iteration
                    ),

                "validation_rows":
                    len(valid),

                "control_metric":
                    control_metric,

                "ablation_metric":
                    experiment_metric,

                "metric_change":
                    (
                        experiment_metric
                        -
                        control_metric
                    ),

                "improved":
                    int(
                        experiment_metric
                        <
                        control_metric
                    ),
            }
        )

        model.save_model(
            str(
                MODEL_DIR
                /
                f"{fold_id}_nonfuel_no_tenure.txt"
            )
        )

        print(
            f"{fold_id} | control RMSE: "
            f"{control_metric:.6f} | "
            f"no-tenure RMSE: "
            f"{experiment_metric:.6f} | "
            f"change: "
            f"{experiment_metric-control_metric:+.6f}"
        )

        # ========================================================
        # OPPORTUNITY
        # ========================================================

        print()
        print(
            "=" * 72
        )

        print(
            f"Training Phase 06B Opportunity: "
            f"{fold_id}"
        )

        print(
            "=" * 72
        )

        opportunity_spec = spec[
            "opportunity"
        ]

        opportunity_params = (
            opportunity_parameters(
                config,
                opportunity_spec,
            )
        )

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

        class_component = (
            opportunity_class_component(
                y_train,
                num_classes,
                float(
                    opportunity_spec[
                        "class_weight_exponent"
                    ]
                ),
            )
        )

        # Opportunity retains the accepted
        # non-temporal Phase 04D weighting.
        opportunity_weight = (
            normalize_weights(
                class_component
            )
        )

        train_dataset = lgb.Dataset(
            X_train,
            label=y_train,
            weight=
                opportunity_weight,
            feature_name=
                ablated_features,
            free_raw_data=False,
        )

        valid_dataset = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_dataset,
            feature_name=
                ablated_features,
            free_raw_data=False,
        )

        model = lgb.train(
            opportunity_params,
            train_dataset,
            num_boost_round=int(
                config[
                    "num_boost_round"
                ]
            ),
            valid_sets=[
                valid_dataset,
            ],
            valid_names=[
                "validation",
            ],
            feval=
                weighted_f1_eval(
                    num_classes
                ),
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=int(
                        config[
                            "early_stopping_rounds"
                        ]
                    ),
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
        )

        experiment_metric = f1_score(
            y_valid,
            prediction,
            average="weighted",
            zero_division=0,
        )

        control_metric = (
            control_lookup[
                "opportunity"
            ][
                fold_id
            ]
        )

        metric_rows.append(
            {
                "task":
                    "opportunity",

                "fold_id":
                    fold_id,

                "features":
                    275,

                "dropped_feature":
                    DROPPED_FEATURE,

                "best_iteration":
                    int(
                        model.best_iteration
                    ),

                "validation_rows":
                    len(valid),

                "control_metric":
                    control_metric,

                "ablation_metric":
                    experiment_metric,

                "metric_change":
                    (
                        experiment_metric
                        -
                        control_metric
                    ),

                "improved":
                    int(
                        experiment_metric
                        >
                        control_metric
                    ),
            }
        )

        model.save_model(
            str(
                MODEL_DIR
                /
                f"{fold_id}_opportunity_no_tenure.txt"
            )
        )

        print(
            f"{fold_id} | control Weighted F1: "
            f"{control_metric:.6f} | "
            f"no-tenure Weighted F1: "
            f"{experiment_metric:.6f} | "
            f"change: "
            f"{experiment_metric-control_metric:+.6f}"
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    metrics = pd.DataFrame(
        metric_rows
    )

    summary_rows = []

    for task, block in metrics.groupby(
        "task"
    ):

        weights = (
            block[
                "validation_rows"
            ]
            /
            block[
                "validation_rows"
            ].sum()
        )

        control_temporal = float(
            np.sum(
                block[
                    "control_metric"
                ]
                *
                weights
            )
        )

        ablation_temporal = float(
            np.sum(
                block[
                    "ablation_metric"
                ]
                *
                weights
            )
        )

        folds_improved = int(
            block[
                "improved"
            ].sum()
        )

        f4 = block[
            block[
                "fold_id"
            ].eq(
                "F4"
            )
        ].iloc[0]

        if task in [
            "fuel",
            "nonfuel",
        ]:

            temporal_improved = (
                ablation_temporal
                <
                control_temporal
            )

            f4_not_worse = (
                float(
                    f4[
                        "ablation_metric"
                    ]
                )
                <=
                float(
                    f4[
                        "control_metric"
                    ]
                )
            )

        else:

            temporal_improved = (
                ablation_temporal
                >
                control_temporal
            )

            f4_not_worse = (
                float(
                    f4[
                        "ablation_metric"
                    ]
                )
                >=
                float(
                    f4[
                        "control_metric"
                    ]
                )
            )

        folds_rule = (
            folds_improved >= 2
        )

        accepted = (
            temporal_improved
            and
            f4_not_worse
            and
            folds_rule
        )

        summary_rows.append(
            {
                "task":
                    task,

                "control_features":
                    276,

                "ablation_features":
                    275,

                "dropped_feature":
                    DROPPED_FEATURE,

                "folds_improved":
                    folds_improved,

                "row_weighted_control_metric":
                    control_temporal,

                "row_weighted_ablation_metric":
                    ablation_temporal,

                "row_weighted_metric_change":
                    (
                        ablation_temporal
                        -
                        control_temporal
                    ),

                "f4_control_metric":
                    float(
                        f4[
                            "control_metric"
                        ]
                    ),

                "f4_ablation_metric":
                    float(
                        f4[
                            "ablation_metric"
                        ]
                    ),

                "f4_metric_change":
                    float(
                        f4[
                            "ablation_metric"
                        ]
                        -
                        f4[
                            "control_metric"
                        ]
                    ),

                "criterion_temporal_improved":
                    int(
                        temporal_improved
                    ),

                "criterion_f4_not_worse":
                    int(
                        f4_not_worse
                    ),

                "criterion_two_of_three_folds":
                    int(
                        folds_rule
                    ),

                "accepted_by_preregistered_rule":
                    int(
                        accepted
                    ),

                "validation_status":
                    "PASS",
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    metrics.to_csv(
        OUTPUT_DIR
        / "fold_comparison.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "task_summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 06B TENURE DRIFT ABLATION COMPLETE"
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
