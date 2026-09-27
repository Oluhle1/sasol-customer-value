"""
Phase 06A: Temporal recency weighting.

Hypothesis:
More recent historical customer-origin snapshots are more representative
of the final competition origin than older snapshots.

Pre-registered weighting:
    temporal_weight = 2 ** (-quarter_lag / 2)

Only F2-F4 are evaluated because F1 has a single training origin.

Accepted Phase 04 model structures are retained exactly.
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

FUEL_ACCEPTED_PATH = (
    PHASE04_DIR
    / "regression_leaf_support"
    / "fold_comparison.csv"
)

NONFUEL_ACCEPTED_PATH = (
    PHASE04_DIR
    / "nonfuel_lower_capacity"
    / "fold_comparison.csv"
)

OPPORTUNITY_ACCEPTED_PATH = (
    PHASE04_DIR
    / "opportunity_weighting"
    / "fold_comparison.csv"
)

OUTPUT_DIR = PHASE06_DIR / "recency_weighting"
MODEL_DIR = ROOT / "models" / "phase06_recency_weighting"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HALF_LIFE_QUARTERS = 2.0


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def weighted_f1_eval(num_classes):

    def evaluate(predictions, dataset):

        labels = dataset.get_label().astype("int64")
        predictions = np.asarray(predictions)

        if predictions.ndim == 1:
            probabilities = (
                predictions
                .reshape(num_classes, -1)
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


def temporal_weights(cutoffs):
    """
    Newest training origin receives raw weight 1.

    Earlier origins decay with a two-quarter half-life.
    Final observation weights are normalized to mean one.
    """

    dates = pd.to_datetime(cutoffs)

    ordered_origins = sorted(
        dates.unique()
    )

    origin_rank = {
        origin: index
        for index, origin in enumerate(
            ordered_origins
        )
    }

    newest_rank = (
        len(ordered_origins)
        - 1
    )

    quarter_lags = np.asarray(
        [
            newest_rank
            - origin_rank[value]
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

    weights = (
        raw_weights
        /
        raw_weights.mean()
    )

    return (
        weights,
        quarter_lags,
    )


def class_weights(
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

    if np.any(counts == 0):
        raise AssertionError(
            "A class is absent from Opportunity training."
        )

    n = len(labels)

    raw_class_weights = (
        n
        /
        (
            num_classes
            * counts
        )
    ) ** float(exponent)

    return raw_class_weights[labels]


def normalize_weights(weights):

    weights = np.asarray(
        weights,
        dtype="float64",
    )

    mean = weights.mean()

    if not np.isfinite(mean) or mean <= 0:
        raise AssertionError(
            "Invalid training weights."
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


def common_parameters(config):

    parameters = dict(
        config["common_parameters"]
    )

    seed = int(
        config["random_seed"]
    )

    parameters.update(
        {
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
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
        config["regression"]
    )

    parameters.pop(
        "prediction_floor",
        None,
    )

    parameters["num_leaves"] = int(
        task_spec["num_leaves"]
    )

    parameters["min_data_in_leaf"] = int(
        task_spec["min_data_in_leaf"]
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
        config["opportunity"]
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

    parameters["num_leaves"] = int(
        task_spec["num_leaves"]
    )

    parameters["min_data_in_leaf"] = int(
        task_spec["min_data_in_leaf"]
    )

    parameters["num_class"] = int(
        task_spec["num_class"]
    )

    # Opportunity early stopping must use
    # the competition-relevant weighted F1.
    parameters["metric"] = "None"

    return parameters


def accepted_metric_lookup(task):

    if task == "fuel":

        data = pd.read_csv(
            FUEL_ACCEPTED_PATH
        )

        data = data[
            data["task"].eq("fuel")
        ].copy()

        value_column = (
            "experiment_validation_rmse"
        )

    elif task == "nonfuel":

        data = pd.read_csv(
            NONFUEL_ACCEPTED_PATH
        )

        value_column = (
            "capacity_validation_rmse"
        )

    elif task == "opportunity":

        data = pd.read_csv(
            OPPORTUNITY_ACCEPTED_PATH
        )

        value_column = (
            "weighted_model_f1"
        )

    else:
        raise ValueError(task)

    return {
        row["fold_id"]:
            float(row[value_column])
        for _, row in data.iterrows()
    }


def main():

    print(
        "Loading Phase 06A inputs..."
    )

    config = json.loads(
        BASELINE_CONFIG_PATH.read_text()
    )

    spec = json.loads(
        FINAL_SPEC_PATH.read_text()
    )

    if spec["status"] != "FROZEN":
        raise AssertionError(
            "Phase 04 final specification is not frozen."
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

    if len(selected_features) != 276:
        raise AssertionError(
            "Expected 276 frozen predictors."
        )

    num_classes = len(class_map)

    if num_classes != 17:
        raise AssertionError(
            "Expected 17 Opportunity classes."
        )

    folds = [
        "F2",
        "F3",
        "F4",
    ]

    accepted_lookup = {
        task:
            accepted_metric_lookup(task)
        for task in [
            "fuel",
            "nonfuel",
            "opportunity",
        ]
    }

    metric_rows = []
    origin_weight_rows = []

    task_definitions = [
        (
            "fuel",
            "CLV_fuel",
        ),
        (
            "nonfuel",
            "CLV_nonfuel",
        ),
        (
            "opportunity",
            "Opportunity_class_id",
        ),
    ]

    # ============================================================
    # TEMPORAL FOLDS
    # ============================================================

    for fold_id in folds:

        fold = membership[
            membership[
                "fold_id"
            ].eq(fold_id)
        ]

        train_ids = set(
            fold.loc[
                fold["role"].eq("train"),
                "snapshot_id",
            ]
        )

        valid_ids = set(
            fold.loc[
                fold["role"].eq("validation"),
                "snapshot_id",
            ]
        )

        train = data[
            data[
                "snapshot_id"
            ].isin(train_ids)
        ].copy()

        valid = data[
            data[
                "snapshot_id"
            ].isin(valid_ids)
        ].copy()

        if len(train) != len(train_ids):
            raise AssertionError(
                f"{fold_id}: training membership mismatch."
            )

        if len(valid) != len(valid_ids):
            raise AssertionError(
                f"{fold_id}: validation membership mismatch."
            )

        (
            time_weight,
            quarter_lag,
        ) = temporal_weights(
            train["cutoff"]
        )

        train[
            "_temporal_weight"
        ] = time_weight

        train[
            "_quarter_lag"
        ] = quarter_lag

        origin_audit = (
            train[
                [
                    "cutoff",
                    "_quarter_lag",
                    "_temporal_weight",
                ]
            ]
            .groupby(
                [
                    "cutoff",
                    "_quarter_lag",
                ],
                as_index=False,
            )
            .agg(
                rows=(
                    "_temporal_weight",
                    "size",
                ),
                normalized_weight=(
                    "_temporal_weight",
                    "first",
                ),
            )
        )

        origin_audit[
            "fold_id"
        ] = fold_id

        origin_weight_rows.append(
            origin_audit
        )

        X_train = train[
            selected_features
        ]

        X_valid = valid[
            selected_features
        ]

        for (
            task,
            target,
        ) in task_definitions:

            print()
            print(
                "=" * 72
            )

            print(
                f"Training Phase 06A "
                f"{task}: {fold_id}"
            )

            print(
                "=" * 72
            )

            if task in [
                "fuel",
                "nonfuel",
            ]:

                task_spec = spec[
                    task
                ]

                parameters = regression_parameters(
                    config,
                    task_spec,
                )

                training_weight = normalize_weights(
                    train[
                        "_temporal_weight"
                    ].to_numpy()
                )

                y_train = train[
                    target
                ].to_numpy(
                    dtype="float64"
                )

                y_valid = valid[
                    target
                ].to_numpy(
                    dtype="float64"
                )

                train_dataset = lgb.Dataset(
                    X_train,
                    label=y_train,
                    weight=training_weight,
                    feature_name=selected_features,
                    free_raw_data=False,
                )

                valid_dataset = lgb.Dataset(
                    X_valid,
                    label=y_valid,
                    reference=train_dataset,
                    feature_name=selected_features,
                    free_raw_data=False,
                )

                model = lgb.train(
                    parameters,
                    train_dataset,
                    num_boost_round=int(
                        config["num_boost_round"]
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
                        task_spec[
                            "prediction_floor"
                        ]
                    ),
                )

                experiment_metric = rmse(
                    y_valid,
                    prediction,
                )

                accepted_metric = (
                    accepted_lookup[
                        task
                    ][
                        fold_id
                    ]
                )

                change = (
                    experiment_metric
                    -
                    accepted_metric
                )

                improved = (
                    experiment_metric
                    <
                    accepted_metric
                )

            else:

                task_spec = spec[
                    "opportunity"
                ]

                parameters = opportunity_parameters(
                    config,
                    task_spec,
                )

                y_train = train[
                    target
                ].to_numpy(
                    dtype="int64"
                )

                y_valid = valid[
                    target
                ].to_numpy(
                    dtype="int64"
                )

                class_component = class_weights(
                    y_train,
                    num_classes,
                    float(
                        task_spec[
                            "class_weight_exponent"
                        ]
                    ),
                )

                training_weight = normalize_weights(
                    train[
                        "_temporal_weight"
                    ].to_numpy()
                    *
                    class_component
                )

                train_dataset = lgb.Dataset(
                    X_train,
                    label=y_train,
                    weight=training_weight,
                    feature_name=selected_features,
                    free_raw_data=False,
                )

                valid_dataset = lgb.Dataset(
                    X_valid,
                    label=y_valid,
                    reference=train_dataset,
                    feature_name=selected_features,
                    free_raw_data=False,
                )

                model = lgb.train(
                    parameters,
                    train_dataset,
                    num_boost_round=int(
                        config["num_boost_round"]
                    ),
                    valid_sets=[
                        valid_dataset,
                    ],
                    valid_names=[
                        "validation",
                    ],
                    feval=weighted_f1_eval(
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

                accepted_metric = (
                    accepted_lookup[
                        task
                    ][
                        fold_id
                    ]
                )

                change = (
                    experiment_metric
                    -
                    accepted_metric
                )

                improved = (
                    experiment_metric
                    >
                    accepted_metric
                )

            metric_rows.append(
                {
                    "task":
                        task,

                    "fold_id":
                        fold_id,

                    "train_rows":
                        len(train),

                    "validation_rows":
                        len(valid),

                    "half_life_quarters":
                        HALF_LIFE_QUARTERS,

                    "best_iteration":
                        int(
                            model.best_iteration
                        ),

                    "accepted_metric":
                        float(
                            accepted_metric
                        ),

                    "experiment_metric":
                        float(
                            experiment_metric
                        ),

                    "metric_change":
                        float(
                            change
                        ),

                    "improved":
                        int(
                            improved
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
                        f"recency_half2.txt"
                    )
                )
            )

            direction = (
                "RMSE"
                if task != "opportunity"
                else "Weighted F1"
            )

            print(
                f"{fold_id} | "
                f"accepted {direction}: "
                f"{accepted_metric:.6f} | "
                f"recency {direction}: "
                f"{experiment_metric:.6f} | "
                f"change: "
                f"{change:+.6f}"
            )

    # ============================================================
    # SUMMARY
    # ============================================================

    metrics = pd.DataFrame(
        metric_rows
    )

    origin_weights = pd.concat(
        origin_weight_rows,
        ignore_index=True,
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

        accepted_temporal = float(
            np.sum(
                block[
                    "accepted_metric"
                ]
                *
                validation_weights
            )
        )

        experiment_temporal = float(
            np.sum(
                block[
                    "experiment_metric"
                ]
                *
                validation_weights
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
                experiment_temporal
                <
                accepted_temporal
            )

            f4_not_worse = (
                float(
                    f4[
                        "experiment_metric"
                    ]
                )
                <=
                float(
                    f4[
                        "accepted_metric"
                    ]
                )
            )

        else:

            temporal_improved = (
                experiment_temporal
                >
                accepted_temporal
            )

            f4_not_worse = (
                float(
                    f4[
                        "experiment_metric"
                    ]
                )
                >=
                float(
                    f4[
                        "accepted_metric"
                    ]
                )
            )

        folds_rule = (
            folds_improved >= 2
        )

        accepted_by_rule = (
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

                "half_life_quarters":
                    HALF_LIFE_QUARTERS,

                "folds_evaluated":
                    len(block),

                "folds_improved":
                    folds_improved,

                "row_weighted_accepted_metric":
                    accepted_temporal,

                "row_weighted_experiment_metric":
                    experiment_temporal,

                "row_weighted_metric_change":
                    (
                        experiment_temporal
                        -
                        accepted_temporal
                    ),

                "f4_accepted_metric":
                    float(
                        f4[
                            "accepted_metric"
                        ]
                    ),

                "f4_experiment_metric":
                    float(
                        f4[
                            "experiment_metric"
                        ]
                    ),

                "f4_metric_change":
                    float(
                        f4[
                            "experiment_metric"
                        ]
                        -
                        f4[
                            "accepted_metric"
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

    origin_weights.to_csv(
        OUTPUT_DIR
        / "origin_weight_audit.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "task_summary.csv",
        index=False,
    )

    # ============================================================
    # CONSOLE
    # ============================================================

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 06A TEMPORAL RECENCY WEIGHTING COMPLETE"
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
        "ORIGIN WEIGHT AUDIT"
    )

    print()

    print(
        origin_weights.to_string(
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
