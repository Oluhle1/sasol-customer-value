"""
Phase 05A: Final model training and competition prediction.

Trains exactly three frozen models:

1. Fuel LightGBM regression
2. Non-fuel LightGBM regression
3. Opportunity LightGBM multiclass

Training population:
    all 22,679 labelled historical customer-origin snapshots

Prediction population:
    all 5,488 competition customers at 2025-12-01

There is no validation, early stopping, tuning or model selection in
this script.

All model decisions are read from the frozen Phase 04M specification.
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

COMPETITION_PATH = (
    PHASE04_DIR
    / "competition_features.csv"
)

FEATURE_MANIFEST_PATH = (
    PHASE04_DIR
    / "model_feature_manifest.csv"
)

CLASS_MAP_PATH = (
    PHASE04_DIR
    / "opportunity_class_map.csv"
)

BASELINE_CONFIG_PATH = (
    ROOT
    / "configs"
    / "lightgbm_baseline.json"
)

FINAL_SPEC_PATH = (
    ROOT
    / "configs"
    / "final_model_spec.json"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "final"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "phase05"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def make_common_parameters(
    baseline_config,
):
    """
    Recover the frozen common LightGBM parameters and deterministic seed.
    """

    parameters = dict(
        baseline_config[
            "common_parameters"
        ]
    )

    seed = int(
        baseline_config[
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
    baseline_config,
    task_spec,
):
    """
    Construct final regression parameters using the frozen specification.
    """

    parameters = make_common_parameters(
        baseline_config
    )

    parameters.update(
        baseline_config[
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
    baseline_config,
    task_spec,
):
    """
    Construct final Opportunity parameters from frozen specification.
    """

    parameters = make_common_parameters(
        baseline_config
    )

    parameters.update(
        baseline_config[
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

    return parameters


def make_class_weights(
    labels,
    num_classes,
    exponent,
):
    """
    Final frozen Opportunity class-weight policy.

        raw_w_c =
            (N / (K * n_c)) ** exponent

    Observation weights are normalized to mean one.
    """

    labels = np.asarray(
        labels,
        dtype="int64",
    )

    counts = np.bincount(
        labels,
        minlength=
            num_classes,
    )

    if np.any(
        counts == 0
    ):

        raise AssertionError(
            "At least one Opportunity class "
            "is absent from final training."
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
            "Final Opportunity weights "
            "do not normalize to mean one."
        )

    return (
        sample_weights,
        counts,
        class_weights,
    )


def main():

    print(
        "Loading frozen Phase 05 inputs..."
    )

    spec = json.loads(
        FINAL_SPEC_PATH.read_text()
    )

    baseline_config = json.loads(
        BASELINE_CONFIG_PATH.read_text()
    )

    if spec[
        "status"
    ] != "FROZEN":

        raise AssertionError(
            "Final model specification "
            "is not frozen."
        )

    train = pd.read_csv(
        SUPERVISED_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    competition = pd.read_csv(
        COMPETITION_PATH,
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

    selected_features = (
        manifest.loc[
            manifest[
                "selected_for_baseline"
            ].eq(1),
            "feature",
        ]
        .tolist()
    )

    # ============================================================
    # FINAL INTEGRITY CHECKS
    # ============================================================

    expected_train_rows = int(
        spec[
            "training_population"
        ][
            "historical_rows"
        ]
    )

    expected_competition_rows = int(
        spec[
            "training_population"
        ][
            "competition_rows"
        ]
    )

    expected_feature_count = int(
        spec[
            "feature_policy"
        ][
            "selected_features"
        ]
    )

    expected_classes = int(
        spec[
            "opportunity"
        ][
            "num_class"
        ]
    )

    if len(
        train
    ) != expected_train_rows:

        raise AssertionError(
            "Final training row count mismatch."
        )

    if len(
        competition
    ) != expected_competition_rows:

        raise AssertionError(
            "Final competition row count mismatch."
        )

    if len(
        selected_features
    ) != expected_feature_count:

        raise AssertionError(
            "Final selected feature count mismatch."
        )

    if len(
        class_map
    ) != expected_classes:

        raise AssertionError(
            "Final Opportunity class count mismatch."
        )

    if train[
        "snapshot_id"
    ].duplicated().any():

        raise AssertionError(
            "Duplicate historical snapshot IDs."
        )

    if competition[
        "snapshot_id"
    ].duplicated().any():

        raise AssertionError(
            "Duplicate competition snapshot IDs."
        )

    if competition[
        "ID"
    ].duplicated().any():

        raise AssertionError(
            "Duplicate competition customer IDs."
        )

    if train[
        [
            "CLV_fuel",
            "CLV_nonfuel",
            "Opportunity_class_id",
        ]
    ].isna().any().any():

        raise AssertionError(
            "Missing final training targets."
        )

    for feature in selected_features:

        if feature not in train.columns:

            raise AssertionError(
                f"Training missing feature: "
                f"{feature}"
            )

        if feature not in competition.columns:

            raise AssertionError(
                f"Competition missing feature: "
                f"{feature}"
            )

    X_train = train[
        selected_features
    ]

    X_competition = competition[
        selected_features
    ]

    # ============================================================
    # FINAL FUEL MODEL
    # ============================================================

    print()
    print(
        "Training final Fuel model..."
    )

    fuel_spec = spec[
        "fuel"
    ]

    fuel_parameters = (
        regression_parameters(
            baseline_config,
            fuel_spec,
        )
    )

    fuel_rounds = int(
        fuel_spec[
            "num_boost_round"
        ]
    )

    fuel_train_data = lgb.Dataset(
        X_train,
        label=
            train[
                "CLV_fuel"
            ].to_numpy(
                dtype="float64"
            ),
        feature_name=
            selected_features,
        free_raw_data=False,
    )

    fuel_model = lgb.train(
        fuel_parameters,
        fuel_train_data,
        num_boost_round=
            fuel_rounds,
        callbacks=[
            lgb.log_evaluation(
                period=0
            )
        ],
    )

    fuel_prediction_raw = (
        fuel_model.predict(
            X_competition,
            num_iteration=
                fuel_rounds,
        )
    )

    fuel_prediction = np.maximum(
        fuel_prediction_raw,
        float(
            fuel_spec[
                "prediction_floor"
            ]
        ),
    )

    fuel_model.save_model(
        str(
            MODEL_DIR
            / "final_fuel.txt"
        )
    )

    # ============================================================
    # FINAL NON-FUEL MODEL
    # ============================================================

    print(
        "Training final Non-fuel model..."
    )

    nonfuel_spec = spec[
        "nonfuel"
    ]

    nonfuel_parameters = (
        regression_parameters(
            baseline_config,
            nonfuel_spec,
        )
    )

    nonfuel_rounds = int(
        nonfuel_spec[
            "num_boost_round"
        ]
    )

    nonfuel_train_data = lgb.Dataset(
        X_train,
        label=
            train[
                "CLV_nonfuel"
            ].to_numpy(
                dtype="float64"
            ),
        feature_name=
            selected_features,
        free_raw_data=False,
    )

    nonfuel_model = lgb.train(
        nonfuel_parameters,
        nonfuel_train_data,
        num_boost_round=
            nonfuel_rounds,
        callbacks=[
            lgb.log_evaluation(
                period=0
            )
        ],
    )

    nonfuel_prediction_raw = (
        nonfuel_model.predict(
            X_competition,
            num_iteration=
                nonfuel_rounds,
        )
    )

    nonfuel_prediction = np.maximum(
        nonfuel_prediction_raw,
        float(
            nonfuel_spec[
                "prediction_floor"
            ]
        ),
    )

    nonfuel_model.save_model(
        str(
            MODEL_DIR
            / "final_nonfuel.txt"
        )
    )

    # ============================================================
    # FINAL OPPORTUNITY MODEL
    # ============================================================

    print(
        "Training final Opportunity model..."
    )

    opportunity_spec = spec[
        "opportunity"
    ]

    opportunity_parameters_final = (
        opportunity_parameters(
            baseline_config,
            opportunity_spec,
        )
    )

    opportunity_rounds = int(
        opportunity_spec[
            "num_boost_round"
        ]
    )

    num_classes = int(
        opportunity_spec[
            "num_class"
        ]
    )

    y_opportunity = train[
        "Opportunity_class_id"
    ].to_numpy(
        dtype="int64"
    )

    (
        opportunity_sample_weights,
        opportunity_class_counts,
        opportunity_class_weights,
    ) = make_class_weights(
        y_opportunity,
        num_classes,
        float(
            opportunity_spec[
                "class_weight_exponent"
            ]
        ),
    )

    opportunity_train_data = lgb.Dataset(
        X_train,
        label=
            y_opportunity,
        weight=
            opportunity_sample_weights,
        feature_name=
            selected_features,
        free_raw_data=False,
    )

    opportunity_model = lgb.train(
        opportunity_parameters_final,
        opportunity_train_data,
        num_boost_round=
            opportunity_rounds,
        callbacks=[
            lgb.log_evaluation(
                period=0
            )
        ],
    )

    opportunity_probability = (
        opportunity_model.predict(
            X_competition,
            num_iteration=
                opportunity_rounds,
        )
    )

    expected_probability_shape = (
        len(
            competition
        ),
        num_classes,
    )

    if (
        opportunity_probability.shape
        != expected_probability_shape
    ):

        raise AssertionError(
            "Unexpected Opportunity "
            f"probability shape: "
            f"{opportunity_probability.shape}"
        )

    probability_row_sum = (
        opportunity_probability.sum(
            axis=1
        )
    )

    if not np.allclose(
        probability_row_sum,
        1.0,
        atol=1e-6,
    ):

        raise AssertionError(
            "Opportunity probabilities "
            "do not sum to one."
        )

    opportunity_class_prediction = (
        np.argmax(
            opportunity_probability,
            axis=1,
        )
        .astype(
            "int64"
        )
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

    opportunity_prediction = [
        class_lookup[
            int(
                class_id
            )
        ]
        for class_id
        in opportunity_class_prediction
    ]

    opportunity_model.save_model(
        str(
            MODEL_DIR
            / "final_opportunity.txt"
        )
    )

    # ============================================================
    # FINAL PREDICTIONS
    # ============================================================

    final_predictions = competition[
        [
            "ID",
            "cutoff",
        ]
    ].copy()

    final_predictions[
        "CLV_fuel"
    ] = fuel_prediction

    final_predictions[
        "CLV_nonfuel"
    ] = nonfuel_prediction

    final_predictions[
        "Opportunity"
    ] = opportunity_prediction

    final_predictions[
        "Opportunity_class_id"
    ] = (
        opportunity_class_prediction
    )

    final_predictions.to_csv(
        OUTPUT_DIR
        / "competition_predictions.csv",
        index=False,
        float_format="%.10g",
    )

    # ============================================================
    # OPPORTUNITY PROBABILITIES
    # ============================================================

    probability_output = competition[
        [
            "ID",
            "cutoff",
        ]
    ].copy()

    for class_id in range(
        num_classes
    ):

        probability_output[
            f"prob_class_{class_id}"
        ] = (
            opportunity_probability[
                :,
                class_id
            ]
        )

    probability_output.to_csv(
        OUTPUT_DIR
        / "opportunity_probabilities.csv",
        index=False,
        float_format="%.10g",
    )

    # ============================================================
    # CLASS WEIGHT AUDIT
    # ============================================================

    class_weight_rows = []

    for class_id in range(
        num_classes
    ):

        class_weight_rows.append(
            {
                "class_id":
                    class_id,

                "Opportunity":
                    class_lookup[
                        class_id
                    ],

                "training_count":
                    int(
                        opportunity_class_counts[
                            class_id
                        ]
                    ),

                "class_weight":
                    float(
                        opportunity_class_weights[
                            class_id
                        ]
                    ),
            }
        )

    class_weight_audit = pd.DataFrame(
        class_weight_rows
    )

    class_weight_audit.to_csv(
        OUTPUT_DIR
        / "final_opportunity_class_weights.csv",
        index=False,
    )

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    importance_rows = []

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

        gains = model.feature_importance(
            importance_type="gain"
        )

        splits = model.feature_importance(
            importance_type="split"
        )

        for (
            feature,
            gain,
            split,
        ) in zip(
            selected_features,
            gains,
            splits,
        ):

            importance_rows.append(
                {
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

    importance = pd.DataFrame(
        importance_rows
    )

    importance.to_csv(
        OUTPUT_DIR
        / "final_feature_importance.csv",
        index=False,
    )

    # ============================================================
    # PREDICTION DISTRIBUTION AUDIT
    # ============================================================

    opportunity_distribution = (
        final_predictions[
            "Opportunity"
        ]
        .value_counts()
        .rename_axis(
            "Opportunity"
        )
        .reset_index(
            name=
                "predicted_count"
        )
    )

    opportunity_distribution[
        "predicted_share"
    ] = (
        opportunity_distribution[
            "predicted_count"
        ]
        /
        len(
            final_predictions
        )
    )

    opportunity_distribution.to_csv(
        OUTPUT_DIR
        / "final_opportunity_distribution.csv",
        index=False,
    )

    prediction_summary = pd.DataFrame(
        [
            {
                "rows":
                    len(
                        final_predictions
                    ),

                "unique_ids":
                    final_predictions[
                        "ID"
                    ].nunique(),

                "fuel_min":
                    float(
                        final_predictions[
                            "CLV_fuel"
                        ].min()
                    ),

                "fuel_mean":
                    float(
                        final_predictions[
                            "CLV_fuel"
                        ].mean()
                    ),

                "fuel_median":
                    float(
                        final_predictions[
                            "CLV_fuel"
                        ].median()
                    ),

                "fuel_max":
                    float(
                        final_predictions[
                            "CLV_fuel"
                        ].max()
                    ),

                "fuel_zero_predictions":
                    int(
                        final_predictions[
                            "CLV_fuel"
                        ].eq(0)
                        .sum()
                    ),

                "nonfuel_min":
                    float(
                        final_predictions[
                            "CLV_nonfuel"
                        ].min()
                    ),

                "nonfuel_mean":
                    float(
                        final_predictions[
                            "CLV_nonfuel"
                        ].mean()
                    ),

                "nonfuel_median":
                    float(
                        final_predictions[
                            "CLV_nonfuel"
                        ].median()
                    ),

                "nonfuel_max":
                    float(
                        final_predictions[
                            "CLV_nonfuel"
                        ].max()
                    ),

                "nonfuel_zero_predictions":
                    int(
                        final_predictions[
                            "CLV_nonfuel"
                        ].eq(0)
                        .sum()
                    ),

                "opportunity_predicted_classes":
                    final_predictions[
                        "Opportunity"
                    ].nunique(),

                "raw_negative_fuel_predictions":
                    int(
                        (
                            fuel_prediction_raw
                            < 0
                        ).sum()
                    ),

                "raw_negative_nonfuel_predictions":
                    int(
                        (
                            nonfuel_prediction_raw
                            < 0
                        ).sum()
                    ),

                "validation_status":
                    "PASS",
            }
        ]
    )

    prediction_summary.to_csv(
        OUTPUT_DIR
        / "final_prediction_summary.csv",
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
        "PHASE 05A FINAL MODEL TRAINING COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "FINAL MODEL SPECIFICATION"
    )

    print()

    print(
        f"Fuel       : "
        f"{fuel_rounds} rounds | "
        f"{fuel_spec['num_leaves']} leaves | "
        f"minleaf {fuel_spec['min_data_in_leaf']}"
    )

    print(
        f"Non-fuel   : "
        f"{nonfuel_rounds} rounds | "
        f"{nonfuel_spec['num_leaves']} leaves | "
        f"minleaf {nonfuel_spec['min_data_in_leaf']}"
    )

    print(
        f"Opportunity: "
        f"{opportunity_rounds} rounds | "
        f"{opportunity_spec['num_leaves']} leaves | "
        f"minleaf {opportunity_spec['min_data_in_leaf']} | "
        f"alpha "
        f"{opportunity_spec['class_weight_exponent']}"
    )

    print()

    print(
        "PREDICTION SUMMARY"
    )

    print()

    print(
        prediction_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "OPPORTUNITY DISTRIBUTION"
    )

    print()

    print(
        opportunity_distribution.to_string(
            index=False
        )
    )

    print()

    print(
        "FINAL OPPORTUNITY CLASS WEIGHTS"
    )

    print()

    print(
        class_weight_audit.to_string(
            index=False
        )
    )

    print()

    print(
        "Models written to:"
    )

    print(
        MODEL_DIR
    )

    print()

    print(
        "Predictions written to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
