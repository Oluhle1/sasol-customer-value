"""
Phase 06R-A3
Frozen LightGBM customer-embedding ablation.

Research question
-----------------
Do longitudinal customer embeddings learned from product purchase
histories add predictive information beyond the 276 frozen handcrafted
features for forward CLV prediction?

Controlled comparison
---------------------
Baseline:
    276 frozen Phase 04 features

Augmented:
    same 276 features + 64 warm-start SGNS dimensions

Targets:
    CLV_fuel
    CLV_nonfuel

Opportunity is deliberately excluded because the current empirical
research basis concerns CLV/value prediction rather than the Sasol
17-class next-opportunity problem.

No model hyperparameters are tuned.

Both variants use exactly the frozen production specification:
    fuel    : leaves=15, minleaf=100, rounds=168
    nonfuel : leaves=15, minleaf=50,  rounds=145

Acceptance rule, inherited from the existing project experiment policy:
    1. row-weighted RMSE improves;
    2. F4 is not worse;
    3. >=3/4 temporal folds improve.
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

EMBEDDING_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "customer_embeddings_warmstart"
    / "customer_embeddings.parquet"
)

BASE_CONFIG_PATH = (
    ROOT
    / "configs"
    / "lightgbm_baseline.json"
)

FINAL_SPEC_PATH = (
    ROOT
    / "configs"
    / "final_model_spec.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "embedding_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


def make_parameters(
    base_config,
    final_spec,
    task,
):

    parameters = dict(
        base_config[
            "common_parameters"
        ]
    )

    parameters.update(
        base_config[
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
        final_spec[
            task
        ][
            "num_leaves"
        ]
    )

    parameters[
        "min_data_in_leaf"
    ] = int(
        final_spec[
            task
        ][
            "min_data_in_leaf"
        ]
    )

    seed = int(
        base_config[
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


def train_model(
    X_train,
    y_train,
    X_valid,
    parameters,
    rounds,
):

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=list(
            X_train.columns
        ),
        free_raw_data=False,
    )

    model = lgb.train(
        parameters,
        train_data,
        num_boost_round=rounds,
        callbacks=[
            lgb.log_evaluation(
                period=0
            )
        ],
    )

    raw_prediction = model.predict(
        X_valid,
        num_iteration=rounds,
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


def main():

    print(
        "Loading frozen Phase 04 model specification..."
    )

    base_config = json.loads(
        BASE_CONFIG_PATH.read_text()
    )

    final_spec = json.loads(
        FINAL_SPEC_PATH.read_text()
    )

    if final_spec[
        "status"
    ] != "FROZEN":

        raise AssertionError(
            "Final model specification is not frozen."
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

    embeddings = pd.read_parquet(
        EMBEDDING_PATH
    )

    embeddings[
        "ID"
    ] = embeddings[
        "ID"
    ].astype(str)

    embeddings[
        "cutoff"
    ] = pd.to_datetime(
        embeddings[
            "cutoff"
        ]
    )

    embedding_features = sorted(
        [
            column
            for column
            in embeddings.columns
            if column.startswith(
                "cust_emb_"
            )
        ]
    )

    if len(
        embedding_features
    ) != 64:

        raise AssertionError(
            "Expected exactly 64 embedding features."
        )

    if embeddings.duplicated(
        [
            "ID",
            "cutoff",
        ]
    ).any():

        raise AssertionError(
            "Duplicate embedding ID-cutoff keys."
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
            "Expected exactly 276 frozen baseline features."
        )

    merged = data.merge(
        embeddings[
            [
                "ID",
                "cutoff",
            ]
            +
            embedding_features
        ],
        on=[
            "ID",
            "cutoff",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(
        merged
    ) != len(
        data
    ):

        raise AssertionError(
            "Embedding merge changed supervised row count."
        )

    missing_embedding_rows = int(
        merged[
            embedding_features
        ].isna().any(
            axis=1
        ).sum()
    )

    if missing_embedding_rows != 0:

        raise AssertionError(
            f"{missing_embedding_rows} supervised rows "
            "lack embeddings."
        )

    augmented_features = (
        selected_features
        +
        embedding_features
    )

    if len(
        augmented_features
    ) != 340:

        raise AssertionError(
            "Expected 340 augmented predictors."
        )

    print(
        f"Historical rows        : {len(merged):,}"
    )

    print(
        f"Baseline predictors    : {len(selected_features)}"
    )

    print(
        f"Embedding predictors   : {len(embedding_features)}"
    )

    print(
        f"Augmented predictors   : {len(augmented_features)}"
    )

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

    metric_rows = []
    prediction_frames = []
    importance_rows = []

    for task, target in tasks:

        parameters = make_parameters(
            base_config,
            final_spec,
            task,
        )

        rounds = int(
            final_spec[
                task
            ][
                "num_boost_round"
            ]
        )

        print()
        print(
            "#" * 78
        )

        print(
            f"TASK: {task.upper()}"
        )

        print(
            f"Frozen rounds: {rounds}"
        )

        print(
            "#" * 78
        )

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

            train = merged[
                merged[
                    "snapshot_id"
                ].isin(
                    train_ids
                )
            ].copy()

            valid = merged[
                merged[
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
                    "training membership mismatch."
                )

            if len(
                valid
            ) != len(
                valid_ids
            ):

                raise AssertionError(
                    f"{task} {fold_id}: "
                    "validation membership mismatch."
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

            print()
            print(
                f"{fold_id} | "
                f"train={len(train):,} | "
                f"valid={len(valid):,}"
            )

            variants = [
                (
                    "baseline_276",
                    selected_features,
                ),
                (
                    "embedding_340",
                    augmented_features,
                ),
            ]

            for (
                variant,
                feature_set,
            ) in variants:

                X_train = train[
                    feature_set
                ]

                X_valid = valid[
                    feature_set
                ]

                (
                    model,
                    raw_prediction,
                    prediction,
                ) = train_model(
                    X_train,
                    y_train,
                    X_valid,
                    parameters,
                    rounds,
                )

                score = rmse(
                    y_valid,
                    prediction,
                )

                metric_rows.append(
                    {
                        "task":
                            task,

                        "fold_id":
                            fold_id,

                        "variant":
                            variant,

                        "train_rows":
                            len(train),

                        "validation_rows":
                            len(valid),

                        "features":
                            len(feature_set),

                        "num_leaves":
                            parameters[
                                "num_leaves"
                            ],

                        "min_data_in_leaf":
                            parameters[
                                "min_data_in_leaf"
                            ],

                        "num_boost_round":
                            rounds,

                        "validation_rmse":
                            score,

                        "raw_negative_predictions":
                            int(
                                (
                                    raw_prediction
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
                    "variant"
                ] = variant

                prediction_frame[
                    "prediction"
                ] = prediction

                prediction_frames.append(
                    prediction_frame
                )

                if variant == "embedding_340":

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

                    for (
                        feature,
                        gain,
                        split,
                    ) in zip(
                        feature_set,
                        gains,
                        splits,
                    ):

                        importance_rows.append(
                            {
                                "task":
                                    task,

                                "fold_id":
                                    fold_id,

                                "feature":
                                    feature,

                                "is_embedding":
                                    int(
                                        feature
                                        in
                                        embedding_features
                                    ),

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
                    f"  {variant:14s} "
                    f"RMSE={score:.6f}"
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

    paired_rows = []
    summary_rows = []

    for task in [
        "fuel",
        "nonfuel",
    ]:

        block = metrics[
            metrics[
                "task"
            ].eq(
                task
            )
        ]

        baseline = (
            block[
                block[
                    "variant"
                ].eq(
                    "baseline_276"
                )
            ][
                [
                    "fold_id",
                    "validation_rows",
                    "validation_rmse",
                ]
            ]
            .rename(
                columns={
                    "validation_rmse":
                        "baseline_rmse",
                }
            )
        )

        augmented = (
            block[
                block[
                    "variant"
                ].eq(
                    "embedding_340"
                )
            ][
                [
                    "fold_id",
                    "validation_rmse",
                ]
            ]
            .rename(
                columns={
                    "validation_rmse":
                        "embedding_rmse",
                }
            )
        )

        paired = baseline.merge(
            augmented,
            on="fold_id",
            validate="one_to_one",
        )

        paired[
            "rmse_change"
        ] = (
            paired[
                "embedding_rmse"
            ]
            -
            paired[
                "baseline_rmse"
            ]
        )

        paired[
            "relative_change_pct"
        ] = (
            100.0
            *
            paired[
                "rmse_change"
            ]
            /
            paired[
                "baseline_rmse"
            ]
        )

        paired[
            "improved"
        ] = (
            paired[
                "embedding_rmse"
            ]
            <
            paired[
                "baseline_rmse"
            ]
        )

        paired.insert(
            0,
            "task",
            task,
        )

        paired_rows.append(
            paired
        )

        weights = (
            paired[
                "validation_rows"
            ]
            /
            paired[
                "validation_rows"
            ].sum()
        )

        weighted_baseline = float(
            np.sum(
                paired[
                    "baseline_rmse"
                ]
                *
                weights
            )
        )

        weighted_embedding = float(
            np.sum(
                paired[
                    "embedding_rmse"
                ]
                *
                weights
            )
        )

        folds_improved = int(
            paired[
                "improved"
            ].sum()
        )

        f4 = paired[
            paired[
                "fold_id"
            ].eq(
                "F4"
            )
        ].iloc[0]

        weighted_improved = (
            weighted_embedding
            <
            weighted_baseline
        )

        f4_not_worse = (
            float(
                f4[
                    "embedding_rmse"
                ]
            )
            <=
            float(
                f4[
                    "baseline_rmse"
                ]
            )
        )

        three_folds = (
            folds_improved >= 3
        )

        accepted = (
            weighted_improved
            and
            f4_not_worse
            and
            three_folds
        )

        task_importance = importance[
            importance[
                "task"
            ].eq(
                task
            )
        ]

        total_gain = float(
            task_importance[
                "gain"
            ].sum()
        )

        embedding_gain = float(
            task_importance.loc[
                task_importance[
                    "is_embedding"
                ].eq(1),
                "gain",
            ].sum()
        )

        embedding_gain_share = (
            embedding_gain
            /
            total_gain
            if total_gain > 0
            else np.nan
        )

        summary_rows.append(
            {
                "task":
                    task,

                "folds_improved":
                    folds_improved,

                "row_weighted_baseline_rmse":
                    weighted_baseline,

                "row_weighted_embedding_rmse":
                    weighted_embedding,

                "row_weighted_rmse_change":
                    weighted_embedding
                    -
                    weighted_baseline,

                "row_weighted_relative_change_pct":
                    100.0
                    *
                    (
                        weighted_embedding
                        -
                        weighted_baseline
                    )
                    /
                    weighted_baseline,

                "f4_baseline_rmse":
                    float(
                        f4[
                            "baseline_rmse"
                        ]
                    ),

                "f4_embedding_rmse":
                    float(
                        f4[
                            "embedding_rmse"
                        ]
                    ),

                "f4_rmse_change":
                    float(
                        f4[
                            "embedding_rmse"
                        ]
                        -
                        f4[
                            "baseline_rmse"
                        ]
                    ),

                "embedding_gain_share":
                    embedding_gain_share,

                "weighted_improved":
                    weighted_improved,

                "f4_not_worse":
                    f4_not_worse,

                "at_least_three_folds_improved":
                    three_folds,

                "accepted_by_rule":
                    accepted,
            }
        )

    paired = pd.concat(
        paired_rows,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    metrics.to_csv(
        OUTPUT_DIR
        /
        "variant_fold_metrics.csv",
        index=False,
    )

    paired.to_csv(
        OUTPUT_DIR
        /
        "paired_fold_comparison.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        /
        "embedding_ablation_summary.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        /
        "validation_predictions.csv",
        index=False,
    )

    importance.to_csv(
        OUTPUT_DIR
        /
        "embedding_feature_importance.csv",
        index=False,
    )

    experiment_config = {
        "phase":
            "06R-A3",

        "baseline_features":
            276,

        "embedding_features":
            64,

        "augmented_features":
            340,

        "targets":
            [
                "CLV_fuel",
                "CLV_nonfuel",
            ],

        "opportunity_tested":
            False,

        "model_capacity":
            {
                "fuel":
                    final_spec[
                        "fuel"
                    ],

                "nonfuel":
                    final_spec[
                        "nonfuel"
                    ],
            },

        "acceptance_rule":
            {
                "row_weighted_rmse_improves":
                    True,

                "F4_not_worse":
                    True,

                "minimum_folds_improved":
                    3,
            },

        "hyperparameter_tuning":
            False,
    }

    (
        OUTPUT_DIR
        /
        "experiment_config.json"
    ).write_text(
        json.dumps(
            experiment_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 100
    )

    print(
        "PHASE 06R-A3 EMBEDDING ABLATION COMPLETE"
    )

    print(
        "=" * 100
    )

    print()
    print(
        paired.to_string(
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
        f"Outputs: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
