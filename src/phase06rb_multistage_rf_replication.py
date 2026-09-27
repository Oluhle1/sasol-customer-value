"""
Phase 06R-B2
Research replication: direct Random Forest vs multi-stage Random Forest.

Research basis
--------------
Yan & Resnick (2024), Quantitative Marketing and Economics:
    CLV =
        probability of return
        * frequency conditional on return
        * average order value conditional on return

Their discriminative multi-stage model uses Random Forest submodels.

Purpose
-------
Test whether the multi-stage decomposition itself improves Sasol
three-month CLV forecasting under our established rolling temporal
validation.

This is NOT yet a production-model change.

No hyperparameter search is performed.

Random Forest library defaults are retained except:
    random_state for reproducibility
    n_jobs=-1 for computation

Missing-value imputation is fitted only on each training fold.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

SUPERVISED_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
    / "supervised_historical.csv"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
    / "model_feature_manifest.csv"
)

AUX_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "multistage_targets"
    / "multistage_targets_all.csv"
)

CONFIG_PATH = (
    ROOT
    / "configs"
    / "label_config.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "multistage_rf_replication"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FOLDS = [
    (
        "F1",
        ["2024-09-01"],
        "2024-12-01",
    ),
    (
        "F2",
        [
            "2024-09-01",
            "2024-12-01",
        ],
        "2025-03-01",
    ),
    (
        "F3",
        [
            "2024-09-01",
            "2024-12-01",
            "2025-03-01",
        ],
        "2025-06-01",
    ),
    (
        "F4",
        [
            "2024-09-01",
            "2024-12-01",
            "2025-03-01",
            "2025-06-01",
        ],
        "2025-09-01",
    ),
]


def rmse(y_true, y_pred):

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def rf_classifier():

    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )


def rf_regressor():

    return RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
    )


def prepare_matrix(
    train,
    valid,
    features,
):

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        train[
            features
        ]
    )

    X_valid = imputer.transform(
        valid[
            features
        ]
    )

    return (
        X_train,
        X_valid,
    )


def direct_rf(
    X_train,
    X_valid,
    y_train,
):

    model = rf_regressor()

    model.fit(
        X_train,
        y_train,
    )

    return model.predict(
        X_valid
    )


def multistage_rf(
    X_train,
    X_valid,
    train,
    active_col,
    frequency_col,
    value_col,
):

    y_active = (
        train[
            active_col
        ]
        .astype(int)
        .to_numpy()
    )

    classifier = (
        rf_classifier()
    )

    classifier.fit(
        X_train,
        y_active,
    )

    prob_active = (
        classifier
        .predict_proba(
            X_valid
        )[:, 1]
    )

    active_mask = (
        y_active == 1
    )

    if active_mask.sum() == 0:

        raise AssertionError(
            "No active customers in training fold."
        )

    frequency_model = (
        rf_regressor()
    )

    frequency_model.fit(
        X_train[
            active_mask
        ],
        train.loc[
            active_mask,
            frequency_col,
        ].to_numpy(),
    )

    value_model = (
        rf_regressor()
    )

    value_model.fit(
        X_train[
            active_mask
        ],
        train.loc[
            active_mask,
            value_col,
        ].to_numpy(),
    )

    predicted_frequency = (
        frequency_model.predict(
            X_valid
        )
    )

    predicted_value = (
        value_model.predict(
            X_valid
        )
    )

    predicted_frequency = np.clip(
        predicted_frequency,
        0.0,
        None,
    )

    predicted_value = np.clip(
        predicted_value,
        0.0,
        None,
    )

    predicted_raw = (
        prob_active
        *
        predicted_frequency
        *
        predicted_value
    )

    return {
        "prob_active":
            prob_active,

        "predicted_frequency":
            predicted_frequency,

        "predicted_value":
            predicted_value,

        "predicted_raw":
            predicted_raw,
    }


def transformed_from_raw(
    raw_prediction,
    scale_constant,
):

    raw_prediction = np.clip(
        raw_prediction,
        0.0,
        None,
    )

    return (
        np.log1p(
            raw_prediction
        )
        /
        scale_constant
    )


def main():

    print(
        "Loading Phase 06R-B2 inputs..."
    )

    supervised = pd.read_csv(
        SUPERVISED_PATH,
        dtype={"ID": str},
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )

    aux = pd.read_csv(
        AUX_PATH,
        dtype={"ID": str},
        keep_default_na=False,
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    config = json.loads(
        CONFIG_PATH.read_text()
    )

    supervised[
        "cutoff"
    ] = supervised[
        "cutoff"
    ].astype(str)

    aux[
        "cutoff"
    ] = aux[
        "cutoff"
    ].astype(str)

    features = (
        manifest.loc[
            manifest[
                "selected_for_baseline"
            ].eq(1),
            "feature",
        ]
        .tolist()
    )

    if len(features) != 276:

        raise AssertionError(
            f"Expected 276 baseline features, found {len(features)}."
        )

    merged = supervised.merge(
        aux,
        on=[
            "ID",
            "cutoff",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(merged) != 22679:

        raise AssertionError(
            "Expected 22,679 historical snapshots."
        )

    auxiliary_columns = [
        "future_fuel_purchase_active",
        "future_fuel_basket_count",
        "future_fuel_litres",
        "future_fuel_litres_per_basket",
        "future_nonfuel_purchase_active",
        "future_nonfuel_basket_count",
        "future_nonfuel_rand",
        "future_nonfuel_rand_per_basket",
    ]

    if merged[
        auxiliary_columns
    ].isna().any().any():

        raise AssertionError(
            "Auxiliary target join produced missing values."
        )

    targets = {
        "fuel": {
            "official_z":
                "CLV_fuel",

            "raw":
                "future_fuel_litres",

            "active":
                "future_fuel_purchase_active",

            "frequency":
                "future_fuel_basket_count",

            "value":
                "future_fuel_litres_per_basket",

            "persistence":
                "m3_fuel_litres",

            "scale":
                float(
                    config[
                        "normalization"
                    ][
                        "CLV_fuel"
                    ]
                ),
        },

        "nonfuel": {
            "official_z":
                "CLV_nonfuel",

            "raw":
                "future_nonfuel_rand",

            "active":
                "future_nonfuel_purchase_active",

            "frequency":
                "future_nonfuel_basket_count",

            "value":
                "future_nonfuel_rand_per_basket",

            "persistence":
                "m3_nonfuel_spend",

            "scale":
                float(
                    config[
                        "normalization"
                    ][
                        "CLV_nonfuel"
                    ]
                ),
        },
    }

    fold_rows = []
    prediction_frames = []

    for (
        fold_id,
        training_cutoffs,
        validation_cutoff,
    ) in FOLDS:

        print()
        print(
            "=" * 76
        )

        print(
            f"Running {fold_id}: "
            f"{training_cutoffs} -> {validation_cutoff}"
        )

        print(
            "=" * 76
        )

        train = merged[
            merged[
                "cutoff"
            ].isin(
                training_cutoffs
            )
        ].copy()

        valid = merged[
            merged[
                "cutoff"
            ].eq(
                validation_cutoff
            )
        ].copy()

        X_train, X_valid = (
            prepare_matrix(
                train,
                valid,
                features,
            )
        )

        for target_name, spec in targets.items():

            print()
            print(
                f"{fold_id} | {target_name}"
            )

            # ====================================================
            # STATUS-QUO / PERSISTENCE BASELINE
            # ====================================================

            persistence_raw = (
                valid[
                    spec[
                        "persistence"
                    ]
                ]
                .to_numpy(
                    dtype="float64"
                )
            )

            persistence_z = (
                transformed_from_raw(
                    persistence_raw,
                    spec[
                        "scale"
                    ],
                )
            )

            y_valid_z = (
                valid[
                    spec[
                        "official_z"
                    ]
                ]
                .to_numpy(
                    dtype="float64"
                )
            )

            persistence_rmse = (
                rmse(
                    y_valid_z,
                    persistence_z,
                )
            )

            # ====================================================
            # DIRECT RANDOM FOREST
            # ====================================================

            direct_prediction = (
                direct_rf(
                    X_train,
                    X_valid,
                    train[
                        spec[
                            "official_z"
                        ]
                    ].to_numpy(
                        dtype="float64"
                    ),
                )
            )

            direct_prediction = np.clip(
                direct_prediction,
                0.0,
                None,
            )

            direct_rmse = (
                rmse(
                    y_valid_z,
                    direct_prediction,
                )
            )

            # ====================================================
            # MULTI-STAGE RANDOM FOREST
            # ====================================================

            stage = multistage_rf(
                X_train,
                X_valid,
                train,
                spec[
                    "active"
                ],
                spec[
                    "frequency"
                ],
                spec[
                    "value"
                ],
            )

            multistage_z = (
                transformed_from_raw(
                    stage[
                        "predicted_raw"
                    ],
                    spec[
                        "scale"
                    ],
                )
            )

            multistage_rmse = (
                rmse(
                    y_valid_z,
                    multistage_z,
                )
            )

            y_active_valid = (
                valid[
                    spec[
                        "active"
                    ]
                ]
                .astype(int)
                .to_numpy()
            )

            if len(
                np.unique(
                    y_active_valid
                )
            ) == 2:

                activity_auc = float(
                    roc_auc_score(
                        y_active_valid,
                        stage[
                            "prob_active"
                        ],
                    )
                )

            else:

                activity_auc = np.nan

            change = (
                multistage_rmse
                -
                direct_rmse
            )

            improved = int(
                change < 0
            )

            print(
                f"Persistence RMSE : "
                f"{persistence_rmse:.6f}"
            )

            print(
                f"Direct RF RMSE   : "
                f"{direct_rmse:.6f}"
            )

            print(
                f"Multi-stage RMSE : "
                f"{multistage_rmse:.6f}"
            )

            print(
                f"Change vs direct : "
                f"{change:+.6f}"
            )

            print(
                f"Activity AUC     : "
                f"{activity_auc:.6f}"
            )

            fold_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "target":
                        target_name,

                    "train_rows":
                        len(
                            train
                        ),

                    "validation_rows":
                        len(
                            valid
                        ),

                    "validation_cutoff":
                        validation_cutoff,

                    "persistence_rmse":
                        persistence_rmse,

                    "direct_rf_rmse":
                        direct_rmse,

                    "multistage_rf_rmse":
                        multistage_rmse,

                    "multistage_change_vs_direct":
                        change,

                    "multistage_improved":
                        improved,

                    "activity_auc":
                        activity_auc,
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "fold_id":
                            fold_id,

                        "target":
                            target_name,

                        "ID":
                            valid[
                                "ID"
                            ].to_numpy(),

                        "actual_z":
                            y_valid_z,

                        "persistence_z":
                            persistence_z,

                        "direct_rf_z":
                            direct_prediction,

                        "multistage_rf_z":
                            multistage_z,

                        "predicted_active_probability":
                            stage[
                                "prob_active"
                            ],

                        "predicted_frequency":
                            stage[
                                "predicted_frequency"
                            ],

                        "predicted_value_per_basket":
                            stage[
                                "predicted_value"
                            ],

                        "predicted_raw_total":
                            stage[
                                "predicted_raw"
                            ],
                    }
                )
            )

    fold_results = pd.DataFrame(
        fold_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    summary_rows = []

    for target_name in [
        "fuel",
        "nonfuel",
    ]:

        subset = fold_results[
            fold_results[
                "target"
            ].eq(
                target_name
            )
        ].copy()

        weights = (
            subset[
                "validation_rows"
            ]
            .to_numpy(
                dtype="float64"
            )
        )

        direct_weighted = float(
            np.average(
                subset[
                    "direct_rf_rmse"
                ],
                weights=weights,
            )
        )

        multistage_weighted = float(
            np.average(
                subset[
                    "multistage_rf_rmse"
                ],
                weights=weights,
            )
        )

        persistence_weighted = float(
            np.average(
                subset[
                    "persistence_rmse"
                ],
                weights=weights,
            )
        )

        folds_improved = int(
            subset[
                "multistage_improved"
            ].sum()
        )

        research_replication_pass = int(
            (
                folds_improved >= 3
            )
            and
            (
                multistage_weighted
                <
                direct_weighted
            )
        )

        summary_rows.append(
            {
                "target":
                    target_name,

                "folds_evaluated":
                    len(
                        subset
                    ),

                "folds_improved_vs_direct_rf":
                    folds_improved,

                "row_weighted_persistence_rmse":
                    persistence_weighted,

                "row_weighted_direct_rf_rmse":
                    direct_weighted,

                "row_weighted_multistage_rf_rmse":
                    multistage_weighted,

                "row_weighted_change_vs_direct":
                    (
                        multistage_weighted
                        -
                        direct_weighted
                    ),

                "research_replication_pass":
                    research_replication_pass,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    fold_results.to_csv(
        OUTPUT_DIR
        / "fold_results.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "replication_summary.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 06R-B2 MULTI-STAGE RF REPLICATION COMPLETE"
    )

    print(
        "=" * 76
    )

    print()
    print(
        "FOLD RESULTS"
    )
    print()

    print(
        fold_results.to_string(
            index=False
        )
    )

    print()
    print(
        "REPLICATION SUMMARY"
    )
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Research interpretation:"
    )

    print(
        "PASS means the published multi-stage structure "
        "beats direct Random Forest in >=3/4 temporal folds "
        "and in the row-weighted mean."
    )

    print()

    print(
        "PASS does NOT automatically mean the model replaces "
        "the accepted LightGBM system."
    )


if __name__ == "__main__":
    main()

