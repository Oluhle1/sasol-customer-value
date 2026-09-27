"""
Phase 06R-B2b
Enhanced Multi-Stage Random Forest reproduction.

Primary source:
Yan & Resnick (2024),
A high-performance turnkey system for customer lifetime value
prediction in retail brands.

Published algorithm:
1. Balance return/churn classes.
2. Train RF return classifier.
3. Recalibrate probabilities on original population.
4. Train RF frequency and value models on returning customers.
5. Generate cross-validated training predictions.
6. Learn sigmoid adjustment of return probability by minimizing
   aggregate raw-value error.
7. Apply frozen adjustment to temporal validation fold.

Important:
The paper does not specify its exact software implementation for
probability calibration or numerical optimizer.

Reproduction choices:
- Platt-style logistic calibration.
- GroupKFold by customer ID for inner OOF predictions.
- Standard two-parameter logistic:
      sigma(p) = 1 / (1 + exp(-t1 * (p - t2)))
- deterministic differential evolution for t1/t2.
- t1 bound [0.1, 50], informed by the paper's reported examples
  with fitted t1 values around 10-24.
- t2 is constrained to [0, 1].

These implementation details are documented adaptations, not claimed
to have been specified by the paper.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution
from scipy.special import expit

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import GroupKFold


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

NAIVE_RESULTS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "multistage_rf_replication"
    / "fold_results.csv"
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
    / "enhanced_multistage_rf"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42
INNER_SPLITS = 5

FOLDS = [
    ("F1", ["2024-09-01"], "2024-12-01"),
    (
        "F2",
        ["2024-09-01", "2024-12-01"],
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
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def rf_regressor():
    return RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def balance_indices(y, rng):
    """
    Downsample the majority class to the minority-class size.

    The Yan & Resnick application downsampled churn because churn
    was the majority class in their setting.

    Sasol's majority class may differ by target/cutoff, so this
    implementation balances whichever class is larger.
    """

    y = np.asarray(y, dtype=int)

    idx0 = np.flatnonzero(
        y == 0
    )
    idx1 = np.flatnonzero(
        y == 1
    )

    if len(idx0) == 0 or len(idx1) == 0:
        raise AssertionError(
            "Return target contains only one class."
        )

    n = min(
        len(idx0),
        len(idx1),
    )

    keep0 = (
        idx0
        if len(idx0) == n
        else rng.choice(
            idx0,
            size=n,
            replace=False,
        )
    )

    keep1 = (
        idx1
        if len(idx1) == n
        else rng.choice(
            idx1,
            size=n,
            replace=False,
        )
    )

    keep = np.concatenate(
        [
            keep0,
            keep1,
        ]
    )

    rng.shuffle(
        keep
    )

    return keep


def prepare_split(
    train_frame,
    valid_frame,
    features,
):

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        train_frame[
            features
        ]
    )

    X_valid = imputer.transform(
        valid_frame[
            features
        ]
    )

    return (
        X_train,
        X_valid,
        imputer,
    )


def fit_balanced_classifier(
    X,
    y,
    seed_offset=0,
):

    rng = np.random.default_rng(
        RANDOM_STATE
        +
        seed_offset
    )

    keep = balance_indices(
        y,
        rng,
    )

    model = rf_classifier()

    model.fit(
        X[
            keep
        ],
        np.asarray(
            y
        )[
            keep
        ],
    )

    return model


def sigmoid_adjust(
    probability,
    t1,
    t2,
):

    return expit(
        float(t1)
        *
        (
            np.asarray(
                probability,
                dtype="float64",
            )
            -
            float(t2)
        )
    )


def fit_sigmoid(
    calibrated_probability,
    clv_return_prediction,
    actual_raw,
):

    probability = np.asarray(
        calibrated_probability,
        dtype="float64",
    )

    conditional_clv = np.asarray(
        clv_return_prediction,
        dtype="float64",
    )

    actual = np.asarray(
        actual_raw,
        dtype="float64",
    )

    actual_total = float(
        actual.sum()
    )

    def objective(params):

        t1, t2 = params

        adjusted = sigmoid_adjust(
            probability,
            t1,
            t2,
        )

        predicted_total = float(
            (
                adjusted
                *
                conditional_clv
            ).sum()
        )

        return abs(
            predicted_total
            -
            actual_total
        )

    result = differential_evolution(
        objective,
        bounds=[
            (0.1, 50.0),
            (0.0, 1.0),
        ],
        seed=RANDOM_STATE,
        polish=True,
        updating="immediate",
    )

    if not result.success:
        raise RuntimeError(
            "Sigmoid optimization failed: "
            + str(
                result.message
            )
        )

    return (
        float(
            result.x[0]
        ),
        float(
            result.x[1]
        ),
        float(
            result.fun
        ),
    )


def make_inner_oof(
    frame,
    features,
    active_col,
    frequency_col,
    value_col,
):

    n = len(
        frame
    )

    groups = (
        frame[
            "ID"
        ]
        .astype(str)
        .to_numpy()
    )

    n_unique_groups = len(
        np.unique(
            groups
        )
    )

    n_splits = min(
        INNER_SPLITS,
        n_unique_groups,
    )

    if n_splits < 2:
        raise AssertionError(
            "Not enough customer groups for inner CV."
        )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    raw_return_oof = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    frequency_oof = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    value_oof = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    y_active_all = (
        frame[
            active_col
        ]
        .astype(int)
        .to_numpy()
    )

    for inner_fold, (
        inner_train_idx,
        inner_valid_idx,
    ) in enumerate(
        splitter.split(
            frame,
            y_active_all,
            groups=groups,
        ),
        start=1,
    ):

        inner_train = frame.iloc[
            inner_train_idx
        ]

        inner_valid = frame.iloc[
            inner_valid_idx
        ]

        (
            X_inner_train,
            X_inner_valid,
            _,
        ) = prepare_split(
            inner_train,
            inner_valid,
            features,
        )

        y_inner_active = (
            inner_train[
                active_col
            ]
            .astype(int)
            .to_numpy()
        )

        classifier = (
            fit_balanced_classifier(
                X_inner_train,
                y_inner_active,
                seed_offset=inner_fold,
            )
        )

        raw_return_oof[
            inner_valid_idx
        ] = (
            classifier
            .predict_proba(
                X_inner_valid
            )[:, 1]
        )

        active_train_mask = (
            y_inner_active
            ==
            1
        )

        if (
            active_train_mask.sum()
            ==
            0
        ):
            raise AssertionError(
                "No active observations in inner training fold."
            )

        frequency_model = (
            rf_regressor()
        )

        frequency_model.fit(
            X_inner_train[
                active_train_mask
            ],
            inner_train.loc[
                inner_train[
                    active_col
                ].eq(1),
                frequency_col,
            ].to_numpy(
                dtype="float64"
            ),
        )

        value_model = (
            rf_regressor()
        )

        value_model.fit(
            X_inner_train[
                active_train_mask
            ],
            inner_train.loc[
                inner_train[
                    active_col
                ].eq(1),
                value_col,
            ].to_numpy(
                dtype="float64"
            ),
        )

        frequency_oof[
            inner_valid_idx
        ] = np.clip(
            frequency_model.predict(
                X_inner_valid
            ),
            0.0,
            None,
        )

        value_oof[
            inner_valid_idx
        ] = np.clip(
            value_model.predict(
                X_inner_valid
            ),
            0.0,
            None,
        )

    if (
        np.isnan(
            raw_return_oof
        ).any()
        or
        np.isnan(
            frequency_oof
        ).any()
        or
        np.isnan(
            value_oof
        ).any()
    ):
        raise AssertionError(
            "Inner OOF predictions are incomplete."
        )

    return {
        "raw_return_probability":
            raw_return_oof,

        "frequency":
            frequency_oof,

        "value":
            value_oof,
    }


def fit_platt_calibrator(
    raw_probability,
    y_active,
):

    calibrator = LogisticRegression(
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    calibrator.fit(
        np.asarray(
            raw_probability
        ).reshape(
            -1,
            1,
        ),
        np.asarray(
            y_active,
            dtype=int,
        ),
    )

    return calibrator


def calibrated_probability(
    calibrator,
    raw_probability,
):

    return (
        calibrator
        .predict_proba(
            np.asarray(
                raw_probability
            ).reshape(
                -1,
                1,
            )
        )[:, 1]
    )


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
        float(
            scale_constant
        )
    )


def main():

    print(
        "Loading Phase 06R-B2b inputs..."
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

    naive_results = pd.read_csv(
        NAIVE_RESULTS_PATH
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

    if len(
        features
    ) != 276:

        raise AssertionError(
            "Expected 276 selected features."
        )

    bad_features = [
        feature
        for feature
        in features
        if not pd.api.types.is_numeric_dtype(
            supervised[
                feature
            ]
        )
    ]

    if bad_features:
        raise AssertionError(
            "Non-numeric selected features: "
            +
            str(
                bad_features
            )
        )

    data = supervised.merge(
        aux,
        on=[
            "ID",
            "cutoff",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(
        data
    ) != 22679:

        raise AssertionError(
            "Expected 22,679 historical snapshots."
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

    result_rows = []
    prediction_frames = []

    for (
        fold_id,
        training_cutoffs,
        validation_cutoff,
    ) in FOLDS:

        print()
        print(
            "=" * 78
        )

        print(
            f"{fold_id}: {training_cutoffs} -> {validation_cutoff}"
        )

        print(
            "=" * 78
        )

        train = data[
            data[
                "cutoff"
            ].isin(
                training_cutoffs
            )
        ].copy()

        valid = data[
            data[
                "cutoff"
            ].eq(
                validation_cutoff
            )
        ].copy()

        for target_name, spec in targets.items():

            print()
            print(
                f"{fold_id} | {target_name}"
            )

            # ----------------------------------------------------
            # Inner cross-validated predictions on outer training
            # ----------------------------------------------------

            inner = make_inner_oof(
                train,
                features,
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

            y_train_active = (
                train[
                    spec[
                        "active"
                    ]
                ]
                .astype(int)
                .to_numpy()
            )

            # ----------------------------------------------------
            # Probability calibration on original population
            # ----------------------------------------------------

            calibrator = fit_platt_calibrator(
                inner[
                    "raw_return_probability"
                ],
                y_train_active,
            )

            calibrated_oof = (
                calibrated_probability(
                    calibrator,
                    inner[
                        "raw_return_probability"
                    ],
                )
            )

            conditional_oof = (
                inner[
                    "frequency"
                ]
                *
                inner[
                    "value"
                ]
            )

            # ----------------------------------------------------
            # Paper's CLV-error sigmoid
            # ----------------------------------------------------

            (
                t1,
                t2,
                optimization_error,
            ) = fit_sigmoid(
                calibrated_oof,
                conditional_oof,
                train[
                    spec[
                        "raw"
                    ]
                ].to_numpy(
                    dtype="float64"
                ),
            )

            # ----------------------------------------------------
            # Fit final submodels on complete outer training set
            # ----------------------------------------------------

            (
                X_train,
                X_valid,
                _,
            ) = prepare_split(
                train,
                valid,
                features,
            )

            final_classifier = (
                fit_balanced_classifier(
                    X_train,
                    y_train_active,
                )
            )

            raw_valid_probability = (
                final_classifier
                .predict_proba(
                    X_valid
                )[:, 1]
            )

            calibrated_valid_probability = (
                calibrated_probability(
                    calibrator,
                    raw_valid_probability,
                )
            )

            adjusted_valid_probability = (
                sigmoid_adjust(
                    calibrated_valid_probability,
                    t1,
                    t2,
                )
            )

            active_train_mask = (
                y_train_active
                ==
                1
            )

            frequency_model = (
                rf_regressor()
            )

            frequency_model.fit(
                X_train[
                    active_train_mask
                ],
                train.loc[
                    train[
                        spec[
                            "active"
                        ]
                    ].eq(1),
                    spec[
                        "frequency"
                    ],
                ].to_numpy(
                    dtype="float64"
                ),
            )

            value_model = (
                rf_regressor()
            )

            value_model.fit(
                X_train[
                    active_train_mask
                ],
                train.loc[
                    train[
                        spec[
                            "active"
                        ]
                    ].eq(1),
                    spec[
                        "value"
                    ],
                ].to_numpy(
                    dtype="float64"
                ),
            )

            predicted_frequency = np.clip(
                frequency_model.predict(
                    X_valid
                ),
                0.0,
                None,
            )

            predicted_value = np.clip(
                value_model.predict(
                    X_valid
                ),
                0.0,
                None,
            )

            conditional_valid = (
                predicted_frequency
                *
                predicted_value
            )

            enhanced_raw = (
                adjusted_valid_probability
                *
                conditional_valid
            )

            enhanced_z = transformed_from_raw(
                enhanced_raw,
                spec[
                    "scale"
                ],
            )

            actual_z = (
                valid[
                    spec[
                        "official_z"
                    ]
                ]
                .to_numpy(
                    dtype="float64"
                )
            )

            enhanced_rmse = rmse(
                actual_z,
                enhanced_z,
            )

            previous = naive_results[
                naive_results[
                    "fold_id"
                ].eq(
                    fold_id
                )
                &
                naive_results[
                    "target"
                ].eq(
                    target_name
                )
            ]

            if len(
                previous
            ) != 1:

                raise AssertionError(
                    "Could not identify unique B2 result."
                )

            direct_rmse = float(
                previous.iloc[
                    0
                ][
                    "direct_rf_rmse"
                ]
            )

            naive_rmse = float(
                previous.iloc[
                    0
                ][
                    "multistage_rf_rmse"
                ]
            )

            y_valid_active = (
                valid[
                    spec[
                        "active"
                    ]
                ]
                .astype(int)
                .to_numpy()
            )

            activity_auc = roc_auc_score(
                y_valid_active,
                calibrated_valid_probability,
            )

            brier = brier_score_loss(
                y_valid_active,
                calibrated_valid_probability,
            )

            print(
                f"Direct RF RMSE     : {direct_rmse:.6f}"
            )

            print(
                f"Naive M-RF RMSE    : {naive_rmse:.6f}"
            )

            print(
                f"Enhanced M-RF RMSE : {enhanced_rmse:.6f}"
            )

            print(
                f"vs naive           : "
                f"{enhanced_rmse - naive_rmse:+.6f}"
            )

            print(
                f"vs direct          : "
                f"{enhanced_rmse - direct_rmse:+.6f}"
            )

            print(
                f"t1 / t2            : "
                f"{t1:.6f} / {t2:.6f}"
            )

            print(
                f"Activity AUC        : "
                f"{activity_auc:.6f}"
            )

            result_rows.append(
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

                    "direct_rf_rmse":
                        direct_rmse,

                    "naive_multistage_rmse":
                        naive_rmse,

                    "enhanced_multistage_rmse":
                        enhanced_rmse,

                    "enhanced_change_vs_naive":
                        enhanced_rmse
                        -
                        naive_rmse,

                    "enhanced_change_vs_direct":
                        enhanced_rmse
                        -
                        direct_rmse,

                    "t1":
                        t1,

                    "t2":
                        t2,

                    "training_total_value_error":
                        optimization_error,

                    "calibrated_activity_auc":
                        activity_auc,

                    "calibrated_activity_brier":
                        brier,
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
                            actual_z,

                        "raw_return_probability":
                            raw_valid_probability,

                        "calibrated_return_probability":
                            calibrated_valid_probability,

                        "adjusted_return_probability":
                            adjusted_valid_probability,

                        "predicted_frequency":
                            predicted_frequency,

                        "predicted_value_per_basket":
                            predicted_value,

                        "enhanced_raw_prediction":
                            enhanced_raw,

                        "enhanced_z_prediction":
                            enhanced_z,
                    }
                )
            )

    results = pd.DataFrame(
        result_rows
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

        subset = results[
            results[
                "target"
            ].eq(
                target_name
            )
        ].copy()

        weights = subset[
            "validation_rows"
        ].to_numpy(
            dtype="float64"
        )

        direct_weighted = float(
            np.average(
                subset[
                    "direct_rf_rmse"
                ],
                weights=weights,
            )
        )

        naive_weighted = float(
            np.average(
                subset[
                    "naive_multistage_rmse"
                ],
                weights=weights,
            )
        )

        enhanced_weighted = float(
            np.average(
                subset[
                    "enhanced_multistage_rmse"
                ],
                weights=weights,
            )
        )

        folds_better_naive = int(
            (
                subset[
                    "enhanced_multistage_rmse"
                ]
                <
                subset[
                    "naive_multistage_rmse"
                ]
            ).sum()
        )

        folds_better_direct = int(
            (
                subset[
                    "enhanced_multistage_rmse"
                ]
                <
                subset[
                    "direct_rf_rmse"
                ]
            ).sum()
        )

        summary_rows.append(
            {
                "target":
                    target_name,

                "row_weighted_direct_rf_rmse":
                    direct_weighted,

                "row_weighted_naive_multistage_rmse":
                    naive_weighted,

                "row_weighted_enhanced_multistage_rmse":
                    enhanced_weighted,

                "folds_enhanced_better_than_naive":
                    folds_better_naive,

                "folds_enhanced_better_than_direct":
                    folds_better_direct,

                "enhancement_replication_pass":
                    int(
                        folds_better_naive >= 3
                        and
                        enhanced_weighted
                        <
                        naive_weighted
                    ),

                "strong_production_signal":
                    int(
                        folds_better_direct >= 3
                        and
                        enhanced_weighted
                        <
                        direct_weighted
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    results.to_csv(
        OUTPUT_DIR
        / "fold_results.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "PHASE 06R-B2b ENHANCED MULTI-STAGE RF COMPLETE"
    )

    print(
        "=" * 78
    )

    print()

    print(
        results.to_string(
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


if __name__ == "__main__":
    main()
