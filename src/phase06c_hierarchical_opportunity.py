"""
Phase 06C: Rule-aware hierarchical Opportunity classifier.

Architecture:

    Macro:
        Growth
        Adoption
        Inactivity
        Stable

    Growth subtype:
        4 official Growth labels

    Adoption subtype:
        11 official Adoption labels

Final 17-class probabilities are composed as:

    P(growth_j) =
        P(Growth) * P(growth_j | Growth)

    P(adoption_k) =
        P(Adoption) * P(adoption_k | Adoption)

    P(Inactivity) =
        P(Inactivity)

    P(Stable) =
        P(Stable)

The experiment uses the accepted 276 predictors and alpha=0.50
class-balancing principle.

No temporal recency weighting is used for Opportunity.
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

PHASE04_DIR = ROOT / "data" / "processed" / "phase04"
PHASE06_DIR = ROOT / "data" / "processed" / "phase06"

SUPERVISED_PATH = PHASE04_DIR / "supervised_historical.csv"
FEATURE_MANIFEST_PATH = PHASE04_DIR / "model_feature_manifest.csv"
FOLD_MEMBERSHIP_PATH = PHASE04_DIR / "temporal_fold_membership.csv"
CLASS_MAP_PATH = PHASE04_DIR / "opportunity_class_map.csv"

BASELINE_CONFIG_PATH = ROOT / "configs" / "lightgbm_baseline.json"
FINAL_SPEC_PATH = ROOT / "configs" / "final_model_spec.json"

CONTROL_PATH = (
    PHASE04_DIR
    / "opportunity_weighting"
    / "fold_comparison.csv"
)

OUTPUT_DIR = PHASE06_DIR / "hierarchical_opportunity"

MODEL_DIR = (
    ROOT
    / "models"
    / "phase06_hierarchical_opportunity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


WEIGHT_EXPONENT = 0.50

MACRO_ORDER = [
    "Growth",
    "Adoption",
    "Inactivity",
    "Stable",
]


def macro_label(label):

    if label == "Inactivity":
        return "Inactivity"

    if label == "Stable":
        return "Stable"

    if label.startswith(
        "Existing-category growth:"
    ):
        return "Growth"

    if label.startswith(
        "New category adoption:"
    ):
        return "Adoption"

    raise ValueError(
        f"Unknown Opportunity label: {label}"
    )


def normalized_class_weights(
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
            "At least one class is absent "
            "from a stage training population."
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

    sample_weights = (
        raw_class_weights[
            labels
        ]
    )

    sample_weights = (
        sample_weights
        /
        sample_weights.mean()
    )

    if not np.isclose(
        sample_weights.mean(),
        1.0,
        atol=1e-12,
    ):

        raise AssertionError(
            "Stage sample weights do not "
            "normalize to mean one."
        )

    return sample_weights


def weighted_f1_eval(
    num_classes,
):

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


def parameters_for_stage(
    config,
    num_classes,
):

    parameters = dict(
        config[
            "common_parameters"
        ]
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
    ] = 31

    parameters[
        "min_data_in_leaf"
    ] = 50

    parameters[
        "num_class"
    ] = int(
        num_classes
    )

    # Make the custom weighted-F1 metric
    # authoritative for early stopping.
    parameters[
        "metric"
    ] = "None"

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


def train_stage(
    X_train,
    y_train,
    X_valid,
    y_valid,
    feature_names,
    config,
    num_classes,
):

    sample_weights = (
        normalized_class_weights(
            y_train,
            num_classes,
            WEIGHT_EXPONENT,
        )
    )

    train_dataset = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sample_weights,
        feature_name=feature_names,
        free_raw_data=False,
    )

    valid_dataset = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=train_dataset,
        feature_name=feature_names,
        free_raw_data=False,
    )

    model = lgb.train(
        parameters_for_stage(
            config,
            num_classes,
        ),
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

    return model


def main():

    print(
        "Loading Phase 06C inputs..."
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
            "Phase 04 specification is not frozen."
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

    class_map = (
        pd.read_csv(
            CLASS_MAP_PATH
        )
        .sort_values(
            "class_id"
        )
    )

    control = pd.read_csv(
        CONTROL_PATH
    )

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
            "Expected 276 accepted predictors."
        )

    if len(class_map) != 17:

        raise AssertionError(
            "Expected 17 final Opportunity classes."
        )

    opportunity_spec = spec[
        "opportunity"
    ]

    if int(
        opportunity_spec[
            "num_leaves"
        ]
    ) != 31:

        raise AssertionError(
            "Expected Opportunity num_leaves=31."
        )

    if int(
        opportunity_spec[
            "min_data_in_leaf"
        ]
    ) != 50:

        raise AssertionError(
            "Expected Opportunity min_data_in_leaf=50."
        )

    if not np.isclose(
        float(
            opportunity_spec[
                "class_weight_exponent"
            ]
        ),
        WEIGHT_EXPONENT,
    ):

        raise AssertionError(
            "Expected accepted weight exponent 0.50."
        )

    # ============================================================
    # FINAL CLASS STRUCTURE
    # ============================================================

    final_labels = (
        class_map[
            "Opportunity"
        ]
        .astype(str)
        .tolist()
    )

    final_class_ids = (
        class_map[
            "class_id"
        ]
        .astype(int)
        .tolist()
    )

    final_id_to_label = dict(
        zip(
            final_class_ids,
            final_labels,
        )
    )

    final_label_to_id = {
        label: class_id
        for class_id, label
        in final_id_to_label.items()
    }

    growth_labels = [
        label
        for label in final_labels
        if label.startswith(
            "Existing-category growth:"
        )
    ]

    adoption_labels = [
        label
        for label in final_labels
        if label.startswith(
            "New category adoption:"
        )
    ]

    if len(growth_labels) != 4:

        raise AssertionError(
            "Expected four Growth classes."
        )

    if len(adoption_labels) != 11:

        raise AssertionError(
            "Expected eleven Adoption classes."
        )

    macro_to_id = {
        label: index
        for index, label
        in enumerate(
            MACRO_ORDER
        )
    }

    growth_to_id = {
        label: index
        for index, label
        in enumerate(
            growth_labels
        )
    }

    adoption_to_id = {
        label: index
        for index, label
        in enumerate(
            adoption_labels
        )
    }

    # ============================================================
    # CONTROL LOOKUP
    # ============================================================

    control_lookup = {
        row[
            "fold_id"
        ]:
            float(
                row[
                    "weighted_model_f1"
                ]
            )
        for _, row
        in control.iterrows()
        if row[
            "fold_id"
        ] in [
            "F2",
            "F3",
            "F4",
        ]
    }

    folds = [
        "F2",
        "F3",
        "F4",
    ]

    metric_rows = []
    class_metric_rows = []
    macro_metric_rows = []

    # ============================================================
    # TEMPORAL VALIDATION
    # ============================================================

    for fold_id in folds:

        print()
        print(
            "=" * 76
        )

        print(
            f"Training hierarchical Opportunity: "
            f"{fold_id}"
        )

        print(
            "=" * 76
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

        if len(train) != len(train_ids):

            raise AssertionError(
                f"{fold_id}: train membership mismatch."
            )

        if len(valid) != len(valid_ids):

            raise AssertionError(
                f"{fold_id}: validation membership mismatch."
            )

        X_train = train[
            features
        ]

        X_valid = valid[
            features
        ]

        train_labels = (
            train[
                "Opportunity"
            ]
            .astype(str)
        )

        valid_labels = (
            valid[
                "Opportunity"
            ]
            .astype(str)
        )

        # ========================================================
        # STAGE 1: MACRO MODEL
        # ========================================================

        y_macro_train = np.asarray(
            [
                macro_to_id[
                    macro_label(
                        label
                    )
                ]
                for label
                in train_labels
            ],
            dtype="int64",
        )

        y_macro_valid = np.asarray(
            [
                macro_to_id[
                    macro_label(
                        label
                    )
                ]
                for label
                in valid_labels
            ],
            dtype="int64",
        )

        macro_model = train_stage(
            X_train,
            y_macro_train,
            X_valid,
            y_macro_valid,
            features,
            config,
            len(
                MACRO_ORDER
            ),
        )

        macro_probabilities = (
            macro_model.predict(
                X_valid,
                num_iteration=
                    macro_model.best_iteration,
            )
        )

        # ========================================================
        # STAGE 2: GROWTH SUBTYPE
        # ========================================================

        growth_train_mask = (
            train_labels
            .isin(
                growth_labels
            )
            .to_numpy()
        )

        growth_valid_mask = (
            valid_labels
            .isin(
                growth_labels
            )
            .to_numpy()
        )

        y_growth_train = np.asarray(
            [
                growth_to_id[
                    label
                ]
                for label
                in train_labels[
                    growth_train_mask
                ]
            ],
            dtype="int64",
        )

        y_growth_valid = np.asarray(
            [
                growth_to_id[
                    label
                ]
                for label
                in valid_labels[
                    growth_valid_mask
                ]
            ],
            dtype="int64",
        )

        growth_model = train_stage(
            X_train.loc[
                growth_train_mask
            ],
            y_growth_train,
            X_valid.loc[
                growth_valid_mask
            ],
            y_growth_valid,
            features,
            config,
            len(
                growth_labels
            ),
        )

        # Conditional subtype probabilities
        # are required for all validation rows.
        growth_probabilities = (
            growth_model.predict(
                X_valid,
                num_iteration=
                    growth_model.best_iteration,
            )
        )

        # ========================================================
        # STAGE 3: ADOPTION SUBTYPE
        # ========================================================

        adoption_train_mask = (
            train_labels
            .isin(
                adoption_labels
            )
            .to_numpy()
        )

        adoption_valid_mask = (
            valid_labels
            .isin(
                adoption_labels
            )
            .to_numpy()
        )

        y_adoption_train = np.asarray(
            [
                adoption_to_id[
                    label
                ]
                for label
                in train_labels[
                    adoption_train_mask
                ]
            ],
            dtype="int64",
        )

        y_adoption_valid = np.asarray(
            [
                adoption_to_id[
                    label
                ]
                for label
                in valid_labels[
                    adoption_valid_mask
                ]
            ],
            dtype="int64",
        )

        adoption_model = train_stage(
            X_train.loc[
                adoption_train_mask
            ],
            y_adoption_train,
            X_valid.loc[
                adoption_valid_mask
            ],
            y_adoption_valid,
            features,
            config,
            len(
                adoption_labels
            ),
        )

        adoption_probabilities = (
            adoption_model.predict(
                X_valid,
                num_iteration=
                    adoption_model.best_iteration,
            )
        )

        # ========================================================
        # COMPOSE FINAL 17-CLASS PROBABILITIES
        # ========================================================

        final_probabilities = np.zeros(
            (
                len(valid),
                17,
            ),
            dtype="float64",
        )

        growth_macro_probability = (
            macro_probabilities[
                :,
                macro_to_id[
                    "Growth"
                ],
            ]
        )

        adoption_macro_probability = (
            macro_probabilities[
                :,
                macro_to_id[
                    "Adoption"
                ],
            ]
        )

        inactivity_id = (
            final_label_to_id[
                "Inactivity"
            ]
        )

        stable_id = (
            final_label_to_id[
                "Stable"
            ]
        )

        final_probabilities[
            :,
            inactivity_id
        ] = (
            macro_probabilities[
                :,
                macro_to_id[
                    "Inactivity"
                ],
            ]
        )

        final_probabilities[
            :,
            stable_id
        ] = (
            macro_probabilities[
                :,
                macro_to_id[
                    "Stable"
                ],
            ]
        )

        for label in growth_labels:

            final_id = (
                final_label_to_id[
                    label
                ]
            )

            subtype_id = (
                growth_to_id[
                    label
                ]
            )

            final_probabilities[
                :,
                final_id
            ] = (
                growth_macro_probability
                *
                growth_probabilities[
                    :,
                    subtype_id
                ]
            )

        for label in adoption_labels:

            final_id = (
                final_label_to_id[
                    label
                ]
            )

            subtype_id = (
                adoption_to_id[
                    label
                ]
            )

            final_probabilities[
                :,
                final_id
            ] = (
                adoption_macro_probability
                *
                adoption_probabilities[
                    :,
                    subtype_id
                ]
            )

        row_sums = (
            final_probabilities.sum(
                axis=1
            )
        )

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-6,
        ):

            raise AssertionError(
                f"{fold_id}: composed probabilities "
                "do not sum to one."
            )

        final_prediction = np.argmax(
            final_probabilities,
            axis=1,
        ).astype(
            "int64"
        )

        y_final_valid = valid[
            "Opportunity_class_id"
        ].to_numpy(
            dtype="int64"
        )

        hierarchical_f1 = f1_score(
            y_final_valid,
            final_prediction,
            average="weighted",
            zero_division=0,
        )

        control_f1 = (
            control_lookup[
                fold_id
            ]
        )

        macro_prediction = np.argmax(
            macro_probabilities,
            axis=1,
        )

        macro_f1 = f1_score(
            y_macro_valid,
            macro_prediction,
            average="weighted",
            zero_division=0,
        )

        growth_prediction = np.argmax(
            growth_model.predict(
                X_valid.loc[
                    growth_valid_mask
                ],
                num_iteration=
                    growth_model.best_iteration,
            ),
            axis=1,
        )

        growth_f1 = f1_score(
            y_growth_valid,
            growth_prediction,
            average="weighted",
            zero_division=0,
        )

        adoption_prediction = np.argmax(
            adoption_model.predict(
                X_valid.loc[
                    adoption_valid_mask
                ],
                num_iteration=
                    adoption_model.best_iteration,
            ),
            axis=1,
        )

        adoption_f1 = f1_score(
            y_adoption_valid,
            adoption_prediction,
            average="weighted",
            zero_division=0,
        )

        metric_rows.append(
            {
                "fold_id":
                    fold_id,

                "train_rows":
                    len(train),

                "validation_rows":
                    len(valid),

                "control_weighted_f1":
                    control_f1,

                "hierarchical_weighted_f1":
                    float(
                        hierarchical_f1
                    ),

                "weighted_f1_change":
                    float(
                        hierarchical_f1
                        -
                        control_f1
                    ),

                "macro_stage_weighted_f1":
                    float(
                        macro_f1
                    ),

                "growth_stage_weighted_f1":
                    float(
                        growth_f1
                    ),

                "adoption_stage_weighted_f1":
                    float(
                        adoption_f1
                    ),

                "macro_best_iteration":
                    int(
                        macro_model.best_iteration
                    ),

                "growth_best_iteration":
                    int(
                        growth_model.best_iteration
                    ),

                "adoption_best_iteration":
                    int(
                        adoption_model.best_iteration
                    ),

                "predicted_classes":
                    int(
                        np.unique(
                            final_prediction
                        ).size
                    ),

                "improved":
                    int(
                        hierarchical_f1
                        >
                        control_f1
                    ),
            }
        )

        # ========================================================
        # FINAL CLASS DIAGNOSTICS
        # ========================================================

        (
            precision,
            recall,
            f1_values,
            support,
        ) = precision_recall_fscore_support(
            y_final_valid,
            final_prediction,
            labels=np.arange(
                17
            ),
            zero_division=0,
        )

        for class_id in range(
            17
        ):

            class_metric_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "class_id":
                        class_id,

                    "Opportunity":
                        final_id_to_label[
                            class_id
                        ],

                    "macro":
                        macro_label(
                            final_id_to_label[
                                class_id
                            ]
                        ),

                    "support":
                        int(
                            support[
                                class_id
                            ]
                        ),

                    "predicted_count":
                        int(
                            (
                                final_prediction
                                ==
                                class_id
                            ).sum()
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

        true_macro_text = np.asarray(
            [
                macro_label(
                    label
                )
                for label
                in valid_labels
            ]
        )

        pred_final_labels = np.asarray(
            [
                final_id_to_label[
                    int(value)
                ]
                for value
                in final_prediction
            ]
        )

        pred_macro_text = np.asarray(
            [
                macro_label(
                    label
                )
                for label
                in pred_final_labels
            ]
        )

        (
            macro_precision,
            macro_recall,
            macro_final_f1,
            macro_support,
        ) = precision_recall_fscore_support(
            true_macro_text,
            pred_macro_text,
            labels=
                MACRO_ORDER,
            zero_division=0,
        )

        for index, macro in enumerate(
            MACRO_ORDER
        ):

            macro_metric_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "macro":
                        macro,

                    "support":
                        int(
                            macro_support[
                                index
                            ]
                        ),

                    "precision":
                        float(
                            macro_precision[
                                index
                            ]
                        ),

                    "recall":
                        float(
                            macro_recall[
                                index
                            ]
                        ),

                    "f1":
                        float(
                            macro_final_f1[
                                index
                            ]
                        ),
                }
            )

        macro_model.save_model(
            str(
                MODEL_DIR
                /
                f"{fold_id}_macro.txt"
            )
        )

        growth_model.save_model(
            str(
                MODEL_DIR
                /
                f"{fold_id}_growth_subtype.txt"
            )
        )

        adoption_model.save_model(
            str(
                MODEL_DIR
                /
                f"{fold_id}_adoption_subtype.txt"
            )
        )

        print(
            f"{fold_id} | "
            f"control F1: "
            f"{control_f1:.6f} | "
            f"hierarchical F1: "
            f"{hierarchical_f1:.6f} | "
            f"change: "
            f"{hierarchical_f1-control_f1:+.6f}"
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    metrics = pd.DataFrame(
        metric_rows
    )

    class_metrics = pd.DataFrame(
        class_metric_rows
    )

    macro_metrics = pd.DataFrame(
        macro_metric_rows
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

    control_temporal = float(
        np.sum(
            metrics[
                "control_weighted_f1"
            ]
            *
            validation_weights
        )
    )

    hierarchical_temporal = float(
        np.sum(
            metrics[
                "hierarchical_weighted_f1"
            ]
            *
            validation_weights
        )
    )

    folds_improved = int(
        metrics[
            "improved"
        ].sum()
    )

    f4 = metrics[
        metrics[
            "fold_id"
        ].eq(
            "F4"
        )
    ].iloc[0]

    criterion_temporal = (
        hierarchical_temporal
        >
        control_temporal
    )

    criterion_f4 = (
        float(
            f4[
                "hierarchical_weighted_f1"
            ]
        )
        >=
        float(
            f4[
                "control_weighted_f1"
            ]
        )
    )

    criterion_folds = (
        folds_improved >= 2
    )

    accepted = (
        criterion_temporal
        and
        criterion_f4
        and
        criterion_folds
    )

    summary = pd.DataFrame(
        [
            {
                "folds_evaluated":
                    len(metrics),

                "folds_improved":
                    folds_improved,

                "row_weighted_control_f1":
                    control_temporal,

                "row_weighted_hierarchical_f1":
                    hierarchical_temporal,

                "row_weighted_f1_change":
                    (
                        hierarchical_temporal
                        -
                        control_temporal
                    ),

                "f4_control_f1":
                    float(
                        f4[
                            "control_weighted_f1"
                        ]
                    ),

                "f4_hierarchical_f1":
                    float(
                        f4[
                            "hierarchical_weighted_f1"
                        ]
                    ),

                "f4_f1_change":
                    float(
                        f4[
                            "hierarchical_weighted_f1"
                        ]
                        -
                        f4[
                            "control_weighted_f1"
                        ]
                    ),

                "f4_predicted_classes":
                    int(
                        f4[
                            "predicted_classes"
                        ]
                    ),

                "criterion_temporal_improved":
                    int(
                        criterion_temporal
                    ),

                "criterion_f4_not_worse":
                    int(
                        criterion_f4
                    ),

                "criterion_two_of_three_folds":
                    int(
                        criterion_folds
                    ),

                "accepted_by_preregistered_rule":
                    int(
                        accepted
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

    class_metrics.to_csv(
        OUTPUT_DIR
        / "class_metrics.csv",
        index=False,
    )

    macro_metrics.to_csv(
        OUTPUT_DIR
        / "macro_metrics.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 06C HIERARCHICAL OPPORTUNITY EXPERIMENT COMPLETE"
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
        "F4 CLASS METRICS"
    )
    print()

    print(
        class_metrics[
            class_metrics[
                "fold_id"
            ].eq(
                "F4"
            )
        ]
        .sort_values(
            "support",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "F4 MACRO METRICS"
    )
    print()

    print(
        macro_metrics[
            macro_metrics[
                "fold_id"
            ].eq(
                "F4"
            )
        ].to_string(
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
