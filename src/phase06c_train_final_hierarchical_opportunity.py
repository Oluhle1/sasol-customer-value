"""
Final Phase 06C hierarchical Opportunity model.

This is an experimental leaderboard submission.

Fuel and non-fuel predictions are NOT regenerated here.
They remain exactly those from the Phase 05 S0 submission.

Only Opportunity is replaced.

Final stage boosting rounds are taken from the Phase 06C F4 validation
result:

    macro = 19
    growth subtype = 113
    adoption subtype = 7
"""

from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PHASE04_DIR = ROOT / "data" / "processed" / "phase04"

TRAIN_PATH = (
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

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "phase06c_final"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "phase06c_final"
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

MACRO_ROUNDS = 19
GROWTH_ROUNDS = 113
ADOPTION_ROUNDS = 7

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


def stage_weights(
    labels,
    num_classes,
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
            "A stage is missing a required class."
        )

    n = len(labels)

    class_weights = (
        n
        /
        (
            num_classes
            * counts
        )
    ) ** WEIGHT_EXPONENT

    sample_weights = (
        class_weights[
            labels
        ]
    )

    sample_weights = (
        sample_weights
        /
        sample_weights.mean()
    )

    return sample_weights


def stage_parameters(
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
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        }
    )

    return parameters


def train_stage(
    X,
    y,
    features,
    config,
    num_classes,
    rounds,
):

    weights = stage_weights(
        y,
        num_classes,
    )

    dataset = lgb.Dataset(
        X,
        label=y,
        weight=weights,
        feature_name=features,
        free_raw_data=False,
    )

    model = lgb.train(
        stage_parameters(
            config,
            num_classes,
        ),
        dataset,
        num_boost_round=int(
            rounds
        ),
        callbacks=[
            lgb.log_evaluation(
                period=0
            )
        ],
    )

    return model


def main():

    print(
        "Loading final Phase 06C inputs..."
    )

    config = json.loads(
        BASELINE_CONFIG_PATH.read_text()
    )

    train = pd.read_csv(
        TRAIN_PATH,
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

    class_map = (
        pd.read_csv(
            CLASS_MAP_PATH
        )
        .sort_values(
            "class_id"
        )
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

    if len(
        features
    ) != 276:

        raise AssertionError(
            "Expected 276 features."
        )

    if len(
        train
    ) != 22679:

        raise AssertionError(
            "Expected 22,679 training rows."
        )

    if len(
        competition
    ) != 5488:

        raise AssertionError(
            "Expected 5,488 competition rows."
        )

    if len(
        class_map
    ) != 17:

        raise AssertionError(
            "Expected 17 Opportunity classes."
        )

    X_train = train[
        features
    ]

    X_comp = competition[
        features
    ]

    final_labels = (
        class_map[
            "Opportunity"
        ]
        .astype(str)
        .tolist()
    )

    final_ids = (
        class_map[
            "class_id"
        ]
        .astype(int)
        .tolist()
    )

    final_id_to_label = dict(
        zip(
            final_ids,
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

    if len(
        growth_labels
    ) != 4:

        raise AssertionError(
            "Expected 4 Growth classes."
        )

    if len(
        adoption_labels
    ) != 11:

        raise AssertionError(
            "Expected 11 Adoption classes."
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

    train_labels = (
        train[
            "Opportunity"
        ]
        .astype(str)
    )

    # ============================================================
    # MACRO
    # ============================================================

    print(
        "Training final 06C macro model..."
    )

    y_macro = np.asarray(
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

    macro_model = train_stage(
        X_train,
        y_macro,
        features,
        config,
        4,
        MACRO_ROUNDS,
    )

    macro_prob = (
        macro_model.predict(
            X_comp,
            num_iteration=
                MACRO_ROUNDS,
        )
    )

    # ============================================================
    # GROWTH
    # ============================================================

    print(
        "Training final 06C Growth subtype model..."
    )

    growth_mask = (
        train_labels
        .isin(
            growth_labels
        )
        .to_numpy()
    )

    y_growth = np.asarray(
        [
            growth_to_id[
                label
            ]
            for label
            in train_labels[
                growth_mask
            ]
        ],
        dtype="int64",
    )

    growth_model = train_stage(
        X_train.loc[
            growth_mask
        ],
        y_growth,
        features,
        config,
        4,
        GROWTH_ROUNDS,
    )

    growth_prob = (
        growth_model.predict(
            X_comp,
            num_iteration=
                GROWTH_ROUNDS,
        )
    )

    # ============================================================
    # ADOPTION
    # ============================================================

    print(
        "Training final 06C Adoption subtype model..."
    )

    adoption_mask = (
        train_labels
        .isin(
            adoption_labels
        )
        .to_numpy()
    )

    y_adoption = np.asarray(
        [
            adoption_to_id[
                label
            ]
            for label
            in train_labels[
                adoption_mask
            ]
        ],
        dtype="int64",
    )

    adoption_model = train_stage(
        X_train.loc[
            adoption_mask
        ],
        y_adoption,
        features,
        config,
        11,
        ADOPTION_ROUNDS,
    )

    adoption_prob = (
        adoption_model.predict(
            X_comp,
            num_iteration=
                ADOPTION_ROUNDS,
        )
    )

    # ============================================================
    # COMPOSE 17-CLASS PROBABILITIES
    # ============================================================

    final_prob = np.zeros(
        (
            len(
                competition
            ),
            17,
        ),
        dtype="float64",
    )

    growth_macro_prob = (
        macro_prob[
            :,
            macro_to_id[
                "Growth"
            ],
        ]
    )

    adoption_macro_prob = (
        macro_prob[
            :,
            macro_to_id[
                "Adoption"
            ],
        ]
    )

    final_prob[
        :,
        final_label_to_id[
            "Inactivity"
        ]
    ] = (
        macro_prob[
            :,
            macro_to_id[
                "Inactivity"
            ],
        ]
    )

    final_prob[
        :,
        final_label_to_id[
            "Stable"
        ]
    ] = (
        macro_prob[
            :,
            macro_to_id[
                "Stable"
            ],
        ]
    )

    for label in growth_labels:

        final_prob[
            :,
            final_label_to_id[
                label
            ]
        ] = (
            growth_macro_prob
            *
            growth_prob[
                :,
                growth_to_id[
                    label
                ]
            ]
        )

    for label in adoption_labels:

        final_prob[
            :,
            final_label_to_id[
                label
            ]
        ] = (
            adoption_macro_prob
            *
            adoption_prob[
                :,
                adoption_to_id[
                    label
                ]
            ]
        )

    row_sum = (
        final_prob.sum(
            axis=1
        )
    )

    if not np.allclose(
        row_sum,
        1.0,
        atol=1e-6,
    ):

        raise AssertionError(
            "Final hierarchical probabilities "
            "do not sum to one."
        )

    pred_class = np.argmax(
        final_prob,
        axis=1,
    ).astype(
        "int64"
    )

    pred_label = [
        final_id_to_label[
            int(value)
        ]
        for value
        in pred_class
    ]

    output = competition[
        [
            "ID",
            "cutoff",
        ]
    ].copy()

    output[
        "Opportunity_class_id"
    ] = pred_class

    output[
        "Opportunity"
    ] = pred_label

    output.to_csv(
        OUTPUT_DIR
        / "hierarchical_opportunity_predictions.csv",
        index=False,
    )

    distribution = (
        output[
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

    distribution[
        "predicted_share"
    ] = (
        distribution[
            "predicted_count"
        ]
        /
        len(
            output
        )
    )

    distribution.to_csv(
        OUTPUT_DIR
        / "hierarchical_opportunity_distribution.csv",
        index=False,
    )

    macro_model.save_model(
        str(
            MODEL_DIR
            / "final_macro.txt"
        )
    )

    growth_model.save_model(
        str(
            MODEL_DIR
            / "final_growth_subtype.txt"
        )
    )

    adoption_model.save_model(
        str(
            MODEL_DIR
            / "final_adoption_subtype.txt"
        )
    )

    print()
    print(
        "=" * 76
    )

    print(
        "FINAL PHASE 06C OPPORTUNITY TRAINING COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Rounds:"
    )

    print(
        f"Macro    : {MACRO_ROUNDS}"
    )

    print(
        f"Growth   : {GROWTH_ROUNDS}"
    )

    print(
        f"Adoption : {ADOPTION_ROUNDS}"
    )

    print()

    print(
        "Predicted classes:"
    )

    print(
        output[
            "Opportunity"
        ].nunique()
    )

    print()

    print(
        "OPPORTUNITY DISTRIBUTION"
    )

    print()

    print(
        distribution.to_string(
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
