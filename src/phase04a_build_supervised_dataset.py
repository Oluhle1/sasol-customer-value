"""
Phase 04A: Supervised dataset assembly and temporal validation design.

Inputs:
- Phase 03D canonical feature matrix;
- Phase 03 frozen model feature policy;
- Phase 02 transformed labels;
- temporal validation configuration.

Outputs:
- modelling-ready historical dataset;
- competition feature matrix;
- selected feature manifest;
- Opportunity class map;
- exact temporal fold membership;
- dataset and fold audits.

No model is trained in this phase.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase03_features"
    / "category_customer_features.csv"
)

MODEL_FEATURE_POLICY_PATH = (
    ROOT
    / "configs"
    / "model_feature_policy.json"
)

TEMPORAL_CONFIG_PATH = (
    ROOT
    / "configs"
    / "temporal_validation.json"
)

LABEL_CONFIG_PATH = (
    ROOT
    / "configs"
    / "label_config.json"
)

PHASE02_LABEL_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase02_labels"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


METADATA_COLUMNS = [
    "snapshot_id",
    "ID",
    "cutoff",
    "snapshot_role",
    "is_labeled",
]

TARGET_COLUMNS = [
    "CLV_fuel",
    "CLV_nonfuel",
    "Opportunity",
]


def load_labels(
    cutoffs,
):
    """
    Load transformed Phase 02 labels and attach forecasting origins.
    """

    frames = []

    audit_rows = []

    for cutoff_text in cutoffs:

        path = (
            PHASE02_LABEL_DIR
            / f"labels_transformed_{cutoff_text}.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        labels = pd.read_csv(
            path,
            dtype={"ID": str},
            keep_default_na=False,
        )

        required = {
            "ID",
            "CLV_fuel",
            "CLV_nonfuel",
            "Opportunity",
        }

        missing = (
            required
            - set(labels.columns)
        )

        if missing:

            raise AssertionError(
                f"{path.name} missing "
                f"columns: {missing}"
            )

        labels = labels[
            [
                "ID",
                "CLV_fuel",
                "CLV_nonfuel",
                "Opportunity",
            ]
        ].copy()

        labels[
            "CLV_fuel"
        ] = pd.to_numeric(
            labels[
                "CLV_fuel"
            ],
            errors="raise",
        )

        labels[
            "CLV_nonfuel"
        ] = pd.to_numeric(
            labels[
                "CLV_nonfuel"
            ],
            errors="raise",
        )

        labels["cutoff"] = (
            pd.Timestamp(
                cutoff_text
            )
        )

        labels[
            "snapshot_id"
        ] = (
            labels["ID"]
            .astype(str)
            +
            "__"
            +
            cutoff_text
        )

        if labels[
            "snapshot_id"
        ].duplicated().any():

            raise AssertionError(
                f"Duplicate label snapshot "
                f"IDs at {cutoff_text}."
            )

        if labels[
            TARGET_COLUMNS
        ].isna().any().any():

            raise AssertionError(
                f"Missing target at "
                f"{cutoff_text}."
            )

        if (
            labels[
                [
                    "CLV_fuel",
                    "CLV_nonfuel",
                ]
            ] < 0
        ).any().any():

            raise AssertionError(
                f"Negative transformed CLV "
                f"target at {cutoff_text}."
            )

        if (
            labels[
                "Opportunity"
            ].eq("")
        ).any():

            raise AssertionError(
                f"Blank Opportunity label "
                f"at {cutoff_text}."
            )

        audit_rows.append(
            {
                "cutoff":
                    cutoff_text,

                "label_rows":
                    len(labels),

                "opportunity_classes":
                    labels[
                        "Opportunity"
                    ].nunique(),

                "fuel_zero_rate":
                    float(
                        labels[
                            "CLV_fuel"
                        ].eq(0).mean()
                    ),

                "nonfuel_zero_rate":
                    float(
                        labels[
                            "CLV_nonfuel"
                        ].eq(0).mean()
                    ),

                "fuel_mean":
                    float(
                        labels[
                            "CLV_fuel"
                        ].mean()
                    ),

                "nonfuel_mean":
                    float(
                        labels[
                            "CLV_nonfuel"
                        ].mean()
                    ),
            }
        )

        frames.append(
            labels
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined[
        "snapshot_id"
    ].duplicated().any():

        raise AssertionError(
            "Duplicate snapshot IDs "
            "across Phase 02 labels."
        )

    return (
        combined,
        pd.DataFrame(
            audit_rows
        ),
    )


def main():

    print(
        "Loading Phase 04 configuration..."
    )

    feature_policy = json.loads(
        MODEL_FEATURE_POLICY_PATH.read_text()
    )

    temporal_config = json.loads(
        TEMPORAL_CONFIG_PATH.read_text()
    )

    label_config = json.loads(
        LABEL_CONFIG_PATH.read_text()
    )

    historical_cutoff_text = (
        label_config[
            "calibration_cutoffs"
        ]
    )

    final_cutoff_text = (
        label_config[
            "public_history_end_exclusive"
        ]
    )

    historical_cutoffs = [
        pd.Timestamp(
            x
        )
        for x in historical_cutoff_text
    ]

    final_cutoff = pd.Timestamp(
        final_cutoff_text
    )

    print(
        "Loading canonical Phase 03 "
        "feature matrix..."
    )

    features = pd.read_csv(
        FEATURE_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    if features[
        "snapshot_id"
    ].duplicated().any():

        raise AssertionError(
            "Duplicate Phase 03 "
            "snapshot IDs."
        )

    # ============================================================
    # FREEZE SELECTED FEATURE SET
    # ============================================================

    candidate_features = [
        column
        for column in features.columns
        if column not in
        METADATA_COLUMNS
    ]

    expected_canonical = int(
        feature_policy[
            "phase03_canonical_feature_count"
        ]
    )

    if len(
        candidate_features
    ) != expected_canonical:

        raise AssertionError(
            f"Expected "
            f"{expected_canonical} "
            f"canonical features, found "
            f"{len(candidate_features)}."
        )

    excluded_features = set(
        feature_policy[
            "excluded_features"
        ].keys()
    )

    missing_exclusions = (
        excluded_features
        - set(candidate_features)
    )

    if missing_exclusions:

        raise AssertionError(
            "Configured exclusions not "
            f"found: {missing_exclusions}"
        )

    selected_features = [
        column
        for column in candidate_features
        if column not in
        excluded_features
    ]

    expected_selected = int(
        feature_policy[
            "baseline_model_feature_count"
        ]
    )

    if len(
        selected_features
    ) != expected_selected:

        raise AssertionError(
            f"Expected "
            f"{expected_selected} "
            f"selected predictors, found "
            f"{len(selected_features)}."
        )

    selected_values = features[
        selected_features
    ].to_numpy(
        dtype="float64"
    )

    if np.isinf(
        selected_values
    ).any():

        raise AssertionError(
            "Selected feature matrix "
            "contains infinity."
        )

    # ============================================================
    # FEATURE MANIFEST
    # ============================================================

    manifest_rows = []

    for position, column in enumerate(
        candidate_features,
        start=1,
    ):

        selected = (
            column
            in selected_features
        )

        exclusion_reason = ""

        retained_equivalent = ""

        if not selected:

            policy = feature_policy[
                "excluded_features"
            ][
                column
            ]

            exclusion_reason = (
                policy[
                    "reason"
                ]
            )

            retained_equivalent = (
                policy[
                    "retained_equivalent"
                ]
            )

        manifest_rows.append(
            {
                "canonical_position":
                    position,

                "feature":
                    column,

                "selected_for_baseline":
                    int(
                        selected
                    ),

                "exclusion_reason":
                    exclusion_reason,

                "retained_equivalent":
                    retained_equivalent,
            }
        )

    feature_manifest = (
        pd.DataFrame(
            manifest_rows
        )
    )

    feature_manifest.to_csv(
        OUTPUT_DIR
        / "model_feature_manifest.csv",
        index=False,
    )

    # ============================================================
    # SPLIT HISTORICAL / COMPETITION FEATURES
    # ============================================================

    historical_features = features[
        features[
            "cutoff"
        ].isin(
            historical_cutoffs
        )
    ].copy()

    competition_features = features[
        features[
            "cutoff"
        ].eq(
            final_cutoff
        )
    ].copy()

    if len(
        historical_features
    ) != 22679:

        raise AssertionError(
            "Historical feature population "
            "is not 22,679."
        )

    if len(
        competition_features
    ) != 5488:

        raise AssertionError(
            "Competition feature population "
            "is not 5,488."
        )

    # ============================================================
    # LOAD PHASE 02 TARGETS
    # ============================================================

    print(
        "Loading transformed Phase 02 labels..."
    )

    labels, label_summary = (
        load_labels(
            historical_cutoff_text
        )
    )

    if len(
        labels
    ) != 22679:

        raise AssertionError(
            "Phase 02 target population "
            "is not 22,679."
        )

    # ============================================================
    # ONE-TO-ONE FEATURE / LABEL RECONCILIATION
    # ============================================================

    print(
        "Reconciling features and labels..."
    )

    historical_ids = set(
        historical_features[
            "snapshot_id"
        ]
    )

    label_ids = set(
        labels[
            "snapshot_id"
        ]
    )

    missing_labels = (
        historical_ids
        - label_ids
    )

    labels_without_features = (
        label_ids
        - historical_ids
    )

    if (
        missing_labels
        or labels_without_features
    ):

        raise AssertionError(
            "Feature/label snapshot mismatch: "
            f"{len(missing_labels)} "
            f"features without labels, "
            f"{len(labels_without_features)} "
            f"labels without features."
        )

    supervised = (
        historical_features[
            [
                "snapshot_id",
                "ID",
                "cutoff",
            ]
            +
            selected_features
        ]
        .merge(
            labels[
                [
                    "snapshot_id",
                    "CLV_fuel",
                    "CLV_nonfuel",
                    "Opportunity",
                ]
            ],
            on="snapshot_id",
            how="left",
            validate="one_to_one",
        )
    )

    if len(
        supervised
    ) != 22679:

        raise AssertionError(
            "Supervised row count changed "
            "during target join."
        )

    if supervised[
        TARGET_COLUMNS
    ].isna().any().any():

        raise AssertionError(
            "Missing target after "
            "feature-label join."
        )

    supervised = supervised.sort_values(
        [
            "cutoff",
            "ID",
        ]
    ).reset_index(
        drop=True
    )

    competition = (
        competition_features[
            [
                "snapshot_id",
                "ID",
                "cutoff",
            ]
            +
            selected_features
        ]
        .sort_values(
            [
                "ID",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ============================================================
    # OPPORTUNITY CLASS MAP
    # ============================================================

    opportunity_classes = sorted(
        supervised[
            "Opportunity"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(
        opportunity_classes
    ) != 17:

        raise AssertionError(
            f"Expected 17 Opportunity "
            f"classes, found "
            f"{len(opportunity_classes)}."
        )

    class_map = pd.DataFrame(
        {
            "class_id":
                range(
                    len(
                        opportunity_classes
                    )
                ),

            "Opportunity":
                opportunity_classes,
        }
    )

    class_map.to_csv(
        OUTPUT_DIR
        / "opportunity_class_map.csv",
        index=False,
    )

    class_lookup = {
        value:
            idx
        for idx, value
        in enumerate(
            opportunity_classes
        )
    }

    supervised[
        "Opportunity_class_id"
    ] = (
        supervised[
            "Opportunity"
        ].map(
            class_lookup
        ).astype(
            "int64"
        )
    )

    # ============================================================
    # PER-CUTOFF JOIN AUDIT
    # ============================================================

    join_audit_rows = []

    for cutoff in historical_cutoffs:

        feature_block = (
            historical_features[
                historical_features[
                    "cutoff"
                ].eq(
                    cutoff
                )
            ]
        )

        label_block = labels[
            labels[
                "cutoff"
            ].eq(
                cutoff
            )
        ]

        supervised_block = (
            supervised[
                supervised[
                    "cutoff"
                ].eq(
                    cutoff
                )
            ]
        )

        feature_set = set(
            feature_block[
                "snapshot_id"
            ]
        )

        label_set = set(
            label_block[
                "snapshot_id"
            ]
        )

        join_audit_rows.append(
            {
                "cutoff":
                    cutoff.date().isoformat(),

                "feature_rows":
                    len(
                        feature_block
                    ),

                "label_rows":
                    len(
                        label_block
                    ),

                "joined_rows":
                    len(
                        supervised_block
                    ),

                "features_without_labels":
                    len(
                        feature_set
                        - label_set
                    ),

                "labels_without_features":
                    len(
                        label_set
                        - feature_set
                    ),

                "opportunity_classes":
                    supervised_block[
                        "Opportunity"
                    ].nunique(),

                "validation_status":
                    "PASS",
            }
        )

    join_audit = pd.DataFrame(
        join_audit_rows
    )

    join_audit.to_csv(
        OUTPUT_DIR
        / "supervised_join_audit.csv",
        index=False,
    )

    # ============================================================
    # TARGET SUMMARY
    # ============================================================

    target_summary_rows = []

    for cutoff in historical_cutoffs:

        block = supervised[
            supervised[
                "cutoff"
            ].eq(
                cutoff
            )
        ]

        target_summary_rows.append(
            {
                "cutoff":
                    cutoff.date().isoformat(),

                "rows":
                    len(block),

                "fuel_mean":
                    float(
                        block[
                            "CLV_fuel"
                        ].mean()
                    ),

                "fuel_median":
                    float(
                        block[
                            "CLV_fuel"
                        ].median()
                    ),

                "fuel_zero_rate":
                    float(
                        block[
                            "CLV_fuel"
                        ].eq(0).mean()
                    ),

                "nonfuel_mean":
                    float(
                        block[
                            "CLV_nonfuel"
                        ].mean()
                    ),

                "nonfuel_median":
                    float(
                        block[
                            "CLV_nonfuel"
                        ].median()
                    ),

                "nonfuel_zero_rate":
                    float(
                        block[
                            "CLV_nonfuel"
                        ].eq(0).mean()
                    ),

                "opportunity_classes":
                    block[
                        "Opportunity"
                    ].nunique(),
            }
        )

    target_summary = pd.DataFrame(
        target_summary_rows
    )

    target_summary.to_csv(
        OUTPUT_DIR
        / "target_summary_by_cutoff.csv",
        index=False,
    )

    # ============================================================
    # EXACT TEMPORAL FOLD MEMBERSHIP
    # ============================================================

    print(
        "Building rolling-origin "
        "validation folds..."
    )

    membership_frames = []
    fold_summary_rows = []

    for fold in temporal_config[
        "folds"
    ]:

        fold_id = fold[
            "fold_id"
        ]

        train_cutoffs = [
            pd.Timestamp(
                value
            )
            for value
            in fold[
                "train_cutoffs"
            ]
        ]

        validation_cutoff = (
            pd.Timestamp(
                fold[
                    "validation_cutoff"
                ]
            )
        )

        train_block = supervised[
            supervised[
                "cutoff"
            ].isin(
                train_cutoffs
            )
        ].copy()

        validation_block = (
            supervised[
                supervised[
                    "cutoff"
                ].eq(
                    validation_cutoff
                )
            ]
            .copy()
        )

        if train_block.empty:
            raise AssertionError(
                f"{fold_id} has no "
                f"training rows."
            )

        if validation_block.empty:
            raise AssertionError(
                f"{fold_id} has no "
                f"validation rows."
            )

        if (
            train_block[
                "cutoff"
            ].max()
            >= validation_cutoff
        ):

            raise AssertionError(
                f"{fold_id} training cutoff "
                f"is not earlier than "
                f"validation."
            )

        # Each target window is exactly
        # three calendar months.
        train_target_ends = (
            train_block[
                "cutoff"
            ]
            +
            pd.DateOffset(
                months=3
            )
        )

        latest_train_target_end = (
            train_target_ends.max()
        )

        if (
            latest_train_target_end
            >
            validation_cutoff
        ):

            raise AssertionError(
                f"{fold_id} training target "
                f"window extends beyond "
                f"validation origin."
            )

        train_membership = (
            train_block[
                [
                    "snapshot_id",
                    "ID",
                    "cutoff",
                ]
            ]
            .copy()
        )

        train_membership[
            "fold_id"
        ] = fold_id

        train_membership[
            "role"
        ] = "train"

        validation_membership = (
            validation_block[
                [
                    "snapshot_id",
                    "ID",
                    "cutoff",
                ]
            ]
            .copy()
        )

        validation_membership[
            "fold_id"
        ] = fold_id

        validation_membership[
            "role"
        ] = "validation"

        membership_frames.extend(
            [
                train_membership,
                validation_membership,
            ]
        )

        fold_summary_rows.append(
            {
                "fold_id":
                    fold_id,

                "train_rows":
                    len(
                        train_block
                    ),

                "validation_rows":
                    len(
                        validation_block
                    ),

                "train_first_cutoff":
                    train_block[
                        "cutoff"
                    ].min()
                    .date()
                    .isoformat(),

                "train_last_cutoff":
                    train_block[
                        "cutoff"
                    ].max()
                    .date()
                    .isoformat(),

                "validation_cutoff":
                    validation_cutoff
                    .date()
                    .isoformat(),

                "latest_train_target_end":
                    latest_train_target_end
                    .date()
                    .isoformat(),

                "target_window_boundary_pass":
                    int(
                        latest_train_target_end
                        <=
                        validation_cutoff
                    ),

                "train_opportunity_classes":
                    train_block[
                        "Opportunity"
                    ].nunique(),

                "validation_opportunity_classes":
                    validation_block[
                        "Opportunity"
                    ].nunique(),

                "validation_fuel_zero_rate":
                    float(
                        validation_block[
                            "CLV_fuel"
                        ].eq(0).mean()
                    ),

                "validation_nonfuel_zero_rate":
                    float(
                        validation_block[
                            "CLV_nonfuel"
                        ].eq(0).mean()
                    ),

                "validation_status":
                    "PASS",
            }
        )

    fold_membership = pd.concat(
        membership_frames,
        ignore_index=True,
    )

    fold_membership = (
        fold_membership[
            [
                "fold_id",
                "role",
                "snapshot_id",
                "ID",
                "cutoff",
            ]
        ]
        .sort_values(
            [
                "fold_id",
                "role",
                "cutoff",
                "ID",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    fold_summary = pd.DataFrame(
        fold_summary_rows
    )

    # Expected fold row counts.
    expected_counts = {
        "F1": (3608, 4223),
        "F2": (7831, 4649),
        "F3": (12480, 4961),
        "F4": (17441, 5238),
    }

    for _, row in fold_summary.iterrows():

        fold_id = row[
            "fold_id"
        ]

        if fold_id not in expected_counts:
            raise AssertionError(
                f"Unexpected fold "
                f"{fold_id}."
            )

        expected_train, expected_valid = (
            expected_counts[
                fold_id
            ]
        )

        if int(
            row[
                "train_rows"
            ]
        ) != expected_train:

            raise AssertionError(
                f"{fold_id} expected "
                f"{expected_train} train "
                f"rows."
            )

        if int(
            row[
                "validation_rows"
            ]
        ) != expected_valid:

            raise AssertionError(
                f"{fold_id} expected "
                f"{expected_valid} "
                f"validation rows."
            )

        if int(
            row[
                "train_opportunity_classes"
            ]
        ) != 17:

            raise AssertionError(
                f"{fold_id} training data "
                f"does not contain all "
                f"17 classes."
            )

        if int(
            row[
                "validation_opportunity_classes"
            ]
        ) != 17:

            raise AssertionError(
                f"{fold_id} validation "
                f"does not contain all "
                f"17 classes."
            )

    fold_membership.to_csv(
        OUTPUT_DIR
        / "temporal_fold_membership.csv",
        index=False,
    )

    fold_summary.to_csv(
        OUTPUT_DIR
        / "temporal_fold_summary.csv",
        index=False,
    )

    # ============================================================
    # SAVE MODELLING DATA
    # ============================================================

    supervised.to_csv(
        OUTPUT_DIR
        / "supervised_historical.csv",
        index=False,
        float_format="%.10g",
    )

    competition.to_csv(
        OUTPUT_DIR
        / "competition_features.csv",
        index=False,
        float_format="%.10g",
    )

    label_summary.to_csv(
        OUTPUT_DIR
        / "phase02_label_load_audit.csv",
        index=False,
    )

    # ============================================================
    # OVERALL AUDIT
    # ============================================================

    overall_audit = pd.DataFrame(
        [
            {
                "canonical_features":
                    len(
                        candidate_features
                    ),

                "excluded_features":
                    len(
                        excluded_features
                    ),

                "selected_features":
                    len(
                        selected_features
                    ),

                "historical_feature_rows":
                    len(
                        historical_features
                    ),

                "label_rows":
                    len(
                        labels
                    ),

                "supervised_rows":
                    len(
                        supervised
                    ),

                "competition_rows":
                    len(
                        competition
                    ),

                "feature_label_missing":
                    len(
                        missing_labels
                    ),

                "label_feature_missing":
                    len(
                        labels_without_features
                    ),

                "opportunity_classes":
                    len(
                        opportunity_classes
                    ),

                "temporal_folds":
                    len(
                        fold_summary
                    ),

                "validation_status":
                    "PASS",
            }
        ]
    )

    overall_audit.to_csv(
        OUTPUT_DIR
        / "phase04a_audit_summary.csv",
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
        "PHASE 04A SUPERVISED DATASET ASSEMBLY COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        overall_audit.to_string(
            index=False
        )
    )

    print()

    print(
        "FEATURE / LABEL JOIN AUDIT"
    )

    print()

    print(
        join_audit.to_string(
            index=False
        )
    )

    print()

    print(
        "TEMPORAL VALIDATION FOLDS"
    )

    print()

    print(
        fold_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "TARGET SUMMARY"
    )

    print()

    print(
        target_summary.to_string(
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
