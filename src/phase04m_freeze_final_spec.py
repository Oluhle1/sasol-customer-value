"""
Phase 04M: Final model specification freeze.

No model is trained.

The script verifies that the machine-readable final model specification
matches the accepted Phase 04 experiment outputs.

It also validates:
- final feature count;
- historical and competition populations;
- accepted F4 metrics;
- accepted best iterations;
- target/class configuration.

The purpose is to prevent silent drift between experimentation and final
competition training.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PHASE04_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
)

SPEC_PATH = (
    ROOT
    / "configs"
    / "final_model_spec.json"
)

BASELINE_CONFIG_PATH = (
    ROOT
    / "configs"
    / "lightgbm_baseline.json"
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

OUTPUT_DIR = (
    PHASE04_DIR
    / "final_spec"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def assert_close(
    observed,
    expected,
    name,
    tolerance=1e-6,
):

    if not np.isclose(
        float(observed),
        float(expected),
        atol=tolerance,
        rtol=0,
    ):

        raise AssertionError(
            f"{name}: expected "
            f"{expected}, observed "
            f"{observed}"
        )


def main():

    print(
        "Loading Phase 04M final specification..."
    )

    spec = json.loads(
        SPEC_PATH.read_text()
    )

    baseline_config = json.loads(
        BASELINE_CONFIG_PATH.read_text()
    )

    supervised = pd.read_csv(
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
    )

    fuel_results = pd.read_csv(
        FUEL_ACCEPTED_PATH
    )

    nonfuel_results = pd.read_csv(
        NONFUEL_ACCEPTED_PATH
    )

    opportunity_results = pd.read_csv(
        OPPORTUNITY_ACCEPTED_PATH
    )

    audit_rows = []

    # ============================================================
    # FROZEN STATUS
    # ============================================================

    if spec[
        "status"
    ] != "FROZEN":

        raise AssertionError(
            "Final specification is not frozen."
        )

    # ============================================================
    # POPULATION
    # ============================================================

    expected_historical = int(
        spec[
            "training_population"
        ][
            "historical_rows"
        ]
    )

    expected_competition = int(
        spec[
            "training_population"
        ][
            "competition_rows"
        ]
    )

    if len(
        supervised
    ) != expected_historical:

        raise AssertionError(
            "Historical population mismatch."
        )

    if len(
        competition
    ) != expected_competition:

        raise AssertionError(
            "Competition population mismatch."
        )

    audit_rows.extend(
        [
            {
                "check":
                    "historical_rows",
                "expected":
                    expected_historical,
                "observed":
                    len(
                        supervised
                    ),
                "status":
                    "PASS",
            },
            {
                "check":
                    "competition_rows",
                "expected":
                    expected_competition,
                "observed":
                    len(
                        competition
                    ),
                "status":
                    "PASS",
            },
        ]
    )

    # ============================================================
    # FEATURES
    # ============================================================

    selected_features = (
        manifest.loc[
            manifest[
                "selected_for_baseline"
            ].eq(1),
            "feature",
        ]
        .tolist()
    )

    expected_features = int(
        spec[
            "feature_policy"
        ][
            "selected_features"
        ]
    )

    if len(
        selected_features
    ) != expected_features:

        raise AssertionError(
            "Selected feature count mismatch."
        )

    missing_historical = [
        feature
        for feature in selected_features
        if feature
        not in supervised.columns
    ]

    missing_competition = [
        feature
        for feature in selected_features
        if feature
        not in competition.columns
    ]

    if missing_historical:

        raise AssertionError(
            f"Historical data missing features: "
            f"{missing_historical}"
        )

    if missing_competition:

        raise AssertionError(
            f"Competition data missing features: "
            f"{missing_competition}"
        )

    audit_rows.append(
        {
            "check":
                "selected_features",
            "expected":
                expected_features,
            "observed":
                len(
                    selected_features
                ),
            "status":
                "PASS",
        }
    )

    # ============================================================
    # CLASS MAP
    # ============================================================

    expected_classes = int(
        spec[
            "opportunity"
        ][
            "num_class"
        ]
    )

    if len(
        class_map
    ) != expected_classes:

        raise AssertionError(
            "Opportunity class count mismatch."
        )

    if supervised[
        "Opportunity_class_id"
    ].nunique() != expected_classes:

        raise AssertionError(
            "Historical Opportunity class "
            "population mismatch."
        )

    audit_rows.append(
        {
            "check":
                "opportunity_classes",
            "expected":
                expected_classes,
            "observed":
                len(
                    class_map
                ),
            "status":
                "PASS",
        }
    )

    # ============================================================
    # ORIGINAL CONFIG SAFETY
    # ============================================================

    if int(
        baseline_config[
            "common_parameters"
        ][
            "num_leaves"
        ]
    ) != 31:

        raise AssertionError(
            "Original baseline num_leaves "
            "has changed."
        )

    if int(
        baseline_config[
            "common_parameters"
        ][
            "min_data_in_leaf"
        ]
    ) != 50:

        raise AssertionError(
            "Original baseline min_data_in_leaf "
            "has changed."
        )

    # ============================================================
    # FUEL ACCEPTED RESULT
    # ============================================================

    fuel_f4 = (
        fuel_results[
            (
                fuel_results[
                    "task"
                ].eq(
                    "fuel"
                )
                &
                fuel_results[
                    "fold_id"
                ].eq(
                    "F4"
                )
            )
        ]
        .iloc[0]
    )

    assert_close(
        fuel_f4[
            "experiment_validation_rmse"
        ],
        spec[
            "fuel"
        ][
            "accepted_f4_metric"
        ],
        "Fuel F4 metric",
    )

    if int(
        fuel_f4[
            "best_iteration"
        ]
    ) != int(
        spec[
            "fuel"
        ][
            "num_boost_round"
        ]
    ):

        raise AssertionError(
            "Fuel final boosting-round "
            "selection does not match F4."
        )

    if int(
        spec[
            "fuel"
        ][
            "num_leaves"
        ]
    ) != 15:

        raise AssertionError(
            "Fuel num_leaves mismatch."
        )

    if int(
        spec[
            "fuel"
        ][
            "min_data_in_leaf"
        ]
    ) != 100:

        raise AssertionError(
            "Fuel min_data_in_leaf mismatch."
        )

    audit_rows.append(
        {
            "check":
                "fuel_final_spec",
            "expected":
                "15 leaves / 100 minleaf / 168 rounds",
            "observed":
                (
                    f"{spec['fuel']['num_leaves']} leaves / "
                    f"{spec['fuel']['min_data_in_leaf']} minleaf / "
                    f"{spec['fuel']['num_boost_round']} rounds"
                ),
            "status":
                "PASS",
        }
    )

    # ============================================================
    # NON-FUEL ACCEPTED RESULT
    # ============================================================

    nonfuel_f4 = (
        nonfuel_results[
            nonfuel_results[
                "fold_id"
            ].eq(
                "F4"
            )
        ]
        .iloc[0]
    )

    assert_close(
        nonfuel_f4[
            "capacity_validation_rmse"
        ],
        spec[
            "nonfuel"
        ][
            "accepted_f4_metric"
        ],
        "Non-fuel F4 metric",
    )

    if int(
        nonfuel_f4[
            "best_iteration"
        ]
    ) != int(
        spec[
            "nonfuel"
        ][
            "num_boost_round"
        ]
    ):

        raise AssertionError(
            "Non-fuel final boosting-round "
            "selection does not match F4."
        )

    if int(
        spec[
            "nonfuel"
        ][
            "num_leaves"
        ]
    ) != 15:

        raise AssertionError(
            "Non-fuel num_leaves mismatch."
        )

    if int(
        spec[
            "nonfuel"
        ][
            "min_data_in_leaf"
        ]
    ) != 50:

        raise AssertionError(
            "Non-fuel min_data_in_leaf mismatch."
        )

    audit_rows.append(
        {
            "check":
                "nonfuel_final_spec",
            "expected":
                "15 leaves / 50 minleaf / 145 rounds",
            "observed":
                (
                    f"{spec['nonfuel']['num_leaves']} leaves / "
                    f"{spec['nonfuel']['min_data_in_leaf']} minleaf / "
                    f"{spec['nonfuel']['num_boost_round']} rounds"
                ),
            "status":
                "PASS",
        }
    )

    # ============================================================
    # OPPORTUNITY ACCEPTED RESULT
    # ============================================================

    opportunity_f4 = (
        opportunity_results[
            opportunity_results[
                "fold_id"
            ].eq(
                "F4"
            )
        ]
        .iloc[0]
    )

    assert_close(
        opportunity_f4[
            "weighted_model_f1"
        ],
        spec[
            "opportunity"
        ][
            "accepted_f4_metric"
        ],
        "Opportunity F4 metric",
    )

    if int(
        opportunity_f4[
            "best_iteration"
        ]
    ) != int(
        spec[
            "opportunity"
        ][
            "num_boost_round"
        ]
    ):

        raise AssertionError(
            "Opportunity final boosting-round "
            "selection does not match F4."
        )

    if not np.isclose(
        float(
            spec[
                "opportunity"
            ][
                "class_weight_exponent"
            ]
        ),
        0.5,
    ):

        raise AssertionError(
            "Opportunity weighting exponent "
            "must be 0.50."
        )

    audit_rows.append(
        {
            "check":
                "opportunity_final_spec",
            "expected":
                "31 leaves / 50 minleaf / alpha 0.50 / 232 rounds",
            "observed":
                (
                    f"{spec['opportunity']['num_leaves']} leaves / "
                    f"{spec['opportunity']['min_data_in_leaf']} minleaf / "
                    f"alpha {spec['opportunity']['class_weight_exponent']} / "
                    f"{spec['opportunity']['num_boost_round']} rounds"
                ),
            "status":
                "PASS",
        }
    )

    # ============================================================
    # TARGET COMPLETENESS
    # ============================================================

    required_targets = [
        "CLV_fuel",
        "CLV_nonfuel",
        "Opportunity",
        "Opportunity_class_id",
    ]

    if supervised[
        required_targets
    ].isna().any().any():

        raise AssertionError(
            "Historical training targets "
            "contain missing values."
        )

    # ============================================================
    # FINAL AUDIT
    # ============================================================

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        OUTPUT_DIR
        / "final_spec_audit.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "status":
                    "FROZEN",

                "historical_rows":
                    len(
                        supervised
                    ),

                "competition_rows":
                    len(
                        competition
                    ),

                "selected_features":
                    len(
                        selected_features
                    ),

                "opportunity_classes":
                    len(
                        class_map
                    ),

                "fuel_num_boost_round":
                    int(
                        spec[
                            "fuel"
                        ][
                            "num_boost_round"
                        ]
                    ),

                "nonfuel_num_boost_round":
                    int(
                        spec[
                            "nonfuel"
                        ][
                            "num_boost_round"
                        ]
                    ),

                "opportunity_num_boost_round":
                    int(
                        spec[
                            "opportunity"
                        ][
                            "num_boost_round"
                        ]
                    ),

                "validation_status":
                    "PASS",
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "final_spec_summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 04M FINAL MODEL SPECIFICATION FREEZE COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "AUDIT"
    )

    print()

    print(
        audit.to_string(
            index=False
        )
    )

    print()

    print(
        "FINAL SPECIFICATION"
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
