"""
Phase 05B: Final submission integrity audit and construction.

Uses the official SampleSubmission.csv as the structural authority.

No modelling occurs here.

The script:
- preserves official ID order;
- verifies one-to-one ID correspondence;
- validates numeric predictions;
- validates Opportunity labels;
- prevents diagnostic columns entering the submission;
- creates the final competition CSV only after all checks pass.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PATH = (
    ROOT
    / "data"
    / "raw"
    / "SampleSubmission.csv"
)

PREDICTION_PATH = (
    ROOT
    / "outputs"
    / "phase05"
    / "competition_predictions.csv"
)

CLASS_MAP_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
    / "opportunity_class_map.csv"
)

OUTPUT_DIR = (
    ROOT
    / "submissions"
)

AUDIT_DIR = (
    ROOT
    / "outputs"
    / "phase05"
    / "submission_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_SUBMISSION_PATH = (
    OUTPUT_DIR
    / "sasol_phase05_frozen_lightgbm_submission.csv"
)


def pass_row(
    check,
    observed,
    expected,
):
    return {
        "check": check,
        "observed": observed,
        "expected": expected,
        "status": "PASS",
    }


def main():

    print(
        "Loading Phase 05B submission inputs..."
    )

    sample = pd.read_csv(
        SAMPLE_PATH,
        dtype={"ID": str},
        keep_default_na=False,
    )

    predictions = pd.read_csv(
        PREDICTION_PATH,
        dtype={"ID": str},
        keep_default_na=False,
    )

    class_map = pd.read_csv(
        CLASS_MAP_PATH,
        keep_default_na=False,
    )

    audit_rows = []

    # ============================================================
    # OFFICIAL SUBMISSION STRUCTURE
    # ============================================================

    required_columns = [
        "ID",
        "CLV_fuel",
        "CLV_nonfuel",
        "Opportunity",
    ]

    if sample.columns.tolist() != required_columns:

        raise AssertionError(
            "Official SampleSubmission columns differ "
            f"from expected structure.\n"
            f"Observed: {sample.columns.tolist()}"
        )

    audit_rows.append(
        pass_row(
            "official_column_order",
            str(sample.columns.tolist()),
            str(required_columns),
        )
    )

    expected_rows = len(
        sample
    )

    if expected_rows != 5488:

        raise AssertionError(
            "Expected official SampleSubmission "
            f"to contain 5488 rows, observed {expected_rows}."
        )

    audit_rows.append(
        pass_row(
            "official_row_count",
            expected_rows,
            5488,
        )
    )

    if sample[
        "ID"
    ].duplicated().any():

        raise AssertionError(
            "Official SampleSubmission contains duplicate IDs."
        )

    # ============================================================
    # PREDICTION POPULATION
    # ============================================================

    if len(
        predictions
    ) != expected_rows:

        raise AssertionError(
            "Prediction row count does not match "
            "SampleSubmission."
        )

    if predictions[
        "ID"
    ].duplicated().any():

        raise AssertionError(
            "Prediction file contains duplicate IDs."
        )

    sample_ids = set(
        sample[
            "ID"
        ]
    )

    prediction_ids = set(
        predictions[
            "ID"
        ]
    )

    missing_predictions = sorted(
        sample_ids
        -
        prediction_ids
    )

    extra_predictions = sorted(
        prediction_ids
        -
        sample_ids
    )

    if missing_predictions:

        raise AssertionError(
            "Missing predictions for official IDs: "
            f"{missing_predictions[:10]}"
        )

    if extra_predictions:

        raise AssertionError(
            "Predictions contain IDs not in "
            "SampleSubmission: "
            f"{extra_predictions[:10]}"
        )

    audit_rows.append(
        pass_row(
            "id_set_match",
            len(
                sample_ids
                & prediction_ids
            ),
            expected_rows,
        )
    )

    # ============================================================
    # BUILD IN OFFICIAL ORDER
    # ============================================================

    required_prediction_columns = {
        "ID",
        "CLV_fuel",
        "CLV_nonfuel",
        "Opportunity",
    }

    missing_prediction_columns = (
        required_prediction_columns
        -
        set(
            predictions.columns
        )
    )

    if missing_prediction_columns:

        raise AssertionError(
            "Prediction file missing required columns: "
            f"{missing_prediction_columns}"
        )

    submission = (
        sample[
            ["ID"]
        ]
        .merge(
            predictions[
                [
                    "ID",
                    "CLV_fuel",
                    "CLV_nonfuel",
                    "Opportunity",
                ]
            ],
            on="ID",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if submission[
        "ID"
    ].tolist() != sample[
        "ID"
    ].tolist():

        raise AssertionError(
            "Submission ID ordering differs from "
            "official SampleSubmission."
        )

    audit_rows.append(
        pass_row(
            "official_id_order_preserved",
            True,
            True,
        )
    )

    # ============================================================
    # MISSING VALUES
    # ============================================================

    missing_by_column = (
        submission[
            required_columns
        ]
        .isna()
        .sum()
    )

    if int(
        missing_by_column.sum()
    ) != 0:

        raise AssertionError(
            "Submission contains missing values:\n"
            f"{missing_by_column}"
        )

    if submission[
        "Opportunity"
    ].eq(
        ""
    ).any():

        raise AssertionError(
            "Submission contains blank Opportunity labels."
        )

    audit_rows.append(
        pass_row(
            "missing_values",
            0,
            0,
        )
    )

    # ============================================================
    # NUMERIC TARGET VALIDATION
    # ============================================================

    for column in [
        "CLV_fuel",
        "CLV_nonfuel",
    ]:

        numeric = pd.to_numeric(
            submission[
                column
            ],
            errors="coerce",
        )

        if numeric.isna().any():

            raise AssertionError(
                f"{column} contains non-numeric values."
            )

        values = numeric.to_numpy(
            dtype="float64"
        )

        if not np.isfinite(
            values
        ).all():

            raise AssertionError(
                f"{column} contains NaN or infinity."
            )

        if (
            values
            < 0
        ).any():

            raise AssertionError(
                f"{column} contains negative predictions."
            )

        submission[
            column
        ] = values

        audit_rows.append(
            pass_row(
                f"{column}_finite_nonnegative",
                True,
                True,
            )
        )

    # ============================================================
    # OPPORTUNITY LABEL VALIDATION
    # ============================================================

    valid_labels = set(
        class_map[
            "Opportunity"
        ].astype(
            str
        )
    )

    if len(
        valid_labels
    ) != 17:

        raise AssertionError(
            "Official Opportunity class map "
            "does not contain 17 unique labels."
        )

    predicted_labels = set(
        submission[
            "Opportunity"
        ].astype(
            str
        )
    )

    invalid_labels = (
        predicted_labels
        -
        valid_labels
    )

    if invalid_labels:

        raise AssertionError(
            "Submission contains invalid Opportunity labels: "
            f"{sorted(invalid_labels)}"
        )

    audit_rows.append(
        pass_row(
            "valid_opportunity_labels",
            len(
                predicted_labels
            ),
            "subset of 17 official classes",
        )
    )

    # ============================================================
    # FINAL COLUMN SAFETY
    # ============================================================

    submission = submission[
        required_columns
    ]

    if submission.columns.tolist() != required_columns:

        raise AssertionError(
            "Final submission column ordering failed."
        )

    if len(
        submission
    ) != 5488:

        raise AssertionError(
            "Final submission does not contain "
            "exactly 5488 rows."
        )

    if submission[
        "ID"
    ].nunique() != 5488:

        raise AssertionError(
            "Final submission does not contain "
            "5488 unique IDs."
        )

    # ============================================================
    # DISTRIBUTION AUDIT
    # ============================================================

    opportunity_distribution = (
        submission[
            "Opportunity"
        ]
        .value_counts()
        .rename_axis(
            "Opportunity"
        )
        .reset_index(
            name="predicted_count"
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
            submission
        )
    )

    regression_summary = pd.DataFrame(
        [
            {
                "rows":
                    len(
                        submission
                    ),

                "unique_ids":
                    submission[
                        "ID"
                    ].nunique(),

                "fuel_min":
                    float(
                        submission[
                            "CLV_fuel"
                        ].min()
                    ),

                "fuel_mean":
                    float(
                        submission[
                            "CLV_fuel"
                        ].mean()
                    ),

                "fuel_median":
                    float(
                        submission[
                            "CLV_fuel"
                        ].median()
                    ),

                "fuel_max":
                    float(
                        submission[
                            "CLV_fuel"
                        ].max()
                    ),

                "nonfuel_min":
                    float(
                        submission[
                            "CLV_nonfuel"
                        ].min()
                    ),

                "nonfuel_mean":
                    float(
                        submission[
                            "CLV_nonfuel"
                        ].mean()
                    ),

                "nonfuel_median":
                    float(
                        submission[
                            "CLV_nonfuel"
                        ].median()
                    ),

                "nonfuel_max":
                    float(
                        submission[
                            "CLV_nonfuel"
                        ].max()
                    ),

                "opportunity_classes_predicted":
                    submission[
                        "Opportunity"
                    ].nunique(),

                "validation_status":
                    "PASS",
            }
        ]
    )

    # ============================================================
    # WRITE ONLY AFTER ALL CHECKS PASS
    # ============================================================

    submission.to_csv(
        FINAL_SUBMISSION_PATH,
        index=False,
        float_format="%.10g",
    )

    # Re-read written file to test serialized output.
    written = pd.read_csv(
        FINAL_SUBMISSION_PATH,
        dtype={"ID": str},
        keep_default_na=False,
    )

    if written.columns.tolist() != required_columns:

        raise AssertionError(
            "Serialized submission columns changed "
            "after writing."
        )

    if len(
        written
    ) != 5488:

        raise AssertionError(
            "Serialized submission row count changed."
        )

    if written[
        "ID"
    ].tolist() != sample[
        "ID"
    ].tolist():

        raise AssertionError(
            "Serialized submission ID order changed."
        )

    audit_rows.append(
        pass_row(
            "serialized_submission_recheck",
            "PASS",
            "PASS",
        )
    )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        AUDIT_DIR
        / "submission_integrity_audit.csv",
        index=False,
    )

    regression_summary.to_csv(
        AUDIT_DIR
        / "submission_summary.csv",
        index=False,
    )

    opportunity_distribution.to_csv(
        AUDIT_DIR
        / "submission_opportunity_distribution.csv",
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
        "PHASE 05B SUBMISSION AUDIT COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "INTEGRITY AUDIT"
    )

    print()

    print(
        audit.to_string(
            index=False
        )
    )

    print()

    print(
        "SUBMISSION SUMMARY"
    )

    print()

    print(
        regression_summary.to_string(
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
        "FINAL SUBMISSION:"
    )

    print(
        FINAL_SUBMISSION_PATH
    )


if __name__ == "__main__":
    main()
