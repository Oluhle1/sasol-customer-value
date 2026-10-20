"""
Build experimental S1 submission.

Control:
    S0 Phase 05 submission

Change:
    replace Opportunity only with final Phase 06C hierarchical predictions

Fuel and non-fuel are preserved byte-for-value from S0 after ID join.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

S0_PATH = (
    ROOT
    / "submissions"
    / "sasol_phase05_frozen_lightgbm_submission.csv"
)

HIERARCHICAL_PATH = (
    ROOT
    / "outputs"
    / "phase06c_final"
    / "hierarchical_opportunity_predictions.csv"
)

OUTPUT_PATH = (
    ROOT
    / "submissions"
    / "sasol_S1_phase06c_hierarchical_opportunity.csv"
)


def main():

    s0 = pd.read_csv(
        S0_PATH,
        dtype={"ID": str},
        keep_default_na=False,
    )

    hierarchy = pd.read_csv(
        HIERARCHICAL_PATH,
        dtype={"ID": str},
        keep_default_na=False,
    )

    if len(s0) != 5488:
        raise AssertionError(
            "S0 row count mismatch."
        )

    if len(hierarchy) != 5488:
        raise AssertionError(
            "06C prediction row count mismatch."
        )

    if s0[
        "ID"
    ].duplicated().any():

        raise AssertionError(
            "S0 contains duplicate IDs."
        )

    if hierarchy[
        "ID"
    ].duplicated().any():

        raise AssertionError(
            "06C predictions contain duplicate IDs."
        )

    merged = (
        s0
        .merge(
            hierarchy[
                [
                    "ID",
                    "Opportunity",
                ]
            ].rename(
                columns={
                    "Opportunity":
                        "Opportunity_06C"
                }
            ),
            on="ID",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if merged[
        "Opportunity_06C"
    ].eq(
        ""
    ).any():

        raise AssertionError(
            "Missing 06C Opportunity predictions."
        )

    if merged[
        "Opportunity_06C"
    ].isna().any():

        raise AssertionError(
            "Missing 06C Opportunity predictions."
        )

    changed = int(
        (
            merged[
                "Opportunity"
            ]
            !=
            merged[
                "Opportunity_06C"
            ]
        ).sum()
    )

    submission = merged[
        [
            "ID",
            "CLV_fuel",
            "CLV_nonfuel",
        ]
    ].copy()

    submission[
        "Opportunity"
    ] = merged[
        "Opportunity_06C"
    ]

    if submission[
        "ID"
    ].tolist() != s0[
        "ID"
    ].tolist():

        raise AssertionError(
            "ID ordering changed."
        )

    # Confirm regressions are exactly unchanged.
    if not submission[
        "CLV_fuel"
    ].equals(
        s0[
            "CLV_fuel"
        ]
    ):

        raise AssertionError(
            "Fuel predictions changed."
        )

    if not submission[
        "CLV_nonfuel"
    ].equals(
        s0[
            "CLV_nonfuel"
        ]
    ):

        raise AssertionError(
            "Non-fuel predictions changed."
        )

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
        float_format="%.10g",
    )

    print()
    print(
        "=" * 72
    )

    print(
        "PHASE 06C EXPERIMENTAL SUBMISSION CREATED"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Rows: {len(submission)}"
    )

    print(
        f"Unique IDs: "
        f"{submission['ID'].nunique()}"
    )

    print(
        f"Opportunity predictions changed "
        f"vs S0: {changed}"
    )

    print(
        f"Predicted classes: "
        f"{submission['Opportunity'].nunique()}"
    )

    print()

    print(
        "Submission:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
