"""
Phase 03A: Build the authoritative customer x forecasting-origin index.

This script creates the row architecture used by all later feature
engineering.

It does NOT create predictive features and does NOT train models.

Historical eligibility is recalculated directly from transaction
history using Sasol's official eligibility implementation and checked
against the Phase 02 reconstructed labels.

Final eligibility is independently checked against test.csv.
"""

from pathlib import Path
import json

import pandas as pd

from reference.label_rules import (
    load_transactions,
    eligible_ids,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"

LABEL_CONFIG_PATH = (
    ROOT / "configs" / "label_config.json"
)

FEATURE_CONFIG_PATH = (
    ROOT / "configs" / "feature_config.json"
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
    / "phase03_snapshots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def make_snapshot_frame(
    ids,
    cutoff,
    role,
    is_labeled,
):
    """
    Construct snapshot-index rows.

    No predictive information is created here.
    """

    cutoff_ts = pd.Timestamp(cutoff)

    frame = pd.DataFrame(
        {
            "ID": pd.Series(
                ids,
                dtype="string",
            ),
        }
    )

    frame["cutoff"] = cutoff_ts

    frame[
        "history_end_exclusive"
    ] = cutoff_ts

    frame["outcome_start"] = cutoff_ts

    frame[
        "outcome_end_exclusive"
    ] = (
        cutoff_ts
        + pd.DateOffset(months=3)
    )

    frame["snapshot_role"] = role
    frame["is_labeled"] = is_labeled

    frame["snapshot_id"] = (
        frame["ID"].astype(str)
        + "__"
        + cutoff_ts.strftime(
            "%Y-%m-%d"
        )
    )

    return frame


def main():

    print(
        "Loading configuration..."
    )

    label_config = json.loads(
        LABEL_CONFIG_PATH.read_text()
    )

    # Load to validate that the feature configuration exists
    # and is valid JSON.
    json.loads(
        FEATURE_CONFIG_PATH.read_text()
    )

    minimum = label_config[
        "minimum_historical_transactions"
    ]

    historical_cutoffs = label_config[
        "calibration_cutoffs"
    ]

    final_cutoff = label_config[
        "public_history_end_exclusive"
    ]

    print(
        "Loading transaction history using "
        "Sasol reference loader..."
    )

    transactions = load_transactions(
        TRAIN_PATH
    )

    max_transaction_time = (
        transactions["_time"].max()
    )

    if not (
        max_transaction_time
        < pd.Timestamp(final_cutoff)
    ):
        raise AssertionError(
            "Training history contains a transaction "
            "at or after the final forecasting cutoff."
        )

    historical_frames = []
    audit_rows = []

    # ============================================================
    # HISTORICAL SNAPSHOTS
    # ============================================================

    for cutoff in historical_cutoffs:

        print(
            f"\nBuilding historical snapshot: "
            f"{cutoff}"
        )

        ids = eligible_ids(
            transactions,
            cutoff,
            minimum,
        )

        ids = pd.Index(
            ids.astype(str)
        )

        label_path = (
            PHASE02_LABEL_DIR
            / f"labels_raw_{cutoff}.csv"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                label_path
            )

        label_ids = pd.read_csv(
            label_path,
            usecols=["ID"],
            dtype={"ID": str},
        )["ID"]

        if label_ids.duplicated().any():
            raise AssertionError(
                f"Duplicate IDs found in "
                f"Phase 02 labels at {cutoff}"
            )

        eligible_set = set(ids)
        label_set = set(label_ids)

        missing_from_labels = (
            eligible_set
            - label_set
        )

        extra_in_labels = (
            label_set
            - eligible_set
        )

        if (
            missing_from_labels
            or extra_in_labels
        ):
            raise AssertionError(
                f"Eligibility mismatch at "
                f"{cutoff}: "
                f"{len(missing_from_labels)} "
                f"missing from labels, "
                f"{len(extra_in_labels)} "
                f"extra in labels."
            )

        frame = make_snapshot_frame(
            sorted(ids),
            cutoff,
            role="historical",
            is_labeled=True,
        )

        historical_frames.append(
            frame
        )

        audit_rows.append(
            {
                "cutoff": cutoff,
                "snapshot_role":
                    "historical",
                "eligible_customers":
                    len(ids),
                "label_customers":
                    len(label_ids),
                "missing_from_labels":
                    len(
                        missing_from_labels
                    ),
                "extra_in_labels":
                    len(
                        extra_in_labels
                    ),
                "validation_status":
                    "PASS",
            }
        )

        print(
            f"Eligible customers: "
            f"{len(ids):,}"
        )

    historical = pd.concat(
        historical_frames,
        ignore_index=True,
    )

    # ============================================================
    # FINAL COMPETITION SNAPSHOT
    # ============================================================

    print(
        f"\nBuilding final competition "
        f"snapshot: {final_cutoff}"
    )

    final_ids = eligible_ids(
        transactions,
        final_cutoff,
        minimum,
    )

    final_ids = pd.Index(
        final_ids.astype(str)
    )

    test = pd.read_csv(
        TEST_PATH,
        dtype={"ID": str},
        keep_default_na=False,
    )

    if "ID" not in test.columns:
        raise AssertionError(
            "test.csv does not contain ID."
        )

    if test["ID"].duplicated().any():
        raise AssertionError(
            "test.csv contains duplicate IDs."
        )

    test_ids = pd.Index(
        test["ID"].astype(str)
    )

    eligible_set = set(final_ids)
    test_set = set(test_ids)

    missing_from_test = (
        eligible_set
        - test_set
    )

    extra_in_test = (
        test_set
        - eligible_set
    )

    if (
        missing_from_test
        or extra_in_test
    ):
        raise AssertionError(
            "Final eligibility does not match "
            "test.csv."
        )

    final = make_snapshot_frame(
        sorted(final_ids),
        final_cutoff,
        role="competition",
        is_labeled=False,
    )

    audit_rows.append(
        {
            "cutoff": final_cutoff,
            "snapshot_role":
                "competition",
            "eligible_customers":
                len(final_ids),
            "label_customers":
                len(test_ids),
            "missing_from_labels":
                len(
                    missing_from_test
                ),
            "extra_in_labels":
                len(
                    extra_in_test
                ),
            "validation_status":
                "PASS",
        }
    )

    # ============================================================
    # COMBINED INDEX
    # ============================================================

    combined = pd.concat(
        [
            historical,
            final,
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "cutoff",
            "ID",
        ]
    ).reset_index(
        drop=True
    )

    if combined[
        "snapshot_id"
    ].duplicated().any():
        raise AssertionError(
            "Duplicate customer-cutoff "
            "snapshot IDs detected."
        )

    if historical[
        "snapshot_id"
    ].duplicated().any():
        raise AssertionError(
            "Historical snapshot IDs "
            "are not unique."
        )

    if final[
        "snapshot_id"
    ].duplicated().any():
        raise AssertionError(
            "Competition snapshot IDs "
            "are not unique."
        )

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================

    historical.to_csv(
        OUTPUT_DIR
        / "historical_snapshot_index.csv",
        index=False,
    )

    final.to_csv(
        OUTPUT_DIR
        / "competition_snapshot_index.csv",
        index=False,
    )

    combined.to_csv(
        OUTPUT_DIR
        / "all_snapshot_index.csv",
        index=False,
    )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        OUTPUT_DIR
        / "snapshot_index_audit.csv",
        index=False,
    )

    # ============================================================
    # RESEARCH CHECKPOINT
    # ============================================================

    print()
    print(
        "=" * 70
    )

    print(
        "PHASE 03A SNAPSHOT ARCHITECTURE COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        audit.to_string(
            index=False
        )
    )

    print()

    print(
        f"Historical snapshots: "
        f"{len(historical):,}"
    )

    print(
        f"Competition snapshots: "
        f"{len(final):,}"
    )

    print(
        f"Total snapshots: "
        f"{len(combined):,}"
    )

    print(
        f"Latest transaction: "
        f"{max_transaction_time}"
    )

    print(
        f"Final cutoff: "
        f"{final_cutoff}"
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
