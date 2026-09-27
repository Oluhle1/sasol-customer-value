"""
Phase 02: Reconstruct Sasol historical label snapshots.

This script DOES NOT redefine the competition labels.

It imports and executes Sasol's official reference implementation
from src/reference/label_rules.py at every official calibration cutoff.

Outputs are written to data/processed/phase02_labels/ and are not
committed to Git because they are derived from competition data.
"""

from pathlib import Path
import json

import pandas as pd

from reference.label_rules import load_transactions, raw_labels, transform


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
CONFIG_PATH = ROOT / "configs" / "label_config.json"

OUTPUT_DIR = ROOT / "data" / "processed" / "phase02_labels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading official Sasol configuration...")
    config = json.loads(CONFIG_PATH.read_text())

    print("Loading transaction history using Sasol reference loader...")
    transactions = load_transactions(TRAIN_PATH)

    summary_rows = []
    opportunity_rows = []

    for cutoff in config["calibration_cutoffs"]:

        print(f"\nProcessing cutoff: {cutoff}")

        # Official raw Sasol outcomes
        raw = raw_labels(
            transactions,
            cutoff,
            config
        )

        # Official competition-transformed targets
        transformed = transform(
            raw,
            config
        )

        # ---------------------------------------------------------
        # Integrity checks
        # ---------------------------------------------------------

        assert len(raw) == len(transformed)
        assert raw["ID"].is_unique
        assert transformed["ID"].is_unique

        assert not raw[
            ["fuel_litres", "nonfuel_rand"]
        ].isna().any().any()

        assert not transformed[
            ["CLV_fuel", "CLV_nonfuel"]
        ].isna().any().any()

        assert (raw["fuel_litres"] >= 0).all()
        assert (raw["nonfuel_rand"] >= 0).all()

        assert set(raw["Opportunity"]).issubset(
            set(config["opportunity_labels"])
        )

        # ---------------------------------------------------------
        # Save snapshot datasets
        # ---------------------------------------------------------

        raw_path = (
            OUTPUT_DIR /
            f"labels_raw_{cutoff}.csv"
        )

        transformed_path = (
            OUTPUT_DIR /
            f"labels_transformed_{cutoff}.csv"
        )

        raw.to_csv(
            raw_path,
            index=False
        )

        transformed.to_csv(
            transformed_path,
            index=False,
            float_format="%.17g"
        )

        # ---------------------------------------------------------
        # Basic target statistics
        # ---------------------------------------------------------

        opportunity = raw["Opportunity"]

        growth_mask = opportunity.str.startswith(
            "Existing-category growth:"
        )

        adoption_mask = opportunity.str.startswith(
            "New category adoption:"
        )

        inactivity_mask = opportunity.eq("Inactivity")
        stable_mask = opportunity.eq("Stable")

        summary_rows.append(
            {
                "cutoff": cutoff,
                "eligible_customers": len(raw),

                "fuel_zero_count":
                    int(raw["fuel_litres"].eq(0).sum()),

                "fuel_zero_rate":
                    raw["fuel_litres"].eq(0).mean(),

                "fuel_mean_litres":
                    raw["fuel_litres"].mean(),

                "fuel_median_litres":
                    raw["fuel_litres"].median(),

                "fuel_p95_litres":
                    raw["fuel_litres"].quantile(0.95),

                "nonfuel_zero_count":
                    int(raw["nonfuel_rand"].eq(0).sum()),

                "nonfuel_zero_rate":
                    raw["nonfuel_rand"].eq(0).mean(),

                "nonfuel_mean_rand":
                    raw["nonfuel_rand"].mean(),

                "nonfuel_median_rand":
                    raw["nonfuel_rand"].median(),

                "nonfuel_p95_rand":
                    raw["nonfuel_rand"].quantile(0.95),

                "opportunity_classes_present":
                    opportunity.nunique(),

                "growth_count":
                    int(growth_mask.sum()),

                "growth_rate":
                    growth_mask.mean(),

                "adoption_count":
                    int(adoption_mask.sum()),

                "adoption_rate":
                    adoption_mask.mean(),

                "inactivity_count":
                    int(inactivity_mask.sum()),

                "inactivity_rate":
                    inactivity_mask.mean(),

                "stable_count":
                    int(stable_mask.sum()),

                "stable_rate":
                    stable_mask.mean(),
            }
        )

        counts = (
            opportunity
            .value_counts()
            .rename_axis("Opportunity")
            .reset_index(name="count")
        )

        counts["cutoff"] = cutoff
        counts["share"] = counts["count"] / len(raw)

        opportunity_rows.append(counts)

        print(
            f"Eligible customers: {len(raw):,}"
        )

        print(
            f"Opportunity classes: "
            f"{opportunity.nunique()}"
        )

    # -------------------------------------------------------------
    # Save cross-cutoff summaries
    # -------------------------------------------------------------

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_DIR / "snapshot_summary.csv",
        index=False
    )

    opportunity_summary = pd.concat(
        opportunity_rows,
        ignore_index=True
    )

    opportunity_summary = opportunity_summary[
        ["cutoff", "Opportunity", "count", "share"]
    ]

    opportunity_summary.to_csv(
        OUTPUT_DIR / "opportunity_class_counts.csv",
        index=False
    )

    print("\n========================================")
    print("PHASE 02 LABEL RECONSTRUCTION COMPLETE")
    print("========================================")

    print(
        summary[
            [
                "cutoff",
                "eligible_customers",
                "fuel_zero_rate",
                "nonfuel_zero_rate",
                "opportunity_classes_present",
                "inactivity_rate",
                "stable_rate",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nOutputs written to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
