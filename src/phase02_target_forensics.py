"""
Phase 02B: Detailed target forensics.

This script analyses the historical labels reconstructed during Phase 02A.

It does NOT:
- engineer predictive features;
- train machine-learning models;
- modify Sasol's label definitions.

It investigates:
- regression target distributions;
- target sparsity;
- fuel/non-fuel relationships;
- Opportunity class imbalance;
- temporal class drift;
- Opportunity transitions;
- cross-target consistency;
- target behaviour by Opportunity class.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "label_config.json"

LABEL_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase02_labels"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase02_forensics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def opportunity_macro(label):
    """
    Analytical grouping only.

    The official prediction target remains the original
    17-class Opportunity label.
    """

    if label.startswith(
        "Existing-category growth:"
    ):
        return "Growth"

    if label.startswith(
        "New category adoption:"
    ):
        return "Adoption"

    return label


def distribution_summary(series):
    """
    Standard descriptive statistics for a numeric target.
    """

    return {
        "count": len(series),
        "zero_count": int(series.eq(0).sum()),
        "zero_rate": float(series.eq(0).mean()),
        "mean": float(series.mean()),
        "std_population": float(series.std(ddof=0)),
        "minimum": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "maximum": float(series.max()),
    }


def main():

    print(
        "Loading Phase 02 configuration..."
    )

    config = json.loads(
        CONFIG_PATH.read_text()
    )

    cutoffs = config[
        "calibration_cutoffs"
    ]

    official_labels = config[
        "opportunity_labels"
    ]

    snapshots = {}

    # ============================================================
    # LOAD ALL RECONSTRUCTED SNAPSHOTS
    # ============================================================

    for cutoff in cutoffs:

        raw_path = (
            LABEL_DIR
            / f"labels_raw_{cutoff}.csv"
        )

        transformed_path = (
            LABEL_DIR
            / f"labels_transformed_{cutoff}.csv"
        )

        if not raw_path.exists():
            raise FileNotFoundError(
                raw_path
            )

        if not transformed_path.exists():
            raise FileNotFoundError(
                transformed_path
            )

        raw = pd.read_csv(
            raw_path,
            dtype={"ID": str}
        )

        transformed = pd.read_csv(
            transformed_path,
            dtype={"ID": str}
        )

        assert raw["ID"].is_unique
        assert transformed["ID"].is_unique

        assert raw["ID"].equals(
            transformed["ID"]
        )

        snapshots[cutoff] = {
            "raw": raw,
            "transformed": transformed,
        }

    # ============================================================
    # 1. TARGET DISTRIBUTIONS
    # ============================================================

    target_rows = []

    for cutoff in cutoffs:

        raw = snapshots[cutoff]["raw"]
        transformed = snapshots[
            cutoff
        ]["transformed"]

        targets = {
            "fuel_litres":
                raw["fuel_litres"],

            "nonfuel_rand":
                raw["nonfuel_rand"],

            "CLV_fuel":
                transformed["CLV_fuel"],

            "CLV_nonfuel":
                transformed["CLV_nonfuel"],
        }

        for target_name, values in targets.items():

            row = {
                "cutoff": cutoff,
                "target": target_name,
            }

            row.update(
                distribution_summary(values)
            )

            target_rows.append(row)

    target_summary = pd.DataFrame(
        target_rows
    )

    target_summary.to_csv(
        OUTPUT_DIR
        / "target_distribution_summary.csv",
        index=False,
    )

    # ============================================================
    # 2. FUEL / NON-FUEL RELATIONSHIP
    # ============================================================

    correlation_rows = []

    for cutoff in cutoffs:

        raw = snapshots[cutoff]["raw"]
        transformed = snapshots[
            cutoff
        ]["transformed"]

        correlation_rows.append(
            {
                "cutoff": cutoff,

                "pearson_raw":
                    raw[
                        [
                            "fuel_litres",
                            "nonfuel_rand",
                        ]
                    ]
                    .corr(
                        method="pearson"
                    )
                    .iloc[0, 1],

                "spearman_raw":
                    raw[
                        [
                            "fuel_litres",
                            "nonfuel_rand",
                        ]
                    ]
                    .corr(
                        method="spearman"
                    )
                    .iloc[0, 1],

                "pearson_transformed":
                    transformed[
                        [
                            "CLV_fuel",
                            "CLV_nonfuel",
                        ]
                    ]
                    .corr(
                        method="pearson"
                    )
                    .iloc[0, 1],

                "spearman_transformed":
                    transformed[
                        [
                            "CLV_fuel",
                            "CLV_nonfuel",
                        ]
                    ]
                    .corr(
                        method="spearman"
                    )
                    .iloc[0, 1],
            }
        )

    correlations = pd.DataFrame(
        correlation_rows
    )

    correlations.to_csv(
        OUTPUT_DIR
        / "target_correlations.csv",
        index=False,
    )

    # ============================================================
    # 3. ZERO-TARGET / INACTIVITY CONSISTENCY
    # ============================================================

    consistency_rows = []

    for cutoff in cutoffs:

        raw = snapshots[cutoff]["raw"]

        fuel_zero = raw[
            "fuel_litres"
        ].eq(0)

        nonfuel_zero = raw[
            "nonfuel_rand"
        ].eq(0)

        inactivity = raw[
            "Opportunity"
        ].eq("Inactivity")

        both_zero = (
            fuel_zero
            & nonfuel_zero
        )

        consistency_rows.append(
            {
                "cutoff": cutoff,

                "customers":
                    len(raw),

                "both_zero":
                    int(
                        both_zero.sum()
                    ),

                "fuel_zero_only":
                    int(
                        (
                            fuel_zero
                            & ~nonfuel_zero
                        ).sum()
                    ),

                "nonfuel_zero_only":
                    int(
                        (
                            ~fuel_zero
                            & nonfuel_zero
                        ).sum()
                    ),

                "neither_zero":
                    int(
                        (
                            ~fuel_zero
                            & ~nonfuel_zero
                        ).sum()
                    ),

                "inactivity":
                    int(
                        inactivity.sum()
                    ),

                "inactivity_with_nonzero_target":
                    int(
                        (
                            inactivity
                            & ~both_zero
                        ).sum()
                    ),

                "both_zero_but_not_inactivity":
                    int(
                        (
                            both_zero
                            & ~inactivity
                        ).sum()
                    ),
            }
        )

    consistency = pd.DataFrame(
        consistency_rows
    )

    consistency.to_csv(
        OUTPUT_DIR
        / "zero_target_consistency.csv",
        index=False,
    )

    # ============================================================
    # 4. FULL 17-CLASS OPPORTUNITY DISTRIBUTION
    # ============================================================

    class_rows = []

    for cutoff in cutoffs:

        raw = snapshots[cutoff]["raw"]

        counts = (
            raw["Opportunity"]
            .value_counts()
            .reindex(
                official_labels,
                fill_value=0
            )
        )

        for label in official_labels:

            count = int(
                counts[label]
            )

            class_rows.append(
                {
                    "cutoff": cutoff,
                    "Opportunity": label,
                    "count": count,
                    "share":
                        count / len(raw),
                    "macro_class":
                        opportunity_macro(
                            label
                        ),
                }
            )

    class_distribution = pd.DataFrame(
        class_rows
    )

    class_distribution.to_csv(
        OUTPUT_DIR
        / "opportunity_class_distribution.csv",
        index=False,
    )

    class_wide = (
        class_distribution
        .pivot(
            index="Opportunity",
            columns="cutoff",
            values="share",
        )
        .reset_index()
    )

    class_wide.to_csv(
        OUTPUT_DIR
        / "opportunity_class_distribution_wide.csv",
        index=False,
    )

    # ============================================================
    # 5. FIRST -> LAST CLASS DRIFT
    # ============================================================

    first_cutoff = cutoffs[0]
    last_cutoff = cutoffs[-1]

    first_dist = (
        class_distribution[
            class_distribution[
                "cutoff"
            ].eq(first_cutoff)
        ][
            [
                "Opportunity",
                "count",
                "share",
            ]
        ]
        .rename(
            columns={
                "count":
                    "first_count",
                "share":
                    "first_share",
            }
        )
    )

    last_dist = (
        class_distribution[
            class_distribution[
                "cutoff"
            ].eq(last_cutoff)
        ][
            [
                "Opportunity",
                "count",
                "share",
            ]
        ]
        .rename(
            columns={
                "count":
                    "last_count",
                "share":
                    "last_share",
            }
        )
    )

    drift = first_dist.merge(
        last_dist,
        on="Opportunity",
        how="outer",
    )

    drift["share_change"] = (
        drift["last_share"]
        - drift["first_share"]
    )

    drift[
        "percentage_point_change"
    ] = (
        drift["share_change"]
        * 100
    )

    drift["relative_change"] = np.where(
        drift["first_share"] > 0,

        (
            drift["last_share"]
            - drift["first_share"]
        )
        / drift["first_share"],

        np.nan,
    )

    drift = drift.sort_values(
        "percentage_point_change",
        ascending=False,
    )

    drift.to_csv(
        OUTPUT_DIR
        / "opportunity_drift_first_to_last.csv",
        index=False,
    )

    # ============================================================
    # 6. MACRO OPPORTUNITY DISTRIBUTION
    # ============================================================

    macro_distribution = (
        class_distribution
        .groupby(
            [
                "cutoff",
                "macro_class",
            ],
            as_index=False,
        )
        .agg(
            count=("count", "sum"),
            share=("share", "sum"),
        )
    )

    macro_distribution.to_csv(
        OUTPUT_DIR
        / "opportunity_macro_distribution.csv",
        index=False,
    )

    # ============================================================
    # 7. CUSTOMER TRANSITIONS
    # ============================================================

    transition_rows = []
    macro_transition_rows = []

    for previous_cutoff, next_cutoff in zip(
        cutoffs[:-1],
        cutoffs[1:],
    ):

        previous = (
            snapshots[
                previous_cutoff
            ]["raw"][
                [
                    "ID",
                    "Opportunity",
                ]
            ]
            .rename(
                columns={
                    "Opportunity":
                        "from_label"
                }
            )
        )

        following = (
            snapshots[
                next_cutoff
            ]["raw"][
                [
                    "ID",
                    "Opportunity",
                ]
            ]
            .rename(
                columns={
                    "Opportunity":
                        "to_label"
                }
            )
        )

        # Only customers eligible in both snapshots
        joined = previous.merge(
            following,
            on="ID",
            how="inner",
        )

        transitions = (
            joined
            .groupby(
                [
                    "from_label",
                    "to_label",
                ]
            )
            .size()
            .rename("count")
            .reset_index()
        )

        from_totals = (
            transitions
            .groupby("from_label")[
                "count"
            ]
            .transform("sum")
        )

        transitions[
            "transition_rate"
        ] = (
            transitions["count"]
            / from_totals
        )

        transitions[
            "from_cutoff"
        ] = previous_cutoff

        transitions[
            "to_cutoff"
        ] = next_cutoff

        transition_rows.append(
            transitions
        )

        # --------------------------------------------
        # Macro transitions
        # --------------------------------------------

        joined[
            "from_macro"
        ] = (
            joined[
                "from_label"
            ]
            .map(
                opportunity_macro
            )
        )

        joined[
            "to_macro"
        ] = (
            joined[
                "to_label"
            ]
            .map(
                opportunity_macro
            )
        )

        macro_transitions = (
            joined
            .groupby(
                [
                    "from_macro",
                    "to_macro",
                ]
            )
            .size()
            .rename("count")
            .reset_index()
        )

        macro_totals = (
            macro_transitions
            .groupby(
                "from_macro"
            )["count"]
            .transform("sum")
        )

        macro_transitions[
            "transition_rate"
        ] = (
            macro_transitions[
                "count"
            ]
            / macro_totals
        )

        macro_transitions[
            "from_cutoff"
        ] = previous_cutoff

        macro_transitions[
            "to_cutoff"
        ] = next_cutoff

        macro_transition_rows.append(
            macro_transitions
        )

    transitions_all = pd.concat(
        transition_rows,
        ignore_index=True,
    )

    transitions_all.to_csv(
        OUTPUT_DIR
        / "opportunity_transitions.csv",
        index=False,
    )

    macro_transitions_all = pd.concat(
        macro_transition_rows,
        ignore_index=True,
    )

    macro_transitions_all.to_csv(
        OUTPUT_DIR
        / "opportunity_macro_transitions.csv",
        index=False,
    )

    # ============================================================
    # 8. TARGET DISTRIBUTION BY OPPORTUNITY CLASS
    # ============================================================

    target_class_rows = []

    for cutoff in cutoffs:

        raw = snapshots[
            cutoff
        ]["raw"]

        for label, group in raw.groupby(
            "Opportunity"
        ):

            target_class_rows.append(
                {
                    "cutoff": cutoff,
                    "Opportunity": label,
                    "macro_class":
                        opportunity_macro(
                            label
                        ),

                    "customers":
                        len(group),

                    "fuel_zero_rate":
                        group[
                            "fuel_litres"
                        ].eq(0).mean(),

                    "fuel_mean":
                        group[
                            "fuel_litres"
                        ].mean(),

                    "fuel_median":
                        group[
                            "fuel_litres"
                        ].median(),

                    "fuel_p90":
                        group[
                            "fuel_litres"
                        ].quantile(
                            0.90
                        ),

                    "nonfuel_zero_rate":
                        group[
                            "nonfuel_rand"
                        ].eq(0).mean(),

                    "nonfuel_mean":
                        group[
                            "nonfuel_rand"
                        ].mean(),

                    "nonfuel_median":
                        group[
                            "nonfuel_rand"
                        ].median(),

                    "nonfuel_p90":
                        group[
                            "nonfuel_rand"
                        ].quantile(
                            0.90
                        ),
                }
            )

    target_by_class = pd.DataFrame(
        target_class_rows
    )

    target_by_class.to_csv(
        OUTPUT_DIR
        / "targets_by_opportunity_class.csv",
        index=False,
    )

    # ============================================================
    # 9. AUTOMATED INTEGRITY TESTS
    # ============================================================

    assert (
        consistency[
            "inactivity_with_nonzero_target"
        ]
        .eq(0)
        .all()
    )

    assert (
        consistency[
            "both_zero_but_not_inactivity"
        ]
        .eq(0)
        .all()
    )

    assert (
        class_distribution
        .groupby("cutoff")[
            "count"
        ]
        .sum()
        .to_numpy()
        ==
        np.array(
            [
                len(
                    snapshots[c][
                        "raw"
                    ]
                )
                for c in cutoffs
            ]
        )
    ).all()

    assert (
        class_distribution
        .groupby("cutoff")[
            "Opportunity"
        ]
        .nunique()
        .eq(
            len(
                official_labels
            )
        )
        .all()
    )

    # ============================================================
    # PRINT RESEARCH CHECKPOINT
    # ============================================================

    print()
    print(
        "=" * 70
    )

    print(
        "PHASE 02B TARGET FORENSICS COMPLETE"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "TARGET CORRELATIONS"
    )

    print(
        correlations.round(
            4
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "ZERO-TARGET CONSISTENCY"
    )

    print(
        consistency.to_string(
            index=False
        )
    )

    print()
    print(
        "MACRO OPPORTUNITY DISTRIBUTION"
    )

    print(
        macro_distribution
        .assign(
            share=lambda x:
                x["share"]
                .round(4)
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "FIRST -> LAST OPPORTUNITY DRIFT"
    )

    print(
        drift[
            [
                "Opportunity",
                "first_share",
                "last_share",
                "percentage_point_change",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
