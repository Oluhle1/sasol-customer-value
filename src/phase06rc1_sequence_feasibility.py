"""
Phase 06R-C1
Sequence-aware CLV feasibility audit.

Research basis:
Bauer & Jannach (2021)
"Improved Customer Lifetime Value Prediction with
Sequence-To-Sequence Learning and Feature-based Models"

This phase DOES NOT train a neural network.

Questions:
1. Can the Sasol history support a 26-week sequential input before
   the official three-calendar-month CLV outcome?
2. Does the sliding-window construction generate multiple training
   examples per typical customer?
3. Do typical customers contain more than one active weekly time step,
   so that meaningful temporal ordering exists?

Sasol adaptation:
- weekly input granularity;
- 26-week input sequence;
- official three-calendar-month target remains unchanged;
- origins are shifted backwards in seven-day increments;
- only origins whose complete future target would be observable before
  the evaluation cutoff are allowed.

No target outcomes are used for the feasibility decision.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from reference.label_rules import load_transactions


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT
    / "data"
    / "raw"
    / "train.csv"
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
    / "sequence_feasibility"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


INPUT_WEEKS = 26
STEP_DAYS = 7

STAGES = [
    (
        "F1",
        pd.Timestamp("2024-12-01"),
    ),
    (
        "F2",
        pd.Timestamp("2025-03-01"),
    ),
    (
        "F3",
        pd.Timestamp("2025-06-01"),
    ),
    (
        "F4",
        pd.Timestamp("2025-09-01"),
    ),
    (
        "FINAL",
        pd.Timestamp("2025-12-01"),
    ),
]


def q(values, p):

    values = pd.Series(
        values,
        dtype="float64",
    )

    if values.empty:
        return np.nan

    return float(
        values.quantile(p)
    )


def candidate_origins(
    evaluation_cutoff,
    data_start,
):

    # The most recent training target must end
    # no later than the evaluation origin.
    latest = (
        evaluation_cutoff
        -
        pd.DateOffset(
            months=3
        )
    )

    origins = []

    current = latest

    while True:

        input_start = (
            current
            -
            pd.Timedelta(
                weeks=INPUT_WEEKS
            )
        )

        if input_start < data_start:
            break

        origins.append(
            current
        )

        current = (
            current
            -
            pd.Timedelta(
                days=STEP_DAYS
            )
        )

    return sorted(
        origins
    )


def eligible_ids(
    frame,
    origin,
    minimum,
):

    history = frame[
        frame["_time"].lt(
            origin
        )
    ]

    counts = (
        history
        .groupby(
            "ID"
        )[
            "TransactionId"
        ]
        .nunique()
    )

    return pd.Index(
        counts[
            counts.ge(
                minimum
            )
        ].index.astype(str)
    )


def active_week_counts(
    frame,
    ids,
    origin,
    included_categories,
    adjustment_items,
):

    input_start = (
        origin
        -
        pd.Timedelta(
            weeks=INPUT_WEEKS
        )
    )

    block = frame[
        frame["_time"].ge(
            input_start
        )
        &
        frame["_time"].lt(
            origin
        )
        &
        frame["ID"].isin(
            ids
        )
    ].copy()

    block["_week"] = (
        (
            block["_time"]
            -
            input_start
        )
        .dt.total_seconds()
        //
        (
            7
            *
            24
            *
            60
            *
            60
        )
    ).astype(
        "int16"
    )

    positive = (
        block["_quantity"].gt(0)
        &
        block["_amount"].gt(0)
    )

    in_scope = (
        block[
            "ItemCategoryLevel2"
        ].isin(
            included_categories
        )
        &
        ~block[
            "ItemName"
        ].isin(
            adjustment_items
        )
    )

    qualifying = block[
        positive
        &
        in_scope
    ]

    fuel = block[
        positive
        &
        ~block[
            "ItemName"
        ].isin(
            adjustment_items
        )
        &
        block[
            "ItemCategoryLevel2"
        ].eq(
            "Fuel"
        )
        &
        block[
            "LineItemType"
        ].eq(
            "Sale_Fuel"
        )
    ]

    nonfuel = block[
        positive
        &
        in_scope
        &
        ~block[
            "ItemCategoryLevel2"
        ].eq(
            "Fuel"
        )
    ]

    active = (
        qualifying
        .groupby(
            "ID"
        )[
            "_week"
        ]
        .nunique()
        .reindex(
            ids,
            fill_value=0,
        )
    )

    fuel_active = (
        fuel
        .groupby(
            "ID"
        )[
            "_week"
        ]
        .nunique()
        .reindex(
            ids,
            fill_value=0,
        )
    )

    nonfuel_active = (
        nonfuel
        .groupby(
            "ID"
        )[
            "_week"
        ]
        .nunique()
        .reindex(
            ids,
            fill_value=0,
        )
    )

    basket_count = (
        qualifying
        .groupby(
            "ID"
        )[
            "TransactionId"
        ]
        .nunique()
        .reindex(
            ids,
            fill_value=0,
        )
    )

    result = pd.DataFrame(
        {
            "ID":
                ids,

            "active_weeks":
                active.to_numpy(
                    dtype="int64"
                ),

            "fuel_active_weeks":
                fuel_active.to_numpy(
                    dtype="int64"
                ),

            "nonfuel_active_weeks":
                nonfuel_active.to_numpy(
                    dtype="int64"
                ),

            "qualifying_baskets":
                basket_count.to_numpy(
                    dtype="int64"
                ),
        }
    )

    return result


def main():

    print(
        "Loading Sasol history..."
    )

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    frame = load_transactions(
        TRAIN_PATH
    )

    frame[
        "ID"
    ] = frame[
        "ID"
    ].astype(str)

    minimum = int(
        config[
            "minimum_historical_transactions"
        ]
    )

    included_categories = {
        category
        for category, rule
        in config[
            "category_mapping"
        ].items()
        if rule[
            "included"
        ]
    }

    adjustment_items = set(
        config[
            "adjustment_items"
        ]
    )

    data_start = (
        frame[
            "_time"
        ]
        .min()
        .normalize()
    )

    data_end = (
        frame[
            "_time"
        ]
        .max()
    )

    print()
    print(
        f"Raw rows       : {len(frame):,}"
    )

    print(
        f"Customers      : {frame['ID'].nunique():,}"
    )

    print(
        f"History start  : {data_start.date()}"
    )

    print(
        f"History end    : {data_end}"
    )

    print(
        f"Input sequence : {INPUT_WEEKS} weekly steps"
    )

    print(
        "Target horizon : official 3 calendar months"
    )

    print(
        "Sliding step   : 7 days"
    )

    stage_rows = []
    origin_rows = []
    customer_rows = []
    density_rows = []

    for (
        stage,
        evaluation_cutoff,
    ) in STAGES:

        origins = candidate_origins(
            evaluation_cutoff,
            data_start,
        )

        latest_train_origin = (
            evaluation_cutoff
            -
            pd.DateOffset(
                months=3
            )
        )

        print()
        print(
            "=" * 86
        )

        print(
            f"{stage} | "
            f"evaluation={evaluation_cutoff.date()} | "
            f"latest train origin={latest_train_origin.date()}"
        )

        print(
            "=" * 86
        )

        if not origins:

            print(
                "No complete 26-week input sequence is available."
            )

            stage_rows.append(
                {
                    "stage":
                        stage,

                    "evaluation_cutoff":
                        evaluation_cutoff,

                    "latest_train_origin":
                        latest_train_origin,

                    "candidate_origins":
                        0,

                    "earliest_origin":
                        pd.NaT,

                    "latest_origin":
                        pd.NaT,

                    "latest_eligible_customers":
                        0,

                    "total_customer_origin_examples":
                        0,

                    "examples_per_customer_p25":
                        np.nan,

                    "examples_per_customer_median":
                        np.nan,

                    "examples_per_customer_p75":
                        np.nan,

                    "share_customers_ge2_examples":
                        np.nan,

                    "latest_active_weeks_p25":
                        np.nan,

                    "latest_active_weeks_median":
                        np.nan,

                    "latest_active_weeks_p75":
                        np.nan,

                    "latest_zero_active_week_share":
                        np.nan,

                    "latest_fuel_active_weeks_median":
                        np.nan,

                    "latest_nonfuel_active_weeks_median":
                        np.nan,

                    "multiple_example_condition":
                        False,

                    "temporal_activity_condition":
                        False,
                }
            )

            continue

        latest_ids = eligible_ids(
            frame,
            latest_train_origin,
            minimum,
        )

        example_counts = pd.Series(
            0,
            index=latest_ids,
            dtype="int64",
        )

        total_examples = 0

        for origin in origins:

            ids = eligible_ids(
                frame,
                origin,
                minimum,
            )

            total_examples += len(
                ids
            )

            overlapping = (
                example_counts.index
                .intersection(
                    ids
                )
            )

            example_counts.loc[
                overlapping
            ] += 1

            origin_rows.append(
                {
                    "stage":
                        stage,

                    "evaluation_cutoff":
                        evaluation_cutoff,

                    "origin":
                        origin,

                    "input_start":
                        origin
                        -
                        pd.Timedelta(
                            weeks=INPUT_WEEKS
                        ),

                    "outcome_end":
                        origin
                        +
                        pd.DateOffset(
                            months=3
                        ),

                    "eligible_customers":
                        len(
                            ids
                        ),
                }
            )

        density = active_week_counts(
            frame,
            latest_ids,
            latest_train_origin,
            included_categories,
            adjustment_items,
        )

        density[
            "stage"
        ] = stage

        density[
            "origin"
        ] = latest_train_origin

        density_rows.append(
            density
        )

        customer_block = pd.DataFrame(
            {
                "ID":
                    example_counts.index,

                "stage":
                    stage,

                "evaluation_cutoff":
                    evaluation_cutoff,

                "examples":
                    example_counts.to_numpy(),
            }
        )

        customer_rows.append(
            customer_block
        )

        example_median = q(
            example_counts,
            0.50,
        )

        active_median = q(
            density[
                "active_weeks"
            ],
            0.50,
        )

        multiple_condition = (
            example_median >= 2
        )

        activity_condition = (
            active_median >= 2
        )

        share_ge2 = float(
            example_counts
            .ge(2)
            .mean()
        )

        zero_activity_share = float(
            density[
                "active_weeks"
            ]
            .eq(0)
            .mean()
        )

        row = {
            "stage":
                stage,

            "evaluation_cutoff":
                evaluation_cutoff,

            "latest_train_origin":
                latest_train_origin,

            "candidate_origins":
                len(
                    origins
                ),

            "earliest_origin":
                min(
                    origins
                ),

            "latest_origin":
                max(
                    origins
                ),

            "latest_eligible_customers":
                len(
                    latest_ids
                ),

            "total_customer_origin_examples":
                total_examples,

            "examples_per_customer_p25":
                q(
                    example_counts,
                    0.25,
                ),

            "examples_per_customer_median":
                example_median,

            "examples_per_customer_p75":
                q(
                    example_counts,
                    0.75,
                ),

            "share_customers_ge2_examples":
                share_ge2,

            "latest_active_weeks_p25":
                q(
                    density[
                        "active_weeks"
                    ],
                    0.25,
                ),

            "latest_active_weeks_median":
                active_median,

            "latest_active_weeks_p75":
                q(
                    density[
                        "active_weeks"
                    ],
                    0.75,
                ),

            "latest_zero_active_week_share":
                zero_activity_share,

            "latest_fuel_active_weeks_median":
                q(
                    density[
                        "fuel_active_weeks"
                    ],
                    0.50,
                ),

            "latest_nonfuel_active_weeks_median":
                q(
                    density[
                        "nonfuel_active_weeks"
                    ],
                    0.50,
                ),

            "multiple_example_condition":
                multiple_condition,

            "temporal_activity_condition":
                activity_condition,
        }

        stage_rows.append(
            row
        )

        print(
            f"Candidate weekly origins       : {len(origins)}"
        )

        print(
            f"Latest eligible customers      : {len(latest_ids):,}"
        )

        print(
            f"Total customer-origin examples : {total_examples:,}"
        )

        print(
            "Examples/customer "
            f"p25={row['examples_per_customer_p25']:.1f} "
            f"median={row['examples_per_customer_median']:.1f} "
            f"p75={row['examples_per_customer_p75']:.1f}"
        )

        print(
            f"Customers with >=2 examples    : {share_ge2:.3%}"
        )

        print(
            "Latest 26w active weeks "
            f"p25={row['latest_active_weeks_p25']:.1f} "
            f"median={row['latest_active_weeks_median']:.1f} "
            f"p75={row['latest_active_weeks_p75']:.1f}"
        )

        print(
            f"All-zero latest sequences      : {zero_activity_share:.3%}"
        )

        print(
            "Median fuel active weeks       : "
            f"{row['latest_fuel_active_weeks_median']:.1f}"
        )

        print(
            "Median nonfuel active weeks    : "
            f"{row['latest_nonfuel_active_weeks_median']:.1f}"
        )

    stages = pd.DataFrame(
        stage_rows
    )

    origins = pd.DataFrame(
        origin_rows
    )

    customers = (
        pd.concat(
            customer_rows,
            ignore_index=True,
        )
        if customer_rows
        else
        pd.DataFrame()
    )

    densities = (
        pd.concat(
            density_rows,
            ignore_index=True,
        )
        if density_rows
        else
        pd.DataFrame()
    )

    # F1 is diagnostic only because the source-faithful
    # 26-week input does not fit before Sep-2024.
    decision_stages = (
        stages[
            stages[
                "stage"
            ].isin(
                [
                    "F2",
                    "F3",
                    "F4",
                    "FINAL",
                ]
            )
        ]
    )

    enough_examples = bool(
        decision_stages[
            "multiple_example_condition"
        ].all()
    )

    enough_temporal_activity = bool(
        decision_stages[
            "temporal_activity_condition"
        ].all()
    )

    c1_pass = (
        enough_examples
        and
        enough_temporal_activity
        and
        len(
            decision_stages
        ) == 4
    )

    stages.to_csv(
        OUTPUT_DIR
        /
        "stage_summary.csv",
        index=False,
    )

    origins.to_csv(
        OUTPUT_DIR
        /
        "candidate_origins.csv",
        index=False,
    )

    if not customers.empty:

        customers.to_csv(
            OUTPUT_DIR
            /
            "customer_example_counts.csv",
            index=False,
        )

    if not densities.empty:

        densities.to_csv(
            OUTPUT_DIR
            /
            "latest_sequence_density.csv",
            index=False,
        )

    decision = {
        "phase":
            "06R-C1",

        "input_weeks":
            INPUT_WEEKS,

        "output_horizon":
            "official three calendar months",

        "sliding_step_days":
            STEP_DAYS,

        "decision_stages":
            [
                "F2",
                "F3",
                "F4",
                "FINAL",
            ],

        "criteria":
            {
                "median_examples_per_customer_at_least_2":
                    True,

                "median_latest_active_weeks_at_least_2":
                    True,
            },

        "multiple_example_condition":
            enough_examples,

        "temporal_activity_condition":
            enough_temporal_activity,

        "c1_pass":
            c1_pass,
    }

    (
        OUTPUT_DIR
        /
        "decision.json"
    ).write_text(
        json.dumps(
            decision,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 100
    )

    print(
        "PHASE 06R-C1 — SEQUENCE FEASIBILITY SUMMARY"
    )

    print(
        "=" * 100
    )

    display_columns = [
        "stage",
        "candidate_origins",
        "latest_eligible_customers",
        "total_customer_origin_examples",
        "examples_per_customer_median",
        "share_customers_ge2_examples",
        "latest_active_weeks_median",
        "latest_zero_active_week_share",
        "latest_fuel_active_weeks_median",
        "latest_nonfuel_active_weeks_median",
        "multiple_example_condition",
        "temporal_activity_condition",
    ]

    print()

    print(
        stages[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "C1 DECISION"
    )

    print(
        "=" * 100
    )

    print(
        f"Multiple-example condition : {enough_examples}"
    )

    print(
        f"Temporal-activity condition: {enough_temporal_activity}"
    )

    print(
        f"C1 PASS                    : {c1_pass}"
    )

    print()
    print(
        f"Outputs: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
