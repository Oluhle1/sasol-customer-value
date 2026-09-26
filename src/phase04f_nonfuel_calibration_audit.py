"""
Phase 04F: Non-fuel occurrence probability calibration audit.

No model is trained.

Uses Phase 04E temporal validation predictions to determine whether
P(CLV_nonfuel > 0 | X) is appropriately calibrated before it is
multiplied by the conditional positive-magnitude estimate.

Diagnostics:
- prevalence versus mean predicted probability;
- probability bias;
- ROC AUC;
- Brier score;
- binary log loss;
- expected calibration error;
- probability-bin reliability tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
    / "nonfuel_twopart"
    / "validation_predictions.csv"
)

FOLD_COMPARISON_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
    / "nonfuel_twopart"
    / "fold_comparison.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
    / "nonfuel_calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def expected_calibration_error(
    truth,
    probability,
    bins=10,
):
    """
    Equal-frequency ECE.

    Each bin contributes in proportion to its number of observations.
    """

    frame = pd.DataFrame(
        {
            "truth":
                np.asarray(
                    truth,
                    dtype="float64",
                ),

            "probability":
                np.asarray(
                    probability,
                    dtype="float64",
                ),
        }
    )

    frame[
        "bin"
    ] = pd.qcut(
        frame[
            "probability"
        ],
        q=bins,
        labels=False,
        duplicates="drop",
    )

    ece = 0.0

    for _, block in frame.groupby(
        "bin"
    ):

        observed = (
            block[
                "truth"
            ].mean()
        )

        predicted = (
            block[
                "probability"
            ].mean()
        )

        ece += (
            len(block)
            /
            len(frame)
            *
            abs(
                observed
                -
                predicted
            )
        )

    return float(
        ece
    )


def main():

    print(
        "Loading Phase 04E probability outputs..."
    )

    data = pd.read_csv(
        INPUT_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    fold_info = pd.read_csv(
        FOLD_COMPARISON_PATH
    )

    required = {
        "fold_id",
        "CLV_nonfuel",
        "positive_probability",
        "conditional_positive_prediction",
        "twopart_prediction",
    }

    missing = (
        required
        - set(
            data.columns
        )
    )

    if missing:

        raise AssertionError(
            f"Missing columns: "
            f"{missing}"
        )

    data[
        "positive_truth"
    ] = (
        data[
            "CLV_nonfuel"
        ].gt(0)
        .astype(int)
    )

    probability = data[
        "positive_probability"
    ]

    if (
        probability.lt(0).any()
        or
        probability.gt(1).any()
    ):

        raise AssertionError(
            "Probability outside [0,1]."
        )

    summary_rows = []
    bin_rows = []

    # ============================================================
    # PER-FOLD CALIBRATION
    # ============================================================

    for fold_id in sorted(
        data[
            "fold_id"
        ].unique()
    ):

        block = data[
            data[
                "fold_id"
            ].eq(
                fold_id
            )
        ].copy()

        y = block[
            "positive_truth"
        ].to_numpy(
            dtype=int
        )

        p = block[
            "positive_probability"
        ].to_numpy(
            dtype="float64"
        )

        prevalence = float(
            y.mean()
        )

        mean_probability = float(
            p.mean()
        )

        info = (
            fold_info[
                fold_info[
                    "fold_id"
                ].eq(
                    fold_id
                )
            ]
            .iloc[0]
        )

        training_prevalence = (
            float(
                info[
                    "positive_train_rows"
                ]
            )
            /
            float(
                info[
                    "train_rows"
                ]
            )
        )

        summary_rows.append(
            {
                "fold_id":
                    fold_id,

                "rows":
                    len(
                        block
                    ),

                "training_positive_prevalence":
                    training_prevalence,

                "validation_positive_prevalence":
                    prevalence,

                "mean_predicted_probability":
                    mean_probability,

                "probability_bias_pred_minus_actual":
                    (
                        mean_probability
                        -
                        prevalence
                    ),

                "roc_auc":
                    float(
                        roc_auc_score(
                            y,
                            p,
                        )
                    ),

                "brier_score":
                    float(
                        brier_score_loss(
                            y,
                            p,
                        )
                    ),

                "binary_log_loss":
                    float(
                        log_loss(
                            y,
                            np.column_stack(
                                [
                                    1.0 - p,
                                    p,
                                ]
                            ),
                            labels=[
                                0,
                                1,
                            ],
                        )
                    ),

                "expected_calibration_error":
                    expected_calibration_error(
                        y,
                        p,
                        bins=10,
                    ),

                "mean_probability_true_zero":
                    float(
                        block.loc[
                            block[
                                "positive_truth"
                            ].eq(0),
                            "positive_probability",
                        ].mean()
                    ),

                "mean_probability_true_positive":
                    float(
                        block.loc[
                            block[
                                "positive_truth"
                            ].eq(1),
                            "positive_probability",
                        ].mean()
                    ),
            }
        )

        # Equal-frequency reliability bins.
        block[
            "probability_bin"
        ] = pd.qcut(
            block[
                "positive_probability"
            ],
            q=10,
            labels=False,
            duplicates="drop",
        )

        for bin_id, bin_block in block.groupby(
            "probability_bin"
        ):

            predicted_mean = float(
                bin_block[
                    "positive_probability"
                ].mean()
            )

            observed_rate = float(
                bin_block[
                    "positive_truth"
                ].mean()
            )

            bin_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "probability_decile":
                        int(
                            bin_id
                        )
                        + 1,

                    "rows":
                        len(
                            bin_block
                        ),

                    "probability_min":
                        float(
                            bin_block[
                                "positive_probability"
                            ].min()
                        ),

                    "probability_max":
                        float(
                            bin_block[
                                "positive_probability"
                            ].max()
                        ),

                    "mean_predicted_probability":
                        predicted_mean,

                    "observed_positive_rate":
                        observed_rate,

                    "calibration_gap_pred_minus_actual":
                        (
                            predicted_mean
                            -
                            observed_rate
                        ),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    reliability = pd.DataFrame(
        bin_rows
    )

    # ============================================================
    # POOLED DIAGNOSTIC
    # ============================================================

    y_all = data[
        "positive_truth"
    ].to_numpy(
        dtype=int
    )

    p_all = data[
        "positive_probability"
    ].to_numpy(
        dtype="float64"
    )

    pooled = pd.DataFrame(
        [
            {
                "rows":
                    len(
                        data
                    ),

                "positive_prevalence":
                    float(
                        y_all.mean()
                    ),

                "mean_predicted_probability":
                    float(
                        p_all.mean()
                    ),

                "probability_bias_pred_minus_actual":
                    float(
                        p_all.mean()
                        -
                        y_all.mean()
                    ),

                "roc_auc":
                    float(
                        roc_auc_score(
                            y_all,
                            p_all,
                        )
                    ),

                "brier_score":
                    float(
                        brier_score_loss(
                            y_all,
                            p_all,
                        )
                    ),

                "binary_log_loss":
                    float(
                        log_loss(
                            y_all,
                            np.column_stack(
                                [
                                    1.0 - p_all,
                                    p_all,
                                ]
                            ),
                            labels=[
                                0,
                                1,
                            ],
                        )
                    ),

                "expected_calibration_error":
                    expected_calibration_error(
                        y_all,
                        p_all,
                        bins=10,
                    ),
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "calibration_summary_by_fold.csv",
        index=False,
    )

    reliability.to_csv(
        OUTPUT_DIR
        / "reliability_by_probability_decile.csv",
        index=False,
    )

    pooled.to_csv(
        OUTPUT_DIR
        / "pooled_calibration_summary.csv",
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
        "PHASE 04F NON-FUEL CALIBRATION AUDIT COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "CALIBRATION SUMMARY"
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "F4 RELIABILITY TABLE"
    )

    print()

    print(
        reliability[
            reliability[
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
        "POOLED CALIBRATION"
    )

    print()

    print(
        pooled.to_string(
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
