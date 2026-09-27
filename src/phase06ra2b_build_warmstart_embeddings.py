"""
Phase 06R-A2b
Longitudinal warm-start customer SGNS embeddings.

Primary methodological source:
Chamberlain et al. (2017),
Customer Lifetime Value Prediction Using Embeddings.

Core published principle:
- embedding axes learned independently at different times are not
  guaranteed to correspond;
- retain embeddings of existing customers;
- initialise new customers at small values;
- continue SGNS learning in the established representation space.

Sasol adaptation:
- product PURCHASE histories instead of ASOS product views,
  following Yan & Resnick (2024);
- quarterly incremental updates;
- Gensim incremental vocabulary/training is used to preserve the
  previous SGNS coordinate system.

Important deviation:
Chamberlain explicitly orders old-old, new-old, old-new and new-new
training pairs during live updating.

Gensim's public Word2Vec training interface does not expose directed
skip-gram pairs in that exact order. Therefore this implementation is
a warm-start SGNS adaptation, not an exact reproduction of their
ordered-pair training algorithm.

No predictive outcomes are used anywhere in this representation build.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from gensim.models import Word2Vec


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "train.csv"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "customer_embeddings_warmstart"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CUTOFFS = [
    pd.Timestamp("2024-09-01"),
    pd.Timestamp("2024-12-01"),
    pd.Timestamp("2025-03-01"),
    pd.Timestamp("2025-06-01"),
    pd.Timestamp("2025-09-01"),
    pd.Timestamp("2025-12-01"),
]

VECTOR_SIZE = 64
WINDOW_RADIUS = 5
NEGATIVE = 5
EPOCHS = 5
SEED = 42

ADJUSTMENT_ITEMS = {
    "TOTAL/ROUNDING DISCO",
    "PENSIONER DISCOUNT",
    "WOW_PROMO",
}


def load_events():

    raw = pd.read_csv(
        RAW_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    raw["_timestamp"] = pd.to_datetime(
        raw["DateTimeZA"],
        format="mixed",
        errors="raise",
    )

    raw["_quantity"] = pd.to_numeric(
        raw["Quantity"],
        errors="raise",
    )

    raw["_amount"] = pd.to_numeric(
        raw["TotalExclAmount"],
        errors="raise",
    )

    events = raw[
        raw["_quantity"].gt(0)
        &
        raw["_amount"].gt(0)
        &
        raw["ID"].ne("")
        &
        raw["TransactionId"].ne("")
        &
        raw["StockItemId"].ne("")
        &
        ~raw["ItemName"].isin(
            ADJUSTMENT_ITEMS
        )
    ].copy()

    events = (
        events
        .sort_values(
            [
                "_timestamp",
                "TransactionId",
                "StockItemId",
                "ID",
            ]
        )
        .drop_duplicates(
            [
                "ID",
                "TransactionId",
                "StockItemId",
            ]
        )
        .reset_index(drop=True)
    )

    return events[
        [
            "ID",
            "TransactionId",
            "StockItemId",
            "_timestamp",
        ]
    ]


def make_sequences(frame):

    if frame.empty:
        return []

    # A product must connect at least two distinct customers
    # to provide cross-customer embedding information.
    customer_counts = (
        frame
        .groupby(
            "StockItemId"
        )["ID"]
        .nunique()
    )

    products = set(
        customer_counts[
            customer_counts.ge(2)
        ].index.astype(str)
    )

    work = frame[
        frame["StockItemId"].isin(
            products
        )
    ].copy()

    work = work.sort_values(
        [
            "StockItemId",
            "_timestamp",
            "TransactionId",
            "ID",
        ]
    )

    sequences = (
        work
        .groupby(
            "StockItemId",
            sort=True,
        )["ID"]
        .apply(
            lambda values:
                values.astype(str).tolist()
        )
        .tolist()
    )

    return [
        sequence
        for sequence in sequences
        if len(sequence) >= 2
    ]


def model_to_frame(
    model,
    cutoff,
):

    customers = sorted(
        model.wv.index_to_key
    )

    matrix = np.vstack(
        [
            model.wv[
                customer
            ]
            for customer
            in customers
        ]
    ).astype(
        "float64"
    )

    columns = [
        f"cust_emb_{i:02d}"
        for i
        in range(
            VECTOR_SIZE
        )
    ]

    result = pd.DataFrame(
        matrix,
        columns=columns,
    )

    result.insert(
        0,
        "cutoff",
        cutoff.strftime(
            "%Y-%m-%d"
        ),
    )

    result.insert(
        0,
        "ID",
        customers,
    )

    return result


def main():

    print(
        "Loading positive Sasol purchase events..."
    )

    events = load_events()

    print(
        f"Basket-product events: {len(events):,}"
    )

    base_cutoff = CUTOFFS[0]

    base = events[
        events["_timestamp"].lt(
            base_cutoff
        )
    ].copy()

    base_sequences = make_sequences(
        base
    )

    print()
    print("=" * 78)
    print("BASE SGNS")
    print("=" * 78)

    print(
        f"Base events     : {len(base):,}"
    )

    print(
        f"Product docs    : {len(base_sequences):,}"
    )

    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW_RADIUS,
        min_count=1,
        workers=1,
        sg=1,
        hs=0,
        negative=NEGATIVE,
        ns_exponent=0.75,
        sample=0.0,
        seed=SEED,
        shrink_windows=False,
        sorted_vocab=1,
    )

    model.build_vocab(
        base_sequences
    )

    model.train(
        base_sequences,
        total_examples=len(
            base_sequences
        ),
        epochs=EPOCHS,
    )

    embedding_frames = [
        model_to_frame(
            model,
            base_cutoff,
        )
    ]

    audit_rows = [
        {
            "cutoff":
                base_cutoff.strftime(
                    "%Y-%m-%d"
                ),

            "stage":
                "base",

            "interval_events":
                len(base),

            "documents":
                len(
                    base_sequences
                ),

            "vocabulary_before":
                0,

            "vocabulary_after":
                len(
                    model.wv
                ),

            "new_customers":
                len(
                    model.wv
                ),

            "existing_active_customers":
                np.nan,

            "active_existing_share":
                np.nan,
        }
    ]

    previous_cutoff = base_cutoff

    for cutoff in CUTOFFS[1:]:

        print()
        print("=" * 78)
        print(
            f"WARM START {previous_cutoff.date()} -> {cutoff.date()}"
        )
        print("=" * 78)

        interval = events[
            events["_timestamp"].ge(
                previous_cutoff
            )
            &
            events["_timestamp"].lt(
                cutoff
            )
        ].copy()

        sequences = make_sequences(
            interval
        )

        old_vocabulary = set(
            model.wv.index_to_key
        )

        active_customers = set(
            interval[
                "ID"
            ].astype(str)
        )

        active_existing = (
            active_customers
            &
            old_vocabulary
        )

        active_new = (
            active_customers
            -
            old_vocabulary
        )

        print(
            f"Interval events       : {len(interval):,}"
        )

        print(
            f"Product documents     : {len(sequences):,}"
        )

        print(
            f"Active customers      : {len(active_customers):,}"
        )

        print(
            f"Existing active       : {len(active_existing):,}"
        )

        print(
            f"New active            : {len(active_new):,}"
        )

        if not sequences:

            raise AssertionError(
                f"No SGNS sequences for interval ending {cutoff}"
            )

        vocabulary_before = len(
            model.wv
        )

        # Existing vectors remain in place.
        # New customer tokens are appended with Gensim's small
        # random initialisation.
        model.build_vocab(
            sequences,
            update=True,
        )

        vocabulary_after = len(
            model.wv
        )

        added = (
            vocabulary_after
            -
            vocabulary_before
        )

        print(
            f"Vocabulary before     : {vocabulary_before:,}"
        )

        print(
            f"Vocabulary after      : {vocabulary_after:,}"
        )

        print(
            f"Vocabulary added      : {added:,}"
        )

        model.train(
            sequences,
            total_examples=len(
                sequences
            ),
            epochs=EPOCHS,
        )

        embedding_frames.append(
            model_to_frame(
                model,
                cutoff,
            )
        )

        audit_rows.append(
            {
                "cutoff":
                    cutoff.strftime(
                        "%Y-%m-%d"
                    ),

                "stage":
                    "incremental_sgns",

                "interval_events":
                    len(
                        interval
                    ),

                "documents":
                    len(
                        sequences
                    ),

                "vocabulary_before":
                    vocabulary_before,

                "vocabulary_after":
                    vocabulary_after,

                "new_customers":
                    added,

                "existing_active_customers":
                    len(
                        active_existing
                    ),

                "active_existing_share":
                    (
                        len(
                            active_existing
                        )
                        /
                        len(
                            active_customers
                        )
                        if active_customers
                        else np.nan
                    ),
            }
        )

        previous_cutoff = cutoff

    embeddings = pd.concat(
        embedding_frames,
        ignore_index=True,
    )

    embedding_columns = [
        column
        for column
        in embeddings.columns
        if column.startswith(
            "cust_emb_"
        )
    ]

    if len(
        embedding_columns
    ) != VECTOR_SIZE:

        raise AssertionError(
            f"Expected {VECTOR_SIZE} dimensions."
        )

    if embeddings.duplicated(
        [
            "ID",
            "cutoff",
        ]
    ).any():

        raise AssertionError(
            "Duplicate customer-cutoff embeddings."
        )

    if not np.isfinite(
        embeddings[
            embedding_columns
        ].to_numpy(
            dtype="float64"
        )
    ).all():

        raise AssertionError(
            "Non-finite embedding values detected."
        )

    audit = pd.DataFrame(
        audit_rows
    )

    embeddings.to_parquet(
        OUTPUT_DIR
        /
        "customer_embeddings.parquet",
        index=False,
    )

    audit.to_csv(
        OUTPUT_DIR
        /
        "embedding_build_audit.csv",
        index=False,
    )

    config = {
        "phase":
            "06R-A2b",

        "method":
            "warm-start incremental SGNS",

        "vector_size":
            VECTOR_SIZE,

        "window_radius":
            WINDOW_RADIUS,

        "total_context_length":
            2 * WINDOW_RADIUS + 1,

        "negative":
            NEGATIVE,

        "ns_exponent":
            0.75,

        "epochs_per_interval":
            EPOCHS,

        "seed":
            SEED,

        "workers":
            1,

        "sample":
            0.0,

        "shrink_windows":
            False,

        "deviation":
            (
                "Gensim incremental SGNS does not implement "
                "Chamberlain's exact ordered old-old/new-old/"
                "old-new/new-new pair schedule."
            ),
    }

    (
        OUTPUT_DIR
        /
        "embedding_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("PHASE 06R-A2b COMPLETE")
    print("=" * 78)
    print()

    print(
        audit.to_string(
            index=False
        )
    )

    print()
    print(
        f"Embedding rows : {len(embeddings):,}"
    )

    print(
        f"Dimensions     : {len(embedding_columns)}"
    )

    print()
    print(
        f"Outputs: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
