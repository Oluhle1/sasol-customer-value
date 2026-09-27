"""
Phase 06R-A2
Leakage-safe, longitudinally aligned customer embeddings.

Research basis
--------------
Chamberlain et al. (KDD 2017):
- customers are embedding tokens;
- product interaction histories define customer contexts;
- Skip-Gram with Negative Sampling;
- fixed context length 11 worked well;
- embedding dimensions in the 32-128 range performed strongly;
- embedding coordinates must remain aligned through time.

Yan & Resnick (QME 2024):
- adapt the approach to product-level PURCHASE data;
- group transactions by product;
- sort customers by purchase time;
- derive product embeddings from customer embeddings;
- refresh existing/new customer representations from later purchases.

Sasol implementation
--------------------
1. Train one SGNS base space using purchase history strictly before
   2024-09-01.
2. Preserve that coordinate system.
3. Advance through each three-month cutoff using product-centroid
   refreshes rather than independent Word2Vec retraining.
4. Never use a transaction on or after the snapshot cutoff when
   constructing that snapshot's embedding.
5. Save generated embeddings under data/processed only.

Paper-supported settings:
- SGNS
- fixed total context length 11
- modified unigram negative-sampling exponent 0.75

Pre-registered implementation choices:
- 64 dimensions: midpoint of published strong 32-128 range
- 5 negative samples: Gensim standard choice; paper does not specify k
- 5 epochs: fixed implementation choice; not tuned
- workers=1 and seed=42 for deterministic reproduction
- min_count=1 because Sasol customer IDs are the tokens
- no frequent-token subsampling
- products require >=2 distinct customers for the base corpus

Refresh implementation:
- product centroid = mean embedding of distinct customers known to
  have purchased that product before the current interval;
- an existing customer's updated embedding is the 50/50 mean of
  its previous embedding and the mean embeddings of products bought
  during the new interval;
- a new customer's embedding is the mean of available product
  embeddings from products bought during the interval;
- for a previously unseen product, a temporary product centroid can
  be inferred from interval purchasers who already have embeddings.

The exact refresh weighting and unseen-product fallback are explicit
Sasol implementation choices because the papers do not specify those
details completely.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from gensim.models import Word2Vec


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "train.csv"

SUPERVISED_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
    / "supervised_historical.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "customer_embeddings"
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

BASE_CUTOFF = CUTOFFS[0]

VECTOR_SIZE = 64

# Chamberlain reports total context length 11.
# Gensim's parameter is radius, therefore +/-5 plus centre = 11.
GENSIM_WINDOW_RADIUS = 5

NEGATIVE = 5
EPOCHS = 5
SEED = 42

ADJUSTMENT_ITEMS = {
    "TOTAL/ROUNDING DISCO",
    "PENSIONER DISCOUNT",
    "WOW_PROMO",
}


def load_purchase_events():
    print("Loading raw Sasol transaction history...")

    raw = pd.read_csv(
        RAW_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required = {
        "DateTimeZA",
        "TransactionId",
        "ID",
        "StockItemId",
        "ItemName",
        "Quantity",
        "TotalExclAmount",
    }

    missing = sorted(
        required
        -
        set(raw.columns)
    )

    if missing:
        raise AssertionError(
            f"Missing required columns: {missing}"
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

    positive = raw[
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

    # A basket-product is one product-purchase event.
    positive = (
        positive
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
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"Raw rows              : {len(raw):,}"
    )

    print(
        f"Basket-product events : {len(positive):,}"
    )

    return positive[
        [
            "ID",
            "TransactionId",
            "StockItemId",
            "_timestamp",
        ]
    ].copy()


def build_product_sequences(frame):
    """
    Products are documents and customer IDs are tokens.

    Only products shared by at least two distinct customers can
    generate cross-customer contexts.
    """

    purchaser_count = (
        frame
        .groupby(
            "StockItemId"
        )["ID"]
        .nunique()
    )

    qualifying_products = set(
        purchaser_count[
            purchaser_count.ge(2)
        ].index.astype(str)
    )

    work = frame[
        frame[
            "StockItemId"
        ].isin(
            qualifying_products
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
            lambda s:
                s.astype(str).tolist()
        )
        .tolist()
    )

    sequences = [
        sequence
        for sequence in sequences
        if len(sequence) >= 2
    ]

    return (
        sequences,
        qualifying_products,
        work,
    )


def train_base_embedding(base_history):
    print()
    print(
        "=" * 78
    )

    print(
        "TRAINING BASE SGNS SPACE"
    )

    print(
        "=" * 78
    )

    (
        sequences,
        qualifying_products,
        corpus_events,
    ) = build_product_sequences(
        base_history
    )

    if not sequences:
        raise AssertionError(
            "No usable product sequences for base embedding."
        )

    print(
        f"Base history events       : {len(base_history):,}"
    )

    print(
        f"Qualifying shared products: {len(qualifying_products):,}"
    )

    print(
        f"Corpus events             : {len(corpus_events):,}"
    )

    print(
        f"Product documents         : {len(sequences):,}"
    )

    model = Word2Vec(
        sentences=sequences,
        vector_size=VECTOR_SIZE,
        window=GENSIM_WINDOW_RADIUS,
        min_count=1,
        workers=1,
        sg=1,
        hs=0,
        negative=NEGATIVE,
        ns_exponent=0.75,
        sample=0.0,
        seed=SEED,
        epochs=EPOCHS,
        shrink_windows=False,
        sorted_vocab=1,
    )

    vectors = {
        str(customer):
            np.asarray(
                model.wv[str(customer)],
                dtype="float64",
            ).copy()
        for customer
        in model.wv.index_to_key
    }

    return (
        vectors,
        model,
        qualifying_products,
        corpus_events,
    )


def make_product_centroids(
    events,
    customer_vectors,
):
    """
    Average one vector per distinct customer-product relationship.

    Repeated purchases do not repeatedly weight the product centroid.
    """

    known = events[
        events["ID"].isin(
            customer_vectors
        )
    ][
        [
            "ID",
            "StockItemId",
        ]
    ].drop_duplicates()

    product_customers = (
        known
        .groupby(
            "StockItemId",
            sort=False,
        )["ID"]
        .apply(list)
    )

    centroids = {}

    for product, customers in product_customers.items():

        matrix = np.vstack(
            [
                customer_vectors[
                    str(customer)
                ]
                for customer
                in customers
            ]
        )

        centroids[
            str(product)
        ] = matrix.mean(
            axis=0
        )

    return centroids


def infer_new_product_centroids(
    interval,
    prior_vectors,
    existing_centroids,
):
    """
    For products unseen in the previous history, use interval buyers
    who already have customer embeddings to establish an aligned
    centroid where possible.
    """

    new_products = set(
        interval[
            "StockItemId"
        ].astype(str)
    ) - set(
        existing_centroids
    )

    if not new_products:
        return {}

    candidates = interval[
        interval[
            "StockItemId"
        ].isin(
            new_products
        )
        &
        interval[
            "ID"
        ].isin(
            prior_vectors
        )
    ][
        [
            "ID",
            "StockItemId",
        ]
    ].drop_duplicates()

    inferred = {}

    for product, group in candidates.groupby(
        "StockItemId"
    ):

        customers = group[
            "ID"
        ].astype(str).tolist()

        if not customers:
            continue

        inferred[
            str(product)
        ] = np.vstack(
            [
                prior_vectors[
                    customer
                ]
                for customer
                in customers
            ]
        ).mean(
            axis=0
        )

    return inferred


def refresh_customer_vectors(
    prior_vectors,
    interval,
    product_centroids,
):
    """
    Refresh in the fixed base coordinate system.

    Existing:
        0.5 * old customer vector
        +
        0.5 * mean(new product embeddings)

    New:
        mean(new product embeddings)
    """

    updated = {
        customer:
            vector.copy()
        for customer, vector
        in prior_vectors.items()
    }

    purchases = (
        interval
        .sort_values(
            [
                "_timestamp",
                "TransactionId",
                "StockItemId",
            ]
        )
        .groupby(
            "ID",
            sort=False,
        )["StockItemId"]
        .apply(
            lambda s:
                s.astype(str).tolist()
        )
    )

    old_updated = 0
    new_created = 0
    unresolved = 0

    for customer, products in purchases.items():

        customer = str(
            customer
        )

        available = [
            product_centroids[
                product
            ]
            for product
            in products
            if product
            in product_centroids
        ]

        if not available:
            unresolved += 1
            continue

        product_mean = np.vstack(
            available
        ).mean(
            axis=0
        )

        if customer in prior_vectors:

            updated[
                customer
            ] = (
                0.5
                *
                prior_vectors[
                    customer
                ]
                +
                0.5
                *
                product_mean
            )

            old_updated += 1

        else:

            updated[
                customer
            ] = product_mean.copy()

            new_created += 1

    return (
        updated,
        old_updated,
        new_created,
        unresolved,
    )


def vectors_to_frame(
    vectors,
    cutoff,
):
    customers = sorted(
        vectors
    )

    matrix = np.vstack(
        [
            vectors[
                customer
            ]
            for customer
            in customers
        ]
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

    result[
        "embedding_available"
    ] = 1

    return result


def historical_coverage(
    embeddings,
):
    if not SUPERVISED_PATH.exists():
        return pd.DataFrame()

    supervised = pd.read_csv(
        SUPERVISED_PATH,
        dtype={
            "ID": str,
            "cutoff": str,
        },
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )

    rows = []

    for cutoff in CUTOFFS[:-1]:

        cutoff_string = cutoff.strftime(
            "%Y-%m-%d"
        )

        population = supervised[
            supervised[
                "cutoff"
            ].eq(
                cutoff_string
            )
        ][
            [
                "ID",
                "cutoff",
            ]
        ].drop_duplicates()

        available = embeddings[
            embeddings[
                "cutoff"
            ].eq(
                cutoff_string
            )
        ][
            [
                "ID",
                "cutoff",
            ]
        ]

        merged = population.merge(
            available.assign(
                embedding_available=1
            ),
            on=[
                "ID",
                "cutoff",
            ],
            how="left",
        )

        covered = int(
            merged[
                "embedding_available"
            ].fillna(
                0
            ).sum()
        )

        rows.append(
            {
                "cutoff":
                    cutoff_string,

                "eligible_customers":
                    len(
                        population
                    ),

                "covered_customers":
                    covered,

                "coverage":
                    covered
                    /
                    len(
                        population
                    )
                    if len(
                        population
                    )
                    else np.nan,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    events = load_purchase_events()

    if (
        events[
            "_timestamp"
        ].max()
        >=
        CUTOFFS[-1]
    ):
        raise AssertionError(
            "Raw history unexpectedly contains events at or after "
            "the final 2025-12-01 forecast origin."
        )

    base_history = events[
        events[
            "_timestamp"
        ].lt(
            BASE_CUTOFF
        )
    ].copy()

    (
        customer_vectors,
        base_model,
        qualifying_products,
        base_corpus_events,
    ) = train_base_embedding(
        base_history
    )

    all_embedding_frames = [
        vectors_to_frame(
            customer_vectors,
            BASE_CUTOFF,
        )
    ]

    audit_rows = [
        {
            "cutoff":
                BASE_CUTOFF.strftime(
                    "%Y-%m-%d"
                ),

            "stage":
                "base_sgns",

            "vectors_available":
                len(
                    customer_vectors
                ),

            "interval_events":
                len(
                    base_history
                ),

            "old_customers_updated":
                np.nan,

            "new_customers_created":
                np.nan,

            "unresolved_interval_customers":
                np.nan,

            "known_product_centroids":
                np.nan,
        }
    ]

    previous_cutoff = BASE_CUTOFF

    for cutoff in CUTOFFS[1:]:

        print()
        print(
            "=" * 78
        )

        print(
            f"REFRESH {previous_cutoff.date()} -> {cutoff.date()}"
        )

        print(
            "=" * 78
        )

        prior_history = events[
            events[
                "_timestamp"
            ].lt(
                previous_cutoff
            )
        ].copy()

        interval = events[
            events[
                "_timestamp"
            ].ge(
                previous_cutoff
            )
            &
            events[
                "_timestamp"
            ].lt(
                cutoff
            )
        ].copy()

        product_centroids = (
            make_product_centroids(
                prior_history,
                customer_vectors,
            )
        )

        new_product_centroids = (
            infer_new_product_centroids(
                interval,
                customer_vectors,
                product_centroids,
            )
        )

        product_centroids.update(
            new_product_centroids
        )

        (
            customer_vectors,
            old_updated,
            new_created,
            unresolved,
        ) = refresh_customer_vectors(
            customer_vectors,
            interval,
            product_centroids,
        )

        print(
            f"Interval events       : {len(interval):,}"
        )

        print(
            f"Product centroids     : {len(product_centroids):,}"
        )

        print(
            f"Existing customers upd: {old_updated:,}"
        )

        print(
            f"New customer vectors  : {new_created:,}"
        )

        print(
            f"Unresolved customers  : {unresolved:,}"
        )

        print(
            f"Total vectors         : {len(customer_vectors):,}"
        )

        all_embedding_frames.append(
            vectors_to_frame(
                customer_vectors,
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
                    "centroid_refresh",

                "vectors_available":
                    len(
                        customer_vectors
                    ),

                "interval_events":
                    len(
                        interval
                    ),

                "old_customers_updated":
                    old_updated,

                "new_customers_created":
                    new_created,

                "unresolved_interval_customers":
                    unresolved,

                "known_product_centroids":
                    len(
                        product_centroids
                    ),
            }
        )

        previous_cutoff = cutoff

    embeddings = pd.concat(
        all_embedding_frames,
        ignore_index=True,
    )

    if embeddings.duplicated(
        [
            "ID",
            "cutoff",
        ]
    ).any():

        raise AssertionError(
            "Duplicate ID-cutoff embeddings detected."
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
            "Expected exactly "
            f"{VECTOR_SIZE} embedding dimensions."
        )

    if not np.isfinite(
        embeddings[
            embedding_columns
        ].to_numpy(
            dtype="float64"
        )
    ).all():

        raise AssertionError(
            "Embedding table contains non-finite values."
        )

    audit = pd.DataFrame(
        audit_rows
    )

    coverage = historical_coverage(
        embeddings
    )

    config = {
        "research_phase":
            "06R-A2",

        "base_cutoff":
            BASE_CUTOFF.strftime(
                "%Y-%m-%d"
            ),

        "cutoffs":
            [
                cutoff.strftime(
                    "%Y-%m-%d"
                )
                for cutoff
                in CUTOFFS
            ],

        "method":
            "SGNS base plus aligned product-centroid refresh",

        "vector_size":
            VECTOR_SIZE,

        "gensim_window_radius":
            GENSIM_WINDOW_RADIUS,

        "total_context_length":
            2
            *
            GENSIM_WINDOW_RADIUS
            +
            1,

        "sg":
            1,

        "negative":
            NEGATIVE,

        "ns_exponent":
            0.75,

        "epochs":
            EPOCHS,

        "min_count":
            1,

        "sample":
            0.0,

        "workers":
            1,

        "seed":
            SEED,

        "shrink_windows":
            False,

        "base_shared_product_count":
            len(
                qualifying_products
            ),

        "base_corpus_events":
            len(
                base_corpus_events
            ),

        "base_vocabulary_customers":
            len(
                base_model.wv
            ),

        "refresh_existing_weight":
            0.5,

        "refresh_new_purchase_weight":
            0.5,
    }

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

    coverage.to_csv(
        OUTPUT_DIR
        /
        "historical_embedding_coverage.csv",
        index=False,
    )

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
    print(
        "=" * 78
    )

    print(
        "PHASE 06R-A2 EMBEDDING BUILD COMPLETE"
    )

    print(
        "=" * 78
    )

    print()
    print(
        audit.to_string(
            index=False
        )
    )

    if not coverage.empty:
        print()
        print(
            "HISTORICAL ELIGIBLE-CUSTOMER COVERAGE"
        )

        print()
        print(
            coverage.to_string(
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
        "Outputs:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
