from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "train.csv"

SUPERVISED_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
    / "supervised_historical.csv"
)

EMBEDDING_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "customer_embeddings_warmstart"
    / "customer_embeddings.parquet"
)

MISSING_FINAL = {
    "4001000000378",
    "4001000001532",
    "4001000002968",
    "4001000005276",
    "4001000005301",
    "4001000005302",
    "4001000005597",
    "4001000005889",
}

ADJUSTMENT_ITEMS = {
    "TOTAL/ROUNDING DISCO",
    "PENSIONER DISCOUNT",
    "WOW_PROMO",
}


def main():

    raw = pd.read_csv(
        RAW_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

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

    embeddings = pd.read_parquet(
        EMBEDDING_PATH
    )

    embeddings["ID"] = (
        embeddings["ID"]
        .astype(str)
    )

    embeddings["cutoff"] = (
        embeddings["cutoff"]
        .astype(str)
    )

    print()
    print("=" * 78)
    print("HISTORICAL SUPERVISED EMBEDDING COVERAGE")
    print("=" * 78)
    print()

    coverage_rows = []

    for cutoff in sorted(
        supervised["cutoff"].unique()
    ):

        eligible_ids = set(
            supervised.loc[
                supervised["cutoff"].eq(cutoff),
                "ID",
            ].astype(str)
        )

        embedded_ids = set(
            embeddings.loc[
                embeddings["cutoff"].eq(cutoff),
                "ID",
            ].astype(str)
        )

        missing = sorted(
            eligible_ids - embedded_ids
        )

        coverage_rows.append(
            {
                "cutoff": cutoff,
                "eligible": len(eligible_ids),
                "covered": len(
                    eligible_ids & embedded_ids
                ),
                "missing": len(missing),
                "coverage": (
                    len(
                        eligible_ids & embedded_ids
                    )
                    /
                    len(eligible_ids)
                ),
            }
        )

        print(
            f"{cutoff} | "
            f"eligible={len(eligible_ids):,} | "
            f"covered={len(eligible_ids & embedded_ids):,} | "
            f"missing={len(missing):,}"
        )

        if missing:
            print(
                "  missing IDs:",
                missing,
            )

    coverage = pd.DataFrame(
        coverage_rows
    )

    print()
    print("=" * 78)
    print("EIGHT MISSING RAW CUSTOMERS")
    print("=" * 78)
    print()

    subset = raw[
        raw["ID"].isin(
            MISSING_FINAL
        )
    ].copy()

    subset["_timestamp"] = pd.to_datetime(
        subset["DateTimeZA"],
        format="mixed",
        errors="raise",
    )

    subset["_quantity"] = pd.to_numeric(
        subset["Quantity"],
        errors="raise",
    )

    subset["_amount"] = pd.to_numeric(
        subset["TotalExclAmount"],
        errors="raise",
    )

    subset["_is_adjustment"] = (
        subset["ItemName"]
        .isin(
            ADJUSTMENT_ITEMS
        )
    )

    subset["_qualifying_embedding_purchase"] = (
        subset["_quantity"].gt(0)
        &
        subset["_amount"].gt(0)
        &
        ~subset["_is_adjustment"]
        &
        subset["TransactionId"].ne("")
        &
        subset["StockItemId"].ne("")
    )

    customer_summary = (
        subset
        .groupby(
            "ID",
            as_index=False,
        )
        .agg(
            rows=(
                "ID",
                "size",
            ),
            baskets=(
                "TransactionId",
                "nunique",
            ),
            products=(
                "StockItemId",
                "nunique",
            ),
            first_date=(
                "_timestamp",
                "min",
            ),
            last_date=(
                "_timestamp",
                "max",
            ),
            positive_quantity_lines=(
                "_quantity",
                lambda s:
                    int(
                        s.gt(0).sum()
                    ),
            ),
            positive_amount_lines=(
                "_amount",
                lambda s:
                    int(
                        s.gt(0).sum()
                    ),
            ),
            qualifying_embedding_lines=(
                "_qualifying_embedding_purchase",
                "sum",
            ),
        )
    )

    print(
        customer_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("PRODUCTS USED BY MISSING CUSTOMERS")
    print("=" * 78)
    print()

    product_pairs = (
        subset[
            [
                "ID",
                "StockItemId",
                "ItemName",
                "ItemCategoryLevel2",
                "_qualifying_embedding_purchase",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "ID",
                "StockItemId",
            ]
        )
    )

    print(
        product_pairs.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("ARE THE EIGHT EVER IN HISTORICAL SUPERVISED DATA?")
    print("=" * 78)
    print()

    historical_presence = (
        supervised[
            supervised["ID"].isin(
                MISSING_FINAL
            )
        ][
            [
                "ID",
                "cutoff",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "ID",
                "cutoff",
            ]
        )
    )

    if historical_presence.empty:
        print(
            "None of the eight customers occur in the historical "
            "supervised modelling cohorts."
        )
    else:
        print(
            historical_presence.to_string(
                index=False
            )
        )

    output_dir = (
        ROOT
        / "data"
        / "processed"
        / "phase06"
        / "customer_embeddings_warmstart"
    )

    coverage.to_csv(
        output_dir
        / "historical_supervised_coverage.csv",
        index=False,
    )

    customer_summary.to_csv(
        output_dir
        / "missing_raw_customer_summary.csv",
        index=False,
    )

    product_pairs.to_csv(
        output_dir
        / "missing_raw_customer_products.csv",
        index=False,
    )

    historical_presence.to_csv(
        output_dir
        / "missing_raw_customer_historical_presence.csv",
        index=False,
    )

    print()
    print("Missing-customer audit complete.")


if __name__ == "__main__":
    main()
