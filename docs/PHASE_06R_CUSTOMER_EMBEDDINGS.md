
# Phase 06R-A — Customer Purchase-History Embeddings

## Research basis

Primary references:

1. Chamberlain et al. (2017)
   Customer Lifetime Value Prediction Using Embeddings.
   KDD 2017.

2. Yan & Resnick (2024)
   A high-performance turnkey system for customer lifetime value
   prediction in retail brands.
   Quantitative Marketing and Economics.

The research hypothesis is that customer representations learned from
product-level behavioural contexts may contain predictive information
not captured by the existing handcrafted behavioural features.

---

# Phase 06R-A1 — Feasibility Audit

Status: PASS

The Sasol transaction history contains sufficient cross-customer
product context to support a customer embedding experiment.

At the final history cutoff:

    customers                  6,204
    products                   6,848
    basket-product events    437,407

Products purchased by >=2 distinct customers covered approximately
98.9% of events and all final competition customers.

Approximately 96% of adjacent product-history contexts connected
different customers.

The product distribution is highly concentrated because fuel
dominates the transaction history.

Top product:

    ULP95

Fuel accounted for approximately:

    76.88% of basket-product events

This was retained because it represents genuine Sasol behaviour and
is directly relevant to the prediction targets.

---

# Phase 06R-A2 — SGNS + Recursive Product-Centroid Refresh

Status:

    COVERAGE: PASS
    REPRESENTATION STABILITY: FAIL
    MODEL USE: REJECT

Method:

1. Train a 64-dimensional SGNS customer embedding using transactions
   strictly before 2024-09-01.
2. Products define customer sequences ordered by purchase time.
3. Preserve the original embedding coordinate system.
4. At each later quarter:
       - derive product centroids from customer embeddings;
       - update existing customers using a 50/50 combination of
         previous customer vector and new product representation;
       - assign new customers from purchased-product embeddings.

The repeated quarterly 50/50 refresh was a Sasol implementation
choice and was not claimed as an exact reproduction of the source
papers.

## Coverage

Historical eligible-customer coverage:

    2024-09-01    3608 / 3608    100%
    2024-12-01    4223 / 4223    100%
    2025-03-01    4649 / 4649    100%
    2025-06-01    4961 / 4961    100%
    2025-09-01    5238 / 5238    100%

Final test:

    5488 / 5488    100%

Only one interval customer was unresolved during the entire recursive
refresh process.

## Representation-health audit

The representation remained full rank at all six snapshots:

    centered matrix rank = 64

However, substantial geometric contraction occurred.

Median dimension standard deviation:

    2024-09-01    0.083336
    2025-12-01    0.024877

Approximately 70% reduction.

Mean random-pair cosine similarity:

    2024-09-01    0.683295
    2025-12-01    0.968956

Median random-pair cosine similarity:

    2024-09-01    0.953341
    2025-12-01    0.999350

Median quarter-to-quarter vector displacement:

    Sep->Dec 2024    0.157578
    Sep->Dec 2025    0.012938

Approximately 92% reduction.

This indicates progressive representation homogenisation.

Customers remain numerically distinct, but their embedding directions
become increasingly similar.

## Interpretation

The recursive centroid averaging procedure gradually contracts the
customer representation toward shared product centroids.

The failure is therefore representational rather than a coverage
failure.

The resulting A2 embeddings will NOT be supplied to the predictive
models.

## Event-definition reconciliation

A2 contains:

    437,372 basket-product events

Adding the positive-amount condition to the positive-quantity,
non-adjustment definition removed 28 basket-product events:

    437,400 -> 437,372

The remaining seven-event difference between the original A1
feasibility count (437,407) and the reconciled non-adjustment count
reflects a minor difference in the A1 diagnostic filtering logic.

A2 uses positive Quantity and positive TotalExclAmount together with
the documented adjustment exclusions, consistent with the project's
qualifying-purchase convention.

## Decision

Recursive product-centroid refresh:

    REJECT

Customer embedding research hypothesis:

    REMAINS OPEN

Next experiment:

    Phase 06R-A2b
    Warm-start / incremental SGNS in a persistent coordinate system.

