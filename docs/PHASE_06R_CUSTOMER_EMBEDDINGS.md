
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


---

# Phase 06R-A2b — Warm-Start Incremental SGNS

Status:

    REPRESENTATION STABILITY: PASS
    FINAL TEST COVERAGE: PASS

Purpose:

Replace the failed recursive centroid update with a longitudinal
warm-start SGNS procedure that preserves the previous embedding
coordinate system while continuing representation learning.

This follows the longitudinal principle described by Chamberlain
et al.: existing customer vectors remain in the learned space and
later customer-product contexts update that space rather than
independently relearning unrelated coordinates.

Product-level purchase histories follow the retail adaptation used
by Yan & Resnick.

The Gensim implementation is an adaptation rather than an exact
reproduction of Chamberlain's ordered old/new pair training schedule.

## Build results

Base:

    cutoff             2024-09-01
    customer vectors        4,374
    product documents       1,467

Subsequent vocabulary sizes:

    2024-12-01    4,950
    2025-03-01    5,377
    2025-06-01    5,676
    2025-09-01    5,927
    2025-12-01    6,196

Existing-customer share among active interval customers:

    Sep-Dec 2024    86.18%
    Dec-Mar 2025    89.90%
    Mar-Jun 2025    92.75%
    Jun-Sep 2025    93.86%
    Sep-Dec 2025    93.62%

This provides substantial overlap for longitudinal representation
learning.

## Representation-health results

Centered matrix rank:

    64 / 64 dimensions at every cutoff

Unique vector share:

    100% at every cutoff

Median dimension standard deviation:

    2024-09-01    0.083336
    2024-12-01    0.122951
    2025-03-01    0.140691
    2025-06-01    0.156365
    2025-09-01    0.165545
    2025-12-01    0.175526

Unlike the rejected centroid-refresh representation, dispersion
increased rather than collapsed.

Mean random-pair cosine similarity:

    2024-09-01    0.683295
    2024-12-01    0.678574
    2025-03-01    0.643759
    2025-06-01    0.624399
    2025-09-01    0.604222
    2025-12-01    0.590582

Median random-pair cosine similarity:

    2024-09-01    0.953341
    2025-12-01    0.652382

The representation therefore becomes more discriminative between
customers rather than converging toward a common direction.

## Temporal continuity

Median cosine similarity for customers whose vectors changed:

    Sep-Dec 2024    0.927303
    Dec-Mar 2025    0.952067
    Mar-Jun 2025    0.960189
    Jun-Sep 2025    0.960016
    Sep-Dec 2025    0.958838

This indicates substantial continuity of customer position while
still allowing meaningful evolution.

## Final competition coverage

    test customers       5,488
    embedded             5,488
    missing                  0
    coverage              100%

Eight of the 6,204 raw-history customers do not have a final SGNS
vector. None are present in the final competition test population.

Their relationship to the historical supervised cohorts is audited
separately before predictive modelling.

## Decision

Warm-start SGNS representation:

    PASS

The failed recursive centroid representation remains rejected.

The warm-start representation is the candidate representation for
Phase 06R-A3.


---

# Phase 06R-A2b — Warm-Start Incremental SGNS

Status:

    REPRESENTATION STABILITY: PASS
    HISTORICAL COVERAGE: PASS
    FINAL TEST COVERAGE: PASS
    CANDIDATE FOR PREDICTIVE ABLATION: YES

## Motivation

Phase A2 demonstrated that repeated product-centroid averaging
progressively homogenised customer vectors.

A2b therefore replaced recursive centroid averaging with incremental
warm-start SGNS in a persistent coordinate system.

Existing customers retain their previously learned coordinates while
new interaction contexts continue training the representation.

This follows the longitudinal principle described by Chamberlain et al.
while using product-level purchase histories in the retail setting
described by Yan & Resnick.

The Gensim implementation is an adaptation rather than an exact
reproduction of Chamberlain's ordered old/new skip-gram pair schedule.

## Build results

Basket-product purchase events:

    437,372

Customer vocabulary:

    2024-09-01    4,374
    2024-12-01    4,950
    2025-03-01    5,377
    2025-06-01    5,676
    2025-09-01    5,927
    2025-12-01    6,196

Share of active interval customers already present in the embedding
space:

    Sep-Dec 2024    86.18%
    Dec-Mar 2025    89.90%
    Mar-Jun 2025    92.75%
    Jun-Sep 2025    93.86%
    Sep-Dec 2025    93.62%

## Representation health

Centered matrix rank:

    64 / 64 at every cutoff

Unique vectors:

    100% at every cutoff

Median per-dimension standard deviation:

    2024-09-01    0.083336
    2024-12-01    0.122951
    2025-03-01    0.140691
    2025-06-01    0.156365
    2025-09-01    0.165545
    2025-12-01    0.175526

Mean random-pair cosine similarity:

    2024-09-01    0.683295
    2024-12-01    0.678574
    2025-03-01    0.643759
    2025-06-01    0.624399
    2025-09-01    0.604222
    2025-12-01    0.590582

Median random-pair cosine similarity:

    2024-09-01    0.953341
    2025-12-01    0.652382

Unlike the rejected centroid method, customer vectors become more
discriminative rather than converging toward a common direction.

## Temporal continuity

Median cosine similarity among customers whose vectors changed:

    Sep-Dec 2024    0.927303
    Dec-Mar 2025    0.952067
    Mar-Jun 2025    0.960189
    Jun-Sep 2025    0.960016
    Sep-Dec 2025    0.958838

The coordinate system therefore preserves substantial continuity while
still allowing customer representations to evolve.

## Coverage

Historical supervised cohorts:

    2024-09-01    3,608 / 3,608    100%
    2024-12-01    4,223 / 4,223    100%
    2025-03-01    4,649 / 4,649    100%
    2025-06-01    4,961 / 4,961    100%
    2025-09-01    5,238 / 5,238    100%

Final competition cohort:

    5,488 / 5,488    100%

## Eight raw customers without embeddings

Eight of the 6,204 raw-history customer IDs never entered the SGNS
vocabulary.

Each has only one distinct transaction basket in the entire available
history.

Consequently:

- none satisfies the minimum three-basket eligibility requirement;
- none appears in any historical supervised modelling cohort;
- none appears in the final competition test cohort.

These eight customers therefore create no missing-value problem for
either temporal validation or competition inference.

## Decision

Warm-start customer representation:

    PASS

Proceed to:

    Phase 06R-A3 — Frozen LightGBM embedding feature ablation

No embedding hyperparameter tuning is permitted before A3.

