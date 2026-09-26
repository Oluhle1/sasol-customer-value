# Phase 03 - Temporal Customer Representation and Feature Engineering

**Project:** Sasol Customer Value Recruitment Challenge  
**Phase:** 03  
**Status:** In Progress  
**Started:** 26 September 2026

---

## 1. Purpose

Phases 01 and 02 established the structure of the source transactions
and the statistical behaviour of the prediction targets.

Phase 03 converts historical transaction data into leakage-free
customer representations that can later be supplied to the selected
LightGBM modelling framework.

No predictive model is trained during Phase 03.

The objective is to construct and validate the feature matrix.

---

## 2. Fundamental Observation

The machine-learning observation is defined as:

    Customer x Forecasting Origin

A customer may therefore appear at several historical forecasting
origins provided the customer satisfies the eligibility requirements
at each origin.

Customer behaviour is allowed to change over time.

---

## 3. Historical Forecasting Origins

The five labelled forecasting origins are:

    2024-09-01
    2024-12-01
    2025-03-01
    2025-06-01
    2025-09-01

The final unlabelled competition forecasting origin is:

    2025-12-01

For every forecasting origin t:

    features use information strictly before t

and historical outcomes use:

    [t, t + 3 calendar months)

No outcome information may enter feature construction.

---

## 4. Expected Snapshot Population

Historical eligible populations established in Phase 02 are:

| Cutoff | Eligible customers |
|---|---:|
| 2024-09-01 | 3,608 |
| 2024-12-01 | 4,223 |
| 2025-03-01 | 4,649 |
| 2025-06-01 | 4,961 |
| 2025-09-01 | 5,238 |

Total historical customer snapshots:

    22,679

Final competition customers:

    5,488

Expected total customer-cutoff observations:

    28,167

---

## 5. Feature-Time Rule

Every predictive variable must satisfy:

    source transaction time < forecasting cutoff

No information occurring at or after the cutoff may be used.

The feature pipeline must therefore be cutoff-aware.

A customer feature cannot be calculated once from the complete
transaction history and reused retrospectively.

---

## 6. Historical Windows

The following calendar-history windows are fixed for Phase 03:

### m1

One calendar month immediately before the forecasting cutoff.

Purpose:

Near-term activity state.

### m3

Three calendar months immediately before the forecasting cutoff.

Purpose:

Directly aligned with the competition's three-month comparison and
forecast horizon.

### m6

Six calendar months immediately before the forecasting cutoff.

Purpose:

Medium-term customer behaviour and recent-quarter versus preceding-
quarter comparison.

### m12

Twelve calendar months immediately before the forecasting cutoff.

Purpose:

Longer-run behavioural context and seasonal coverage where available.

### lifetime

All observed customer history strictly before the forecasting cutoff.

Purpose:

Customer tenure, accumulated behavioural history and complete category
exposure.

Customers with less history than a requested window will use all
history actually available within that window.

Observation depth and tenure will later make the amount of available
history explicit to the model.

---

## 7. Planned Feature Families

Feature engineering will be performed in controlled families.

### 7.1 Customer lifecycle

Examples:

    tenure
    observation depth
    first purchase recency
    recent activity state

### 7.2 Recency

Examples:

    days since last basket
    days since last fuel purchase
    days since last non-fuel purchase

### 7.3 Frequency

Examples:

    basket count
    active days
    baskets per active month

Frequency will count distinct TransactionId values rather than
transaction lines.

### 7.4 Value

Fuel and non-fuel behaviour will be represented separately.

Examples:

    fuel litres
    non-fuel spend
    spend per basket
    litres per fuel basket

### 7.5 Multi-window behaviour

The same core behaviours will be calculated across:

    m1
    m3
    m6
    m12
    lifetime

This preserves both immediate and long-run customer state.

### 7.6 Momentum

Recent behaviour will later be compared with longer-run behaviour.

Examples include:

    recent quarter versus previous quarter
    recent month versus three-month average

The exact features will be defined before implementation.

### 7.7 Purchase regularity

Customer purchasing cadence will later be represented using
inter-basket timing.

Examples:

    mean purchase gap
    median purchase gap
    purchase-gap variability

### 7.8 Category state

Category history must distinguish:

    never purchased
    historically purchased but recently dormant
    recently purchased
    continuously active

This distinction is required by the Sasol Opportunity rules.

### 7.9 Category breadth

The representation will include the number and diversity of historically
observed qualifying categories.

This reflects customer maturity and the shrinking space of possible
new-category adoption.

---

## 8. Research Hypotheses Carried Forward

Phase 01 and Phase 02 established the following hypotheses.

### H1 - Recency and inactivity

Longer time since recent activity is expected to contain information
about future inactivity.

### H2 - Frequency and future customer value

Customers with greater recent and historical transaction intensity may
display different future fuel and non-fuel value.

### H3 - Recent versus historical behaviour

Recent behaviour should be interpreted relative to a customer's longer-
term history rather than only through absolute totals.

### H4 - Category exposure and adoption

A category that has previously been purchased cannot qualify as a new
category adoption.

Complete historical category exposure is therefore important.

### H5 - Dormancy and reactivation

A historically known category with no recent-quarter activity is
different from both a continuously active category and a never-observed
category.

### H6 - Fuel and non-fuel behaviour

Fuel and non-fuel customer histories should remain separately
represented because the corresponding future values are related but not
equivalent.

These are modelling hypotheses rather than assumed truths.

Their usefulness will later be tested through temporal validation.

---

## 9. Phase 03 Subphases

Phase 03A - Snapshot architecture

Phase 03B - Core behavioural features

Phase 03C - Multi-window behaviour, momentum and regularity

Phase 03D - Category state, maturity and reactivation

Phase 03E - Feature integrity, leakage and drift audit

---

## 10. Phase 03A Objective

Phase 03A creates the authoritative customer-cutoff index.

It does not yet create predictive features.

The output defines exactly which customer exists at which historical
forecasting origin and distinguishes labelled historical observations
from the final competition population.

Phase 03A must verify that:

- historical eligibility independently matches Phase 02 labels;
- final eligibility exactly matches test.csv;
- customer-cutoff combinations are unique;
- no ineligible customer enters a snapshot;
- the final historical transaction occurs before the competition cutoff.

---

## 11. Current Status

Phase 03A snapshot architecture is being implemented.

No Phase 03 predictive feature has yet been created.

No machine-learning model has been trained.

---

## 12. Phase 03A Results - Snapshot Architecture

The authoritative Customer x Forecasting Origin index was successfully
constructed.

Historical customer eligibility was recalculated directly from the
transaction history at every forecasting origin using the official
Sasol eligibility implementation.

Each historical population was independently reconciled with the
corresponding Phase 02 target population.

The final forecasting population was independently reconciled with
test.csv.

### Snapshot populations

| Cutoff | Snapshot role | Eligible customers | Validation |
|---|---|---:|---|
| 2024-09-01 | Historical | 3,608 | PASS |
| 2024-12-01 | Historical | 4,223 | PASS |
| 2025-03-01 | Historical | 4,649 | PASS |
| 2025-06-01 | Historical | 4,961 | PASS |
| 2025-09-01 | Historical | 5,238 | PASS |
| 2025-12-01 | Competition | 5,488 | PASS |

For every historical snapshot:

    missing_from_labels = 0
    extra_in_labels = 0

For the final competition snapshot:

    eligible IDs missing from test.csv = 0
    extra IDs in test.csv = 0

### Observation counts

Historical labelled observations:

    22,679

Final competition observations:

    5,488

Combined snapshot observations:

    28,167

The resulting supervised-learning architecture is therefore:

    22,679 labelled Customer x Forecasting Origin observations

followed by:

    5,488 unlabelled Customer x Final Forecasting Origin observations

---

## 13. Final Historical Boundary Verification

The latest transaction timestamp in the released transaction history is:

    2025-11-30 23:39:14.567

The final competition forecasting cutoff is:

    2025-12-01 00:00:00

Therefore the complete supplied history occurs strictly before the final
forecasting origin.

This satisfies the project's core temporal constraint:

    feature source time < forecasting cutoff

---

## 14. Snapshot Metadata

Each snapshot contains the following structural metadata:

    ID
    cutoff
    history_end_exclusive
    outcome_start
    outcome_end_exclusive
    snapshot_role
    is_labeled
    snapshot_id

These variables define the experiment rather than customer behaviour.

In particular:

    outcome_start
    outcome_end_exclusive
    snapshot_role
    is_labeled

must not be supplied to the predictive models as customer features.

The cutoff variable will primarily control feature construction and
temporal validation.

Any eventual use of calendar information derived from the cutoff must
be separately justified and documented rather than entering the model
accidentally through metadata.

---

## 15. Phase 03A Integrity Decision

Phase 03A is considered complete because:

- historical eligibility exactly reproduces Phase 02 populations;
- final eligibility exactly reproduces test.csv;
- every Customer x Forecasting Origin combination is unique;
- no post-cutoff transaction exists in the released history for the
  final competition snapshot;
- labelled and unlabelled populations are clearly distinguished;
- and no predictive feature has yet been constructed.

The snapshot index is now the authoritative row structure for all
subsequent feature engineering.

---

## 16. Phase 03A Conclusion

The project now has a validated temporal modelling spine.

Rather than calculating one static representation per customer,
features will be generated independently for each:

    Customer x Forecasting Origin

observation.

This preserves changing customer state through time and prevents later
transactions from contaminating earlier historical representations.

Phase 03A is complete.

The next stage is:

    Phase 03B - Core Behavioural Features

Phase 03B will construct the first predictive variables covering:

- customer lifecycle;
- recency;
- transaction frequency;
- active days;
- fuel activity;
- non-fuel activity;
- and customer value.

No momentum or category-state features will be introduced until those
core behavioural variables have been validated.


---

## 17. Phase 03B Results - Core Behavioural Features

Phase 03B successfully created the first predictive customer
representation.

The resulting feature matrix contains:

    28,167 Customer x Forecasting Origin rows

and:

    57 predictive variables

for every observation.

The row population exactly matches the authoritative Phase 03A
snapshot index.

---

## 18. Core Feature Integrity Results

All six forecasting origins passed feature-generation integrity checks.

For every forecasting origin:

- snapshot row counts were preserved;
- predictive feature count remained 57;
- duplicate snapshot IDs were zero;
- infinite numeric values were zero;
- every eligible customer had a valid last historical basket;
- every eligible customer retained at least three lifetime baskets;
- historical windows were monotonic with respect to basket counts;
- and all features were constructed exclusively from transactions
  strictly before the forecasting cutoff.

The resulting matrix is therefore considered structurally valid.

---

## 19. Historical Non-Fuel Participation

The proportion of eligible customers who had never made a qualifying
historical non-fuel purchase declined substantially through time.

| Cutoff | Never non-fuel | Share |
|---|---:|---:|
| 2024-09-01 | 1,806 | 50.06% |
| 2024-12-01 | 1,726 | 40.87% |
| 2025-03-01 | 1,675 | 36.03% |
| 2025-06-01 | 1,593 | 32.11% |
| 2025-09-01 | 1,528 | 29.17% |
| 2025-12-01 | 1,419 | 25.86% |

This provides predictor-side evidence of customer maturation.

As more transaction history becomes available, increasing numbers of
customers accumulate non-fuel purchase experience.

This observation is consistent with the Phase 02 finding that
new-category adoption becomes less frequent as customer histories
mature.

---

## 20. Historical Fuel Participation

Customers with no qualifying historical fuel purchase remain uncommon.

Observed values were:

| Cutoff | Never fuel | Share |
|---|---:|---:|
| 2024-09-01 | 23 | 0.64% |
| 2024-12-01 | 39 | 0.92% |
| 2025-03-01 | 48 | 1.03% |
| 2025-06-01 | 63 | 1.27% |
| 2025-09-01 | 73 | 1.39% |
| 2025-12-01 | 83 | 1.51% |

The increasing absolute count does not imply that existing customers
are losing their historical fuel status.

The eligible population itself expands through time, and later entrants
can become eligible through non-fuel transaction histories without ever
recording qualifying fuel activity.

This supports the decision to preserve fuel and non-fuel histories as
separate feature families.

---

## 21. Customers Without Qualifying Historical Activity

One eligible customer at each of the first four forecasting origins
had no qualifying positive purchase despite satisfying the minimum
historical basket requirement.

Observed missing qualifying recency values were:

    2024-09-01     1
    2024-12-01     1
    2025-03-01     1
    2025-06-01     1
    2025-09-01     0
    2025-12-01     0

This is not treated as missing-data corruption.

Eligibility counts all historical baskets, while qualifying purchase
activity follows the narrower official Sasol in-scope positive purchase
definition.

A customer may therefore be historically eligible without yet having
a qualifying purchase.

The missing recency value is retained as a meaningful customer state.

---

## 22. Recent Net Value Checks

No customer had a negative one-month net fuel or non-fuel value at any
forecasting origin.

Observed counts were:

    negative m1 fuel net values     = 0

    negative m1 non-fuel net values = 0

This result does not justify clipping negative values generally.

The raw transaction history contains legitimate signed returns and the
feature pipeline continues to preserve those values.

Longer historical windows may still contain negative line-level effects.

---

## 23. Core Representation Available After Phase 03B

Each customer snapshot now contains information describing:

### Observed lifecycle

    observed_tenure_days

### Recency

    days_since_last_basket
    days_since_last_qualifying_purchase
    days_since_last_fuel_purchase
    days_since_last_nonfuel_purchase

### Historical participation

    has_ever_fuel
    has_ever_nonfuel

### Frequency

For each of:

    m1
    m3
    m6
    m12
    lifetime

the representation contains:

    basket_count
    active_days
    qualifying_basket_count
    qualifying_active_days

### Fuel behaviour

For each historical window:

    fuel_basket_count
    fuel_active_days
    fuel_litres

### Non-fuel behaviour

For each historical window:

    nonfuel_basket_count
    nonfuel_active_days
    nonfuel_spend

The Phase 03B representation therefore captures current activity level
and historical scale, but it does not yet explicitly represent the
direction or regularity of behaviour.

---

## 24. Phase 03B Limitation

Absolute historical totals alone cannot distinguish customers with
different trajectories.

For example, two customers may have identical recent three-month fuel
volume while one is increasing and the other is declining relative to
their previous behaviour.

Similarly, a customer with no recent purchases may represent:

- long-term inactivity;
- temporary dormancy;
- or an historically active customer approaching reactivation.

Phase 03B therefore provides the behavioural levels required for a
baseline representation, but additional temporal structure is still
required.

---

## 25. Phase 03B Decision

Phase 03B is considered complete because:

- 57 documented predictive features were generated;
- all 28,167 authoritative snapshot rows were preserved;
- no duplicates or infinite values were introduced;
- historical windows respect the forecasting cutoff;
- basket and qualifying activity remain conceptually separated;
- fuel and non-fuel behaviour remain separately represented;
- legitimate missing recency states are preserved;
- and all core feature families passed integrity checks.

No machine-learning model has yet been trained.

The project may now proceed to:

    Phase 03C - Momentum, Previous-Period Comparison and Purchase Regularity

