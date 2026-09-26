# Phase 03C - Momentum, Previous-Period Comparison and Purchase Regularity

**Project:** Sasol Customer Value Recruitment Challenge
**Phase:** 03C
**Status:** In Progress
**Started:** 26 September 2026

---

## 1. Purpose

Phase 03B created a validated representation of customer behavioural
levels across multiple historical windows.

Phase 03C extends that representation to describe behavioural direction,
state changes and purchase cadence.

No machine-learning model is trained during this phase.

---

## 2. Research Motivation

Absolute customer totals cannot distinguish between different temporal
trajectories.

For example, two customers may have the same recent three-month fuel
volume even though one is increasing and the other is declining.

Phase 03C therefore introduces features describing:

- the immediately preceding three-month period;
- recent versus previous-quarter change;
- short-term acceleration;
- dormancy and reactivation;
- and historical purchase regularity.

---

## 3. Previous-Quarter Definition

The existing Phase 03B windows are:

    m3 = [cutoff - 3 months, cutoff)

and:

    m6 = [cutoff - 6 months, cutoff)

Therefore:

    prev3 = m6 - m3

represents:

    [cutoff - 6 months, cutoff - 3 months)

for additive behavioural measures.

---

## 4. Quarter-on-Quarter Momentum

For additive behavioural metric x:

    QoQ Delta = x_m3 - x_prev3

Positive values indicate increased recent activity.

Negative values indicate decreased recent activity.

Zero indicates unchanged aggregate activity.

Relative changes are calculated for fuel litres and non-fuel spend only
when the previous-quarter value is positive.

---

## 5. Short-Term Acceleration

The most recent calendar month is compared with the average of the
preceding two calendar months:

    recent month deviation
        =
    m1 - ((m3 - m1) / 2)

This describes whether activity in the latest month is above or below
the customer's immediately preceding two-month rate.

---

## 6. Historical Reactivation and Dormancy

Reactivation is defined using historical information only.

For a behavioural stream:

    previous quarter activity = 0

and:

    recent quarter activity > 0

indicates historical reactivation.

Conversely:

    previous quarter activity > 0

and:

    recent quarter activity = 0

indicates newly dormant behaviour.

These states are calculated separately for:

- overall qualifying purchase activity;
- fuel;
- non-fuel.

They must not be confused with future Opportunity labels.

---

## 7. Purchase Regularity

Distinct baskets are ordered chronologically within customer.

Inter-purchase gaps are calculated as:

    gap_i = basket_time_i - basket_time_(i-1)

Gap statistics are produced for:

- lifetime general basket activity;
- recent six-month general basket activity;
- lifetime fuel activity;
- recent six-month fuel activity;
- lifetime non-fuel activity;
- recent six-month non-fuel activity.

Where insufficient historical baskets exist to calculate a gap, the
feature remains missing.

This is meaningful missingness rather than a data error.

---

## 8. Dormancy Relative to Customer Cadence

Current recency is compared with the customer's historical mean
inter-purchase gap.

For example:

    basket_recency_to_lifetime_gap
        =
    days_since_last_basket
        /
    lifetime_gap_mean_days

This allows identical absolute recency to have different meaning for
high-frequency and low-frequency customers.

---

## 9. Expected Feature Expansion

Phase 03B contains:

    57 predictive features

Phase 03C adds:

    47 predictive features

Expected combined representation:

    104 predictive features

No category-specific features are introduced during Phase 03C.

Those remain reserved for Phase 03D.

---

## 10. Earliest-Snapshot Limitation

The released transaction history begins on 19 March 2024.

Consequently, the previous-quarter portion of the earliest 2024-09-01
snapshot is partially left-truncated.

This limitation is documented rather than artificially filled.

Later temporal validation will account for the fact that early
forecasting origins contain less available history than later origins.

---

## 11. Current Status

Phase 03C feature-generation code is being implemented.

No Phase 03C result has yet been accepted.

No machine-learning model has been trained.

---

## 12. Phase 03C Results

Phase 03C successfully expanded the temporal customer representation
from:

    57 predictive features

to:

    104 predictive features

while preserving all:

    28,167 Customer x Forecasting Origin observations.

All six forecasting origins passed the Phase 03C integrity checks.

Observed integrity results included:

- zero duplicate snapshot IDs;
- zero infinite numeric values;
- exact quarter reconstruction;
- no negative inter-purchase gaps;
- no conflicting reactivation/dormancy states;
- and unchanged snapshot populations.

---

## 13. Historical Reactivation and Dormancy

Reactivation and dormancy were found to be common customer states.

For overall qualifying purchase activity:

| Cutoff | Reactivated | Became dormant |
|---|---:|---:|
| 2024-09-01 | 349 | 199 |
| 2024-12-01 | 459 | 291 |
| 2025-03-01 | 402 | 415 |
| 2025-06-01 | 369 | 512 |
| 2025-09-01 | 422 | 479 |
| 2025-12-01 | 506 | 463 |

Reactivation therefore affects approximately 7-11% of eligible
customers depending on forecasting origin.

New dormancy affects approximately 6-10%.

These states are sufficiently common that treating them explicitly is
justified.

---

## 14. Fuel Reactivation and Dormancy

Fuel behaviour displayed the following transitions:

| Cutoff | Reactivated fuel | Became dormant fuel |
|---|---:|---:|
| 2024-09-01 | 360 | 208 |
| 2024-12-01 | 458 | 302 |
| 2025-03-01 | 405 | 429 |
| 2025-06-01 | 368 | 512 |
| 2025-09-01 | 428 | 471 |
| 2025-12-01 | 488 | 470 |

Fuel reactivation and dormancy occur regularly, despite fuel being the
dominant transaction stream.

This reinforces the need to distinguish:

    continuously active fuel customers

from:

    historically active but recently dormant customers.

---

## 15. Non-Fuel Reactivation and Dormancy

Non-fuel behaviour is more intermittent than fuel behaviour.

Observed transitions were:

| Cutoff | Reactivated non-fuel | Became dormant non-fuel |
|---|---:|---:|
| 2024-09-01 | 601 | 471 |
| 2024-12-01 | 810 | 420 |
| 2025-03-01 | 656 | 691 |
| 2025-06-01 | 683 | 706 |
| 2025-09-01 | 682 | 623 |
| 2025-12-01 | 751 | 632 |

At the final competition cutoff approximately:

    13.7%

of customers had reactivated non-fuel activity relative to the
preceding quarter.

This is substantially higher than the corresponding fuel reactivation
rate.

Non-fuel behaviour therefore appears more episodic and should not be
treated as a scaled version of fuel behaviour.

---

## 16. General Purchase Regularity

All eligible customer snapshots had sufficient historical basket
activity to calculate lifetime general inter-purchase gaps.

Observed:

    missing_lifetime_gap = 0

at every forecasting origin.

This is consistent with the competition eligibility requirement of at
least three historical baskets.

The general lifetime cadence features are therefore available for the
entire modelling population.

---

## 17. Fuel Purchase Regularity

Missing lifetime fuel-gap features remained uncommon.

| Cutoff | Missing fuel lifetime gap |
|---|---:|
| 2024-09-01 | 42 |
| 2024-12-01 | 62 |
| 2025-03-01 | 65 |
| 2025-06-01 | 82 |
| 2025-09-01 | 98 |
| 2025-12-01 | 105 |

At the final cutoff this represents approximately 1.9% of customers.

Fuel purchase cadence can therefore be estimated for the vast majority
of the eligible population.

---

## 18. Non-Fuel Purchase Regularity

Non-fuel cadence is available for a substantially smaller proportion of
customers.

| Cutoff | Missing non-fuel lifetime gap |
|---|---:|
| 2024-09-01 | 2,493 |
| 2024-12-01 | 2,563 |
| 2025-03-01 | 2,583 |
| 2025-06-01 | 2,511 |
| 2025-09-01 | 2,480 |
| 2025-12-01 | 2,429 |

At the final competition forecasting origin:

    2,429 / 5,488

or approximately:

    44.3%

of customers do not have sufficient repeated non-fuel activity to
calculate a lifetime non-fuel inter-purchase gap.

This is treated as meaningful behavioural missingness.

It indicates that many eligible customers have either never made a
qualifying non-fuel purchase or have insufficient repeated non-fuel
history to establish a purchasing cadence.

No arbitrary imputation is performed during feature construction.

---

## 19. Behavioural Interpretation of Recency Relative to Cadence

Absolute recency alone can be misleading because customers operate at
different natural frequencies.

The Phase 03C representation therefore includes:

    basket_recency_to_lifetime_gap

    fuel_recency_to_lifetime_gap

    nonfuel_recency_to_lifetime_gap

These features express current recency relative to the customer's own
historical purchasing cadence.

A customer whose current recency greatly exceeds the usual purchase gap
may represent a stronger dormancy signal than another customer with the
same absolute recency but a naturally infrequent purchase pattern.

---

## 20. Phase 03C Representation

Phase 03C adds four important behavioural dimensions to the Phase 03B
representation.

### Previous-quarter state

The immediately preceding three-month period is represented separately
from the recent three-month period.

### Direction

Quarter-on-quarter changes distinguish increasing, stable and declining
activity.

### State transition

Reactivation and newly dormant states distinguish changing customer
activity from static customer history.

### Cadence

Inter-purchase gaps describe how regularly customers historically
transact.

The customer representation can now describe not only behavioural level
but also behavioural direction and timing.

---

## 21. Phase 03C Decision

Phase 03C is considered complete because:

- all 28,167 snapshot rows were preserved;
- the feature count increased from 57 to 104 as designed;
- previous-quarter reconstruction passed for all additive measures;
- reactivation and dormancy indicators were internally consistent;
- no negative temporal gaps were introduced;
- no infinite feature values were created;
- meaningful missing cadence states were preserved;
- and all six forecasting origins passed validation.

No predictive model has yet been trained.

The project may now proceed to:

    Phase 03D - Category State, Category Maturity and Reactivation

