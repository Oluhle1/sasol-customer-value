# Phase 03D - Category State, Category Maturity and Reactivation

**Project:** Sasol Customer Value Recruitment Challenge
**Phase:** 03D
**Status:** In Progress
**Started:** 26 September 2026

---

## 1. Purpose

Phase 03D extends the temporal customer representation with original
Level-2 category history.

This phase is directly motivated by the official Sasol Opportunity
label-generation rules.

No machine-learning model is trained during Phase 03D.

---

## 2. Why Original Categories Must Be Preserved

Opportunity candidates compete using the original source Level-2
categories before the final Opportunity category mapping is applied.

Several original source categories may ultimately map to the same
output label such as "Other", but they must not be pooled before the
winning candidate is determined.

Therefore Phase 03D preserves each included original Level-2 category
separately.

The final mapped Opportunity labels are not used as feature-engineering
categories.

---

## 3. Included Category Population

The official configuration contains 27 Level-2 categories.

The following three categories are excluded from qualifying Opportunity
activity:

    Carrier Bags
    Print media
    Value added services

The feature representation therefore uses:

    24 included original categories

The literal categories:

    NA
    Unknown

remain valid categories and are preserved.

---

## 4. Category Purchase Definition

Historical category purchase indicators and basket counts require:

    Quantity > 0

and:

    TotalExclAmount > 0

and the category must be included under the official configuration.

Named adjustment items are excluded.

Basket counts use distinct TransactionId values rather than transaction
rows.

---

## 5. Category Net Spend

Category spend features use signed source:

    TotalExclAmount

Returns therefore reduce historical net category spend.

No clipping is applied.

For the Fuel category, category-level Opportunity features use source
monetary TotalExclAmount rather than fuel litres because Opportunity
candidates compete using monetary increments.

The existing CLV-oriented fuel features continue to use litres.

---

## 6. Per-Category Features

For each of the 24 included original categories, Phase 03D creates:

    ever_purchased
    seen_before_m3
    m3_basket_count
    prev3_basket_count
    m3_net_spend
    prev3_net_spend
    days_since_last_purchase

This produces:

    24 x 7 = 168

category-specific predictive features.

---

## 7. Category State Interpretation

The representation supports several customer-category states.

### Never observed

    ever_purchased = 0

The category remains theoretically eligible for future new-category
adoption.

### Historically known but recently dormant

    ever_purchased = 1
    m3_basket_count = 0

### Recently active

    m3_basket_count > 0

### Recently new to observed history

    seen_before_m3 = 0
    m3_basket_count > 0

### Recently reactivated

    seen_before_m3 = 1
    prev3_basket_count = 0
    m3_basket_count > 0

### Recently became dormant

    prev3_basket_count > 0
    m3_basket_count = 0

These are historical states only.

They are not future Opportunity labels.

---

## 8. Category Maturity Features

Phase 03D creates nine customer-level summary variables:

    lifetime_category_breadth
    pre_m3_category_breadth
    m3_category_breadth
    prev3_category_breadth
    unseen_category_count
    dormant_known_category_count
    recent_new_to_history_category_count
    recent_reactivated_category_count
    recent_became_dormant_category_count

These variables describe how mature and diversified the customer's
historical relationship with Sasol categories has become.

---

## 9. Adoption Opportunity Space

The feature:

    unseen_category_count

is calculated as:

    included categories - lifetime category breadth

Because future adoption requires a category to have no previous
qualifying purchase, this feature describes the number of original
included categories that remain historically unseen.

It is not itself a future target and uses only pre-cutoff information.

---

## 10. Expected Feature Count

Phase 03C representation:

    104 features

Category-specific features:

    168 features

Category maturity summaries:

    9 features

Expected Phase 03D total:

    281 predictive features

---

## 11. Integrity Requirements

Phase 03D must verify that:

- all 28,167 snapshot rows are preserved;
- exactly 24 included original categories are represented;
- category feature names are unique;
- no negative category basket counts exist;
- recent category purchases imply ever_purchased = 1;
- seen_before_m3 implies ever_purchased = 1;
- category recency is missing exactly when no historical qualifying
  purchase exists;
- lifetime category breadth equals the sum of ever-purchased category
  indicators;
- unseen category count plus lifetime breadth equals 24;
- no infinite feature values are produced;
- and the final predictive feature count equals 281.

---

## 12. Current Status

Phase 03D feature generation is being implemented.

No Phase 03D result has yet been accepted.

No machine-learning model has been trained.

---

## 13. Phase 03D Results

Phase 03D successfully expanded the customer representation from:

    104 predictive features

to:

    281 predictive features

while preserving:

    28,167 Customer x Forecasting Origin observations.

The feature matrix contains:

    24 included original Level-2 categories

with seven features per category and nine category-maturity summary
features.

Every forecasting origin passed the Phase 03D integrity checks.

Observed results included:

- zero duplicate snapshot IDs;
- zero infinite values;
- correct category-recency missingness;
- exact reconciliation between category breadth and per-category
  historical exposure;
- exact reconciliation between unseen categories and total included
  category count;
- and preservation of the complete snapshot population.

---

## 14. Growth of Observed Category Breadth

Customer category breadth increases systematically through time.

| Cutoff | Mean observed breadth | Median | Mean unseen categories |
|---|---:|---:|---:|
| 2024-09-01 | 2.44 | 1 | 21.56 |
| 2024-12-01 | 2.87 | 2 | 21.13 |
| 2025-03-01 | 3.16 | 2 | 20.84 |
| 2025-06-01 | 3.44 | 3 | 20.56 |
| 2025-09-01 | 3.67 | 3 | 20.33 |
| 2025-12-01 | 3.88 | 3 | 20.12 |

Customers therefore accumulate experience across additional Sasol
categories as their observed histories mature.

This is the predictor-side counterpart of the Phase 02 decline in
New-category adoption prevalence.

---

## 15. Shrinking Observed Adoption Space

Customers with at least one category newly appearing in their observed
history during the most recent quarter declined through time.

Observed customer counts were:

    2024-09-01    1,357
    2024-12-01    1,502
    2025-03-01    1,296
    2025-06-01    1,171
    2025-09-01    1,146
    2025-12-01    1,142

At the earliest snapshot this represents approximately 37.6% of
eligible customers.

At the final competition cutoff it represents approximately 20.8%.

Recent new-category events similarly decline from:

    2,560

to:

    1,956.

This pattern is consistent with customers accumulating observed category
history and therefore having fewer categories remaining that are new to
their recorded history.

---

## 16. Category Reactivation

Observed category reactivation becomes increasingly measurable as the
available historical record lengthens.

Customer counts with at least one recently reactivated category were:

    2024-09-01        0
    2024-12-01      470
    2025-03-01      615
    2025-06-01      808
    2025-09-01    1,003
    2025-12-01    1,138

At the final cutoff there were:

    1,623

individual category reactivation events.

Reactivation is therefore a substantial component of customer-category
behaviour.

---

## 17. Earliest-Snapshot Left Truncation

The zero category-reactivation count at the 2024-09-01 cutoff requires
careful interpretation.

The transaction dataset begins on 19 March 2024.

For the 2024-09-01 forecasting origin, almost all observed history
before the latest three-month period falls inside the immediately
preceding three-month comparison period.

Consequently, the data provides very limited opportunity to observe a
category that is historically known, absent during the previous quarter,
and then active again during the recent quarter.

The zero result is therefore substantially influenced by left
truncation and should not be interpreted as evidence that category
reactivation did not exist commercially during this period.

The terms "lifetime" and "ever purchased" in the feature names refer to
the complete transaction history observable in the released dataset
before a cutoff, not necessarily the customer's true commercial
lifetime before the dataset begins.

---

## 18. Category Dormancy

Customers frequently stop purchasing individual categories even while
remaining active elsewhere.

Customers with at least one category becoming dormant were:

    2024-09-01    1,045
    2024-12-01    1,279
    2025-03-01    1,701
    2025-06-01    1,790
    2025-09-01    1,726
    2025-12-01    1,797

At the final cutoff there were:

    3,272

category-level dormancy events.

This confirms that category behaviour is dynamic rather than purely
accumulative.

The predictive representation must therefore distinguish historically
known categories from currently active categories.

---

## 19. Missing Category Recency

Missing category-recency values exactly reconcile with historically
unseen category slots.

This is expected behaviour.

For any customer-category combination:

    ever_purchased = 0

implies:

    days_since_last_purchase = missing.

The missing value does not represent data corruption.

It means no qualifying purchase for that category has been observed
before the forecasting cutoff.

This meaningful missingness is preserved for later LightGBM modelling.

---

## 20. Phase 03D Interpretation

After Phase 03D the customer representation can distinguish, for each
original included category:

    never observed

    historically observed but dormant

    recently active

    newly observed in recent history

    reactivated after a dormant quarter

    became dormant after previous-quarter activity

It also retains recent and previous-quarter monetary values separately.

This closely mirrors the historical states from which Sasol's future
Opportunity labels are generated without using future information or
target-derived encodings.

---

## 21. Phase 03D Decision

Phase 03D is considered complete because:

- all 28,167 snapshot observations were preserved;
- all 24 included original categories were represented separately;
- the final feature count reached 281 as designed;
- category purchase states passed all logical reconciliation checks;
- category-recency missingness behaved exactly as intended;
- category maturity and unseen-category space were quantified;
- historical reactivation and dormancy states were represented;
- and no target or future information entered feature construction.

No machine-learning model has yet been trained.

The project may now proceed to:

    Phase 03E - Feature Integrity, Leakage and Temporal Drift Audit

