# Phase 06R-B — Research-Backed Multi-Stage CLV

## Research question

Can the multi-stage customer-value decomposition used in published
non-contractual retail CLV research improve Sasol three-month
fuel-volume and non-fuel-spend forecasting?

Primary research source:

Yan, Y. & Resnick, N. (2024).
"A high-performance turnkey system for customer lifetime value
prediction in retail brands."
Quantitative Marketing and Economics, 22, 169-192.
DOI: 10.1007/s11129-023-09272-x

The published system was evaluated across 12 retail brands and six
evaluation periods.

The paper decomposes future customer value into:

    Prob(return)
        ×
    Frequency conditional on return
        ×
    Average value conditional on return

For Sasol this is adapted as:

Fuel:
    Prob(future fuel purchase)
        ×
    expected future fuel baskets conditional on purchase
        ×
    expected net litres per fuel basket conditional on purchase

Non-fuel:
    Prob(future non-fuel purchase)
        ×
    expected future non-fuel baskets conditional on purchase
        ×
    expected net rand per non-fuel basket conditional on purchase

The fuel-volume formulation is a Sasol-specific adaptation because
the source paper predicts monetary CLV rather than litres.

---

# Phase 06R-B1 — Auxiliary Outcome Construction

The official Sasol outcomes were decomposed into auxiliary future
components using the same three-month outcome windows as the official
label generator.

Generated components:

    future_fuel_purchase_active
    future_fuel_basket_count
    future_fuel_litres
    future_fuel_litres_per_basket

    future_nonfuel_purchase_active
    future_nonfuel_basket_count
    future_nonfuel_rand
    future_nonfuel_rand_per_basket

Integrity checks passed across all 22,679 historical customer snapshots.

Official fuel target agreement:

    PASS

Official non-fuel target agreement:

    PASS

Component reconstruction:

    PASS

For every historical snapshot:

    future_purchase_active
        ==
    future_raw_target > 0

There were zero active-but-zero-net-value cases for either fuel or
non-fuel.

## B1 temporal behaviour

Fuel future activity rate:

    2024-09-01    0.8811
    2024-12-01    0.8231
    2025-03-01    0.7621
    2025-06-01    0.7271
    2025-09-01    0.7075

Non-fuel future activity rate:

    2024-09-01    0.4235
    2024-12-01    0.3758
    2025-03-01    0.3500
    2025-06-01    0.3447
    2025-09-01    0.3494

Conditional fuel basket intensity remained comparatively stable:

    mean litres per active fuel basket
    approximately 23.6-24.1 litres

Conditional non-fuel basket value also remained comparatively stable:

    mean rand per active non-fuel basket
    approximately R65-R69

Interpretation:

The largest temporal change occurs in whether customers return,
rather than in basket value conditional on returning.

This provides a strong empirical reason to test a return × frequency
× value decomposition.

---

# Phase 06R-B2 — Naive Multi-Stage Random Forest

Purpose:

Test whether the decomposition itself improves prediction before
implementing the full enhanced method described by Yan & Resnick.

Models:

1. Persistence baseline
2. Direct Random Forest regression
3. Naive multi-stage Random Forest:

       Prob(return)
           ×
       expected frequency
           ×
       expected value per basket

Validation:

    F1: Sep 2024 -> Dec 2024
    F2: Sep+Dec 2024 -> Mar 2025
    F3: through Mar 2025 -> Jun 2025
    F4: through Jun 2025 -> Sep 2025

## Fuel results

                Direct RF      Multi-stage RF      Change
    F1          0.660793       0.936287            +0.275493
    F2          0.664799       0.895865            +0.231066
    F3          0.698385       0.888791            +0.190406
    F4          0.633434       0.886615            +0.253180

Row-weighted:

    Direct RF       0.664034
    Multi-stage RF  0.900435

Change:

    +0.236401

Multi-stage improved:

    0 / 4 folds

## Non-fuel results

                Direct RF      Multi-stage RF      Change
    F1          0.883685       1.460411            +0.576726
    F2          0.814478       1.296427            +0.481949
    F3          0.780165       1.243766            +0.463601
    F4          0.784168       1.244555            +0.460388

Row-weighted:

    Direct RF       0.812552
    Multi-stage RF  1.304793

Change:

    +0.492241

Multi-stage improved:

    0 / 4 folds

## Activity discrimination

Fuel activity AUC:

    F1  0.883293
    F2  0.894218
    F3  0.905598
    F4  0.917937

Non-fuel activity AUC:

    F1  0.777669
    F2  0.794868
    F3  0.827979
    F4  0.821554

The activity classifier therefore contains useful signal even though
the final multiplicative CLV prediction performs poorly.

## B2 decision

The naive multi-stage Random Forest formulation is REJECTED.

It performed worse than direct RF in all eight target/fold
comparisons.

However, B2 is not a complete reproduction of Yan & Resnick's
Enhanced Multi-Stage Random Forest.

The published EM-RF additionally:

1. balances the return/churn training classes;
2. recalibrates return probabilities on the original population;
3. learns a sigmoid transformation of the return probability;
4. estimates the sigmoid parameters from cross-validated training
   predictions;
5. chooses the transformation to reduce aggregate CLV error.

Therefore B2 rejects only the naive multiplicative implementation.

It does not yet reject the published EM-RF method.

---

# Next registered experiment — Phase 06R-B2b

Reproduce the published enhanced return-probability adjustment before
changing model family.

Primary question:

    Does probability calibration plus the learned CLV-error sigmoid
    materially improve the naive multi-stage RF on Sasol?

Secondary question:

    Is the enhanced multi-stage RF competitive with direct RF?

No LightGBM adaptation will be attempted until this question is
answered.

