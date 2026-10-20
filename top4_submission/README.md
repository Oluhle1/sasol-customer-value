# Top-4 submission — Opportunity head model

Separate from the frozen Phase 04–06 pipeline. Fuel and non-fuel values are copied unchanged from the audited Phase 05 submission. Only the Opportunity label can change.

## Why this experiment

On the public leaderboard of 28 September 2026, Oluhle is 21st at 0.30225. Fourth place is 0.31034. About two-thirds of that gap is Opportunity weighted F1 (0.524 versus 0.538–0.546). Fuel RMSE and non-fuel RMSE are already inside the top-10 band.

The accepted 17-class model almost never emits an Adoption label. Stable is over-predicted (about 40% of predictions against about 27% of customers on the latest fold). Phase 06C multiplied macro and subtype probabilities, so a single Stable score was compared with one eleventh of the Adoption score. That submission scored 0.284 publicly and was rejected.

## Pre-registered candidate

Train a 5-class LightGBM head on the same 276 features and the same square-root class weights:

1. Inactivity
2. Stable
3. Existing-category growth: Fuel
4. Other existing-category growth
5. New-category adoption

Fuel growth, Stable and Inactivity stay as their own classes, because those three dominate weighted F1. Adoption is one class at this stage, so its probability is not split across eleven labels before the decision. If the head predicts other growth or adoption, a second model trained only on that branch picks the official subtype. The subtype probability is not multiplied back into the head probability.

Tree parameters, seed, and weighted-F1 early stopping match Phase 04D. No feature is added. No fuel or non-fuel model is retrained.

## Acceptance rule

Outer folds are F2, F3 and F4. The control is a 17-class square-root-weighted LightGBM trained in this same run, decided by argmax.

Accept the head model only if all three hold:

1. Row-weighted F2–F4 weighted F1 improves.
2. F4 weighted F1 does not fall.
3. At least two of the three folds improve.

A second candidate is already specified and is used only if the head model fails this rule. It keeps the 17-class probabilities and searches a small Stable penalty and Fuel / Inactivity boost on the previous labelled cutoff only, then applies that fixed shift to the next cutoff. The same three conditions apply. The outer fold is not used to choose the shift.

The competition file is written only after a candidate passes.

## Result

The head model passed. The probability shift did not.

| Fold | Control weighted F1 | Head weighted F1 | Change |
|---|---:|---:|---:|
| F2 | 0.4751 | 0.4799 | +0.0047 |
| F3 | 0.4966 | 0.4978 | +0.0012 |
| F4 | 0.5001 | 0.5085 | +0.0083 |
| Row-weighted | 0.4911 | 0.4960 | +0.0048 |

All three folds improved, including the latest one. On that latest fold the gains are Stable, Fuel growth, Beverages growth and Food & beverage growth, plus a few Adoption labels that the 17-class model almost never emitted. Inactivity F1 dipped by 0.004.

`sasol_top4_opportunity_submission.csv` keeps the Phase 05 fuel and non-fuel values and changes 847 Opportunity labels.

`sasol_top4_on_0302_submission.csv` is the file to upload. It keeps the fuel and non-fuel values from the public 0.302250514 submission and applies the same new Opportunity labels. 1,318 of those labels differ from that submission. Both files have the 5,488 official IDs in sample order. The December 2025 predictions include 110 Adoption customers.

A local gain of 0.008 weighted F1 is worth about 0.003 on the Zindi score if it appears on the public slice. That is a real step and it is not enough, on its own, to move 0.302 into the 0.310 band where 4th place sits.

## Public result

`sasol_top4_on_0302_submission.csv` scored 0.295100903. Fuel and non-fuel were the 0.302250514 values, so that drop of 0.00715 is an Opportunity weighted-F1 drop of about 0.018, from 0.5239 to 0.5060. That matches this model's own September 2025 validation F1 of 0.508.

`sasol_top4_opportunity_submission.csv` scored 0.294823176. It uses the same Opportunity labels and the Phase 05 fuel and non-fuel values. The 0.00028 gap between these two public scores is the whole public difference between those two value models. The 0.302250514 file beats Phase 05 by 0.01286, and about 0.01258 of that is its Opportunity labels (about +0.031 weighted F1 on the public slice). Phase 05 Opportunity is about 0.492 public weighted F1. The head model is about 0.506. The 0.302250514 labels are about 0.524.

The 1,318 changed labels moved customers out of Fuel growth, Stable, and Food & beverage growth into Other growth, Beverages growth, Inactivity, and Adoption. The 0.302250514 labels point the other way relative to Phase 05, and those labels remain the stronger public set. This file should not be one of the two submissions chosen for the private leaderboard.
