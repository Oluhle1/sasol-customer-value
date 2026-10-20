# Sprint 17 E5 — XGBoost rank4 member

Pre-registered 2026-10-17, before any fit. Machine copy:
`outputs/sprint17/proposal_xgb_rank4.json`. Do not edit the gates after seeing
scores.

## Why this is a different family

Sprint 16 closed LGBM capacity/seed variants (E8), a CatBoost member (E8), and
the neural family on the 305 features (E9). XGBoost is installed (3.2.0) and
has **never been used in this repository**; the WSDM Cup 2018 KKBox winner
took first place with an XGBoost base model blended with LightGBM. The
remaining lever is not more tuning of the shipped head but a different tree
growth algorithm as an additional member of the rank4 product.

## Protocol

Frames are identical to the production rank4 head (`run_rank4.fit_rank`):
stacked monthly cutoffs `≤ origin−3m`, recency weights
`0.70 + 0.05·(months−9)`, early stopping 60 on the previous month, refit at
best rounds, seeds 7/23 averaged, 305 features, no new features, no
hyperparameter search.

XGBoost params (frozen): `multi:softprob`, `n_estimators 1200`, `lr 0.04`,
`max_depth 6`, `min_child_weight 35`, `subsample 0.85`, `colsample_bytree
0.70`, `reg_lambda 1.2`, `tree_method hist`.

Variants:

- `xgb`: `rank4_combine(catdyn, p4_xgb, full)`
- `mix`: `rank4_combine(catdyn, 0.5·p4_std2 + 0.5·p4_xgb, full)`

## Gates (same bars as sprint 17 E1–E3)

Screen on the pool (April, May, July, August 2025) versus r4full+vR:

- mean weighted-F1 gain ≥ **+0.004**, and no pool origin worse than **−0.002**.

One pre-registered fallback per variant: coordinate-ascent retune of vR on the
pool only (same grid as E3), then re-screen at the same bars.

Evaluation, only if the screen passes: June and September fit with the same
protocol; each must gain ≥ **+0.0015** and their mean ≥ **+0.004**.

## Export

Only if both gates pass: December fit (train `≤ 2025-09-01`), rebuild the
passing variant at 2025-12-01, copy CLV columns byte-identical from
`submissions/sasol_sprint16_r4full_vR.csv`, and write
`submissions/sasol_sprint17_xgbmix.csv` with an audit JSON. If both variants
pass, keep the one with the higher minimum fold gain. Rejected variants are
never exported.
