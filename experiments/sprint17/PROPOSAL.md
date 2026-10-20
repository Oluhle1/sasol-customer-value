# Sprint 17 E1 — weekly fuel-rand rank4

Pre-registered 2026-10-16, before any fit. The machine-readable copy of this
gate is `outputs/sprint17/proposal_weekly_rank4.json`. The fit reads that
file. Do not edit the gate after seeing scores.

## Why this is a different family

Public best is `submissions/sasol_sprint16_r4full_vR.csv` at 0.313217559.
The 16 Oct rank-10 cut is 0.316711718. Sprint 16 closed multipliers,
product transforms, rank4 capacity and seeds, CatBoost, neural nets on the
305 features, and CLV blend weights. Remaining errors are confident swaps
among Inactivity, Stable, and Fuel growth.

Those three labels are a threshold on next-quarter category rand. Fuel
growth needs a historical fuel purchase, a future qualifying basket, at
least R20 more fuel rand than the previous calendar quarter, and at least
25% more. The 305 features store monthly fuel litres, quarterly fuel rand
(Sale_Fuel only), and 14/28/56-day visit counts. They do not store the
week path of fuel-category rand, which is the unit the rule uses.

Fuel cadence and category dynamics stay as they are. This run only replaces
the four-class probabilities inside the r4full product.

## Frozen design

- Features: 16 weeks strictly before the cutoff. Week 0 is the latest.
  Each week contributes signed log fuel-category rand, signed log Sale_Fuel
  litres, log1p qualifying fuel baskets, and signed log shop rand.
- Scalars: last-4 and previous-4 fuel rand, their difference, the last-4
  run-rate versus the official +R20/+25% need, share of the last 8 weeks
  with a qualifying fuel basket, last-4 shop rand, and a last-4 activity flag.
- Head: production rank4 hyperparameters, seeds 7 and 23. No search.
- Product: r4full. Category-dynamics still splits the non-big-three mass.
- Decode: frozen vR. CLV columns copied from the current best file.

## Gates

Pool, and the only place a result can be inspected before evaluation:
2025-04-01, 2025-05-01, 2025-07-01, 2025-08-01.

Screen, versus production r4full+vR: mean weighted-F1 gain at least 0.004,
and no pool origin worse than -0.002. If this fails, June and September
are not fit and nothing is exported.

Evaluation, only after the screen passes: 2025-06-01 and 2025-09-01 must
each gain at least 0.0015, and their mean must gain at least 0.004.
Only then is a December file written.
