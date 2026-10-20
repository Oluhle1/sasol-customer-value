# Sprint 17 E4 — log-space product blend (decision layer only)

Pre-registered 2026-10-17, before any evaluation. Machine copy:
`outputs/sprint17/proposal_logit_blend.json`. Do not edit the gate after seeing
scores.

## Why this is new

Sprint 16 E7/E8 closed probability-space product transforms and cached-array
mixtures: `blend_search.json` only swept arithmetic mixtures of the cached
products. The 2019 Google Analytics winner's runner-up family and the Elo
Merchant Category winners blended **logits**, not probabilities; that space has
never been evaluated here.

## Hypothesis and grid

Geometric mixture `softmax((1−g)·log catdyn17 + g·log r4full)` versus the
arithmetic mixture at the same `g`, `g ∈ {0.25, 0.5, 0.75}`, decoded with the
frozen vR vector. Single best grid point by (pool mean gain, pool min gain,
geom over arith, smaller |g−1|).

`r4full` = `rank4_combine(category_dynamics, rank4_std2, "full")`, the shipped
product in `sasol_sprint16_r4full_vR.csv` (public 0.313217559).

## Gates

Screen on the pool (April, May, July, August 2025) versus r4full+vR:

- pool mean gain ≥ **+0.001** and no pool origin worse than **−0.001**.

Only if the screen passes, evaluation folds June and September are scored with
the same frozen product/decode: each must gain ≥ **+0.001** over r4full+vR and
their mean must be ≥ **+0.002**. No vector retune at any stage.

## Export

Only if both gates pass: rebuild the selected grid point at 2025-12-01 from the
cached catdyn and rank4 arrays, decode with vR, copy CLV columns byte-identical
from `submissions/sasol_sprint16_r4full_vR.csv`, and write
`submissions/sasol_sprint17_logitblend.csv` plus an audit JSON. Otherwise
nothing is exported.
