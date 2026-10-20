# Sprint 17 E3 — big-three head reparameterisation (cascade family)

Pre-registered 2026-10-16, before any fit. Machine copy: `outputs/sprint17/proposal_binary_cascade.json`. Do not edit the gates after seeing scores.

## Why this is a different family, not E1 again

Sprint 16 E7/E8 closed rank4 capacity/seeds/transforms; E9/E9b closed the neural family on the 305 features; sprint 17 E1 closed the weekly fuel-rand feature block and E2 closed the BG/NBD comeback overlay.

E1's failure mode is informative: the production errors are rows where the *shape* already agrees with the wrong label, so more.Level2-granular features cannot separate those cells. The remaining structural lever is the estimator itself. The shipped rank4 head is one 4-way LGBM softmax on [Inactivity, Stable, Fuel, Other]; all three big-three boundaries share the same tree pool and the ~8% `Other` class contributes gradient that dilutes the boundaries that carry ~56% of public F1 mass (oracle top-2 gain is inside the big three).

E3 tests two reparameterisations that break that sharing:

- `big3_softmax`: a 3-class LGBM softmax (Inactivity/Stable/Fuel) trained **only on rows whose target is one of the big three**, i.e. `Other` no longer competes for the same trees.
- `binary`: two independent BCE heads (Inactivity vs rest, Fuel vs rest); `pStable = 1 − pI − pF` (clipped) — heads choose their own feature interactions and the cascade identity closes the row.

Both reuse the exact sprint-16 production protocol for the rank4 head (stacked monthly cutoffs `≤ cutoff−3m`, recency weights `0.70 + 0.05·(months−9)`, early stop 60 on the previous month, refit with best rounds, seeds 7/23 averaged, production STD hyperparameters, 305 features). No new features. No hyperparameter search.

## Product and decode

`r4full` recombination with the big-three probabilities replaced:

- `big3_softmax`: `out[big3] = p3 * (1 − p4_other) / max(p3.sum, eps)`; `out[other] = p4_other * cond(catdyn)` (p4_other from the cached std2 rank4).
- `binary`: same, with `p3 = [pI, max(0, 1−pI−pF), pF]`.

Decode starts frozen at vR. Because independent-head scales differ from softmax scales, exactly one pre-registered fallback per variant is allowed: coordinate-ascent retune of vR **on the pool origins only** (2025-04/05/07/08), then re-screen. This follows the E5 lesson (tune the vector on the shipped product); it touches no evaluation fold.

## Gates (same bars as E1/E2)

Screen on the pool (April, May, July, August 2025), versus production r4full+vR with the cached std2 head:

- mean weighted-F1 gain ≥ **+0.004**, and no pool origin worse than **−0.002**.

If both variants fail, fit nothing further (June/September are never fit from a failed screen).

Evaluation, only if the screen passes: June and September fit with the same protocol; each must gain ≥ **+0.0015** and their mean ≥ **+0.004**. Only then fit December (train ≤ 2025-09-01, `latest = 2025-09-01`), build the candidate with CLV columns copied from `submissions/sasol_sprint16_r4full_vR.csv`, and record the audit. December is public evidence only; nothing is tuned on the folds.

## Recorded outcomes

Written on completion to `outputs/sprint17/binary_cascade_decision.json`. Rejected variants are never exported; the baseline files stay immutable.
