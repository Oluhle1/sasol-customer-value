# Sprint 17 E7 — conditional stacking

Pre-registered 2026-10-17, before any fit. Machine copy:
`outputs/sprint17/proposal_stacking.json`. Gates frozen.

Every combine so far is fixed weights or multipliers. This is the first learned
combiner: a small LGBM meta-classifier over the 17 catdyn probabilities, the 4
rank4 std probabilities, the 4 XGB rank4 probabilities where cached, and five
context columns (n_tx, recency_days, fuel_z_mean, shop_z_mean, crossbuy_q1).

Training is strictly causal: to predict origin O the meta trains only on
origins X with `X + 3m <= O` that have member caches — 07 trains on 03/04,
08 on 03/04/05, June on 03, September on 03..06, December on 03..09.
2025-03/04/05 cannot be scored (no eligible training origins) and are not.

Gates: screen on July/August mean gain ≥ +0.004 and no origin worse than
−0.002; folds June/September each ≥ +0.0015 and mean ≥ +0.004. No fallback.
December export only on a full pass, CLV copied byte-identical from the
current best.
