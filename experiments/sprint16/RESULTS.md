# Sprint 16 — top-10 gap round (Oct 16, 2026)

Status after the competition closed: `sasol_sprint16_r4full_vR.csv`
(0.313217559) remained the best public file. It was not the file selected
for the private leaderboard. See `SOLUTION.md` and
`docs/final_outcome_2026-10-19/README.md`.



Scope: close as much of the gap to the public top-10 cut (0.316711718, rank 10)
from the current best public score 0.309411451 (`submissions/sasol_top10_rank4.csv`).
Gap decomposition from the live leaderboard: nearly all of it is Opportunity F1
(current 0.536248187 vs ~0.5545 at rank 10); fuel (0.5891127) and non-fuel
(0.724222712) RMSEs are mid-pack with far smaller attainable swings.

All work here is December-export-oriented and protocol-gated. June/September 2025
remain evaluation folds; no multiplier or threshold was tuned on them.

## Confirmed facts before this round

- score ≈ 0.3 * (1 - fuel/0.78) + 0.3 * (1 - nf/0.78) + 0.4 * weighted-F1
  (fitted on 30+ leaderboard rows; both RMSE scales ≈ 0.78).
- Current best = `sasol_top10_rank4.csv` (Opportunity = catdyn probs with the
  four-class rank4 "full" recombination, accepted multipliers; CLV columns from
  `sasol_oct10_small_fuel_values.csv`). Baseline snapshot:
  `outputs/sprint16/baseline_0309411451.csv`, sha256 `ee559dac...`.
- Individual and greedy ensemble results on June/September:
  `outputs/sprint16/individual_models.csv`, `outputs/sprint16/greedy_history.json`.

## E1 — Opportunity probability blend (accepted)

Recipe: `0.4 * category_dynamics + 0.4 * early_history + 0.2 * catdyn+rank4_full`,
decode `argmax(prob * accepted_multipliers)`.

| fold | catdyn (baseline) | blend | gain |
|---|---|---|---|
| 2025-06-01 | 0.509527 | 0.512113 | +0.002586 |
| 2025-09-01 | 0.516557 | 0.519127 | +0.002570 |

Gate (pre-registered in `outputs/sprint16/proposal_december_blend.json`): both
folds ≥ +0.001 — passed. Equal-weight subset search over all cached 17-class
arrays and weight search are recorded in `outputs/sprint16/blend_search.json`;
no 4th component improved the minimum fold gain.

## E2 — Pooled decision-multiplier calibration (rejected)

Pre-registration: `outputs/sprint16/proposal_pooled_multipliers.json`.
Coordinate ascent over the frozen multiplier grid, pooled mean weighted F1 on six
non-evaluation origins (2024-12, 2025-03/04/05/07/08): 0.481911 -> 0.485641.
Gate G1: June +0.002593 but **September -0.001808**; G2 (blend): both folds
negative. Rejected per protocol; accepted multipliers retained.
Decision record: `outputs/sprint16/pooled_multiplier_decision.json`.

## E3 — December export (complete)

- `early_history` refit on the 16-origin early panel for cutoff 2025-12-01 with
  the frozen `run_monthly.predict` procedure (`fit_december_early.py`; 838 s,
  seeds 7/23 at 74/77 rounds, save `outputs/sprint16/2025-12-01_early_history.npy`).
- `candidate_blend_2025-12-01.csv` (`build_december_candidate.py`): replication
  guard passed (rank4_full decode of December catdyn probs reproduces all 5488
  baseline Opportunity labels exactly); 281 label changes; CLV columns byte-identical.

## E4 — Multiplier transfer diagnostic (exploratory, decode candidates)

`multiplier_transfer.py` compares four decode vectors on all cached origins
(decision-layer only, no model fitting): accepted / uniform / vD (tuned on
2024-12 alone) / v5 (tuned on 2025-03/04/05/07/08, i.e. pool minus December).

| vector | Dec-2024 | pool5 mean | 2025-06 | 2025-09 |
|---|---|---|---|---|
| accepted | 0.428659 | 0.492561 | 0.509527 | 0.516557 |
| uniform | 0.432897 | 0.488188 | 0.503523 | 0.500302 |
| vD | 0.452554 | 0.478513 | 0.503044 | 0.490520 |
| v5 | 0.429855 | 0.495853 | 0.516100 | 0.517490 |

vD is a textbook single-origin overfit (great on 2024-12, bad everywhere else).
v5 improves catdyn on both evaluation folds (+0.00657 June, +0.00093 Sept) and
on Dec-2024 (+0.0012). Evaluated against the **current submission** (local
June 0.509630 / Sept 0.520405 for r4full+accepted), blend and vector changes:

| candidate (probability product + vector) | June Δ | Sept Δ | mean Δ |
|---|---|---|---|
| blend + accepted (E1, registered) | +0.00248 | −0.00127 | +0.00061 |
| r4full + v5 (decode only) | +0.00244 | −0.00051 | +0.00097 |
| blend + v5 | +0.00321 | −0.00295 | +0.00013 |
| catdyn + v5 | +0.00647 | −0.00291 | +0.00178 |

No candidate dominates on both folds; these were exported for public probing
(see file table). Nothing here is promoted by a passing gate; `uniform` and
`vD` files are recorded for completeness but are not recommended.

## E5 — vR: product-matched multiplier calibration (new probes)

Public results of the first four probes (recorded in
`outputs/sprint16/public_scores_2026-10-16.json`):

| file | public score | delta |
|---|---|---|
| `sasol_sprint16_blend.csv` (blend + accepted) | **0.310373619** | +0.000962 (new best) |
| `sasol_sprint16_blend_v5.csv` | 0.309631118 | +0.000220 |
| `sasol_sprint16_catdyn_v5.csv` | 0.309945351 | +0.000534 |
| `sasol_sprint16_r4full_v5.csv` | 0.308187911 | −0.001224 |

v5's effect on the public LB depends on the product: catdyn +0.00078, blend
−0.00074, r4full −0.00122. Lesson: tune decode vectors on the exact product
shipped. Re-tuning the coordinate grid on the **r4full product** over the four
non-evaluation origins with rank4 coverage (2025-04/05/07/08) yields **vR** =
Beverages-growth 2.0, Inactivity 0.8, adoption 3.0, others as accepted
(`outputs/sprint16/vR_decision.json`). Evaluated after fitting:

| product | June gain | Sept gain |
|---|---|---|
| blend | +0.00179 | +0.00515 |
| r4full | +0.00147 | +0.00577 |
| catdyn | +0.00520 | +0.00479 |

vs the current submission (local 0.509630 / 0.520405): blend+vR +0.00428 /
+0.00388 — positive on both folds, the only candidate so far to be so. New
probe files (upload pending):

| file (submissions/) | product + vector | changes vs baseline |
|---|---|---|
| `sasol_sprint16_blend_vR.csv` | blend + vR | 412 |
| `sasol_sprint16_r4full_vR.csv` | r4full + vR | 291 |
| `sasol_sprint16_catdyn_vR.csv` | catdyn + vR | 439 |

Caveat: Dec-2024 (catdyn) is mildly negative under vR (−0.0018); December
transfer is what the public probe measures.

## E6 — vR public outcome and follow-up probes

vR results on the public LB (recorded in
`outputs/sprint16/public_scores_vR_2026-10-16.json`):

| file | public score | delta vs prior best |
|---|---|---|
| `sasol_sprint16_r4full_vR.csv` | **0.313217559** | +0.002844 (new best) |
| `sasol_sprint16_catdyn_vR.csv` | 0.310734505 | +0.000361 |
| `sasol_sprint16_blend_vR.csv` | 0.309967901 | −0.000406 |

vR effect by product on public: r4full **+0.00381**, catdyn +0.00157, blend
−0.00041. The r4full product with its own tuned vector is the strongest December
combination; December behaves September-like (Sept locally prefers r4full+vR
too). Gap to the rank-10 cut shrinks from 0.007300 to 0.003494.

Follow-up probes built from local fold evidence (both improve on both folds vs
r4full+vR):

| file (submissions/) | change | June Δ | Sept Δ | label changes vs r4full+vR |
|---|---|---|---|---|
| `sasol_sprint16_r4full_vR2.csv` | Stable 0.85 → 0.90 (extended grid) | +0.00101 | +0.00070 | 73 |
| `sasol_sprint16_rank4early_vR.csv` | product 0.6*r4full + 0.4*early, vector vR | +0.00349 | +0.00018 | 216 |

Follow-up probe outcomes (`outputs/sprint16/public_scores_vR2_2026-10-16.json`):
both regressed — `r4full_vR2` 0.311350092 (−0.00187) and `rank4early_vR`
0.309780905 (−0.00344). The December public surface is sharp around
r4full+vR; every perturbation (Stable shift, early/catdyn dilution, v5) lost
0.002-0.003. Local fold deltas below ~0.001 do not transfer reliably. Best
submission remains `sasol_sprint16_r4full_vR.csv` (0.313217559).

## Candidate files (all 5488 rows, CLV columns frozen from baseline)

| file (submissions/) | product | vector | changes vs baseline |
|---|---|---|---|
| `sasol_sprint16_blend.csv` | blend | accepted | 281 |
| `sasol_sprint16_r4full_v5.csv` | r4full | v5 | 485 |
| `sasol_sprint16_blend_v5.csv` | blend | v5 | 464 |
| `sasol_sprint16_catdyn_v5.csv` | catdyn | v5 | 468 |

Source copies and audits: `outputs/sprint16/candidate_*`, `candidate_blend_audit.json`,
`multiplier_variant_audit.json`, `decode_candidate_audit.json`.

## Closed paths (not revisited)

- Fuel/non-fuel RMSE: all candidate blends (cat_shop, calendar, fuel cadence,
  lapsed fuel, recipe re-weighting) either shipped already or failed their gates;
  no validated RMSE gain remains in the cached assets.
- Rank5 / remainder / corrector mislabel repairs: failed their gates.
- Headroom diagnostics: top-2 oracle +0.2296 but median wrong-margin ~0.16, i.e.
  model errors are confident; no decision-layer fix has access to that mass.

## E7: extended tuning set and product transforms (both rejected)

Goal: beat vR by improving the *tuning set* or the *product transform*.

- 8-origin tuning (`tune_vR8.py`): fitted rank4 for the old origins
  2024-12..2025-03 on the early panel (the only fittable option there) and
  pooled with the 4 production-like origins. The equal-weight objective was
  dominated by the old origins' weak-p4 quirks (huge +0.01..+0.03 F1 swings
  on subset populations) and produced vectors (F&B 0.85, adoption 2.0) that
  collapsed on the evaluation folds: vR8 June 0.5074 (-0.0037), Sept 0.5094
  (-0.0168); vR7 Sept 0.5127 (-0.0135). Rejected; recorded in
  `outputs/sprint16/pool_r4full_vector_vr8.json`.
- Product transforms (`tune_product_transforms.py`): conditional sharpening
  cond^g, p4-mass temperature, and global product temperature all failed to
  beat vR on the pool (best: massg 0.8, +0.00025 pooled). Recorded in
  `outputs/sprint16/product_transform_search.json`.
- Conclusion: the decode/product surface is locally optimal at vR; the same
  principle (tune only on the 4 production-like origins) must be applied to
  model-level rank4 changes instead.

## E8: exhaustive push beyond vR (all rejected on non-eval evidence)

Every remaining lever was tested on the pool origins (and folds where
promising); none matched the vR signature (large, balanced gain):

- rank4 model variants (`fit_rank4_variants.py`): 4-seed averaging std4
  -0.0011 pooled, capacity capbig -0.0021, capsma -0.0003. Local optimum.
- rank4 CatBoost diversity member (`fit_rank4_catboost.py`): -0.0131 alone,
  averaged with LGBM -0.0031 pooled. Rejected.
- cached-array combo sweep: best mean(lgb,capbig,capsma) +0.0004 pooled;
  everything else <= 0. Not probe-worthy.
- adoption subgroup split (`tune_adoption_split.py`): +0.00004 pooled;
  the only accepted move duplicated the public-failed Stable 0.9. Rejected.
- mirror probes on the public evidence (Stable 0.80, Beverages 1.6,
  Inactivity 0.90): all lose on pool AND folds (e.g. stable0.80
  -0.0018 pool, -0.0022/-0.0022 folds). Public optimum is sharp; not probing.
- headroom diagnostic: errors are ~85% within-big-three pairwise flips
  (top-2 coverage 0.76); decode/transforms cannot fix these.
- rank4 already trains on the rich category feature set (cat_*/prev__*
  included in the 305); no missing feature block. CLV axis fully screened
  in oct10 decisions.

Conclusion: r4full+vR is the validated ceiling of this model family.
Closing the gap to 0.3167+ needs a materially different model/feature
family, not further tuning.

## E9-E11: new model families and CLV ensembles (all closed)

- E9/E9b multi-task neural nets (`fit_multitask_nn.py`, `fit_nn_scaling.py`,
  proposals in `outputs/sprint16/proposal_multitask_nn.json`,
  `proposal_nn_scaling.json`): single-seed MLP failed the pool screen
  (nn -0.0049, nn+catdyn -0.0027); the scaling diagnostic with 2 seeds and
  dropout 0.3 failed too, even on the larger fold training sets (June
  -0.0034/-0.0009, Sept -0.0062/-0.0022). Neural family closed, no probe.
- E10 CLV blend-weight tuning (`tune_clv_ensembles.py`,
  `proposal_clv_ensemble.json`): the shipped fuel recipe reproduces exactly
  as 0.40*category_values + 0.35*two_part + 0.25*fuel_residual (max diff
  9e-16), but exhaustive simplex tuning gained only 0.00005 pooled RMSE and
  lost on both folds; nonfuel likewise null. Recorded in
  `outputs/sprint16/clv_ensemble_result.json`. Also noted: the shipped
  nonfuel column is not the two_part array (max diff 0.18), so no exact
  anchor exists for its validation.
- E11 stacking over saved probability sources: infeasible to validate
  honestly - monthly/early_history sources exist only for June/Sept/Dec, not
  the pool origins, so December-only members would have zero non-eval
  validation. Not pursued.

## Next actions

1. Final submission: `submissions/sasol_sprint16_r4full_vR.csv`
   (0.313217559, verified 5488 rows, CLV frozen, sha256
   `67478c9c861376834f54976a828bcd753a81c3103a3c15398fdff8a4195a593e`).
   It would sit ~rank 19 on the 16 Oct leaderboard snapshot (rank-19 score
   0.313175441), up from ~rank 42 at round start; the rank-10 cut in that
   snapshot is 0.316711718.
2. E7-E11 closed every remaining lever (tuning set, transforms, rank4
   variants, diversity members, adoption split, mirror probes, neural nets,
   CLV ensembles, stacking feasibility). Status: stopping and keeping
   `sasol_sprint16_r4full_vR.csv` as the final submission. Reopen only with
   materially new information (e.g. raw transaction sequences, external
   data) whose pool gain approaches vR's (+0.0080 pooled, +0.0015/+0.0058
   folds).
