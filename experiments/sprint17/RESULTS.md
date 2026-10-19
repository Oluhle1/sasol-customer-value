# Sprint 17 results

Public file under test: `submissions/sasol_sprint16_r4full_vR.csv` (0.313217559).
Rank-10 cut used for this round: 0.316711718. Gap: 0.003494.

## E1 — weekly fuel-rand rank4 (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL.md`.
Decision: `outputs/sprint17/weekly_rank4_decision.json`.

The production rank4 head was retrained with 71 added features: 16 weeks of
fuel-category rand, Sale_Fuel litres, qualifying fuel baskets, and shop rand,
plus the run-rate against the official +R20/+25% rule. Hyperparameters and
the vR decode stayed frozen.

| origin | gain vs r4full+vR |
|---|---|
| 2025-04-01 | -0.003822 |
| 2025-05-01 | -0.002885 |
| 2025-07-01 | -0.001466 |
| 2025-08-01 | -0.001215 |

Mean gain -0.002347. The screen required +0.004 and no origin worse than
-0.002. June and September were not fit. Nothing was exported.

The block is aligned with quarterly fuel rand (correlation about 0.77) and
took 13% of split gain. True Fuel customers are accelerating into the cutoff
and true Stable customers are decelerating, but the production mistakes are
the rows where that shape already agrees with the wrong label. Retraining
on it made the head worse.

## E2 — inactivity comeback override (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E2.md`.
Decision: `outputs/sprint17/comeback_rule_decision.json`.

On the pool, customers called Inactivity who are truly Fuel growth have
shorter fuel recency and a higher BG/NBD expectation than true inactivity
(Cohen's d about 0.5, same sign on April–May and July–August). The frozen
rule flips Inactivity to Fuel growth when recency and that expectation clear
a grid point. The selected point also requires Fuel to be the second-ranked
class.

Best point: recency at most 45 days, `fuel_bgnbd_z` at least 0.5, second
class Fuel. It flips 19 to 40 rows per pool origin. Mean weighted-F1 gain
+0.000575, minimum +0.000392. The screen required +0.004. June and September
were not scored. Nothing was exported.

The same weekly block correlates with the shipped fuel residual at about
0.01 to 0.08 on the pool, and the correlation is smaller on the later
origins. That is not a fuel-RMSE refit.

## E3 — big-three cascade heads (rejected, corrected rerun)

Pre-registration: `experiments/sprint17/PROPOSAL_E3.md`.
Decision: `outputs/sprint16/binary_cascade_decision.json`.

The first E3 attempt was voided before any scoring was read: its caches were
all-zero (a `fit_head` tuple-return bug, class-code vs class-name alignment,
wrong member names, and filenames with `:`/`&`). Invalid caches were moved to
`outputs/sprint16/cascade_invalid/`; `fit_binary_cascade.py` was fixed and the
variant rerun from scratch.

| variant | pool mean gain | pool min gain | screen |
|---|---|---|---|
| big3_softmax | -0.002611 | -0.007182 | fail |
| big3_softmax + vR retune | -0.002494 | -0.006581 | fail |
| binary (P(inact), P(fuel), stable = remainder) | -0.017500 | -0.033175 | fail |
| binary + vR retune | -0.001925 | -0.008916 | fail |

The fixed heads train normally (real early-stopped rounds), but replacing the
17-class big-three mass with dedicated heads loses on every origin. The
binary complement construction is far worse. June and September were not fit,
nothing was exported. This closes the cascade/head-replacement family.

## E4 — log-space product blend (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E4.md`.
Decision: `outputs/sprint17/logitblend_decision.json`.

Zero-fit decision-layer test: geometric and arithmetic mixtures of the
category-dynamics product and the r4full product, `g` in {0.25, 0.5, 0.75},
decoded with the frozen vR vector. All six grid points have negative pool
mean gain; the best (`geom g=0.75`) reaches mean -0.000436, minimum -0.003367.

| grid point | pool mean gain | pool min gain |
|---|---|---|
| arith_g0.75 | -0.000538 | -0.003149 |
| geom_g0.75 | -0.000436 | -0.003367 |
| others (arith/geom 0.25, 0.5) | <= -0.001493 | <= -0.006746 |

Screen required +0.001; folds were not scored and nothing was exported.
Endpoint note: catdyn+vR beats r4full+vR on June (0.514722 vs 0.511101) but
loses on September (0.521349 vs 0.526178), so a fold-directed choice would
not be stable anyway; the mixture is worse than r4full on the pool.

## E5 — XGBoost rank4 member (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E5.md`.
Decision: `outputs/sprint17/xgb_rank4_decision.json`.

XGBoost 3.2.0, depth-wise growth (`max_depth 6`, `hist`), fitted on the exact
production rank4 frames (same 305 features, stacking, weights, early stopping
and seeds). Its ranks 7/23 averaged into the existing product two ways. The
first run hit an infrastructure bug when reloading its own cache
(`allow_pickle=False` on an object-dtype ids array); the flag was fixed and the
variant rescored from the same cached probabilities without refitting, gates
unchanged. No fold was fit with XGBoost.

| variant | pool mean gain | pool min gain | retune mean | retune min | screen |
|---|---|---|---|---|---|
| xgb | -0.000854 | -0.001475 | +0.000490 | -0.003144 | fail |
| mix (0.5 std + 0.5 xgb) | -0.000456 | -0.001576 | +0.000815 | -0.000803 | fail |

XGB needs about 1.5x the boosting rounds of the LGBM head and lands on the
same plate: small negative mean, one or two origins slightly positive (mix is
+0.001972 on August). A different tree growth algorithm does not change the
ceiling of this feature panel. June and September were not scored, nothing was
exported. This closes the remaining estimator-family member for rank4.

## E6 — price/discount/novelty feature block (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E6.md`.
Decision: `outputs/sprint17/price_rank4_decision.json`.

The first new-signal-source experiment: 23 frozen columns built from raw
fields the 305 panel never uses (`TotalModifierAmount` discount lines,
`StockItemId` product novelty, `UnitSellingExclPrice` recency price) —
discount incidence/depth 90d/30d/trend, item entropy, new-item shares and
per-category price changes, all strictly pre-cutoff. Build ran in 49s with an
exact future-leakage check (truncated vs full data, max diff 0). The rank4
head was refit with 328 columns using the production recipe; March 2025 got a
production baseline refit for both arms (0.482292 vs 0.485365).

| origin | gain vs r4full+vR |
|---|---|
| 2025-03-01 | +0.003073 |
| 2025-04-01 | +0.000784 |
| 2025-05-01 | -0.003877 |
| 2025-07-01 | -0.001752 |
| 2025-08-01 | +0.004235 |

Mean +0.000493, minimum -0.003877. The screen required +0.004 and no origin
worse than -0.002. June and September were not fit, nothing was exported. The
block is noisy-positive on March/August and negative on May/July; the panel's
four existing price features plus 305 columns already capture most of what is
learnable in the raw fields. `px_fuel_disc_share_90` is nearly degenerate
(0-4 nonzero rows per origin). No feature was added, removed or redefined
after the frozen list.

## E7 — conditional stacking (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E7.md`.
Decision: `outputs/sprint17/stacking_decision.json`.

First learned combiner: LGBM meta-classifier over catdyn(17) + rank4 std(4) +
XGB rank4(4, NaN where uncached) + five context columns, trained strictly
causally (origin O trains on X with `X + 3m <= O`).
July (train 03/04): -0.052598. August (train 03/04/05): -0.022524.
Mean -0.037561. Small per-origin meta training sets cannot replace the frozen
vR vector; the fixed decode beats a learned one everywhere. Screen failed,
June/September not scored, nothing exported.

## E8 — EM prior correction (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E8.md`.
Decision: `outputs/sprint17/em_prior_decision.json`.

Saerens-Latinne-Decaestecker EM reweighting of the r4full product toward the
O-3m label prior, frozen vR decode. Every origin negative:
03 -0.019493, 04 -0.011355, 05 -0.018889, 07 -0.010984, 08 -0.009185;
mean -0.013981. The vR multipliers already encode the useful prior tilt;
moving marginals per customer only destroys mass where the labels live. Screen
failed, nothing exported.

## E9 — TabM member (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E9.md`.
Decision: `outputs/sprint17/tabm_decision.json`.

TabM-style batch-ensemble MLP (k=8, rank-1 adapters, 128-wide, AdamW, early
stop on previous-month logloss) under venv torch 2.14.1+cpu, averaged 50/50
with the LGBM rank4 std member. Fits were fast (March 10.5s, August 22s), so
the 15-minute abort rule did not trigger.

| origin | gain vs r4full+vR |
|---|---|
| 2025-03-01 | -0.039951 |
| 2025-04-01 | -0.027959 |
| 2025-05-01 | -0.032114 |
| 2025-07-01 | -0.012160 |
| 2025-08-01 | -0.007047 |

Mean -0.023846, minimum -0.039951. The third neural family on the 305
features loses to the LGBM head the same way MLP and scaling variants did in
sprint 16. Screen failed, June/September not fit, nothing exported.

## E10 — pretrained ItemName embeddings (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E10.md` and
`outputs/sprint17/proposal_item_embeddings.json` (frozen before any download).
Decision: `outputs/sprint17/embed_rank4_decision.json`.

The first pretrained-model arm: all-MiniLM-L6-v2 mean-pooled embeddings of
every distinct (StockItemId, ItemName) row (17,777 rows, 6,849 ids), per-id
means, PCA(16) fit on the item vectors only, then 27 per-customer features
(all-history and 90-day projected means, taste shift, semantic novelty,
engagement norm) joined to the 305 panel columns. Builders:
`build_item_embeddings.py` (venv torch, 72s), `build_embed_features.py`
(leakage check exact on April, 120s).

| origin | gain vs r4full+vR |
|---|---|
| 2025-03-01 | -0.007390 |
| 2025-04-01 | +0.000751 |
| 2025-05-01 | +0.001798 |
| 2025-07-01 | -0.000463 |
| 2025-08-01 | +0.000119 |

Mean -0.001037, minimum -0.007390. Product semantics, like raw-field price
features (E6 +0.0005) and every estimator-family swap, sit below the noise
floor of this panel; the frozen vR decode plus the 305-column LGBM head is
the ceiling. Screen failed, folds and December not fit, nothing exported.

## E11 — GRU over item sequences (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E11.md` and
`outputs/sprint17/proposal_gru_seq.json` (frozen before any tensor build).
Decision: `outputs/sprint17/gru_decision.json`.

The neural input type never tried: raw transaction sequence. Last K=64 lines
strictly before each cutoff (amount > 0, adjustments excluded), channels =
PCA(16) of the E10 MiniLM item vectors + log1p(amount) + log1p(days-before) +
is_fuel + mask; GRU(20->96) -> dropout -> Linear(96,4) on the rank collapse,
trained exactly like E9 (temporal split, recency weights, prev-month early
stop, AdamW 2e-3, batch 256, seed 7), mixed 50/50 with the LGBM rank4 std
member. Builder `build_item_sequences.py` (35s, leakage exact), fitter
`fit_gru_rank4.py` (March fit 178s, under the 20-min abort).

| origin | gain vs r4full+vR |
|---|---|
| 2025-03-01 | -0.015566 |
| 2025-04-01 | -0.009067 |
| 2025-05-01 | -0.011298 |
| 2025-07-01 | -0.008792 |
| 2025-08-01 | -0.000401 |

Mean -0.009025, minimum -0.015566. Best neural family so far (TabM was
-0.024, MLP worse) and nearly neutral at August, but still far below the
gate; the 305-column LGBM head with the frozen vR decode absorbs everything
the sequence order, cadence, and pretrained item semantics can add. FT-
Transformer (305 tokens) was measured at >13 s/step on random data before
fitting and was not run at all; it is not a viable arm on this CPU. Screen
failed, folds and December not fit, nothing exported.

## E12 — public transfer probe (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E12.md` and
`outputs/sprint17/proposal_transfer_probe.json` (frozen before any export).
Scores reported by the user after manual uploads; ledger:
`outputs/sprint17/public_scores_probes_2026-10-17.json`.

December-only probe exports of the two least-negative rejected candidates,
built by `export_probe.py` with CLV copied byte-identical from the shipped
best (`sasol_sprint16_r4full_vR.csv`):

| candidate | local screen mean | public | delta vs 0.313217559 | promoted |
|---|---|---|---|---|
| pricefeat (E6) | +0.0005 | 0.311092232 | -0.002125 | no |
| grumix (E11) | -0.0090 | 0.307605724 | -0.005612 | no |

Neither met the pre-registered +0.003 promotion bar. This closes the
question of whether the strict local gate was hiding a leaderboard winner:
it was not. The transfer pattern held again — strong local gains transfer
(r4full_vR +0.0028), sub-noise and negative local candidates also lose on
public. Selection stays `sasol_sprint16_r4full_vR.csv` plus `blend_vR`.

## E13 — TabPFN GPU rank4 member (rejected)

User-reported Colab CUDA run; decision JSON already recorded in
`outputs/sprint17/tabpfn_decision.json`. Frozen 10,000-row cap, 305 features,
50/50 rank4 mix. Screen gains: March +0.002788, April +0.005378,
May -0.004763, July -0.001907, August -0.000389. Mean +0.000222 versus
required +0.004; minimum -0.004763 versus required -0.002: both failed.
First fit 453.7 seconds; full screen 2441.8 seconds. Evaluation folds and
December were not fitted; no export. The prediction NPZ caches have not
been returned locally. This rejects the registered configuration only.

## E14 — FT-Transformer GPU rank4 member (rejected)

Pre-registration: `experiments/sprint17/PROPOSAL_E14.md` and
`outputs/sprint17/proposal_fttransformer.json`. User-reported Colab Tesla T4
screen decision: mean -0.002880853253247606, minimum -0.008958293130844397;
failed both frozen criteria. Gains: March -0.0028525382333110527,
April -0.008958293130844397, May +0.0018290508449827492,
July -0.0035499983097287413, August -0.0008724874373365887.
Original five fit times totaled 683.2 seconds; 0.3 seconds in the pasted
final JSON was a cached rerun. June/September and December not fitted; no
submission exported. Prediction caches remain on the user's Drive.
Feature-layer investigation follows in `experiments/feature_rebuild/`;
new features constructed there have no candidate-model scores yet.

## Status

Keep `submissions/sasol_sprint16_r4full_vR.csv`. No sprint 17 file was written.
E1-E14 all rejected (including the Colab TabPFN and FT-Transformer screens): tune/transform variants, four estimator families (XGB,
TabM, GRU sequence, decision-layer stacking/EM), two raw-field feature
sources (price/discount/novelty +0.0005, pretrained embeddings -0.001), and
the December transfer probes of the two least-negative candidates (pricefeat
-0.0021, grumix -0.0056 public). Neural arms in total: MLP and scaling
(sprint 16), TabM (-0.024), GRU sequence (-0.009); FT-Transformer infeasible
on this CPU. Post-close note: the operational reminder to select
`sasol_sprint16_r4full_vR.csv` plus `blend_vR` for private scoring was not
executed; the actual outcome is recorded below.

## Final outcome (2026-10-19)

Competition closed 18 Oct 2026; the private leaderboard was revealed 19 Oct
2026. The private-scoring slots had been left on S0-era submissions: the
portfolio's final private score 0.302880883 (rank 116) is S0
(`submissions/sasol_phase05_frozen_lightgbm_submission.csv`, public
0.289392053) — a +0.01348883 transfer, 98.1st percentile of the 156-pair
field, and still 0.009842 below the top-10 cut (0.312723243). The sprint 16
pair (`r4full_vR` 0.313217559 + `blend_vR` 0.309967901) was never privately
scored: a selection error, not a modelling loss. Conditioned on the 156
same-submission public->private pairs, the intended pair's central projection
is mid-table (neighborhood median private 0.309721, ~rank 39; local linear
~0.310254, ~rank 34), with only 10.5-16.7% of comparable rows beating the
top-10 cut; the naive market-wide median transfer (rank ~5) is an optimistic
upper bound. Full record: `docs/final_outcome_2026-10-19/README.md`; curated
evidence `outputs/sprint17/final_private_leaderboard_2026-10-19.json` (raw
dump and reproducible script in the same directory). Scores provisional
until Zindi code review completes (stated 2026-11-08).
