# Feature layer construction — 18 October 2026

Construction accepted; predictive performance is untested.

- Typed original lines: 467,880, using the authoritative loader.
- Unique customer receipts: 365,638.
- Original-category receipts: 404,210.
- Cutoffs: all 14 entries in the frozen category-dynamics panel.
- Candidate columns: 21 shop cadence, 57 basket attachment, 366 category
  state, 28 coverage/ratios and 17 fine category/site = 489 separately stored.
- Storage: `outputs/feature_rebuild/`, including a portable Colab ZIP.
- No accepted feature cache, label generator or submission was modified.
- No models were fitted and no submission was exported.

## Verification

Synthetic cases covered returns, repeated basket totals, eligibility versus
qualifying activity, literal NA/Unknown, first-seen versus reactivated
categories, independent Other-mapped original categories, same-day versus
same-basket association, zero gaps, no history, exact cutoff exclusion,
future-row deletion and inconsistent receipt rejection.

All real receipts had consistent time/site/repeated totals. Signed integer
fuel, shop and category sums reconciled with the raw typed lines. All 70
feature arrays passed shape, identity, finite-value and ID-order checks.
336 original-category spend comparisons (24 categories x 14 cutoffs) and
category breadth matched the immutable baseline representation. December
attachment counts reproduced the independent raw-data audit: 1,944 mixed
receipts, 1,835 customers with fuel and shop purchases, 970 of those without
a mixed receipt. The consumer rejected reversed customer order.

At March, September and December, every feature in all five blocks was
exactly unchanged after all future lines were deleted and receipts rebuilt.
The original source/baseline/best-submission hashes remained unchanged.
Initial full construction: 156.7 seconds; peak resident memory 933,620 KiB.
Machine reports and source fingerprints are under `outputs/feature_rebuild/`.

## Next experiment

Preregister an Opportunity fit using only the shop-cadence block added to
the accepted 305 features (326 total), keeping the model/rows/seeds/blend/vR
fixed. Do not train all 489 columns at once or select blocks using June or
September outcomes. Construction passing is not model acceptance.

## Colab E1 prepared

The shop-cadence fitting proposal and self-contained 42.54 MB bundle are ready.
Checks: seven frozen baseline scores, purged splits, IDs/order and 326 columns;
synthetic-only LightGBM fit/cache, stage gates and 5,488-row unchanged-baseline
export. Runtime versions are pinned; cache identity binds code/data/versions.
No real-customer candidate fit was run locally. The completed Colab result is recorded below. Evidence: `outputs/feature_rebuild/shop_runner_verification.json`.

## E1 shop cadence — rejected (user-reported Colab result)

Evidence summary: `outputs/feature_rebuild/shop_cadence_reported_screen.json`.
Screen gains: March -0.0046533015, April +0.0007193479,
May -0.0041141156, July -0.0016048976, August +0.0003351933.
Mean -0.0018635547 (required >=+0.004), minimum -0.0046533015
(required >=-0.002): failed both. Five origin fits totaled 251.5 seconds;
run elapsed 256.7 seconds. June/September and December were skipped and
no submission was exported. The later shell exit 127 was from Python display
code pasted into bash, after the model completed, and does not invalidate
the saved result. Prediction NPZ files have not been returned locally.

## E2 basket attachment — prepared, not fitted

Pre-registration: `PROPOSAL_BASKET_ATTACHMENT.md` and
`outputs/feature_rebuild/proposal_basket_attachment.json`. Test only the
57 basket-attachment columns appended to the accepted 305 (362 inputs).
The head, row eligibility, seeds, package versions, purged round selection,
full refit, cached comparator, product, vR, timing and numerical gates are
unchanged from E1. No rejected shop-cadence columns are carried forward.
Separate archive/notebook and prediction caches prevent E1/E2 collisions.
The single-cell launcher uses Python streaming, with no bash magic cell.
Verification: all seven frozen baseline guards and input/split/schema checks;
synthetic-only fitting, cache identity, stage gating and CLV-preserving export.
No real-customer E2 candidate fit has run locally; no E2 scores exist yet.

## E2 basket attachment — rejected (user-reported Colab result)

Evidence summary: `outputs/feature_rebuild/basket_attachment_reported_screen.json`.
Screen gains: March -0.0023547435, April -0.0012965283,
May -0.0041895943, July +0.0012773806, August +0.0013453871.
Mean -0.0010436197 (required >=+0.004), minimum -0.0041895943
(required >=-0.002): failed both. Five origin fits totaled 272.9 seconds;
run elapsed 275.1 seconds. June/September and December were skipped, no
submission exported. The duplicated pasted JSON describes one completed run;
Drive-already-mounted was informational. Prediction caches remain on Drive.

## E3 category state — prepared, not fitted

Pre-registration: `PROPOSAL_CATEGORY_STATE.md` and
`outputs/feature_rebuild/proposal_category_state.json`. Test the 366 frozen
category-state columns plus the original 305 (671 inputs), with no E1/E2
features. Preserve E2's model/head, row eligibility, seeds, pinned packages,
purged inner selection/full refit, cached comparator, product, vR and gates.
Distinct original categories retain their separate historical states and
spend requirements; no official labels/scales are regenerated or changed.
Separate ZIP/notebook and Python launcher preserve E1/E2 result caches.
Checks: all seven frozen baseline scores, IDs/schema/purged splits;
synthetic-only fitting/cache validation, stage gating and CLV/ID-preserving
export. No real-customer E3 fits or E3 scores exist yet.

## E3 category state — rejected (user-reported Colab result)

Evidence: `outputs/feature_rebuild/category_state_reported_screen.json`.
Mean gain -0.0017185330, minimum -0.0076681540; failed both frozen screen
criteria. March/April/May lost, July/August gained slightly. Fits 321.7s,
run elapsed326.5s. No June/September/December fits or submission export.
The repeated pasted JSON represents a single completed run.

## Scope change explicitly requested by user

After E1-E3 rejections, the user requested high-risk, materially different
last-hours experiments. A separate `experiments/moonshot/` batch combines
all794 numeric inputs, customer-ID GPU CatBoost, class weighting and a
17-candidate LambdaRank formulation. Six views are fixed before fitting.
Original promotion gates remain. A separate screen-only exploratory-export
rule is predeclared; probe files are never promoted models. No real-customer
moonshot fit has run locally. See that directory's README and frozen
`outputs/moonshot/proposal.json`.
