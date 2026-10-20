# Feature rebuild E1 — shop cadence

Pre-registered 18 October 2026 before any real-customer candidate fit.
Machine proposal: `outputs/feature_rebuild/proposal_shop_cadence.json`.

Test exactly the 21 shop-cadence columns appended to the original 305
features. Use the production LightGBM rank4 recipe, original float64 baseline
values, seeds 7/23, recency weights and full eligible-history refit after
purged inner early stopping. Candidate rank4 probabilities replace the
baseline rank4 head in the existing full category-dynamics product, with
frozen vR; there is no additional rank4 blend or multiplier search.

Screen March/April/May/July/August: mean gain >=0.004 and each >=-0.002.
Only after passing, evaluate June/September: mean >=0.004 and each >=0.0015.
December export only after both pass; copy official IDs and original CLV
strings from the best submission. Stop any origin taking over 20 minutes.
No feature/parameter fallback after results. No baseline refits.

A new portable bundle contains float64 original inputs (rather than the
float32 neural bundle), only the 21 candidate features, cached baseline
probabilities, frozen vR, proposal and scripts. Package versions are pinned
to numpy 2.4.4, scikit-learn 1.8.0 and LightGBM 4.6.0. CPU deterministic
column-wise single-thread execution is fixed. Pinning and input preservation
reduce platform differences but do not guarantee bit-identical training
across compilers/operating systems. Archived baselines are reproduced from
saved probabilities, not freshly fitted on Colab.

Preparation verification reproduced seven baseline scores without fitting;
synthetic-only tests exercised the fitting, resume and gate paths and
verified CLV/ID-preserving export. No actual candidate scores exist yet.
