# Feature rebuild E3 — category state

Pre-registered before any real-customer E3 candidate fitting. Machine proposal:
`outputs/feature_rebuild/proposal_category_state.json`.

E1 shop cadence and E2 basket attachment failed their screens. E2's
user-reported mean was -0.00104362, minimum -0.00418959. Neither rejected
feature block carries forward.

Test exactly the frozen 366-column category_state block plus the original
305 float64 inputs (671 total). This covers 15 fields for each of 24 original
included categories and six customer-level category-maturity counts.
The state features distinguish first observed, reactivated and newly dormant
categories; retain qualifying purchase counts/days, signed recent/prior spend,
recency and known historical growth requirements. They are features only:
no label rules, original-category competition or target scales are changed.
Some definitions existed in the original Phase03 pipeline; this tests their
incremental value against the currently accepted 305-feature representation.

Keep E2's LightGBM head, seeds 7/23, row eligibility, purged inner early
stopping/full refit, cached comparator, frozen category probabilities and vR.
Screen March/April/May/July/August: mean >=+0.004 and each >=-0.002.
Only after screen pass: June/September each >=+0.0015 and mean >=+0.004.
Only after both: December export, source IDs/order and CLV strings preserved.
Abort an origin over 20 minutes. No feature selection or parameter fallback.
No accepted-baseline refits or local real-customer candidate fits.

Separate ZIP, Drive folder, proposal, manifest, decision and prediction
cache names prevent collisions with E1/E2. Versions remain numpy 2.4.4,
scikit-learn 1.8.0 and LightGBM 4.6.0; CPU runtime is sufficient. The launcher
uses a single normal Python cell and streams the training process.
