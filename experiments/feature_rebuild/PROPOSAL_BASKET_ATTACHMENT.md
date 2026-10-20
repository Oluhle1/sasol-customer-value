# Feature rebuild E2 — basket attachment

Pre-registered before any real-customer E2 candidate fit. Machine proposal:
`outputs/feature_rebuild/proposal_basket_attachment.json`.

E1 shop-cadence was rejected (user-reported screen mean -0.00186355,
minimum -0.00465330). Do not carry its 21 features into this experiment.

Test only the frozen 57-column basket_attachment block added to the original
305 float64 features (362 total): same-receipt fuel/shop purchase counts and
shares, separate same-day association, basket breadth, mixed versus shop-only
tickets, recency and quarter changes. No new feature selection or thresholds.
Production rank4 LightGBM recipe, seeds 7/23, purged inner early stopping,
full eligible-history refit, cached baseline comparison, frozen category
probabilities and vR are identical to E1. No accepted baseline refits.

Screen March/April/May/July/August: mean >=+0.004 and each >=-0.002.
Only after screen pass: June/September each >=+0.0015 and mean >=+0.004.
Only after both: December export, official IDs/order and original CLV strings
preserved. Abort any origin taking over 20 minutes. No fallback tuning.

The self-contained archive and notebook are separate from E1, with distinct
Drive folder, proposal, manifest, decision and prediction cache names.
Versions remain numpy 2.4.4, scikit-learn 1.8.0 and LightGBM 4.6.0. CPU
runtime is sufficient. Stream training from Python rather than a bash magic
cell to avoid mixing result-display Python into shell instructions.
