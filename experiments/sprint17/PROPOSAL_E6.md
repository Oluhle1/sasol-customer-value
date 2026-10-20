# Sprint 17 E6 — price/discount/novelty feature block

Pre-registered 2026-10-17, before any fit or feature computation. Machine copy:
`outputs/sprint17/proposal_price_features.json`. The feature list is frozen;
after the first screen nothing may be added, removed or redefined.

## Why this is the last real lever

Every sprint 16/17 rejection was a new model or a new combine on the **same 305
columns**. The raw file carries fields the panel never touches:

- `TotalModifierAmount`: discount lines are 6.2% of all lines and concentrate
  in exactly the adoptable categories (Beverages 17619, Food & beverage 4164,
  Confectionary 2958, Snacking 2207). The panel has zero discount features.
- `StockItemId`: product-level novelty (has this customer ever bought this
  product?) is absent; the panel knows category recency only.
- `UnitSellingExclPrice`: the panel has four price features total, none
  per-category, none recency-windowed.

The hypothesis: ramp-up (growth) and trial (adoption) show up as discount
response and product novelty before they show up in category rand, which is
what the big three classes are decided on.

## Frozen block (23 features)

Global: mod line share 90d, mod intensity 90d/30d/trend, item entropy 90d,
new-item share 90d. Fuel (Sale_Fuel): discount share 90d, mean R/l 30d, price
change 90 vs prior 90, distinct grades 90d, grade delta. Per category
(Beverages, Food & beverage, Confectionary, Snacking, Ice-cream, Tobacco):
new-item share, price change. Exact definitions in the proposal JSON.

All windows are strictly before the cutoff (90d = `[T-90d, T)`); adjustment
items are excluded; features align to the panel frame index. No labels, no
future transactions.

## Gates (round standard)

Screen on five production-like pool origins — March, April, May, July, August
2025 (March gains a rank4 cache refit for both arms) — versus r4full+vR:
mean gain ≥ **+0.004** and no origin worse than **−0.002**.

Only if the screen passes: June and September, each ≥ **+0.0015**, mean
≥ **+0.004**. Only then is a December file written
(`submissions/sasol_sprint17_pricefeat.csv`, CLV copied byte-identical from
the current best, audit JSON recorded).
