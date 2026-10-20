# Sprint 17 E2 — inactivity comeback override

Pre-registered after E1 failed, and before this rule is scored. The machine
copy is `outputs/sprint17/proposal_comeback_rule.json`. Do not edit the grid
or the gate after seeing scores.

## Why this is not E1 again

E1 retrained rank4 on weekly fuel rand and lost on every pool origin. The
weekly path matches the class the production model already predicts. The
residual that still replicates from April–May onto July–August is different:
customers called Inactivity who are truly Fuel growth have a shorter fuel
recency and a higher BG/NBD fuel expectation than customers who really are
inactive. Cohen's d on the pool is about -0.5 for recency and +0.5 for
`fuel_bgnbd_z`.

This experiment does not refit a model and does not add features. It overrides
the frozen r4full+vR label on that slice only.

## Frozen rule

Start from production r4full decoded with vR. Replace Inactivity with
Existing-category growth: Fuel when all of these hold:

- fuel_recency is at or below T
- fuel_bgnbd_z is at or above B
- if the grid point says so, Fuel is the second-ranked class

T, B, and the second-class switch are the grid in the JSON. The selected
point is the one with the highest minimum pool-origin gain, then the highest
mean gain. No other point is scored on June or September.

## Gates

Same screen and export bar as E1. Pool origins are April, May, July, and
August 2025. Mean weighted-F1 gain at least 0.004 and no origin worse than
-0.002. June and September are scored only for that one selected point, and
only if the screen passes. Each must gain at least 0.0015 and the mean must
gain at least 0.004 before a December file is written. CLV stays on
`submissions/sasol_sprint16_r4full_vR.csv`.
