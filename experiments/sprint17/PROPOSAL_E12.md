# Sprint 17 E12 — public transfer probe (pricefeat, grumix)

Pre-registered 2026-10-17 before any export. Machine copy:
`outputs/sprint17/proposal_transfer_probe.json`.

The local gate rejects effects below +0.004 because the public split (30% of
test, SE ~0.002-0.003) cannot resolve smaller differences anyway. This probe
tests that logic directly on the two least-negative rejected candidates:

- `pricefeat` (E6): local +0.0005, mixed sign across origins
- `grumix` (E11): local -0.0090, neutral at August

Both get December-only probe exports with CLV copied byte-identical. Decision
rule: promote only on a public delta >= +0.003 vs 0.313217559; anything less
is recorded as noise and the selection stays `r4full_vR` + `blend_vR`. No
re-parameterization after seeing scores.
