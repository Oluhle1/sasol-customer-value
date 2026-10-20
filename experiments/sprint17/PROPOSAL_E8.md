# Sprint 17 E8 — EM prior correction

Pre-registered 2026-10-17, before any run. Machine copy:
`outputs/sprint17/proposal_em_prior.json`. Zero fitting: cached arrays only.

The r4full product's predicted class marginal need not match the true label
prior. Saerens-Latinne-Decaestecker EM reweights each customer's 17-class
distribution toward a target prior and renormalizes; decode stays
`argmax(w * vR)` with frozen vR. Target prior: label frequencies at origin
O minus three months (outcome window fully observed at O), floored 1/17.

Gates (decision-layer bars, like E4): pool origins 03/04/05/07/08 mean gain
≥ +0.001 and minimum ≥ −0.001; then June/September each ≥ +0.001 and mean
≥ +0.002. December export only on a full pass.
