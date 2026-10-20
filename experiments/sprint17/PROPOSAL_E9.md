# Sprint 17 E9 — TabM member

Pre-registered 2026-10-17, before any fit. Machine copy:
`outputs/sprint17/proposal_tabm.json`. Gates frozen.

The neural family on the 305 features failed in sprint 16 (E9/nn_scaling) with
conventional MLPs. TabM (batch-ensemble MLP with rank-1 weight adapters) is
the current tabular SOTA family and the only untested architecture class. It
enters the product as a member averaged 50/50 with the LGBM rank4 std head
(the E5-least-bad form): `p4 = 0.5*std + 0.5*tabm`, then the usual full
combination and frozen vR decode.

Architecture and training are frozen in the proposal JSON: k=8 members,
128-wide 2-layer trunk, rank-1 adapters on every weight, dropout 0.1,
AdamW 2e-3, batch 1024, ≤50 epochs, early stop 8 on previous-month logloss,
seed 7, per-cutoff standardization clipped at ±8, identical stacking blocks.

Screen: March/April/May/July/August mean ≥ +0.004, min ≥ −0.002. Folds:
June/September each ≥ +0.0015, mean ≥ +0.004. No fallback. Abort rule: if the
March member fit exceeds 15 wall-clock minutes the experiment is abandoned and
reported with timing only. December export only on a full pass.
