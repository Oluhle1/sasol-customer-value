# Sprint 17 E11 — GRU over item-purchase sequences

Pre-registered 2026-10-17, before building any tensor. Machine copy:
`outputs/sprint17/proposal_gru_seq.json`. Block frozen; no iteration after
the first screen.

## Why

Every neural member on the 305 aggregated columns lost to the LGBM head
(MLP and scaling in sprint 16, TabM -0.024 in E9). FT-Transformer on 305
tokens was measured infeasible on this CPU (>13 s/step on random data,
>30 min/epoch) and is not run. The input type never tried is the raw
transaction sequence with pretrained item semantics: order, cadence, and
MiniLM item geometry, compressed to PCA(16).

## Frozen design

Per customer, last **K=64** lines strictly before the cutoff (adjustment
items excluded, amount > 0), channels: item PCA(16), log1p(amount),
log1p(days-before-cutoff), is_fuel, plus a mask channel (20 total, frozen).
GRU(20 -> 96, one layer) -> dropout 0.1 -> Linear(96, 4) on the rank
collapse, trained exactly like E9 (temporal inner split, recency weights,
prev-month early stop patience 6, max 30 epochs, AdamW 2e-3, batch 256,
seed 7). Member mix identical to E9: `p4 = 0.5*rank4_std2 + 0.5*gru`;
frozen vR decode.

## Gates

Screen on March/April/May/July/August: mean >= **+0.004**, min >= **-0.002**;
folds June/September each >= **+0.0015**, mean >= **+0.004**. No fallback.
Abort if the March fit exceeds 20 minutes. December export
(`submissions/sasol_sprint17_grumix.csv`) only on a full pass, CLV
byte-identical.
