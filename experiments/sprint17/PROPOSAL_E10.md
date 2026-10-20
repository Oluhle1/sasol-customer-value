# Sprint 17 E10 — pretrained ItemName embeddings as features

Pre-registered 2026-10-17, before any download or computation. Machine copy:
`outputs/sprint17/proposal_item_embeddings.json`. Block frozen; no iteration
after the first screen.

## Why

Pretrained openly available models are allowed (AutoML services are not).
ItemName is the one raw field with semantics a GBM cannot see through
StockItemIds: products that mean similar things (flavours, categories, pack
formats) get nearby vectors. E9 (TabM) showed another architecture on the 305
columns fails; this changes the *inputs*, not the learner.

## Frozen design

sentence-transformers/all-MiniLM-L6-v2 (mean pooling, unnormalized), PCA(16)
fit on the 6,849 item vectors only (no labels, no transactions). An item
vector is the mean over its 17,777 distinct (id, name) rows — 2,742 ids carry
more than one name, so name-level averaging is part of the frozen mapping.
Counts verified before any computation (2026-10-17).
Per customer, strictly pre-cutoff, spend-weighted over purchased items:

- 16 dims: all-history mean  - 8 dims: 90-day mean
- `emb_shift_cos` = taste shift between windows
- `emb_recent_outlier` = how semantically new the recent items are
- `emb_norm` = engagement magnitude

27 columns total, joined to the 305 panel columns (332 features) and fed to
the unchanged production rank4 recipe.

## Gates

Screen on March/April/May/July/August: mean ≥ **+0.004**, min ≥ **−0.002**;
folds June/September each ≥ **+0.0015**, mean ≥ **+0.004**. No fallback.
December export (`submissions/sasol_sprint17_embfeat.csv`) only on a full
pass, CLV copied byte-identical.
