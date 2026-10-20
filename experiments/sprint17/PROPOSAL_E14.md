# Sprint 17 E14 — GPU FT-Transformer (Colab)

Pre-registered 2026-10-18 before any candidate fit. Machine specification:
`outputs/sprint17/proposal_fttransformer.json` (included in the bundle).

E13 TabPFN failed its frozen screen (mean +0.000222, worst -0.004763).
E14 tests a different, never-fitted architecture: the authors' numerical
FT-Transformer. Local CPU timing previously prevented this experiment.
GPU availability enables the test; no improvement is assumed.

Two attention blocks, width 96, eight heads, all 305 features, no row cap;
AdamW 1e-4, batch 128, at most 30 epochs, patience five, seed seven.
Use only released data. At outer T, inner validation is T-3 months and
training origins end at T-6 months, with no label overlap. Imputation and
scaling use inner training only. Use the best inner-validation checkpoint,
without a later refit. Recency-weighted four-class cross entropy; frozen
50/50 mix with the existing LGBM rank4 member and frozen vR decoding.

Screen March/April/May/July/August: mean >= +0.004 and each >= -0.002.
Only after screen pass, evaluate June/September: mean >= +0.004 and each
>= +0.0015. Abort immediately if any fit exceeds 20 minutes. No parameter
or batch-size fallback after seeing results. December export only after
both gates pass, preserving the source IDs and CLV strings.

`build_fttransformer_bundle.py` reuses the existing E13 plain-array archive;
it does not fit or regenerate accepted models. The new archive includes a
runnable notebook, hash manifest, proposal, runner and frozen source data.
`--check` verifies hashes, baseline scores and purged splits without torch.
Completed origins are cached on Drive with input/code/version identities.

Implementation reference:
https://github.com/yandex-research/rtdl-revisiting-models/tree/main/package
