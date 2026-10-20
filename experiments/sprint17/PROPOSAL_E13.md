# Sprint 17 E13 — TabPFN rank4 member (Colab GPU)

Pre-registered 2026-10-17 before any Colab run. Machine copy:
`outputs/sprint17/proposal_tabpfn.json`.

The last model class our hardware blocked: a pretrained tabular foundation
model. Executed on free Colab (compute platform only). Frozen design: TabPFN
on the 4-class rank collapse, trained on all monthly origins <= cutoff-3m
truncated to the most recent <= 10,000 rows (API limit), no sample weights,
mixed 0.5/0.5 with the LGBM rank4 std member, frozen vR decode.

Gates identical to E9/E11: screen mean >= **+0.004**, min >= **-0.002**;
folds each >= **+0.0015**, mean >= **+0.004**; December export only on a
full pass (`submissions/sasol_sprint17_tabpfnmix.csv`, CLV byte-identical);
if exported, the usual +0.003 public-probe promotion rule applies. Abort if
the March member takes > 20 minutes.
