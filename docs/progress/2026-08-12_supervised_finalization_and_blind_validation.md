# Supervised finalization and blind-validation gate — 2026-08-12

## Authorized scope

Complete Phases 0–4 only: freeze the immutable Field 1 benchmark at
`da95fd59b7dc5d8ba4add96ed43b398bb7c93749`, audit grouped spectral and
center/context reliance without retraining the nested models, freeze Field 1
deployment ensembles, and write an interim supervised report. Stop at the
Field 2 compatibility gate.

Field 2 is reflectance-only, is not georectified in the authorized data section,
and remains a hard no-access domain. No Field 2 image, header, scientific array,
coordinate, label, mask, preview, normalization statistic, sample, or prediction
may be opened or produced in this work. Metadata from the pre-existing locked
archive inventory may be used only to establish that it is not an authorized,
analysis-ready data source. Cubes 24 and 28 remain a previously inspected Field
1 early-growth sensitivity set and cannot substitute for blind Field 2 data.

## Startup record

- Branch: `agent/ssl-multicube-setup`
- Starting HEAD: `da95fd59b7dc5d8ba4add96ed43b398bb7c93749`
- Upstream divergence at startup: 0 behind / 0 ahead
- Required annotated tag was absent and was created at the exact starting SHA:
  `supervised-confident-v1-da95fd5`
- No repository `AGENTS.md` applies.
- Protected pre-existing working-tree entries:
  - modified: `train_chickpea_classifier.py`
  - untracked: `chickpea_modeling_before_cuda_fix.yaml`
  - untracked: `configs/data_manifest.csv`
  - untracked: `label_mapping.json`
  - untracked: `label_mapping_test.json`
  - untracked: `xgb_chickpea_model_test.json`
  - untracked: `xgb_chickpea_model_weighted.json`

These entries will not be modified, deleted, staged, or committed.

## Frozen scientific decisions

1. The nested Field 1 benchmark and its reported metrics are immutable. New
   interpretability results are post-hoc reliance diagnostics and cannot select
   bands, architecture, thresholds, epochs, or labels.
2. Grouped occlusion uses physical wavelength intervals of approximately
   10–15 nm. Primary replacement is the fold-training mean; interpolation from
   adjacent retained wavelengths is sensitivity analysis.
3. Center-only and context-only interventions are routed through the two frozen
   branches separately. They are destructive frozen-model reliance tests, not
   retrained center-only or context-only models.
4. Interpretability uses a deterministic class/fold/cube/spatial-group-stratified
   subset of the already frozen outer-test population. It preserves equal
   spatial-group aggregation and model-seed pairing. It is not substituted for
   the exhaustive benchmark metrics.
5. Benjamini–Hochberg adjustment is prespecified within each
   occlusion-method × route × metric family. Stable regions additionally require
   positive estimated degradation and positive direction in at least four of
   five outer folds.
6. Fixed spectral indices are descriptive only. Their formulas and nearest
   wavelengths are frozen before results are inspected; no thresholds are fit.
   A filename containing “NDVI” is never accepted as formula evidence.
7. Integrated gradients are omitted because the mandatory two-method, three-route
   grouped occlusion plus branch audit is already computationally substantial.
   This optional method is not needed to satisfy the primary interpretability
   question.
8. Deployment duration is frozen from Field 1 inner-validation selected epochs:
   median epoch 10 for with-alley and 11 for without-alley. Deployment uses the
   selected `center_context_fusion`, seeds 42–44, fixed cross-entropy and existing
   optimizer settings. No post-hoc probability calibration is fit; the frozen
   deployment output is the arithmetic mean of three uncalibrated softmax vectors.
9. The historical legacy result remains a historical internal comparator. A
   legacy all-Field-1 deployment model will not be fabricated because no frozen,
   scientifically equivalent reconstruction contract exists.

## Execution plan and gates

- [x] Record startup state and inspect repository instructions.
- [x] Verify the model input, band policy, label rules, nested contracts, tests,
  and metadata-only Field 2 availability state.
- [x] Create the immutable annotated benchmark tag.
- [x] Freeze and verify the benchmark contract and checkpoint manifest.
- [x] Run grouped wavelength occlusion and branch-reliance audits on CUDA.
- [x] Run fixed-index descriptive audit and generate publication figures.
- [x] Prepare eligible Field 1 deployment populations and normalization.
- [x] Pass two-GPU real-patch forward/backward/save/reload gate.
- [x] Train and freeze six deployment checkpoints using independent GPU workers.
- [x] Write the interim supervised report and exact Field 2 product checklist.
- [x] Validate the complete contract and checkpoint hash chain.
- [x] Stop with Field 2 compatibility status `blocked_not_georectified_or_authorized`.

## Completion findings

- All benchmark hashes and metrics matched the immutable commit.
- The grouped audit evaluated 38 physical wavelength groups, two occlusion
  methods, three routes, five outer folds, and three seeds on 3,016 deterministic
  observations from 136 spatial groups. No model was retrained.
- Mean replacement found broad corrected reliance, but adjacent-band
  interpolation retained stable macro-F1 evidence only at 401.84–414.07 nm,
  confirming substantial neighboring-band redundancy.
- Center neutralization decreased macro-F1 by 0.670249; context neutralization by
  0.294874; spatial averaging and shuffling context also caused nonzero losses.
- NDVI/GNDVI/NDRE/NIR-red were descriptive only. SAVI L=0.5 was withheld because
  unit-reflectance scaling provenance is absent.
- Six deployment checkpoints were frozen from eligible primary Field 1 only.
  Summed training time was 285.23 seconds; peak VRAM was 0.351 GiB.
- Field 2 remained completely unopened and unpredicted. Cubes 24/28 were not used
  as substitutes. Work stopped at the required compatibility gate.

No self-supervised learning is authorized in this task.
