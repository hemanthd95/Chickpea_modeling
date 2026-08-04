# Metadata layout

`metadata/local/` is workstation-generated and excluded from Git because it may
contain absolute paths or data-derived records. Keep two kinds of output separate:

- Canonical pipeline contracts remain directly in `metadata/local/` when existing
  scripts depend on their stable names, for example `authoritative_manifest.csv`.
- Reviewable stage outputs belong in `metadata/local/reports/<stage>/`.

Current report stages:

- `spatial_grouping/`: cube footprints, overlap, group membership, and group-level
  authoritative class counts, accompanied by a PNG visual-QC overview.
- `normalization/`: fold-specific training-only spectral statistics and PNG
  diagnostics. The hashed normalization contract remains under `contracts/`.
- `sampling/`: leakage-aware observed training-centre summaries and visual QC.
  The indexed centre pool is a training contract, never a replacement for
  exhaustive held-out evaluation. Candidate patches containing internal
  georectification NoData are excluded and reported by cube.
- `patch_qc/`: center-label, patch-shape, normalization checks, and an observed
  NIR–red–green patch montage for investigator review before model training.
- `gpu_benchmark/`: independent hardware-throughput diagnostics for each GPU;
  these engineering measurements are not model results.
- `models/supervised_smoke_fold1/`: bounded diagnostic learning curves,
  classification report, and confusion matrix. These are pipeline checks, not
  primary five-fold results.
- `models/supervised_convergence_fold1/`: larger Fold-1 convergence diagnostic,
  per-patch probabilities, early-stopping history, and observed patch-purity error
  analysis. This stage remains diagnostic and cannot replace five-fold results.
- `models/supervised_architecture_ablation_fold1/`: controlled three-seed
  comparison of center-spectrum, spatial-average, and fused center-context
  classifiers, including class and worst-cube metrics plus visual QC.

Accepted local analytical contracts are written to `metadata/local/contracts/`.
Each contract includes hashes and the visual artifact reviewed by the investigator.

Major analytical stages should emit a clearly named PNG preview alongside their
tabular reports whenever spatial or statistical structure can be meaningfully
visualized. A PNG supports investigator review but never replaces numeric QC.

Raw imagery and source masks never belong in metadata directories. Report folders
may be deleted and regenerated from the documented scripts.

- `contracts/field1_supervised_architecture_contract.yaml`: hashed Fold-1
  architecture decision freezing center-context fusion as primary and retaining
  spatial-average and center-spectrum supervised baselines before five-fold work.
