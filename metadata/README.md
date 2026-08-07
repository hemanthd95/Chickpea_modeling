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

- `models/supervised_fivefold_balanced/`: resumable 45-run, five-fold ×
  three-architecture × three-seed balanced supervised benchmark, with isolated
  fold outputs, an aggregate PNG, and a hashed completion contract. It measures
  development stability and is not exhaustive held-out-mask evaluation.

- `reports/models/supervised_outer_evaluation/`: exhaustive nested Field 1
  supervised metrics, fold/cube/5 m-group confusion counts, cluster-bootstrap
  uncertainty, timing reports, and the final supervised PNG.
- `reports/models/supervised_outer_evaluation/architecture_paired_*.csv` and
  `supervised_outer_architecture_comparison.png`: paired 5 m-group uncertainty
  and fold-consistency audit for the frozen architecture comparison.


- `reports/overlap_sensitivity/`: native raster resolution, deterministic
  one-observation-per-ground-cell support, repeated-view removal, cross-cube label
  disagreement, and PNG visual QC before overlap-sensitivity inference.
- `contracts/field1_ground_cell_overlap_audit_contract.yaml`: hashes the
  predeclared 1× and 2× native-GSD support audit; it is not a performance result.


- `reports/overlap_sensitivity/overlap_label_*.csv` and
  `overlap_label_consistency_overview.png`: repeated-view class-pair,
  boundary/interior, fold, and 5 m-group diagnostics. These reports diagnose
  mask/georegistration consistency and never modify authoritative labels.


- `reports/models/supervised_overlap_sensitivity/`: native-GSD, one-view
  frozen-model evaluation for both deterministic first-view and unanimous-label
  populations, with fold/group/cube confusion counts and PNG visual QC.
- `contracts/field1_supervised_overlap_sensitivity_contract.yaml`: hashes the
  completed overlap sensitivity and records that no retraining or model selection
  occurred.


- `reports/mask_visualization/chickpea_only/`: one authoritative chickpea-only
  PNG per Field 1 cube.
- `reports/mask_visualization/source_vs_authoritative/`: source chickpea masks
  beside model-ready masks and precedence removals.
- `reports/mask_visualization/field1_chickpea_masks_overview.png` and
  `field1_chickpea_mask_summary.csv`: all-cube investigator overview and
  full-resolution mask statistics.


- `local/reports/planter_geometry_evidence/`: read-only PCA/derivative evidence
  audit for row-parallel structure and numbered nonparallel planter-turn/wheel
  candidates. Geometry evidence is not a class label.
- `local/contracts/field1_planter_geometry_evidence_contract.yaml`: hashes and
  guardrails for the planter-geometry review.

- `local/reports/standardized_planter_turn_audit/`: consistent reflectance-PCA transverse-band evidence, per-cube profiles, and candidate tables. This replaces the overly permissive generic nonparallel-line candidates for decision making.
- `local/contracts/field1_standardized_planter_turn_audit_contract.yaml`: immutable audit record for the standardized turn-band review; no masks or models are changed.

- `local/annotations/planter_tracks/`: investigator-drawn tyre tracks, alley boundaries, and uncertain structures. The browser annotator saves JSON, vertex-level CSV, and GeoJSON; these are review evidence and never automatic class labels.
