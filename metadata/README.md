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

- `local/annotations/chickpea_regions/`: investigator row-region polygons saved
  as synchronized JSON, vertex CSV, and EPSG:32617 GeoJSON. These are uncertain
  spatial eligibility priors, never direct chickpea labels.
- `local/reports/mask_refinement/chickpea_region_annotation_qc/`: polygon export
  reconciliation, normalized geometry QC, observed-footprint clipping, alley
  overlap, 0–0.15 m edge-tolerance sensitivity, and per-cube/all-cube PNG review.
- `local/contracts/field1_chickpea_region_annotation_qc_contract.yaml`: hashed
  audit record confirming that no authoritative mask or model was changed.

- `local/reports/mask_refinement/polygon_guided_candidate_profile/`: polygon-first
  observed NDVI support, green/red/red-edge/NIR feature profiles, per-cube PNGs,
  and an all-cube candidate overview. Historical masks are comparison-only.
- `local/contracts/field1_polygon_guided_candidate_profile_contract.yaml`: hashes
  the polygon-first evidence profile before any spectral separator or mask is
  materialized.

- `local/reports/mask_refinement/polygon_soil_source_audit/`: compares the
  failed raw-band NDVI diagnostic with provenance-tracked soil masks and inventories
  stored NDVI products inside investigator polygons. Unknown NDVI scaling is never
  inferred, and label-expansion cubes remain unresolved without validated soil evidence.
- `local/contracts/field1_polygon_soil_source_audit_contract.yaml`: hashes the
  read-only soil-source audit and confirms that no label or model changed.

- `local/reports/mask_refinement/polygon_guided_candidate_materialization/`:
  review-only categorical candidates built from investigator polygons, explicit
  alleys, provenance-tracked soil masks, and the validated stored-index rule for
  cubes 12/14/15. Core non-soil pixels remain vegetation candidates—not accepted
  chickpea labels—and every run writes per-cube and overview PNGs.
- `local/contracts/field1_polygon_guided_candidate_materialization_contract.yaml`:
  hashes candidate inputs and outputs while recording that authoritative masks
  and trained models were not changed.

- `local/reports/mask_refinement/polygon_guided_vegetation_separability/`:
  cube-balanced weak-reference spectra, leave-one-cube-out transfer metrics,
  per-cube review-score summaries, and individual/all-cube PNG score maps. The
  scores are diagnostics only and are never categorical chickpea labels.
- `local/contracts/field1_polygon_guided_vegetation_separability_contract.yaml`:
  records the frozen reference roles, transfer gate, band selection, and the
  guarantee that no mask or supervised model changed.
- `local/annotations/vegetation_references/` contains investigator-
confirmed chickpea and weed reference points collected after the historical
mask weak-reference transfer gate failed. These are spectral calibration
references, not replacement masks. The synchronized JSON, point CSV, and
GeoJSON are generated by `scripts/run_vegetation_reference_annotator.py`.
- `local/reports/mask_refinement/investigator_vegetation_reference_audit/`:
  point-level spectral QC and leave-one-cube-out chickpea/weed transfer
  diagnostics. Each annotation circle is reduced to one median spectrum after
  soil and NoData exclusion, preventing neighbouring pixels from being treated
  as independent reference observations.
- `local/contracts/field1_investigator_vegetation_reference_audit_contract.yaml`:
  hashes the clean-reference inputs and audit outputs while confirming that no
  mask, probability raster, supervised model, or Field 2 data changed.
