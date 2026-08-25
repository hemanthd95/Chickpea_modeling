# Study protocol

## Research question

Can a self-supervised spectral-spatial representation learned from UAV
hyperspectral imagery recover stable, externally verifiable phenotypic subgroups
within pixels labelled broadly as weeds?

## Claims and terminology

Until botanical validation is available, outputs are called **latent weed
phenotypes**, **weed spectral-spatial groups**, or **candidate weed classes**.
Cluster IDs are not described as weed species.

## Data-use rule

Only observed project data are permitted. Synthetic pixels, synthetic spectra,
generated masks, simulated labels, and fabricated coordinates may not enter model
training, validation, testing, figures, or reported metrics. Normal stochastic
augmentation of observed patches is permitted and must be documented.

## Dataset roles

### Field 1: development

- Single confirmed acquisition date: 2025-05-06.
- Four directories dated 2025-06-06, 2025-06-15, 2025-06-23, and 2025-06-30
  are student processing batches, not separate acquisitions.
- Original approximately 150-band Resonon Pika-L reflectance cubes.
- Manual chickpea, weed, and soil information from mask products and CSV tables.
- Used for preprocessing decisions, model fitting, ablation studies, spatially
  blocked cross-validation, clustering choices, and supervised baselines.

### Field 2: locked external test

- No Field 2 labels, masks, or RTK tall-grass coordinates may influence the
  primary Field 1 pipeline.
- Freeze wavelengths, bad-band handling, normalization, patch size, encoder,
  clustering method, cluster number-selection rule, and cluster prototypes before
  revealing tall-grass validation.
- Apply the frozen pipeline to Field 2, then evaluate RTK point enrichment.

## Primary analysis

1. Validate ENVI metadata, wavelength vectors, spatial dimensions, CRS, NoData,
   and mask registration.
2. Remove invalid/bad bands using documented sensor/data criteria.
3. Fit normalization parameters using training blocks only.
4. Sample observed vegetation-centred patches while tracking cube, processing
   batch, plot assignment, and spatial block.
5. Train a contrastive spectral-spatial encoder on Field 1.
6. Cluster only centre pixels belonging to the Field 1 weed mask.
7. Choose clustering complexity using stability, not a forced five-cluster answer.
8. Freeze the pipeline and deploy it to Field 2.
9. Reveal RTK tall-grass points and calculate cluster enrichment with spatially
   appropriate null sampling.

## Required comparisons

- Raw reflectance clustering without SSL.
- PCA clustering without SSL.
- Original denoising-autoencoder concept with corrected aligned augmentation.
- Contrastive SSL using full reflectance.
- Traditional supervised models for chickpea/weed/soil.
- Supervised spectral-spatial model with imbalance-aware loss.
- Center-spectrum, spatial-average, and fused center-context supervised
  architectures on identical observed samples and spatial folds.

## Legacy PCA-derivative products

First and second differences calculated across PCA components are retained only
as a legacy ablation. They are not spectral derivatives because component order
does not represent wavelength. Scientifically interpretable derivatives, if
used, must be calculated across wavelength-ordered reflectance after an explicit
smoothing protocol.

## Leakage controls

- No random pixel split.
- Because adjacent 30 m AGL swaths overlap at 7 m flight-line spacing, all
  observations of the same mapped ground area must share one split. Split IDs
  are assigned in map coordinates, not independently by cube.
- No neighboring-patch overlap across train/test spatial blocks.
- Architecture selection uses repeated seeds and considers macro-F1, chickpea F1,
  and worst-cube macro-F1; no single best-seed score may determine selection.
- No global normalization fitted using validation or Field 2.
- No cluster-number tuning against Field 2 RTK outcomes.
- No smoothing before primary spatial-coherence evaluation.
- Report both object/pixel counts and spatially clustered uncertainty.

## Geometric chickpea prior

- Confirmed Field 1 plot polygons define where planted chickpea may occur.
- The 1.5 m field-boundary tolerance is used only for field-identity QC and is
  never used as the chickpea constraint buffer.
- Apply plot constraints as a documented post-model refinement, preserving raw
  predictions for comparison.
- Test outward plot buffers of 0, 0.15, 0.30, 0.50, and 1.0 m using Field 1
  spatial validation only. No buffer is selected using Field 2.
- Outside the allowed plot region, suppress chickpea; assign weed only through
  the vegetation decision, otherwise retain soil/unlabelled status.

## Primary evidence

- Cluster stability across seeds and held-out Field 1 blocks.
- Transfer across held-out cubes, plots, and spatial blocks within Field 1.
- Spectral and spatial separability with uncertainty.
- Field 2 RTK tall-grass cluster enrichment.
- Sensitivity to patch size, band set, cluster method, and cluster count.
