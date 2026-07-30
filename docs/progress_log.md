# Daily progress log

Append a dated entry for every work session that changes data interpretation,
code, experimental design, or results. Do not rewrite prior entries; append
corrections.

## 2026-07-30

### Completed

- Reviewed the original single-cube denoising-autoencoder code.
- Identified channel-misalignment, spatial-leakage, all-pixel clustering, forced
  cluster-count, and validation limitations.
- Created a reproducible PyTorch/CUDA project scaffold.
- Added data inventory, manifest validation, and image/mask inspection tools.
- Confirmed Field 1 contains available masks and development data.
- Confirmed Field 2 is untouched and contains independent RTK locations for one
  tall-grass weed class.
- Identified four Field 1 reflectance acquisition directories.
- Confirmed first and second derivatives were calculated across PCA components;
  reclassified these as legacy ablation inputs rather than spectral derivatives.
- Established Field 1 development and locked Field 2 external-validation roles.
- Prohibited synthetic observations in all reported experiments.

### Pending for next session

- Run full project preflight and review local inventories.
- Resolve cube IDs across reflectance, masks, and CSV tables.
- Inspect wavelength vectors, band counts, NoData, CRS, mask values, and spatial
  alignment.
- Define Field 1 spatial blocks before any model split.
- Implement the authoritative full-band dataset loader after metadata review.

### Decisions frozen

- Primary external validation is inductive: fit on Field 1, deploy frozen model
  to Field 2, then reveal RTK tall-grass coordinates.
- Five weed groups are a biological prior, not a forced clustering result.
- Cluster outputs are candidate weed phenotypes until botanically validated.
