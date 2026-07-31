# Daily progress log

Append a dated entry for every work session that changes data interpretation,
code, experimental design, or results. Do not rewrite prior entries; append
corrections.

## 2026-07-31

### Started

- Reran the improved preflight successfully on the HP Z6 with both RTX 5000 Ada
  GPUs available.
- Confirmed 32 catalog cube IDs: 31 original reflectance cubes plus unmatched
  derived ID `field1_cube2`.
- Confirmed 18 cubes have complete reflectance, CSV labels, combined masks, and
  individual chickpea/weed/soil masks.
- Confirmed Cube 24 and Cube 28 label CSVs are exact fingerprint duplicates.
- Identified mask-version ambiguity: Cube 31 has two chickpea and two weed masks;
  Cube 36 has five soil masks. A single authoritative mask per class must be
  selected using registration and provenance checks.
- Added metadata-only QC for ENVI headers, cube byte sizes, wavelength coverage,
  TIFF masks, CSV schemas, and cross-cube duplicate detection.

### Next gate

- Run `scripts/run_data_qc.py` and review all generated QC reports before writing
  the full-band loader or training any model.

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
- Successfully ran the HP Z6 preflight in Python 3.11.15 and PyTorch 2.5.1.
- Confirmed CUDA access to two NVIDIA RTX 5000 Ada Generation GPUs.
- Cataloged 449 observed Field 1 files totaling approximately 18.12 GiB.
- Identified 31 original reflectance cubes totaling approximately 12.81 GiB.
- Confirmed the original reflectance collection contains 31 matching source-mask
  BIPs and 31 ENVI header pairs.
- Confirmed 18 cubes have both label CSVs and Emmanuel mask products:
  22, 24, 28, 30, 31, 32, 35, 36, 38, 39, 40, 45, 47, 49, 50, 54, 55, and 56.
- Identified 13 full-reflectance cubes without the curated label-table/mask pair:
  12, 14, 15, 16, 23, 27, 29, 37, 46, 51, 57, 58, and 59.
- Detected that the Cube 24 and Cube 28 label CSVs have identical file size and
  fingerprint; this must be checked for an accidental duplicate before training.
- Observed an unmatched derived cube ID 2, which requires filename/source review.
- Improved cube-ID parsing to recognize Resonon filenames containing `GigE_XX`.

### Pending for next session

- Run full project preflight and review local inventories.
- Rerun the improved preflight to produce corrected cube IDs, cube coverage, and
  duplicate-candidate reports.
- Verify whether Cube 24 and Cube 28 label CSVs are intentionally identical.
- Inspect wavelength vectors, band counts, NoData, CRS, mask values, and spatial
  alignment.
- Define Field 1 spatial blocks before any model split.
- Implement the authoritative full-band dataset loader after metadata review.

### Decisions frozen

- Primary external validation is inductive: fit on Field 1, deploy frozen model
  to Field 2, then reveal RTK tall-grass coordinates.
- Five weed groups are a biological prior, not a forced clustering result.
- Cluster outputs are candidate weed phenotypes until botanically validated.
