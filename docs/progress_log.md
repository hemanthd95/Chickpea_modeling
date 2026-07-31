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

### Metadata-QC findings

- All 31 reflectance cubes are readable 150-band BIP arrays stored as uint16,
  little-endian, with correct byte sizes and matching data files.
- All reflectance headers report 150 wavelengths spanning 389.64–1030.82 nm and
  include ENVI map information.
- All 31 one-band source-mask BIPs match their reflectance cube dimensions and
  byte sizes.
- All 18 label CSVs share the same 17-column schema, but the Cube 24 and Cube 28
  tables are exact duplicates despite different image dimensions (1575×1017 vs
  1267×777); at least one association is invalid.
- Four combined-mask files (Cubes 31, 36, 38, and 39) are small, ungeoreferenced,
  four-band 8-bit images and are excluded from analytical mask selection.
- Four chickpea masks (Cubes 31, 36, 38, and 39) contain 0–255 gradients rather
  than binary values; Cube 31 also has a valid aligned binary alternative.
- Every soil and weed TIFF inspected is binary. Fourteen combined masks are
  single-band categorical rasters with values 0, 1, 2, and 3.
- Added a deep audit to count the full Label columns and measure label-to-mask
  overlap without assuming the meaning of codes 1, 2, and 3.

### Label-table audit correction and findings

- The 18 CSVs are labeled-pixel feature tables, not full-raster tables; therefore
  their row counts should not equal total cube pixels and cannot be reshaped into
  spatial label maps.
- Across the 18 table associations, 14,336,210 labeled rows were found: label 1
  = 7,491,236 (52.25%), label 2 = 1,440,944 (10.05%), and label 3 = 5,404,030
  (37.69%). The smallest class is consistent with chickpea scarcity, but mapping
  is not accepted until mask counts verify it.
- Similar row and class counts occur for Cube pairs 24/28, 30/32, and 35/56.
- Correction: the catalog's earlier lightweight fingerprint included file
  modification time. It was a fast change detector, not definitive proof of
  byte-identical content. Full SHA-256 hashing is now required for label-table
  duplicate conclusions.
- Revised the deep audit to compare label-class totals against observed binary
  mask pixel counts and to compute full SHA-256 hashes for all label CSVs.

### Provenance resolution

- Full SHA-256 confirmed three byte-identical CSV pairs: Cube 24/28, Cube 30/32,
  and Cube 35/56.
- Mask-count agreement assigns the valid copies to Cubes 24, 32, and 35,
  respectively. CSV associations for Cubes 28, 30, and 56 are excluded until
  regenerated from their own data.
- Confirmed label mapping: 1 = soil, 2 = chickpea, and 3 = weed. Chickpea and weed
  mask counts agree exactly for most non-duplicated tables; soil masks include
  additional valid/background-adjacent pixels and therefore differ modestly.
- Cube 31 generated and GPS weed masks are not identical (IoU 0.811867). The
  generated mask exactly reproduces the CSV count and is the default for baseline
  reproducibility; the GPS mask is retained as a sensitivity analysis.
- Encoded decisions in `configs/data_decisions.yaml`; no source file was changed.
- Added RGB/mask preview generation for visual QA before authoritative-manifest
  creation.

### Visual-QA interpretation

- Generated observed-band RGB previews for all 18 labeled cubes.
- Fourteen cubes supplied three valid binary mask files; Cubes 36, 38, and 39
  supplied valid weed/soil masks only; Cube 31 supplied the selected masks plus a
  sensitivity weed variant.
- The first montage used contours around pixel-level masks, producing excessive
  visual density. It is retained as a diagnostic but is not treated as proof of
  mask quality.
- Added authoritative-manifest construction and full-resolution quantitative
  checks for class-mask overlaps, missing masks, and unclassified pixels.

### Authoritative-manifest findings

- Built a 31-cube manifest: 18 weed-SSL-ready cubes, 15 three-class
  raster-supervised-ready cubes, and 15 valid legacy label tables.
- Fifteen three-mask cubes contain class overlaps; none contains triple overlap.
- Across labeled-mask unions, 547,817 of 13,734,935 pixels overlap (3.99%).
  Per-cube overlap ranges from 1.73% to 10.40% among three-mask cubes.
- The soil-mask pixel excess closely tracks overlap on trusted CSV associations,
  suggesting the historical label-table construction gave vegetation priority
  over soil. Pair-specific overlap measurement was added before freezing this
  precedence rule.

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
