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

### Mask precedence frozen

- Pairwise overlap totals: chickpea–soil = 502,998 pixels, chickpea–weed =
  44,819 pixels (Cube 24 only), and weed–soil = 0 pixels.
- Cube 24's chickpea–weed overlap exactly equals the excess in its raw chickpea
  mask relative to the trusted CSV, confirming the historical order.
- Frozen dynamic precedence: weed > chickpea > soil > unlabeled. Source masks
  remain unchanged.
- Added the production ENVI/mask data layer, spatial-block identifiers, and an
  observed-only patch dataset.
- Added deterministic profiling of up to 10,000 observed spectra per cube so
  bad-band decisions are derived from the 31 real cubes.

### Supporting archive inventory

- Metadata correction from the investigator: all imagery was acquired on
  2025-05-06. The four June folder dates record student processing, not repeated
  acquisitions. Catalog and manifest schemas now separate `acquisition_date`
  from `processing_batch`; no temporal-transfer claim will be made.
- Flight geometry confirmed as 30 m AGL, 1 m/s, with 7 m swath spacing. Adjacent
  cube overlap is expected and will be quantified rather than treated as a QC
  defect. Map-coordinate spatial groups will keep repeat observations of the same
  ground area in one fold to prevent cross-cube leakage.
- Plot polygons may belong to Field 1, Field 2, or both. They are not authorized
  as chickpea constraints until coordinate-based field assignment is complete.
- Once assigned, plot interiors provide a chickpea-support prior and the spaces
  between plot blocks provide alley exclusions. Outside-plot vegetation is not
  automatically relabelled as weed until registration uncertainty and boundary
  buffers are quantified.

- Safely extracted 520 supporting files into an isolated, ignored raw-data
  directory and recorded per-file SHA-256 hashes.
- Initially identified 36 Field 2/tall-grass records by explicit filenames.
  A subsequent provenance pass identified three additional Cube 34 GeoTIFF/XML
  records tied to the Field 2 workflow despite lacking `Field 2` in their names;
  the corrected lock contains 39 records. Their contents and coordinates were
  not interpreted.
- Confirmed all 81 observed shapefile datasets contain the required SHP, SHX,
  DBF, and PRJ components.
- Found 35 numbered plot groups: 10–26, 37–45, and 64–72. These are treated as
  an observed subset, not assumed missing plots, until the experimental design
  and combined-plot layer are inspected.
- Added a metadata-only supporting GIS audit. It inspects unlocked Field 1
  geometry, CRS, feature counts, and schemas while never opening locked or
  unassigned records (including Project 50/51 until their role is confirmed).
- Removed the optional Fiona dependency after the workstation check. The audit
  now reads SHP, DBF, and PRJ metadata directly and leaves the conda environment
  unchanged.
- The dependency-free audit inspected 75 Field 1 shapefiles and 41 Field 1 CSVs
  with no structural issues; 39 locked and 26 unassigned records stayed unopened.
- All 75 layers use WGS 84 / UTM Zone 17N. The combined layer contains 35
  polygons matching the observed plot IDs 10–26, 37–45, and 64–72.
- Plot GIS attributes contain geometry bookkeeping and area only; RTK CSVs contain
  coordinate/quality fields but no treatment, cultivar, or planting-date mapping.
- Added a focused audit for the newly observed Field 1 Cube 20 reflectance and
  competing mask variants before considering it for the authoritative manifest.
- Cube 20 reflectance is a valid 150-band uint16 EPSG:32617 cube (989×1517 at
  1.5 cm). One binary chickpea mask already matches this exact grid.
- Twelve initial dimension warnings were reinterpreted: most masks use a valid
  alternate 1092×1591 georeferenced grid and require nearest-neighbor reprojection;
  multiband/continuous RGB and desktop products are visualization-only.
- The non-desktop combined Cube 20 mask is categorical (0/1/2/3). Added aligned
  agreement tests against each binary variant before selecting or deriving masks.
- Corrected plot interpretation: current-cube coverage is not field identity.
  Named Field 1 boundary support and cube coverage are now reported separately;
  exact polygon containment remains required before geometric label refinement.

### Spectral profile and supporting archive

- Profiled 310,000 observed spectra from all 31 Field 1 cubes with zero sampled
  NoData values.
- Upper-edge saturation is severe: 0.86% at 869.34 nm, 1.54% at 873.77 nm,
  greater than 50% by 913.77 nm, and effectively complete above 972 nm.
- Frozen primary band set to indices 3–113 (111 bands, 401.84–869.34 nm), using
  the nominal 400 nm lower bound and a global saturation threshold below 1%.
- Retained sub-400 nm and extension through 882.63 nm only as declared
  sensitivity analyses.
- Added a safe, non-overwriting importer for `OneDrive_2026-07-31.zip`. It
  extracts to a separate data folder, hashes all files, and locks Field 2/tall-
  grass assets from content inspection during Field 1 development.

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
