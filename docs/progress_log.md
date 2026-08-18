# Daily progress log

Append a dated entry for every work session that changes data interpretation,
code, experimental design, or results. Do not rewrite prior entries; append
corrections.

## 2026-08-14

### Field 2 georectified readiness inventory completed; annotation remains blocked

- Audited 40 authorized Field 2 cube identifiers across 120 ENVI headers and
  120 payloads using read-only memory maps and `GDAL_PAM_ENABLED=NO`.
- All 120 header/payload pairs passed one-to-one resolution and byte-size gates;
  independent before/after SHA-256 snapshots of all 240 source files were
  identical.
- All 111 frozen Field 1 model wavelengths matched every Field 2 reflectance
  cube within 0.00003 nm, with strictly increasing order and no duplicates.
- All 80 reflectance-to-PCA/index grid comparisons had identical CRS, affine
  transform, dimensions, GSD, bounds, and pixel centers. No resampling occurred.
- Generated deterministic sampled QC, 40 prediction-free review previews, four
  overview figures, footprint/overlap tables, and a detailed readiness report.
- PCA and stored-index products contain transformed nonzero values outside the
  reflectance zero-filled footprint. Because no explicit NoData value or aligned
  valid-data mask was supplied, all cubes remain blocked for actual annotation
  pending a frozen authoritative valid-support rule; biological roles remain
  `unreviewed`.
- No label, prediction, probability, model training, or SSL experiment was
  produced. The frozen Field 1 benchmark and checkpoints were unchanged.

## 2026-08-12

### Supervised benchmark finalized; Field 2 compatibility stop frozen

- Tagged immutable benchmark commit
  `da95fd59b7dc5d8ba4add96ed43b398bb7c93749` as
  `supervised-confident-v1-da95fd5` and verified all 15 primary nested checkpoint
  hashes, contracts, wavelengths, and reported metrics.
- Completed a Field 1-only grouped reliance audit over 38 physical wavelength
  groups, two occlusion methods, three branch routes, five folds, and three seeds.
  Adjacent-band interpolation left 401.84–414.07 nm as the only corrected stable
  macro-F1 region, emphasizing correlated-band redundancy.
- Frozen-model ablations established reliance on both branches: center and
  context neutralization reduced macro-F1 by 0.670249 and 0.294874 respectively.
  These are destructive reliance diagnostics, not retrained model comparisons.
- Fixed-index audit used exact 550.54/669.09/720.67/798.91 nm bands. SAVI L=0.5
  was not computed because unit-reflectance scale provenance is absent.
- Frozen three-seed with- and without-alley deployment ensembles using only 13
  eligible primary Field 1 cubes. Two-GPU real-patch checks and all six checkpoint
  reloads passed; no legacy deployment model was fabricated.
- Stopped at Field 2 status `blocked_not_georectified_or_authorized`: authorized
  paths are blank, the manifest has zero Field 2 rows, and no Field 2 header,
  scientific array, coordinate, label, or prediction was opened. Cubes 24/28
  remain previously inspected sensitivity data, never blind substitutes.
- Wrote the interim report and exact georectified-product checklist in
  `docs/progress/2026-08-12_final_supervised_model_report.md`.

### Confident-label nested supervised benchmark passed

- Confirmed PyTorch 2.10.0+cu130 sees exactly two NVIDIA RTX 5000 Ada Generation
  GPUs, then ran every repository Python command with the explicit project Conda
  interpreter. No DataParallel or CPU fallback was used.
- Materialized provenance-preserving Field 1 confident labels with soil,
  chickpea, weed, and unresolved states. Fold-specific vegetation separators
  exclude both Test and Stop folds. Cubes 24/28 remain phenology sensitivity,
  Cubes 12/14/15 remain positive-support expansion, and Cube 32 remains sparse
  transfer-warning evidence only.
- Froze patch-safe, 0.15 m thinned, ground-deduplicated nested samples and an
  exhaustive 5,924,107-observation primary evaluation population. Deterministic
  rerun, role separation, boundary safety, NoData, and Field 2 lock gates passed.
- Passed a real-data CUDA smoke test before full fitting. Independent workers on
  both GPUs completed 30 center-context checkpoints: five outer folds, three
  seeds, and with-/without-alley variants.
- On equal-weight held-out spatial groups, the with-alley ensemble achieved
  balanced accuracy 0.932545 and macro-F1 0.876089 versus 0.785344 and 0.648250
  for the frozen legacy model. Paired bootstrap improvements were +0.148134
  balanced accuracy and +0.228297 macro-F1, with both 95% intervals above zero.
- Alley enrichment materially improved recall and macro performance, but its
  weed-precision interval crossed zero. Worst-cube macro-F1 remains 0.431107 and
  Cube 32 remains an explicit failure/transfer warning. No Field 2 claim is made.
- Added task-specific configuration, label composition, data preparation,
  validation, dual-GPU training, paired evaluation, tests, and the full report in
  `docs/progress/2026-08-12_confident_label_supervised.md`. All 41 tests pass.

## 2026-08-04

### Multi-GPU stall isolated for hardware benchmarking

- The revised diagnostic again produced no completed batch after 20 minutes.
  Interruption occurred inside PyTorch `DataParallel.parallel_apply` during the
  model forward pass, after the DataLoader had supplied a batch.
- System memory was healthy (120 GiB available, no swap use), and no Python
  process remained after interruption. This evidence supersedes the provisional
  disk-cache diagnosis.
- Added a short independent per-GPU training-step benchmark to compare both RTX
  5000 Ada devices under deterministic and cuDNN-autotuned execution before
  changing the scientific training pipeline. Benchmark tensors are engineering
  inputs only and can never enter fitted models or reported scientific results.
- Both GPUs passed independently. At deterministic batch 256, GPU 0 processed
  134,976 and GPU 1 processed 176,473 engineering samples/s; peak allocated
  memory was only 0.106 GiB. Deterministic execution imposed no meaningful
  throughput penalty.
- Isolated `nn.DataParallel` as the failed execution path. The smoke test is now
  pinned to dedicated GPU 1 with batch 256. The full benchmark will use
  multi-process DistributedDataParallel only after the single-GPU data path is
  validated.
- The three-epoch Fold-1 smoke test completed successfully on GPU 1. Validation
  accuracy and balanced accuracy were 0.5192 and macro-F1 was 0.5188 on an
  exactly balanced 6,000-patch diagnostic set.
- Per-class F1 was 0.5197 soil, 0.5268 chickpea, and 0.5101 weed. Chickpea had
  the highest recall (0.5980) but lowest precision (0.4707), reflecting
  overprediction: 2,541 patches were predicted chickpea versus 2,000 true.
- The largest directional error was weed-to-chickpea (783/2,000; 39.15%).
  Soil recall was 0.4520, with soil divided almost evenly between chickpea and
  weed errors. These results support spectral-spatial ambiguity but do not yet
  test hidden weed phenotypes.
- Training and validation losses decreased through epoch 3 (0.9199 and 0.9701),
  with a small gap and no convergence. The smoke test therefore passes the
  pipeline gate but is prohibited from serving as the supervised baseline.
- Staged the next Fold-1 convergence diagnostic without overwriting the smoke
  test: 20,000 observed training and 5,000 observed validation patches per class,
  up to 20 epochs, minimum six epochs, and macro-F1 early stopping on GPU 1.
- Added per-patch class probabilities and a post-run observed-label audit that
  quantifies center-class fraction, unlabeled fraction, accuracy by purity bin,
  and cube-by-class performance. Field 2 remains locked and no synthetic sample
  can enter model fitting.
- The convergence diagnostic stopped at epoch 11 and restored epoch 6, where
  balanced accuracy was 0.6170 and macro-F1 was 0.6138. Training loss continued
  downward after epoch 6 while validation performance became unstable, confirming
  spatial overfitting rather than incomplete optimization.
- Epoch-6 class performance was: soil precision/recall/F1 =
  0.6617/0.7832/0.7173; chickpea = 0.4860/0.4676/0.4766; weed =
  0.7026/0.6002/0.6474. Chickpea is now the limiting supervised class.
- Soil accuracy rose monotonically from 0.6075 in patches with center-class
  fraction at most 0.25 to 0.9864 above 0.90, confirming a strong neighborhood-
  mixing effect for soil.
- Chickpea accuracy was nonmonotonic and highly cube-dependent (0.2676–0.5962).
  Weed accuracy also varied substantially by cube (0.4629–0.7703); its pooled
  decrease with purity was confounded by cubes, because high-purity weed was
  accurate in Cubes 24/32 but poor in Cubes 40/45/54.
- These findings do not validate hidden weed classes, but they establish a
  defensible signal of inter-cube weed heterogeneity that the SSL stage must test
  for repeatability. The next supervised gate is a center-aware architecture
  ablation: pixel spectrum, spatial-average CNN, and fused center-plus-context
  models on the same frozen Fold-1 samples.
- Staged a controlled three-seed center-aware architecture ablation using the
  identical 60,000 training and 15,000 validation patches. It compares a center-
  spectrum MLP, the existing spatial-average CNN, and a fused center-plus-context
  classifier with matched early-stopping rules on GPU 1.
- The ablation reports parameter count, overall and class F1, seed variability,
  and mean/worst-cube macro-F1. No automatic winner is declared; review of both
  central accuracy and spatial robustness is required before five-fold training.

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
- Cube 20 mask provenance resolved: refined weed exactly matches categorical
  class 3 (IoU 1.0); both soil variants contain all categorical soil plus the
  same 10,145 extra pixels (IoU 0.982005); chickpea binary variants disagree and
  are rejected. The categorical mask is frozen as the sole three-class source.
- On the reflectance grid the categorical source yields 537,983 soil, 43,407
  chickpea, and 231,017 weed pixels. Added deterministic nearest-neighbor
  materialization with hashes and mutual-exclusivity checks; raw files stay intact.
- Materialization passed: four distinct hashed products were written on the exact
  989×1517 grid with zero class overlap and 687,906 unlabeled pixels. Cube 20 is
  now integrated as archive-recovered provenance, increasing the authoritative
  collection to 32 reflectance, 19 weed-SSL, and 16 three-class supervised cubes.
- Exact field-boundary containment is 100% for 32 plots; plots 10, 19, and 64 are
  76.76%, 91.74%, and 81.42% inside, respectively, but all reach 100% within the
  1.5 m field-identity tolerance. This tolerance is explicitly prohibited from
  serving as the chickpea constraint buffer.
- All 35 numbered plots fall completely within the 1.5 m buffered bounding box of
  the field boundary candidate. This supports Field 1 identity, while exact
  polygon containment remains the final gate before spatial label constraints.
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

### Thirty-two-cube spectral verification and spatial grouping

- Reprofiled 320,000 observed spectra across all 32 authoritative cubes after
  integrating Cube 20. The primary band policy remains indices 3–113 (111 bands,
  401.84–869.34 nm): saturation is 0.854% at band 113 and 1.543% at band 114.
- Verified all 32 manifest records use the true 2025-05-06 acquisition date; the
  June directories remain processing batches only.
- Across available raw masks, class abundance is approximately chickpea : weed :
  soil = 1 : 4.00 : 6.76. Supervised metrics therefore require class-balanced
  reporting rather than overall accuracy alone.
- Added globally anchored 5 m map-coordinate group indexing. Identical ground
  locations observed in overlapping cubes receive the same group ID because cube
  identity is excluded from the key.
- The index writes cube footprints, cube-to-group membership, and conservative
  overlap pairs for review. It deliberately does not assign folds yet.
- Proposed a 0.30 m sampling exclusion around future fold boundaries and recorded
  it as a candidate, not a frozen decision. Field 2 remains unopened.

### Spatial-index findings and report organization

- Indexed 221 unique 5 m map-coordinate groups across the 32 Field 1 cubes; 147
  groups (66.5%) occur in multiple cubes and 70 cube pairs share at least one
  group. All cubes belong to one overlap-connected component.
- The indexed footprint spans approximately 92.44 m east–west and 87.43 m
  north–south in EPSG:32617. Pixel size is 1.5 cm for 31 cubes and 2.0 cm for one
  cube.
- Cube-held-out evaluation is rejected as the primary split because overlapping
  cubes connect the full collection and 82.4% of cube-group membership records
  occur on repeated ground.
- Added authoritative class-count summaries by map group so contiguous folds can
  be optimized for chickpea representation without inspecting Field 2.
- New reviewable CSVs now write beneath `metadata/local/reports/<stage>/`.
  Canonical files required by downstream scripts retain their stable locations.

### Spatial edge-index correction and visual-QC policy

- The first spatial index used block-centre containment and missed a valid label
  pixel group along a rotated cube edge. No labels or source data were changed.
- Replaced the approximation with chunked enumeration of every raster pixel
  centre. The revised index exactly includes all blocks touched by observed
  pixels while reading no reflectance values.
- Added a four-panel PNG overview of labeled-cube coverage and authoritative
  soil, chickpea, and weed fractions by 5 m group.
- Investigator requested visual verification after each major task. Future stages
  will include meaningful PNG QC alongside numeric reports when applicable.

### Class geography and candidate-fold optimizer

- Authoritative precedence counts across 174 labeled spatial groups are 8,171,964
  soil (56.18%), 1,238,391 chickpea (8.51%), and 5,136,987 weed (35.31%) pixel
  observations. Counts intentionally retain repeat observations within a group.
- Thirty-six labeled groups contain no chickpea. Simple easting quintiles assign
  only 8.4% of chickpea to one fold; northing quintiles range from 12.1% to 29.4%,
  so neither naive split is accepted.
- Added a deterministic search over spatial-slab angles and boundaries. The
  objective balances spatial groups, labeled groups, soil, weed, all labels, and
  chickpea, with chickpea receiving the highest weight.
- Candidate folds must meet minimum group and labeled-group counts. Disconnected
  fold components receive a strong penalty. The output remains review-only until
  its numeric summary and PNG are approved.

### Candidate spatial folds accepted for contract materialization

- The 110° candidate produces five single-component contiguous folds. Chickpea
  allocation is 17.87–22.44% per fold (221,294–277,919 observations), and total
  labeled allocation is 18.13–22.16% per fold.
- Soil allocation ranges from 13.36% to 25.63% and weed from 16.36% to 24.70%.
  This remaining compositional heterogeneity is retained as a realistic spatial
  generalization challenge rather than erased by noncontiguous reassignment.
- Added explicit acceptance gates and a contract-materialization script. The
  frozen local contract records assignment, boundary, configuration, and input
  hashes and preserves the reviewed PNG. The 0.30 m boundary exclusion remains a
  sampling-stage rule, not a relabeling operation.

### Spatial contract frozen and normalization staged

- Materialized the accepted 266-group, five-fold Field 1 contract. Assignment
  SHA-256 is `53bef09e72fc112c8473c10bb16d54cd202bce055377ab91f776b2e8ad391b09`;
  assignment and boundary hashes independently reverified from the attached files.
- Added deterministic fold-specific normalization over a cube-balanced sample of
  valid Field 1 pixels. Each held-out fold and training centres within 0.30 m of
  its boundary are excluded from that fold's statistics.
- Normalization uses only primary bands 3–113 and records population mean,
  standard deviation, minimum, maximum, sample counts, configuration hashes, and
  a PNG spectral diagnostic. Unlabeled Field 1 pixels remain eligible because the
  SSL representation task is unsupervised. Field 2 remains locked.

### Normalization passed and sample indexing staged

- Fold-specific normalization used 640,000 cube-balanced valid spectra. Training
  sample counts range from 489,451 to 519,519; boundary exclusion removes only
  0.61–1.45% of the sampled pool.
- Across retained bands, fold-specific mean curves differ by about 0.7–2.4% on
  average. The largest relative mean spread is 6.37% near 677.65 nm. All bands
  have nonzero variance; no scaling instability was detected.
- Added an observed-only sample-centre index capped within cube × map group ×
  class strata. The cap controls training imbalance without synthesizing pixels.
- Each centre records fold, class, patch-edge validity, and separate training- and
  validation-side safety for every held-out fold under the 0.30 m boundary rule.
- The centre index is explicitly training-candidate-only. Primary evaluation will
  use exhaustive tiled prediction on held-out authoritative masks so test
  prevalence is not altered by balanced sampling.

### Training-centre pool passed and observed patch QC staged

- Indexed 710,693 observed candidate centres from 19 cubes: 281,617 soil,
  164,750 chickpea, and 264,326 weed. The resulting pool reduces the raw class
  imbalance to approximately chickpea : weed : soil = 1 : 1.60 : 1.71.
- Each fold retains 126,908–131,191 training-eligible chickpea centres and at
  least 28,276 validation-diagnostic centres per class. Boundary removal remains
  modest and no class or fold is starved.
- Added a pretraining visual gate that loads real 15×15×111 patches, checks the
  authoritative center label, patch shape, and fold-specific normalization, and
  renders an observed NIR–red–green montage. False colours are diagnostic only
  and are not model inputs or biological labels.

### Patch-QC correction: internal georectification NoData

- Label, shape, and normalization checks passed for all 1,500 inspected patches,
  but 202 patches (13.47%) contained at least one all-band-zero spatial pixel.
  This is a patch-neighborhood issue despite valid authoritative centre labels.
- Training on these patches could let the encoder learn georectification-footprint
  geometry. The candidate-index builder now uses an integral image per cube and
  requires every 15×15 patch pixel to contain observed reflectance.
- Added per-cube exclusion reporting and made patch NoData a hard QC failure.
- Corrected the montage sampler to show one observed example per class per fold
  rather than selecting all examples from Fold 1.


### NoData-safe patch pool passed and supervised diagnostic staged

- Rebuilt the observed candidate-centre contract after requiring every 15×15
  neighborhood to contain measured reflectance. The corrected pool contains
  684,521 centres: 270,118 soil, 159,976 chickpea, and 254,427 weed.
- Repeated patch QC checked 500 examples per class across all five folds. All
  1,500 patches passed centre-label, shape, finite-normalization, and full-patch
  observation checks; no patch contains an all-band-zero spatial pixel.
- Added a bounded Fold-1 supervised spectral-spatial diagnostic using 20,000
  observed training patches and 5,000 observed validation patches per class.
  It uses training-only Fold-1 normalization, mixed precision, and all visible
  GPUs, and emits learning curves, a classification report, and a normalized
  confusion-matrix PNG.
- This smoke test is explicitly diagnostic-only. It does not replace exhaustive
  held-out inference, the complete five-fold supervised baseline, or the SSL
  comparison, and it creates no synthetic observations.


### First GPU diagnostic interrupted and workload corrected

- The initial supervised smoke test was manually stopped after 34 minutes because
  no epoch had completed. Both RTX 5000 Ada GPUs were active, but memory use was
  below 1 GiB per device and random multi-cube ENVI patch reads made the pipeline
  input-bound. No result or model-performance claim was retained.
- Reduced the diagnostic to 5,000 observed training patches and 2,000 observed
  validation patches per class for three epochs, increased the batch size to
  1,024, and added batch-level elapsed-time reporting. The revised run remains
  balanced, spatially held out, GPU accelerated, and diagnostic-only.
- The full benchmark will not inherit this convenience subsample. Its loader must
  be redesigned for cube-local reads or cached shards before five-fold training.

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

### Supervised architecture decision frozen

- Completed the controlled Fold-1 three-seed architecture ablation on identical
  observed samples. Center-context fusion achieved mean macro-F1 0.643905
  (SD 0.005936), exceeding spatial-average CNN by 0.029469 absolute (4.80%
  relative) and winning the paired comparison for seeds 42, 43, and 44.
- Spatial-average CNN retained the strongest mean chickpea F1 (0.474566), while
  center-spectrum MLP retained the strongest mean worst-cube macro-F1 (0.536734).
  The three models therefore serve distinct primary, spatial-context baseline,
  and spectral-only baseline roles rather than being reduced to a single result.
- Added a deterministic contract freezer that verifies all nine
  architecture/seed runs, class metrics, cube metrics, source hashes, and reviewed
  PNG before recording the primary architecture and both baselines.
- The contract explicitly prohibits architecture switching after the five-fold
  benchmark. Fold 1 remains development-only; Field 2 remains unopened.

### Resumable dual-GPU five-fold supervised benchmark staged

- Added a contract-gated five-fold benchmark over the three frozen architectures
  and seeds 42, 43, and 44, yielding 45 planned Fold 1–5 runs.
- The scheduler launches independent single-GPU subprocesses on CUDA devices 0
  and 1. It never uses DataParallel; at most one fold owns each GPU.
- Completed folds are validated from their nine architecture/seed summaries and
  skipped on restart. Fold-specific reports and checkpoints remain isolated.
- The aggregate stage produces run-, fold-, class-, and cube-level CSVs, a
  three-panel PNG, and a hashed local benchmark contract.
- This benchmark uses balanced observed validation subsets to measure stability.
  It is explicitly not the later exhaustive held-out-mask evaluation and does
  not access Field 2.

### Balanced five-fold supervised benchmark passed

- Completed all 45 frozen runs (five spatial folds × three architectures × three
  seeds) using independent workers on both RTX 5000 Ada GPUs. The run completed
  substantially faster than the conservative estimate, confirming that the
  corrected loader and cached data path are no longer the training bottleneck.
- Center-context fusion ranked first in every held-out fold. Across the 15 nested
  fold × seed runs, mean macro-F1 was 0.687735 versus 0.661296 for
  spatial-average CNN and 0.624518 for center-spectrum MLP.
- Fusion improved over spatial-average CNN by 0.026439 absolute (4.00% relative).
  Fold-mean gains were positive in all five folds and ranged from 0.015684 to
  0.038847.
- Fusion improved mean soil F1 by 0.039971 and weed F1 by 0.045033 relative to
  spatial-average CNN. Chickpea F1 was effectively tied but slightly lower
  (0.526116 versus 0.531804; delta -0.005688).
- Fold geography dominates seed variability: fusion's within-fold seed SD was
  0.001021–0.009114, while its across-run SD was 0.050069. Chickpea remains the
  unstable class (fusion F1 SD 0.113815), ranging from fold means 0.393987 to
  0.665918.
- Worst-cube macro-F1 remains weak and must be audited for cube-level class
  support before interpretation. The next evaluation contract will distinguish
  exhaustive pixel-observation, spatial-group-balanced, cube-level, and
  overlap-sensitivity results; folds, not the 15 nested runs, are the primary
  independent units for uncertainty.
- These results validate the supervised architecture choice but do not test
  hidden weed subclasses or the SSL hypothesis. Field 2 remains unopened.

### Nested primary evaluation protocol staged

- Identified an essential distinction between the completed development benchmark
  and primary evaluation: development runs used outer-fold labels for early
  stopping and therefore cannot support unbiased outer-fold performance claims.
- Added a deterministic nested spatial protocol. For each outer test fold, one
  different spatial fold is reserved for early stopping and the remaining three
  folds alone supply supervised gradient updates.
- Primary retraining will also recompute normalization without both outer and
  inner folds. Outer labels are prohibited from early stopping, threshold tuning,
  architecture selection, or geometric postprocessing decisions.
- The frozen protocol specifies three-seed probability ensembles, exhaustive
  eligible authoritative labeled-pixel observations, equal-weight spatial-group
  summaries, cube and overlap sensitivity analyses, and 1,000-replicate
  spatial-group cluster bootstrap uncertainty.
- Unconstrained predictions remain primary. Plot/alley chickpea constraints are
  reserved for a declared sensitivity analysis. Field 2 remains locked.

### Nested training-only normalization staged

- Added deterministic cube-balanced normalization for the nested primary protocol.
  For each outer evaluation, both the outer test fold and its separate inner
  early-stopping fold are excluded from mean, variance, minimum, and maximum
  calculations.
- Training-side samples within 0.30 m of either excluded-fold boundary are also
  removed. This prevents preprocessing statistics or patch neighborhoods from
  carrying information across Test or Stop boundaries.
- The stage verifies frozen fold, role-table, protocol, manifest, band-policy, and
  configuration hashes; rejects zero-variance bands; and emits summary CSVs,
  spectral mean/SD visual QC, and a local immutable contract.
- Existing outer-only normalization remains valid for development diagnostics but
  is prohibited for the nested primary evaluation. Field 2 remains locked.

### Nested fitting and inner-validation sample freeze staged

- Added a deterministic selector over the existing 684,521 fully observed,
  authoritative Field 1 candidate centres. For each outer evaluation it requires
  simultaneous safety from both outer and inner boundaries.
- Exact model-selection samples are frozen at 20,000 fitting and 5,000 inner
  validation centres per class per outer fold, totaling 375,000 role-specific
  rows. Training uses only the three Fit folds; Stop samples come only from the
  declared inner fold; outer Test labels are absent.
- The stage records eligible pool sizes and cube coverage before sampling,
  verifies normalization and protocol hashes, and emits a class/fold availability
  PNG. These balanced samples support fitting and early stopping only; exhaustive
  outer evaluation retains observed prevalence.
- Sample selection uses deterministic seed 42 in the builder without changing the
  already frozen primary-evaluation configuration. Field 2 remains locked.

### Dual-GPU nested checkpoint training staged

- Nested sample freezing passed with 375,000 exact observed role-specific rows.
  The smallest eligible Fit pool is 90,723 chickpea centres and the smallest
  Stop pool is 27,604 chickpea centres, comfortably above frozen selections.
- Extended the validated architecture trainer with a contract-gated nested mode.
  It reads exact frozen samples and outer+inner-excluded normalization, rejects
  any outer-fold sample in Fit or Stop roles, and records outer-test access as
  false.
- Added a resumable two-GPU scheduler for 45 checkpoint runs. Each outer fold
  trains the three frozen architectures across seeds 42–44; independent
  subprocesses own CUDA 0 and CUDA 1, avoiding DataParallel.
- Checkpoints are selected exclusively by inner-validation macro-F1. Aggregated
  inner metrics and the PNG are model-selection diagnostics only, never outer
  performance claims. Completion freezes hashes for all 45 checkpoints before
  outer-test inference is permitted. Field 2 remains locked.

### Nested checkpoints passed and outer-test support audit staged

- Frozen all 45 nested checkpoints after inner-validation-only early stopping.
  Center-context fusion led spatial-average CNN on every Stop-fold pairing by
  0.024293–0.041830 macro-F1. Across nested diagnostics, fusion mean macro-F1 was
  0.686752, with soil/chickpea/weed F1 of 0.783548/0.514095/0.762613.
- These values remain checkpoint-selection diagnostics, not test results.
  Selected epochs vary substantially by outer role; fusion averages 4.87 epochs
  and ranges from 1 to 11, supporting fold-specific early stopping.
- Added a post-checkpoint exhaustive support audit that is the first permitted
  opening of Field 1 outer-test labels. It counts all patch-safe, fully observed,
  boundary-safe labeled observations by fold, cube, class, and 5 m group without
  generating predictions.
- The audit explicitly marks cube-fold records missing any class as invalid for
  three-class cube macro-F1, quantifies the eventual inference volume, and emits
  class-support and cube-completeness visual QC. Field 2 remains locked.


### Corrected outer support accounting and GPU inference pilot staged

- Corrected the cube-fold completeness report so 46 combinations with zero eligible
  observations are classified as noncontributing rather than invalid. The exhaustive
  support now contains 49 contributing cube-fold records: 36 contain all three
  classes and 13 contain two classes. The scientific support total remains unchanged
  at 12,782,603 eligible observed Field 1 predictions.
- Added a contract-gated real-data GPU inference engineering pilot. It loads the
  frozen three-seed center-context ensemble, gathers 15×15×111 observed patches on
  one GPU without random ENVI reads, and benchmarks batch sizes 256, 512, and 1024.
- The pilot computes no target metric and cannot become a reported performance
  result. It writes throughput and peak-VRAM CSVs, a PNG, and a frozen selected-batch
  contract before exhaustive inference is implemented. Field 2 remains locked.

### Outer-inference pilot checkpoint discovery correction

- The first pilot invocation stopped before data or GPU inference because checkpoint
  discovery omitted the repository's `results/` directory. The frozen checkpoints
  themselves were intact under `results/supervised_nested_training/outer_fold*/`.
- Added the exact results root while retaining checkpoint metadata matching and
  SHA-256 verification against the frozen 45-checkpoint contract. No model was
  retrained, no metric was computed, and Field 2 remained locked.

### Observed-data inference pilot passed and exhaustive evaluation staged

- The real Field 1 pilot verified all three Fold-1 center-context checkpoint hashes
  and three-seed probability ensembling. Batch 512 achieved 200,273 observed
  predictions/s at 0.785 GiB peak allocated VRAM and outperformed batches 256 and
  1,024. Probability sums were accurate within 2.38e-7.
- Added resumable independent-GPU workers for the five frozen outer folds. Each
  worker evaluates all three frozen architectures and all three seeds at observed
  prevalence, using the exact nested normalization and boundary-safe support.
- Individual pixel probabilities are not persisted. Exact confusion counts are
  accumulated by fold, cube, and frozen 5 m spatial group, enabling pixel-weighted,
  equal-group, complete-cube, and 1,000-replicate group-cluster-bootstrap summaries.
- The aggregate stage verifies 12,782,603 predictions per architecture and produces
  CSV results, a four-panel PNG, and a hashed evaluation contract. Field 2 remains
  locked.

### Exhaustive predictions completed; aggregation compatibility corrected

- Completed all five outer folds for all three architectures: 12,782,603 eligible
  observations per architecture and 38,347,809 architecture-level predictions in
  total. Each fold matched its frozen support count exactly and wrote a hashed,
  resumable fold contract.
- Final plotting stopped after metric and bootstrap tables were created because the
  installed Matplotlib version removed the `labels=` boxplot argument. Replaced it
  with `tick_labels=`. Rerunning skips all five verified inference folds and repeats
  aggregation only; no prediction work is lost or duplicated.

### Exhaustive supervised outer evaluation passed; paired audit staged

- Final equal-5 m-group metrics confirm center-context fusion as the strongest
  supervised model: balanced accuracy 0.699235 and macro-F1 0.671206, compared
  with 0.674820/0.635110 for spatial-average CNN and 0.652364/0.646696 for the
  center-spectrum MLP.
- Fusion achieved soil/chickpea/weed F1 of 0.893603/0.300510/0.819504. Chickpea
  remains the limiting class; pixel-weighted recall is approximately 0.48, with
  substantial confusion into both soil and weed. This limitation must be reported,
  not hidden by the strong soil and weed scores.
- Added a paired spatial-group bootstrap audit using identical resampled 5 m groups
  for fusion and each comparator. It reports confidence intervals for performance
  differences and fold-wise geographic consistency before the supervised baseline
  is frozen for SSL comparison. No new inference is performed and Field 2 remains
  locked.


### Paired supervised audit passed; overlap sensitivity staged

- Paired equal-5 m-group bootstrap inference confirms center-context fusion over
  both frozen comparators. Fusion improved macro-F1 by 0.024510 versus the
  center-spectrum MLP (95% interval 0.010499–0.039788) and by 0.036095 versus
  spatial-average CNN (0.029376–0.042737).
- Fusion materially improved chickpea F1 over both comparators, although its
  absolute exhaustive chickpea F1 remains low and is a declared limitation.
  Center-spectrum retained higher soil F1, and Fold 3 showed a small fusion
  macro-F1 deficit versus center-spectrum; neither exception is concealed.
- Added a deterministic observed-data overlap audit before sensitivity inference.
  It derives median native GSD from raster transforms, evaluates predeclared 1×
  and 2× native-GSD ground grids, retains the first lexicographic cube observation
  without using labels or predictions, and quantifies removed repeats and
  cross-cube label disagreement.
- This audit verifies all 12,782,603 exhaustive support observations, emits CSVs,
  PNG visual QC, and a hashed local contract. It performs no model inference and
  keeps Field 2 locked.


### Native-GSD overlap support passed; label-consistency diagnosis staged

- The native raster resolution is 1.5 cm for 18 labeled cubes and 2.0 cm for
  Cube 28. At the predeclared 1× median-GSD grid, 11,581,768 of 12,782,603
  exhaustive observations remain after deterministic de-duplication (90.61%).
- Class retention is balanced: 89.94% soil, 91.49% chickpea, and 91.45% weed.
  The class mixture therefore changes by less than half a percentage point.
- The 2× grid retains only 23.31% because each 3 cm cell deliberately aggregates
  roughly four native pixels; it is retained as a coarse-scale stress test and is
  not interpreted as literal repeated-observation removal.
- At 1× GSD, 576,067 repeated observations disagree with the first retained label,
  or 47.97% of removed observations (4.51% of all observations). Added a
  predeclared 5×5 label-neighborhood audit to distinguish class-boundary
  misregistration from interior label inconsistency and to separate cross-cube
  repeats from any within-cube grid collisions. No labels are changed.


### Repeated-view label inconsistency confirmed; model sensitivity frozen

- All 1,200,835 native-GSD repeats are cross-cube observations; there are no
  within-cube grid collisions. Cross-cube disagreement ranges from 43.80% to
  58.02% across outer folds, so it is not isolated to one geographic split.
- Boundary structure explains a substantial share but not all disagreement:
  neither-view-interior pairs disagree 50.62%, one-interior pairs 45.67%, and
  pairs for which both 5×5 neighborhoods are class-interior still disagree
  22.95%.
- Chickpea is especially unstable across repeated views: a reference chickpea
  label is repeated as chickpea only 11.04%, versus 57.04% soil and 31.92% weed.
  This is treated as observed mask/georegistration uncertainty, not biological
  evidence of hidden weed subclasses.
- Before sensitivity predictions, froze two label-transparent populations at
  native GSD: one deterministic lexicographic view per cell, and a stricter
  population retaining unique cells plus repeated cells only when every observed
  label agrees. Conflicting cells are excluded only from the latter sensitivity;
  primary exhaustive results remain unchanged.
- Added resumable dual-GPU inference for all three frozen supervised
  architectures. Models are not retrained or selected from sensitivity results,
  individual probabilities are not persisted, and Field 2 remains locked.


### Investigator-requested chickpea-only mask review staged

- Paused overlap-sensitivity inference before execution to support visual review of
  every authoritative Field 1 chickpea mask.
- Added full-resolution counts, one chickpea-only PNG per cube, a high-resolution
  all-cube overview, and source-versus-authoritative provenance panels. Pixels
  removed by the frozen weed > chickpea > soil precedence are shown separately.
- The overview contains only model-ready chickpea pixels on black; no soil, weed,
  false-colour imagery, or synthetic data are displayed. Raw masks are never
  modified, and Field 2 remains locked.

## 2026-08-17

### Field 2 reflectance valid-support masks frozen

- Derived read-only Field 2 support solely from finite, nonzero reflectance in
  frozen model-input Python bands 3–113 (ENVI bands 4–114); no PCA, stored
  index, prediction, probability, label, or biological assumption entered the
  rule.
- Audited finite-any-nonzero, finite-all-nonzero, full-spectrum-any-nonzero,
  and positive-norm thresholds above 0, 1, 5, and 10 stored units. All seven
  candidates agreed pixel-for-pixel across all 40 cubes.
- Retained every observed component without morphology. Cube 02's coherent
  upper scan strips were explicitly reviewed and retained; its two remaining
  fragments total 29 pixels. No other cube has an unexplained non-small
  interior component.
- Materialized 40 ignored uint8 GeoTIFF masks with exact source grids and
  hashes. All 40 cubes now have status `annotation_ready_prediction_free`,
  while every investigator role remains `unreviewed`.
- Confirmed that PCA and stored-index rasters each contain 18,349,592 nonzero
  pixels outside authoritative reflectance support. The source rasters were
  not altered or resampled.
- Before/after path, size, `mtime_ns`, and SHA-256 snapshots of all 240 source
  files are identical. No supervised or SSL training occurred, and the frozen
  Field 1 benchmark remained unchanged.


### Prediction-free Field 2 cube-role review prepared

- Validated the frozen Field 1 finalization and the exact 40-cube Field 2
  readiness, source, and valid-support contracts before preparing review data.
- Built checksum-tracked false-colour, supplied-PCA, stored-index, valid-support,
  and support-outline views for all 40 cubes. All biological roles remain
  `unreviewed`; generated PNGs and local manifests remain ignored.
- Added a localhost browser reviewer with zoom/pan/reset, navigation, independent
  role flags, logical validation, atomic save/resume, explicit clear-current, an
  audit table, and a reviewed/unreviewed overview.
- Implemented a separate all-40-reviewed role-freeze gate, the later deterministic
  blind sampling generator, and a future point-annotation interface skeleton.
  Freeze and sampling were not run, and the point annotator was not launched.
- No checkpoint, prediction, probability, Field 1 label, automatic biological
  role, Field 2 categorical label, or training operation entered this work.


### Investigator-reviewed Field 2 cube roles frozen

- Froze review revision 86 only after exact JSON/CSV agreement, 40/40 reviewed
  records, logical/audit passes, preview rehashing, and frozen source/support
  contract validation. No investigator decision was inferred or altered.
- Frozen roles comprise 20 primary three-class, 8 challenge-only, 7
  chickpea-absent negative-control, and 5 sensitivity-only cubes, with no
  excluded or unreviewed cubes. All 40 confidence values are high.
- Added a write-once contract, frozen role table, biological-support summary,
  visual overview, and a validator that cross-checks every hashed input/output
  and detects later input or contract mutation.
- The contract records freeze implementation commit `bcfefaa4bb702879e95fbaec297fd18d6e7b6afe`
  and SHA-256 `a76425dd259f4fb245db6d51824a99d4f960e044ca5aab2cfc114578bcf06c96`.
- The blind point-sampling frame and sampling contract remain absent. No point
  annotator, inference, pseudo-labeling, training, or model selection ran.


### Prediction-free Field 2 blind sampling frozen

- Materialized the predeclared 800-point main frame and a disjoint 396-point
  reserve from frozen valid support, frozen cube roles, ground coordinates,
  deterministic 5 m blocks, and within-cube stored-scalar rank strata only.
- Achieved exact per-cube allocations for all 40 cubes. The 1,196 combined
  points have unique IDs and ground cells, zero support/context violations,
  and minimum separation 0.250599 m across 216 spatial blocks.
- Froze first-stage/final inclusion probabilities and design weights, immutable
  input/output hashes, allocation and stratum/block audits, and a prediction-free
  map. Same-seed selection reproduced byte-identical main and reserve frames.
- Locked the reserve from annotation behind a separate immutable authorization
  and prespecified support gates unrelated to predictions or model performance.
- Completed and launched the main-only point annotator with 800 neutral records,
  no default label, no reserve endpoint, atomic save/resume, audit exports, and
  explicit role-contradiction handling.
- No checkpoint, model output, prediction, probability, pseudo-label, training,
  model selection, or automatic biological inference entered the workflow.


### Full-resolution natural-RGB point-annotation display frozen

- Added a write-once display addendum and 40 ignored full-resolution RGB PNGs
  derived only from Field 2 reflectance and frozen valid-support masks. Every
  cube uses 669.09/550.54/479.80 nm for red/green/blue, a cube-level 2nd–98th
  percentile stretch, uint8 output, and black outside support.
- Made natural RGB the default while retaining false colour, PCA, stored index,
  and support context as selectable views. The interface now provides complete
  cube and centered magnified views, 2×/4×/8×/16× nearest-neighbor zoom, an
  unfilled sampled-pixel outline, optional high-zoom grid, and shortcuts.
- Versioned every point record against the display contract. Older reviewed
  labels would be preserved and flagged for visual re-review; because no point
  annotation file existed, the initialized revision has 800 main records, zero
  reviewed, zero labeled, and zero requiring re-review.
- The live API exposes exactly 800 main samples and no reserve samples. All 94
  tests and the frozen Field 1, Field 2 role, Field 2 sampling, and RGB-display
  validators passed; frozen main/reserve/combined hashes remained unchanged.
- No checkpoint, prediction, probability, pseudo-label, suggested class,
  training, Field 1 label, PCA value, or stored-index value entered RGB.


### Field 2 point-viewer interaction fixed

- Diagnosed the inert image as missing canvas pointer/wheel listeners; Firefox
  found no JavaScript errors, overlay interception, or control hit-test failure.
- Added an immutable yellow sampled-pixel outline and independent cyan inspection
  cursor, click-to-inspect magnifier, wheel zoom, drag pan, reset, exact
  2×/4×/8×/16× controls, high-zoom grid, and nearest-neighbor rendering.
- Kept natural RGB as default and retained false-color, PCA, stored-index, and
  support views without changing the frozen sample on layer switches.
- Replaced label/confidence selects with explicit clickable buttons and verified
  all form, navigation, save, review, layer, and zoom controls remain unobstructed.
- All 97 tests and the frozen Field 1 finalization, Field 2 cube-role, sampling,
  and RGB-display validators passed. The live API still exposes exactly 800 main
  samples and no reserve samples; sampling, display, and annotation hashes are
  unchanged.


### Prediction-free Field 2 area annotation prepared

- Added a separate port-8774 area annotator for investigator-defined research
  crop area, alley, outside-field, and uncertain-boundary contextual domains.
- Added full polygon drawing/editing, undo/redo, pan/zoom, support-aware geometry
  audits, pixel/CRS vertices, atomic JSON/GeoJSON/CSV/audit/overview save outputs,
  and an explicit—not silent—unassigned-support outside-field action.
- Implemented but did not execute a write-once freeze workflow requiring all 40
  cubes reviewed; it builds deterministic support-clipped masks and coordinate-
  free main/locked-reserve memberships without assigning biological labels.
- Blocked point annotation until the area contract is frozen, added immutable
  zone/domain display, alley label restrictions, boundary-correction handling,
  and the pre-freeze `weed_soil_mixed` biological option.
- All 108 tests and every frozen Field 1/Field 2 source, support, role, sampling,
  and RGB validator passed. Point labels/reviews remain zero; all sampling and
  point-annotation hashes are unchanged.


### Field 2 area polygon finalization corrected without rewriting geometry

- Backed up and hashed the revision-25 area package before diagnosis; all 225
  polygons and original investigator vertices remain byte-identical.
- Reproduced the cube12 crop failure in isolated Firefox: tool switching silently
  cleared a valid draft, while double-click appended two terminal pointer events
  without finalizing.
- Preserved unfinished drafts across tool changes, locked draft zone identity,
  unified Finish/Enter/double-click completion, added explicit cancel and detailed
  validation-stage counts, and separated original from operational clipped geometry.
- Generated a non-applying 54-row terminal-vertex reconciliation table with 53
  review-only candidates and one manual-review/no-change record.
- Presented exact overlap pairs without priority and retained mixed-mode review
  gates for cubes 11, 16, and 17. Area geometry remains unfrozen.
