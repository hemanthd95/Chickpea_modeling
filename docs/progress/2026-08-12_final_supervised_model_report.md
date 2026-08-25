# Interim final supervised model report — Field 1 freeze

Benchmark reference: `da95fd59b7dc5d8ba4add96ed43b398bb7c93749`

Annotated tag: `supervised-confident-v1-da95fd5`
External-validation status: **stopped before Field 2 compatibility; no Field 2 data opened**

## 1. Executive summary

The confident-label Field 1 supervised benchmark is verified and frozen. Its
primary three-seed `center_context_fusion` ensemble with alley-aware labels
achieved equal-spatial-group balanced accuracy 0.932545 and macro-F1 0.876089.
The corresponding without-alley result was 0.884095/0.835364; the frozen legacy
result was 0.785344/0.648250. Paired 5 m spatial-group bootstrap intervals favor
the alley-aware confident-label model for balanced accuracy, macro-F1, and all
three class F1 scores.

A post-hoc grouped spectral audit was completed on the 15 already-frozen nested
checkpoints without retraining. It demonstrates reliance on both the center
spectrum and spatial context, with especially strong mean-replacement effects at
401.84–414.07 nm and around red absorption/red-edge wavelengths. Only
401.84–414.07 nm remained a stable macro-F1 region under adjacent-wavelength
interpolation after multiplicity correction, demonstrating substantial
redundancy among neighboring hyperspectral bands. These results cannot be used
to revise the frozen band set.

Two all-eligible-primary-Field-1 deployment ensembles are frozen before any
external imagery is opened: with alley labels and without alley additions, each
with seeds 42–44. No scientifically equivalent legacy all-Field-1 deployment
contract exists, so none was fabricated.

Field 2 is not georectified or populated in the authorized data section. The
pipeline stops here. Cubes 24 and 28 remain a previously inspected early-growth
stress test and are not blind external data.

## 2. Scientific question and claim boundary

The supervised question is whether conservative, provenance-preserving Field 1
labels improve soil/chickpea/weed discrimination under nested spatial
evaluation, and which physical wavelength regions and model branches the frozen
classifier relies on.

Valid claims are restricted to the observed Field 1 acquisition and the frozen
evaluation design. The results support improved internal spatial generalization
relative to the frozen comparators. They do not establish blind external Field 2
performance, universal field transfer, biological causality of individual
wavelengths, or hidden weed subclasses.

## 3. Dataset and cube inventory

All imagery was acquired on 2025-05-06; later folder dates describe processing,
not repeated acquisition dates.

| Cube role | Cubes | Use |
| --- | --- | --- |
| Primary Field 1 | 20, 22, 30, 31, 35, 40, 45, 47, 49, 50, 54, 55, 56 | Nested fitting/evaluation and deployment fitting |
| Positive-support expansion | 12, 14, 15 | High-confidence chickpea support sensitivity; not promoted to primary deployment fitting |
| Early-growth sensitivity | 24, 28 | Previously inspected phenology stress test; never blind data |
| Transfer warning | 32 | Sparse investigator points only; no dense probability labels and no deployment fitting |
| Field 2 | none authorized | Compatibility not run; no data opened |

The primary nested evaluation contains 5,924,107 deduplicated ground-cell
observations. The interpretability audit uses a deterministic class/fold/
spatial-group-stratified subset of 3,016 observations across 136 frozen 5 m
groups; it does not replace the exhaustive benchmark.

## 4. Class-label construction and investigator process

Labels are soil = 0, chickpea = 1, weed = 2, unresolved = 255. The confident
vegetation thresholds were fixed at P(chickpea) <= 0.15 for weed and >= 0.80 for
chickpea. The middle interval and polygon-edge belt remain unresolved. NoData is
always unresolved.

Investigator row-region polygons define incomplete positive-support eligibility,
not exhaustive absence. Investigator alley footprints prohibit chickpea and have
precedence. Of 354 vegetation-reference points, 348 passed spectral QC. Their
leave-one-cube-out separator achieved mean AUC 0.9556 and balanced accuracy
0.9115; Cube 32 remained a transfer warning. Each final label retains explicit
provenance, and source masks/annotations remain immutable.

The network receives none of these geometries as input. Repository inspection
confirms that its only input is the normalized reflectance patch.

## 5. Alley rule and measured effect

The configured 0.05 m inward alley core contributed 1,147,513 final weed-
provenance pixels and marked 1,717,525 existing soil pixels with alley-core
provenance. Alley precedence removed 100,783 historical chickpea pixels from
eligible chickpea support; zero final chickpea labels remain in full alleys.

On identical primary ground cells, with-alley minus without-alley differences
were +0.048362 balanced accuracy (95% interval 0.039377–0.057520), +0.041054
macro-F1 (0.026683–0.055562), +0.064463 chickpea F1 (0.042634–0.090731), and
+0.055913 weed F1 (0.032158–0.078505). Weed precision changed by -0.017635 and
its interval crossed zero (-0.041055–0.001808), so no precision benefit is
claimed.

## 6. Leakage controls and nested evaluation

For each of five outer Test folds, a distinct fold was reserved for Stop and the
remaining three folds alone supplied gradients. Normalization excluded Test,
Stop, and boundary-buffer support. Full 15×15 patch safety, 0.15 m center
separation, global 0.015 m ground-cell deduplication, and frozen 5 m groups were
enforced. Outer labels did not choose architecture, thresholds, bands, epochs,
or normalization.

The primary ensemble averages the three seed softmax vectors. Metrics use equal
weight per frozen spatial group. Confidence intervals use 1,000 paired cluster
bootstrap replicates; pixels are never treated as independent inferential units.

## 7. Architecture and inputs

`center_context_fusion` consumes 111 accepted channels spanning approximately
401.84–869.34 nm in a 15×15 patch. One branch applies an MLP to the complete
center spectrum. The context branch applies a compact CNN to the full spatial
patch. Their embeddings are concatenated and classified into three classes.

Input geometry is 111×15×15. Alley masks, polygons, soil masks, thresholds, and
spectral indices are label-construction or post-hoc analysis tools—not neural
network channels.

## 8. Normalization and hyperparameters

Nested models use fold-specific training-only mean and standard deviation.
Deployment normalization is recomputed separately for each label variant from
eligible primary Field 1 fitting centers only. Field 2 and cubes 12/14/15/24/28/32
contribute nothing.

Training uses AdamW, learning rate 0.0003, weight decay 0.0001, cross-entropy,
automatic mixed precision, batch 1,024, eight loader workers, and seeds 42–44.
Nested checkpoints use inner-validation early stopping. Deployment durations are
fixed before fitting from median nested selected epochs: 10 with-alley and 11
without-alley. No external calibration is fit; deployment probabilities are the
arithmetic mean of three uncalibrated softmax outputs.

Across the 15 nested runs per variant, mean inner-validation macro-F1/balanced
accuracy was 0.900044/0.900063 with alleys and 0.902413/0.902508 without
alleys. Standard deviations were 0.017250/0.017220 and 0.012285/0.012304,
respectively. Median selected epochs were 10 and 11. These inner-validation
values selected stopping duration inside each outer split; they are not the
outer-test result.

## 9. GPU execution and runtime

The explicit interpreter was
`/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python`. PyTorch
2.10.0+cu130 used CUDA runtime 13.0 with driver 580.173.02. Exactly two NVIDIA
RTX 5000 Ada Generation GPUs were visible. Independent processes owned one GPU
each; DataParallel and CPU fallback were prohibited.

Real patches passed forward/backward, checkpoint save/reload, and inference on
both GPUs for batches 256, 512, and 1,024. Batch 1,024 was the largest tested and
peaked at 0.353 GiB, far below the 25.6 GiB 20%-margin ceiling.

The six deployment fits consumed 285.23 summed seconds, averaged about 29–30k
training samples/s, peaked at 0.351 GiB, and averaged 145,444 reloaded-checkpoint
inference patches/s. Final optimization accuracy averaged 0.943641 with alleys
and 0.957018 without alleys. These are fitting diagnostics, not generalization
metrics. Training curves are stored in
`metadata/local/reports/supervised_finalization_v1/deployment/deployment_training_curves.png`.

## 10. Frozen Field 1 performance

| Model | Accuracy | Balanced accuracy | Macro-F1 | Soil F1 | Chickpea F1 | Weed F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Confident labels, with alleys | 0.929973 | 0.932545 | 0.876089 | 0.959268 | 0.856123 | 0.812877 |
| Confident labels, without alleys | 0.918150 | 0.884095 | 0.835364 | 0.956540 | 0.792535 | 0.757018 |
| Frozen legacy nested model | 0.809215 | 0.785344 | 0.648250 | 0.897941 | 0.327577 | 0.719230 |

Per-class precision/recall/F1 is:

| Model | Soil P/R/F1 | Chickpea P/R/F1 | Weed P/R/F1 |
| --- | --- | --- | --- |
| Confident, with alleys | 0.986714 / 0.933308 / 0.959268 | 0.771034 / 0.962321 / 0.856123 | 0.739779 / 0.902005 / 0.812877 |
| Confident, without alleys | 0.967188 / 0.946124 / 0.956540 | 0.679907 / 0.949886 / 0.792535 | 0.757761 / 0.756276 / 0.757018 |
| Frozen legacy | 0.992319 / 0.819956 / 0.897941 | 0.207686 / 0.774909 / 0.327577 | 0.681672 / 0.761168 / 0.719230 |

Against legacy, the with-alley model improved balanced accuracy by +0.148134
(0.120900–0.175723), macro-F1 by +0.228297 (0.207597–0.247755), chickpea F1 by
+0.529335 (0.481188–0.573297), and weed F1 by +0.094152
(0.069749–0.119792). Soil precision decreased by 0.005627 while soil recall
increased by 0.113446; this tradeoff is not concealed.

Row-normalized with-alley confusion is:

| True / predicted | Soil | Chickpea | Weed |
| --- | ---: | ---: | ---: |
| Soil | 0.933308 | 0.010335 | 0.056356 |
| Chickpea | 0.002835 | 0.962321 | 0.034844 |
| Weed | 0.067534 | 0.030461 | 0.902005 |

Fold macro-F1 was 0.933117, 0.924878, 0.810488, 0.804526, and 0.924061 for
outer folds 1–5. Cube 20 is the clear failure mode: macro-F1 0.431107, weed F1
0.208605, and chickpea F1 0.387989. The next worst cube macro-F1 was 0.902087.
The cube-level heterogeneity prohibits a universal transfer claim.

## 11. Calibration

The original exhaustive paired evaluator persisted class decisions rather than
individual probabilities, so full-primary NLL/Brier/ECE cannot be reconstructed
without a new exhaustive probability pass. On the fixed interpretability subset,
baseline averages across fold/seed were NLL 0.242591, Brier score 0.128392, and
15-bin ECE 0.037191. These subset values are descriptive and are not substituted
for blind external calibration.

## 12. Grouped wavelength importance

Thirty-eight contiguous physical-wavelength groups were audited. For every outer
fold and seed, each group was replaced by the fold-training mean and, separately,
interpolated from neighboring retained bands. Interventions were routed through
both branches, the center branch alone, and the context branch alone. Deltas are
paired within 5 m groups; 95% intervals use fold-stratified group bootstrap and
Benjamini–Hochberg correction within method/route/metric families.

Compact primary mean-replacement feature table:

| Wavelength interval | Center macro-F1 decrease | Context decrease | Both decrease | Chickpea-F1 decrease, both | Positive folds | Corrected status | Interpretation | Caveat |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 401.84–414.07 nm | 0.104755 | 0.207441 | 0.271966 | 0.561122 | 5/5 | stable | violet/blue-edge pigment or acquisition signal | remains stable under one-sided edge interpolation, but sensor-edge/illumination sensitivity is plausible |
| 664.81–673.37 nm | 0.070604 | 0.066070 | 0.161948 | 0.205337 | 5/5 | stable for mean replacement | red chlorophyll absorption | adjacent-band interpolation is not significant after correction |
| 677.65–686.23 nm | 0.043119 | 0.059599 | 0.130250 | 0.194813 | 5/5 | stable for mean replacement | red absorption shoulder | adjacent-band interpolation is exploratory after correction |
| 703.43–712.04 nm | 0.076004 | 0.003503 | 0.088364 | 0.022764 | 5/5 macro-F1 | stable for mean replacement | early red-edge transition | context effect and interpolation evidence are not stable |

Under adjacent-wavelength interpolation, only 401.84–414.07 nm remained stable
for macro-F1 across all three routes after correction; its both-branch decrease
was 0.119110 (0.105505–0.132697). This divergence from mean replacement is
evidence of local spectral redundancy, not permission to discard bands.

Adjacent hyperspectral bands are strongly correlated. Occlusion and attribution
magnitudes measure frozen-model reliance under a specific intervention; they do
not uniquely identify causal wavelengths or biochemical mechanisms.

## 13. Center-versus-context reliance

Neutralizing the center spectrum decreased macro-F1 by 0.670249
(0.633429–0.701318). Neutralizing context decreased it by 0.294874
(0.269927–0.317139). Neutralizing both decreased it by 0.670737
(0.634643–0.704117). Replacing context with its spatial mean decreased macro-F1
by 0.135626; deterministic within-patch spatial shuffling decreased it by
0.050914.

The center branch is indispensable, while the nonzero context-neutralization,
spatial-mean, and shuffle effects show reliance on contextual values and spatial
arrangement. These destructive frozen-model interventions are not equivalent to
training purpose-built center-only or context-only architectures. Historical
development comparators remain contextual evidence only: center-spectrum macro-
F1 0.646696, spatial-average CNN 0.635110, and fusion 0.671206 under the earlier
label contract.

## 14. Fixed spectral-index audit

Exact nearest wavelengths are 550.54, 669.09, 720.67, and 798.91 nm. NDVI,
GNDVI, NDRE, and NIR/red ratio were computed as fixed dimensionless descriptors;
no threshold was optimized. None is a neural-network input.

NDVI one-versus-rest direction-free AUC was 0.965873 for soil, 0.885466 for
chickpea, and 0.619505 for weed; multiclass mutual information was 0.528910.
GNDVI AUC was 0.895963/0.819566/0.609055, and NDRE was
0.762868/0.741963/0.544710. The indices separate soil/vegetation and some
chickpea structure better than weed, but do not establish a fixed classifier.

SAVI with L=0.5 was deliberately not reported. The ENVI metadata gives no unit-
reflectance scaling provenance; adding the dimensioned 0.5 constant to stored
DN-like values would be scientifically uninterpretable. This is an explicit
missing-provenance result, not a failed threshold search.

| Index | Confident-label construction | Field 2 annotation sampling | Neural-network input | Post-hoc interpretation |
| --- | --- | --- | --- | --- |
| NDVI | No | No—Field 2 not opened | No | Yes, descriptive only |
| GNDVI | No | No—Field 2 not opened | No | Yes, descriptive only |
| NDRE | No | No—Field 2 not opened | No | Yes, descriptive only |
| NIR/red ratio | No | No—Field 2 not opened | No | Yes, descriptive only |
| SAVI, L=0.5 | No | No—Field 2 not opened | No | Proposed but not computed because scale provenance is missing |

## 15. Deployment ensemble freeze

The with-alley deployment population contains 144,744 unique, thinned centers:
89,573 soil, 16,395 chickpea, and 38,776 weed. Without-alley contains 124,649:
89,591, 16,486, and 18,572. All eligible primary support is retained after the
prespecified safety/deduplication/thinning rules; there is no oversampling.

Six checkpoints are frozen, reload-verified, and SHA-256 manifested under
`results/field1_deployment_v1/`. The primary deployment ensemble is with-alley;
without-alley is the deployment comparator. A legacy deployment model is absent
because reconstructing one now would change the historical scientific protocol.
The historical nested legacy metric remains the honest comparator.

## 16. Field 2 compatibility gate — mandatory stop

Current authorized inventory is exact:

- `configs/paths.local.yaml` has `field2.imagery: ""`,
  `field2.tall_grass_rtk: ""`, and `field2.locked: true`.
- The authorized manifest has zero Field 2 rows.
- No Field 2 header, scientific array, coordinate, label, mask, preview, or model
  output was opened.
- A pre-existing locked archive metadata inventory is not an authorized or
  analysis-ready scientific source.

Compatibility status is `blocked_not_georectified_or_authorized`. Wavelength,
band-order, radiometric, CRS/transform, NoData, GSD, and preview checks were not
run because doing so would require opening unauthorized Field 2 products.

Exact products required before compatibility QC:

1. An immutable cube manifest with unique IDs, acquisition/sensor/processing
   provenance, and relative paths.
2. For every cube, a paired georectified ENVI `.hdr` and `.bip`. The header must
   declare dimensions, 150 bands, BIP interleave, data type, byte order,
   wavelength vector/units, map info, and NoData semantics.
3. A wavelength QC table proving increasing nanometer order and compatibility
   for model channels 3–113: 111 channels spanning 401.84–869.34 nm. Any spectral
   resampling rule and tolerance must be frozen before labels are opened.
4. Radiometric provenance: units, scale, calibration process, dark/white
   reference handling, saturation encoding, and compatibility with Field 1.
5. Georeferencing provenance: CRS/WKT, affine transform, pixel size, bounds, and
   error/residual summary. Field 1 primary compatibility targets EPSG:32617 and
   nominal 0.015 m GSD; deviations require a frozen decision.
6. Explicit NoData/saturation definitions and aligned valid-data support capable
   of screening complete 15×15 patches.
7. A prediction-free cube-footprint/overlap index in the declared CRS.
8. SHA-256 and byte-size manifest for every supplied product.
9. Prediction-free NIR-red-green/natural-color source views or a frozen recipe
   to generate them from the georectified cube.
10. Prediction-free PCA preview sources/recipe with PCA fit scope and component
    sign/orientation provenance.
11. An authorized Field 2 annotation boundary/exclusion layer independent of
    supervised outputs.
12. After imagery compatibility—not before—an immutable investigator point file
    supporting soil, chickpea, weed, uncertain, invalid/NoData, confidence,
    blinded duplicates, inclusion probabilities, and checksums.

The machine-readable checklist is
`metadata/local/reports/supervised_finalization_v1/field2_georectified_product_checklist.csv`.
Supplying these products authorizes only a read-only compatibility audit unless a
later instruction explicitly changes the lock. It does not authorize training,
normalization, annotation sampling, or prediction.

## 17. Previously inspected Field 1 early-growth stress test — not blind external validation

No new Cube 24/28 processing was performed. The already-frozen with-alley
ensemble result on the pooled sensitivity role was accuracy 0.966099, balanced
accuracy 0.949009, and macro-F1 0.898512. Soil precision/recall/F1 was
0.981367/0.980700/0.981034, chickpea was
0.661055/0.938112/0.775583, and weed was
0.949872/0.928214/0.938918. Supports were 555,089 soil, 12,054 chickpea, and
205,041 weed pixels under the historical sensitivity evaluation. These values
were known before this finalization, are not newly generated blind evidence,
and are never pooled with the primary nested metrics.

## 18. Pending external sections

Blind Field 2 performance, external confusion matrices, annotation repeatability,
weighted prevalence metrics, and block-cluster confidence intervals are not
available and are not fabricated. Cubes 24/28 are not rerun or relabeled as a
blind set. Their prior sensitivity result remains explicitly titled “Previously
inspected Field 1 early-growth stress test — not blind external validation” and
is never pooled with primary or future external metrics.

## 19. Limitations and error analysis

- Confident labels are selective weak supervision, not exhaustive botanical
  ground truth; unresolved pixels are excluded.
- Cube 20 remains a major geographic failure mode.
- Cube 32 demonstrates transfer instability and retains sparse-only authority.
- Fold and cube variability matters more than seed variability; pixels are not
  independent replicates.
- The importance subset is balanced across local support and cannot reproduce
  exhaustive prevalence metrics.
- Mean replacement is a stronger distribution shift than interpolation.
- Physical explanations for wavelength reliance remain hypotheses.
- No blind external evidence currently exists.

## 20. Reproduction commands

Run from `/home/hemanthd95/Chickpea_modeling` with the explicit interpreter:

```bash
/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/freeze_confident_supervised_benchmark.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --finalization configs/supervised_finalization_v1.yaml --bands configs/spectral_bands.yaml

MPLCONFIGDIR=/tmp/chickpea_matplotlib /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_grouped_importance_audit.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --finalization configs/supervised_finalization_v1.yaml --bands configs/spectral_bands.yaml --mode full

/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/prepare_field1_deployment_data.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --finalization configs/supervised_finalization_v1.yaml --bands configs/spectral_bands.yaml

MPLCONFIGDIR=/tmp/chickpea_matplotlib /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/train_field1_deployment_ensembles.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --finalization configs/supervised_finalization_v1.yaml --bands configs/spectral_bands.yaml --mode preflight

MPLCONFIGDIR=/tmp/chickpea_matplotlib /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/train_field1_deployment_ensembles.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --finalization configs/supervised_finalization_v1.yaml --bands configs/spectral_bands.yaml --mode full

/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/freeze_field2_compatibility_gate.py --paths configs/paths.local.yaml --finalization configs/supervised_finalization_v1.yaml

/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/validate_supervised_finalization.py --paths configs/paths.local.yaml --finalization configs/supervised_finalization_v1.yaml

/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python -m pytest -q
```

## 21. Contracts, manifests, Git reference, and next protocol

Machine-readable artifacts are under
`metadata/local/contracts/supervised_finalization_v1/`; reports and figures are
under `metadata/local/reports/supervised_finalization_v1/`. The benchmark,
interpretability, deployment-data, GPU-preflight, deployment-ensemble, checkpoint-
manifest, and Field 2 stop contracts all record the benchmark SHA and Field 2
lock state.

The immutable scientific reference and annotated tag are
`da95fd59b7dc5d8ba4add96ed43b398bb7c93749` and
`supervised-confident-v1-da95fd5`. Finalization code is committed only on the
descendant branch `agent/ssl-multicube-setup`; reproductions should record the
then-current `git rev-parse HEAD` alongside the immutable benchmark reference.

Self-supervised learning was not started. A later SSL protocol should remain
Field-1-only during development, use the frozen supervised benchmark without
retrospective reinterpretation, keep Field 2 excluded from pretraining for this
inductive benchmark, and predeclare its representation, clustering, and external
evaluation rules before any Field 2 labels or predictions are opened.
