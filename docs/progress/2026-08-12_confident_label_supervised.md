# Confident-label supervised learning — 2026-08-12

## Status

The Field 1 confident-label pipeline completed its prerequisite CUDA gate,
real-data smoke test, nested fitting, and paired held-out evaluation. The result
is a validated, provenance-preserving candidate dataset and benchmark; it is not
an unqualified replacement for the source masks. Field 2 remained locked.

All repository Python commands used the project interpreter directly:

```text
/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python
```

The host exposed exactly two `NVIDIA RTX 5000 Ada Generation` GPUs. PyTorch was
2.10.0+cu130, its CUDA runtime was 13.0, and `torch.cuda.is_available()` was
true. Training used independent single-GPU processes assigned to `cuda:0` and
`cuda:1`; `torch.nn.DataParallel` was not used and CPU fallback was prohibited.

## Frozen label semantics

The categorical output uses soil = 0, chickpea = 1, weed = 2, and unresolved =
255. The middle probability interval remains unresolved. The configured
vegetation thresholds are P(chickpea) <= 0.15 for confident weed and
P(chickpea) >= 0.80 for confident chickpea. Polygon edges use a 0.05 m uncertain
band. Investigator-drawn alleys have precedence over chickpea, and a 0.05 m
inward alley buffer defines the primary alley core.

Every resolved pixel records one provenance code. The sources distinguish
authoritative soil, scalar-index expansion soil, alley-core soil, alley-core
weed, polygon-core probability chickpea/weed, and investigator-confirmed
chickpea/weed points. NoData is always unresolved.

Vegetation probabilities are fitted separately for each outer evaluation role.
Both the outer Test fold and separate inner Stop fold are excluded from fitting.
Cube 32 is never densely resolved by its probability raster: its 26 accepted
investigator points remain sparse sensitivity evidence only. Cubes 24 and 28
are phenology sensitivity cubes. Cubes 12, 14, and 15 are positive-support
expansion cubes and cannot create confident weed labels. These roles are
disjoint from the primary evaluation population.

## Materialized support

Across all declared roles, the candidate contains 6,930,741 soil, 441,995
chickpea, 1,440,422 weed, and 19,322,295 unresolved pixels. The primary role
contains 5,749,823 soil, 322,196 chickpea, 1,080,446 weed, and 12,027,000
unresolved pixels.

The configured alley core contributes 1,147,513 final weed-provenance pixels and
marks 1,717,525 already resolved soil pixels with alley-core provenance. It does
not manufacture additional soil where authoritative soil already resolved the
same cells. Alley precedence removes 100,783 historical chickpea pixels from
eligible chickpea support. No confident chickpea remains inside a full alley.

The inward-buffer sensitivity is:

| Inward buffer | Core pixels | Soil | Weed |
| ---: | ---: | ---: | ---: |
| 0.00 m | 3,000,250 | 1,796,211 | 1,204,039 |
| 0.03 m | 2,936,146 | 1,759,259 | 1,176,887 |
| 0.05 m | 2,865,043 | 1,717,525 | 1,147,518 |
| 0.10 m | 2,743,972 | 1,644,641 | 1,099,331 |

The five-pixel difference between the configured core weed count and final
alley-weed provenance is due to final point/role resolution bookkeeping; both
counts are retained rather than forced to agree. The complete cube, fold,
spatial-group, class, and provenance counts are in
`metadata/local/reports/confident_labels_v1/`.

## Leakage controls and sampling

The nested protocol reserves one outer fold for Test and a different fold for
Stop. Only the remaining three folds supply gradients. Normalization excludes
both Test and Stop plus the declared boundary buffer. A 15 x 15 patch is accepted
only when its full context is safe. Training centers are separated by at least
0.15 m and deduplicated on a global 0.015 m ground grid, so overlapping cube
observations cannot cross roles or be counted twice.

There is no oversampling. Each role is balanced down to the scarce class, so the
effective per-class Fit/Stop counts vary by outer fold:

| Outer fold | With-alley Fit / Stop | Without-alley Fit / Stop |
| ---: | ---: | ---: |
| 1 | 10,825 / 3,116 | 10,891 / 3,146 |
| 2 | 9,649 / 3,641 | 9,717 / 3,641 |
| 3 | 9,579 / 2,357 | 9,617 / 2,406 |
| 4 | 9,366 / 3,287 | 9,398 / 3,292 |
| 5 | 9,633 / 3,023 | 9,715 / 3,023 |

The primary evaluation is exhaustive after the same ground-cell deduplication:
5,924,107 observations total across the five outer folds.

## CUDA engineering gate and training

The real-data benchmark tested batches 256, 512, and 1,024. Batch 1,024 was
selected at 14,897.58 samples/s and 0.351 GiB peak allocated VRAM. The required
one-batch, full-epoch, validation, checkpoint-save, and checkpoint-reload smoke
test passed before the full run. Its full-epoch throughput was 10,735.98
samples/s, peak VRAM was 0.351 GiB, and the projected 15-checkpoint runtime was
0.11 hours.

The full run trained three seeds for each of five outer folds for both the
with-alley and without-alley variants: 30 checkpoints total. Sum checkpoint
runtime was 670.23 seconds; estimated two-GPU wall time was 398.83 seconds. Mean
training throughput was 18,096.70 samples/s and maximum peak allocated VRAM was
0.361 GiB. Progress, device name, throughput, memory, and estimated runtime were
printed by every worker and retained in the timing reports.

## Primary held-out result

Metrics below use the same exhaustive, deduplicated primary ground cells and
equal-weight 5 m spatial groups. The cleaned model is the three-seed ensemble.

| Model | Balanced accuracy | Macro-F1 | Soil F1 | Chickpea F1 | Weed F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Confident labels, with alleys | 0.932545 | 0.876089 | 0.959268 | 0.856123 | 0.812877 |
| Confident labels, without alleys | 0.884095 | 0.835364 | 0.956540 | 0.792535 | 0.757018 |
| Frozen legacy supervised model | 0.785344 | 0.648250 | 0.897941 | 0.327577 | 0.719230 |

Against the frozen legacy model, the paired spatial-group bootstrap estimates a
balanced-accuracy difference of +0.148134 (95% interval 0.120900 to 0.175723)
and macro-F1 difference of +0.228297 (0.207597 to 0.247755). Chickpea F1 improves
by +0.529335 (0.481188 to 0.573297), and weed F1 improves by +0.094152 (0.069749
to 0.119792). Soil precision is 0.005627 lower, while soil recall is 0.113446
higher; the tradeoff is preserved in the paired report.

With-alley versus without-alley training improves balanced accuracy by +0.048362
(0.039377 to 0.057520), macro-F1 by +0.041054 (0.026683 to 0.055562), chickpea
F1 by +0.064463 (0.042634 to 0.090731), and weed recall by +0.145346 (0.118236
to 0.173251). The weed-precision interval crosses zero, so a precision benefit
is not claimed.

Geographic variation remains material. With-alley fold macro-F1 ranges from
0.804526 to 0.933117, and its worst cube macro-F1 is 0.431107. Across the three
individual seeds, balanced accuracy is 0.926703 +/- 0.002797 and macro-F1 is
0.869421 +/- 0.003361. Folds, not seeds or pixels, remain the primary geographic
replicates.

## Sensitivity findings and limitations

- Cubes 24 and 28 achieve ensemble balanced accuracy 0.949009 and chickpea F1
  0.775583, but remain phenology sensitivity results.
- Expansion Cubes 12, 14, and 15 achieve chickpea F1 0.936350 on their declared
  positive-support role. This is not an exhaustive negative-class test.
- Cube 32 remains a failure/transfer warning. Only 13 chickpea and 13 weed
  investigator points are authoritative there; scoring dense predictions against
  that sparse support produces chickpea precision 0.003034 and is not comparable
  to the primary dense population.
- Investigator-reference-only summaries omit soil reference support. Their
  three-class balanced accuracy and macro-F1 therefore should not be interpreted;
  class-specific vegetation metrics are the meaningful quantities.
- Confident labels are selective, not exhaustive truth. Unresolved pixels stay out
  of fitting and reported support, and point-level threshold precision is not
  treated as proof of dense-pixel accuracy.
- The benchmark substantially improves on the frozen legacy baseline, but the
  weakest cubes and the Cube 32 transfer warning prohibit a claim of universal
  Field 1 generalization. No Field 2 transfer claim is made.

## Validation gates

All automated gates passed: Field 2 lock, input/output hashes, threshold policy,
sample/evaluation/normalization hashes, deterministic rerun, zero chickpea in
alleys, zero labels in NoData, Cube 32 dense-resolution prohibition, role
separation, ground-cell deduplication, valid classes, fold-specific reference
separation, patch QC, and CUDA smoke completion. The repository test suite also
passes: 41 tests.

## Exact reproduction commands

Run from `/home/hemanthd95/Chickpea_modeling`. Keep the explicit interpreter;
do not rely on `conda activate` persisting.

```bash
nvidia-smi

/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Visible GPUs:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/build_confident_label_dataset.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --bands configs/spectral_bands.yaml

/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/prepare_confident_supervised_data.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --bands configs/spectral_bands.yaml

CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/chickpea_matplotlib /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_confident_supervised_training.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --bands configs/spectral_bands.yaml --mode benchmark

CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/chickpea_matplotlib /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_confident_supervised_training.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --bands configs/spectral_bands.yaml --mode smoke

MPLCONFIGDIR=/tmp/chickpea_matplotlib /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/validate_confident_label_gates.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --bands configs/spectral_bands.yaml --require-cuda-smoke

MPLCONFIGDIR=/tmp/chickpea_matplotlib /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_confident_supervised_training.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --bands configs/spectral_bands.yaml --mode full --variants with_alley,without_alley

MPLCONFIGDIR=/tmp/chickpea_matplotlib /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/evaluate_confident_supervised_models.py --paths configs/paths.local.yaml --config configs/confident_labels_v1.yaml --bands configs/spectral_bands.yaml --mode full

/home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python -m pytest -q
```

Generated datasets, contracts, checkpoints, and reports remain in ignored local
data/result directories. Source masks and the authoritative manifest are not
modified.
