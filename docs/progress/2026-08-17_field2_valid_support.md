# Field 2 reflectance valid-support freeze

Date: 2026-08-17

Status: **all cubes annotation-ready (prediction-free)**

## Scientific rule

The frozen rule is `finite_positive_norm_gt_0`: all zero-based Python bands 3–113 (ENVI bands 4–114) must be finite, and the maximum absolute stored reflectance across those bands must exceed zero. It uses reflectance only. PCA and the stored scalar index were display/diagnostic layers and never mask inputs.

The selected robustness comparator is `finite_positive_norm_gt_1`. Maximum per-cube disagreement was 0.000000000; the configured gate is 0.001000. Thresholds tested were [0, 1, 5, 10], together with finite-any-nonzero, finite-all-nonzero, and full-spectrum-any-nonzero.

All candidate rules agreed exactly. Thus low-amplitude nonzero pixels did not form a fringe: no valid pixel had maximum model-band magnitude at or below 10 stored units. Cube 02 contains one main body, two coherent non-small upper scan strips, and two tiny boundary remnants; all are retained as observed support. No morphology, component removal, hole filling, convex hull, or biological assumption was applied.

## Per-cube support

| cube_id | valid_fraction | components | largest_component_fraction | status |
|---|---:|---:|---:|---|
| field2_cube02 | 0.441651 | 5 | 0.950210 | annotation_ready_prediction_free |
| field2_cube03 | 0.492360 | 2 | 0.999857 | annotation_ready_prediction_free |
| field2_cube04 | 0.476512 | 2 | 0.999830 | annotation_ready_prediction_free |
| field2_cube05 | 0.486966 | 2 | 0.999879 | annotation_ready_prediction_free |
| field2_cube06 | 0.480592 | 2 | 0.999868 | annotation_ready_prediction_free |
| field2_cube07 | 0.469087 | 2 | 0.999886 | annotation_ready_prediction_free |
| field2_cube08 | 0.473529 | 2 | 0.999825 | annotation_ready_prediction_free |
| field2_cube09 | 0.470734 | 2 | 0.999949 | annotation_ready_prediction_free |
| field2_cube11 | 0.508397 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube12 | 0.484749 | 2 | 0.999920 | annotation_ready_prediction_free |
| field2_cube13 | 0.480490 | 2 | 0.999982 | annotation_ready_prediction_free |
| field2_cube14 | 0.479370 | 2 | 0.999843 | annotation_ready_prediction_free |
| field2_cube16 | 0.483797 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube17 | 0.476809 | 2 | 0.999981 | annotation_ready_prediction_free |
| field2_cube18 | 0.455579 | 2 | 0.999863 | annotation_ready_prediction_free |
| field2_cube19 | 0.477029 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube20 | 0.492988 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube21 | 0.467470 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube22 | 0.474286 | 2 | 0.999854 | annotation_ready_prediction_free |
| field2_cube23 | 0.485278 | 2 | 0.999777 | annotation_ready_prediction_free |
| field2_cube24 | 0.460779 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube25 | 0.492231 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube26 | 0.455419 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube27 | 0.496501 | 2 | 0.999843 | annotation_ready_prediction_free |
| field2_cube28 | 0.489434 | 2 | 0.999827 | annotation_ready_prediction_free |
| field2_cube29 | 0.487757 | 2 | 0.999981 | annotation_ready_prediction_free |
| field2_cube30 | 0.483424 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube31 | 0.507346 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube32 | 0.481380 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube33 | 0.480283 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube34 | 0.476847 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube35 | 0.517691 | 2 | 0.999930 | annotation_ready_prediction_free |
| field2_cube36 | 0.488919 | 2 | 0.999991 | annotation_ready_prediction_free |
| field2_cube37 | 0.486668 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube38 | 0.479483 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube39 | 0.498984 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube40 | 0.478041 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube41 | 0.485288 | 2 | 0.999964 | annotation_ready_prediction_free |
| field2_cube42 | 0.478462 | 1 | 1.000000 | annotation_ready_prediction_free |
| field2_cube43 | 0.518598 | 1 | 1.000000 | annotation_ready_prediction_free |

## Transformed products

PCA/index rasters were not altered or resampled. Their nonzero pixels outside reflectance support were counted and suppressed only in review displays:

| product | total nonzero | nonzero outside support | finite inside | nonfinite inside |
|---|---:|---:|---:|---:|
| pca | 35393403 | 18349592 | 17043811 | 0 |
| stored_index | 35393237 | 18349592 | 17043811 | 0 |

## Integrity and prohibited operations

Materialized masks: 40. Annotation-ready cubes: 40; blocked cubes: 0. Source before/after differences: `[]`.

No source product was modified. No biological label, supervised prediction, probability, model retraining, or SSL training was generated. The frozen Field 1 benchmark was unchanged.

Generated machine-local CSVs, previews, masks, and the YAML contract are intentionally ignored by Git.
