# Data dictionary

| Source key | Intended content | Experimental role |
|---|---|---|
| `field1.derived_ssl` | PCA, PCA-component differences, NDVI, masks | Legacy/baseline analysis |
| `field1.label_csvs` | Cube-level chickpea, weed, and soil mask details | Label audit and supervised baselines |
| `field1.masks_emmanuel` | Completed Field 1 mask products | Development labels |
| `field1.reflectance_processing_batches` | Original georectified Pika-L cubes grouped by student processing date | Primary model inputs; all acquired 2025-05-06 |
| `field2.imagery` | Untouched Field 2 imagery | Locked external deployment |
| `field2.tall_grass_rtk` | RTK tall-grass coordinates | Revealed only after model freeze |

The local preflight creates machine-readable file inventories under
`metadata/local/`. Each row records source key, acquisition date, relative path,
extension, size, and inferred cube identifier.

Label codes are defined in `configs/data_decisions.yaml`: 1 = soil, 2 = chickpea,
and 3 = weed. Excluded duplicate table associations and mask variants are also
recorded there with reasons.

The primary hyperspectral band policy is stored in `configs/spectral_bands.yaml`.
Supporting files extracted from the OneDrive archive are stored separately under
`data/OneDrive_2026-07-31_raw/`; their local inventory is not committed.
