# Data dictionary

| Source key | Intended content | Experimental role |
|---|---|---|
| `field1.derived_ssl` | PCA, PCA-component differences, NDVI, masks | Legacy/baseline analysis |
| `field1.label_csvs` | Cube-level chickpea, weed, and soil mask details | Label audit and supervised baselines |
| `field1.masks_emmanuel` | Completed Field 1 mask products | Development labels |
| `field1.reflectance_dates` | Original georectified Pika-L cubes and ENVI headers | Primary model inputs |
| `field2.imagery` | Untouched Field 2 imagery | Locked external deployment |
| `field2.tall_grass_rtk` | RTK tall-grass coordinates | Revealed only after model freeze |

The local preflight creates machine-readable file inventories under
`metadata/local/`. Each row records source key, acquisition date, relative path,
extension, size, and inferred cube identifier.

Label codes are defined in `configs/data_decisions.yaml`: 1 = soil, 2 = chickpea,
and 3 = weed. Excluded duplicate table associations and mask variants are also
recorded there with reasons.
