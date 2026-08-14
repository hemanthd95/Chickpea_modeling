# Field 2 georectified-data readiness audit

Date: 2026-08-14

Audit code commit at execution: `c5a69bef38a71a29306c4005b65531b27a7539ae`

Status: **prediction-free read-only readiness audit complete**

## Scope and safeguards

The audit inspected only the three explicitly authorized Field 2 roots. A complete SHA-256 snapshot was written before raster arrays were opened. All ENVI payloads were accessed through read-only NumPy memory maps; Rasterio was used in reader mode with `GDAL_PAM_ENABLED=NO`. Output paths were rejected if they resolved inside a source root.

No model checkpoint was loaded. No prediction, probability, categorical label, annotation point, model training, or SSL experiment was generated. The frozen Field 1 benchmark and its artifacts were not modified.

## Inventory

- Cubes (40): field2_cube02, field2_cube03, field2_cube04, field2_cube05, field2_cube06, field2_cube07, field2_cube08, field2_cube09, field2_cube11, field2_cube12, field2_cube13, field2_cube14, field2_cube16, field2_cube17, field2_cube18, field2_cube19, field2_cube20, field2_cube21, field2_cube22, field2_cube23, field2_cube24, field2_cube25, field2_cube26, field2_cube27, field2_cube28, field2_cube29, field2_cube30, field2_cube31, field2_cube32, field2_cube33, field2_cube34, field2_cube35, field2_cube36, field2_cube37, field2_cube38, field2_cube39, field2_cube40, field2_cube41, field2_cube42, field2_cube43
- Products: {"pca": 40, "reflectance": 40, "stored_index": 40}
- Headers: 120; payloads: 120; valid one-to-one pairs: 120
- Missing/ambiguous pairs: 0

Derivative PCA, separate valid-data masks, georeferencing sidecars, and georectification residual/control-point reports were not supplied. Their absence is reported and was not inferred to mean acquisition failure.

## Integrity and spectral compatibility

All payload sizes matched ENVI dimensions, data type, interleave, byte order, and header offset: **True**. Reflectance wavelength rows matched the frozen Field 1 band indices 3–113 within the configured tolerance: **True**. No spectral reordering, interpolation, truncation, or resampling was performed.

## Spatial alignment and overlap

Grid-alignment status counts: {"exact_alignment": 80}. CRS, affine, dimensions, GSD, bounds, rotation/shear, and pixel-center alignment were compared directly. No product was resampled. Georectification residuals/control-point errors were `not_supplied`.

Although all grids align exactly, PCA and stored-index products contain transformed, nonzero values outside the reflectance raster's zero-filled footprint. Raw support agreement is 0.442–0.519. The diagnostic previews apply the operational reflectance support for display, but this is not treated as an authoritative NoData definition.

Cross-cube footprint overlaps are reported in `field2_cube_overlap.csv`; source cubes were not deduplicated or changed.

## Radiometry, NoData, and sampled QC

Each reflectance header records measured-reference correction with 100% reflectivity scaled to 10,000 stored integer units. The history also records a sensor saturation value of 4095 before reflectance correction, but does not state a post-correction saturation encoding. Therefore saturation fractions are left unresolved rather than guessed. Field 2 radiometric processing is documented, but compatibility with Field 1 scale remains unresolved because the frozen Field 1 report states its headers do not prove unit-reflectance scale.

No explicit ENVI `data ignore value` or separate valid-data mask was supplied. Operational preview support uses finite, nonzero reflectance at the documented preview bands only to suppress obvious display background; it remains distinct from an authoritative NoData rule. Sampled per-band statistics quantify zero, nonfinite, and constant fractions. Stored index files are described as scalar-index products: the processing history names an NDVI transform but does not supply the exact formula and source-band provenance.

Warnings observed: derived_product_nonzero_outside_reflectance_support, high_sampled_zero_fraction, low_operational_valid_fraction, no_explicit_nodata_or_valid_mask.

## Annotation readiness

Annotation-ready cubes: none. The previews are suitable for investigator review, but annotation is blocked until an authoritative valid-data/NoData rule or aligned mask is supplied and frozen.

Blocked/excluded cubes:
| cube_id | exclusion_reason |
|---|---|
| field2_cube02 | unresolved_nodata_and_valid_support_definition |
| field2_cube03 | unresolved_nodata_and_valid_support_definition |
| field2_cube04 | unresolved_nodata_and_valid_support_definition |
| field2_cube05 | unresolved_nodata_and_valid_support_definition |
| field2_cube06 | unresolved_nodata_and_valid_support_definition |
| field2_cube07 | unresolved_nodata_and_valid_support_definition |
| field2_cube08 | unresolved_nodata_and_valid_support_definition |
| field2_cube09 | unresolved_nodata_and_valid_support_definition |
| field2_cube11 | unresolved_nodata_and_valid_support_definition |
| field2_cube12 | unresolved_nodata_and_valid_support_definition |
| field2_cube13 | unresolved_nodata_and_valid_support_definition |
| field2_cube14 | unresolved_nodata_and_valid_support_definition |
| field2_cube16 | unresolved_nodata_and_valid_support_definition |
| field2_cube17 | unresolved_nodata_and_valid_support_definition |
| field2_cube18 | unresolved_nodata_and_valid_support_definition |
| field2_cube19 | unresolved_nodata_and_valid_support_definition |
| field2_cube20 | unresolved_nodata_and_valid_support_definition |
| field2_cube21 | unresolved_nodata_and_valid_support_definition |
| field2_cube22 | unresolved_nodata_and_valid_support_definition |
| field2_cube23 | unresolved_nodata_and_valid_support_definition |
| field2_cube24 | unresolved_nodata_and_valid_support_definition |
| field2_cube25 | unresolved_nodata_and_valid_support_definition |
| field2_cube26 | unresolved_nodata_and_valid_support_definition |
| field2_cube27 | unresolved_nodata_and_valid_support_definition |
| field2_cube28 | unresolved_nodata_and_valid_support_definition |
| field2_cube29 | unresolved_nodata_and_valid_support_definition |
| field2_cube30 | unresolved_nodata_and_valid_support_definition |
| field2_cube31 | unresolved_nodata_and_valid_support_definition |
| field2_cube32 | unresolved_nodata_and_valid_support_definition |
| field2_cube33 | unresolved_nodata_and_valid_support_definition |
| field2_cube34 | unresolved_nodata_and_valid_support_definition |
| field2_cube35 | unresolved_nodata_and_valid_support_definition |
| field2_cube36 | unresolved_nodata_and_valid_support_definition |
| field2_cube37 | unresolved_nodata_and_valid_support_definition |
| field2_cube38 | unresolved_nodata_and_valid_support_definition |
| field2_cube39 | unresolved_nodata_and_valid_support_definition |
| field2_cube40 | unresolved_nodata_and_valid_support_definition |
| field2_cube41 | unresolved_nodata_and_valid_support_definition |
| field2_cube42 | unresolved_nodata_and_valid_support_definition |
| field2_cube43 | unresolved_nodata_and_valid_support_definition |

All `investigator_role` entries remain `unreviewed`; no biological role was inferred from cube number, filename, image content, or statistics. The future vocabulary is `soil`, `weed`, `tall_grass_weed`, `chickpea`, `chickpea_soil_mixed`, `chickpea_weed_mixed`, `uncertain`, and `nodata_invalid`.

## Source immutability result

Independent before/after snapshot differences: []. Final status: **PASS**.

## Reproduction

```bash
GDAL_PAM_ENABLED=NO MPLCONFIGDIR=/tmp/chickpea_matplotlib /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/audit_field2_georectified_inventory.py --paths configs/paths.local.yaml --config configs/field2_readiness.yaml
```

Main machine-local tables and previews are under `metadata/local/reports/field2_readiness`; the machine-readable contract is `metadata/local/contracts/field2_georectified_inventory_contract.yaml`.
