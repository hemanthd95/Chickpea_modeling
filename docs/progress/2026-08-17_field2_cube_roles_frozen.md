# Field 2 investigator cube roles frozen

Date: 2026-08-17

Starting repository commit: `8f06d4d413d600b89d4507297e0c6849f6ac9cfe`

Freeze implementation and execution commit: `bcfefaa4bb702879e95fbaec297fd18d6e7b6afe`

Status: **all 40 prediction-free investigator decisions frozen; point sampling not generated**

## Pre-freeze safeguards

The remote branch was fetched before work began; local and `origin/agent/ssl-multicube-setup` initially had divergence `0 0`. Existing changes to `train_chickpea_classifier.py` and unrelated untracked files were preserved and excluded from both commits.

The frozen Field 1 finalization validator passed without changing any contract or artifact. It verified benchmark commit `da95fd59b7dc5d8ba4add96ed43b398bb7c93749`, tag `supervised-confident-v1-da95fd5`, 15 nested checkpoints, and 6 deployment checkpoints by identity and SHA-256. No checkpoint was loaded or deserialized as a model.

The Field 2 read-only revalidation passed for the exact configured 40 cubes, all 240 source files, and all 40 frozen uint8 support masks. Current source, reflectance, and support-mask hashes match their manifests. Masks contain only `{0, 1}` and retain their frozen grids. The governing hashes remain:

- Source manifest: `0dbe286a152748f0ab8d338b095b308d977cfcc3568999b86e7f9f4e306ee44e`
- Valid-support manifest: `1f48ffe67d7ad2cbf95cca5c2e6e98d15245843435c519bbb8a67841c2bbf3e3`
- Review-package manifest: `81de7834c1e3f538a5b1b19f905dbda36521baab5a14e84da261678a4fdceaf2`

The localhost reviewer was stopped before the input hashes were captured. Saved review revision 86 contains 40 unique expected IDs in exact order. All 40 records have `reviewed=true`, a non-`unreviewed` primary role, a populated confidence and timestamp, and source-preview references. JSON and CSV agree field-for-field. The audit contains 40 `pass` rows, 40 matching-preview results, and no issue text. Direct rehashing of all preview layers found no changes.

## Frozen investigator decisions

All 40 confidence values are `high`. Decisions were copied exactly; no role, flag, confidence, note, exclusion reason, timestamp, reviewer identifier, or preview reference was inferred or corrected.

| Primary role | Count | Cubes |
|---|---:|---|
| `primary_three_class` | 20 | 07, 08, 09, 11, 12, 13, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 33, 34 |
| `challenge_only` | 8 | 03, 04, 06, 14, 16, 23, 24, 32 |
| `chickpea_absent_negative_control` | 7 | 02, 05, 31, 39, 41, 42, 43 |
| `sensitivity_only` | 5 | 35, 36, 37, 38, 40 |
| `exclude_with_reason` | 0 | none |
| `unreviewed` | 0 | none |

| Investigator flag | Count |
|---|---:|
| `chickpea_visible` | 20 |
| `chickpea_very_small_or_immature` | 14 |
| `chickpea_absent_confirmed` | 7 |
| `ordinary_weed_visible` | 40 |
| `tall_grass_weed_visible` | 34 |
| `soil_visible` | 40 |
| `mixed_pixels_common` | 0 |
| `annotation_difficult` | 0 |
| `georectification_concern` | 0 |
| `other_visual_concern` | 0 |

## Frozen inputs and outputs

Input review hashes:

- JSON: `67f0627e0fcffa199676c7996445a0c68e02a72d1d5c9cf6202279c55506be33`
- CSV: `24f628e8bc9bcbea60a51698f1abd5fb122786e895db3c797d90f7d6a3ebb973`
- Audit CSV: `5268ce4a3cb0e50ae025e66764062df4b70479037a0daabefa285d69acd02fc6`
- Saved overview: `999de677fbd7f5e51f67647ee7f0fb071a819d3f9b8d9ff6699eb2efe51972e8`

The write-once freeze command generated:

- Contract: `metadata/local/contracts/field2_cube_role_contract.yaml`
- Frozen table: `metadata/local/contracts/field2_cube_role_table.csv`
- Summary: `metadata/local/reports/field2_cube_role_freeze/field2_cube_role_summary.csv`
- Frozen overview: `metadata/local/reports/field2_cube_role_freeze/field2_cube_role_frozen_overview.png`

The contract SHA-256 is `a76425dd259f4fb245db6d51824a99d4f960e044ca5aab2cfc114578bcf06c96`. It records freeze timestamp `2026-08-17T19:04:04.018579+00:00`, exact commit `bcfefaa4bb702879e95fbaec297fd18d6e7b6afe`, review schema `field2_cube_role_reviews_v1`, revision 86, all per-cube decisions and preview references, all input/output hashes, and explicit prediction-free/no-checkpoint/no-training statements. The freeze command refuses to overwrite any existing frozen output.

`scripts/validate_field2_cube_role_contract.py` revalidates the contract, original reviews, audit, frozen table, summary, overview, package manifest, preview references, valid-support manifest, role totals, flag totals, and freeze commit. Tests deliberately mutate a frozen input and a contract decision and require validation failure.

## Prohibited operations

The blind point-sampling CSV and sampling contract remain absent. The sampling generator and point annotator were not launched. No model output was inspected; no checkpoint was loaded as a model; no prediction, probability, pseudo-label, categorical Field 2 label, threshold selection, normalization, training, or model selection occurred. Source imagery, transformed products, support masks, saved investigator reviews, and the frozen Field 1 benchmark were not modified.
