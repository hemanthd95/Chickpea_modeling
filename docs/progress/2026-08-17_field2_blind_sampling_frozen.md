# Field 2 prediction-free blind sampling frozen

Date: 2026-08-17

Starting commit: `7d6ed4be797c1ef7c4987f5564c9434e2665c40c`

Sampling implementation and materialization commit: `0a2b230b8d776e6b7219f320bff99c0792b6cbd4`

Status: **800-point main frame frozen; 396-point reserve frozen and locked; main-only annotator ready**

## Frozen-input validation

The remote branch was fetched before work began and initial local/remote divergence was `0 0`. Unrelated modified and untracked files were preserved. The Field 1 finalization validator passed for benchmark `da95fd59b7dc5d8ba4add96ed43b398bb7c93749`, its benchmark tag, 15 nested checkpoints, and 6 deployment checkpoints. Checkpoints were checksum-verified but never loaded or deserialized as models.

Field 2 source, support, preview, investigator-review, and cube-role hashes all matched before sampling:

- Field 2 source manifest: `0dbe286a152748f0ab8d338b095b308d977cfcc3568999b86e7f9f4e306ee44e`
- Valid-support manifest: `1f48ffe67d7ad2cbf95cca5c2e6e98d15245843435c519bbb8a67841c2bbf3e3`
- Review-package manifest: `81de7834c1e3f538a5b1b19f905dbda36521baab5a14e84da261678a4fdceaf2`
- Frozen cube-role contract: `a76425dd259f4fb245db6d51824a99d4f960e044ca5aab2cfc114578bcf06c96`

All 240 source files and 40 support-mask hashes were unchanged after materialization. No source, transformed product, support mask, review, role decision, or preview was modified.

## Predeclared design

Seed `20260817` and all SHA-256 tie-breaking rules are frozen. Candidate pixels must be finite stored-scalar values inside frozen reflectance-derived support and far enough from the raster/preview border to display the complete 110-preview-pixel medium context. No support erosion, morphology, biological filter, prediction, label, probability, embedding, feature, or cluster is used.

Within each cube, the stored scalar product is used only for empirical rank strata: 0–20%, 20–40%, 40–60%, 60–80%, and 80–100%. It is not treated as authoritative NDVI or a biological class. Candidates are collapsed into 0.10 m geographic ground cells, overlapping cube views are resolved by seeded deterministic rank, and the eligible frame is globally thinned to at least 0.25 m separation. Allocation is balanced over the five rank strata and round-robin over deterministic 5 m spatial blocks.

The first-stage inclusion probability is the reciprocal of valid-context pixels in the selected within-cube ground cell multiplied by overlapping cube views. The final probability is the selected quota divided by the eligible population within cube × rank stratum × 5 m block. Overall probability is their product and the design weight is its reciprocal. The declared target population is prediction-free, spatially thinned valid-context support.

## Exact allocation

| Frozen role | Cubes | Main per cube | Main total | Reserve per cube | Reserve total |
|---|---:|---:|---:|---:|---:|
| `primary_three_class` | 20 | 20 | 400 | 10 | 200 |
| `challenge_only` | 8 | 25 | 200 | 12 | 96 |
| `chickpea_absent_negative_control` | 7 | 20 | 140 | 10 | 70 |
| `sensitivity_only` | 5 | 12 | 60 | 6 | 30 |
| **Total** | **40** |  | **800** |  | **396** |

Every cube achieved its exact requested main and reserve count. Main and reserve are disjoint. The combined frame has 1,196 unique immutable sample IDs and 1,196 unique geographic ground-cell IDs.

Rank-stratum counts are:

| Frame | 0–20 | 20–40 | 40–60 | 60–80 | 80–100 |
|---|---:|---:|---:|---:|---:|
| Main | 163 | 163 | 158 | 158 | 158 |
| Reserve | 88 | 83 | 75 | 75 | 75 |

Main points cover 211 spatial blocks, reserve points cover 174, and the combined frame covers 216. Combined nearest-neighbor separation is 0.250599 m minimum, 0.743774 m median, and 3.659781 m maximum. Cross-cube ground-cell duplicates, support violations, invalid-context/border violations, and main/reserve intersections are all zero.

First-stage inclusion probabilities range from 0.02 to 1.0; final probabilities range from 0.00990099 to 1.0; overall probabilities range from 0.000307692 to 0.5; design weights range from 2.0 to 3250.0. A same-input/same-seed repeat produced byte-identical main and reserve tables before freezing.

## Frozen products

Sampling contract:

- Path: `metadata/local/contracts/field2_blind_sampling_frame_contract.yaml`
- SHA-256: `78c0378105e23cb7cc42da0871f144f06532bf7b5ed83ea51c262b4094e229d2`
- Freeze timestamp: `2026-08-17T20:06:39.137492+00:00`
- Materialization commit: `0a2b230b8d776e6b7219f320bff99c0792b6cbd4`

| Product | SHA-256 |
|---|---|
| Main frame | `355980b7dddc302a2b82f0193495bca5ce1b2bc9e31a4c8f72a95a20af8ffe0a` |
| Reserve frame | `37db9d20e86f40d3fe330cf352c10ded1463ebdbe0226bc7ff09ec1204d287fc` |
| Combined frame | `8b0efeeeeeb461f43d40247c62da3e1cc6e18273bb39fe8a1f5fa89eaaab170d` |
| Allocation audit | `a263e96e898a114572a1508da5d7066ddc835b8aa9ede84209e7ba7667466b42` |
| Stratum/block audit | `45798bff890157aa7876b793bb0a8bd4fccba65bdab11dfef0fc3faf2077a93c` |
| Sampling audit | `8aeac7ecf30ee8e588058760038b91b3096da9ce13ef2c5cdc0152377a45cfda` |
| Sampling overview | `9596eba955b770502b7a66eec52f56aa40efdc82a6b57dffeb32da2b79e3f688` |

The materializer refuses to overwrite any frozen output. `scripts/validate_field2_blind_sampling_contract.py` independently verifies hashes, frame composition, stable sample IDs, exact per-cube allocations, probabilities, weights, separation, input contracts, reserve policy, and mutation resistance.

## Main-only point annotator

The localhost interface is available at `http://127.0.0.1:8773`. It serves exactly 800 main samples and zero reserve samples. Initial state has 800 neutral records, zero reviewed records, and no default biological label. Views are limited to checksum-validated false-colour reflectance, supplied PCA, stored scalar index, and frozen support outline. The crosshair marks the central pixel; close and medium contexts are available.

The interface supports cube/role filters, navigation, keyboard shortcuts, explicit reviewed state, confidence, optional notes/reviewer ID, atomic JSON/CSV/audit/overview saves, and resume. Selecting chickpea on an absent-negative-control cube records a role-contradiction flag without changing the frozen role or disabling the label.

Relaunch command:

```bash
MPLCONFIGDIR=/tmp/chickpea_matplotlib GDAL_PAM_ENABLED=NO /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_field2_point_annotator.py --paths configs/paths.local.yaml --config configs/field2_blind_evaluation.yaml
```

## Locked reserve policy

The reserve frame is frozen but not served by the annotator, shown in the annotation interface, or annotated. After all 800 main samples are reviewed, release requires a separate immutable authorization record and at least one prespecified support gate: fewer than 75 confident chickpea labels in primary/challenge cubes, fewer than 100 confident ordinary-weed labels, fewer than 75 confident tall-grass labels, fewer than 100 confident soil labels, fewer than four contributing chickpea cubes, fewer than four contributing tall-grass cubes, or effective design-weighted sample size below 50 for a primary evaluation class.

Reserve release may not depend on supervised predictions, observed model errors, probabilities, model confidence, or preliminary accuracy.

## Safety result

No supervised or SSL checkpoint was loaded, no model output was inspected, and no prediction, probability, pseudo-label, automatic biological class, training run, model selection, threshold selection, or Field 2 normalization occurred. The frozen Field 1 benchmark remained unchanged.
