# Field 2 prediction-free blind review

Date: 2026-08-17

Starting implementation commit: `e2376b8824bac440dbde8c036a643de29bdac9c5`

Status: **40-cube review package ready; investigator roles remain unreviewed**

## Frozen-state validation

- The Field 1 benchmark commit `da95fd59b7dc5d8ba4add96ed43b398bb7c93749`, finalization commit `c5a69bef38a71a29306c4005b65531b27a7539ae`, and tag `supervised-confident-v1-da95fd5` were validated. All 15 nested and 6 deployment checkpoints required by the frozen finalization are present.
- The exact 40-cube Field 2 inventory was independently checked against the readiness and valid-support contracts.
- The current 240-file source snapshot reproduces SHA-256 `0dbe286a152748f0ab8d338b095b308d977cfcc3568999b86e7f9f4e306ee44e`.
- The valid-support manifest reproduces SHA-256 `1f48ffe67d7ad2cbf95cca5c2e6e98d15245843435c519bbb8a67841c2bbf3e3`. All 40 uint8 masks contain only `{0, 1}` and retain the exact source CRS, affine grid, dimensions, and bounds. Source reflectance and mask hashes match their frozen records.
- All 40 cubes have frozen readiness status `annotation_ready_prediction_free`; their investigator role is still `unreviewed`.

## Review package and interface

`scripts/prepare_field2_cube_role_review.py` produced six checksum-tracked local PNGs per cube: a combined review sheet, false-colour reflectance, supplied PCA components 1–3, stored scalar index, frozen valid support, and the support outline on reflectance. The previews are masked to frozen reflectance support and contain no predictions, probabilities, Field 1 labels, candidate categorical masks, or automatic biological interpretation.

The package contains all 40 configured cubes and 240 layer images. Its ignored local outputs are:

- `metadata/local/reports/field2_cube_role_review/field2_cube_role_review_manifest.csv`
- `metadata/local/reports/field2_cube_role_review/field2_cube_role_review_package.yaml`
- `metadata/local/reports/field2_cube_role_review/layers/`

The localhost reviewer provides layer switching, zoom, pan, reset view, previous/next navigation, save/resume, and an explicitly confirmed clear-current-review control. Saves are validated and atomically replaced. Investigator outputs are written only after the investigator selects **Save all reviews**:

- `metadata/local/annotations/field2_cube_roles/field2_cube_role_reviews.json`
- `metadata/local/annotations/field2_cube_roles/field2_cube_role_reviews.csv`
- `metadata/local/annotations/field2_cube_roles/field2_cube_role_review_audit.csv`
- `metadata/local/annotations/field2_cube_roles/field2_cube_role_review_overview.png`

Launch the reviewer with the project interpreter:

```bash
MPLCONFIGDIR=/tmp/chickpea_matplotlib GDAL_PAM_ENABLED=NO /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_field2_cube_role_reviewer.py --paths configs/paths.local.yaml --config configs/field2_blind_evaluation.yaml
```

## Investigator schema

Each cube has one primary role: `primary_three_class`, `chickpea_absent_negative_control`, `challenge_only`, `sensitivity_only`, `exclude_with_reason`, or `unreviewed`. Ten independent flags preserve visible chickpea, immature chickpea, confirmed absence, ordinary weed, tall-grass weed, soil, mixed pixels, annotation difficulty, georectification concern, and another visual concern. Notes, exclusion reason, confidence, completion status and timestamp, optional reviewer ID, and source-preview hashes are retained.

The save gate rejects inconsistent chickpea-visible/absent flags, incomplete three-class or negative-control evidence, exclusions without reasons, invalid reviewed states, and changed preview hashes. Low confidence remains a valid investigator decision and is never converted to exclusion.

## Explicit freeze and future blind sampling

`scripts/freeze_field2_cube_role_contract.py` is implemented but was not run. It refuses to create `metadata/local/contracts/field2_cube_role_contract.yaml` unless all exact 40 cubes are explicitly reviewed, all logical rules pass, preview and frozen-manifest hashes match, and provenance is prediction-free.

`scripts/build_field2_blind_annotation_sampling_frame.py` is implemented but was not run. Its first data-access gate requires the frozen role contract. The future deterministic design uses seed `20260817`, projected 5 m groups, 0.25 m minimum separation, one deterministic view per geographic ground cell, empirical stored-index rank quintiles (not biological classes), investigator roles and flags, and recorded inclusion probabilities and design weights. Counts remain explicitly proposed pending the reviewed role distribution.

`scripts/run_field2_point_annotator.py` provides the later prediction-free point-interface skeleton, including close/medium context, PCA/index/support views, a crosshair, and a distinct `tall_grass_weed` label. It was not launched and refuses to start without a frozen sampling-frame contract.

## Safety result

No supervised or SSL checkpoint was opened, no prediction or probability was generated, no biological role or categorical Field 2 label was assigned automatically, and no model was trained. Field 2 source products, frozen support masks, investigator annotations, and the frozen Field 1 benchmark were not modified. Generated review assets remain under the ignored `metadata/local/` tree and are not committed.
