# Field 2 prediction-free area annotation setup

Date: 2026-08-18

Status: **area annotator ready for investigator review; geometry not frozen; point annotation blocked**

## Phase 0 safety gate

The remote branch was fetched before implementation. Starting local and remote commit: `fb7999a845aaf30807a060f86287d188f51e11ab`; starting divergence: `0 0`. The unrelated modified classifier and unrelated untracked YAML, CSV, JSON, and model files were preserved and excluded.

Read-only validation passed for the frozen Field 1 finalization; all 240 Field 2 source files and the 40-cube georectified readiness contract; all 40 frozen valid-support masks; the 40-cube role contract; the blind-sampling contract; and the natural-RGB display addendum. The source snapshot exactly matched its frozen baseline.

The point package remains revision 1 with exactly 800 main records, zero biological labels, zero reviewed point annotations, and `reserve_frame_exposed: false`. Its SHA-256 remains `22d9f4cf701f3e5576686e7e1328c18c0e46c12917501cb6dba3774e0e14460c`.

Frozen sampling hashes remained byte-identical:

| Sampling product | SHA-256 |
|---|---|
| Main, 800 points | `355980b7dddc302a2b82f0193495bca5ce1b2bc9e31a4c8f72a95a20af8ffe0a` |
| Reserve, 396 locked points | `37db9d20e86f40d3fe330cf352c10ded1463ebdbe0226bc7ff09ec1204d287fc` |
| Combined, 1,196 points | `8b0efeeeeeb461f43d40247c62da3e1cc6e18273bb39fe8a1f5fa89eaaab170d` |

## Area annotator

`scripts/run_field2_area_annotator.py` serves a separate localhost interface on port 8774. Natural RGB is the default; false color, PCA, stored scalar index, valid support, and support outline remain available only for investigator interpretation.

Each of the 40 cubes has an explicit coverage mode, confidence, notes, review state, reviewer, frozen source hashes, support hash, RGB hash, CRS, and transform. Mixed cubes accept multiple polygons of these contextual types:

- `research_crop_area` — transparent green
- `alley` — transparent orange
- `outside_research_field` — transparent blue
- `uncertain_boundary` — transparent purple

The interface supports polygon drawing, finish, vertex editing/insertion/deletion, polygon movement/deletion, undo/redo, opacity, per-zone visibility, wheel/fixed zoom, pan, reset, layer switching, cube navigation, notes, confidence, review, and atomic save/resume. The right-hand panel occupies the full viewport and its navigation, review, and save controls remain permanently visible.

“Treat unassigned valid support as outside research field” is a separate confirmed investigator action and is never silently enabled.

Saved outputs, once the investigator uses Save, are:

- `metadata/local/annotations/field2_area_zones/field2_area_annotations.json`
- `metadata/local/annotations/field2_area_zones/field2_area_annotations.geojson`
- `metadata/local/annotations/field2_area_zones/field2_area_vertices.csv`
- `metadata/local/annotations/field2_area_zones/field2_area_geometry_audit.csv`
- `metadata/local/annotations/field2_area_zones/field2_area_annotation_overview.png`

No file currently exists at those paths because browser verification deliberately did not save or create investigator geometry.

Original pixel vertices are retained. Geospatial vertices are derived deterministically through the frozen affine transform and recorded with the source CRS. Operational masks are clipped to frozen valid support without changing the polygon. Self-intersections, incompatible overlaps, outside-raster fractions, and outside-support fractions are audited. No morphology, Hough transform, segmentation, color inference, or model correction exists in the workflow.

## Freeze and point workflow

`scripts/freeze_field2_area_contract.py` is implemented but was not executed. It refuses overwrite, requires all 40 cubes reviewed, rejects self-intersections and incompatible zone overlaps, creates deterministic support-clipped zone masks, and computes main and locked-reserve zone memberships without writing coordinates. It does not assign a biological label.

The point annotator now refuses to launch until that frozen contract is present and hash-valid. After freeze it merges only coordinate-free main membership, preserves all 800 original coordinates, applies alley label restrictions, exposes `boundary_needs_correction`, and reports outside-field and uncertain domains separately. `weed_soil_mixed` was added while all point labels remain empty.

## Verification

- Focused Field 2 area/blind suite: **42 passed**.
- Full repository suite: **108 passed**.
- Live Firefox at 1440×900: polygon creation, undo/redo, layer switching, wheel zoom, sticky viewport controls, and all tested hit targets passed; JavaScript console errors: zero.
- Area API: 40 cubes, natural RGB default, reserve exposure false, biological-label controls unavailable.
- Point launch gate: correctly refused with `Field 2 area geometry is not frozen; point annotation remains blocked`.
- Frozen Field 1 finalization, Field 2 source/readiness, valid support, cube role, blind sampling, and RGB display: **passed**.

## Relaunch

```bash
MPLCONFIGDIR=/tmp/chickpea_matplotlib GDAL_PAM_ENABLED=NO /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_field2_area_annotator.py --paths configs/paths.local.yaml --config configs/field2_area_annotation.yaml --port 8774 --no-browser
```

URL: `http://127.0.0.1:8774`

No checkpoint was loaded. No prediction, probability, suggested label, automatic geometry, biological point label, sampling coordinate, sampling table, reserve protection, source raster, frozen contract, or existing annotation was changed.
