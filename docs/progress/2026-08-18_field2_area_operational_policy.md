# Field 2 operational area-geometry policy

Date: 2026-08-18

Status: **operational policy implemented; raw annotations unchanged; annotation incomplete; not frozen**

## Investigator decision

Small polygon overlaps, self-intersections, slight raster-edge extensions, and near-duplicate closing vertices are audit warnings rather than annotation or freeze blockers. Investigators do not need to redraw the existing 225 polygons.

Raw investigator vertices remain immutable. Derived operational geometry uses this deterministic pixel-domain precedence, from highest to lowest:

`outside_research_field > alley > uncertain_boundary > research_crop_area`

The interface states: “Small overlaps are resolved automatically: outside > alley > uncertain > research crop.” It shows raw overlap pairs and the effective operational coverage mode. A record containing polygons always has effective operational mode `mixed_manual_boundaries`, including cubes 11, 16, and 17 whose raw coverage mode remains blank.

`research_crop_area` remains an investigator-identified planted-row/research-plot domain and is not an automatic chickpea label.

## Operational geometry

The source annotation JSON remains revision 25 with 225 polygons and 1,602 original vertices. Derived geometry results are:

- 171 polygons: no repair
- 53 polygons: final terminal vertex removed from operational geometry only
- 1 polygon: deterministic raster-domain validity repair
- 227 retained operational polygon components
- 225/225 operational polygons rasterize successfully

The remaining invalid source polygon is `field2_cube18-polygon-1787084615605` (`alley`). Raster-domain make-valid retained all three resulting polygon components. Raw area was 19,064.608346 pixels²; operational area is 19,065 pixels²; absolute change is 0.391654 pixels² (0.002054%). Its 21 raw vertices remain untouched.

Operational products are stored separately under:

`metadata/local/annotations/field2_area_zones/operational_geometry/7415888767d54551/`

- `field2_area_operational_geometry.geojson`
- `field2_area_operational_geometry_audit.csv`
- `field2_area_operational_membership_summary.csv`

Every product records source annotation SHA-256 `7415888767d545514319cc5c175ee1b4ccc29015a4c44702775fa5ee73554b93`.

## Transparent membership and future freeze

Operational compilation retains one raw membership mask for each zone, a compact raw-membership bitmask, the winning membership, and a precedence-applied mask. Future main and locked-reserve membership rows will record:

- `raw_zone_memberships`
- `winning_operational_membership`
- `precedence_rule_applied`
- `effective_operational_mode`

Future freeze outputs will include both winning zone masks and raw-membership bitmask rasters plus a complete pixel-count precedence summary. Reserve membership outputs continue to omit coordinates.

Consequences are explicit: research-crop pixels form the main in-field domain; outside-field pixels are retained as OOD and excluded from primary accuracy; uncertain pixels are supplementary; alley pixels cannot receive chickpea or chickpea-mixed labels. No area polygon assigns any biological label.

The freeze command was updated but not executed because only 3/40 cubes are reviewed.

## Preservation and safety

The five raw annotation artifacts remain byte-identical:

- JSON: `7415888767d545514319cc5c175ee1b4ccc29015a4c44702775fa5ee73554b93`
- GeoJSON: `e6366f31d49c6ea7758ed41e708266f48143d318a78077435f1e5fc8e95a6937`
- vertex CSV: `fd91c42ad3d85af2804d0fa96c064c62e2da4e91ee0c42aa40438496fb89adb6`
- geometry audit CSV: `86f9aa92399d9fb45f5c6a787fe40d68ed332a3c5243452358e7b403e15f34ee`
- overview PNG: `e378f793b8b10c86f65d56029be0106566fd8b220407fcd8e70ba421d1db9778`

No checkpoint, prediction, probability, suggested label, biological auto-label, reserve release, frozen sampling change, or area freeze was performed.

## Verification

- Focused Field 2 area/blind suite: **55 passed**.
- Full repository suite: **121 passed**.
- All 225 operational polygons rasterized; terminal/validity repair counts were exactly 53/1.
- Cube12 seven-vertex Finish/Enter/double-click, zoom/pan, cancellation, multiple-crop, and save/reload regressions passed.
- Frozen Field 1 finalization and Field 2 source/readiness, valid-support, cube-role, blind-sampling, and RGB-display validators passed.
