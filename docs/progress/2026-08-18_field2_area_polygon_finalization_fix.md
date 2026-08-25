# Field 2 area polygon-finalization defect correction

Date: 2026-08-18

Status: **interface corrected; existing investigator geometry unchanged; area not frozen**

## Preserved annotation baseline

Before diagnosis, the five existing area-package artifacts were hashed and copied byte-for-byte to `/tmp/field2-area-annotation-backup.HvfgXU`. The source JSON was revision 25 with 225 polygons: 159 alley, 52 research crop area, 9 outside research field, and 5 uncertain boundary. Three cubes were reviewed. Cubes 11, 16, and 17 retained polygons with blank coverage modes.

The original package hashes were:

- JSON: `7415888767d545514319cc5c175ee1b4ccc29015a4c44702775fa5ee73554b93`
- GeoJSON: `e6366f31d49c6ea7758ed41e708266f48143d318a78077435f1e5fc8e95a6937`
- vertex CSV: `fd91c42ad3d85af2804d0fa96c064c62e2da4e91ee0c42aa40438496fb89adb6`
- geometry audit CSV: `86f9aa92399d9fb45f5c6a787fe40d68ed332a3c5243452358e7b403e15f34ee`
- overview PNG: `e378f793b8b10c86f65d56029be0106566fd8b220407fcd8e70ba421d1db9778`

## Root cause and correction

The UI and backend use the same exact `research_crop_area` identifier. The crop-specific backend path did not filter, deduplicate, or clip submitted vertices before validation.

The reproducible failure was shared browser draft state: `setTool()` silently replaced the in-progress vertex array with an empty array whenever the investigator selected Pan, Edit, or another non-draw tool. A seven-vertex cube12 crop therefore became a zero-vertex draft before Finish and produced the minimum-vertex error. Zone switching could also retain a draft while changing its zone type.

Separately, browser double-click generated two pointer-up events. Both were appended as vertices, and no double-click completion handler existed. This explains the extra near-terminal vertices seen in the saved audit.

The correction keeps a draft and its zone type until successful finalization or explicit cancellation. Tool changes preserve it; incompatible zone switches and cube navigation are blocked with a count-bearing warning. Finish, Enter, and double-click use one finalization function. Vertices are added on click: the first click of a double-click supplies the terminal point, while the second completes without appending another point. Validation reports displayed, submitted, distinct-pixel, and clipped counts and never silently deduplicates source vertices.

Original pixel vertices remain the source geometry. Map vertices are derived only for new or edited polygons that do not already have them. Raster-bounds and valid-support clipping are separate operational derivations. A no-edit save is a true no-op and leaves all artifacts byte-identical.

`research_crop_area` is defined in the interface and area schema as “investigator-identified planted-row/research-plot domain; not a biological chickpea label.”

## Audit reconciliation and overlaps

The saved audit contains 54 self-intersecting polygons. A local, non-applying reconciliation table was generated at:

`metadata/local/annotations/field2_area_zones/field2_area_terminal_vertex_reconciliation.csv`

It contains all 54 requested rows and source hash provenance. Fifty-three rows have a terminal-only operational proposal; one remains manual-review/no-change. Every `proposal_applied` value is false, and investigator decision/reviewer/timestamp fields are blank. No source polygon or vertex was modified.

The five nonzero saved overlap records were inspected: cube05 alley/uncertain, cube08 uncertain/alley, cube12 alley/uncertain, cube12 crop/alley, and cube17 uncertain/alley. The interface now lists polygon IDs and zone pairs, explicitly states that no priority is assigned, and visibly presents both cube12 conflicts. Cubes 11, 16, and 17 show a mixed-mode-required warning and cannot be reviewed until `mixed_manual_boundaries` is selected.

## Safety

The area-freeze command was not run. Reserve points were not exposed. No checkpoint, model, prediction, probability, automatic segmentation, biological label, frozen contract, sampling frame, source product, support mask, or preview was used or changed.

## Verification

- Isolated Firefox cube12 reproduction before the fix: a seven-vertex crop succeeded directly, switching to Pan cleared the draft and reproduced the minimum-vertex error, and double-click appended two terminal vertices without finishing.
- Isolated Firefox after the fix: direct Finish retained seven vertices, Pan preserved all seven, and double-click completed with one terminal click.
- Isolated browser-to-storage round trip: the new cube12 crop reloaded with exactly seven original pixel vertices and seven derived map vertices; every one of the pre-existing 225 polygon objects remained exactly equal.
- Focused Field 2 area/blind suite: **49 passed**.
- Full repository suite: **115 passed**.
- Frozen Field 1 finalization and Field 2 source/readiness, valid-support, cube-role, blind-sampling, and RGB-display validators: **passed**.
