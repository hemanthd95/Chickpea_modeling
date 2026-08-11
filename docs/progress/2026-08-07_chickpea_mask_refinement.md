# Chickpea weak-mask refinement — 2026-08-07

## 2026-08-11 investigator-reference QC correction

- Investigator-confirmed point classes remain the reference authority.
- Polygon support and exact-centre soil status are retained as registration/QC
  warnings, not label vetoes. Soil and NoData pixels inside each sampling circle
  are still excluded before its bandwise median spectrum is calculated.
- This prevents a rasterized boundary or one noisy centre pixel from discarding
  an otherwise usable 3 cm reference neighbourhood, as occurred in cube 20.

The investigator documented the Field 1 mask provenance. Chickpea masks began as
Spectronon Magic-feature selections and were subsequently filtered with OpenCV
Hough geometry. Soil was defined using NDVI < 0.3; weed is the residual class
that is neither chickpea nor soil. Visual review showed remaining chickpea labels
between crop rows and within vehicle/pedestrian alleys.

These masks are therefore treated as versioned weak labels rather than independent
botanical ground truth. Raw and current authoritative masks remain immutable.

The next stage builds review-only candidates by intersecting current chickpea
labels with two independently interpretable spatial priors:

1. exact verified Field 1 experimental-plot polygons, which exclude plot gaps and
   alleys; and
2. observed straight-row corridors detected from the current mask using a dominant
   Hough orientation.

Plot buffers of 0.00 and 0.15 m and row half-widths of 0.06, 0.10, and 0.15 m are
predeclared. All six variants are written separately with pixel-retention tables
and PNG comparisons. No variant is automatically selected, no removed pixel is
yet reassigned, no supervised model is retrained, and no held-out performance is
used to tune geometry. If the investigator accepts a variant, removed chickpea
pixels will be reassigned to weed under a new authoritative-mask version with a
complete provenance and rollback contract. Field 2 remains locked.

## Engineering correction after the first attempted run

The initial combined constraint stopped on Cube 22 because the verified plot
polygons do not intersect every labeled cube. Cube 20 also retained only 6.4%
when row support was restricted to short Hough segments. These are method-design
failures, not candidate results, and no authoritative mask was changed.

Plot polygons are now explicitly review-only until their spatial completeness is
verified cube by cube. Each cube receives row-only candidates independently of
plot coverage, while plot-and-row intersections are retained as separate visual
diagnostics. Dominant-orientation Hough lines are extended across each raster to
reconstruct full straight rows from fragmented plant detections. This correction
does not select a threshold and remains subject to investigator visual review.

The completed diagnostic retained 98.6–100% of current chickpea labels, proving
that extending every inlier Hough segment created an over-dense corridor set and
provided no useful refinement. Plot intersection was absent in 10 of 16 cubes and
ranged only from 0.9% to 36.2% where present, independently confirming that the
plot layer is spatially incomplete for this mask collection.

The next candidate iteration clusters parallel Hough segments by perpendicular
offset, merges duplicate detections along the same physical row, and requires at
least 1.5 m of longitudinal span and 1.5 m total line support before extending a
row centerline. Reports now include accepted row-cluster count and median spacing.
The ineffective candidate set is rejected and cannot become authoritative.

## Agronomic row spacing supplied and fixed-lattice review staged

The investigator supplied the planter spacing: exactly 3 ft (0.9144 m) between
rows. The free-spacing Hough clusters are therefore superseded. Hough is now used
only to estimate image-space orientation, while a 0.9144 m lattice is fitted by
phase to the observed mask. This prevents dense weed bands from inventing extra
rows at 0.24–0.40 m spacing.

A conservative local-orientation diagnostic compares 0.60 m support along and
across the row direction. This is designed to reject transverse structures such
as the known weed band in Cube 47 even where they cross a valid lattice corridor.

For alley review, the verified numbered plots form an experimental convex-hull
envelope. Within that envelope, current chickpea pixels outside plot polygons
buffered by 0.15 m are marked as alley candidates; pixels outside the envelope
are unaffected because the plot layer is incomplete across Field 1. Row,
orientation, and alley removals remain separately reported and visualized. No
authoritative label is changed at this stage.


## Fixed-lattice composite rejected after visual and quantitative review

The 0.9144 m spacing is agronomically correct, but a single lattice phase per
cube is not sufficient for cubes containing multiple disconnected planting
blocks. At the widest predeclared corridor (0.15 m), the row/orientation
candidate retained 867,500 of 1,238,391 chickpea pixels (70.05%). Retention was
especially implausible in Cubes 22 (49.83%), 24 (55.67%), 31 (41.15%), 32
(53.13%), and 47 (48.89%). Visual review showed the lattice aligning with one
block while rejecting genuine rows in another block with a different phase.

The convex-hull alley diagnostic is also rejected as a label constraint. It
removed a further 130,576 pixels relative to the 0.15 m row-only candidate and
reduced total retention to 59.51%. It falsely marked long genuine crop rows as
alley pixels in Cubes 20–32 because the verified numbered-plot layer is spatially
incomplete and its convex hull does not encode the two actual alleys.

The local-orientation diagnostic did successfully flag the transverse weed band
in Cube 47. It should therefore be retained only as component-level evidence,
not used as an unrestricted pixel-level deletion rule.

No candidate in this iteration may replace an authoritative mask. The next
candidate design will:

1. split disconnected planting blocks before fitting the fixed 0.9144 m lattice;
2. fit an independent lattice phase within each block while preserving the known
   spacing and dominant row orientation;
3. use local orientation to flag transverse connected components;
4. use only an explicitly identified alley/boundary layer, never a convex hull
   inferred from incomplete plot polygons; and
5. preserve an uncertain class during review rather than automatically converting
   every rejected chickpea pixel to weed.


## Investigator-guided six-row design and cube decisions

The planter has six planting units with 3 ft (0.9144 m) spacing between adjacent
rows. The next geometric candidate must therefore detect six-row blocks rather
than fit one unrestricted lattice across an entire cube. Each disconnected block
will share the known orientation and spacing but receive an independently fitted
phase and longitudinal extent.

The investigator identified Cubes 24 and 28 as recently planted substantially
closer to the 2025-05-06 image-collection date. They are prospectively excluded
from the revised primary supervised mask training and evaluation because their
phenological stage is not comparable. They remain immutable in the inventory and
will be retained in a separately reported sensitivity analysis; previous results
are not deleted or rewritten.

For Cube 20, orange pixels in the rejected overview are chickpea by default.
Only the upper area between rows and within the alleys requires removal through
investigator-guided spatial review. This is explicit evidence that the rejected
convex-hull rule cannot be reused.

Raw Spectronon BIP masks for Cubes 12, 14, and 15 are staged as optional label
expansion. They have strong chickpea signal but substantial weed contamination,
so they cannot enter training automatically. They require the same six-row
geometry plus investigator visual approval and will be versioned separately from
the current authoritative masks.


## Planter-turn and wheel-track evidence added

The investigator noted that the six-row planter completed one pass, turned in
the headland, and began the next pass. Tractor wheel marks and turning structures
are distinctly visible in the PCA, first-derivative PCA, and second-derivative
PCA imagery. These features provide independent geometric evidence for separating
adjacent six-row planting blocks and estimating how many planter passes and turns
are represented.

The next audit will reconstruct planter passes at the georeferenced Field 1 level,
not independently count features inside every overlapping UAV cube. Candidate
wheel tracks and turn arcs will be detected from PCA-derived structure, transformed
to map coordinates, and de-duplicated where cubes observe the same ground area.
Each accepted pass will define one six-row block with 0.9144 m centerline spacing
and its own fitted lattice phase and longitudinal extent.

PCA-derived features are geometry evidence only. They cannot assign chickpea,
weed, or soil labels by themselves. Per-cube PCA component signs and ordering may
vary, so detection will use local edge/ridge magnitude and spatial geometry rather
than assuming a globally consistent PCA intensity. Numbered pass, wheel-track,
and turn candidates will be rendered for investigator approval before they
constrain any mask.


## PCA planter-geometry audit implemented

A new read-only script, `scripts/audit_planter_geometry_evidence.py`, inventories
the PCA, first-difference PCA, and second-difference PCA products for each
authoritatively masked cube. It creates locally stretched visualizations, derives
a structural composite from both derivative products, estimates the dominant row
orientation from the observed chickpea mask, and separates row-parallel structural
segments from nonparallel evidence.

Nearby nonparallel segments are grouped into numbered investigator-review
candidates. Candidate centroids are retained in the reflectance raster CRS so a
later field-level step can de-duplicate structures observed in overlapping UAV
cubes. The audit deliberately does not call these candidates tractor turns until
the investigator confirms them.

Outputs include per-cube six-panel PNGs, a Field 1 overview PNG, product-inventory
CSV, segment CSV, numbered-candidate CSV, and a provenance contract. Missing or
ambiguous derivative products and shape mismatches are reported and skipped rather
than silently substituted. No mask is modified and no model is retrained.


## PCA geometry audit result and correction

The first PCA-derived Hough audit completed on 15 cubes; cube 20 was skipped because no student-derived PCA/derivative files exist for it. Review showed that the generic `nonparallel` rule was not specific enough: 38–82% of detected segments were labeled nonparallel in most cubes, and 15 of 21 connected candidates covered approximately 129–209 m². Those candidates are retained as a rejected diagnostic and must not drive mask edits.

A replacement audit now recomputes a consistent three-component PCA directly from each original observed 150-band reflectance cube, including cube 20. It rotates the dominant crop-row direction upright and detects only broad peaks in cross-row derivative energy. This targets transverse headland/turn/wheel-track bands such as the visually distinct structure in cube 47. The stage remains investigator review only; it does not change labels, retrain models, or access Field 2.


## Investigator tyre-track annotation

A localhost-only browser annotator was added so the investigator can trace visible tyre-track centrelines directly on aligned standardized PCA first-difference, second-difference, or PCA RGB layers. Each saved polyline is retained in preview-pixel, original-raster, and EPSG map coordinates. JSON, vertex CSV, and GeoJSON outputs are written under `metadata/local/annotations/planter_tracks/`. Annotations are geometric evidence only and do not become chickpea/weed labels automatically.


## Standardized PCA validity correction

The first standardized reflectance-PCA run stopped on cube 20 because the provisional audit rejected a spatial spectrum whenever any individual band equaled 65535. Existing project QC had already established the correct NoData rule as `any spectral band > 0`; valid Pika-L observations can contain 65535 in individual bands. The audit now follows the frozen project rule, records finite/observed/valid preview counts per cube, and reports those counts in any future failure message. No output from the failed run was accepted.

## Annotation save-safety correction

The planter-track annotator now keeps save status visible in the sticky toolbar, prints successful save counts and paths in the serving terminal, and automatically finishes a valid in-progress polyline before saving or switching views. A browser-side `Download backup JSON` action preserves the current in-memory annotations even when the localhost POST fails. A one-point unfinished feature blocks the action with an explicit correction message rather than being silently omitted.

## Investigator annotation receipt and geometry interpretation

The investigator completed all 16 browser-review cubes and supplied the three synchronized exports: 128 annotations with 1,000 vertices. They comprise 102 tyre-track footprints, 25 alley footprints, and one uncertain structure. JSON, vertex CSV, and EPSG:32617 GeoJSON IDs and coordinates reconcile exactly, and no traced footprint self-intersects. Although the UI called the objects lines, the traces are nearly closed outlines: median mapped footprint widths are approximately 0.39 m for tyre tracks and 1.90 m for alleys. They must therefore be polygonized as investigator-drawn regions rather than buffered as centrelines.

A non-destructive annotation QC stage was added. It rasterizes the footprints on each original cube grid and reports how many current chickpea pixels fall inside alley or tyre footprints. Alley intersections are proposed weed-relabel candidates for investigator review; tyre footprints remain planter-geometry evidence and do not trigger automatic relabeling. Cubes 24 and 28 remain documented phenology sensitivity cases. The stage writes only reports, PNGs, and a provenance contract—never a replacement mask.

## Annotation seam-normalization correction

The first footprint-impact audit stopped on a Cube 22 tyre outline because closing its near-returned endpoint introduced one small seam crossing. Full review identified 13 such near-closed traces (12 tyre footprints and one alley footprint). Their investigator-drawn endpoints were at most 0.60 m apart. The audit now deterministically snaps the final near-duplicate endpoint to the first point when the separation is no more than 0.75 m, records the operation in geometry QC, and confirms that all 128 normalized footprints are non-self-intersecting. Original JSON, CSV, and GeoJSON exports remain immutable.

## Alley-impact result and candidate policy

The annotation impact audit completed across 16 cubes. For the 14 primary-candidate cubes, 94,053 of 1,167,954 current chickpea pixels (8.05%) fall inside investigator-drawn alley footprints. The largest proposed corrections are Cube 40 (25,277 pixels; 29.91%), Cube 47 (22,728; 21.63%), and Cube 49 (12,563; 12.13%). Cubes 24 and 28 remain sensitivity-only and cannot enter refined primary training or evaluation.

Tyre footprints are not safe direct removal masks: several magenta intersections visibly follow legitimate chickpea rows, including Cubes 35 and 45. A reversible materialization stage therefore applies only one rule: current chickpea inside an investigator-drawn alley becomes weed. Soil is unchanged, no chickpea is added, tyre annotations cause no class change, and all candidate products remain outside the authoritative manifest pending visual approval.


## Chickpea-growing polygon annotation staged

A separate localhost browser workflow now prepares aligned NIR-red-green, NDVI,
standardized PCA, PCA first-difference, PCA second-difference, and current-mask
overlay layers before collecting investigator polygons. The interface supports
multiple chickpea-growing blocks, explicit non-chickpea exclusions, and uncertain
boundaries per cube. Existing alley and tyre footprints appear only as locked
reference overlays and cannot be overwritten by this annotation set.

The review set contains every cube with a current chickpea mask plus Cubes 12,
14, and 15 as label-expansion candidates. Cubes 24 and 28 remain visibly marked
as phenology sensitivity cases. Polygons are spatial eligibility priors rather
than pixel labels: no polygon can directly convert all enclosed pixels to
chickpea. Subsequent candidates must separately remove soil using observed NDVI,
apply row/planter geometry, and evaluate conservative observed spectral evidence.
Ambiguous pixels remain uncertain. No authoritative mask or trained model is
changed by either annotation script, and Field 2 remains locked.

## Investigator chickpea-region annotations received

The investigator completed all 19 cubes, including optional label-expansion
Cubes 12, 14, and 15. The synchronized JSON, vertex CSV, and EPSG:32617 GeoJSON
exports contain 413 row-region polygons and 2,068 vertices; all annotation IDs,
vertex counts, coordinates, cube IDs, and analysis roles reconcile exactly.

Because chickpea rows are narrow, polygon edges are explicitly treated as
uncertain measurements rather than exact class boundaries. A non-destructive
audit normalizes only centimetre-scale closing seams (maximum 0.30 m), evaluates
outward tolerances of 0, 0.05, 0.10, and 0.15 m, clips regions to observed raster
support, and gives investigator-drawn alleys precedence. Oversized polygons do
not create chickpea labels; they only define where later observed NDVI, row
geometry, and conservative spectral evidence may be evaluated. The audit reports
current chickpea pixels outside each tolerant prior and renders per-cube and
all-cube PNGs. No mask is materialized until those diagnostics are reviewed.

## Polygon-first authority accepted by investigator

After reviewing the all-cube polygon QC, the investigator directed that the
hand-drawn chickpea row polygons be trusted over the historical chickpea masks.
At the 0.15 m audit tolerance, the historical mask retained only 68.63% of its
primary-cube chickpea pixels; disagreements were particularly large in Cubes 35,
40, 45, 47, and 54 and visually coincide with previously suspected weed noise.
Historical magenta pixels outside the polygon prior therefore cannot be rescued
automatically.

The frozen next-stage policy is polygon-first but not polygon-equals-chickpea.
Observed NoData and explicit alleys are excluded first. Within the polygon core,
NDVI below 0.30 is provisionally soil; remaining vegetation is profiled using
observed green, red, red-edge, and NIR features before a weed/chickpea separator
is frozen. A 0.05 m outward belt is retained as uncertain and requires stronger
spectral evidence. The old chickpea mask remains available solely for comparison.
Cubes 24 and 28 remain sensitivity-only, and Cubes 12, 14, and 15 remain label-
expansion candidates. No authoritative label changes during profiling.


## Polygon spectral profile result and soil-source correction

The polygon-first profile completed for all 19 annotated cubes. Investigator
polygons provide useful spatial restriction and generally contain stronger NIR
vegetation response than their outward edge belts and surrounding observations.
The historical chickpea mask often has spectral distributions similar to, or
worse than, the rest of the polygon core, so it remains comparison-only.

The provisional direct NDVI calculation from stored 670 and 800 nm Pika-L values
classified zero soil pixels in every polygon and every uncertain edge belt. This
cannot be accepted as evidence that the polygons contain no soil: even observations
outside the polygons had 5th-percentile NDVI around 0.47–0.55 and median NDVI
around 0.63–0.75. The direct stored-value calculation therefore does not reproduce
the investigator's original Spectronon NDVI < 0.30 soil workflow and is rejected
as a label source.

A read-only soil-source audit now intersects the provenance-tracked soil masks
with the investigator polygons, inventories stored NDVI products, checks their
shape and numeric scale, and compares any plausible product with existing soil
masks. No unknown scaling is inferred. Cubes 12, 14, and 15 remain unresolved for
soil until a stored NDVI product is validated because they do not have approved
soil masks. No mask is changed and no model is retrained by this audit.


## Stored NDVI threshold rejected; cross-cube separability audit staged

After excluding the common -2.0 Spectronon NoData sentinel, all 18 stored
one-band products were readable on the physical [-1, 1] domain. Nevertheless,
their first-percentile values were 0.41–0.47 and the documented NDVI < 0.30 rule
selected zero soil pixels in every cube. Agreement with all available
provenance-tracked soil masks was therefore effectively zero. These products
cannot be used as NDVI soil masks for Cubes 12, 14, or 15 under the documented
threshold.

A new read-only audit treats the stored value as an uninterpreted scalar index
and tests whether it separates existing soil from non-soil consistently. A
direction and cutoff are fitted on balanced samples from all but one primary
cube and evaluated on the held-out cube. Cubes 24 and 28 are excluded from
threshold fitting and retained only for sensitivity. A transferred rule is
review-eligible only if mean leave-one-cube-out AUC is at least 0.75, balanced
accuracy is at least 0.70, and the direction is stable. Even a passing rule does
not automatically materialize a mask.

## Stored-index transfer gate passed; review candidates staged

Across 13 primary reference cubes, the lower-valued scalar-index rule achieved
mean leave-one-cube-out AUC 0.9903 and mean balanced accuracy 0.9414. Held-out
balanced accuracy ranged from 0.9344 to 0.9488, and fitted thresholds ranged only
from 0.7206 to 0.7229. The index is therefore transferable as an empirically
calibrated soil discriminator, but it is not reinterpreted as literal NDVI.

The next stage writes mutually exclusive, review-only support layers. Existing
cubes retain their provenance-tracked soil masks. Only Cubes 12, 14, and 15 use
the frozen scalar-index rule, producing provisional soil fractions of 32.98%,
26.15%, and 34.48% inside eligible polygon cores. Investigator polygons remain
spatial priors, explicit alleys take precedence, and a 0.05 m outward belt stays
uncertain. Core non-soil is not yet chickpea: it remains pending observed
weed-versus-chickpea separation. No authoritative mask or model input changes.

## Polygon-guided vegetation separability audit staged

The review-only materialization completed for all 19 annotated cubes. Across
the polygon cores it preserves 696,057 soil pixels and 1,281,131 non-soil
vegetation candidates; the latter are explicitly not assumed to be chickpea.
Cubes 12, 14, and 15 use only the validated scalar-index soil rule, while all
existing labeled cubes retain their provenance-tracked soil masks.

Before any weed/chickpea relabeling, a fixed linear spectral audit now uses
only pixels where the polygon-core non-soil layer agrees with an existing
historical chickpea or weed label. These are weak audit references, not new
ground truth. Cube-balanced samples are evaluated by leaving out one entire
primary cube at a time. Cubes 24 and 28 are excluded from reference fitting and
remain sensitivity-only; Cubes 12, 14, and 15 receive projection scores only.
The audit produces spectral curves, held-out metrics, and spatial score PNGs.
It cannot write categorical masks or retrain a model.

## Weak-reference gate result and clean point annotation

A weak-reference chickpea-versus-weed audit was completed after polygon/soil
materialization. The global spectral separator failed the frozen transfer gate:
mean leave-one-cube-out AUC was 0.5707 and mean balanced accuracy was 0.5462.
Scores clustered near 0.5 across all cubes. This demonstrates that historical
chickpea and weed masks remain too contaminated to provide spectral truth; it
does not demonstrate that the classes are intrinsically inseparable.

A separate localhost vegetation-reference point annotator was therefore added.
It reuses the already prepared NIR-red-green, PCA, derivative, NDVI, and
historical-mask layers, while showing the investigator row polygons and planter
footprints as locked context. The investigator marks only small, unmistakable
chickpea or weed interiors (target: ten points per class per primary cube), with
zoom and 0/3/5 cm sampling radii. It writes a new JSON/CSV/GeoJSON reference set
and cannot alter the trusted row polygons, authoritative masks, or model inputs.

## Investigator reference spectral audit staged

The next read-only audit treats every investigator point as one independent
reference observation. Its 0/3/5 cm neighbourhood is sampled in map space;
soil, nonfinite, and unobserved pixels are removed; and the remaining spectra
are reduced to one bandwise median. Chickpea centres must fall in polygon-core
non-soil support and weed centres must fall outside that core. The separator is
then evaluated by leaving out a whole primary cube at a time. Cubes 24 and 28
remain sensitivity-only, Cubes 12/14/15 remain projection-only, historical
chickpea/weed masks are not used as truth, and no categorical mask is written.
