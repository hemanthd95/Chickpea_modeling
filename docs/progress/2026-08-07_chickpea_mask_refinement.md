# Chickpea weak-mask refinement — 2026-08-07

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
