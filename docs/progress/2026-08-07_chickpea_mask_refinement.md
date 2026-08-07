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
