# Field 2 area-domain freeze and prediction-free point annotation

Date: 2026-08-19

Status: **area domains frozen; point labels remain unfrozen and unstarted**

## Frozen raw inputs

The investigator-completed area package is revision 75 with 40/40 reviewed,
40/40 high-confidence records, 438 polygons, and 2,778 original vertices.
Every raw polygon and vertex remains unchanged. The immutable raw hashes are:

- annotations JSON: `9e45943b2b8817fdbea63e28d94aea6bcd46852cbe2750985631a39b8da560d0`
- annotations GeoJSON: `8b22e4b2db413b10bf0876b812283b7c4056991cca4a2ed5839d8e901c1eeda4`
- vertex CSV: `bb5a0ebd4b6fe6f936656a57d1c2b1caecfe97a427d2aeae640435d4a87f12c2`
- geometry audit CSV: `00c2fc2c4382b4b4d21ef1a1613e7f829c6d7423e61cca30eba3acacafca8217`
- overview PNG: `a8f6e30214cf923b906f4b8210fb0b2a2c02064a024c96d97d7300e637f6047d`

The freeze contract SHA-256 is
`25c85c11b40efe384aeb89163e172221c42016d0b36fcaaa814a09cf4ed8badf`.
It records creation commit `3ee5bff41d8e2624200c7af5632fb672d392ac07`.

## Operational geometry and masks

Operational geometry is stored separately and clipped to frozen valid support.
Of the 438 raw polygons, 334 required no pre-support validity repair, 103 used
terminal-vertex reconciliation operationally only, and one used deterministic
raster-polygonize validity repair. Support clipping retained 701 polygonal
components. Warnings comprise 104 raw self-intersections, 103 near-terminal
artifacts, 96 valid-support clips, 33 raster-edge clips, and one deterministic
validity repair; all are recoverable audit warnings.

Every cube has an aligned uint8 domain raster with codes 0–5 and a raw-membership
bit raster. The 40-row operational mask manifest SHA-256 is
`27381fbecea6aff6e6a3d608e36ca1b86d5f57d27bdda391e716ed905af8ed08`.
The freeze built all 89 products twice and verified byte-identical SHA-256 maps.
The independent validator then recompiled all products from the raw package and
matched every frozen byte.

Precedence is:

`outside_research_field > alley > uncertain_boundary > research_crop_area > unassigned_valid_support`

Cubes 11, 16, 17, 36, and 37 preserve blank raw coverage modes and reconcile to
effective `mixed_manual_boundaries` because reviewed investigator polygons are
present. Recorded all-outside modes remain all outside operationally even where
preserved alley polygons are present; both raw membership and the winning outside
membership remain auditable.

## Frozen sample memberships

All 800 main and 396 locked-reserve samples received exactly one valid-support
domain assignment without changing sample coordinates or identities. Membership
tables omit coordinates. The reserve membership table remains locked and is not
available to the annotation server.

| Domain | Main | Locked reserve |
|---|---:|---:|
| alley | 307 | 153 |
| outside_research_field | 249 | 121 |
| research_crop_area | 5 | 3 |
| unassigned_valid_support | 239 | 119 |

No frozen sample fell in invalid support, and main/reserve intersection remains
zero. Counts are also frozen by cube, cube role, stored-index rank stratum, and
spatial block.

## Main-only point interface

The point annotator accepts only the 800-point main frame and exposes zero reserve
records. It provides natural RGB by default, PCA and stored-index context, fixed
nearest-neighbor magnifiers, a larger pan/zoom viewer, and raw source-reflectance
spectra for Python bands 3–113. The optional valid 3×3 median is display-only.
Displayed NDVI, GNDVI, and NDRE values use exact selected wavelengths and explicit
formulas without thresholds. The stored scalar layer is explicitly not described
as authoritative NDVI.

The label set adds `other_weed`. Alley and outside-field records permit soil,
weed, mixed weed/soil, uncertain, and invalid labels but disable chickpea and
chickpea-mixed labels. A boundary-correction flag is available without changing
the frozen domain or forcing a biological label. No label has a default or is
suggested automatically.

## Verification and safety

- Focused Field 2 area/point suite: **57 passed**.
- Full repository suite: **123 passed**.
- Frozen Field 1 benchmark validator passed.
- Field 2 source/readiness, valid-support, cube-role, blind-sampling, and RGB
  validators passed with exact hashes.
- Independent Field 2 area validator passed with deterministic recompile.
- Existing point annotations remain 800 main records, zero reviewed, and zero
  biological labels; point labels were not frozen.
- No checkpoint was loaded or deserialized. No model, prediction, probability,
  threshold suggestion, pseudo-label, class envelope, or automatic biological
  label was generated.
