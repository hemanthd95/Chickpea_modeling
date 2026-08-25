# Field 2 capacity-aware area-stratified sampling v2

Date: 2026-08-25

Status: **v2 main frame frozen and ready for prediction-free annotation; reserve locked**

## Scientific design

The frozen 20-cube `primary_three_class` role was not changed. A separate derived
attribute, `crop_sampling_eligible`, is true only when a primary cube has nonzero
operational `research_crop_area` support. Fourteen primary cubes are eligible:

`07, 08, 12, 13, 21, 22, 25, 26, 27, 28, 29, 30, 33, 34`.

Cubes `09, 11, 17, 18, 19, 20` remain primary evaluation cubes but have zero
operational crop support. They are represented through guaranteed unassigned-
support sampling and are neither exclusions nor negative controls. All-outside
cubes `38, 41, 42` receive guaranteed outside-domain points and no nonexistent
alley points.

| Operational domain | Main | Locked reserve |
|---|---:|---:|
| research_crop_area | 400 | 200 |
| alley | 160 | 80 |
| outside_research_field | 120 | 60 |
| unassigned_valid_support | 120 | 60 |
| **Total** | **800** | **400** |

Selection used frozen valid support, frozen operational area masks, 0.10 m ground
cells, 5 m spatial blocks, a 0.25 m combined-frame separation target, and stored-
scalar empirical ranks only. Every one of the 1,196 v1 main/reserve ground cells
was excluded. Main and reserve were selected jointly and have no intersection.
The overview contains main coordinates only; reserve coordinates are omitted.

Eight crop allocation rows exported capacity shortfalls and seven rows received
same-domain redistribution. This reflects small valid, separated capacity in
some crop polygons; global crop totals remained exact. No alley, outside, or
unassigned point was substituted for crop. The complete initial/final allocation
and zero-shortfall rows are in `allocation_redistribution.csv`; capacity and
cube/domain/rank/block details are in the other v2 audits.

## Preservation and supersession

The immutable supersession amendment records that v1 preceded area annotation,
contained five crop-domain main points, had no biological annotations, and was
never used for Field 2 model evaluation. V1 frames and the neutral v1 annotation
package remain byte-identical. V2 uses a separate sampling directory and point-
annotation namespace.

Frozen hashes at materialization:

- area contract: `25c85c11b40efe384aeb89163e172221c42016d0b36fcaaa814a09cf4ed8badf`
- v1 sampling contract: `78c0378105e23cb7cc42da0871f144f06532bf7b5ed83ea51c262b4094e229d2`
- v1 main frame: `355980b7dddc302a2b82f0193495bca5ce1b2bc9e31a4c8f72a95a20af8ffe0a`
- v1 reserve frame: `37db9d20e86f40d3fe330cf352c10ded1463ebdbe0226bc7ff09ec1204d287fc`
- v1 neutral annotations: `22d9f4cf701f3e5576686e7e1328c18c0e46c12917501cb6dba3774e0e14460c`
- v2 sampling contract: `2881dc196c70d211a8eaf25a6b87c67652babb1e3f4602ede3f648d0b34fee87`

## Point annotator

The v2 annotator serves exactly 800 main points and has no reserve route. Natural
RGB is the default; false color, PCA, stored scalar, support, magnification, and
raw spectra remain prediction-free inspection aids. Frozen role and operational
domain are displayed. No biological label is preselected. Alley and outside
records disable chickpea and chickpea-containing mixed labels and retain
`boundary_needs_correction` for visible boundary disagreements.

The new namespace initialized at revision 1 with 800 records, zero reviewed, and
zero labels. Relaunch command:

```bash
MPLCONFIGDIR=/tmp/chickpea_matplotlib GDAL_PAM_ENABLED=NO /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_field2_point_annotator_v2.py --paths configs/paths.local.yaml --config configs/field2_area_sampling_v2.yaml --port 8773 --no-browser
```

## Verification

- Focused area/sampling/point tests: **61 passed**.
- Complete repository tests: **127 passed**.
- Field 2 source/support, cube-role, v1 sampling, natural-RGB display, frozen
  area deterministic recompile, and v2 sampling validators passed.
- Frozen Field 1 supervised-finalization validation passed with Field 2 still
  locked, unopened, and unpredicted by that pipeline.
- All 1,200 v2 points match frozen valid support and their recorded operational
  domain; every operational crop component has a coverage audit row.
- No checkpoint, model, prediction, model probability, embedding, Field 1 output,
  pseudo-label, or automatic biological label was used.
