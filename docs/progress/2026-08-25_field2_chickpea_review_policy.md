# Field 2 v2 prediction-free chickpea review policy

Date: 2026-08-25

Status: **NDVI/reference/evaluation policy frozen before model use; 477-point investigator queue open**

## Frozen rule and provenance

The policy uses raw reflectance only: red 669.09 nm (Python band 67, ENVI
band 68), NIR 798.91 nm (Python band 97, ENVI band 98), and
`(NIR - Red) / (NIR + Red)`. NDVI below 0.30 is rule-derived soil, NDVI at
or above 0.30 is vegetation, and nonfinite input or a nonfinite/zero
denominator is `nodata_invalid`. The stored scalar product is not authoritative
NDVI. The threshold was predeclared and was not fitted to Field 2 labels.

Area constraints then produce the following prediction-free provisional
references:

- Alley soil/vegetation becomes `soil`/`field_weed`; chickpea is impossible.
- Outside-field soil/vegetation becomes `soil`/`outside_weed_ood`; chickpea is
  impossible and the OOD weed is not collapsed into an ordinary Field 1 weed.
- Crop/unassigned soil becomes rule-derived `soil`; crop/unassigned vegetation
  remains pending for the investigator and carries no investigator provenance
  until a decision is saved.

Every completed reference uses one of the seven declared provenance values.
Rule references remain separate from investigator responses. The frozen policy
contract records the policy, spectral config, v2 config/contract, Field 1
confident-label contract, source-product/support audit tables, wavelengths,
indices, counts, and hashes. The machine-local contract SHA-256 is
`2ffaf752e3615a186d680303227cf71bc823e684a0fd10bb6f8b9d12a0d7506a`.

## Field 1 documentation audit

The fixed 0.30 rule was audited against the frozen Field 1 confident-label
rasters without optimization. In the raw-reflectance units read from those
cubes, every audited confident pixel was at or above 0.30: 6,930,741/6,930,741
soil-labelled pixels, 441,995/441,995 chickpea-labelled pixels, and
1,440,422/1,440,422 weed-labelled pixels were classified as vegetation; no
audited pixel was invalid. Thus this audit provides **no empirical Field 1
support for soil discrimination at 0.30 in these raw units**. The threshold was
nevertheless left unchanged as explicitly predeclared. Rule-assisted results
must remain a secondary endpoint and must not be described as an independent
gold standard.

## Frozen workload

The 800 main points resolve to 99 rule-based soil, 146 alley field weed, 78
outside OOD weed, 477 manual crop/unassigned vegetation reviews, and zero
invalid points. The 400-point reserve remains locked and was neither opened nor
served.

| Cube | Crop review | Unassigned review | Total review |
|---|---:|---:|---:|
| field2_cube02 | 0 | 2 | 2 |
| field2_cube03 | 0 | 2 | 2 |
| field2_cube04 | 0 | 2 | 2 |
| field2_cube05 | 0 | 2 | 2 |
| field2_cube06 | 0 | 1 | 1 |
| field2_cube07 | 43 | 2 | 45 |
| field2_cube08 | 29 | 1 | 30 |
| field2_cube09 | 0 | 10 | 10 |
| field2_cube11 | 0 | 9 | 9 |
| field2_cube12 | 27 | 2 | 29 |
| field2_cube13 | 5 | 2 | 7 |
| field2_cube14 | 0 | 2 | 2 |
| field2_cube16 | 0 | 1 | 1 |
| field2_cube17 | 0 | 9 | 9 |
| field2_cube18 | 0 | 9 | 9 |
| field2_cube19 | 0 | 10 | 10 |
| field2_cube20 | 0 | 9 | 9 |
| field2_cube21 | 28 | 1 | 29 |
| field2_cube22 | 27 | 2 | 29 |
| field2_cube23 | 0 | 2 | 2 |
| field2_cube24 | 0 | 1 | 1 |
| field2_cube25 | 28 | 2 | 30 |
| field2_cube26 | 25 | 2 | 27 |
| field2_cube27 | 20 | 2 | 22 |
| field2_cube28 | 9 | 1 | 10 |
| field2_cube29 | 25 | 1 | 26 |
| field2_cube30 | 55 | 1 | 56 |
| field2_cube31 | 0 | 0 | 0 |
| field2_cube32 | 0 | 2 | 2 |
| field2_cube33 | 25 | 2 | 27 |
| field2_cube34 | 25 | 2 | 27 |
| field2_cube35 | 0 | 2 | 2 |
| field2_cube36 | 0 | 1 | 1 |
| field2_cube37 | 0 | 2 | 2 |
| field2_cube38 | 0 | 0 | 0 |
| field2_cube39 | 0 | 1 | 1 |
| field2_cube40 | 0 | 2 | 2 |
| field2_cube41 | 0 | 0 | 0 |
| field2_cube42 | 0 | 0 | 0 |
| field2_cube43 | 0 | 2 | 2 |

The complete cube/domain table, including source-domain totals, rule-soil and
invalid counts, is frozen machine-locally as
`metadata/local/reports/field2_chickpea_review_policy/field2_v2_manual_workload_by_cube_domain.csv`.

## Annotation instructions

1. Label only the pixel inside the permanent yellow outline: **“Label the
   frozen yellow center pixel—not the surrounding row.”** Image clicks move only
   the cyan inspection cursor and never select a class.
2. Choose one large decision: Chickpea, Not chickpea (`weed_unspecified`),
   Chickpea/weed mixed, or Uncertain. Nothing is preselected or suggested.
3. Choose High, Medium, or Low confidence. Notes and Boundary needs correction
   are optional. For Not chickpea only, ordinary/tall-grass/other weed may be
   recorded voluntarily as an optional subtype.
4. Use Save & Next to persist and advance to the next unreviewed point. Previous
   revisits the preceding queue item, Next unreviewed jumps forward, Skip moves
   without a decision, and Clear response clears only the displayed response
   after confirmation. Relaunch resumes at the first unreviewed point.

Natural RGB remains the default. PCA and the other frozen layers, the fixed and
inspection magnifiers, raw spectrum, domain, cube role, and sample ID remain
available as prediction-free inspection aids.

## Frozen evaluation policy

The primary endpoint is investigator-reviewed chickpea versus non-chickpea,
reporting chickpea precision, recall, F1, PR-AUC, and false-positive rate with
equal-cube and design-weighted estimates plus confidence-stratified sensitivity.
Mixed and uncertain responses are excluded from primary hard labels and reported
separately. The secondary three-class endpoint is explicitly rule-assisted.
Outside-field points are a separate open-set/OOD challenge; alley points are
Field-1-type in-distribution weed/soil evaluation. No threshold or model may be
tuned on Field 2.

No checkpoint, model, prediction, probability, embedding, pseudo-label, or
automatic biological label was loaded or generated while freezing or testing
this workflow.

## Verification

- Focused Field 2 review/sampling suite: **44 passed**.
- Complete repository suite: **134 passed** (one non-failing Matplotlib
  background-thread warning from the isolated browser fixture).
- Policy, v2 sampling, v1 sampling, natural-RGB display, source/support,
  cube-role, deterministic area-recompile, and frozen Field 1 finalization
  validators all passed.
- The isolated Firefox test saved and resumed through a temporary annotation
  root, verified native control clicks and reserve HTTP 404, and left production
  unchanged.
- Production v2 annotation store: 800 records, zero labels, zero reviewed,
  revision 1, SHA-256
  `86c0f95b4d8cb20954780628a11581731edead1641c3d68e266d5d54e437c504`.
- Frozen v2 main/reserve hashes remain
  `620d168b4a2e98fa80f82cf9df4273f824053a3ea0ea3b92d2adca400831e811`
  and `addca72d0b5dde173d09ce0a318225838afb2b25231bb5793979be0c263e602f`.

## Relaunch

```bash
MPLCONFIGDIR=/tmp/chickpea_matplotlib GDAL_PAM_ENABLED=NO /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_field2_point_annotator_v2.py --paths configs/paths.local.yaml --config configs/field2_area_sampling_v2.yaml --policy-config configs/field2_chickpea_review_policy.yaml --port 8773 --no-browser
```
