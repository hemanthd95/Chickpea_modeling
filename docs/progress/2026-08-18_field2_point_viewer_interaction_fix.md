# Field 2 point-viewer interaction fix

Date: 2026-08-18

Status: **interactive inspection viewer verified; frozen sampling and annotations unchanged**

## Diagnosis

The existing natural-RGB canvas rendered correctly, and browser hit-testing placed both canvases and every annotation control at the top of their respective hit targets with `pointer-events: auto`. Firefox reported no page JavaScript errors. Previous/next, layer, zoom, label, confidence, note, review, and save controls were not covered by an overlay and remained independently interactive.

The defect was isolated to the image viewer: neither canvas had click, pointer, drag, or wheel listeners. A canvas click therefore reached the canvas but had no effect. No coordinate or annotation-state defect was involved.

The saved annotation package was recorded before diagnosis as revision 1, 800 records, 0 reviewed, with SHA-256 `22d9f4cf701f3e5576686e7e1328c18c0e46c12917501cb6dba3774e0e14460c`. Browser diagnosis and verification did not invoke Save; the same revision and hash remained after the work.

## Viewer behavior

The viewer now separates immutable sample position from visual inspection state:

- A thin yellow square permanently outlines the frozen full-resolution sampled pixel without filling its color.
- A cyan crosshair identifies the independently movable inspection pixel.
- Clicking the cube updates only the inspection pixel and its optional magnifier. It does not move the frozen sample or select a biological class.
- The page states: “Sample location is frozen; clicking only changes the inspection view.”
- Pointer drag pans, the mouse wheel zooms around its cursor, Reset view restores the initial fit and inspection position, and buttons provide exact 2×, 4×, 8×, and 16× views.
- Pixel grids are available at 8× and 16×, with nearest-neighbor rendering at high zoom.
- The frozen-sample magnifier is always present. The inspection magnifier can be shown or hidden independently.
- Natural RGB remains the default. False color, PCA, stored index, and support views remain available, and layer changes preserve frozen and inspection coordinates.
- Labels and confidence are explicit buttons. All annotation, navigation, layer, zoom, review, note, and save controls remain separate from the canvas pointer handlers. The only form overlay is explicitly `pointer-events: none` and behind the controls.

## Verification

- Focused Field 2 review/sampling/viewer suite: **31 passed**.
- Full repository suite: **97 passed**.
- Live Firefox pointer/wheel verification: **passed**; click changed inspection only, drag panned, wheel zoomed, reset and layer preservation passed, control hit targets were unobstructed, and browser console errors were empty.
- Frozen Field 1 supervised finalization: **passed**; 15 nested and 6 deployment checkpoints were identity/hash verified only, without model loading.
- Frozen Field 2 cube-role contract: **passed**.
- Frozen Field 2 blind-sampling contract: **passed**; main 800, reserve 396, combined 1,196, reserve locked.
- Frozen natural-RGB display addendum: **passed**; all 40 previews reproduced in memory with exact hashes.
- Live API: **800 main samples**, one `main` frame value, `reserve_exposed: false`, natural RGB default.

Frozen hashes after the viewer change:

| Product | SHA-256 |
|---|---|
| Main sampling CSV | `355980b7dddc302a2b82f0193495bca5ce1b2bc9e31a4c8f72a95a20af8ffe0a` |
| Reserve sampling CSV | `37db9d20e86f40d3fe330cf352c10ded1463ebdbe0226bc7ff09ec1204d287fc` |
| Combined sampling CSV | `8b0efeeeeeb461f43d40247c62da3e1cc6e18273bb39fe8a1f5fa89eaaab170d` |
| Annotation-display addendum | `1854458fb428b7c4b7eb1ec15bbc5b354dc21573f35944ef2cefbbb36ef08a7f` |
| Saved point annotations | `22d9f4cf701f3e5576686e7e1328c18c0e46c12917501cb6dba3774e0e14460c` |

## Relaunch

```bash
MPLCONFIGDIR=/tmp/chickpea_matplotlib GDAL_PAM_ENABLED=NO /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_field2_point_annotator.py --paths configs/paths.local.yaml --config configs/field2_blind_evaluation.yaml --port 8773 --no-browser
```

URL: `http://127.0.0.1:8773`

No checkpoint was loaded. No prediction, probability, suggested label, pseudo-label, training, model selection, sample coordinate, sampling table, frozen contract, reserve protection, or saved annotation was changed.
