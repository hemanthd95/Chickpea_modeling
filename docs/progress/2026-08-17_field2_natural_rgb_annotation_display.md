# Field 2 natural-RGB annotation display frozen

Date: 2026-08-17

Starting commit: `8b621b381f6097451def9af34930714b578f6acb`

Display implementation and materialization commit: `3c9f5e1e13abc8fe48898c3dc53f0cd82a5db1f2`

Status: **40 deterministic full-resolution natural-RGB displays frozen; main-only annotator running; reserve locked**

## Pre-change state and safeguards

The remote branch was fetched before work began; initial local/remote divergence was `0 0`. Unrelated modified and untracked files were preserved and excluded from both commits. No Field 2 point-annotation file existed before the display change, so the starting counts were 0 reviewed, 0 labeled, and 0 requiring visual re-review.

The frozen Field 1 finalization validator passed for benchmark `da95fd59b7dc5d8ba4add96ed43b398bb7c93749`, its benchmark tag, 15 nested checkpoints, and 6 deployment checkpoints. Checkpoints were verified only by file identity and SHA-256; none was loaded or deserialized. The frozen Field 2 cube-role and blind-sampling validators also passed before materialization.

The display materializer is write-once and refuses any existing preview root, manifest, or addendum. It snapshots all authorized Field 2 source files before and after rendering, rehashes all 40 support masks, and rehashes all three sampling tables. No existing source, support, cube-role, or sampling contract is rewritten.

## Frozen display recipe

The renderer uses only each Field 2 reflectance cube and its frozen valid-support mask. It reads wavelengths from the ENVI reflectance header and selects the closest available band to each target. All 40 cubes have the same wavelength grid and therefore the same selection:

| Channel | Target | Selected wavelength | Python index (0-based) | ENVI band (1-based) | Absolute difference |
|---|---:|---:|---:|---:|---:|
| Red | 670.00 nm | 669.09 nm | 67 | 68 | 0.91 nm |
| Green | 550.00 nm | 550.54 nm | 39 | 40 | 0.54 nm |
| Blue | 480.00 nm | 479.80 nm | 22 | 23 | 0.20 nm |

For each cube and channel, the 2nd and 98th percentiles are computed once from finite pixels inside frozen valid support. Values are linearly stretched and clipped to uint8 0–255. Every pixel outside support is set to RGB `(0, 0, 0)`. There is no adaptive histogram equalization, biological mask, point-specific enhancement, PCA contribution, stored-index contribution, Field 1 contribution, or model output.

Each in-memory render is encoded twice with the fixed PNG settings and required to be byte-identical before it is written. The independent validator then re-reads all 40 ENVI headers and sources, reselects the bands, recomputes the stretches, re-renders all previews in memory, and requires all 40 SHA-256 hashes to match. The immutable manifest and addendum retain per-cube selected bands, exact wavelengths, stretch endpoints, reflectance and support hashes, relative preview paths, and preview SHA-256 hashes.

## Frozen local products

- Display addendum: `metadata/local/contracts/field2_annotation_display_addendum.yaml`
- Display addendum SHA-256: `1854458fb428b7c4b7eb1ec15bbc5b354dc21573f35944ef2cefbbb36ef08a7f`
- Natural-RGB manifest: `metadata/local/reports/field2_annotation_display/field2_natural_rgb_manifest.csv`
- Natural-RGB manifest SHA-256: `cc14f4371485a9610181f480124feffb681e7415af01afbf9c9edcfb5bcb6f41`
- Full-resolution preview directory: `metadata/local/reports/field2_annotation_display/natural_rgb/`
- Preview count: 40; unique preview hashes: 40
- Freeze timestamp: `2026-08-17T20:42:07.991647+00:00`

These machine-local products are intentionally ignored and were not committed. The addendum records implementation commit `3c9f5e1e13abc8fe48898c3dc53f0cd82a5db1f2` as the exact code state used for materialization.

## Annotator behavior and display-version integrity

Natural RGB is the default layer. Existing false-colour reflectance, PCA, stored scalar index, and valid-support/context views remain selectable and are never used to suggest a class. The interface shows a complete-cube overview and a centered magnified neighborhood, 2×/4×/8×/16× zoom, nearest-neighbor magnification, a thin unfilled sampled-pixel square, and an optional grid at 8×/16×. Layer and zoom keyboard shortcuts preserve the selected sample, and exact RGB wavelengths are displayed.

The annotation schema now records the display-recipe version, display-addendum hash, and cube RGB hash for every point. A label-preserving migration marks only previously reviewed records from an older display version as requiring visual re-review. A dedicated filter exposes those records. Explicit review under the new display clears the re-review flag. Biological labels are never assigned or changed by migration.

Because no annotations existed, the neutral package was initialized directly at `field2_annotation_display_natural_rgb_v1`. A live save/resume check created revision 1 with exactly 800 main records, 0 reviewed, 0 labeled, and 0 requiring re-review. The API exposes 800 main samples and zero reserve samples, reports `reserve_exposed: false`, and serves natural RGB as the default.

Relaunch command:

```bash
MPLCONFIGDIR=/tmp/chickpea_matplotlib GDAL_PAM_ENABLED=NO /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/run_field2_point_annotator.py --paths configs/paths.local.yaml --config configs/field2_blind_evaluation.yaml --port 8773 --no-browser
```

URL: `http://127.0.0.1:8773`

## Validation results

- Full repository test suite: **94 passed**.
- Focused Field 2 review/sampling/display suite: **28 passed**.
- Frozen Field 1 supervised finalization: **passed**; 15 nested and 6 deployment checkpoints hash-verified without model loading.
- Frozen Field 2 cube-role contract: **passed**; exact 40-cube inventory and reviewed role/flag totals unchanged.
- Frozen Field 2 blind-sampling contract: **passed**; main 800, reserve 396, combined 1,196, reserve locked.
- Frozen natural-RGB annotation-display addendum: **passed**; 40 previews, reflectance-only references, frozen support references, exact selected bands, wavelength tolerance, preview hashes, black unsupported pixels, and prediction-free provenance.
- Live API and save/resume: **passed**; 800 main, 0 reserve, 0 labels, 0 reviewed, 0 re-review, natural RGB default.

Frozen sampling hashes remained byte-identical:

| Sampling product | SHA-256 |
|---|---|
| Main | `355980b7dddc302a2b82f0193495bca5ce1b2bc9e31a4c8f72a95a20af8ffe0a` |
| Reserve | `37db9d20e86f40d3fe330cf352c10ded1463ebdbe0226bc7ff09ec1204d287fc` |
| Combined | `8b0efeeeeeb461f43d40247c62da3e1cc6e18273bb39fe8a1f5fa89eaaab170d` |

## Scientific safety result

No checkpoint was loaded; no prediction, probability, pseudo-label, suggested biological class, model output, training, model selection, or threshold selection was generated or inspected. No PCA, stored scalar index, Field 1 label, or model artifact contributed to natural RGB. Frozen source imagery, support masks, cube roles, sample IDs, coordinates, main/reserve tables, and contracts remained unchanged. The reserve remains locked and inaccessible.
