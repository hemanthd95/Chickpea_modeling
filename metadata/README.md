# Metadata layout

`metadata/local/` is workstation-generated and excluded from Git because it may
contain absolute paths or data-derived records. Keep two kinds of output separate:

- Canonical pipeline contracts remain directly in `metadata/local/` when existing
  scripts depend on their stable names, for example `authoritative_manifest.csv`.
- Reviewable stage outputs belong in `metadata/local/reports/<stage>/`.

Current report stages:

- `spatial_grouping/`: cube footprints, overlap, group membership, and group-level
  authoritative class counts, accompanied by a PNG visual-QC overview.
- `normalization/`: fold-specific training-only spectral statistics and PNG
  diagnostics. The hashed normalization contract remains under `contracts/`.
- `sampling/`: leakage-aware observed training-centre summaries and visual QC.
  The indexed centre pool is a training contract, never a replacement for
  exhaustive held-out evaluation.
- `patch_qc/`: center-label, patch-shape, normalization checks, and an observed
  NIR–red–green patch montage for investigator review before model training.

Accepted local analytical contracts are written to `metadata/local/contracts/`.
Each contract includes hashes and the visual artifact reviewed by the investigator.

Major analytical stages should emit a clearly named PNG preview alongside their
tabular reports whenever spatial or statistical structure can be meaningfully
visualized. A PNG supports investigator review but never replaces numeric QC.

Raw imagery and source masks never belong in metadata directories. Report folders
may be deleted and regenerated from the documented scripts.
