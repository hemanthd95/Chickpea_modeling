# Metadata layout

`metadata/local/` is workstation-generated and excluded from Git because it may
contain absolute paths or data-derived records. Keep two kinds of output separate:

- Canonical pipeline contracts remain directly in `metadata/local/` when existing
  scripts depend on their stable names, for example `authoritative_manifest.csv`.
- Reviewable stage outputs belong in `metadata/local/reports/<stage>/`.

Current report stages:

- `spatial_grouping/`: cube footprints, overlap, group membership, and group-level
  authoritative class counts.

Raw imagery and source masks never belong in metadata directories. Report folders
may be deleted and regenerated from the documented scripts.
