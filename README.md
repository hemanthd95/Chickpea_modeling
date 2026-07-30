# Chickpea hyperspectral representation learning

This repository tests whether self-supervised learning (SSL) can discover stable,
spectrally and spatially distinct subgroups inside manually labelled weed pixels
from UAV Resonon Pika-L imagery. All reported experiments use only the project's
observed imagery, masks, tables, and RTK observations. Synthetic observations are
not permitted in training, validation, testing, or reported results.

## Scientific scope

The defensible claim is **weed phenotype discovery**, not automatic weed-species
identification. Cluster IDs become biological classes only after field or expert
validation. The one RTK-surveyed tall-grass class is used as an external test and
is never used to fit the unsupervised clusters.

The analysis is deliberately hierarchical:

1. Use manual masks to separate chickpea, weed, and soil.
2. Train SSL representations on vegetation-centred patches.
3. Cluster only embeddings whose centre pixels are inside weed masks.
4. Validate cluster stability across seeds, cubes, and fields.
5. Test whether one cluster enriches for the independently surveyed tall grass.
6. Compare SSL features with supervised baselines using field-held-out splits.

This prevents the much larger soil class, chickpea rows, and alleys from
artificially driving the hidden weed clusters.

## Quick start on the HP Z6

```bash
cd /home/hemanthd95/Chickpea_modeling
conda env update -n chickpea_modeling -f environment.yml
conda activate chickpea_modeling

cp configs/paths.local.example.yaml configs/paths.local.yaml
# Review paths.local.yaml, then:
python scripts/project_preflight.py --paths configs/paths.local.yaml
```

The preflight writes local inventories to `metadata/local/`. Review those reports
before model training. Files under `metadata/local/` contain workstation paths and
are intentionally not versioned.

## Experimental boundary

- **Field 1:** development, spatially blocked cross-validation, supervised
  baselines, SSL fitting, and model selection.
- **Field 2:** locked external field. Its RTK tall-grass coordinates are not used
  until the Field 1 pipeline is frozen.
- **Primary inputs:** calibrated/georectified 150-band reflectance and wavelength
  metadata.
- **Derived baseline:** PCA and NDVI.
- **Legacy ablation only:** first/second differences calculated across PCA
  components. Component order represents explained variance rather than
  wavelength, so these are not interpreted as spectral derivatives.

See `docs/study_protocol.md` for the preregistered analysis logic and
`docs/progress_log.md` for the daily record.

Then run the leakage-safe starter experiment:

```bash
python -m chickpea_ssl.train --config configs/experiment.yaml
```

## Data policy

Large imagery, model weights, derived arrays, and results are intentionally
excluded from Git. The repository stores code, configuration, manifests without
private data, and summary metrics.

## Why the earlier notebook needs revision

The original denoising autoencoder was a useful proof of concept, but:

- PCA, derivative, and NDVI patches received different random rotations/flips,
  destroying pixel-to-pixel channel alignment.
- Random patches from the same cube were used without field-held-out testing,
  allowing spatial leakage.
- K-means was fit to every pixel, so soil and chickpea could consume clusters
  intended to represent five weed types.
- Seven clusters were fixed in advance without model-selection or stability
  analysis.
- A globally pooled reconstruction bottleneck can favor scene appearance over
  fine spectral differences.
- Median filtering can inflate apparent spatial coherence.
- Differences between ordered PCA components are not physically interpretable as
  wavelength derivatives.

The new framework treats those items as explicit controls.
