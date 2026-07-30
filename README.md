# Chickpea hyperspectral representation learning

This repository tests whether self-supervised learning (SSL) can discover stable,
spectrally and spatially distinct subgroups inside manually labelled weed pixels
from UAV Resonon Pika-L imagery.

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

python scripts/inventory_data.py \
  --root "/home/hemanthd95/Chickpea_modeling/data/Kusi_interns/Masks for modelling/Training_data_for_ss_model" \
  --output configs/data_manifest.csv

python scripts/validate_manifest.py --manifest configs/data_manifest.csv

python scripts/inspect_manifest.py \
  --manifest configs/data_manifest.csv \
  --output configs/data_inspection.csv
```

Open `configs/data_manifest.csv` and correct any file roles that could not be
resolved from filenames. Each row is one cube; paths for PCA, first derivative,
second derivative, NDVI, and either a combined categorical mask or three binary
masks must refer to the same spatial footprint. `reflectance` is optional in the
initial experiment and reserved for the later full-band comparison.

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

The new framework treats those items as explicit controls.
