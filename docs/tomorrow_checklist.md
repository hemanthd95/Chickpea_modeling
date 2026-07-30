# Next-session checklist

Run from `/home/hemanthd95/Chickpea_modeling`.

```bash
git switch agent/ssl-multicube-setup
git pull origin agent/ssl-multicube-setup
conda env update -n chickpea_modeling -f environment.yml
conda activate chickpea_modeling
cp -n configs/paths.local.example.yaml configs/paths.local.yaml
python scripts/project_preflight.py --paths configs/paths.local.yaml
```

Review or share these non-imagery reports:

```text
metadata/local/catalog_summary.csv
metadata/local/project_catalog.csv
```

Do not add Field 2 RTK coordinates to a development manifest. Do not copy raw
imagery, masks, coordinates, model weights, or local absolute paths into Git.
