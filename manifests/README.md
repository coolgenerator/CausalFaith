# Data Manifests

This directory stores small checksum manifests for data artifacts that are too
large to commit to git.

After preprocessing, create a manifest:

```bash
python scripts/write_data_manifest.py \
  data/processed/k562_essential \
  --output manifests/k562_essential_processed.json
```

Upload the corresponding `data/processed/k562_essential/` directory as a GitHub
Release asset or other shared artifact. Teammates can download the directory
and compare file sizes and SHA-256 checksums against the committed manifest.
