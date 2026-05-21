# Data Protocol

## Scope

This protocol defines the shared data layer: downloading or importing Replogle
Perturb-seq data, applying common QC, defining the 300-gene working subset, and
producing metadata that Modules A-C can consume without re-deriving preprocessing
choices.

## Data Sources

Primary datasets:

- Replogle et al. K562 essential-scale CRISPRi Perturb-seq.
- Replogle et al. RPE1 essential-scale CRISPRi Perturb-seq.

Preferred access route:

- `python scripts/download_causalbench_data.py --data-dir data/causalbench --dataset all --source scverse`
- AnnData files: `data/causalbench/k562.h5ad` and
  `data/causalbench/rpe1.h5ad`

Fallback route:

- `pertpy.data.replogle_2022_k562_essential`
- `pertpy.data.replogle_2022_rpe1`
- any `.h5ad` file with cell-level perturbation labels and gene expression.

## Shared QC

The preprocessing script computes these cell-level covariates:

- `n_counts`: total UMI/count depth per cell.
- `n_genes_by_counts`: number of detected genes per cell.
- `pct_counts_mt`: percentage of counts assigned to mitochondrial genes.
- `gemgroup`: sequencing/batch/library field when available; otherwise `unknown`.
- `perturbation`: normalized copy of the perturbation-label column.

For the default scverse/PertPy K562 essential file, use `--perturbation-col gene`
because `var_names` are gene symbols. Use `--gemgroup-col batch` for the batch
covariate.

Default filters:

- `n_counts >= 500`
- `n_genes_by_counts >= 200`
- `pct_counts_mt <= 20`

These defaults are intentionally conservative. If the retained cell count falls
too low, record the changed threshold in `preprocess_summary.json`.

## Reproducible K562 Run

The project-standard K562 preprocessing run is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[benchmark]"

python scripts/download_causalbench_data.py \
  --data-dir data/causalbench \
  --dataset k562 \
  --source scverse

PYTHONUNBUFFERED=1 python scripts/preprocess_replogle.py \
  --input-h5ad data/causalbench/k562.h5ad \
  --dataset-name k562_essential \
  --perturbation-col gene \
  --gemgroup-col batch \
  --output-dir data/processed/k562_essential \
  --subset-size 300
```

Known-good K562 output:

- input shape: `310385 cells x 8563 genes`;
- output subset shape: `310385 cells x 300 genes`;
- control perturbation label: `non-targeting`;
- selected genes: `300`;
- matched random subsets: `20`.

After rerunning, regenerate the manifest:

```bash
python scripts/write_data_manifest.py \
  data/processed/k562_essential \
  --output manifests/k562_essential_processed.json
```

Use this quick check before handing off the data:

```bash
python - <<'PY'
import json
import anndata as ad
import pandas as pd

adata = ad.read_h5ad("data/processed/k562_essential/processed_300_gene_subset.h5ad")
genes = pd.read_csv("data/processed/k562_essential/gene_subset_300.csv")
with open("data/processed/k562_essential/preprocess_summary.json", encoding="utf-8") as handle:
    summary = json.load(handle)

assert adata.shape == (310385, 300), adata.shape
assert "log1p" in adata.layers
assert len(genes) == 300
assert summary["selected_genes"] == 300
print("K562 preprocessing check passed.")
PY
```

Do not use `--write-full` for the standard hand-off unless another module
explicitly needs the full QC-filtered AnnData file.

## 300-Gene Subset

The primary 300-gene subset is chosen without using downstream causal-discovery
performance. A gene must:

- be measured in the expression matrix;
- be targeted by a perturbation label;
- have at least `min_cells_per_perturbation` cells after QC;
- have nonzero control expression.

Genes are ranked by a weighted score:

```text
0.55 * perturbation-cell coverage rank + 0.45 * control-expression rank
```

This makes the subset stable, well-powered, and defensible while avoiding
selection on the faithfulness effect size itself.

## Required Outputs

For each dataset, deliver:

- `processed_300_gene_subset.h5ad`
- `gene_subset_300.csv`
- `gene_metadata_all_candidates.csv`
- `matched_random_subsets.csv`
- `cell_metadata.csv`
- `preprocess_summary.json`

`processed_full.h5ad` is optional and should only be written with
`--write-full`, because real K562/RPE1 full copies can be many gigabytes.

## Hand-Off Contract

Module A needs:

- `processed_300_gene_subset.h5ad`
- `cell_metadata.csv`
- perturbation, library-size, mitochondrial-fraction, and gemgroup columns.

Modules B/C need:

- `gene_subset_300.csv`
- exact gene order used in the subset;
- `matched_random_subsets.csv` for sensitivity checks.

No downstream module should silently change the cell filters or gene set. Any
change creates a new `subset_id` and must be recorded in the results table.

## Sharing Artifacts

Do not commit raw `.h5ad` files to normal git. If the team is only using GitHub,
store generated data as a GitHub Release asset or through Git LFS, then commit a
checksum manifest under `manifests/`:

```bash
python scripts/write_data_manifest.py \
  data/processed/k562_essential \
  --output manifests/k562_essential_processed.json
```
