# CausalFaith

Data pipeline for testing epsilon-interventional faithfulness on real
CausalBench / Replogle Perturb-seq data.

This repository currently focuses on a reproducible data and diagnostics layer:

- downloading and caching CausalBench/Replogle K562 and RPE1 data;
- preprocessing Replogle K562-essential and RPE1 screens;
- construction of a defensible 300-gene working subset;
- Module A faithfulness matrix computation on raw and residualized expression;
- Module B intervention-quality and confounding diagnostics;
- metadata outputs that Modules A-C can use without changing QC or gene-set
  choices.

## Quick Start

Create an environment from a fresh clone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[benchmark]"
```

This installs the local preprocessing package plus CausalBench. Add
`.[benchmark,singlecell]` only if you want to load data directly through
`pertpy` instead of using the downloaded `.h5ad` files.

### Reproduce the K562 preprocessing run

This is the known-good command sequence for the processed K562 output used by
the project.

Download the K562 AnnData file:

```bash
python scripts/download_causalbench_data.py \
  --data-dir data/causalbench \
  --dataset k562 \
  --source scverse
```

This creates or reuses `data/causalbench/k562.h5ad`. The scverse/PertPy mirror is
the default because it is much smaller and more automation-friendly than the
original Figshare raw files.

Run preprocessing with the verified K562 column mapping:

```bash
PYTHONUNBUFFERED=1 python scripts/preprocess_replogle.py \
  --input-h5ad data/causalbench/k562.h5ad \
  --dataset-name k562_essential \
  --perturbation-col gene \
  --gemgroup-col batch \
  --output-dir data/processed/k562_essential \
  --subset-size 300
```

Expected terminal summary:

```text
Loaded AnnData with shape 310385 cells x 8563 genes.
QC retained 310385 / 310385 cells.
Inferred control labels: ['non-targeting']
Selected 300 genes.
Wrote processed outputs to data/processed/k562_essential
```

Quickly verify the produced files:

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

Finally, regenerate the checksum manifest after any fresh run:

```bash
python scripts/write_data_manifest.py \
  data/processed/k562_essential \
  --output manifests/k562_essential_processed.json
```

The main outputs are:

- `processed_300_gene_subset.h5ad`: same cells, restricted to selected genes.
- `gene_subset_300.csv`: selected genes plus coverage/expression metadata.
- `cell_metadata.csv`: cell-level perturbation and QC covariates.
- `preprocess_summary.json`: parameters and QC counts for reproducibility.

Use `--write-full` only if you really need the full QC-filtered AnnData; it can
be many gigabytes.

## Integrated Analysis Workflow

Run the analysis modules after the K562 preprocessing step has produced
`data/processed/k562_essential/processed_300_gene_subset.h5ad` and
`data/processed/k562_essential/gene_subset_300.csv`.

### Module A: faithfulness matrices

Module A computes Wasserstein-1 faithfulness scores for perturbation-gene pairs,
using both raw log-normalized expression and expression residualized against
available technical covariates.

```bash
python scripts/run_faithfulness.py \
  --processed-dir data/processed/k562_essential \
  --output-dir results/faithfulness
```

Main outputs:

- `F_raw_I1.csv`, `F_raw_I2.csv`: faithfulness matrices on the two stratified folds.
- `F_residualized_I1.csv`, `F_residualized_I2.csv`: covariate-residualized matrices.
- `faithfulness_summary.json`: distribution summaries for each matrix.
- `top_edges_*.csv`: highest-scoring perturbation-gene pairs.
- `diagnostic_plots/`: histograms, epsilon curves, and heatmaps.

### Module B: intervention quality and confounding diagnostics

Module B quantifies knockdown strength, batch imbalance, and cell-cycle bias.

```bash
python scripts/run_intervention_quality.py \
  --processed-dir data/processed/k562_essential \
  --output-dir results/diagnostics
```

Cell-cycle scoring uses the full raw K562 matrix and streams chunks to avoid
materializing the full normalized matrix in memory:

```bash
python scripts/run_cell_cycle_scoring.py \
  --raw-h5ad data/causalbench/k562.h5ad \
  --output-csv data/processed/k562_essential/cell_cycle_scores.csv
```

Then join the cell-cycle scores to the processed metadata and compute
confounding diagnostics:

```bash
python scripts/run_confounding.py \
  --processed-dir data/processed/k562_essential \
  --cellcycle-csv data/processed/k562_essential/cell_cycle_scores.csv \
  --output-dir results/diagnostics
```

Finally, assemble the per-perturbation diagnostic table:

```bash
python scripts/build_perturbation_quality_table.py \
  --diagnostics-dir results/diagnostics
```

Main outputs:

- `kd_efficiency.csv`: per-perturbation knockdown efficiency and KD strata.
- `batch_divergence.csv`: Jensen-Shannon batch divergence against controls.
- `cellcycle_bias.csv`: phase-shift p-values and cell-cycle score differences.
- `perturbation_quality_table.csv`: joined master table with `diag_` columns.
- `plots/`: marginal and joint diagnostic figures.

The integration branch includes a small committed snapshot under
`results/diagnostics/` for review. New generated outputs still belong under
ignored paths such as `results/` or `data/` unless the team explicitly decides
to commit a reproducible artifact.

## Workflow Quality Checks

Before opening or updating a PR, run the checks that match the change size.

For lightweight code and documentation changes:

```bash
python -m compileall causalfaith scripts
python -m ruff check causalfaith scripts
```

For package and import sanity:

```bash
python -m pip install -e ".[dev,benchmark]"
python - <<'PY'
import causalfaith.faithfulness
import causalfaith.intervention_quality
import causalfaith.confounding

print("module import check passed")
PY
```

For data workflow quality, prefer a staged check instead of rerunning the most
expensive steps first:

1. Verify preprocessing outputs with the shape and manifest checks above.
2. Run Module A with `--max-genes 20` to validate the faithfulness path quickly.
3. Run `run_intervention_quality.py` on the processed subset.
4. Run the full cell-cycle scoring only when the raw K562 `.h5ad` is available
   and the machine has enough memory for chunked dense slices.
5. Rebuild `perturbation_quality_table.csv` and confirm row counts, missing
   values, and diagnostic strata in the summary JSON files.

### Other data options

To download both K562 and RPE1:

```bash
python scripts/download_causalbench_data.py \
  --data-dir data/causalbench \
  --dataset all \
  --source scverse
```

If you also want CausalBench's own processed `.npz` files for
`causalbench_run`, add `--make-npz`. If you need the benchmark evaluation
resources, add `--with-evaluation-resources`.

Or run preprocessing from `pertpy`:

```bash
python scripts/preprocess_replogle.py \
  --dataset k562_essential \
  --output-dir data/processed/k562_essential \
  --subset-size 300
```

## CausalBench Smoke Test

This optional command checks that CausalBench itself can run. The parameters
below are deliberately cheap smoke-test settings, not final experiment settings:

```bash
mkdir -p results/causalbench

causalbench_run \
  --dataset_name weissmann_k562 \
  --output_directory results/causalbench/k562_pc_smoke \
  --data_directory data/causalbench \
  --training_regime observational \
  --model_name pc \
  --subset_data 0.05 \
  --model_seed 0 \
  --do_filter \
  --max_path_length -1 \
  --omission_estimation_size 0
```

Useful CausalBench parameters:

- `--dataset_name`: `weissmann_k562` or `weissmann_rpe1`.
- `--training_regime`: `observational`, `partial_interventional`, or
  `interventional`.
- `--model_name`: CausalBench method to run, such as `pc`, `ges`, `gies`,
  `grnboost`, or `DCDI-G`.
- `--subset_data`: fraction of training cells to use. Keep this small for smoke
  tests; use `1.0` for full runs.
- `--do_filter`: applies CausalBench's strong-perturbation filter. This is useful
  for reproducing filtered CausalBench runs, but it is separate from this
  project's preprocessing.
- `--omission_estimation_size`: set to `0` for smoke tests; larger values add
  false-omission-rate estimation cost.

For this repository's data pipeline, the important parameters are instead
`preprocess_replogle.py`'s `--input-h5ad`, `--perturbation-col`,
`--gemgroup-col`, `--subset-size`, and QC thresholds.

## Pipeline Outputs

The preprocessing step writes:

- `processed_300_gene_subset.h5ad`
- `gene_subset_300.csv`
- `gene_metadata_all_candidates.csv`
- `matched_random_subsets.csv`
- `cell_metadata.csv`
- `preprocess_summary.json`

## Sharing Data Artifacts

`data/` is ignored by git because raw `.h5ad` files and processed single-cell
artifacts can be large. If the team is only using GitHub, the recommended
hand-off is:

1. Run preprocessing once.
2. Upload `data/processed/k562_essential/` as a GitHub Release artifact, or use
   Git LFS if the repo has LFS configured.
3. Commit a small checksum manifest:

```bash
python scripts/write_data_manifest.py \
  data/processed/k562_essential \
  --output manifests/k562_essential_processed.json
```

The next person should download the shared folder into the same path and verify
that file sizes and SHA-256 checksums match the manifest. This keeps normal git
history small while still preventing everyone from rerunning preprocessing.

## Project Documents

- [Data Protocol](docs/data_protocol.md)
- [Source Notes](docs/source_notes.md)
