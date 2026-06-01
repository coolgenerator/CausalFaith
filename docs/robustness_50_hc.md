# 50-Gene High-Confidence Robustness Pilot

Run date: 2026-05-31

This pilot checks whether the apparent benefit of interventional data survives
across multiple high-confidence 50-gene subsets and a small parameter grid. It
is intended as a robustness screen, not an exhaustive benchmark.

## Scope

- Subsets: 3 high-confidence 50-gene subsets.
  - `subset_01`: deterministic top-ranked genes.
  - `subset_02`, `subset_03`: weighted samples from the high-confidence pool.
- Ground truth: pooled ChIP-Atlas, CORUM, and STRING; STRING restricted to
  physical links with `combined_score >= 700`.
- Methods: PC, FCI, GRNBoost.
- Regimes: observational vs interventional.
- Grid:
  - `alpha005_trees50`: PC/FCI `alpha=0.005`, GRNBoost `n_estimators=50`.
  - `alpha010_trees100`: PC/FCI `alpha=0.01`, GRNBoost `n_estimators=100`.
  - `alpha020_trees200`: PC/FCI `alpha=0.02`, GRNBoost `n_estimators=200`.
- Sampling budget: observational max 500 cells; interventional max 1000 cells.
- Bootstrap: 200 resamples per run.

GES/GIES were not included in this pilot because they are much slower and GIES
currently needs a fallback path in this repository. They should be evaluated in
a separate focused run.

## Key Aggregate Results

Mean AUPRC uplift is computed as `interventional - observational` across the
9 subset/config runs.

| Source | Method | Mean baseline | Mean interventional | Mean delta | Positive runs | Strong-positive runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CORUM | GRNBoost | 0.1269 | 0.1807 | +0.0538 | 9/9 | 9/9 |
| CORUM | FCI | 0.1190 | 0.1356 | +0.0167 | 9/9 | 6/9 |
| CORUM | PC | 0.1146 | 0.1229 | +0.0083 | 9/9 | 1/9 |
| Pooled | GRNBoost | 0.1998 | 0.2398 | +0.0400 | 9/9 | 8/9 |
| Pooled | FCI | 0.1856 | 0.1930 | +0.0075 | 9/9 | 2/9 |
| Pooled | PC | 0.1821 | 0.1856 | +0.0035 | 9/9 | 0/9 |
| STRING | GRNBoost | 0.1776 | 0.2121 | +0.0345 | 9/9 | 7/9 |
| STRING | FCI | 0.1641 | 0.1693 | +0.0052 | 9/9 | 2/9 |
| STRING | PC | 0.1610 | 0.1636 | +0.0026 | 8/9 | 0/9 |
| ChIP-Atlas | GRNBoost | 0.0066 | 0.0039 | -0.0026 | 2/9 | 0/9 |

`Strong-positive` means the bootstrap probability `P(delta > 0) >= 0.95`.
ChIP-Atlas has only 4-5 positives in these subsets, so it is too sparse to
serve as the primary readout.

Precision@25 also improves in the main biological references:

| Source | Method | Mean baseline P@25 | Mean interventional P@25 | Mean delta P@25 |
| --- | --- | ---: | ---: | ---: |
| CORUM | PC | 0.3022 | 0.4667 | +0.1644 |
| CORUM | FCI | 0.3289 | 0.4533 | +0.1244 |
| CORUM | GRNBoost | 0.2578 | 0.3733 | +0.1156 |
| Pooled | PC | 0.3822 | 0.4933 | +0.1111 |
| Pooled | GRNBoost | 0.3556 | 0.4444 | +0.0889 |
| Pooled | FCI | 0.4356 | 0.5111 | +0.0756 |
| STRING | PC | 0.3556 | 0.4667 | +0.1111 |
| STRING | GRNBoost | 0.3156 | 0.4178 | +0.1022 |
| STRING | FCI | 0.4089 | 0.4489 | +0.0400 |

## Interpretation

The pilot strengthens the claim that interventional information is useful in
this project, especially for GRNBoost. The direction is stable across all
9 subset/config runs on CORUM, pooled, and STRING references. FCI shows a
smaller but still consistent gain. PC has modest AUPRC uplift but larger
Top-K precision gains.

The result does not prove that every causal method benefits from intervention
data. It also does not yet cover full 300-gene runs, GES/GIES, larger cell
budgets, or external held-out datasets.

## Reproduce

Build the subsets:

```bash
PYTHONUNBUFFERED=1 python scripts/build_benchmark_subsets.py \
  --processed-dir data/processed/k562_essential \
  --diagnostics-dir results/diagnostics \
  --gt-dir data/causalbench \
  --output-dir results/robustness_50_hc \
  --n-subsets 3 \
  --n-genes 50 \
  --max-family-size 8 \
  --seed 42 \
  --string-min-score 700 \
  --string-physical-only
```

Run the pilot:

```bash
PYTHONUNBUFFERED=1 python scripts/run_robustness_pilot.py \
  --subsets-dir results/robustness_50_hc \
  --processed-dir data/processed/k562_essential \
  --gt-dir data/causalbench \
  --methods pc fci grnboost \
  --regimes observational interventional \
  --max-obs-cells 500 \
  --max-all-cells 1000 \
  --n-bootstrap 200 \
  --n-jobs 4
```

Summarize:

```bash
PYTHONUNBUFFERED=1 python scripts/summarize_robustness_grid.py \
  --subsets-dir results/robustness_50_hc
```

Main output files:

- `results/robustness_50_hc/subsets_manifest.csv`
- `results/robustness_50_hc/summary/robustness_uplift_summary.csv`
- `results/robustness_50_hc/summary/robustness_topk_summary.csv`
- `results/robustness_50_hc/summary/robustness_summary.json`
