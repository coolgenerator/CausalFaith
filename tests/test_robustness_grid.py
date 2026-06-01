from __future__ import annotations

import pandas as pd

from scripts.build_benchmark_subsets import weighted_capped_sample
from scripts.summarize_robustness_grid import (
    load_topk_rows,
    load_uplift_rows,
    summarize_topk,
    summarize_uplift,
)


def test_weighted_capped_sample_honors_family_cap() -> None:
    ranked = pd.DataFrame({
        "gene": ["RPL1", "RPL2", "RPS1", "RPS2", "A", "B", "C", "D"],
        "benchmark_score": [100, 95, 90, 85, 80, 70, 60, 50],
        "gene_family": ["RPL", "RPL", "RPS", "RPS", "A", "B", "C", "D"],
        "selection_pool": ["strict"] * 8,
    })

    sampled = weighted_capped_sample(
        ranked,
        n_genes=4,
        max_family_size=1,
        seed=11,
        score_power=2.0,
        candidate_multiplier=2,
    )

    assert len(sampled) == 4
    assert sampled["gene"].is_unique
    assert sampled["gene_family"].value_counts().max() == 1


def test_summarize_robustness_grid_aggregates_nested_runs(tmp_path) -> None:
    metrics_dir = tmp_path / "subset_01" / "alpha005_trees50" / "metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame({
        "source": ["corum", "corum"],
        "method": ["pc", "pc"],
        "comparison_regime": ["interventional", "interventional"],
        "metric": ["auprc", "auprc"],
        "delta": [0.1, -0.05],
        "p_delta_gt_0": [0.99, 0.1],
        "baseline_value": [0.2, 0.3],
        "comparison_value": [0.3, 0.25],
    }).to_csv(metrics_dir / "causal_uplift.csv", index=False)
    pd.DataFrame({
        "source": ["corum", "corum"],
        "method": ["pc", "pc"],
        "comparison_regime": ["interventional", "interventional"],
        "k": [25, 25],
        "delta_precision": [0.2, 0.0],
        "precision_baseline": [0.4, 0.5],
        "precision_comparison": [0.6, 0.5],
    }).to_csv(metrics_dir / "causal_uplift_topk.csv", index=False)

    uplift_summary = summarize_uplift(load_uplift_rows(tmp_path))
    topk_summary = summarize_topk(load_topk_rows(tmp_path))

    uplift_row = uplift_summary.iloc[0]
    assert uplift_row["n_runs"] == 2
    assert uplift_row["mean_delta"] == 0.025
    assert uplift_row["positive_fraction"] == 0.5
    assert uplift_row["strong_positive_fraction"] == 0.5

    topk_row = topk_summary.iloc[0]
    assert topk_row["n_runs"] == 2
    assert topk_row["mean_delta_precision"] == 0.1
    assert topk_row["positive_fraction"] == 0.5
