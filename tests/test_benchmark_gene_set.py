from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from scripts import build_benchmark_gene_set as builder


def test_select_genes_honors_family_cap(monkeypatch, tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    diagnostics_dir = tmp_path / "diagnostics"
    gt_dir = tmp_path / "gt"
    processed_dir.mkdir()
    diagnostics_dir.mkdir()
    gt_dir.mkdir()

    genes = ["RPL1", "RPL2", "RPL3", "RPL4", "A", "B", "C"]
    pd.DataFrame({
        "gene": genes,
        "selection_score": [100, 99, 98, 97, 30, 29, 28],
    }).to_csv(processed_dir / "gene_subset_300.csv", index=False)
    pd.DataFrame({
        "perturbation": genes,
        "diag_status": ["full"] * len(genes),
        "diag_kd_stratum": ["strong"] * len(genes),
        "diag_batch_stratum": ["low"] * len(genes),
        "diag_kd_residual_ratio": [0.1] * len(genes),
        "diag_batch_js_divergence": [0.01] * len(genes),
        "perturbation_cells": [100] * len(genes),
    }).to_csv(diagnostics_dir / "perturbation_quality_table.csv", index=False)

    def fake_edge_sets(_gt_dir, _genes, _sources, **_kwargs):
        return {
            "toy": {
                ("RPL1", "A"),
                ("RPL2", "A"),
                ("RPL3", "A"),
                ("RPL4", "A"),
                ("A", "B"),
                ("B", "C"),
                ("C", "A"),
            }
        }

    monkeypatch.setattr(builder, "load_ground_truth_edge_sets", fake_edge_sets)
    selected = builder.select_genes(SimpleNamespace(
        processed_dir=processed_dir,
        diagnostics_dir=diagnostics_dir,
        gt_dir=gt_dir,
        preferred_kd_strata=["strong"],
        allowed_batch_strata=["low"],
        n_genes=5,
        max_family_size=2,
        string_min_score=700,
        string_physical_only=True,
    ))

    assert len(selected) == 5
    assert selected["gene"].str.startswith("RPL").sum() == 2
    assert set(selected["selection_pool"]) == {"strict"}
