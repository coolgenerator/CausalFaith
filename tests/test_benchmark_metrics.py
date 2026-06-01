from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import run_benchmark_metrics as metrics


def test_edge_scores_use_max_duplicate_score_and_zero_missing_edges() -> None:
    universe = metrics.directed_universe(["A", "B", "C"])
    edges = pd.DataFrame(
        [
            {"source_gene": "A", "target_gene": "B", "score": 0.2},
            {"source_gene": "A", "target_gene": "B", "score": 0.9},
            {"source_gene": "C", "target_gene": "A", "score": 0.1},
            {"source_gene": "X", "target_gene": "Y", "score": 1.0},
        ]
    )

    scores = metrics.edge_scores(edges, universe)

    assert len(scores) == 6
    assert scores[0] == 0.9  # A -> B
    assert scores[4] == 0.1  # C -> A
    assert np.count_nonzero(scores) == 2


def test_labels_auc_and_topk_include_fallback_provenance() -> None:
    universe = metrics.directed_universe(["A", "B", "C"])
    labels = metrics.labels_for({("A", "B"), ("B", "C")}, universe)
    scores = np.array([0.9, 0.2, 0.0, 0.8, 0.1, 0.0])

    auroc, auprc = metrics.safe_auc(labels, scores)
    rows = metrics.topk_rows(
        "gies_interventional",
        "toy",
        labels,
        scores,
        [2],
        {"fallback_source": "ges_interventional", "fallback_reason": "timeout"},
    )

    assert auroc == 1.0
    assert auprc == 1.0
    assert rows[0]["precision_at_k"] == 1.0
    assert rows[0]["recall_at_k"] == 1.0
    assert rows[0]["is_fallback"] is True
    assert rows[0]["fallback_source"] == "ges_interventional"


def test_safe_auc_returns_nan_for_degenerate_labels() -> None:
    auroc, auprc = metrics.safe_auc(np.zeros(4, dtype=int), np.arange(4))

    assert np.isnan(auroc)
    assert np.isnan(auprc)
