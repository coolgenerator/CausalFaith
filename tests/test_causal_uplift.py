from __future__ import annotations

import numpy as np

from scripts.run_causal_uplift import (
    bootstrap_delta,
    metric_value,
    precision_recall_at_k,
    split_method_regime,
)


def test_split_method_regime_keeps_underscored_method_name() -> None:
    assert split_method_regime("mean_difference_interventional") == (
        "mean_difference",
        "interventional",
    )


def test_bootstrap_delta_reports_positive_signal() -> None:
    labels = np.array([1, 1, 0, 0, 0, 0])
    baseline = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.6])
    comparison = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4])

    assert metric_value(labels, comparison, "auprc") > metric_value(labels, baseline, "auprc")
    result = bootstrap_delta(labels, baseline, comparison, "auprc", n_bootstrap=50, seed=7)

    assert result["ci_low"] > 0
    assert result["p_delta_gt_0"] == 1.0


def test_precision_recall_at_k() -> None:
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.4, 0.9, 0.8, 0.1])

    precision, recall = precision_recall_at_k(labels, scores, 2)

    assert precision == 1.0
    assert recall == 1.0
