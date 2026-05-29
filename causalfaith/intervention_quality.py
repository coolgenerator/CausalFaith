from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class KDConfig:
    perturb_col: str = "perturbation"
    control_label: str = "non-targeting"
    counts_col: str = "ncounts"  # column in adata.obs with per-cell total counts
    target_sum: float = 10_000.0  # standard scanpy default
    min_cells: int = 10
    # KD strata thresholds (residual ratio):
    #   strong  < strong_threshold
    #   medium  in [strong_threshold, weak_threshold)
    #   weak    >= weak_threshold
    strong_threshold: float = 0.20
    weak_threshold: float = 0.50


def _to_dense_column(X, mask: np.ndarray, j: int) -> np.ndarray:
    """Return X[mask, j] as a 1-D dense float array, handling sparse and dense."""
    col = X[mask, j]
    if sparse.issparse(col):
        col = col.toarray()
    return np.asarray(col, dtype=np.float64).ravel()


def normalize_counts(values: np.ndarray, totals: np.ndarray, target_sum: float) -> np.ndarray:
    """Per-cell normalize to target_sum total counts."""
    totals = np.maximum(totals.astype(np.float64), 1.0)
    return values * (target_sum / totals)


def classify_stratum(ratio: float, strong: float, weak: float) -> str:
    if not np.isfinite(ratio):
        return "undefined"
    if ratio < strong:
        return "strong"
    if ratio < weak:
        return "medium"
    return "weak"


def compute_kd_efficiency(
    adata: ad.AnnData,
    config: KDConfig | None = None,
    targets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute per-perturbation KD efficiency.

    Parameters
    ----------
    adata
        AnnData with raw counts in .X and a perturbation column in .obs.
    config
        KDConfig with column names, normalization target, and strata thresholds.
    targets
        Optional list of perturbation labels to process. If None, processes
        every non-control perturbation that has a matching gene in var_names.

    Returns
    -------
    DataFrame with one row per perturbation, columns:
        perturbation, n_cells, target_in_subset,
        ctrl_norm_mean, pert_norm_mean, residual_ratio, kd_stratum
    """
    if config is None:
        config = KDConfig()

    obs = adata.obs
    X = adata.X
    var_names = pd.Index(adata.var_names.astype(str))

    if config.perturb_col not in obs.columns:
        raise ValueError(f"Missing column '{config.perturb_col}' in adata.obs.")
    if config.counts_col not in obs.columns:
        raise ValueError(f"Missing column '{config.counts_col}' in adata.obs.")

    totals = obs[config.counts_col].to_numpy(dtype=np.float64)
    ctrl_mask = (obs[config.perturb_col] == config.control_label).to_numpy()
    n_ctrl = int(ctrl_mask.sum())
    if n_ctrl < config.min_cells:
        raise ValueError(
            f"Only {n_ctrl} control cells (label='{config.control_label}'); "
            f"need at least {config.min_cells}."
        )

    if targets is None:
        targets = sorted(
            set(obs[config.perturb_col].unique()) - {config.control_label}
        )

    rows = []
    for k, target in enumerate(targets):
        pert_mask = (obs[config.perturb_col] == target).to_numpy()
        n_pert = int(pert_mask.sum())
        in_subset = target in var_names

        if not in_subset:
            rows.append({
                "perturbation": target,
                "n_cells": n_pert,
                "target_in_subset": False,
                "ctrl_norm_mean": np.nan,
                "pert_norm_mean": np.nan,
                "residual_ratio": np.nan,
                "kd_stratum": "target_not_measured",
            })
            continue

        if n_pert < config.min_cells:
            rows.append({
                "perturbation": target,
                "n_cells": n_pert,
                "target_in_subset": True,
                "ctrl_norm_mean": np.nan,
                "pert_norm_mean": np.nan,
                "residual_ratio": np.nan,
                "kd_stratum": "insufficient",
            })
            continue

        j = var_names.get_loc(target)
        ctrl_vals = _to_dense_column(X, ctrl_mask, j)
        pert_vals = _to_dense_column(X, pert_mask, j)

        ctrl_norm_mean = normalize_counts(ctrl_vals, totals[ctrl_mask], config.target_sum).mean()
        pert_norm_mean = normalize_counts(pert_vals, totals[pert_mask], config.target_sum).mean()

        if ctrl_norm_mean <= 0:
            ratio = np.nan
            stratum = "undefined"
        else:
            ratio = pert_norm_mean / ctrl_norm_mean
            stratum = classify_stratum(ratio, config.strong_threshold, config.weak_threshold)

        rows.append({
            "perturbation": target,
            "n_cells": n_pert,
            "target_in_subset": True,
            "ctrl_norm_mean": float(ctrl_norm_mean),
            "pert_norm_mean": float(pert_norm_mean),
            "residual_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
            "kd_stratum": stratum,
        })

        if (k + 1) % 50 == 0:
            print(f"  processed {k + 1}/{len(targets)} perturbations", flush=True)

    return pd.DataFrame(rows)


def summarize_kd(df: pd.DataFrame) -> dict:
    """Quick summary stats over a KD efficiency table."""
    valid = df[df["kd_stratum"].isin(["strong", "medium", "weak"])]
    counts = df["kd_stratum"].value_counts().to_dict()
    if valid.empty:
        return {"n_total": int(len(df)), "stratum_counts": counts}
    ratios = valid["residual_ratio"].to_numpy()
    return {
        "n_total": int(len(df)),
        "stratum_counts": counts,
        "ratio_mean": float(np.mean(ratios)),
        "ratio_median": float(np.median(ratios)),
        "ratio_q25": float(np.quantile(ratios, 0.25)),
        "ratio_q75": float(np.quantile(ratios, 0.75)),
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
    }