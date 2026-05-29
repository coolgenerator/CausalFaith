from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency


@dataclass(frozen=True)
class ConfoundingConfig:
    perturb_col: str = "perturbation"
    control_label: str = "non-targeting"
    batch_col: str = "gemgroup"
    phase_col: str = "phase"
    s_score_col: str = "S_score"
    g2m_score_col: str = "G2M_score"
    min_cells: int = 10


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence (the squared JS distance from scipy).
    
    Returns a number in [0, 1] where 0 = identical distributions.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p = p / p.sum()
    q = q / q.sum()
    d = jensenshannon(p, q, base=2)
    if not np.isfinite(d):
        return np.nan
    return float(d ** 2)


def compute_batch_divergence(
    obs: pd.DataFrame,
    config: ConfoundingConfig | None = None,
) -> pd.DataFrame:
    """For each perturbation, compute JS divergence of its batch
    distribution against control's batch distribution.
    """
    if config is None:
        config = ConfoundingConfig()

    if config.batch_col not in obs.columns:
        raise ValueError(f"Missing batch column '{config.batch_col}' in obs.")

    batch_levels = sorted(obs[config.batch_col].dropna().astype(str).unique())

    ctrl_mask = (obs[config.perturb_col] == config.control_label).to_numpy()
    if ctrl_mask.sum() < config.min_cells:
        raise ValueError(f"Too few control cells: {ctrl_mask.sum()}")

    ctrl_counts = (
        obs.loc[ctrl_mask, config.batch_col]
        .astype(str)
        .value_counts()
        .reindex(batch_levels, fill_value=0)
        .to_numpy()
    )

    targets = sorted(set(obs[config.perturb_col].unique()) - {config.control_label})
    rows = []
    for k, target in enumerate(targets):
        pert_mask = (obs[config.perturb_col] == target).to_numpy()
        n_pert = int(pert_mask.sum())
        n_batches_in_pert = int(
            obs.loc[pert_mask, config.batch_col].astype(str).nunique()
        )

        if n_pert < config.min_cells:
            rows.append({
                "perturbation": target,
                "n_cells": n_pert,
                "batch_js_divergence": np.nan,
                "n_batches_covered": n_batches_in_pert,
                "diagnostic_status": "insufficient",
            })
            continue

        pert_counts = (
            obs.loc[pert_mask, config.batch_col]
            .astype(str)
            .value_counts()
            .reindex(batch_levels, fill_value=0)
            .to_numpy()
        )

        js = _js_divergence(pert_counts, ctrl_counts)
        rows.append({
            "perturbation": target,
            "n_cells": n_pert,
            "batch_js_divergence": js,
            "n_batches_covered": n_batches_in_pert,
            "diagnostic_status": "ok",
        })

        if (k + 1) % 200 == 0:
            print(f"  batch divergence: {k + 1}/{len(targets)}", flush=True)

    return pd.DataFrame(rows)


def compute_cellcycle_bias(
    obs: pd.DataFrame,
    config: ConfoundingConfig | None = None,
) -> pd.DataFrame:
    """For each perturbation, compute:
    - chi-square p-value of phase distribution vs control
    - mean difference in S_score and G2M_score vs control
    """
    if config is None:
        config = ConfoundingConfig()

    required = [config.phase_col, config.s_score_col, config.g2m_score_col]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise ValueError(
            f"Missing cell-cycle columns: {missing}. "
            f"Did you run scripts/run_cell_cycle_scoring.py first?"
        )

    phase_levels = sorted(obs[config.phase_col].dropna().astype(str).unique())

    ctrl_mask = (obs[config.perturb_col] == config.control_label).to_numpy()
    if ctrl_mask.sum() < config.min_cells:
        raise ValueError(f"Too few control cells: {ctrl_mask.sum()}")

    ctrl_phase = (
        obs.loc[ctrl_mask, config.phase_col]
        .astype(str)
        .value_counts()
        .reindex(phase_levels, fill_value=0)
        .to_numpy()
    )
    ctrl_s_mean = float(obs.loc[ctrl_mask, config.s_score_col].mean())
    ctrl_g2m_mean = float(obs.loc[ctrl_mask, config.g2m_score_col].mean())

    targets = sorted(set(obs[config.perturb_col].unique()) - {config.control_label})
    rows = []
    for k, target in enumerate(targets):
        pert_mask = (obs[config.perturb_col] == target).to_numpy()
        n_pert = int(pert_mask.sum())

        if n_pert < config.min_cells:
            rows.append({
                "perturbation": target,
                "n_cells": n_pert,
                "phase_chi2_pvalue": np.nan,
                "s_score_diff": np.nan,
                "g2m_score_diff": np.nan,
                "diagnostic_status": "insufficient",
            })
            continue

        pert_phase = (
            obs.loc[pert_mask, config.phase_col]
            .astype(str)
            .value_counts()
            .reindex(phase_levels, fill_value=0)
            .to_numpy()
        )

        # chi-square on 2 x K contingency
        try:
            table = np.vstack([pert_phase, ctrl_phase])
            # drop zero-sum columns to avoid warning
            keep = table.sum(axis=0) > 0
            table = table[:, keep]
            if table.shape[1] < 2:
                pvalue = np.nan
            else:
                _, pvalue, _, _ = chi2_contingency(table)
                pvalue = float(pvalue)
        except Exception:
            pvalue = np.nan

        pert_s_mean = float(obs.loc[pert_mask, config.s_score_col].mean())
        pert_g2m_mean = float(obs.loc[pert_mask, config.g2m_score_col].mean())

        rows.append({
            "perturbation": target,
            "n_cells": n_pert,
            "phase_chi2_pvalue": pvalue,
            "s_score_diff": pert_s_mean - ctrl_s_mean,
            "g2m_score_diff": pert_g2m_mean - ctrl_g2m_mean,
            "diagnostic_status": "ok",
        })

        if (k + 1) % 200 == 0:
            print(f"  cellcycle bias: {k + 1}/{len(targets)}", flush=True)

    return pd.DataFrame(rows)


def summarize_batch_divergence(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["batch_js_divergence"])
    if valid.empty:
        return {"n_total": int(len(df)), "n_valid": 0}
    js = valid["batch_js_divergence"].to_numpy()
    return {
        "n_total": int(len(df)),
        "n_valid": int(len(valid)),
        "js_mean": float(js.mean()),
        "js_median": float(np.median(js)),
        "js_q25": float(np.quantile(js, 0.25)),
        "js_q75": float(np.quantile(js, 0.75)),
        "js_q90": float(np.quantile(js, 0.90)),
        "js_max": float(js.max()),
    }


def summarize_cellcycle_bias(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["phase_chi2_pvalue"])
    if valid.empty:
        return {"n_total": int(len(df)), "n_valid": 0}
    # Bonferroni-corrected significance count
    n = len(valid)
    bonferroni = 0.05 / max(n, 1)
    n_sig = int((valid["phase_chi2_pvalue"] < bonferroni).sum())
    return {
        "n_total": int(len(df)),
        "n_valid": n,
        "bonferroni_threshold": bonferroni,
        "n_phase_shifted_bonferroni": n_sig,
        "s_score_diff_median": float(valid["s_score_diff"].median()),
        "g2m_score_diff_median": float(valid["g2m_score_diff"].median()),
        "s_score_diff_max_abs": float(valid["s_score_diff"].abs().max()),
        "g2m_score_diff_max_abs": float(valid["g2m_score_diff"].abs().max()),
    }