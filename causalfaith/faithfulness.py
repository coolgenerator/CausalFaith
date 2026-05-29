from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


@dataclass
class FaithfulnessConfig:
    perturb_col: str = "perturbation"
    control_label: str = "non-targeting"
    min_cells: int = 10
    seed: int = 42
    max_cells_per_group: int = 5000


def read_expression(adata: ad.AnnData, layer: str = "log1p") -> pd.DataFrame:
    """
    Return expression matrix as cells x genes DataFrame.
    Prefer adata.layers['log1p'] if it exists.
    """
    if layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    if sparse.issparse(X):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)
    return pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)


def read_gene_list(gene_path: Path) -> list[str]:
    genes_df = pd.read_csv(gene_path)

    for col in ["gene", "gene_symbol", "symbol", "gene_name"]:
        if col in genes_df.columns:
            return genes_df[col].astype(str).tolist()

    return genes_df.iloc[:, 0].astype(str).tolist()


def choose_existing_column(obs: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in obs.columns:
            return col
    return None


def make_stratified_folds(
    obs: pd.DataFrame,
    perturb_col: str,
    seed: int = 42,
) -> pd.Series:
    """
    Split cells into I1 / I2, stratified by perturbation target.
    """
    counts = obs[perturb_col].value_counts()
    valid_labels = counts[counts >= 2].index

    obs_use = obs[obs[perturb_col].isin(valid_labels)].copy()

    idx_i1, idx_i2 = train_test_split(
        obs_use.index,
        test_size=0.5,
        random_state=seed,
        stratify=obs_use[perturb_col],
    )

    fold = pd.Series("unused", index=obs.index, name="fold")
    fold.loc[idx_i1] = "I1"
    fold.loc[idx_i2] = "I2"

    return fold


def residualize_expression(
    X: pd.DataFrame,
    obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Residualize expression against technical covariates:
    library size / counts, mitochondrial fraction, and gemgroup / batch if available.
    """
    covariate_parts = []

    count_col = choose_existing_column(
        obs,
        ["n_counts", "total_counts", "library_size", "nCount_RNA"],
    )
    mt_col = choose_existing_column(
        obs,
        ["pct_counts_mt", "mitochondrial_fraction", "percent_mito", "mt_frac"],
    )
    batch_col = choose_existing_column(
        obs,
        ["gemgroup", "batch", "gem_group", "library"],
    )

    used_covariates = []

    if count_col is not None:
        covariate_parts.append(np.log1p(obs[[count_col]].astype(float)))
        used_covariates.append(count_col)

    if mt_col is not None:
        covariate_parts.append(obs[[mt_col]].astype(float))
        used_covariates.append(mt_col)

    if batch_col is not None:
        enc = OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
        )
        batch_encoded = enc.fit_transform(obs[[batch_col]].astype(str))
        batch_df = pd.DataFrame(
            batch_encoded,
            index=obs.index,
            columns=enc.get_feature_names_out([batch_col]),
        )
        covariate_parts.append(batch_df)
        used_covariates.append(batch_col)

    if not covariate_parts:
        print("No technical covariates found. Returning raw expression.")
        return X.copy()

    C = pd.concat(covariate_parts, axis=1)
    C = C.loc[X.index]

    model = LinearRegression()
    model.fit(C.values, X.values)

    fitted = model.predict(C.values)
    residuals = X.values - fitted

    print(f"Residualized using covariates: {used_covariates}")
    return pd.DataFrame(residuals, index=X.index, columns=X.columns)


def _maybe_downsample(index_values: pd.Index, max_n: int, rng: np.random.Generator) -> pd.Index:
    if max_n is None or max_n <= 0:
        return index_values

    if len(index_values) <= max_n:
        return index_values

    chosen = rng.choice(index_values.to_numpy(), size=max_n, replace=False)
    return pd.Index(chosen)


def compute_faithfulness_matrix(
    X: pd.DataFrame,
    obs: pd.DataFrame,
    genes: list[str],
    fold_name: str,
    config: FaithfulnessConfig,
) -> pd.DataFrame:
    """
    F[i, j] = W1(
        expression of gene j in control cells,
        expression of gene j in cells where gene i is perturbed
    )
    """
    rng = np.random.default_rng(config.seed)

    obs_fold = obs[obs["fold"] == fold_name].copy()
    X_fold = X.loc[obs_fold.index, genes]

    control_idx = obs_fold[obs_fold[config.perturb_col] == config.control_label].index
    control_idx = _maybe_downsample(control_idx, config.max_cells_per_group, rng)

    if len(control_idx) < config.min_cells:
        raise ValueError(
            f"Too few control cells in {fold_name}: {len(control_idx)}. "
            f"Check control_label='{config.control_label}'."
        )

    F = pd.DataFrame(np.nan, index=genes, columns=genes)

    control_values = X_fold.loc[control_idx, genes].to_numpy()

    for k, gene_i in enumerate(genes):
        pert_idx = obs_fold[obs_fold[config.perturb_col] == gene_i].index
        pert_idx = _maybe_downsample(pert_idx, config.max_cells_per_group, rng)

        if len(pert_idx) < config.min_cells:
            continue

        pert_values = X_fold.loc[pert_idx, genes].to_numpy()

        for j, gene_j in enumerate(genes):
            F.loc[gene_i, gene_j] = wasserstein_distance(
                control_values[:, j],
                pert_values[:, j],
            )

        if (k + 1) % 25 == 0:
            print(f"{fold_name}: finished {k + 1}/{len(genes)} perturbation genes", flush=True)

    return F


def summarize_matrix(F: pd.DataFrame) -> dict:
    values = F.to_numpy().ravel()
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return {"num_scores": 0}

    return {
        "num_scores": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
    }