from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


PERTURBATION_COLUMN_CANDIDATES = (
    "perturbation",
    "condition",
    "gene",
    "gene_id",
    "gene_target",
    "target",
    "target_gene",
    "knockdown",
    "guide_target",
    "sgRNA_target",
)

GEMGROUP_COLUMN_CANDIDATES = (
    "gemgroup",
    "gem_group",
    "batch",
    "batch_id",
    "library",
    "library_id",
    "sample",
    "orig.ident",
)

CONTROL_LABEL_HINTS = (
    "control",
    "non-targeting",
    "nontargeting",
    "non_targeting",
    "ntc",
    "safe-targeting",
    "safe_targeting",
    "unperturbed",
)


@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str
    subset_size: int = 300
    min_counts: int = 500
    min_genes: int = 200
    max_pct_mt: float = 20.0
    min_cells_per_perturbation: int = 50
    random_subset_count: int = 20
    random_seed: int = 13
    perturbation_col: str | None = None
    gemgroup_col: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def infer_column(columns: Iterable[str], candidates: Iterable[str], explicit: str | None = None) -> str | None:
    column_set = set(columns)
    if explicit:
        if explicit not in column_set:
            raise ValueError(f"Requested column '{explicit}' is absent from AnnData.obs.")
        return explicit
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def canonicalize_gene_name(value: object) -> str:
    text = str(value).strip()
    for sep in ("+", ",", "|"):
        if sep in text:
            text = text.split(sep)[0].strip()
    if "_" in text and text.lower().startswith(("knockdown_", "perturbation_")):
        text = text.split("_", 1)[1]
    return text


def infer_control_labels(values: pd.Series) -> list[str]:
    labels = []
    for value in values.dropna().astype(str).unique():
        normalized = value.lower().replace(" ", "").replace("-", "_")
        if any(hint.replace("-", "_") in normalized for hint in CONTROL_LABEL_HINTS):
            labels.append(value)
    return sorted(labels)


def matrix_sum(x, axis: int) -> np.ndarray:
    summed = np.asarray(x.sum(axis=axis)).ravel()
    return summed.astype(float)


def subset_columns(adata: ad.AnnData, genes: list[str]) -> ad.AnnData:
    index = pd.Index(adata.var_names)
    present = [gene for gene in genes if gene in index]
    missing = sorted(set(genes) - set(present))
    if missing:
        raise ValueError(f"{len(missing)} requested genes are absent, including: {missing[:10]}")
    return adata[:, present].copy()


def add_qc_covariates(adata: ad.AnnData) -> None:
    var_names = pd.Index(adata.var_names.astype(str))
    mt_mask = var_names.str.upper().str.startswith("MT-")

    if "ncounts" in adata.obs:
        adata.obs["n_counts"] = pd.to_numeric(adata.obs["ncounts"], errors="coerce")
    elif "UMI_count" in adata.obs:
        adata.obs["n_counts"] = pd.to_numeric(adata.obs["UMI_count"], errors="coerce")
    else:
        adata.obs["n_counts"] = matrix_sum(adata.X, axis=1)

    if "ngenes" in adata.obs:
        adata.obs["n_genes_by_counts"] = pd.to_numeric(adata.obs["ngenes"], errors="coerce")
    else:
        if sparse.issparse(adata.X):
            adata.obs["n_genes_by_counts"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()
        else:
            adata.obs["n_genes_by_counts"] = (adata.X > 0).sum(axis=1)

    if "percent_mito" in adata.obs:
        pct_mt = pd.to_numeric(adata.obs["percent_mito"], errors="coerce").fillna(0.0)
        adata.obs["pct_counts_mt"] = pct_mt * 100.0 if pct_mt.max() <= 1.0 else pct_mt
    elif mt_mask.any():
        mt_counts = matrix_sum(adata[:, mt_mask].X, axis=1)
        denom = np.maximum(adata.obs["n_counts"].to_numpy(dtype=float), 1.0)
        adata.obs["pct_counts_mt"] = 100.0 * mt_counts / denom
    else:
        adata.obs["pct_counts_mt"] = 0.0


def filter_cells(adata: ad.AnnData, config: PreprocessConfig) -> tuple[ad.AnnData, dict]:
    before = int(adata.n_obs)
    keep = (
        (adata.obs["n_counts"] >= config.min_counts)
        & (adata.obs["n_genes_by_counts"] >= config.min_genes)
        & (adata.obs["pct_counts_mt"] <= config.max_pct_mt)
    )
    filtered = adata[keep.to_numpy()]
    return filtered, {"cells_before_qc": before, "cells_after_qc": int(filtered.n_obs)}


def add_log1p_layer(adata: ad.AnnData, scale: float = 10_000.0) -> None:
    counts = adata.X.copy()
    totals = matrix_sum(counts, axis=1)
    totals = np.maximum(totals, 1.0)
    if sparse.issparse(counts):
        normalized = sparse.diags(scale / totals) @ counts
        normalized = normalized.tocsr(copy=False)
        normalized.data = np.log1p(normalized.data)
    else:
        normalized = counts * (scale / totals[:, None])
        normalized = np.log1p(normalized)
    adata.layers["log1p"] = normalized


def mean_control_log1p_expression(adata: ad.AnnData, control_mask: np.ndarray) -> np.ndarray:
    control = adata[control_mask, :].X.copy()
    totals = matrix_sum(control, axis=1)
    totals = np.maximum(totals, 1.0)
    if sparse.issparse(control):
        normalized = sparse.diags(10_000.0 / totals) @ control
        normalized = normalized.tocsr(copy=False)
        normalized.data = np.log1p(normalized.data)
    else:
        normalized = control * (10_000.0 / totals[:, None])
        normalized = np.log1p(normalized)
    return matrix_sum(normalized, axis=0) / max(int(np.sum(control_mask)), 1)


def build_gene_metadata(
    adata: ad.AnnData,
    perturbation_col: str,
    control_labels: list[str],
) -> pd.DataFrame:
    obs = adata.obs[[perturbation_col]].copy()
    obs["target_gene"] = obs[perturbation_col].map(canonicalize_gene_name)
    target_counts = obs["target_gene"].value_counts()

    control_mask = obs[perturbation_col].astype(str).isin(control_labels)
    if control_mask.sum() == 0:
        control_mask = np.ones(adata.n_obs, dtype=bool)
    else:
        control_mask = control_mask.to_numpy()

    gene_index = pd.Index(adata.var_names.astype(str))
    expression = mean_control_log1p_expression(adata, control_mask)

    metadata = pd.DataFrame({"gene": gene_index, "control_mean_log1p": expression})
    metadata["perturbation_cells"] = metadata["gene"].map(target_counts).fillna(0).astype(int)
    metadata["is_targeted_and_measured"] = metadata["perturbation_cells"] > 0
    return metadata


def select_gene_subset(
    gene_metadata: pd.DataFrame,
    subset_size: int,
    min_cells_per_perturbation: int,
) -> pd.DataFrame:
    candidates = gene_metadata[
        (gene_metadata["is_targeted_and_measured"])
        & (gene_metadata["perturbation_cells"] >= min_cells_per_perturbation)
        & (gene_metadata["control_mean_log1p"] > 0)
    ].copy()

    if candidates.empty:
        raise ValueError("No genes passed the subset filters. Lower min_cells_per_perturbation.")

    candidates["coverage_rank"] = candidates["perturbation_cells"].rank(pct=True)
    candidates["expression_rank"] = candidates["control_mean_log1p"].rank(pct=True)
    candidates["selection_score"] = 0.55 * candidates["coverage_rank"] + 0.45 * candidates["expression_rank"]
    selected = candidates.sort_values(
        ["selection_score", "perturbation_cells", "control_mean_log1p", "gene"],
        ascending=[False, False, False, True],
    ).head(subset_size)
    selected = selected.reset_index(drop=True)
    selected["subset_id"] = "primary_coverage_expression"
    selected["subset_rank"] = np.arange(1, len(selected) + 1)
    return selected


def build_matched_random_subsets(
    selected: pd.DataFrame,
    gene_metadata: pd.DataFrame,
    subset_size: int,
    subset_count: int,
    seed: int,
    min_cells_per_perturbation: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = gene_metadata[
        (gene_metadata["is_targeted_and_measured"])
        & (gene_metadata["perturbation_cells"] >= min_cells_per_perturbation)
        & (gene_metadata["control_mean_log1p"] > 0)
    ].copy()
    if len(pool) < subset_size:
        return pd.DataFrame()

    pool["expr_bin"] = pd.qcut(
        pool["control_mean_log1p"].rank(method="first"),
        q=min(5, len(pool)),
        labels=False,
        duplicates="drop",
    )
    pool["cells_bin"] = pd.qcut(
        pool["perturbation_cells"].rank(method="first"),
        q=min(5, len(pool)),
        labels=False,
        duplicates="drop",
    )
    selected_bins = selected.merge(pool[["gene", "expr_bin", "cells_bin"]], on="gene", how="left")
    bin_counts = selected_bins.groupby(["expr_bin", "cells_bin"], dropna=False).size()

    rows = []
    primary_genes = set(selected["gene"])
    for subset_idx in range(subset_count):
        subset_rows = []
        used = set()
        for (expr_bin, cells_bin), count in bin_counts.items():
            candidates = pool[
                (pool["expr_bin"] == expr_bin)
                & (pool["cells_bin"] == cells_bin)
                & (~pool["gene"].isin(primary_genes | used))
            ]
            if len(candidates) < count:
                candidates = pool[
                    (~pool["gene"].isin(primary_genes | used))
                    & (pool["expr_bin"].sub(expr_bin).abs() <= 1)
                    & (pool["cells_bin"].sub(cells_bin).abs() <= 1)
                ]
            if len(candidates) < count:
                candidates = pool[~pool["gene"].isin(primary_genes | used)]
            sampled_idx = rng.choice(candidates.index.to_numpy(), size=int(count), replace=False)
            sampled = pool.loc[sampled_idx].copy()
            used.update(sampled["gene"])
            subset_rows.append(sampled)
        random_subset = pd.concat(subset_rows, ignore_index=True).head(subset_size)
        random_subset["subset_id"] = f"matched_random_{subset_idx + 1:02d}"
        random_subset["subset_rank"] = np.arange(1, len(random_subset) + 1)
        rows.append(random_subset)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_outputs(
    adata: ad.AnnData,
    selected_genes: pd.DataFrame,
    random_subsets: pd.DataFrame,
    gene_metadata: pd.DataFrame,
    output_dir: Path,
    summary: dict,
    write_full: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_full:
        full = adata.copy()
        add_log1p_layer(full)
        full.write_h5ad(output_dir / "processed_full.h5ad")
    subset = subset_columns(adata, selected_genes["gene"].tolist())
    add_log1p_layer(subset)
    subset.write_h5ad(output_dir / "processed_300_gene_subset.h5ad")
    selected_genes.to_csv(output_dir / "gene_subset_300.csv", index=False)
    gene_metadata.to_csv(output_dir / "gene_metadata_all_candidates.csv", index=False)
    if not random_subsets.empty:
        random_subsets.to_csv(output_dir / "matched_random_subsets.csv", index=False)
    adata.obs.to_csv(output_dir / "cell_metadata.csv")

    import json

    with (output_dir / "preprocess_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
