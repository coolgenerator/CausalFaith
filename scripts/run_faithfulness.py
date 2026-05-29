from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalfaith.faithfulness import (
    FaithfulnessConfig,
    compute_faithfulness_matrix,
    make_stratified_folds,
    read_expression,
    read_gene_list,
    residualize_expression,
    summarize_matrix,
)


def plot_histogram(F: pd.DataFrame, output_path: Path, title: str) -> None:
    values = F.to_numpy().ravel()
    values = values[~np.isnan(values)]

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=50)
    plt.xlabel("Faithfulness score: Wasserstein-1 distance")
    plt.ylabel("Number of gene pairs")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_epsilon_curve(F: pd.DataFrame, output_path: Path, title: str) -> None:
    values = F.to_numpy().ravel()
    values = values[~np.isnan(values)]

    eps_grid = np.quantile(values, np.linspace(0, 0.99, 100))
    proportions = [(values >= eps).mean() for eps in eps_grid]

    plt.figure(figsize=(7, 5))
    plt.plot(eps_grid, proportions)
    plt.xlabel("epsilon threshold")
    plt.ylabel("Proportion of pairs with F >= epsilon")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_heatmap(F: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(F.to_numpy(), aspect="auto")
    plt.colorbar(label="Wasserstein-1 distance")
    plt.xlabel("Affected gene j")
    plt.ylabel("Perturbed gene i")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_top_edges(F: pd.DataFrame, output_path: Path, n: int = 50) -> None:
    top_df = (
        F.stack()
        .reset_index()
        .rename(
            columns={
                "level_0": "perturbed_gene_i",
                "level_1": "affected_gene_j",
                0: "F_score",
            }
        )
        .sort_values("F_score", ascending=False)
        .head(n)
    )
    top_df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perturb-col", default="perturbation")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cells-per-group", type=int, default=5000)
    parser.add_argument("--max-genes", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "diagnostic_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = args.processed_dir / "processed_300_gene_subset.h5ad"
    gene_path = args.processed_dir / "gene_subset_300.csv"

    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing file: {h5ad_path}")

    if not gene_path.exists():
        raise FileNotFoundError(f"Missing file: {gene_path}")

    print(f"Loading AnnData from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    print(f"Loading genes from {gene_path}")
    genes = read_gene_list(gene_path)

    obs = adata.obs.copy()

    if args.perturb_col not in obs.columns:
        raise ValueError(
            f"Missing perturbation column '{args.perturb_col}'. "
            f"Available columns: {list(obs.columns)}"
        )

    print("Top perturbation labels:")
    print(obs[args.perturb_col].value_counts().head(20))

    print("Reading expression matrix")
    X_raw = read_expression(adata, layer="log1p")

    genes = [g for g in genes if g in X_raw.columns]

    if args.max_genes and args.max_genes > 0:
        genes = genes[: args.max_genes]
        print(f"Using only first {len(genes)} genes for test run.")
    else:
        print(f"Using {len(genes)} genes.")

    obs["fold"] = make_stratified_folds(
        obs,
        perturb_col=args.perturb_col,
        seed=args.seed,
    )

    fold_path = args.output_dir / "fold_assignment.csv"
    obs[[args.perturb_col, "fold"]].to_csv(fold_path)
    print(f"Wrote fold assignment to {fold_path}")

    config = FaithfulnessConfig(
        perturb_col=args.perturb_col,
        control_label=args.control_label,
        min_cells=args.min_cells,
        seed=args.seed,
        max_cells_per_group=args.max_cells_per_group,
    )

    print("Residualizing expression matrix")
    X_resid = residualize_expression(X_raw[genes], obs)

    summary = {}

    for version_name, X in [("raw", X_raw[genes]), ("residualized", X_resid[genes])]:
        for fold_name in ["I1", "I2"]:
            print(f"Computing F matrix: version={version_name}, fold={fold_name}")

            F = compute_faithfulness_matrix(
                X=X,
                obs=obs,
                genes=genes,
                fold_name=fold_name,
                config=config,
            )

            csv_path = args.output_dir / f"F_{version_name}_{fold_name}.csv"
            npy_path = args.output_dir / f"F_{version_name}_{fold_name}.npy"

            F.to_csv(csv_path)
            np.save(npy_path, F.to_numpy())

            print(f"Wrote {csv_path}")
            print(f"Wrote {npy_path}")

            summary_key = f"{version_name}_{fold_name}"
            summary[summary_key] = summarize_matrix(F)

            plot_histogram(
                F,
                plot_dir / f"hist_{version_name}_{fold_name}.png",
                f"Faithfulness score distribution: {version_name}, {fold_name}",
            )

            plot_epsilon_curve(
                F,
                plot_dir / f"epsilon_curve_{version_name}_{fold_name}.png",
                f"Epsilon threshold curve: {version_name}, {fold_name}",
            )

            plot_heatmap(
                F,
                plot_dir / f"heatmap_{version_name}_{fold_name}.png",
                f"Faithfulness matrix heatmap: {version_name}, {fold_name}",
            )

            save_top_edges(
                F,
                args.output_dir / f"top_edges_{version_name}_{fold_name}.csv",
                n=50,
            )

    summary_path = args.output_dir / "faithfulness_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote summary to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()