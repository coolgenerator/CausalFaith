from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

from causalfaith.intervention_quality import (
    KDConfig,
    compute_kd_efficiency,
    summarize_kd,
)


def plot_ratio_histogram(df: pd.DataFrame, out_path: Path, strong: float, weak: float) -> None:
    valid = df.dropna(subset=["residual_ratio"])
    plt.figure(figsize=(8, 5))
    plt.hist(valid["residual_ratio"], bins=40, edgecolor="black")
    plt.axvline(strong, color="green", linestyle="--", label=f"strong threshold = {strong}")
    plt.axvline(weak, color="red", linestyle="--", label=f"weak threshold = {weak}")
    plt.xlabel("Residual expression ratio (perturbed / control)")
    plt.ylabel("Number of perturbations")
    plt.title("KD efficiency: distribution of residual ratios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_ratio_vs_expression(df: pd.DataFrame, out_path: Path) -> None:
    valid = df.dropna(subset=["residual_ratio", "ctrl_norm_mean"])
    plt.figure(figsize=(8, 5))
    plt.scatter(
        valid["ctrl_norm_mean"],
        valid["residual_ratio"],
        alpha=0.5,
        s=20,
    )
    plt.xscale("log")
    plt.xlabel("Control mean expression (normalized counts, log scale)")
    plt.ylabel("Residual ratio")
    plt.title("KD efficiency vs target expression level")
    plt.axhline(1.0, color="grey", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_stratum_counts(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["kd_stratum"].value_counts()
    ordered = ["strong", "medium", "weak", "insufficient", "target_not_measured", "undefined"]
    counts = counts.reindex([s for s in ordered if s in counts.index])
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar", edgecolor="black")
    plt.ylabel("Number of perturbations")
    plt.title("KD stratum counts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True,
                        help="Directory containing processed_300_gene_subset.h5ad")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write kd_efficiency.csv and plots")
    parser.add_argument("--perturb-col", default="perturbation")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--counts-col", default="ncounts")
    parser.add_argument("--target-sum", type=float, default=10_000.0)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--strong-threshold", type=float, default=0.20)
    parser.add_argument("--weak-threshold", type=float, default=0.50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = args.processed_dir / "processed_300_gene_subset.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing input: {h5ad_path}")

    print(f"Loading AnnData from {h5ad_path}", flush=True)
    adata = ad.read_h5ad(h5ad_path)
    print(f"  shape = {adata.shape}", flush=True)

    config = KDConfig(
        perturb_col=args.perturb_col,
        control_label=args.control_label,
        counts_col=args.counts_col,
        target_sum=args.target_sum,
        min_cells=args.min_cells,
        strong_threshold=args.strong_threshold,
        weak_threshold=args.weak_threshold,
    )

    print("Computing KD efficiency per perturbation...", flush=True)
    df = compute_kd_efficiency(adata, config=config)
    print(f"  computed {len(df)} rows", flush=True)

    csv_path = args.output_dir / "kd_efficiency.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}", flush=True)

    summary = summarize_kd(df)
    summary["config"] = {
        "perturb_col": config.perturb_col,
        "control_label": config.control_label,
        "counts_col": config.counts_col,
        "target_sum": config.target_sum,
        "min_cells": config.min_cells,
        "strong_threshold": config.strong_threshold,
        "weak_threshold": config.weak_threshold,
    }
    summary_path = args.output_dir / "kd_efficiency_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote {summary_path}", flush=True)

    print("Drawing plots...", flush=True)
    plot_ratio_histogram(
        df, plot_dir / "kd_residual_ratio_hist.png",
        strong=config.strong_threshold, weak=config.weak_threshold,
    )
    plot_ratio_vs_expression(df, plot_dir / "kd_ratio_vs_expression.png")
    plot_stratum_counts(df, plot_dir / "kd_stratum_counts.png")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
