from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

from causalfaith.confounding import (
    ConfoundingConfig,
    compute_batch_divergence,
    compute_cellcycle_bias,
    summarize_batch_divergence,
    summarize_cellcycle_bias,
)


def plot_batch_histogram(df: pd.DataFrame, out_path: Path) -> None:
    valid = df.dropna(subset=["batch_js_divergence"])
    plt.figure(figsize=(8, 5))
    plt.hist(valid["batch_js_divergence"], bins=40, edgecolor="black")
    plt.xlabel("Batch JS divergence vs control")
    plt.ylabel("Number of perturbations")
    plt.title("Batch confounding: distribution of JS divergence")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_batch_vs_ncells(df: pd.DataFrame, out_path: Path) -> None:
    valid = df.dropna(subset=["batch_js_divergence"])
    plt.figure(figsize=(8, 5))
    plt.scatter(valid["n_cells"], valid["batch_js_divergence"], alpha=0.5, s=20)
    plt.xscale("log")
    plt.xlabel("Cells per perturbation (log scale)")
    plt.ylabel("Batch JS divergence")
    plt.title("Batch divergence vs perturbation cell count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cellcycle_score_diff(df: pd.DataFrame, out_path: Path) -> None:
    valid = df.dropna(subset=["s_score_diff", "g2m_score_diff"])
    plt.figure(figsize=(8, 7))
    plt.scatter(valid["s_score_diff"], valid["g2m_score_diff"], alpha=0.5, s=20)
    plt.axhline(0, color="grey", linestyle=":", alpha=0.5)
    plt.axvline(0, color="grey", linestyle=":", alpha=0.5)
    plt.xlabel("S_score diff (perturbed - control)")
    plt.ylabel("G2M_score diff (perturbed - control)")
    plt.title("Cell cycle shift per perturbation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pvalue_histogram(df: pd.DataFrame, out_path: Path) -> None:
    valid = df.dropna(subset=["phase_chi2_pvalue"])
    plt.figure(figsize=(8, 5))
    plt.hist(valid["phase_chi2_pvalue"], bins=40, edgecolor="black")
    plt.xlabel("Chi-square p-value (phase shift vs control)")
    plt.ylabel("Number of perturbations")
    plt.title("Phase distribution test p-values")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--cellcycle-csv", type=Path, required=True,
                        help="CSV from run_cell_cycle_scoring.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perturb-col", default="perturbation")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--batch-col", default="gemgroup")
    parser.add_argument("--min-cells", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = args.processed_dir / "processed_300_gene_subset.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing: {h5ad_path}")
    if not args.cellcycle_csv.exists():
        raise FileNotFoundError(
            f"Missing cell cycle CSV: {args.cellcycle_csv}. "
            "Run scripts/run_cell_cycle_scoring.py first."
        )

    print(f"Loading AnnData from {h5ad_path}", flush=True)
    adata = ad.read_h5ad(h5ad_path)
    obs = adata.obs.copy()
    print(f"  obs shape = {obs.shape}", flush=True)
    # we don't need expression for confounding, free it
    del adata

    print(f"Loading cell cycle scores from {args.cellcycle_csv}", flush=True)
    cc = pd.read_csv(args.cellcycle_csv, index_col=0)
    print(f"  cc shape = {cc.shape}", flush=True)

    # Align cell IDs. obs.index and cc.index should both be cell barcodes.
    overlap = obs.index.intersection(cc.index)
    print(f"  cell ID overlap: {len(overlap)} / {len(obs)} obs cells", flush=True)
    if len(overlap) < 0.95 * len(obs):
        raise RuntimeError(
            "Cell ID overlap is suspiciously low. Check that "
            "run_cell_cycle_scoring.py was run on the same source dataset."
        )

    obs = obs.join(cc, how="left")
    print(f"  obs columns after join: {list(obs.columns)[-5:]}", flush=True)

    config = ConfoundingConfig(
        perturb_col=args.perturb_col,
        control_label=args.control_label,
        batch_col=args.batch_col,
        min_cells=args.min_cells,
    )

    # --- Batch divergence ---
    print("\n[1/2] Computing batch JS divergence...", flush=True)
    df_batch = compute_batch_divergence(obs, config=config)
    batch_path = args.output_dir / "batch_divergence.csv"
    df_batch.to_csv(batch_path, index=False)
    print(f"Wrote {batch_path}", flush=True)

    summary_batch = summarize_batch_divergence(df_batch)
    print("Batch summary:", json.dumps(summary_batch, indent=2))

    plot_batch_histogram(df_batch, plot_dir / "batch_js_hist.png")
    plot_batch_vs_ncells(df_batch, plot_dir / "batch_js_vs_ncells.png")

    # --- Cell cycle bias ---
    print("\n[2/2] Computing cell cycle bias...", flush=True)
    df_cc = compute_cellcycle_bias(obs, config=config)
    cc_path = args.output_dir / "cellcycle_bias.csv"
    df_cc.to_csv(cc_path, index=False)
    print(f"Wrote {cc_path}", flush=True)

    summary_cc = summarize_cellcycle_bias(df_cc)
    print("Cell cycle summary:", json.dumps(summary_cc, indent=2))

    plot_cellcycle_score_diff(df_cc, plot_dir / "cellcycle_score_diff.png")
    plot_pvalue_histogram(df_cc, plot_dir / "cellcycle_pvalue_hist.png")

    summary_all = {"batch": summary_batch, "cellcycle": summary_cc}
    summary_path = args.output_dir / "confounding_summary.json"
    with open(summary_path, "w", encoding="utf-8") as h:
        json.dump(summary_all, h, indent=2)
    print(f"\nWrote {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
