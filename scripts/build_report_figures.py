"""Generate synthesis figures for the project summary report.

Reads committed result tables and writes headline charts into
docs/report_assets/. Pure read-only on results; safe to re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

METHOD_COLORS = {"grnboost": "#2563eb", "fci": "#16a34a", "pc": "#ea580c"}
SOURCE_ORDER = ["corum", "stringdb", "pooled", "chipAtlas"]
SOURCE_LABEL = {"corum": "CORUM", "stringdb": "STRING", "pooled": "Pooled",
                "chipAtlas": "ChIP-Atlas"}


def fig_auprc_uplift():
    df = pd.read_csv(ROOT / "results/robustness_50_hc/summary/robustness_uplift_summary.csv")
    d = df[df["metric"] == "auprc"].copy()
    methods = ["grnboost", "fci", "pc"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(SOURCE_ORDER))
    w = 0.26
    for i, m in enumerate(methods):
        vals, errs = [], []
        for s in SOURCE_ORDER:
            row = d[(d["method"] == m) & (d["source"] == s)]
            vals.append(float(row["mean_delta"].iloc[0]) if len(row) else 0.0)
            errs.append(float(row["sd_delta"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=3,
               label=m.upper() if m != "grnboost" else "GRNBoost",
               color=METHOD_COLORS[m])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABEL[s] for s in SOURCE_ORDER])
    ax.set_ylabel("Mean AUPRC uplift\n(interventional − observational)")
    ax.set_title("Interventional data improves AUPRC across biological references\n"
                 "3 subsets × 3 configs (n=9 runs per bar); error bars = SD")
    ax.legend(title="Method", frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_auprc_uplift.png", bbox_inches="tight")
    plt.close(fig)


def fig_precision_uplift():
    df = pd.read_csv(ROOT / "results/robustness_50_hc/summary/robustness_topk_summary.csv")
    d = df[df["k"] == 50].copy()
    methods = ["grnboost", "fci", "pc"]
    srcs = ["corum", "stringdb", "pooled"]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(srcs))
    w = 0.26
    for i, m in enumerate(methods):
        vals = []
        for s in srcs:
            row = d[(d["method"] == m) & (d["source"] == s)]
            vals.append(float(row["mean_delta_precision"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + (i - 1) * w, vals, w,
               label=m.upper() if m != "grnboost" else "GRNBoost",
               color=METHOD_COLORS[m])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABEL[s] for s in srcs])
    ax.set_ylabel("Mean Precision@50 uplift")
    ax.set_title("Top-k precision also improves under interventional data (k=50)")
    ax.legend(title="Method", frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_precision_uplift.png", bbox_inches="tight")
    plt.close(fig)


def fig_kd_strata():
    with open(ROOT / "results/diagnostics/kd_efficiency_summary.json") as fh:
        s = json.load(fh)
    sc = s["stratum_counts"]
    labels = ["strong\n(<0.20)", "medium\n(0.20–0.50)", "weak\n(≥0.50)"]
    vals = [sc["strong"], sc["medium"], sc["weak"]]
    colors = ["#16a34a", "#eab308", "#dc2626"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    bars = ax.bar(labels, vals, color=colors)
    total = sum(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}\n{v/total*100:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of perturbations")
    ax.set_title("Knockdown efficiency strata\n(300 KD-measurable perturbations)")
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_kd_strata.png", bbox_inches="tight")
    plt.close(fig)


def fig_faithfulness_dist():
    with open(ROOT / "results/faithfulness/faithfulness_summary.json") as fh:
        fs = json.load(fh)
    keys = ["raw_I1", "raw_I2", "residualized_I1", "residualized_I2"]
    stats = ["q25", "median", "q75", "q90", "q95"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(stats))
    w = 0.2
    palette = ["#2563eb", "#60a5fa", "#16a34a", "#86efac"]
    for i, k in enumerate(keys):
        vals = [fs[k][st] for st in stats]
        ax.bar(x + (i - 1.5) * w, vals, w, label=k, color=palette[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Q25", "Median", "Q75", "Q90", "Q95"])
    ax.set_ylabel("Wasserstein-1 faithfulness score")
    ax.set_title("Faithfulness score distribution is stable across folds and\n"
                 "raw vs covariate-residualized expression (Spearman ρ≈0.96)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_faithfulness_dist.png", bbox_inches="tight")
    plt.close(fig)


def fig_positive_runs_heatmap():
    df = pd.read_csv(ROOT / "results/robustness_50_hc/summary/robustness_uplift_summary.csv")
    d = df[df["metric"] == "auprc"].copy()
    methods = ["grnboost", "fci", "pc"]
    mat = np.zeros((len(methods), len(SOURCE_ORDER)))
    for i, m in enumerate(methods):
        for j, s in enumerate(SOURCE_ORDER):
            row = d[(d["method"] == m) & (d["source"] == s)]
            mat[i, j] = float(row["strong_positive_fraction"].iloc[0]) if len(row) else np.nan
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SOURCE_ORDER)))
    ax.set_xticklabels([SOURCE_LABEL[s] for s in SOURCE_ORDER])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(["GRNBoost", "FCI", "PC"])
    for i in range(len(methods)):
        for j in range(len(SOURCE_ORDER)):
            ax.text(j, i, f"{mat[i,j]*100:.0f}%", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_title("Fraction of runs with strong-positive uplift\nP(Δ AUPRC > 0) ≥ 0.95")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_strong_positive_heatmap.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_auprc_uplift()
    fig_precision_uplift()
    fig_kd_strata()
    fig_faithfulness_dist()
    fig_positive_runs_heatmap()
    print("Wrote report figures to", ASSETS)
