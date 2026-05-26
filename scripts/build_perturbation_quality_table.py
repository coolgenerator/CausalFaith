from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIAG_PREFIX = "diag_"


def load_kd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={
        "n_cells": "n_cells_kd",
        "target_in_subset": "target_in_subset",
        "ctrl_norm_mean": f"{DIAG_PREFIX}kd_ctrl_norm_mean",
        "pert_norm_mean": f"{DIAG_PREFIX}kd_pert_norm_mean",
        "residual_ratio": f"{DIAG_PREFIX}kd_residual_ratio",
        "kd_stratum": f"{DIAG_PREFIX}kd_stratum",
    })


def load_batch(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={
        "n_cells": "n_cells_batch",
        "batch_js_divergence": f"{DIAG_PREFIX}batch_js_divergence",
        "n_batches_covered": f"{DIAG_PREFIX}batch_n_batches_covered",
        "diagnostic_status": f"{DIAG_PREFIX}batch_status",
    })


def load_cellcycle(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={
        "n_cells": "n_cells_cc",
        "phase_chi2_pvalue": f"{DIAG_PREFIX}cellcycle_phase_chi2_p",
        "s_score_diff": f"{DIAG_PREFIX}cellcycle_s_score_diff",
        "g2m_score_diff": f"{DIAG_PREFIX}cellcycle_g2m_score_diff",
        "diagnostic_status": f"{DIAG_PREFIX}cellcycle_status",
    })


def add_batch_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Stratify perturbations into low / medium / high batch confounding,
    using tertiles of the observed JS divergence distribution.
    """
    col = f"{DIAG_PREFIX}batch_js_divergence"
    stratum_col = f"{DIAG_PREFIX}batch_stratum"
    valid = df[col].dropna()
    if len(valid) == 0:
        df[stratum_col] = "undefined"
        return df

    q33 = float(np.quantile(valid, 1 / 3))
    q67 = float(np.quantile(valid, 2 / 3))

    def classify(x):
        if not np.isfinite(x):
            return "undefined"
        if x < q33:
            return "low"
        if x < q67:
            return "medium"
        return "high"

    df[stratum_col] = df[col].apply(classify)
    return df


def add_cellcycle_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Flag perturbations with Bonferroni-significant phase shift."""
    p_col = f"{DIAG_PREFIX}cellcycle_phase_chi2_p"
    flag_col = f"{DIAG_PREFIX}cellcycle_phase_shifted"
    valid = df[p_col].dropna()
    bonferroni = 0.05 / max(len(valid), 1)
    df[flag_col] = df[p_col].apply(
        lambda x: True if np.isfinite(x) and x < bonferroni else False
    )
    return df


def assign_overall_status(df: pd.DataFrame) -> pd.DataFrame:
    """A coarse status combining KD and confounding info."""
    kd_status_col = f"{DIAG_PREFIX}kd_stratum"
    batch_status_col = f"{DIAG_PREFIX}batch_status"

    def classify(row):
        kd_ok = row.get(kd_status_col) in ["strong", "medium", "weak"]
        batch_ok = row.get(batch_status_col) == "ok"
        if kd_ok and batch_ok:
            return "full"
        if kd_ok and not batch_ok:
            return "kd_only"
        if not kd_ok and batch_ok:
            return "kd_unmeasurable"
        return "insufficient"

    df[f"{DIAG_PREFIX}status"] = df.apply(classify, axis=1)
    return df


def plot_kd_vs_batch(df: pd.DataFrame, out_path: Path) -> None:
    sub = df.dropna(subset=[
        f"{DIAG_PREFIX}kd_residual_ratio",
        f"{DIAG_PREFIX}batch_js_divergence",
    ])
    if sub.empty:
        return
    color_map = {"strong": "tab:green", "medium": "tab:orange",
                 "weak": "tab:red", "undefined": "tab:grey"}
    colors = sub[f"{DIAG_PREFIX}kd_stratum"].map(color_map).fillna("tab:grey")
    plt.figure(figsize=(8, 6))
    plt.scatter(
        sub[f"{DIAG_PREFIX}kd_residual_ratio"],
        sub[f"{DIAG_PREFIX}batch_js_divergence"],
        c=colors, alpha=0.6, s=24,
    )
    for label, color in color_map.items():
        plt.scatter([], [], c=color, label=label)  # legend handles only
    plt.xlabel("KD residual ratio")
    plt.ylabel("Batch JS divergence vs control")
    plt.title("KD efficiency vs batch confounding (per perturbation)")
    plt.legend(title="KD stratum")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_kd_vs_cellcycle(df: pd.DataFrame, out_path: Path) -> None:
    sub = df.dropna(subset=[
        f"{DIAG_PREFIX}kd_residual_ratio",
        f"{DIAG_PREFIX}cellcycle_phase_chi2_p",
    ]).copy()
    if sub.empty:
        return
    sub["neg_log10_p"] = -np.log10(sub[f"{DIAG_PREFIX}cellcycle_phase_chi2_p"].clip(lower=1e-300))
    plt.figure(figsize=(8, 6))
    plt.scatter(
        sub[f"{DIAG_PREFIX}kd_residual_ratio"],
        sub["neg_log10_p"],
        alpha=0.6, s=24,
    )
    bonferroni = 0.05 / max(len(sub), 1)
    plt.axhline(-np.log10(bonferroni), color="red", linestyle="--",
                label=f"Bonferroni 0.05 / n")
    plt.xlabel("KD residual ratio")
    plt.ylabel("-log10(p) for phase shift")
    plt.title("KD efficiency vs cell-cycle disruption (per perturbation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-dir", type=Path, required=True,
                        help="Directory with kd_efficiency.csv, batch_divergence.csv, cellcycle_bias.csv")
    args = parser.parse_args()

    d = args.diagnostics_dir
    kd_path = d / "kd_efficiency.csv"
    batch_path = d / "batch_divergence.csv"
    cc_path = d / "cellcycle_bias.csv"

    for p in [kd_path, batch_path, cc_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    print("Loading three diagnostic tables...", flush=True)
    df_kd = load_kd(kd_path)
    df_batch = load_batch(batch_path)
    df_cc = load_cellcycle(cc_path)
    print(f"  KD: {df_kd.shape}, batch: {df_batch.shape}, cc: {df_cc.shape}", flush=True)

    print("Outer-joining on perturbation...", flush=True)
    df = df_kd.merge(df_batch, on="perturbation", how="outer")
    df = df.merge(df_cc, on="perturbation", how="outer")

    # consolidate cell count columns (they should all agree)
    n_cols = [c for c in ["n_cells_kd", "n_cells_batch", "n_cells_cc"] if c in df.columns]
    df["n_cells"] = df[n_cols].bfill(axis=1).iloc[:, 0]
    df = df.drop(columns=n_cols)

    print("Adding derived strata and status...", flush=True)
    df = add_batch_strata(df)
    df = add_cellcycle_strata(df)
    df = assign_overall_status(df)

    # reorder columns: identity → counts → diag_*
    id_cols = ["perturbation", "n_cells", "target_in_subset"]
    diag_cols = sorted([c for c in df.columns if c.startswith(DIAG_PREFIX)])
    other_cols = [c for c in df.columns if c not in id_cols + diag_cols]
    df = df[id_cols + diag_cols + other_cols]

    out_path = d / "perturbation_quality_table.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  shape={df.shape}", flush=True)

    print("\nStatus counts:")
    print(df[f"{DIAG_PREFIX}status"].value_counts())

    print("\nKD stratum × batch stratum cross-tab:")
    print(pd.crosstab(
        df[f"{DIAG_PREFIX}kd_stratum"].fillna("missing"),
        df[f"{DIAG_PREFIX}batch_stratum"].fillna("missing"),
    ))

    plot_dir = d / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_kd_vs_batch(df, plot_dir / "joint_kd_vs_batch.png")
    plot_kd_vs_cellcycle(df, plot_dir / "joint_kd_vs_cellcycle.png")
    print(f"Wrote joint plots to {plot_dir}/")

    # compact final summary for member 5 / final report
    summary = {
        "n_perturbations": int(len(df)),
        "status_counts": df[f"{DIAG_PREFIX}status"].value_counts().to_dict(),
        "kd_stratum_counts": df[f"{DIAG_PREFIX}kd_stratum"].value_counts().to_dict(),
        "batch_stratum_counts": df[f"{DIAG_PREFIX}batch_stratum"].value_counts().to_dict(),
        "n_cellcycle_phase_shifted": int(df[f"{DIAG_PREFIX}cellcycle_phase_shifted"].sum()),
    }
    summary_path = d / "perturbation_quality_summary.json"
    with open(summary_path, "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)
    print(f"\nWrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()