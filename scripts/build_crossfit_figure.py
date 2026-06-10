"""Reproduce the cross-fit per-edge join and visualize the Result-3 null finding.

Re-runs the cross-fitting from causalfaith.crossfit on the committed faithfulness
matrices + method edge folds + pooled ground truth, then writes:

- results/crossfit/per_edge_faithfulness_recall.csv : per-edge (f_score_mean,
  recall_mean) for methods with enough predicted edges (source file for Table 2).
- docs/report_assets/fig_crossfit_null.png : the Module C figure showing that
  higher faithfulness does NOT translate into higher recovery.

Ground truth dir defaults to data/causalbench (CORUM zip, STRING info/links;
ChIP-Atlas comes from the causalscbench package). STRING physical-only is off
because only the full protein.links file is present locally.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from causalfaith import crossfit as cf

ROOT = Path(__file__).resolve().parents[1]
FDIR = ROOT / "results" / "faithfulness"
MDIR = ROOT / "results" / "methods"
GTDIR = ROOT / "data" / "causalbench"
OUT_CSV = ROOT / "results" / "crossfit" / "per_edge_faithfulness_recall.csv"
OUT_FIG = ROOT / "docs" / "report_assets" / "fig_crossfit_null.png"

REGIME = "interventional"
METHODS = ["meandifference", "grnboost", "ges", "gies", "pc", "fci"]
COLORS = {"meandifference": "#2563eb", "grnboost": "#ea580c", "ges": "#16a34a",
          "gies": "#9333ea", "pc": "#0891b2", "fci": "#dc2626"}


def _edges(method: str, fold: str) -> pd.DataFrame:
    return cf.load_method_edges(MDIR / f"{method}_{REGIME}_{fold}_edges.csv")


def main():
    F_I1 = cf.load_f_matrix(FDIR / "F_raw_I1.npy")
    F_I2 = cf.load_f_matrix(FDIR / "F_raw_I2.npy")
    genes = [str(g) for g in F_I1.index]

    gt_sets = cf.load_ground_truth_edge_sets(
        GTDIR, genes, sources=["chipAtlas", "corum", "stringdb"],
        string_min_score=700, string_physical_only=False,
    )
    gt_pooled: set[tuple[str, str]] = set()
    for s in gt_sets.values():
        gt_pooled |= s
    print(f"genes={len(genes)}  gt sources={ {k: len(v) for k, v in gt_sets.items()} }  "
          f"pooled={len(gt_pooled)}")

    per_edge, rho_rows = {}, []
    for m in METHODS:
        try:
            df = cf.cross_fit(F_I1, _edges(m, "I2"), F_I2, _edges(m, "I1"), gt_pooled)
        except FileNotFoundError:
            continue
        valid = df.dropna(subset=["f_score_mean", "recall_mean"])
        n = len(valid)
        rho = (float(stats.spearmanr(valid["f_score_mean"], valid["recall_mean"]).correlation)
               if n >= 10 else np.nan)
        rho_rows.append({"method": m, "n_edges": n, "spearman_rho": rho})
        per_edge[m] = valid.assign(method=m)
        print(f"  {m:15s} n={n:6d}  rho={rho}")

    # committable per-edge source file
    pd.concat(per_edge.values(), ignore_index=True)[
        ["method", "source_gene", "target_gene", "f_score_mean", "recall_mean"]
    ].to_csv(OUT_CSV, index=False)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel A: recovery rate vs faithfulness decile, methods with many edges
    for m in ["meandifference", "grnboost"]:
        if m not in per_edge:
            continue
        d = per_edge[m]
        q = pd.qcut(d["f_score_mean"], 10, labels=False, duplicates="drop")
        g = d.groupby(q).agg(x=("f_score_mean", "median"),
                             y=("recall_mean", "mean")).reset_index(drop=True)
        rho = next(r["spearman_rho"] for r in rho_rows if r["method"] == m)
        axA.plot(g["x"], g["y"], "o-", color=COLORS[m],
                 label=f"{'GRNBoost' if m=='grnboost' else 'MeanDifference'} "
                       f"(ρ={rho:+.3f})")
    axA.set_xlabel("Faithfulness score (decile median)")
    axA.set_ylabel("Recovery rate\n(fraction of edges in pooled ground truth)")
    axA.set_title("Higher faithfulness does not raise recovery")
    axA.legend(frameon=False, fontsize=8)
    axA.grid(alpha=0.3)

    # Panel B: Spearman rho across all methods (forest)
    rho_df = pd.DataFrame(rho_rows).dropna(subset=["spearman_rho"])
    rho_df = rho_df.sort_values("spearman_rho")
    y = np.arange(len(rho_df))
    axB.scatter(rho_df["spearman_rho"], y, color=[COLORS[m] for m in rho_df["method"]],
                zorder=3)
    for yi, (_, r) in zip(y, rho_df.iterrows()):
        axB.text(r["spearman_rho"], yi + 0.18, f"  n={r['n_edges']}",
                 fontsize=7, va="bottom")
    axB.axvline(0, color="black", lw=0.8)
    axB.set_yticks(y)
    axB.set_yticklabels([{"meandifference": "MeanDiff", "grnboost": "GRNBoost"}
                         .get(m, m.upper()) for m in rho_df["method"]])
    axB.set_xlabel("Spearman ρ (faithfulness vs recovery)")
    axB.set_xlim(-0.2, 0.2)
    axB.set_title("All methods cluster near zero")
    axB.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("\nWrote:", OUT_CSV, OUT_FIG, sep="\n  ")


if __name__ == "__main__":
    main()
