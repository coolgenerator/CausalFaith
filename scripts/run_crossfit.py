"""
Module C: Cross-fitting analysis and final stratified comparisons.

Usage (once member 4's edge lists are available):

    python scripts/run_crossfit.py \
        --faithfulness-dir  results/faithfulness \
        --methods-dir       results/methods \
        --diagnostics-dir   results/diagnostics \
        --gt-dir            data/causalbench/evaluation_resources \
        --processed-dir     data/processed/k562_essential \
        --output-dir        results/crossfit

Cross-fitting mode requires fold-specific edge files:
    results/methods/{method}_{regime}_I1_edges.csv
    results/methods/{method}_{regime}_I2_edges.csv

If only full-dataset edge files exist ({method}_{regime}_edges.csv),
the script falls back to a single-split analysis (no averaging).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from causalfaith.crossfit import (
    CrossFitConfig,
    compute_correlation_table,
    cross_fit,
    cross_fit_one_split,
    load_f_matrix,
    load_fold_assignment,
    load_ground_truth_edges,
    load_method_edges,
    load_perturbation_quality,
    run_stratified_analysis,
    sensitivity_gene_subset,
    sensitivity_raw_vs_residualized,
)


def discover_method_edge_files(
    methods_dir: Path,
    methods: list[str],
    regimes: list[str],
) -> dict[str, dict]:
    """
    Scan methods_dir for edge CSVs.

    Returns dict mapping label → {"full": path, "I1": path|None, "I2": path|None}
    """
    found = {}
    for method in methods:
        for regime in regimes:
            label = f"{method}_{regime}"
            entry = {"full": None, "I1": None, "I2": None}

            for fold in ["I1", "I2", ""]:
                suffix = f"_{fold}" if fold else ""
                candidate = methods_dir / f"{method}_{regime}{suffix}_edges.csv"
                if candidate.exists():
                    if fold:
                        entry[fold] = candidate
                    else:
                        entry["full"] = candidate

            if any(v is not None for v in entry.values()):
                found[label] = entry

    return found


def load_gene_metadata(processed_dir: Path) -> pd.DataFrame | None:
    path = processed_dir / "gene_subset_300.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # index by gene symbol
    gene_col = next((c for c in ["gene", "gene_symbol", "symbol"] if c in df.columns), None)
    if gene_col:
        df = df.set_index(gene_col)
    return df


def run_analysis(args: argparse.Namespace) -> None:
    faith_dir = Path(args.faithfulness_dir)
    methods_dir = Path(args.methods_dir)
    diag_dir = Path(args.diagnostics_dir)
    gt_dir = Path(args.gt_dir)
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = CrossFitConfig(results_dir=out_dir)

    # ── load faithfulness matrices ────────────────────────────────────────────
    print("Loading faithfulness matrices...")
    f_versions: dict[str, dict[str, pd.DataFrame | None]] = {}
    for version in ["raw", "residualized"]:
        f_versions[version] = {}
        for fold in ["I1", "I2"]:
            p = faith_dir / f"F_{version}_{fold}.csv"
            if p.exists():
                f_versions[version][fold] = load_f_matrix(p)
                print(f"  Loaded {p.name}: {f_versions[version][fold].shape}")
            else:
                f_versions[version][fold] = None
                print(f"  Missing: {p.name}")

    # pick the primary F matrix for stratified analysis (raw, I1)
    F_primary = f_versions["raw"].get("I1") or f_versions["residualized"].get("I1")
    if F_primary is None:
        print("ERROR: No faithfulness matrix found. Run scripts/run_faithfulness.py first.")
        sys.exit(1)

    # ── load supporting data ──────────────────────────────────────────────────
    print("Loading perturbation quality table...")
    quality_path = diag_dir / "perturbation_quality_table.csv"
    if not quality_path.exists():
        print(f"ERROR: Missing {quality_path}")
        sys.exit(1)
    quality_table = load_perturbation_quality(quality_path)
    print(f"  {len(quality_table)} perturbations loaded")

    gene_metadata = load_gene_metadata(processed_dir)

    matched_subsets_path = processed_dir / "matched_random_subsets.csv"
    matched_subsets = pd.read_csv(matched_subsets_path) if matched_subsets_path.exists() else None

    # ── load ground truth ─────────────────────────────────────────────────────
    print("Loading ground truth edges...")
    genes = list(F_primary.index)
    try:
        gt_edges = load_ground_truth_edges(gt_dir, genes, config.gt_sources)
        print(f"  {len(gt_edges)} ground truth edges in 300-gene subset")
    except FileNotFoundError as exc:
        print(f"WARNING: {exc}")
        print("Skipping correlation and recall analyses; stratified analysis will be skipped.")
        gt_edges = None

    # ── discover method edge files ────────────────────────────────────────────
    print("Scanning for method edge files...")
    edge_files = discover_method_edge_files(methods_dir, config.methods, config.regimes)
    if not edge_files:
        print(
            f"No method edge files found in {methods_dir}.\n"
            "Expected: {{method}}_{{regime}}_edges.csv  (full) or\n"
            "          {{method}}_{{regime}}_I1_edges.csv / _I2_edges.csv  (fold-split)\n"
            "Continuing with sensitivity analysis only."
        )
    else:
        print(f"  Found edge files for: {list(edge_files.keys())}")

    # ── cross-fitting correlation table ───────────────────────────────────────
    if gt_edges is not None and edge_files:
        print("\nRunning cross-fitting correlation analysis...")
        method_results: dict[str, pd.DataFrame] = {}

        for label, paths in edge_files.items():
            print(f"  {label}")
            try:
                F_raw_I1 = f_versions["raw"]["I1"]
                F_raw_I2 = f_versions["raw"]["I2"]

                if paths["I1"] and paths["I2"] and F_raw_I1 is not None and F_raw_I2 is not None:
                    # proper cross-fitting
                    edges_I1 = load_method_edges(paths["I1"])
                    edges_I2 = load_method_edges(paths["I2"])
                    result = cross_fit(F_raw_I1, edges_I2, F_raw_I2, edges_I1,
                                       gt_edges, gene_metadata)
                    result["cross_fitted"] = True

                elif paths["full"] and F_raw_I1 is not None:
                    # fallback: single split (faithfulness on I1, methods on full data)
                    print(f"    (single-split fallback — no fold-specific edge files)")
                    edges_full = load_method_edges(paths["full"])
                    result = cross_fit_one_split(F_raw_I1, edges_full, gt_edges, gene_metadata)
                    result = result.rename(columns={"f_score": "f_score_mean",
                                                     "in_ground_truth": "recall_mean"})
                    result["cross_fitted"] = False

                else:
                    print(f"    skipped (no usable edge files)")
                    continue

                method_results[label] = result

            except Exception as exc:
                print(f"    ERROR: {exc}")
                continue

        if method_results:
            corr_table = compute_correlation_table(method_results)
            corr_path = out_dir / "correlation_table.csv"
            corr_table.to_csv(corr_path, index=False)
            print(f"\nWrote {corr_path}")
            print(corr_table.to_string(index=False))

    # ── stratified analysis ───────────────────────────────────────────────────
    if gt_edges is not None and edge_files:
        print("\nRunning stratified analysis...")

        # use full-dataset edges where available, else I2 as proxy
        method_edges_for_strat: dict[str, pd.DataFrame] = {}
        for label, paths in edge_files.items():
            p = paths["full"] or paths["I2"] or paths["I1"]
            if p:
                try:
                    method_edges_for_strat[label] = load_method_edges(p)
                except Exception as exc:
                    print(f"  Skipping {label}: {exc}")

        strat_df = run_stratified_analysis(
            method_edges_for_strat, gt_edges, quality_table, F_primary, config
        )
        strat_path = out_dir / "stratified_rankings.csv"
        strat_df.to_csv(strat_path, index=False)
        print(f"Wrote {strat_path} ({len(strat_df)} rows)")

    # ── sensitivity: raw vs residualized ─────────────────────────────────────
    print("\nRunning sensitivity analysis: raw vs residualized F...")
    sensitivity_rows = []
    for fold in ["I1", "I2"]:
        F_raw = f_versions["raw"][fold]
        F_resid = f_versions["residualized"][fold]
        if F_raw is not None and F_resid is not None:
            result = sensitivity_raw_vs_residualized(F_raw, F_resid)
            result["fold"] = fold
            sensitivity_rows.append(result)
            print(f"  {fold}: rho={result['rho']:.3f}, mean_rank_change={result['mean_rank_change']:.3f}")

    if sensitivity_rows:
        sens_path = out_dir / "sensitivity_raw_vs_resid.json"
        with open(sens_path, "w") as fh:
            json.dump(sensitivity_rows, fh, indent=2)
        print(f"Wrote {sens_path}")

    # ── sensitivity: 300-gene vs random subsets ───────────────────────────────
    if matched_subsets is not None:
        print("\nRunning sensitivity analysis: 300-gene vs random subsets...")
        subset_df = sensitivity_gene_subset(F_primary, matched_subsets)
        subset_path = out_dir / "sensitivity_gene_subsets.csv"
        subset_df.to_csv(subset_path, index=False)
        print(f"Wrote {subset_path}")
        print(subset_df.to_string(index=False))

    # ── write run summary ─────────────────────────────────────────────────────
    summary = {
        "faithfulness_matrices_loaded": {
            v: {fold: (f_versions[v][fold] is not None) for fold in ["I1", "I2"]}
            for v in ["raw", "residualized"]
        },
        "n_method_edge_files": len(edge_files),
        "n_gt_edges": len(gt_edges) if gt_edges else 0,
        "output_dir": str(out_dir),
    }
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nDone. Results in {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--faithfulness-dir", default="results/faithfulness",
                        help="Directory with F_raw_I1.csv etc. (default: results/faithfulness)")
    parser.add_argument("--methods-dir", default="results/methods",
                        help="Directory with {method}_{regime}_edges.csv files")
    parser.add_argument("--diagnostics-dir", default="results/diagnostics",
                        help="Directory with perturbation_quality_table.csv (member 3 output)")
    parser.add_argument("--gt-dir", required=True,
                        help="Directory with ground truth edge CSVs "
                             "(ChIP-Atlas, CORUM, StringDB)")
    parser.add_argument("--processed-dir", default="data/processed/k562_essential",
                        help="Processed data directory (gene_subset_300.csv etc.)")
    parser.add_argument("--output-dir", default="results/crossfit",
                        help="Output directory for correlation_table.csv etc.")
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
