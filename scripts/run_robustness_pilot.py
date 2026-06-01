from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


CONFIGS = {
    "alpha005_trees50": {"alpha": "0.005", "n_estimators": "50"},
    "alpha010_trees100": {"alpha": "0.01", "n_estimators": "100"},
    "alpha020_trees200": {"alpha": "0.02", "n_estimators": "200"},
}


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets-dir", default="results/robustness_50_hc")
    parser.add_argument("--processed-dir", default="data/processed/k562_essential")
    parser.add_argument("--gt-dir", default="data/causalbench")
    parser.add_argument("--methods", nargs="+", default=["pc", "fci", "grnboost"])
    parser.add_argument("--regimes", nargs="+", default=["observational", "interventional"])
    parser.add_argument("--max-obs-cells", type=int, default=500)
    parser.add_argument("--max-all-cells", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subsets_dir = Path(args.subsets_dir)
    manifest = pd.read_csv(subsets_dir / "subsets_manifest.csv")
    py = sys.executable

    for subset in manifest.itertuples(index=False):
        subset_dir = subsets_dir / subset.subset_id
        gene_list = subset_dir / "gene_list.txt"
        for config_name, config in CONFIGS.items():
            run_dir = subset_dir / config_name
            methods_dir = run_dir / "methods"
            metrics_dir = run_dir / "metrics"
            run([
                py,
                "scripts/run_methods.py",
                "--processed-dir", args.processed_dir,
                "--output-dir", str(methods_dir),
                "--gene-list", str(gene_list),
                "--methods", *args.methods,
                "--regimes", *args.regimes,
                "--max-obs-cells", str(args.max_obs_cells),
                "--max-all-cells", str(args.max_all_cells),
                "--alpha", config["alpha"],
                "--n-estimators", config["n_estimators"],
                "--n-jobs", str(args.n_jobs),
            ], args.dry_run)
            run([
                py,
                "scripts/run_benchmark_metrics.py",
                "--methods-dir", str(methods_dir),
                "--gt-dir", args.gt_dir,
                "--gene-list", str(gene_list),
                "--output-dir", str(metrics_dir),
                "--string-min-score", "700",
                "--string-physical-only",
                "--ks", "10", "25", "50", "100", "200",
            ], args.dry_run)
            run([
                py,
                "scripts/run_causal_uplift.py",
                "--methods-dir", str(methods_dir),
                "--gt-dir", args.gt_dir,
                "--gene-list", str(gene_list),
                "--output-dir", str(metrics_dir),
                "--string-min-score", "700",
                "--string-physical-only",
                "--n-bootstrap", str(args.n_bootstrap),
                "--ks", "25", "50", "100",
            ], args.dry_run)

    print("\nRobustness pilot complete.")


if __name__ == "__main__":
    main()
