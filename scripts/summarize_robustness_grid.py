from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_uplift_rows(subsets_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(subsets_dir.glob("subset_*/alpha*/metrics/causal_uplift.csv")):
        subset_id = path.parents[2].name
        config = path.parents[1].name
        df = pd.read_csv(path)
        df.insert(0, "config", config)
        df.insert(0, "subset_id", subset_id)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_topk_rows(subsets_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(subsets_dir.glob("subset_*/alpha*/metrics/causal_uplift_topk.csv")):
        subset_id = path.parents[2].name
        config = path.parents[1].name
        df = pd.read_csv(path)
        df.insert(0, "config", config)
        df.insert(0, "subset_id", subset_id)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_uplift(uplift: pd.DataFrame) -> pd.DataFrame:
    if uplift.empty:
        return pd.DataFrame()
    group_cols = ["source", "method", "comparison_regime", "metric"]
    return (
        uplift
        .groupby(group_cols, as_index=False)
        .agg(
            n_runs=("delta", "size"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            sd_delta=("delta", "std"),
            positive_fraction=("delta", lambda s: float((s > 0).mean())),
            strong_positive_fraction=("p_delta_gt_0", lambda s: float((s >= 0.95).mean())),
            mean_baseline=("baseline_value", "mean"),
            mean_comparison=("comparison_value", "mean"),
        )
        .sort_values(["source", "metric", "mean_delta"], ascending=[True, True, False])
    )


def summarize_topk(topk: pd.DataFrame) -> pd.DataFrame:
    if topk.empty:
        return pd.DataFrame()
    group_cols = ["source", "method", "comparison_regime", "k"]
    return (
        topk
        .groupby(group_cols, as_index=False)
        .agg(
            n_runs=("delta_precision", "size"),
            mean_delta_precision=("delta_precision", "mean"),
            median_delta_precision=("delta_precision", "median"),
            positive_fraction=("delta_precision", lambda s: float((s > 0).mean())),
            mean_precision_baseline=("precision_baseline", "mean"),
            mean_precision_comparison=("precision_comparison", "mean"),
        )
        .sort_values(["source", "k", "mean_delta_precision"], ascending=[True, True, False])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets-dir", default="results/robustness_50_hc")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    subsets_dir = Path(args.subsets_dir)
    output_dir = Path(args.output_dir) if args.output_dir else subsets_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    uplift = load_uplift_rows(subsets_dir)
    topk = load_topk_rows(subsets_dir)
    uplift_summary = summarize_uplift(uplift)
    topk_summary = summarize_topk(topk)

    uplift.to_csv(output_dir / "robustness_uplift_all.csv", index=False)
    topk.to_csv(output_dir / "robustness_topk_all.csv", index=False)
    uplift_summary.to_csv(output_dir / "robustness_uplift_summary.csv", index=False)
    topk_summary.to_csv(output_dir / "robustness_topk_summary.csv", index=False)

    summary = {
        "n_uplift_rows": int(len(uplift)),
        "n_topk_rows": int(len(topk)),
        "n_subsets": int(uplift["subset_id"].nunique()) if not uplift.empty else 0,
        "n_configs": int(uplift["config"].nunique()) if not uplift.empty else 0,
        "outputs": {
            "uplift_all": str(output_dir / "robustness_uplift_all.csv"),
            "topk_all": str(output_dir / "robustness_topk_all.csv"),
            "uplift_summary": str(output_dir / "robustness_uplift_summary.csv"),
            "topk_summary": str(output_dir / "robustness_topk_summary.csv"),
        },
    }
    (output_dir / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not uplift_summary.empty:
        print("\nUplift summary:")
        print(uplift_summary.to_string(index=False))


if __name__ == "__main__":
    main()
