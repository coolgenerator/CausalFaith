from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from causalfaith.crossfit import load_ground_truth_edge_sets, load_method_edges
from scripts.run_benchmark_metrics import (
    directed_universe,
    discover_full_edge_files,
    edge_scores,
    labels_for,
    load_fallback_manifest,
    method_label,
    read_gene_list,
)


REGIME_SUFFIXES = ["partial_interventional", "interventional", "observational"]


def split_method_regime(label: str) -> tuple[str, str]:
    for suffix in REGIME_SUFFIXES:
        suffix_text = f"_{suffix}"
        if label.endswith(suffix_text):
            return label[: -len(suffix_text)], suffix
    raise ValueError(f"Cannot split method/regime label: {label}")


def metric_value(labels: np.ndarray, scores: np.ndarray, metric: str) -> float:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return np.nan
    if metric == "auroc":
        return float(roc_auc_score(labels, scores))
    if metric == "auprc":
        return float(average_precision_score(labels, scores))
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_delta(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    comparison_scores: np.ndarray,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        base = metric_value(y, baseline_scores[idx], metric)
        comp = metric_value(y, comparison_scores[idx], metric)
        if np.isfinite(base) and np.isfinite(comp):
            deltas.append(comp - base)
    if not deltas:
        return {
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_delta_gt_0": np.nan,
            "n_bootstrap_used": 0,
        }
    arr = np.asarray(deltas, dtype=float)
    return {
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_delta_gt_0": float((arr > 0).mean()),
        "n_bootstrap_used": int(len(arr)),
    }


def precision_recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    kk = min(k, len(scores))
    if kk == 0:
        return np.nan, np.nan
    positives = int(labels.sum())
    order = np.argsort(-scores, kind="mergesort")[:kk]
    tp = int(labels[order].sum())
    return tp / kk, tp / positives if positives else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods-dir", required=True)
    parser.add_argument("--gt-dir", default="data/causalbench")
    parser.add_argument("--gene-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fallback-manifest", default=None)
    parser.add_argument("--string-min-score", type=int, default=700)
    parser.add_argument("--string-physical-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", nargs="+", type=int, default=[25, 50, 100])
    parser.add_argument("--include-fallback", action="store_true")
    args = parser.parse_args()

    methods_dir = Path(args.methods_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fallback_manifest = (
        Path(args.fallback_manifest)
        if args.fallback_manifest
        else methods_dir / "method_edge_fallbacks.csv"
    )
    fallback_by_label = load_fallback_manifest(fallback_manifest)

    genes = read_gene_list(Path(args.gene_list))
    universe = directed_universe(genes)
    edge_sets = load_ground_truth_edge_sets(
        Path(args.gt_dir),
        genes,
        ["chipAtlas", "corum", "stringdb"],
        string_min_score=args.string_min_score,
        string_physical_only=args.string_physical_only,
    )
    pooled = set()
    for edges in edge_sets.values():
        pooled.update(edges)
    source_sets = {"pooled": pooled, **edge_sets}

    scores_by_label = {}
    label_parts = {}
    for path in discover_full_edge_files(methods_dir):
        label = method_label(path)
        if label in fallback_by_label and not args.include_fallback:
            continue
        scores_by_label[label] = edge_scores(load_method_edges(path), universe)
        label_parts[label] = split_method_regime(label)

    by_method: dict[str, dict[str, str]] = {}
    for label, (method, regime) in label_parts.items():
        by_method.setdefault(method, {})[regime] = label

    uplift_rows = []
    topk_rows = []
    for source, gt_edges in source_sets.items():
        labels = labels_for(gt_edges, universe)
        for method, regimes in sorted(by_method.items()):
            if "observational" not in regimes:
                continue
            baseline_label = regimes["observational"]
            baseline_scores = scores_by_label[baseline_label]
            for regime in ["partial_interventional", "interventional"]:
                if regime not in regimes:
                    continue
                comparison_label = regimes[regime]
                comparison_scores = scores_by_label[comparison_label]
                for metric in ["auprc", "auroc"]:
                    baseline_value = metric_value(labels, baseline_scores, metric)
                    comparison_value = metric_value(labels, comparison_scores, metric)
                    delta = comparison_value - baseline_value
                    boot = bootstrap_delta(
                        labels,
                        baseline_scores,
                        comparison_scores,
                        metric,
                        args.n_bootstrap,
                        args.seed,
                    )
                    uplift_rows.append({
                        "source": source,
                        "method": method,
                        "comparison_regime": regime,
                        "baseline_regime": "observational",
                        "metric": metric,
                        "baseline_value": baseline_value,
                        "comparison_value": comparison_value,
                        "delta": delta,
                        **boot,
                        "baseline_label": baseline_label,
                        "comparison_label": comparison_label,
                        "comparison_is_fallback": comparison_label in fallback_by_label,
                        "fallback_source": fallback_by_label.get(comparison_label, {}).get(
                            "fallback_source", ""
                        ),
                    })

                for k in args.ks:
                    base_precision, base_recall = precision_recall_at_k(
                        labels, baseline_scores, k
                    )
                    comp_precision, comp_recall = precision_recall_at_k(
                        labels, comparison_scores, k
                    )
                    topk_rows.append({
                        "source": source,
                        "method": method,
                        "comparison_regime": regime,
                        "k": k,
                        "precision_baseline": base_precision,
                        "precision_comparison": comp_precision,
                        "delta_precision": comp_precision - base_precision,
                        "recall_baseline": base_recall,
                        "recall_comparison": comp_recall,
                        "delta_recall": comp_recall - base_recall,
                        "comparison_is_fallback": comparison_label in fallback_by_label,
                    })

    uplift = pd.DataFrame(uplift_rows).sort_values(
        ["source", "metric", "delta"],
        ascending=[True, True, False],
        na_position="last",
    )
    topk = pd.DataFrame(topk_rows).sort_values(
        ["source", "k", "delta_precision"],
        ascending=[True, True, False],
        na_position="last",
    )

    uplift_path = output_dir / "causal_uplift.csv"
    topk_path = output_dir / "causal_uplift_topk.csv"
    uplift.to_csv(uplift_path, index=False)
    topk.to_csv(topk_path, index=False)

    positive = uplift[uplift["delta"].gt(0)]
    summary = {
        "n_genes": len(genes),
        "n_universe_edges": int(len(universe)),
        "n_comparisons": int(len(uplift)),
        "n_positive_deltas": int(len(positive)),
        "n_positive_deltas_p_gt_0_ge_0_95": int(
            positive["p_delta_gt_0"].ge(0.95).sum()
        ) if not positive.empty else 0,
        "string_min_score": int(args.string_min_score),
        "string_physical_only": bool(args.string_physical_only),
        "include_fallback": bool(args.include_fallback),
        "outputs": {
            "uplift": str(uplift_path),
            "topk": str(topk_path),
        },
    }
    summary_path = output_dir / "causal_uplift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {uplift_path}")
    print(f"Wrote {topk_path}")
    print(f"Wrote {summary_path}")
    print(uplift.to_string(index=False))


if __name__ == "__main__":
    main()
