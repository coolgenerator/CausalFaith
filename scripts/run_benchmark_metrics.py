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


def read_gene_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        gene_col = next(
            (c for c in ["gene", "gene_symbol", "symbol", "perturbation"] if c in df.columns),
            df.columns[0],
        )
        return df[gene_col].dropna().astype(str).tolist()
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def discover_full_edge_files(methods_dir: Path) -> list[Path]:
    files = []
    for path in sorted(methods_dir.glob("*_edges.csv")):
        if path.name.endswith("_I1_edges.csv") or path.name.endswith("_I2_edges.csv"):
            continue
        files.append(path)
    return files


def method_label(path: Path) -> str:
    return path.name.removesuffix("_edges.csv")


def load_fallback_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    manifest = pd.read_csv(path)
    if manifest.empty:
        return {}
    if "fold" in manifest.columns:
        manifest = manifest[manifest["fold"].eq("full")]
    return {
        str(row.target_label): {
            "fallback_source": str(row.source_label),
            "fallback_reason": str(row.reason),
        }
        for row in manifest.itertuples(index=False)
    }


def directed_universe(genes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source_gene": source, "target_gene": target}
            for source in genes
            for target in genes
            if source != target
        ]
    )


def edge_scores(edges: pd.DataFrame, universe: pd.DataFrame) -> np.ndarray:
    filtered = edges[
        edges["source_gene"].isin(set(universe["source_gene"]))
        & edges["target_gene"].isin(set(universe["target_gene"]))
    ].copy()
    if filtered.empty:
        return np.zeros(len(universe), dtype=float)
    score_map = (
        filtered
        .groupby(["source_gene", "target_gene"])["score"]
        .max()
        .to_dict()
    )
    return np.array(
        [
            float(score_map.get((row.source_gene, row.target_gene), 0.0))
            for row in universe.itertuples(index=False)
        ],
        dtype=float,
    )


def labels_for(gt_edges: set[tuple[str, str]], universe: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            int((row.source_gene, row.target_gene) in gt_edges)
            for row in universe.itertuples(index=False)
        ],
        dtype=int,
    )


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return np.nan, np.nan
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def topk_rows(
    method_regime: str,
    source: str,
    labels: np.ndarray,
    scores: np.ndarray,
    ks: list[int],
    fallback_info: dict[str, str] | None = None,
) -> list[dict]:
    order = np.argsort(-scores, kind="mergesort")
    positives = int(labels.sum())
    rows = []
    for k in ks:
        kk = min(k, len(scores))
        top = order[:kk]
        tp = int(labels[top].sum())
        rows.append({
            "method_regime": method_regime,
            "source": source,
            "k": kk,
            "true_positives_at_k": tp,
            "precision_at_k": tp / kk if kk else np.nan,
            "recall_at_k": tp / positives if positives else np.nan,
            "n_positives": positives,
            "is_fallback": bool(fallback_info),
            "fallback_source": fallback_info.get("fallback_source") if fallback_info else "",
            "fallback_reason": fallback_info.get("fallback_reason") if fallback_info else "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods-dir", required=True)
    parser.add_argument("--gt-dir", default="data/causalbench")
    parser.add_argument("--gene-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fallback-manifest", default=None,
                        help="Optional method_edge_fallbacks.csv for provenance annotations")
    parser.add_argument("--string-min-score", type=int, default=700,
                        help="Minimum STRING score to include in ground truth labels")
    parser.add_argument("--string-physical-only", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use STRING physical links only for ground truth labels")
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 25, 50, 100, 200])
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

    metric_rows = []
    topk_metric_rows = []
    for path in discover_full_edge_files(methods_dir):
        label = method_label(path)
        fallback_info = fallback_by_label.get(label, {})
        edges = load_method_edges(path)
        scores = edge_scores(edges, universe)
        predicted_nonzero = int((scores > 0).sum())
        for source, gt_edges in source_sets.items():
            labels = labels_for(gt_edges, universe)
            auroc, auprc = safe_auc(labels, scores)
            metric_rows.append({
                "method_regime": label,
                "source": source,
                "n_universe_edges": len(universe),
                "n_positives": int(labels.sum()),
                "n_predicted_nonzero": predicted_nonzero,
                "auroc": auroc,
                "auprc": auprc,
                "is_fallback": bool(fallback_info),
                "fallback_source": fallback_info.get("fallback_source", ""),
                "fallback_reason": fallback_info.get("fallback_reason", ""),
            })
            topk_metric_rows.extend(
                topk_rows(label, source, labels, scores, args.ks, fallback_info)
            )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["source", "auprc", "auroc"],
        ascending=[True, False, False],
        na_position="last",
    )
    topk = pd.DataFrame(topk_metric_rows)

    metrics_path = output_dir / "benchmark_metrics.csv"
    topk_path = output_dir / "benchmark_topk.csv"
    metrics.to_csv(metrics_path, index=False)
    topk.to_csv(topk_path, index=False)

    summary = {
        "n_genes": len(genes),
        "n_universe_edges": int(len(universe)),
        "n_method_regimes": int(metrics["method_regime"].nunique()) if not metrics.empty else 0,
        "ground_truth_sources": {
            source: int(len(edges)) for source, edges in source_sets.items()
        },
        "string_min_score": int(args.string_min_score),
        "string_physical_only": bool(args.string_physical_only),
        "fallback_method_regimes": fallback_by_label,
        "outputs": {
            "metrics": str(metrics_path),
            "topk": str(topk_path),
        },
    }
    summary_path = output_dir / "benchmark_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {metrics_path}")
    print(f"Wrote {topk_path}")
    print(f"Wrote {summary_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
