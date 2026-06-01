from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from causalfaith.crossfit import load_ground_truth_edge_sets


FAMILY_PREFIXES = [
    "RPL",
    "RPS",
    "PSMA",
    "PSMB",
    "PSMC",
    "PSMD",
    "EIF",
    "CCT",
]


def pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    return values.rank(pct=True, ascending=ascending).fillna(0.0)


def source_degrees(edge_sets: dict[str, set[tuple[str, str]]], genes: list[str]) -> pd.DataFrame:
    rows = []
    for gene in genes:
        row = {"gene": gene}
        pooled = set()
        for source, edges in edge_sets.items():
            degree = sum(1 for src, tgt in edges if src == gene or tgt == gene)
            row[f"gt_degree_{source}"] = degree
            pooled.update(edges)
        row["gt_degree_pooled"] = sum(1 for src, tgt in pooled if src == gene or tgt == gene)
        rows.append(row)
    return pd.DataFrame(rows)


def family_key(gene: str) -> str:
    for prefix in FAMILY_PREFIXES:
        if gene.startswith(prefix):
            return prefix
    match = re.match(r"^[A-Z]+", gene)
    return match.group(0) if match else gene


def capped_select(ranked: pd.DataFrame, n_genes: int, max_family_size: int) -> pd.DataFrame:
    if max_family_size <= 0:
        return ranked.head(n_genes)

    selected_rows = []
    family_counts: dict[str, int] = {}
    for _, row in ranked.iterrows():
        family = str(row["gene_family"])
        if family_counts.get(family, 0) >= max_family_size:
            continue
        selected_rows.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected_rows) >= n_genes:
            break
    return pd.DataFrame(selected_rows)


def select_genes(args: argparse.Namespace) -> pd.DataFrame:
    processed_dir = Path(args.processed_dir)
    diagnostics_dir = Path(args.diagnostics_dir)
    gene_table = pd.read_csv(processed_dir / "gene_subset_300.csv")
    quality = pd.read_csv(diagnostics_dir / "perturbation_quality_table.csv")

    if "gene" not in gene_table.columns:
        raise ValueError("gene_subset_300.csv must contain a 'gene' column")
    if "perturbation" not in quality.columns:
        raise ValueError("perturbation_quality_table.csv must contain a 'perturbation' column")

    genes = gene_table["gene"].astype(str).tolist()
    edge_sets = load_ground_truth_edge_sets(
        Path(args.gt_dir),
        genes,
        ["chipAtlas", "corum", "stringdb"],
        string_min_score=args.string_min_score,
        string_physical_only=args.string_physical_only,
    )
    degree_df = source_degrees(edge_sets, genes)

    df = (
        gene_table
        .merge(quality, left_on="gene", right_on="perturbation", how="left")
        .merge(degree_df, on="gene", how="left")
    )
    df["gt_degree_pooled"] = df["gt_degree_pooled"].fillna(0).astype(int)
    df["gene_family"] = df["gene"].astype(str).apply(family_key)

    preferred_kd = set(args.preferred_kd_strata)
    allowed_batch = set(args.allowed_batch_strata)
    strict_mask = (
        df["diag_status"].eq("full")
        & df["diag_kd_stratum"].isin(preferred_kd)
        & df["diag_batch_stratum"].isin(allowed_batch)
        & df["gt_degree_pooled"].gt(0)
    )
    relaxed_mask = (
        df["diag_status"].eq("full")
        & df["diag_batch_stratum"].isin(allowed_batch)
        & df["gt_degree_pooled"].gt(0)
    )

    df["selection_pool"] = "excluded"
    df.loc[relaxed_mask, "selection_pool"] = "relaxed"
    df.loc[strict_mask, "selection_pool"] = "strict"

    kd_raw = 1.0 - pd.to_numeric(df["diag_kd_residual_ratio"], errors="coerce").clip(0, 1)
    batch_raw = -pd.to_numeric(df["diag_batch_js_divergence"], errors="coerce")
    df["benchmark_score"] = (
        0.40 * pct_rank(df["gt_degree_pooled"])
        + 0.20 * pct_rank(df.get("selection_score", pd.Series(0, index=df.index)))
        + 0.20 * pct_rank(kd_raw)
        + 0.10 * pct_rank(batch_raw)
        + 0.10 * pct_rank(df.get("perturbation_cells", pd.Series(0, index=df.index)))
    )

    strict = df[strict_mask].sort_values(
        ["benchmark_score", "gt_degree_pooled", "selection_score"],
        ascending=False,
    )
    selected = capped_select(strict, args.n_genes, args.max_family_size)

    if len(selected) < args.n_genes:
        fallback = (
            df[relaxed_mask & ~df["gene"].isin(selected["gene"])]
            .sort_values(["benchmark_score", "gt_degree_pooled", "selection_score"], ascending=False)
        )
        fallback = capped_select(
            pd.concat([selected, fallback], ignore_index=True),
            args.n_genes,
            args.max_family_size,
        )
        fallback = fallback[~fallback["gene"].isin(selected["gene"])]
        selected = pd.concat([selected, fallback], ignore_index=True)

    if len(selected) < args.n_genes:
        fallback = (
            df[~df["gene"].isin(selected["gene"])]
            .sort_values(["benchmark_score", "gt_degree_pooled", "selection_score"], ascending=False)
        )
        fallback = capped_select(
            pd.concat([selected, fallback], ignore_index=True),
            args.n_genes,
            args.max_family_size,
        )
        fallback = fallback[~fallback["gene"].isin(selected["gene"])]
        selected = pd.concat([selected, fallback], ignore_index=True)

    selected = selected.copy()
    selected.insert(0, "benchmark_rank", range(1, len(selected) + 1))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/k562_essential")
    parser.add_argument("--diagnostics-dir", default="results/diagnostics")
    parser.add_argument("--gt-dir", default="data/causalbench")
    parser.add_argument("--output-dir", default="results/benchmark_50")
    parser.add_argument("--n-genes", type=int, default=50)
    parser.add_argument("--preferred-kd-strata", nargs="+", default=["strong", "medium"])
    parser.add_argument("--allowed-batch-strata", nargs="+", default=["low", "medium"])
    parser.add_argument("--max-family-size", type=int, default=8,
                        help="Maximum genes per simple family prefix; <=0 disables cap")
    parser.add_argument("--string-min-score", type=int, default=700,
                        help="Minimum STRING score to include in benchmark GT degree")
    parser.add_argument("--string-physical-only", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use STRING physical links only for benchmark GT degree")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_genes(args)
    selected_path = output_dir / "benchmark_genes.csv"
    selected.to_csv(selected_path, index=False)

    gene_list_path = output_dir / "gene_list.txt"
    gene_list_path.write_text("\n".join(selected["gene"].astype(str)) + "\n", encoding="utf-8")

    degree_cols = [c for c in selected.columns if c.startswith("gt_degree_")]
    summary = {
        "n_genes": int(len(selected)),
        "genes": selected["gene"].astype(str).tolist(),
        "selection_pools": selected["selection_pool"].value_counts().to_dict(),
        "kd_strata": selected["diag_kd_stratum"].value_counts().to_dict(),
        "batch_strata": selected["diag_batch_stratum"].value_counts().to_dict(),
        "gene_families": selected["gene_family"].value_counts().to_dict(),
        "ground_truth_degree_totals": {
            col: int(selected[col].sum()) for col in degree_cols
        },
        "string_min_score": int(args.string_min_score),
        "string_physical_only": bool(args.string_physical_only),
    }
    summary_path = output_dir / "benchmark_gene_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {selected_path}")
    print(f"Wrote {gene_list_path}")
    print(f"Wrote {summary_path}")
    print(
        selected[
            ["benchmark_rank", "gene", "gene_family", "benchmark_score", "selection_pool", *degree_cols]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
