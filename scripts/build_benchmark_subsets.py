from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from scripts.build_benchmark_gene_set import capped_select, select_genes
except ModuleNotFoundError:  # pragma: no cover - used when run as scripts/foo.py
    from build_benchmark_gene_set import capped_select, select_genes


def weighted_capped_sample(
    ranked: pd.DataFrame,
    n_genes: int,
    max_family_size: int,
    seed: int,
    score_power: float,
    candidate_multiplier: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = ranked.head(max(n_genes * candidate_multiplier, n_genes)).copy()
    selected_rows = []
    family_counts: dict[str, int] = {}
    remaining = pool.index.to_numpy()

    scores = pd.to_numeric(pool["benchmark_score"], errors="coerce").fillna(0).to_numpy()
    weights = np.maximum(scores, 1e-6) ** score_power

    while len(selected_rows) < n_genes and len(remaining) > 0:
        remaining_positions = np.array([pool.index.get_loc(idx) for idx in remaining])
        p = weights[remaining_positions]
        p = p / p.sum()
        chosen_idx = rng.choice(remaining, p=p)
        row = pool.loc[chosen_idx]
        family = str(row["gene_family"])
        remaining = remaining[remaining != chosen_idx]
        if max_family_size > 0 and family_counts.get(family, 0) >= max_family_size:
            continue
        selected_rows.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1

    if len(selected_rows) < n_genes:
        selected_genes = {row["gene"] for row in selected_rows}
        fill_pool = ranked[~ranked["gene"].isin(selected_genes)]
        selected = pd.DataFrame(selected_rows)
        filled = capped_select(
            pd.concat([selected, fill_pool], ignore_index=True),
            n_genes,
            max_family_size,
        )
        return filled.head(n_genes).copy()

    return pd.DataFrame(selected_rows).head(n_genes).copy()


def write_subset(selected: pd.DataFrame, output_dir: Path, subset_id: str, seed: int) -> None:
    subset_dir = output_dir / subset_id
    subset_dir.mkdir(parents=True, exist_ok=True)
    selected = selected.copy().reset_index(drop=True)
    selected.insert(0, "benchmark_rank", range(1, len(selected) + 1))
    selected.to_csv(subset_dir / "benchmark_genes.csv", index=False)
    (subset_dir / "gene_list.txt").write_text(
        "\n".join(selected["gene"].astype(str)) + "\n",
        encoding="utf-8",
    )
    summary = {
        "subset_id": subset_id,
        "seed": int(seed),
        "n_genes": int(len(selected)),
        "genes": selected["gene"].astype(str).tolist(),
        "selection_pools": selected["selection_pool"].value_counts().to_dict(),
        "gene_families": selected["gene_family"].value_counts().to_dict(),
        "ground_truth_degree_totals": {
            col: int(selected[col].sum())
            for col in selected.columns
            if col.startswith("gt_degree_")
        },
    }
    (subset_dir / "benchmark_gene_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/k562_essential")
    parser.add_argument("--diagnostics-dir", default="results/diagnostics")
    parser.add_argument("--gt-dir", default="data/causalbench")
    parser.add_argument("--output-dir", default="results/robustness_50_hc")
    parser.add_argument("--n-subsets", type=int, default=3)
    parser.add_argument("--n-genes", type=int, default=50)
    parser.add_argument("--max-family-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-power", type=float, default=3.0)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--string-min-score", type=int, default=700)
    parser.add_argument("--string-physical-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    selection_args = SimpleNamespace(
        processed_dir=args.processed_dir,
        diagnostics_dir=args.diagnostics_dir,
        gt_dir=args.gt_dir,
        preferred_kd_strata=["strong", "medium"],
        allowed_batch_strata=["low", "medium"],
        n_genes=args.n_genes,
        max_family_size=args.max_family_size,
        string_min_score=args.string_min_score,
        string_physical_only=args.string_physical_only,
    )
    ranked = select_genes(selection_args).drop(columns=["benchmark_rank"], errors="ignore")

    # Rebuild the candidate pool by requesting more genes from the deterministic selector.
    selection_args.n_genes = min(args.n_genes * args.candidate_multiplier, 300)
    candidates = select_genes(selection_args).drop(columns=["benchmark_rank"], errors="ignore")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for i in range(args.n_subsets):
        subset_id = f"subset_{i + 1:02d}"
        if i == 0:
            selected = ranked.head(args.n_genes).copy()
            seed = args.seed
            mode = "deterministic_top"
        else:
            seed = args.seed + i
            selected = weighted_capped_sample(
                candidates,
                args.n_genes,
                args.max_family_size,
                seed,
                args.score_power,
                args.candidate_multiplier,
            )
            mode = "weighted_sample"
        write_subset(selected, output_dir, subset_id, seed)
        manifest_rows.append({
            "subset_id": subset_id,
            "seed": seed,
            "mode": mode,
            "n_genes": len(selected),
            "gene_list": str(output_dir / subset_id / "gene_list.txt"),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "subsets_manifest.csv", index=False)
    print(manifest.to_string(index=False))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
