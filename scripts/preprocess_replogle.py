#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from causalfaith.preprocessing import (
    GEMGROUP_COLUMN_CANDIDATES,
    PERTURBATION_COLUMN_CANDIDATES,
    PreprocessConfig,
    add_qc_covariates,
    build_gene_metadata,
    build_matched_random_subsets,
    filter_cells,
    infer_column,
    infer_control_labels,
    select_gene_subset,
    write_outputs,
)


PERTPY_LOADERS = {
    "k562_essential": "replogle_2022_k562_essential",
    "rpe1": "replogle_2022_rpe1",
}


def load_from_pertpy(dataset: str) -> ad.AnnData:
    if dataset not in PERTPY_LOADERS:
        raise ValueError(f"Unknown pertpy dataset '{dataset}'. Expected one of {sorted(PERTPY_LOADERS)}.")
    import pertpy as pt

    data_module = getattr(pt, "dt", None) or getattr(pt, "data", None)
    if data_module is None:
        raise RuntimeError("Could not find pertpy data module. Try `pip install pertpy`.")
    loader = getattr(data_module, PERTPY_LOADERS[dataset])
    return loader()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Replogle Perturb-seq data for CausalFaith."
    )
    parser.add_argument("--dataset", choices=sorted(PERTPY_LOADERS), help="Dataset to load from pertpy.")
    parser.add_argument("--input-h5ad", type=Path, help="Existing AnnData file to preprocess.")
    parser.add_argument("--dataset-name", default=None, help="Name used in output metadata.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=300)
    parser.add_argument("--min-counts", type=int, default=500)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-pct-mt", type=float, default=20.0)
    parser.add_argument("--min-cells-per-perturbation", type=int, default=50)
    parser.add_argument("--random-subset-count", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--perturbation-col", default=None)
    parser.add_argument("--gemgroup-col", default=None)
    parser.add_argument(
        "--write-full",
        action="store_true",
        help="Also write the full QC-filtered AnnData. This can be very large.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.dataset) == bool(args.input_h5ad):
        raise SystemExit("Provide exactly one of --dataset or --input-h5ad.")

    dataset_name = args.dataset_name or args.dataset or args.input_h5ad.stem
    config = PreprocessConfig(
        dataset_name=dataset_name,
        subset_size=args.subset_size,
        min_counts=args.min_counts,
        min_genes=args.min_genes,
        max_pct_mt=args.max_pct_mt,
        min_cells_per_perturbation=args.min_cells_per_perturbation,
        random_subset_count=args.random_subset_count,
        random_seed=args.random_seed,
        perturbation_col=args.perturbation_col,
        gemgroup_col=args.gemgroup_col,
    )

    adata = load_from_pertpy(args.dataset) if args.dataset else ad.read_h5ad(args.input_h5ad)
    print(f"Loaded AnnData with shape {adata.n_obs} cells x {adata.n_vars} genes.", flush=True)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    perturbation_col = infer_column(
        adata.obs.columns, PERTURBATION_COLUMN_CANDIDATES, args.perturbation_col
    )
    if perturbation_col is None:
        raise SystemExit(
            "Could not infer the perturbation column. Pass --perturbation-col explicitly."
        )

    gemgroup_col = infer_column(adata.obs.columns, GEMGROUP_COLUMN_CANDIDATES, args.gemgroup_col)
    if gemgroup_col is None:
        adata.obs["gemgroup"] = "unknown"
        gemgroup_col = "gemgroup"

    adata.obs["perturbation"] = adata.obs[perturbation_col].astype(str)
    adata.obs["gemgroup"] = adata.obs[gemgroup_col].astype(str)

    add_qc_covariates(adata)
    print("Computed or copied QC covariates.", flush=True)
    adata, qc_summary = filter_cells(adata, config)
    print(
        f"QC retained {qc_summary['cells_after_qc']} / {qc_summary['cells_before_qc']} cells.",
        flush=True,
    )

    control_labels = infer_control_labels(adata.obs["perturbation"])
    print(f"Inferred control labels: {control_labels}", flush=True)
    gene_metadata = build_gene_metadata(adata, "perturbation", control_labels)
    print("Built gene metadata.", flush=True)
    selected = select_gene_subset(
        gene_metadata,
        subset_size=config.subset_size,
        min_cells_per_perturbation=config.min_cells_per_perturbation,
    )
    print(f"Selected {len(selected)} genes.", flush=True)
    random_subsets = build_matched_random_subsets(
        selected,
        gene_metadata,
        subset_size=config.subset_size,
        subset_count=config.random_subset_count,
        seed=config.random_seed,
        min_cells_per_perturbation=config.min_cells_per_perturbation,
    )
    print("Built matched random subsets.", flush=True)

    summary = {
        **config.to_dict(),
        **qc_summary,
        "n_genes_after_qc": int(adata.n_vars),
        "perturbation_col_source": perturbation_col,
        "gemgroup_col_source": gemgroup_col,
        "control_labels": control_labels,
        "selected_genes": int(len(selected)),
        "random_subsets": int(random_subsets["subset_id"].nunique()) if not random_subsets.empty else 0,
    }
    write_outputs(
        adata,
        selected,
        random_subsets,
        gene_metadata,
        args.output_dir,
        summary,
        write_full=args.write_full,
    )
    print(f"Wrote processed outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
