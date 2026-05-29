from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


@dataclass
class CrossFitConfig:
    methods: list[str] = field(default_factory=lambda: [
        "pc", "fci", "ges", "gies", "dcdi_g", "grnboost",
    ])
    regimes: list[str] = field(default_factory=lambda: [
        "observational", "partial_interventional", "interventional",
    ])
    faithfulness_quantile: float = 0.5
    min_edges: int = 10
    gt_sources: list[str] = field(default_factory=lambda: [
        "chipAtlas", "corum", "stringdb",
    ])
    results_dir: Path = Path("results/crossfit")


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_col(columns, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ── data loaders ──────────────────────────────────────────────────────────────

def load_f_matrix(path: Path) -> pd.DataFrame:
    """Load faithfulness matrix from .npy (with companion .csv for labels) or .csv."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path, index_col=0)
        return pd.DataFrame(arr)
    if path.suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    raise ValueError(f"Unsupported format: {path.suffix}. Use .npy or .csv")


def load_method_edges(path: Path) -> pd.DataFrame:
    """
    Load edge list from CSV.
    Expected columns: source_gene, target_gene, score.
    Also accepts common aliases: i/j, from/to, weight.
    """
    df = pd.read_csv(path)
    aliases = {
        "i": "source_gene", "from": "source_gene",
        "j": "target_gene", "to": "target_gene",
        "weight": "score",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    missing = {"source_gene", "target_gene", "score"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Found: {list(df.columns)}")
    return df[["source_gene", "target_gene", "score"]].copy()


def load_ground_truth_edges(
    gt_dir: Path,
    genes: list[str],
    sources: Optional[list[str]] = None,
) -> set[tuple[str, str]]:
    """
    Load ground truth edges from CausalBench evaluation resource directory,
    filtered to the provided gene list.

    Looks for CSV/TSV files matching source name patterns (chipAtlas, corum, stringdb).
    Expects source/target columns (or gene1/gene2, regulator/regulated).

    Raises FileNotFoundError if no matching files are found — download evaluation
    resources with: python scripts/download_causalbench_data.py --with-evaluation-resources
    """
    gt_dir = Path(gt_dir)
    gene_set = set(genes)
    edges: set[tuple[str, str]] = set()

    patterns = sources or ["chipAtlas", "corum", "stringdb", "ground_truth"]
    found_any = False

    for pattern in patterns:
        for f in sorted(gt_dir.glob(f"*{pattern}*")):
            if f.suffix not in (".csv", ".tsv"):
                continue
            sep = "\t" if f.suffix == ".tsv" else ","
            try:
                df = pd.read_csv(f, sep=sep)
            except Exception:
                continue
            src = _find_col(df.columns, ["source", "gene1", "from", "source_gene", "regulator"])
            tgt = _find_col(df.columns, ["target", "gene2", "to", "target_gene", "regulated"])
            if src is None or tgt is None:
                continue
            for s, t in zip(df[src].astype(str), df[tgt].astype(str)):
                if s in gene_set and t in gene_set and s != t:
                    edges.add((s, t))
            found_any = True

    if not found_any:
        raise FileNotFoundError(
            f"No ground truth files found in {gt_dir} for patterns {patterns}.\n"
            "Download with: python scripts/download_causalbench_data.py "
            "--with-evaluation-resources"
        )
    return edges


def load_perturbation_quality(path: Path) -> pd.DataFrame:
    """Load member 3's perturbation_quality_table.csv, indexed by perturbation gene."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def load_fold_assignment(path: Path) -> pd.Series:
    """
    Load fold_assignment.csv produced by run_faithfulness.py.
    Returns Series mapping perturbation label → I1/I2/unused.
    """
    df = pd.read_csv(path, index_col=0)
    fold_col = next((c for c in df.columns if "fold" in c.lower()), None)
    if fold_col is None:
        raise ValueError(f"No 'fold' column in {path}. Columns: {list(df.columns)}")
    return df[fold_col]


# ── per-edge operations ───────────────────────────────────────────────────────

def edges_to_recall_labels(
    edges_df: pd.DataFrame,
    gt_edges: set[tuple[str, str]],
) -> pd.DataFrame:
    """Add binary in_ground_truth column to a predicted edge list."""
    df = edges_df.copy()
    gt_set = gt_edges
    df["in_ground_truth"] = [
        int((str(s), str(t)) in gt_set)
        for s, t in zip(df["source_gene"], df["target_gene"])
    ]
    return df


def f_matrix_to_edge_scores(F: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten F matrix to long-form edge DataFrame.
    Columns: source_gene, target_gene, f_score.
    Drops NaN entries and self-edges.
    """
    genes_i = F.index.tolist()
    genes_j = F.columns.tolist()
    rows = []
    for gene_i in genes_i:
        for gene_j in genes_j:
            if gene_i == gene_j:
                continue
            val = F.loc[gene_i, gene_j]
            if pd.isna(val):
                continue
            rows.append({"source_gene": gene_i, "target_gene": gene_j, "f_score": float(val)})
    return pd.DataFrame(rows)


def merge_faithfulness_recall(
    F_edges: pd.DataFrame,
    recall_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Inner join on (source_gene, target_gene) between F edges and recall-labelled edges."""
    return F_edges.merge(
        recall_edges[["source_gene", "target_gene", "in_ground_truth"]],
        on=["source_gene", "target_gene"],
        how="inner",
    )


# ── network features ──────────────────────────────────────────────────────────

def add_network_features(
    edges_df: pd.DataFrame,
    gene_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Append per-edge covariates used in partial correlation:
    - source_degree, target_degree: normalized out/in-degree in predicted network
    - source_ctrl_expr, target_ctrl_expr: control expression from gene_metadata
    """
    df = edges_df.copy()
    n_genes = len(set(df["source_gene"]) | set(df["target_gene"]))
    denom = max(n_genes - 1, 1)

    out_deg = df.groupby("source_gene").size() / denom
    in_deg = df.groupby("target_gene").size() / denom
    df["source_degree"] = df["source_gene"].map(out_deg).fillna(0.0)
    df["target_degree"] = df["target_gene"].map(in_deg).fillna(0.0)

    if gene_metadata is not None:
        expr_col = _find_col(
            gene_metadata.columns,
            ["ctrl_mean_expression", "control_mean", "mean_expression",
             "ctrl_mean", "mean_ctrl", "ctrl_norm_mean"],
        )
        if expr_col is not None:
            expr_map = gene_metadata[expr_col].to_dict()
            df["source_ctrl_expr"] = df["source_gene"].map(expr_map)
            df["target_ctrl_expr"] = df["target_gene"].map(expr_map)

    return df


# ── partial correlation ───────────────────────────────────────────────────────

def partial_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    control_cols: list[str],
) -> dict[str, float]:
    """
    Partial Spearman correlation of x_col and y_col after removing the
    linear effect of control_cols on both (rank-transform first).
    """
    present = [c for c in control_cols if c in df.columns]
    sub = df[[x_col, y_col] + present].dropna()
    n = len(sub)
    if n < 10:
        return {"rho": np.nan, "pvalue": np.nan, "n": n}

    ranked = sub.rank()

    if present:
        C = ranked[present].values
        reg = LinearRegression()
        reg.fit(C, ranked[x_col].values)
        x_resid = ranked[x_col].values - reg.predict(C)
        reg.fit(C, ranked[y_col].values)
        y_resid = ranked[y_col].values - reg.predict(C)
    else:
        x_resid = ranked[x_col].values
        y_resid = ranked[y_col].values

    rho, pvalue = stats.pearsonr(x_resid, y_resid)
    return {"rho": float(rho), "pvalue": float(pvalue), "n": n}


# ── cross-fitting ─────────────────────────────────────────────────────────────

def cross_fit_one_split(
    F_train: pd.DataFrame,
    edges_test: pd.DataFrame,
    gt_edges: set[tuple[str, str]],
    gene_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    One cross-fitting half:
    - F_train: faithfulness estimated on one cell fold
    - edges_test: method edges predicted from the held-out fold
    Returns per-edge DataFrame with f_score, in_ground_truth, network features.
    """
    recall_df = edges_to_recall_labels(edges_test, gt_edges)
    F_edges = f_matrix_to_edge_scores(F_train)
    merged = merge_faithfulness_recall(F_edges, recall_df)
    return add_network_features(merged, gene_metadata)


def cross_fit(
    F_I1: pd.DataFrame,
    edges_I2: pd.DataFrame,
    F_I2: pd.DataFrame,
    edges_I1: pd.DataFrame,
    gt_edges: set[tuple[str, str]],
    gene_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Full cross-fitting (Chernozhukov et al. 2018):
    average two splits to avoid finite-sample noise.

    Split A: faithfulness on I1, recall on I2.
    Split B: faithfulness on I2, recall on I1.

    Returns per-edge DataFrame with averaged f_score_mean and recall_mean.
    """
    split_a = cross_fit_one_split(F_I1, edges_I2, gt_edges, gene_metadata)
    split_b = cross_fit_one_split(F_I2, edges_I1, gt_edges, gene_metadata)
    split_a["split"] = "A"
    split_b["split"] = "B"

    combined = pd.concat([split_a, split_b], ignore_index=True)
    averaged = (
        combined
        .groupby(["source_gene", "target_gene"], as_index=False)
        .agg(
            f_score_mean=("f_score", "mean"),
            recall_mean=("in_ground_truth", "mean"),
            source_degree=("source_degree", "mean"),
            target_degree=("target_degree", "mean"),
            n_splits=("split", "count"),
        )
    )
    for col in ["source_ctrl_expr", "target_ctrl_expr"]:
        if col in combined.columns:
            averaged[col] = combined.groupby(
                ["source_gene", "target_gene"]
            )[col].mean().values
    return averaged


# ── correlation table ─────────────────────────────────────────────────────────

def compute_correlation_table(
    method_results: dict[str, pd.DataFrame],
    control_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For each method/regime, compute:
    - Raw Spearman correlation between f_score_mean and recall_mean
    - Partial Spearman controlling for network centrality and expression

    Parameters
    ----------
    method_results : mapping from "method_regime" label to cross-fitted edge DataFrame
    control_cols   : covariates to partial out; defaults to degree + expression

    Returns DataFrame sorted by partial_rho descending.
    """
    if control_cols is None:
        control_cols = [
            "source_degree", "target_degree",
            "source_ctrl_expr", "target_ctrl_expr",
        ]

    rows = []
    for label, df in method_results.items():
        valid = df.dropna(subset=["f_score_mean", "recall_mean"])
        n = len(valid)
        if n < 10:
            rows.append({"method_regime": label, "n_edges": n,
                         "spearman_rho": np.nan, "spearman_p": np.nan,
                         "partial_rho": np.nan, "partial_p": np.nan})
            continue

        raw = stats.spearmanr(valid["f_score_mean"], valid["recall_mean"], nan_policy="omit")
        spearman_rho = float(getattr(raw, "statistic", getattr(raw, "correlation", np.nan)))
        partial = partial_spearman(valid, "f_score_mean", "recall_mean", control_cols)

        rows.append({
            "method_regime": label,
            "n_edges": n,
            "spearman_rho": spearman_rho,
            "spearman_p": float(raw.pvalue),
            "partial_rho": partial["rho"],
            "partial_p": partial["pvalue"],
        })

    return (
        pd.DataFrame(rows)
        .sort_values("partial_rho", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


# ── stratified analysis ───────────────────────────────────────────────────────

def _recall_stats(
    edges_df: pd.DataFrame,
    gt_edges: set[tuple[str, str]],
    source_genes: set[str],
) -> dict:
    predicted = edges_df[edges_df["source_gene"].isin(source_genes)]
    n_predicted = len(predicted)
    gt_sub = {(s, t) for s, t in gt_edges if s in source_genes}
    n_gt = len(gt_sub)
    if n_predicted == 0 or n_gt == 0:
        return {"n_predicted": n_predicted, "n_gt": n_gt,
                "precision": np.nan, "recall": np.nan, "f1": np.nan}
    pred_set = set(zip(predicted["source_gene"].astype(str),
                       predicted["target_gene"].astype(str)))
    tp = len(pred_set & gt_sub)
    prec = tp / n_predicted
    rec = tp / n_gt
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"n_predicted": n_predicted, "n_gt": n_gt,
            "precision": prec, "recall": rec, "f1": f1}


def run_stratified_analysis(
    method_edges: dict[str, pd.DataFrame],
    gt_edges: set[tuple[str, str]],
    quality_table: pd.DataFrame,
    F: pd.DataFrame,
    config: CrossFitConfig,
) -> pd.DataFrame:
    """
    Three key contrasts from the proposal:

    1. faithfulness stratum (high/low median split on row-mean F)
       → compares causal methods vs non-causal baseline
    2. kd_efficiency stratum (strong/medium/weak, status==full)
       → GIES vs DCDI-G contrast for imperfect intervention
    3. batch_divergence stratum (low/medium/high, status in full|kd_unmeasurable)
       → PC vs FCI contrast for causal sufficiency

    Returns long-form DataFrame:
        contrast, method_regime, stratum, n_genes,
        n_predicted, n_gt, precision, recall, f1
    """
    # faithfulness strata: median split on row-mean F
    row_means = F.mean(axis=1, skipna=True).dropna()
    threshold = row_means.quantile(config.faithfulness_quantile)
    faith_strata = {
        "high": set(row_means[row_means >= threshold].index),
        "low": set(row_means[row_means < threshold].index),
    }

    # quality table strata
    kd_col = "diag_kd_stratum"
    batch_col = "diag_batch_stratum"

    kd_full = quality_table[quality_table["diag_status"] == "full"]
    batch_all = quality_table[quality_table["diag_status"].isin(["full", "kd_unmeasurable"])]

    all_rows = []
    for label, edges_df in method_edges.items():
        # ── contrast 1: faithfulness ──────────────────────────────────────────
        for stratum, genes in faith_strata.items():
            s = _recall_stats(edges_df, gt_edges, genes)
            all_rows.append({"contrast": "faithfulness", "method_regime": label,
                              "stratum": stratum, "n_genes": len(genes), **s})

        # ── contrast 2: KD efficiency ─────────────────────────────────────────
        if kd_col in quality_table.columns:
            for stratum_val, grp in kd_full.groupby(kd_col):
                genes = set(grp.index.astype(str))
                s = _recall_stats(edges_df, gt_edges, genes)
                all_rows.append({"contrast": "kd_efficiency", "method_regime": label,
                                  "stratum": stratum_val, "n_genes": len(genes), **s})

        # ── contrast 3: batch divergence ──────────────────────────────────────
        if batch_col in quality_table.columns:
            for stratum_val, grp in batch_all.groupby(batch_col):
                genes = set(grp.index.astype(str))
                s = _recall_stats(edges_df, gt_edges, genes)
                all_rows.append({"contrast": "batch_divergence", "method_regime": label,
                                  "stratum": stratum_val, "n_genes": len(genes), **s})

    cols = ["contrast", "method_regime", "stratum", "n_genes",
            "n_predicted", "n_gt", "precision", "recall", "f1"]
    return pd.DataFrame(all_rows, columns=cols) if all_rows else pd.DataFrame(columns=cols)


# ── sensitivity analysis ──────────────────────────────────────────────────────

def sensitivity_raw_vs_residualized(
    F_raw: pd.DataFrame,
    F_resid: pd.DataFrame,
) -> dict:
    """
    Measure agreement between raw and residualized F matrices.
    Returns Spearman rho, p-value, n valid pairs, and mean absolute rank change.
    """
    raw_vals = F_raw.values.ravel().astype(float)
    resid_vals = F_resid.values.ravel().astype(float)
    mask = ~(np.isnan(raw_vals) | np.isnan(resid_vals))
    n = int(mask.sum())
    if n < 10:
        return {"rho": np.nan, "pvalue": np.nan, "n": n, "mean_rank_change": np.nan}

    rv, sv = raw_vals[mask], resid_vals[mask]
    rho, pv = stats.spearmanr(rv, sv)
    raw_rank = pd.Series(rv).rank(pct=True)
    resid_rank = pd.Series(sv).rank(pct=True)
    mean_rank_change = float((raw_rank - resid_rank).abs().mean())
    return {
        "rho": float(rho), "pvalue": float(pv),
        "n": n, "mean_rank_change": mean_rank_change,
    }


def sensitivity_gene_subset(
    F: pd.DataFrame,
    matched_subsets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare faithfulness distribution between the 300-gene selected subset
    and matched random subsets from matched_random_subsets.csv.

    Returns one row per subset with mean/median faithfulness and n_genes.
    """
    rows = []

    row_means_300 = F.mean(axis=1, skipna=True).dropna()
    rows.append({
        "subset": "300_gene_selected",
        "n_genes": len(row_means_300),
        "mean_faithfulness": float(row_means_300.mean()),
        "median_faithfulness": float(row_means_300.median()),
        "std_faithfulness": float(row_means_300.std()),
    })

    for col in matched_subsets.columns:
        random_genes = matched_subsets[col].dropna().astype(str).tolist()
        overlap = [g for g in random_genes if g in F.index]
        if not overlap:
            continue
        rm = F.loc[overlap].mean(axis=1, skipna=True).dropna()
        rows.append({
            "subset": f"random_{col}",
            "n_genes": len(rm),
            "mean_faithfulness": float(rm.mean()),
            "median_faithfulness": float(rm.median()),
            "std_faithfulness": float(rm.std()),
        })

    return pd.DataFrame(rows)
