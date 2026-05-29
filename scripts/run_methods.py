"""
Member 4 — Causal discovery methods runner.

Runs PC, GES, GIES, FCI, GRNBoost, MeanDifference on the 300-gene K562 subset
across three training regimes (observational, partial_interventional, interventional).
Outputs standardised edge lists consumed by Module C (run_crossfit.py).

Output files (one per method × regime):
    results/methods/{method}_{regime}_edges.csv   columns: source_gene, target_gene, score

With --fold-split, also writes fold-specific files:
    results/methods/{method}_{regime}_I1_edges.csv
    results/methods/{method}_{regime}_I2_edges.csv

Usage:
    # Full run (all methods × regimes)
    python scripts/run_methods.py \\
        --processed-dir data/processed/k562_essential \\
        --fold-assignment results/faithfulness/fold_assignment.csv \\
        --output-dir results/methods

    # Single method smoke-test
    python scripts/run_methods.py \\
        --methods grnboost ges \\
        --regimes observational \\
        --output-dir results/methods

    # With fold-specific outputs for cross-fitting
    python scripts/run_methods.py --fold-split \\
        --processed-dir data/processed/k562_essential \\
        --fold-assignment results/faithfulness/fold_assignment.csv \\
        --output-dir results/methods
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CONTROL_LABEL = "non-targeting"
ALL_METHODS = ["grnboost", "meandifference", "ges", "pc", "fci", "gies"]
ALL_REGIMES = ["observational", "partial_interventional", "interventional"]


# ── data preparation ──────────────────────────────────────────────────────────

def load_adata(processed_dir: Path) -> ad.AnnData:
    path = processed_dir / "processed_300_gene_subset.h5ad"
    print(f"Loading {path}")
    adata = ad.read_h5ad(path)
    print(f"  shape: {adata.shape}")
    return adata


def get_expression(adata: ad.AnnData, cell_mask: np.ndarray) -> np.ndarray:
    """Return log1p expression matrix (cells × genes) as dense float32."""
    X = adata.layers["log1p"][cell_mask]
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def prepare_data(
    adata: ad.AnnData,
    regime: str,
    fold_cells: Optional[pd.Index] = None,
    max_obs_cells: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Prepare expression data for a given regime.

    Returns dict with keys:
        X_obs       : (n_ctrl, n_genes) control expression
        X_all       : (n_all, n_genes) all cells expression (partial_interventional)
        environments: list of (X_env, intervention_set) for GIES
        genes       : list of gene names
        pert_obs    : obs DataFrame for all non-control cells (used by MeanDifference)
    """
    rng = np.random.default_rng(seed)
    genes = list(adata.var_names)
    obs = adata.obs.copy()

    # restrict to fold if specified
    if fold_cells is not None:
        obs = obs.loc[obs.index.intersection(fold_cells)]

    ctrl_idx = obs.index[obs["perturbation"] == CONTROL_LABEL]
    all_idx = obs.index

    # downsample control cells if needed
    if len(ctrl_idx) > max_obs_cells:
        ctrl_idx = pd.Index(
            rng.choice(ctrl_idx, size=max_obs_cells, replace=False)
        )

    ctrl_mask = adata.obs.index.isin(ctrl_idx)
    all_mask = adata.obs.index.isin(all_idx)
    pert_mask = adata.obs.index.isin(obs.index[obs["perturbation"] != CONTROL_LABEL])

    X_obs = get_expression(adata, ctrl_mask)
    X_all = get_expression(adata, all_mask)
    pert_obs = obs[obs["perturbation"] != CONTROL_LABEL]

    # build GIES environments: env 0 = control, env k = perturbation k
    gene_index = {g: i for i, g in enumerate(genes)}
    environments = [(X_obs, [])]  # (expression_matrix, intervention_target_indices)

    # only use perturbations whose target is in the 300-gene subset
    measurable_perts = sorted(
        set(pert_obs["perturbation"].unique()) & set(genes)
    )
    for pert_gene in measurable_perts:
        p_mask = adata.obs.index.isin(
            pert_obs.index[pert_obs["perturbation"] == pert_gene]
        )
        if p_mask.sum() < 5:
            continue
        X_p = get_expression(adata, p_mask)
        environments.append((X_p, [gene_index[pert_gene]]))

    return {
        "X_obs": X_obs,
        "X_all": X_all,
        "environments": environments,
        "genes": genes,
        "pert_obs": pert_obs,
        "ctrl_idx": ctrl_idx,
    }


def select_X(data: dict, regime: str) -> np.ndarray:
    """Return the expression matrix appropriate for this regime."""
    if regime == "observational":
        return data["X_obs"]
    return data["X_all"]  # partial_interventional and interventional fall back to all cells


# ── graph → edge list conversion ──────────────────────────────────────────────

def causallearn_graph_to_edges(graph: np.ndarray, genes: list[str]) -> pd.DataFrame:
    """
    Convert causal-learn graph matrix to edge DataFrame.

    Encoding (causal-learn convention):
        graph[i][j] = -1  →  arrowhead mark at j for the i-j edge
        graph[i][j] =  1  →  tail mark at j for the i-j edge
        graph[i][j] =  0  →  no edge

    Directed edge i → j : graph[j][i] = -1 AND graph[i][j] = 1 → score 1.0
    Undirected edge i — j: graph[i][j] = -1 AND graph[j][i] = -1 → score 0.5 each direction
    Bidirected / circle marks (FCI PAG): score 0.3 each direction
    """
    n = len(genes)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            gij = int(graph[i][j])
            gji = int(graph[j][i])
            if gij == 0 and gji == 0:
                continue
            # directed j → i : graph[i][j]=1 (tail at i), graph[j][i]=-1 (arrow at i)
            if gij == 1 and gji == -1:
                rows.append({"source_gene": genes[j], "target_gene": genes[i], "score": 1.0})
            # directed i → j : graph[j][i]=1, graph[i][j]=-1
            elif gij == -1 and gji == 1:
                rows.append({"source_gene": genes[i], "target_gene": genes[j], "score": 1.0})
            # undirected or ambiguous
            else:
                rows.append({"source_gene": genes[i], "target_gene": genes[j], "score": 0.5})
                rows.append({"source_gene": genes[j], "target_gene": genes[i], "score": 0.5})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source_gene", "target_gene", "score"]
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def adj_matrix_to_edges(adj: np.ndarray, genes: list[str]) -> pd.DataFrame:
    """
    Convert a weighted/binary adjacency matrix to edge DataFrame.
    adj[i][j] > 0 means edge i → j with that weight.
    """
    rows = []
    for i in range(len(genes)):
        for j in range(len(genes)):
            if i != j and adj[i, j] > 0:
                rows.append({
                    "source_gene": genes[i],
                    "target_gene": genes[j],
                    "score": float(adj[i, j]),
                })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source_gene", "target_gene", "score"]
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ── method implementations ────────────────────────────────────────────────────

def run_grnboost(X: np.ndarray, genes: list[str], n_jobs: int = 4,
                 n_estimators: int = 500) -> pd.DataFrame:
    """
    GRNBoost2-equivalent: for each target gene j, fit an ExtraTrees regressor
    and use feature importances as edge weights (source_gene i → target_gene j).

    ExtraTreesRegressor matches GRNBoost2 behavior and supports n_jobs for
    true parallelism (GradientBoostingRegressor does not have n_jobs).
    """
    from sklearn.ensemble import ExtraTreesRegressor

    n_genes = len(genes)
    rows = []

    for j, target in enumerate(genes):
        # regress target gene on all others
        feature_idx = [i for i in range(n_genes) if i != j]
        X_feat = X[:, feature_idx]
        y = X[:, j]

        reg = ExtraTreesRegressor(
            n_estimators=n_estimators, max_features="sqrt",
            random_state=42, n_jobs=n_jobs,
        )
        reg.fit(X_feat, y)

        for k, imp in enumerate(reg.feature_importances_):
            if imp > 0:
                rows.append({
                    "source_gene": genes[feature_idx[k]],
                    "target_gene": target,
                    "score": float(imp),
                })

        if (j + 1) % 50 == 0:
            print(f"    GRNBoost: {j + 1}/{n_genes} genes done", flush=True)

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source_gene", "target_gene", "score"]
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def run_mean_difference(
    adata: ad.AnnData,
    data: dict,
    regime: str,
) -> pd.DataFrame:
    """
    MeanDifference baseline: for each perturbed gene i, score edge i→j by
    |mean(X_j | do(X_i)) - mean(X_j | control)|.

    For observational regime: falls back to computing absolute pairwise
    Pearson correlation (no intervention info available).
    """
    genes = data["genes"]
    gene_index = {g: k for k, g in enumerate(genes)}

    if regime == "observational":
        # Pearson correlation as proxy
        X = data["X_obs"]
        corr = np.abs(np.corrcoef(X.T))
        np.fill_diagonal(corr, 0.0)
        return adj_matrix_to_edges(corr, genes)

    # interventional / partial_interventional
    ctrl_mask = adata.obs.index.isin(data["ctrl_idx"])
    X_ctrl = get_expression(adata, ctrl_mask)
    ctrl_mean = X_ctrl.mean(axis=0)  # (n_genes,)

    rows = []
    pert_obs = data["pert_obs"]
    for pert_gene in sorted(set(pert_obs["perturbation"].unique()) & set(genes)):
        p_idx = adata.obs.index.isin(
            pert_obs.index[pert_obs["perturbation"] == pert_gene]
        )
        if p_idx.sum() < 5:
            continue
        X_p = get_expression(adata, p_idx)
        diff = np.abs(X_p.mean(axis=0) - ctrl_mean)

        for j, g_j in enumerate(genes):
            if g_j == pert_gene:
                continue
            rows.append({
                "source_gene": pert_gene,
                "target_gene": g_j,
                "score": float(diff[j]),
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source_gene", "target_gene", "score"]
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def run_pc(X: np.ndarray, genes: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """PC algorithm (Peter-Clark). Uses Fisher-z conditional independence test."""
    from causallearn.search.ConstraintBased.PC import pc as _pc

    print(f"    Running PC on {X.shape[0]} cells × {X.shape[1]} genes, alpha={alpha}")
    cg = _pc(X, alpha=alpha, indep_test="fisherz", stable=True, show_progress=False)
    return causallearn_graph_to_edges(cg.G.graph, genes)


def run_ges(X: np.ndarray, genes: list[str]) -> pd.DataFrame:
    """GES (Greedy Equivalence Search). Observational BIC score."""
    from causallearn.search.ScoreBased.GES import ges as _ges

    print(f"    Running GES on {X.shape[0]} cells × {X.shape[1]} genes")
    rec = _ges(X, score_func="local_score_BIC", maxP=None)
    return causallearn_graph_to_edges(rec["G"].graph, genes)


def run_fci(X: np.ndarray, genes: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """
    FCI (Fast Causal Inference). Allows latent confounders.
    Outputs a PAG (Partial Ancestral Graph) — the key PC vs FCI contrast.
    """
    from causallearn.search.ConstraintBased.FCI import fci as _fci

    print(f"    Running FCI on {X.shape[0]} cells × {X.shape[1]} genes, alpha={alpha}")
    g, _ = _fci(X, independence_test_method="fisherz", alpha=alpha, show_progress=False)
    return causallearn_graph_to_edges(g.graph, genes)


def run_gies(data: dict) -> pd.DataFrame:
    """
    GIES (Greedy Interventional Equivalence Search).
    Uses all available intervention environments.
    Only meaningful for 'interventional' regime.
    """
    import gies as _gies

    environments = data["environments"]
    genes = data["genes"]
    n = len(genes)

    gies_data = [env[0] for env in environments]
    gies_I = [list(env[1]) for env in environments]  # gies.fit_bic requires lists, not sets

    print(f"    Running GIES: {n} genes, {len(environments)} environments "
          f"({sum(1 for e in environments if not e[1])} observational, "
          f"{sum(1 for e in environments if e[1])} interventional)")

    dag_adj, _ = _gies.fit_bic(gies_data, gies_I)

    return adj_matrix_to_edges(dag_adj, genes)


# ── orchestration ─────────────────────────────────────────────────────────────

def run_one(
    method: str,
    regime: str,
    adata: ad.AnnData,
    data: dict,
    alpha: float,
    n_jobs: int,
    n_estimators: int = 500,
) -> pd.DataFrame:
    """Run a single method × regime combination."""
    X = select_X(data, regime)

    if method == "grnboost":
        return run_grnboost(X, data["genes"], n_jobs=n_jobs, n_estimators=n_estimators)

    elif method == "meandifference":
        return run_mean_difference(adata, data, regime)

    elif method == "pc":
        return run_pc(X, data["genes"], alpha=alpha)

    elif method == "ges":
        return run_ges(X, data["genes"])

    elif method == "fci":
        return run_fci(X, data["genes"], alpha=alpha)

    elif method == "gies":
        if regime != "interventional":
            # GIES without intervention info = GES
            print(f"    GIES in {regime} mode → running as GES (no intervention info)")
            return run_ges(X, data["genes"])
        return run_gies(data)

    else:
        raise ValueError(f"Unknown method: {method}")


def save_edges(edges: pd.DataFrame, output_dir: Path, method: str, regime: str,
               fold: Optional[str] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_suffix = f"_{fold}" if fold else ""
    path = output_dir / f"{method}_{regime}{fold_suffix}_edges.csv"
    edges.to_csv(path, index=False)
    print(f"    Wrote {path.name} ({len(edges)} edges)")
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processed-dir", default="data/processed/k562_essential")
    parser.add_argument("--fold-assignment", default="results/faithfulness/fold_assignment.csv",
                        help="fold_assignment.csv from run_faithfulness.py")
    parser.add_argument("--output-dir", default="results/methods")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, metavar="METHOD",
                        help=f"Methods to run (default: all). Choices: {ALL_METHODS}")
    parser.add_argument("--regimes", nargs="+", default=ALL_REGIMES,
                        choices=ALL_REGIMES, metavar="REGIME",
                        help=f"Training regimes (default: all). Choices: {ALL_REGIMES}")
    parser.add_argument("--fold-split", action="store_true",
                        help="Also produce I1/I2 fold-specific edge files for cross-fitting")
    parser.add_argument("--max-obs-cells", type=int, default=10_000,
                        help="Max control cells to use for observational methods (default: 10000)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for PC and FCI (default: 0.05)")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Parallel jobs for GRNBoost (default: 4)")
    parser.add_argument("--n-estimators", type=int, default=500,
                        help="GRNBoost trees per gene (default: 500; use 100 for fast runs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = load_adata(processed_dir)
    genes = list(adata.var_names)
    print(f"Genes: {len(genes)}, Cells: {adata.n_obs}")

    # load fold assignment if needed
    fold_assignment: Optional[pd.DataFrame] = None
    if args.fold_split:
        fa_path = Path(args.fold_assignment)
        if fa_path.exists():
            fold_assignment = pd.read_csv(fa_path, index_col=0)
            print(f"Loaded fold assignment: {fa_path}")
        else:
            print(f"WARNING: fold assignment not found at {fa_path}. Skipping fold-split.")

    total = len(args.methods) * len(args.regimes)
    done = 0

    for regime in args.regimes:
        print(f"\n{'='*60}")
        print(f"Regime: {regime}")
        print(f"{'='*60}")

        # full-dataset data
        data_full = prepare_data(adata, regime,
                                  max_obs_cells=args.max_obs_cells, seed=args.seed)

        # fold-specific data (if requested)
        data_folds: dict[str, dict] = {}
        if fold_assignment is not None:
            for fold in ["I1", "I2"]:
                fold_cells = fold_assignment.index[fold_assignment["fold"] == fold]
                data_folds[fold] = prepare_data(
                    adata, regime, fold_cells=fold_cells,
                    max_obs_cells=args.max_obs_cells, seed=args.seed,
                )

        for method in args.methods:
            done += 1
            print(f"\n[{done}/{total}] {method} × {regime}")
            t0 = time.time()

            try:
                # full-dataset run
                edges = run_one(method, regime, adata, data_full, args.alpha, args.n_jobs,
                               n_estimators=args.n_estimators)
                save_edges(edges, output_dir, method, regime)

                # fold-specific runs
                for fold, data_fold in data_folds.items():
                    print(f"    Fold {fold}")
                    edges_fold = run_one(method, regime, adata, data_fold,
                                         args.alpha, args.n_jobs,
                                         n_estimators=args.n_estimators)
                    save_edges(edges_fold, output_dir, method, regime, fold=fold)

            except Exception as exc:
                print(f"    ERROR: {exc}")
                import traceback
                traceback.print_exc()

            print(f"    Done in {time.time() - t0:.1f}s")

    print(f"\nAll methods complete. Edge files in {output_dir}/")


if __name__ == "__main__":
    main()
