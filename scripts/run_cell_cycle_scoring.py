"""One-shot script: score cell cycle on raw k562.h5ad using a memory-
efficient chunked implementation. Saves a lightweight CSV with cell_id,
S_score, G2M_score, phase.

Why custom: scanpy's score_genes_cell_cycle materializes large
temporary arrays. On a 16GB Mac with the full Replogle K562 dataset
(310k cells x 8.5k genes) it OOMs. We replicate the same algorithm
(Tirosh et al. 2016, Satija lab) but stream cells in chunks.

Algorithm (per gene set, applied to S genes and to G2M genes):
  1. Compute mean expression across ALL cells for every gene.
  2. Bin all genes into 25 expression bins.
  3. For each target gene, sample 50 control genes from the same bin.
  4. score(cell) = mean(target genes in cell) - mean(control genes in cell).
Phase assignment: max(S_score, G2M_score) if positive, else G1.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


# Regev lab cell cycle gene set, same as scanpy tutorial.
S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
    "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
    "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
    "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
]

G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
    "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
    "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
    "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]

N_BINS = 25
N_CTRL_PER_GENE = 50
CHUNK_SIZE = 30_000  # cells per chunk


def _per_cell_lognormalize_chunk(X_chunk, target_sum: float = 1e4):
    """Per-cell normalize + log1p one chunk.
    Returns a dense float32 array (chunk_cells x genes).
    """
    if sparse.issparse(X_chunk):
        X_chunk = X_chunk.toarray()
    X_chunk = X_chunk.astype(np.float32, copy=False)
    totals = X_chunk.sum(axis=1)
    totals = np.maximum(totals, 1.0).astype(np.float32)
    X_chunk = X_chunk * (target_sum / totals[:, None])
    np.log1p(X_chunk, out=X_chunk)
    return X_chunk


def _compute_gene_mean_streaming(adata: ad.AnnData, chunk_size: int) -> np.ndarray:
    """Pass 1: compute per-gene mean (over log-normalized cells), streaming."""
    n_cells, n_genes = adata.shape
    gene_sum = np.zeros(n_genes, dtype=np.float64)
    n_done = 0
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        X_chunk = _per_cell_lognormalize_chunk(adata.X[start:end])
        gene_sum += X_chunk.sum(axis=0)
        n_done = end
        print(f"  pass1 (gene means): {n_done}/{n_cells} cells", flush=True)
        del X_chunk
        gc.collect()
    return (gene_sum / n_cells).astype(np.float32)


def _build_control_genes(
    gene_names: np.ndarray,
    gene_mean: np.ndarray,
    targets: list[str],
    n_bins: int,
    n_ctrl: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """For each target gene, pick n_ctrl control genes from the same
    expression bin. Returns a 1-D array of unique control-gene indices.
    """
    available = set(gene_names)
    target_idx = np.array([np.where(gene_names == g)[0][0] for g in targets if g in available])
    target_set = set(target_idx.tolist())

    # rank-based binning to handle uneven distributions
    ranks = pd.Series(gene_mean).rank(method="average").to_numpy()
    bin_ids = pd.cut(ranks, bins=n_bins, labels=False, include_lowest=True)

    ctrl = set()
    for ti in target_idx:
        same_bin = np.where(bin_ids == bin_ids[ti])[0]
        same_bin = same_bin[~np.isin(same_bin, list(target_set))]
        if len(same_bin) == 0:
            continue
        n_pick = min(n_ctrl, len(same_bin))
        picked = rng.choice(same_bin, size=n_pick, replace=False)
        ctrl.update(picked.tolist())
    return np.array(sorted(ctrl), dtype=np.int64)


def _score_streaming(
    adata: ad.AnnData,
    target_idx: np.ndarray,
    ctrl_idx: np.ndarray,
    chunk_size: int,
    label: str,
) -> np.ndarray:
    """Pass 2: per-cell score = mean(targets) - mean(ctrls), streaming."""
    n_cells = adata.shape[0]
    out = np.empty(n_cells, dtype=np.float32)
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        X_chunk = _per_cell_lognormalize_chunk(adata.X[start:end])
        target_mean = X_chunk[:, target_idx].mean(axis=1)
        ctrl_mean = X_chunk[:, ctrl_idx].mean(axis=1)
        out[start:end] = target_mean - ctrl_mean
        print(f"  pass2 ({label} score): {end}/{n_cells} cells", flush=True)
        del X_chunk
        gc.collect()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-h5ad", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.raw_h5ad.exists():
        raise FileNotFoundError(f"Missing input: {args.raw_h5ad}")

    print(f"Loading raw AnnData (backed='r') from {args.raw_h5ad}", flush=True)
    # backed='r' keeps the matrix on disk; we slice chunks as we need them.
    adata = ad.read_h5ad(args.raw_h5ad, backed="r")
    print(f"  shape = {adata.shape}", flush=True)

    gene_names = adata.var_names.to_numpy().astype(str)
    available = set(gene_names)
    s_use = [g for g in S_GENES if g in available]
    g2m_use = [g for g in G2M_GENES if g in available]
    print(
        f"Using {len(s_use)}/{len(S_GENES)} S genes, "
        f"{len(g2m_use)}/{len(G2M_GENES)} G2M genes.",
        flush=True,
    )
    if len(s_use) < 10 or len(g2m_use) < 10:
        raise RuntimeError("Too few cell-cycle markers in var_names.")

    print("\n[Pass 1] streaming per-gene mean over log-normalized data...", flush=True)
    gene_mean = _compute_gene_mean_streaming(adata, args.chunk_size)

    rng = np.random.default_rng(args.seed)
    print("\nBuilding control gene sets...", flush=True)
    s_target_idx = np.array([np.where(gene_names == g)[0][0] for g in s_use])
    g2m_target_idx = np.array([np.where(gene_names == g)[0][0] for g in g2m_use])
    s_ctrl_idx = _build_control_genes(
        gene_names, gene_mean, s_use, N_BINS, N_CTRL_PER_GENE, rng,
    )
    g2m_ctrl_idx = _build_control_genes(
        gene_names, gene_mean, g2m_use, N_BINS, N_CTRL_PER_GENE, rng,
    )
    print(f"  S control genes: {len(s_ctrl_idx)}", flush=True)
    print(f"  G2M control genes: {len(g2m_ctrl_idx)}", flush=True)

    print("\n[Pass 2a] streaming S_score...", flush=True)
    s_score = _score_streaming(adata, s_target_idx, s_ctrl_idx, args.chunk_size, "S")

    print("\n[Pass 2b] streaming G2M_score...", flush=True)
    g2m_score = _score_streaming(adata, g2m_target_idx, g2m_ctrl_idx, args.chunk_size, "G2M")

    print("\nAssigning phases...", flush=True)
    # scanpy convention: pick max score if positive, else G1
    phase = np.empty(len(s_score), dtype=object)
    s_higher = s_score > g2m_score
    phase[:] = "G1"
    phase[s_higher & (s_score > 0)] = "S"
    phase[(~s_higher) & (g2m_score > 0)] = "G2M"

    out = pd.DataFrame({
        "S_score": s_score,
        "G2M_score": g2m_score,
        "phase": phase,
    }, index=adata.obs_names)
    out.index.name = "cell_id"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv)
    print(f"\nWrote {args.output_csv} ({len(out)} cells)", flush=True)

    print("\nPhase counts:")
    print(out["phase"].value_counts())
    print("\nScore stats:")
    print(out[["S_score", "G2M_score"]].describe())

    adata.file.close()
    del adata
    gc.collect()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()