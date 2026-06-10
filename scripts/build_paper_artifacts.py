"""Generate paper source artifacts that back the report's headline numbers.

Reads the raw/residualized faithfulness matrices (.npy) and writes small,
committable summary files (no large h5ad/npy):

- results/faithfulness/fold_stability.json : cross-fold (I1 vs I2) Spearman
  rank stability for raw and residualized faithfulness.
- results/faithfulness/threshold_table.csv : per-fold fraction of self vs
  downstream pairs whose raw Wasserstein-1 faithfulness exceeds each epsilon
  threshold, plus medians (backs Result 2 of the paper).
- results/faithfulness/threshold_summary.json : the headline scalars
  (self >= 0.5, downstream >= 0.2 / >= 0.3, medians), per fold and averaged.

Pure read-only on the matrices; safe to re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
FDIR = ROOT / "results" / "faithfulness"

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FOLDS = ["I1", "I2"]


def _load(name: str) -> np.ndarray:
    return np.load(FDIR / f"{name}.npy")


def _split(F: np.ndarray):
    n = F.shape[0]
    diag = np.diag(F)
    off = F[~np.eye(n, dtype=bool)]
    return diag, off


def build_fold_stability() -> dict:
    out = {}
    for kind in ["raw", "residualized"]:
        a = _load(f"F_{kind}_I1").ravel()
        b = _load(f"F_{kind}_I2").ravel()
        rho = float(spearmanr(a, b).correlation)
        out[kind] = {"fold_a": "I1", "fold_b": "I2", "spearman_rho": rho,
                     "n_pairs": int(a.size)}
    with open(FDIR / "fold_stability.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def build_threshold_table() -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        F = _load(f"F_raw_{fold}")
        diag, off = _split(F)
        for pair_type, vals in [("self", diag), ("downstream", off)]:
            for t in THRESHOLDS:
                n_ge = int(np.sum(vals >= t))
                rows.append({
                    "matrix": "raw",
                    "fold": fold,
                    "pair_type": pair_type,
                    "threshold": t,
                    "n_total": int(vals.size),
                    "n_ge": n_ge,
                    "frac_ge": n_ge / vals.size,
                    "pct_ge": round(100 * n_ge / vals.size, 2),
                })
    df = pd.DataFrame(rows)
    df.to_csv(FDIR / "threshold_table.csv", index=False)
    return df


def build_threshold_summary() -> dict:
    per_fold = {}
    for fold in FOLDS:
        F = _load(f"F_raw_{fold}")
        diag, off = _split(F)
        per_fold[fold] = {
            "self_ge_0.5_pct": round(100 * float(np.mean(diag >= 0.5)), 2),
            "self_median": float(np.median(diag)),
            "downstream_ge_0.2_pct": round(100 * float(np.mean(off >= 0.2)), 2),
            "downstream_ge_0.3_pct": round(100 * float(np.mean(off >= 0.3)), 2),
            "downstream_median": float(np.median(off)),
        }
    keys = list(per_fold["I1"].keys())
    averaged = {k: round(np.mean([per_fold[f][k] for f in FOLDS]), 4) for k in keys}
    out = {"per_fold": per_fold, "fold_averaged": averaged,
           "thresholds": THRESHOLDS}
    with open(FDIR / "threshold_summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    fs = build_fold_stability()
    tbl = build_threshold_table()
    summ = build_threshold_summary()
    print("fold_stability:", {k: round(v["spearman_rho"], 4) for k, v in fs.items()})
    print("\nthreshold_table (raw):")
    print(tbl.pivot_table(index=["pair_type", "threshold"], columns="fold",
                          values="pct_ge").to_string())
    print("\nthreshold_summary fold_averaged:", summ["fold_averaged"])
    print("\nWrote:", FDIR / "fold_stability.json", FDIR / "threshold_table.csv",
          FDIR / "threshold_summary.json", sep="\n  ")
