from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def edge_path(methods_dir: Path, label: str, fold: str) -> Path:
    suffix = "" if fold == "full" else f"_{fold}"
    return methods_dir / f"{label}{suffix}_edges.csv"


def fill_missing_edges(
    methods_dir: Path,
    source_label: str,
    target_label: str,
    folds: list[str],
    reason: str,
    manifest_path: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    methods_dir = Path(methods_dir)
    rows = []
    created_at = datetime.now(timezone.utc).isoformat()

    for fold in folds:
        source = edge_path(methods_dir, source_label, fold)
        target = edge_path(methods_dir, target_label, fold)
        if not source.exists():
            raise FileNotFoundError(f"Missing source fallback file: {source}")
        if target.exists() and not overwrite:
            action = "kept_existing"
        else:
            shutil.copyfile(source, target)
            action = "copied"
        rows.append({
            "target_label": target_label,
            "source_label": source_label,
            "fold": fold,
            "target_file": str(target),
            "source_file": str(source),
            "fallback_type": "edge_copy",
            "reason": reason,
            "action": action,
            "created_at": created_at,
        })

    new_rows = pd.DataFrame(rows)
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        keep = ~(
            existing["target_label"].eq(target_label)
            & existing["fold"].isin(folds)
        )
        manifest = pd.concat([existing[keep], new_rows], ignore_index=True)
    else:
        manifest = new_rows
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return new_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill missing method edge files from an explicit fallback source."
    )
    parser.add_argument("--methods-dir", required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--folds", nargs="+", default=["full", "I1", "I2"])
    parser.add_argument("--reason", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    methods_dir = Path(args.methods_dir)
    manifest_path = (
        Path(args.manifest) if args.manifest else methods_dir / "method_edge_fallbacks.csv"
    )
    rows = fill_missing_edges(
        methods_dir=methods_dir,
        source_label=args.source_label,
        target_label=args.target_label,
        folds=args.folds,
        reason=args.reason,
        manifest_path=manifest_path,
        overwrite=args.overwrite,
    )
    print(rows.to_string(index=False))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
