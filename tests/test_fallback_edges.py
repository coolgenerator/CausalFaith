from __future__ import annotations

import pandas as pd

from scripts.fill_missing_method_edges import fill_missing_edges
from scripts.run_benchmark_metrics import load_fallback_manifest


def write_edge_file(path) -> None:
    pd.DataFrame(
        [{"source_gene": "A", "target_gene": "B", "score": 0.7}]
    ).to_csv(path, index=False)


def test_fill_missing_edges_copies_all_folds_and_writes_manifest(tmp_path) -> None:
    methods_dir = tmp_path / "methods"
    methods_dir.mkdir()
    for suffix in ["", "_I1", "_I2"]:
        write_edge_file(methods_dir / f"ges_interventional{suffix}_edges.csv")

    manifest_path = methods_dir / "method_edge_fallbacks.csv"
    rows = fill_missing_edges(
        methods_dir=methods_dir,
        source_label="ges_interventional",
        target_label="gies_interventional",
        folds=["full", "I1", "I2"],
        reason="GIES timeout in smoke test",
        manifest_path=manifest_path,
    )

    assert len(rows) == 3
    assert (methods_dir / "gies_interventional_edges.csv").exists()
    assert (methods_dir / "gies_interventional_I1_edges.csv").exists()
    assert (methods_dir / "gies_interventional_I2_edges.csv").exists()

    manifest = pd.read_csv(manifest_path)
    assert set(manifest["fold"]) == {"full", "I1", "I2"}
    assert set(manifest["source_label"]) == {"ges_interventional"}
    assert set(manifest["target_label"]) == {"gies_interventional"}

    loaded = load_fallback_manifest(manifest_path)
    assert loaded["gies_interventional"]["fallback_source"] == "ges_interventional"
    assert loaded["gies_interventional"]["fallback_reason"] == "GIES timeout in smoke test"
