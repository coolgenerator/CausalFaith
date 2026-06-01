from __future__ import annotations

import gzip

import pandas as pd

from causalfaith.crossfit import load_ground_truth_edge_sets, load_ground_truth_edges


def test_load_ground_truth_edge_sets_reads_generic_csv_source(tmp_path) -> None:
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    pd.DataFrame(
        [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "C"},
            {"source": "X", "target": "A"},
        ]
    ).to_csv(gt_dir / "toy_edges.csv", index=False)

    edge_sets = load_ground_truth_edge_sets(gt_dir, ["A", "B", "C"], ["toy"])
    pooled = load_ground_truth_edges(gt_dir, ["A", "B", "C"], ["toy"])

    assert edge_sets == {"toy": {("A", "B"), ("B", "C")}}
    assert pooled == {("A", "B"), ("B", "C")}


def write_gzip_text(path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)


def test_string_loader_filters_by_score_and_physical_file(tmp_path) -> None:
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    write_gzip_text(
        gt_dir / "protein.info.txt.gz",
        "#string_protein_id\tpreferred_name\tprotein_size\tannotation\n"
        "9606.P1\tA\t1\t-\n"
        "9606.P2\tB\t1\t-\n"
        "9606.P3\tC\t1\t-\n",
    )
    write_gzip_text(
        gt_dir / "protein.physical.links.txt.gz",
        "protein1 protein2 experimental database textmining combined_score\n"
        "9606.P1 9606.P2 0 0 0 900\n"
        "9606.P1 9606.P3 0 0 0 500\n",
    )
    write_gzip_text(
        gt_dir / "protein.links.txt.gz",
        "protein1 protein2 neighborhood fusion cooccurence coexpression experimental "
        "database textmining combined_score\n"
        "9606.P2 9606.P3 0 0 0 0 0 0 0 950\n",
    )

    edge_sets = load_ground_truth_edge_sets(
        gt_dir,
        ["A", "B", "C"],
        ["stringdb"],
        string_min_score=700,
        string_physical_only=True,
    )

    assert edge_sets == {"stringdb": {("A", "B"), ("B", "A")}}
