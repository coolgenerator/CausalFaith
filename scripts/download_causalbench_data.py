#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen


DATASET_CHOICES = ("k562", "rpe1", "all")
SOURCE_CHOICES = ("scverse", "figshare-raw")
H5AD_URLS = {
    "scverse": {
        "k562": (
            "https://exampledata.scverse.org/pertpy/replogle_2022_k562_essential.h5ad",
            "k562.h5ad",
        ),
        "rpe1": (
            "https://exampledata.scverse.org/pertpy/replogle_2022_rpe1.h5ad",
            "rpe1.h5ad",
        ),
    },
    "figshare-raw": {
        "k562": ("https://ndownloader.figshare.com/files/35773219", "k562_raw.h5ad"),
        "rpe1": ("https://ndownloader.figshare.com/files/35775606", "rpe1_raw.h5ad"),
    },
}
MIN_H5AD_BYTES = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and cache CausalBench/Replogle data into a project data directory."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/causalbench"))
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="all")
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default="scverse",
        help="Data source for raw AnnData downloads. scverse is smaller and automation-friendly.",
    )
    parser.add_argument(
        "--make-npz",
        action="store_true",
        help="Also build CausalBench processed .npz files used by causalbench_run.",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="When building .npz files, apply CausalBench's strong-perturbation filter.",
    )
    parser.add_argument(
        "--with-evaluation-resources",
        action="store_true",
        help="Also download/build benchmark evaluation resources such as CORUM and StringDB.",
    )
    return parser.parse_args()


def require_causalbench() -> None:
    try:
        import causalscbench  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "CausalBench is not installed. Run `pip install causalbench` first."
        ) from exc


def download_url_if_needed(url: str, path: Path, min_bytes: int = MIN_H5AD_BYTES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= min_bytes:
        return path
    if path.exists():
        path.unlink()

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"Downloading {url} -> {path}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, tmp_path.open("wb") as handle:
        expected = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        next_report = 100 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                if expected:
                    pct = 100.0 * downloaded / expected
                    print(f"  downloaded {downloaded / 1024**2:.0f} MiB / {expected / 1024**2:.0f} MiB ({pct:.1f}%)")
                else:
                    print(f"  downloaded {downloaded / 1024**2:.0f} MiB")
                next_report += 100 * 1024 * 1024

    size = tmp_path.stat().st_size
    if size < min_bytes:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is too small ({size} bytes): {path}")
    tmp_path.replace(path)
    return path


def download_h5ad(data_dir: Path, dataset: str, source: str) -> list[Path]:
    paths = []
    if dataset in {"k562", "all"}:
        url, filename = H5AD_URLS[source]["k562"]
        paths.append(download_url_if_needed(url, data_dir / filename))
    if dataset in {"rpe1", "all"}:
        url, filename = H5AD_URLS[source]["rpe1"]
        paths.append(download_url_if_needed(url, data_dir / filename))
    return paths


def build_npz(data_dir: Path, apply_filter: bool) -> list[Path]:
    from causalscbench.data_access.create_dataset import CreateDataset

    return [Path(path) for path in CreateDataset(str(data_dir), apply_filter).load()]


def build_evaluation_resources(data_dir: Path, dataset: str) -> None:
    from causalscbench.data_access.create_evaluation_datasets import CreateEvaluationDatasets

    dataset_names = []
    if dataset in {"k562", "all"}:
        dataset_names.append("weissmann_k562")
    if dataset in {"rpe1", "all"}:
        dataset_names.append("weissmann_rpe1")
    for dataset_name in dataset_names:
        CreateEvaluationDatasets(str(data_dir), dataset_name).load()


def main() -> None:
    args = parse_args()
    require_causalbench()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    h5ad_paths = download_h5ad(args.data_dir, args.dataset, args.source)
    print("Downloaded or found raw AnnData files:")
    for path in h5ad_paths:
        print(f"  {path}")

    if args.make_npz:
        npz_paths = build_npz(args.data_dir, args.filter)
        print("Built or found CausalBench processed .npz files:")
        for path in npz_paths:
            print(f"  {path}")

    if args.with_evaluation_resources:
        build_evaluation_resources(args.data_dir, args.dataset)
        print(f"Downloaded or found evaluation resources under {args.data_dir}")


if __name__ == "__main__":
    main()
