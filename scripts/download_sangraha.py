#!/usr/bin/env python3
"""Materialize Sangraha verified samples (or complete raw Parquet shards) locally.

Examples:
    uv run python scripts/download_sangraha.py
    uv run python scripts/download_sangraha.py --languages eng hin --rows 50000
    uv run python scripts/download_sangraha.py --full --languages mar
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

REPO_ID = "ai4bharat/sangraha"
CONFIG = "verified"
DEFAULT_LANGUAGES = ("eng", "hin", "mar")
DEFAULT_ROOT = Path("/home/harshad/Workspace/Learning/DeepLearning/Datasets/ai4bharat/sangraha")
EXPECTED_COLUMNS = ("doc_id", "type", "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=DEFAULT_LANGUAGES, default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--rows", type=int, default=100_000, help="Rows per language in sample mode (default: 100000).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT, help=f"Dataset root (default: {DEFAULT_ROOT}).")
    parser.add_argument("--revision", default="main", help="Hugging Face revision to resolve (default: main).")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Parquet write batch size (default: 10000).")
    parser.add_argument("--full", action="store_true", help="Download every original Parquet shard instead of a sample.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing completed output.")
    args = parser.parse_args()
    if args.rows <= 0 or args.batch_size <= 0:
        parser.error("--rows and --batch-size must be positive")
    return args


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_revision(revision: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(REPO_ID, revision=revision).sha


def ensure_target(path: Path, manifest: Path, overwrite: bool) -> bool:
    """Prepare a target. Return False when an already-complete target is reusable."""
    if path.exists() or manifest.exists():
        if not overwrite:
            if path.exists() and manifest.exists():
                print(f"Skipping completed output: {path}")
                return False
            raise FileExistsError(
                f"Refusing incomplete existing output for {path}. Remove it or use --overwrite."
            )
        path.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def sample_paths(root: Path, language: str, rows: int) -> tuple[Path, Path]:
    directory = root / "samples" / CONFIG / language
    stem = f"sangraha-{CONFIG}-{language}-first-{rows}"
    return directory / f"{stem}.parquet", directory / f"{stem}.manifest.json"


def write_sample(language: str, rows: int, batch_size: int, root: Path, revision: str, overwrite: bool) -> None:
    output, manifest = sample_paths(root, language, rows)
    if not ensure_target(output, manifest, overwrite):
        return
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)

    # Import after cache variables are set so streaming leaves no durable HF cache.
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"Streaming {rows:,} rows: {CONFIG}/{language} -> {output}")
    dataset = load_dataset(
        REPO_ID,
        data_dir=f"{CONFIG}/{language}",
        # Limiting data_dir to one language makes the Parquet loader expose its
        # single split as "train" (rather than the repository-level language name).
        split="train",
        streaming=True,
        revision=revision,
    )
    written = 0
    writer: pq.ParquetWriter | None = None
    try:
        batch: list[dict[str, Any]] = []
        for row in dataset:
            batch.append({column: row.get(column) for column in EXPECTED_COLUMNS})
            if len(batch) == batch_size or written + len(batch) == rows:
                table = pa.Table.from_pylist(batch, schema=pa.schema([(c, pa.string()) for c in EXPECTED_COLUMNS]))
                if writer is None:
                    writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
                writer.write_table(table)
                written += len(batch)
                print(f"  {language}: {written:,}/{rows:,} rows", file=sys.stderr)
                batch.clear()
            if written == rows:
                break
        if written != rows:
            raise RuntimeError(f"Only {written:,} rows were available for {CONFIG}/{language}; expected {rows:,}.")
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    partial.replace(output)
    write_json(manifest, {
        "created_at": now(), "dataset": REPO_ID, "revision": revision,
        "config": CONFIG, "split": language, "selection": "first",
        "requested_rows": rows, "written_rows": written,
        "columns": list(EXPECTED_COLUMNS), "format": "parquet", "path": str(output),
    })


def iter_parquet_paths(language: str, revision: str) -> Iterable[str]:
    from huggingface_hub import HfApi

    prefix = f"{CONFIG}/{language}/"
    for entry in HfApi().list_repo_tree(REPO_ID, repo_type="dataset", revision=revision, path_in_repo=prefix, recursive=True):
        path = getattr(entry, "path", "")
        if path.startswith(prefix) and path.endswith(".parquet"):
            yield path


def download_file(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(destination)


def download_full(language: str, root: Path, revision: str, overwrite: bool) -> None:
    directory = root / "raw" / CONFIG / language
    manifest = directory / "download.manifest.json"
    files = sorted(iter_parquet_paths(language, revision))
    if not files:
        raise RuntimeError(f"No Parquet shards found for {CONFIG}/{language} at {revision}.")
    directory.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for remote_path in files:
        output = directory / Path(remote_path).name
        if output.exists() and not overwrite:
            print(f"Skipping existing shard: {output}")
        else:
            print(f"Downloading {remote_path} -> {output}")
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{remote_path}"
            download_file(url, output)
        downloaded.append(output.name)
    write_json(manifest, {
        "created_at": now(), "dataset": REPO_ID, "revision": revision,
        "config": CONFIG, "split": language, "format": "parquet", "files": downloaded,
    })


def main() -> None:
    args = parse_args()
    resolved_revision = resolve_revision(args.revision)
    print(f"Using {REPO_ID} revision {resolved_revision}")

    # Streaming may use package-managed temporary files; keep them isolated and remove them at exit.
    with tempfile.TemporaryDirectory(prefix="sangraha-hf-") as cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        for language in args.languages:
            if args.full:
                download_full(language, args.output_root, resolved_revision, args.overwrite)
            else:
                write_sample(language, args.rows, args.batch_size, args.output_root, resolved_revision, args.overwrite)


if __name__ == "__main__":
    main()
