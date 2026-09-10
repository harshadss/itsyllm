#!/usr/bin/env python3
"""Materialize samples from the English Ultra-FineWeb-L3 train splits.

The local Parquet files contain only a ``text`` column, normalized from the
dataset's ``content`` column, so they work directly with the packaging and
token-estimation scripts.

Examples:
    uv run python scripts/download_ultra_fineweb_l3.py
    uv run python scripts/download_ultra_fineweb_l3.py --subsets Ultra-FineWeb-L3-en-QA --rows 50000
    uv run python scripts/download_ultra_fineweb_l3.py --keep-uid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ID = "openbmb/Ultra-FineWeb-L3"
# These are the short subset names used in project commands.  The repository's
# current config names add "-Synthetic"; keep that implementation detail here.
SUBSETS = {
    "Ultra-FineWeb-L3-en-QA": "Ultra-FineWeb-L3-en-QA-Synthetic",
    "Ultra-FineWeb-L3-en-Multi-Style": "Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
}
DEFAULT_SUBSETS = tuple(SUBSETS)
DEFAULT_ROWS = 1_000_000
DEFAULT_ROOT = Path("/home/harshad/Workspace/Learning/DeepLearning/Datasets/openbmb/ultra-fineweb-l3")
TEXT_COLUMN = "content"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=DEFAULT_SUBSETS,
        default=list(DEFAULT_SUBSETS),
        help="English subsets to sample (default: both).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=f"Rows per subset in sample mode (default: {DEFAULT_ROWS}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Dataset root (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument("--revision", default="main", help="Hugging Face revision to resolve (default: main).")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Parquet write batch size (default: 10000).")
    parser.add_argument(
        "--keep-uid",
        action="store_true",
        help="Also write the source uid column for provenance or deduplication.",
    )
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
            raise FileExistsError(f"Refusing incomplete existing output for {path}. Remove it or use --overwrite.")
        path.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def sample_paths(root: Path, subset: str, rows: int) -> tuple[Path, Path]:
    directory = root / "samples" / subset
    stem = f"{subset}-first-{rows}"
    return directory / f"{stem}.parquet", directory / f"{stem}.manifest.json"


def write_sample(
    subset: str,
    rows: int,
    batch_size: int,
    root: Path,
    revision: str,
    keep_uid: bool,
    overwrite: bool,
) -> None:
    output, manifest = sample_paths(root, subset, rows)
    if not ensure_target(output, manifest, overwrite):
        return
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)

    # Import after cache variables are set so streaming leaves no durable HF cache.
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq

    config = SUBSETS[subset]
    print(f"Streaming {rows:,} rows: {config}/train -> {output}", flush=True)
    dataset = load_dataset(REPO_ID, config, split="train", streaming=True, revision=revision)
    columns = ("uid", "text") if keep_uid else ("text",)
    schema = pa.schema([(column, pa.string()) for column in columns])
    written = 0
    skipped_empty = 0
    writer: pq.ParquetWriter | None = None
    try:
        batch: list[dict[str, str]] = []
        for row in dataset:
            content = row.get(TEXT_COLUMN)
            if not isinstance(content, str) or not content.strip():
                skipped_empty += 1
                continue
            document = {"text": content}
            if keep_uid:
                uid = row.get("uid")
                if not isinstance(uid, str):
                    raise ValueError(f"Expected a string uid in {config}/train.")
                document["uid"] = uid
            batch.append(document)
            if len(batch) == batch_size or written + len(batch) == rows:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
                writer.write_table(table)
                written += len(batch)
                print(f"  {subset}: {written:,}/{rows:,} rows", file=sys.stderr, flush=True)
                batch.clear()
            if written == rows:
                break
        if written != rows:
            raise RuntimeError(f"Only {written:,} non-empty rows were available for {config}/train; expected {rows:,}.")
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    print(f"Finalizing {subset} Parquet file...", flush=True)
    partial.replace(output)
    write_json(
        manifest,
        {
            "created_at": now(),
            "dataset": REPO_ID,
            "revision": revision,
            "config": config,
            "split": "train",
            "selection": "first non-empty rows",
            "requested_rows": rows,
            "written_rows": written,
            "skipped_empty_or_null": skipped_empty,
            "columns": list(columns),
            "source_text_column": TEXT_COLUMN,
            "format": "parquet",
            "path": str(output),
        },
    )
    print(f"Completed {subset}: {written:,} rows -> {output}", flush=True)


def main() -> None:
    args = parse_args()
    # Streaming may use package-managed temporary files; keep them isolated and remove them at exit.
    with tempfile.TemporaryDirectory(prefix="ultra-fineweb-l3-hf-") as cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        resolved_revision = resolve_revision(args.revision)
        print(f"Using {REPO_ID} revision {resolved_revision}", flush=True)
        for index, subset in enumerate(args.subsets, start=1):
            print(f"Starting subset {index}/{len(args.subsets)}: {subset}", flush=True)
            write_sample(
                subset,
                args.rows,
                args.batch_size,
                args.output_root,
                resolved_revision,
                args.keep_uid,
                args.overwrite,
            )
        print("Cleaning up temporary Hugging Face cache...", flush=True)
    print("All requested subsets completed.", flush=True)


if __name__ == "__main__":
    main()
