#!/usr/bin/env python3
"""Stream a shuffled, score-filtered Ultra-FineWeb sample to local Parquet.

The source corpus is never materialized locally.  The sample is randomized with
a bounded streaming shuffle, so it is suitable for training but is not an
exact uniform sample of every qualifying source row.

Examples:
    uv run python scripts/download_ultra_fineweb.py
    uv run python scripts/download_ultra_fineweb.py --rows 100000 --shuffle-buffer-size 250000
    uv run python scripts/download_ultra_fineweb.py --score-threshold 0.8 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ID = "openbmb/Ultra-FineWeb"
SPLIT = "en"
DEFAULT_ROWS = 3_000_000
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_SHUFFLE_BUFFER_SIZE = 100_000
DEFAULT_ROOT = Path("/home/harshad/Workspace/Learning/DeepLearning/Datasets/openbmb/ultra-fineweb")
SOURCE_TEXT_COLUMN = "content"
OUTPUT_COLUMNS = ("text", "score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help=f"Rows to write (default: {DEFAULT_ROWS}).")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Keep rows with score strictly greater than this value (default: {DEFAULT_SCORE_THRESHOLD}).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Streaming-shuffle seed (default: 42).")
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER_SIZE,
        help=(
            "Maximum number of qualifying rows retained for streaming shuffle "
            f"(default: {DEFAULT_SHUFFLE_BUFFER_SIZE}). Larger values improve mixing but use more RAM."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Dataset root (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument("--revision", default="main", help="Hugging Face revision to resolve (default: main).")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Parquet write batch size (default: 10000).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing completed output.")
    args = parser.parse_args()
    if args.rows <= 0 or args.batch_size <= 0 or args.shuffle_buffer_size <= 0:
        parser.error("--rows, --batch-size, and --shuffle-buffer-size must be positive")
    if not math.isfinite(args.score_threshold) or not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be a finite value from 0 through 1")
    return args


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_revision(revision: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(REPO_ID, revision=revision).sha


def sample_paths(root: Path, threshold: float, rows: int) -> tuple[Path, Path]:
    threshold_label = format(threshold, "g").replace(".", "p")
    directory = root / "samples" / SPLIT
    stem = f"ultra-fineweb-{SPLIT}-score-gt-{threshold_label}-shuffled-{rows}"
    return directory / f"{stem}.parquet", directory / f"{stem}.manifest.json"


def ensure_target(path: Path, manifest: Path, overwrite: bool) -> bool:
    """Prepare a target. Return False when a completed target is reusable."""
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


def score_exceeds_threshold(row: dict[str, Any], threshold: float) -> bool:
    """Parse the source's string-typed score and apply a numeric threshold."""
    try:
        score = float(row["score"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(score) and score > threshold


def write_sample(args: argparse.Namespace, revision: str) -> None:
    output, manifest = sample_paths(args.output_root, args.score_threshold, args.rows)
    if not ensure_target(output, manifest, args.overwrite):
        return
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)

    # These imports occur after the temporary cache locations have been set.
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(
        f"Streaming {args.rows:,} shuffled rows: {REPO_ID}/{SPLIT} "
        f"where score > {args.score_threshold:g} -> {output}",
        flush=True,
    )
    dataset = load_dataset(
        REPO_ID,
        split=SPLIT,
        streaming=True,
        revision=revision,
    ).select_columns([SOURCE_TEXT_COLUMN, "score"])
    # The repository's source Parquet schema stores score as a string. A
    # numeric Arrow predicate cannot bind to that column, so filter lazily in
    # Python before filling the bounded shuffle buffer.
    dataset = dataset.filter(score_exceeds_threshold, fn_kwargs={"threshold": args.score_threshold})
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    schema = pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("score", pa.float32()),
        ]
    )
    written = 0
    skipped_empty = 0
    writer: pq.ParquetWriter | None = None
    try:
        batch: list[dict[str, Any]] = []
        for row in dataset:
            raw_score = row.get("score")
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                raise ValueError("A score-filtered row did not contain a numeric score.") from None
            if not math.isfinite(score) or score <= args.score_threshold:
                raise ValueError("A score-filtered row did not satisfy the configured score threshold.")

            content = row.get(SOURCE_TEXT_COLUMN)
            if not isinstance(content, str) or not content.strip():
                skipped_empty += 1
                continue
            batch.append(
                {
                    "text": content,
                    "score": score,
                }
            )
            if len(batch) == args.batch_size or written + len(batch) == args.rows:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
                writer.write_table(table)
                written += len(batch)
                print(f"  {written:,}/{args.rows:,} rows", file=sys.stderr, flush=True)
                batch.clear()
            if written == args.rows:
                break
        if written != args.rows:
            raise RuntimeError(
                f"Only {written:,} qualifying non-empty rows were available; expected {args.rows:,}."
            )
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    print("Finalizing Parquet file...", flush=True)
    partial.replace(output)
    write_json(
        manifest,
        {
            "created_at": now(),
            "dataset": REPO_ID,
            "revision": revision,
            "split": SPLIT,
            "selection": "streaming bounded-buffer shuffle after score filter; not exact global-uniform sampling",
            "score_filter": {"column": "score", "operator": ">", "value": args.score_threshold},
            "shuffle": {"seed": args.seed, "buffer_size": args.shuffle_buffer_size},
            "requested_rows": args.rows,
            "written_rows": written,
            "skipped_empty_or_null": skipped_empty,
            "columns": list(OUTPUT_COLUMNS),
            "source_text_column": SOURCE_TEXT_COLUMN,
            "format": "parquet",
            "path": str(output),
        },
    )
    print(f"Completed {written:,} rows -> {output}", flush=True)


def main() -> None:
    args = parse_args()
    # Isolate package-managed temporary files so streaming does not leave a
    # durable Hugging Face cache beside the requested output.
    with tempfile.TemporaryDirectory(prefix="ultra-fineweb-hf-") as cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")
        os.environ["HF_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        # Kept for older huggingface_hub releases used by some environments.
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        revision = resolve_revision(args.revision)
        print(f"Using {REPO_ID} revision {revision}", flush=True)
        write_sample(args, revision)
        print("Cleaning up temporary Hugging Face cache...", flush=True)


if __name__ == "__main__":
    main()
