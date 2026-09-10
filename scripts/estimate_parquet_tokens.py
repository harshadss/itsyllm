#!/usr/bin/env python3
"""Estimate training tokens in one or more Parquet files without writing a dataset.

The count matches ``package_parquet_dataset.py``: each non-empty document is
encoded as ``BOS + text tokens + EOS``.

Example:
    uv run python scripts/estimate_parquet_tokens.py \
      --tokenizer artifacts/tokenizers/example data/train-*.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Input Parquet files.")
    parser.add_argument("--tokenizer", required=True, type=Path, help="SentencePiece .model file or its artifact directory.")
    parser.add_argument("--text-column", default="text", help="Parquet text column (default: text).")
    parser.add_argument("--batch-rows", type=int, default=8192, help="Parquet rows and tokenization inputs per batch (default: 8192).")
    parser.add_argument(
        "--tokenizer-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="SentencePiece encoding threads (default: available CPU count).",
    )
    args = parser.parse_args()
    if args.batch_rows <= 0 or args.tokenizer_threads <= 0:
        parser.error("--batch-rows and --tokenizer-threads must be positive")
    return args


def tokenizer_model_path(path: Path) -> Path:
    return path / "tokenizer.model" if path.is_dir() else path


def count_file(
    path: Path,
    tokenizer: spm.SentencePieceProcessor,
    text_column: str,
    batch_rows: int,
    tokenizer_threads: int,
) -> tuple[int, int, int]:
    """Return document, token, and skipped-empty counts for one Parquet file."""
    parquet = pq.ParquetFile(path)
    if text_column not in parquet.schema_arrow.names:
        raise ValueError(f"Column {text_column!r} is missing from {path}")

    document_count = 0
    token_count = 0
    skipped_empty = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=[text_column]):
        texts = [text for text in batch.column(0).to_pylist() if isinstance(text, str) and text.strip()]
        skipped_empty += batch.num_rows - len(texts)
        if not texts:
            continue

        encoded_documents = tokenizer.encode(
            texts,
            out_type=int,
            add_bos=True,
            add_eos=True,
            num_threads=tokenizer_threads,
        )
        document_count += len(encoded_documents)
        token_count += sum(len(document) for document in encoded_documents)

    return document_count, token_count, skipped_empty


def main() -> None:
    args = parse_args()
    input_paths = [path.resolve() for path in args.paths]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input Parquet file(s) not found:\n" + "\n".join(missing))

    model_path = tokenizer_model_path(args.tokenizer).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    if tokenizer.bos_id() < 0 or tokenizer.eos_id() < 0:
        raise ValueError("Tokenizer must define both BOS and EOS IDs.")

    total_documents = 0
    total_tokens = 0
    total_skipped = 0
    for path in input_paths:
        documents, tokens, skipped = count_file(
            path,
            tokenizer,
            args.text_column,
            args.batch_rows,
            args.tokenizer_threads,
        )
        total_documents += documents
        total_tokens += tokens
        total_skipped += skipped
        print(f"{path}: {documents:,} documents, {tokens:,} tokens, {skipped:,} empty or null rows skipped")

    print()
    print(f"Total documents: {total_documents:,}")
    print(f"Total tokens (including BOS/EOS): {total_tokens:,}")
    print(f"Empty or null rows skipped: {total_skipped:,}")


if __name__ == "__main__":
    main()
