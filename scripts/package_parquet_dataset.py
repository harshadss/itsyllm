#!/usr/bin/env python3
"""Tokenize Parquet documents once and write a memory-mappable training dataset.

Example:
    uv run python scripts/package_parquet_dataset.py \
      --tokenizer artifacts/tokenizers/example \
      --output artifacts/datasets/example.bin data/*.parquet
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Input Parquet files.")
    parser.add_argument("--tokenizer", required=True, type=Path, help="SentencePiece .model file or its artifact directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output token file, conventionally ending in .bin.")
    parser.add_argument("--text-column", default="text", help="Parquet text column (default: text).")
    parser.add_argument("--batch-rows", type=int, default=8192, help="Parquet rows per read batch (default: 8192).")
    parser.add_argument("--max-documents", type=int, help="Stop after this many non-empty documents.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dataset.")
    args = parser.parse_args()
    if args.batch_rows <= 0:
        parser.error("--batch-rows must be positive")
    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents must be positive")
    return args


def tokenizer_model_path(path: Path) -> Path:
    return path / "tokenizer.model" if path.is_dir() else path


def sidecar_paths(output: Path) -> tuple[Path, Path]:
    stem = output.with_suffix("")
    return stem.with_name(stem.name + ".document_offsets.bin"), stem.with_suffix(".json")


def prepare_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists. Use --overwrite to replace it:\n{names}")
    for path in existing:
        path.unlink()


def write_tokens(handle: object, token_ids: list[int]) -> None:
    values = np.asarray(token_ids, dtype="<i2")
    handle.write(values.tobytes())  # type: ignore[attr-defined]


def package(args: argparse.Namespace) -> None:
    input_paths = [path.resolve() for path in args.paths]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input Parquet file(s) not found:\n" + "\n".join(missing))

    model_path = tokenizer_model_path(args.tokenizer).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    bos_id, eos_id = tokenizer.bos_id(), tokenizer.eos_id()
    if bos_id < 0 or eos_id < 0:
        raise ValueError("Tokenizer must define both BOS and EOS IDs.")
    if tokenizer.vocab_size() > np.iinfo(np.int16).max:
        raise ValueError("Tokenizer vocabulary exceeds the signed int16 dataset format.")

    output = args.output.resolve()
    offsets_path, metadata_path = sidecar_paths(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_outputs([output, offsets_path, metadata_path], args.overwrite)
    partial_output = output.with_suffix(output.suffix + ".partial")
    partial_offsets = offsets_path.with_suffix(offsets_path.suffix + ".partial")
    for path in (partial_output, partial_offsets):
        path.unlink(missing_ok=True)

    document_count = 0
    token_count = 0
    skipped_empty = 0
    try:
        with partial_output.open("wb") as token_file, partial_offsets.open("wb") as offsets_file:
            for input_path in input_paths:
                parquet = pq.ParquetFile(input_path)
                if args.text_column not in parquet.schema_arrow.names:
                    raise ValueError(f"Column {args.text_column!r} is missing from {input_path}")
                for batch in parquet.iter_batches(batch_size=args.batch_rows, columns=[args.text_column]):
                    for text in batch.column(0).to_pylist():
                        if not isinstance(text, str) or not text.strip():
                            skipped_empty += 1
                            continue
                        token_ids = [bos_id, *tokenizer.encode(text, out_type=int), eos_id]
                        np.asarray([token_count], dtype="<u8").tofile(offsets_file)
                        write_tokens(token_file, token_ids)
                        document_count += 1
                        token_count += len(token_ids)
                        if document_count % 10_000 == 0:
                            print(f"Packed {document_count:,} documents and {token_count:,} tokens.")
                        if args.max_documents is not None and document_count >= args.max_documents:
                            break
                    if args.max_documents is not None and document_count >= args.max_documents:
                        break
                if args.max_documents is not None and document_count >= args.max_documents:
                    break
            np.asarray([token_count], dtype="<u8").tofile(offsets_file)

        if document_count == 0:
            raise RuntimeError("No non-empty documents were found.")
        partial_output.replace(output)
        partial_offsets.replace(offsets_path)
        metadata = {
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "format": "little-endian signed int16 token IDs",
            "token_file": output.name,
            "document_offsets_file": offsets_path.name,
            "token_dtype": "<i2",
            "offset_dtype": "<u8",
            "token_count": token_count,
            "document_count": document_count,
            "bos_id": bos_id,
            "eos_id": eos_id,
            "tokenizer_model": str(model_path),
            "tokenizer_vocab_size": tokenizer.vocab_size(),
            "input_parquet_files": [str(path) for path in input_paths],
            "text_column": args.text_column,
            "skipped_empty_or_null": skipped_empty,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {output}")
        print(f"Wrote {offsets_path}")
        print(f"Wrote {metadata_path}")
    except Exception:
        partial_output.unlink(missing_ok=True)
        partial_offsets.unlink(missing_ok=True)
        raise


def main() -> None:
    package(parse_args())


if __name__ == "__main__":
    main()
