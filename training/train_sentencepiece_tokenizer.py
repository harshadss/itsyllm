#!/usr/bin/env python3
"""Train a SentencePiece tokenizer directly from one or more Parquet text columns.

The default config trains the first joint English/Hindi/Marathi tokenizer:
    uv run python training/train_sentencepiece_tokenizer.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
import sentencepiece as spm

DEFAULT_CONFIG = Path("configs/tokenizers/sangraha_verified_sample_unigram_v1.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"Training TOML (default: {DEFAULT_CONFIG}).")
    parser.add_argument("--output-dir", type=Path, help="Override the artifact directory from the config.")
    parser.add_argument("--max-sentences", type=int, help="Cap input sentences for a quick smoke run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing artifact directory.")
    args = parser.parse_args()
    if args.max_sentences is not None and args.max_sentences <= 0:
        parser.error("--max-sentences must be positive")
    return args


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    for section in ("metadata", "corpus", "trainer", "output"):
        if section not in config:
            raise ValueError(f"Missing [{section}] in {path}")
    return config


class ParquetSentenceIterator:
    """Yield non-empty text values in bounded PyArrow record batches."""

    def __init__(self, paths: list[Path], column: str, batch_rows: int, limit: int, max_sentence_length: int) -> None:
        self.paths = paths
        self.column = column
        self.batch_rows = batch_rows
        self.limit = limit
        self.max_sentence_length = max_sentence_length
        self.yielded = 0
        self.empty_or_null = 0
        self.too_long = 0

    def __iter__(self) -> Iterator[str]:
        for path in self.paths:
            parquet = pq.ParquetFile(path)
            if self.column not in parquet.schema_arrow.names:
                raise ValueError(f"Column {self.column!r} is missing from {path}")
            for batch in parquet.iter_batches(batch_size=self.batch_rows, columns=[self.column]):
                for text in batch.column(0).to_pylist():
                    if not isinstance(text, str) or not text.strip():
                        self.empty_or_null += 1
                        continue
                    if len(text.encode("utf-8")) > self.max_sentence_length:
                        self.too_long += 1
                        continue
                    self.yielded += 1
                    yield text
                    if self.yielded >= self.limit:
                        return


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def train(config_path: Path, output_override: Path | None, max_sentences: int | None, overwrite: bool) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    corpus = config["corpus"]
    trainer = dict(config["trainer"])
    output = (output_override or Path(config["output"]["directory"])).resolve()
    model_basename = config["output"]["model_basename"]
    paths = [Path(value) for value in corpus["paths"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input Parquet file(s):\n" + "\n".join(missing))

    configured_limit = int(corpus["input_sentence_size"])
    sentence_limit = min(configured_limit, max_sentences) if max_sentences else configured_limit
    if sentence_limit <= 0:
        raise ValueError("corpus.input_sentence_size must be positive")
    prepare_output(output, overwrite)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    temporary_prefix = temporary_dir / model_basename
    iterator = ParquetSentenceIterator(
        paths,
        corpus["text_column"],
        int(corpus["batch_rows"]),
        sentence_limit,
        int(trainer["max_sentence_length"]),
    )

    trainer.update({
        "model_prefix": str(temporary_prefix),
        "input_sentence_size": sentence_limit,
        "shuffle_input_sentence": bool(corpus["shuffle_input_sentence"]),
    })
    try:
        print(f"Training {config['metadata']['name']} on up to {sentence_limit:,} Parquet rows.")
        # The Python bindings require an Iterator object, not merely an Iterable.
        spm.SentencePieceTrainer.train(sentence_iterator=iter(iterator), **trainer)
        if iterator.yielded == 0:
            raise RuntimeError("No non-empty text rows were found in the configured corpus.")

        model_path = temporary_dir / f"{model_basename}.model"
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        expected_vocab = int(trainer["vocab_size"])
        if processor.get_piece_size() != expected_vocab:
            raise RuntimeError(f"Expected {expected_vocab} pieces, got {processor.get_piece_size()}.")
        for field in ("unk", "bos", "eos", "pad"):
            piece = trainer[f"{field}_piece"]
            token_id = int(trainer[f"{field}_id"])
            if processor.piece_to_id(piece) != token_id:
                raise RuntimeError(f"Expected {piece} to have ID {token_id}.")
        for symbol in trainer["user_defined_symbols"]:
            # SentencePiece can add the normal whitespace marker before a
            # symbol. The symbol itself must nevertheless be one indivisible piece.
            if processor.encode(symbol, out_type=str).count(symbol) != 1:
                raise RuntimeError(f"Protocol token did not remain atomic: {symbol}")
        if processor.unk_id() in processor.encode("byte fallback probe: \U0010ffff", out_type=int):
            raise RuntimeError("Byte fallback did not prevent an unknown token for unseen Unicode.")

        manifest = {
            "created_at": utc_now(),
            "name": config["metadata"]["name"],
            "description": config["metadata"]["description"],
            "config_path": str(config_path),
            "input_parquet_files": [str(path) for path in paths],
            "text_column": corpus["text_column"],
            "requested_sentence_limit": sentence_limit,
            "yielded_sentences": iterator.yielded,
            "skipped_empty_or_null": iterator.empty_or_null,
            "skipped_too_long": iterator.too_long,
            "sentencepiece_version": spm.__version__,
            "trainer": trainer,
            "vocab_size": processor.get_piece_size(),
        }
        (temporary_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copy2(config_path, temporary_dir / "config.toml")
        temporary_dir.replace(output)
        print(f"Wrote tokenizer artifacts to {output}")
        return output
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    train(args.config, args.output_dir, args.max_sentences, args.overwrite)


if __name__ == "__main__":
    main()
