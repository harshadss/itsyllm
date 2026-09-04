#!/usr/bin/env python3
"""Inspect SentencePiece tokenization for text supplied through standard input.

Example:
    printf 'नमस्ते, world!' | uv run python scripts/test_tokenizer.py \
      artifacts/tokenizers/sangraha_verified_sample_unigram_v1/tokenizer.model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to a SentencePiece .model file.")
    parser.add_argument("--no-bos", action="store_true", help="Do not prepend the model's BOS token.")
    parser.add_argument("--no-eos", action="store_true", help="Do not append the model's EOS token.")
    return parser.parse_args()


def boundary_token(processor: spm.SentencePieceProcessor, token_id: int, name: str) -> tuple[int, str]:
    if token_id < 0:
        raise ValueError(f"This model has no {name} token; use --no-{name.lower()}.")
    return token_id, processor.id_to_piece(token_id)


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Tokenizer model not found: {args.model}")
    text = sys.stdin.read()
    if not text:
        raise ValueError("No input received on stdin.")

    processor = spm.SentencePieceProcessor(model_file=str(args.model))
    pieces = processor.encode(text, out_type=str)
    ids = processor.encode(text, out_type=int)
    tokens: list[tuple[int, str]] = []
    if not args.no_bos:
        tokens.append(boundary_token(processor, processor.bos_id(), "BOS"))
    tokens.extend(zip(ids, pieces, strict=True))
    if not args.no_eos:
        tokens.append(boundary_token(processor, processor.eos_id(), "EOS"))

    print(f"Model: {args.model}")
    print(f"Input: {text!r}")
    print(f"Token count: {len(tokens)}")
    print("index\tid\tpiece")
    for index, (token_id, piece) in enumerate(tokens):
        print(f"{index}\t{token_id}\t{piece!r}")
    print("IDs:", [token_id for token_id, _ in tokens])
    print("Pieces:", [piece for _, piece in tokens])


if __name__ == "__main__":
    main()
