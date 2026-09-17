#!/usr/bin/env python3
"""Render normalized SFT JSONL, chunk it, and write memory-mappable artifacts.

Each JSONL record must contain an ordered ``messages`` array.  Messages are
rendered with the project's role vocabulary, with a supervised EOS token after
every assistant response. Long conversations are split into independent, overlapping
chunks.  Every output chunk has ``BOS`` and ``EOS`` boundaries, an offset entry
that makes it a separate FlexAttention block, and a per-token SFT loss mask.

Examples:
    uv run python scripts/package_sft_jsonl_dataset.py \
      --tokenizer artifacts/tokenizers/sangraha_ultrafineweb_l3_en_indic_unigram_v1 \
      /path/to/indic-align.jsonl /path/to/ultradata-sft.jsonl

    uv run python scripts/package_sft_jsonl_dataset.py \
      --tokenizer artifacts/tokenizers/sangraha_ultrafineweb_l3_en_indic_unigram_v1 \
      --context-length 4096 --overlap 256 --output /path/to/sft.bin data/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm


DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_OVERLAP = 256
DEFAULT_SYSTEM_PROMPT = "You are a small language model assistant trained by AbleEdge AI."
DEFAULT_OUTPUT = Path("/home/harshad/Workspace/Learning/DeepLearning/Datasets/itsyllm/sft/sft.bin")
ROLE_MARKERS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
THINKING_MARKERS = ("<|thinking|>", "<|end_thinking|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Input JSONL files from one or more SFT downloaders.")
    parser.add_argument("--tokenizer", required=True, type=Path, help="SentencePiece .model file or its artifact directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output token file (default: {DEFAULT_OUTPUT}).")
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help=f"Maximum tokens per chunk including BOS/EOS (default: {DEFAULT_CONTEXT_LENGTH}).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Repeated content tokens at the start of each non-first chunk (default: {DEFAULT_OVERLAP}).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt injected ahead of every source conversation.",
    )
    parser.add_argument("--no-system-prompt", action="store_true", help="Do not inject a system prompt.")
    parser.add_argument("--max-documents", type=int, help="Stop after this many valid source JSONL records.")
    parser.add_argument("--flush-every", type=int, default=1_000, help="Flush artifacts after this many chunks (default: 1000).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output artifacts.")
    args = parser.parse_args()
    if args.context_length < 3:
        parser.error("--context-length must allow BOS, at least one content token, and EOS (minimum: 3)")
    if not 0 <= args.overlap < args.context_length - 2:
        parser.error("--overlap must be non-negative and smaller than context length minus BOS/EOS")
    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents must be positive")
    if args.flush_every <= 0:
        parser.error("--flush-every must be positive")
    if args.no_system_prompt and args.system_prompt != DEFAULT_SYSTEM_PROMPT:
        parser.error("--no-system-prompt cannot be combined with a custom --system-prompt")
    if not args.no_system_prompt and not args.system_prompt.strip():
        parser.error("--system-prompt cannot be blank; use --no-system-prompt to omit it")
    return args


def tokenizer_model_path(path: Path) -> Path:
    return path / "tokenizer.model" if path.is_dir() else path


def sidecar_paths(output: Path) -> tuple[Path, Path, Path]:
    stem = output.with_suffix("")
    offsets = stem.with_name(stem.name + ".document_offsets.bin")
    loss_mask = stem.with_name(stem.name + ".loss_mask.bin")
    return offsets, loss_mask, stem.with_suffix(".json")


def prepare_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists. Use --overwrite to replace it:\n{names}")
    for path in existing:
        path.unlink()


def token_dtype_for_vocab(vocab_size: int) -> np.dtype[Any]:
    """Use the smallest signed dtype that can represent every token ID."""
    max_token_id = vocab_size - 1
    if max_token_id <= np.iinfo(np.int16).max:
        return np.dtype("<i2")
    if max_token_id <= np.iinfo(np.int32).max:
        return np.dtype("<i4")
    if max_token_id <= np.iinfo(np.int64).max:
        return np.dtype("<i8")
    raise ValueError("Tokenizer vocabulary is too large for supported token ID dtypes.")


def write_tokens(handle: Any, token_ids: list[int], token_dtype: np.dtype[Any]) -> None:
    handle.write(np.asarray(token_ids, dtype=token_dtype).tobytes())


def write_mask(handle: Any, mask: list[bool]) -> None:
    handle.write(np.asarray(mask, dtype=np.uint8).tobytes())


def nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def validate_messages(value: object, path: Path, line_number: int) -> list[dict[str, str]]:
    """Validate the supported source schema and preserve only renderable fields."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{path}:{line_number}: messages must be a non-empty array")
    messages: list[dict[str, str]] = []
    expected_role = "user"
    for index, source_message in enumerate(value):
        if not isinstance(source_message, dict):
            raise ValueError(f"{path}:{line_number}: message {index} is not an object")
        role = source_message.get("role")
        if role != expected_role:
            raise ValueError(
                f"{path}:{line_number}: message {index} must have role {expected_role!r}; got {role!r}"
            )
        content = nonempty_string(source_message.get("content"))
        if content is None:
            raise ValueError(f"{path}:{line_number}: {role} message {index} has empty content")
        message = {"role": role, "content": content}
        reasoning = source_message.get("reasoning_content")
        if role == "assistant" and reasoning is not None:
            reasoning_text = nonempty_string(reasoning)
            if reasoning_text is None:
                raise ValueError(f"{path}:{line_number}: assistant message {index} has empty reasoning_content")
            message["reasoning_content"] = reasoning_text
        elif role == "user" and reasoning not in (None, ""):
            raise ValueError(f"{path}:{line_number}: user message {index} unexpectedly has reasoning_content")
        messages.append(message)
        expected_role = "assistant" if expected_role == "user" else "user"
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{path}:{line_number}: conversation ends with a user-only turn")
    return messages


def append_fragment(parts: list[str], spans: list[tuple[int, int]], text: str, trainable: bool, byte_offset: int) -> int:
    """Append rendered text and record its byte span when it contributes SFT loss."""
    parts.append(text)
    end = byte_offset + len(text.encode("utf-8"))
    if trainable:
        spans.append((byte_offset, end))
    return end


def render_document(
    messages: list[dict[str, str]], system_prompt: str | None
) -> tuple[str, list[tuple[int, int]], list[int]]:
    """Render text, supervised byte spans, and assistant-turn end byte offsets."""
    parts: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    assistant_turn_ends: list[int] = []
    byte_offset = 0
    if system_prompt is not None:
        byte_offset = append_fragment(parts, assistant_spans, f"{ROLE_MARKERS['system']}\n{system_prompt}\n", False, byte_offset)

    for message in messages:
        role = message["role"]
        trainable = role == "assistant"
        # Role markers identify the turn but are prompt/context tokens, not
        # assistant response targets.  Reasoning delimiters remain supervised:
        # the model must learn to open and close explicit reasoning content.
        byte_offset = append_fragment(parts, assistant_spans, f"{ROLE_MARKERS[role]}\n", False, byte_offset)
        if trainable and "reasoning_content" in message:
            byte_offset = append_fragment(parts, assistant_spans, f"{THINKING_MARKERS[0]}\n", True, byte_offset)
            byte_offset = append_fragment(parts, assistant_spans, message["reasoning_content"] + "\n", True, byte_offset)
            byte_offset = append_fragment(parts, assistant_spans, f"{THINKING_MARKERS[1]}\n", True, byte_offset)
        byte_offset = append_fragment(parts, assistant_spans, message["content"] + "\n", trainable, byte_offset)
        if trainable:
            assistant_turn_ends.append(byte_offset)
    return "".join(parts), assistant_spans, assistant_turn_ends


def spans_intersect(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def tokenize_document(
    tokenizer: spm.SentencePieceProcessor,
    rendered: str,
    assistant_spans: list[tuple[int, int]],
    assistant_turn_ends: list[int],
) -> tuple[list[int], list[bool]]:
    """Tokenize through each answer and append a supervised EOS ID directly.

    SentencePiece control tokens must be inserted by ID, not encoded as text.
    Splitting at turn ends also prevents pieces from spanning an EOS boundary.
    """
    encoded = rendered.encode("utf-8")
    if not assistant_turn_ends or assistant_turn_ends[-1] != len(encoded):
        raise ValueError("Conversation must end with an assistant turn.")
    if tokenizer.eos_id() < 0:
        raise ValueError("Tokenizer must define an EOS ID.")
    token_ids: list[int] = []
    loss_mask: list[bool] = []
    start = 0
    for end in assistant_turn_ends:
        if not start < end <= len(encoded):
            raise ValueError("Assistant turn ends must be strictly increasing byte offsets.")
        proto = tokenizer.encode(encoded[start:end].decode("utf-8"), out_type="proto")
        if not proto.pieces:
            raise ValueError("Tokenizer produced no tokens for a rendered non-empty turn")
        for piece in proto.pieces:
            token_ids.append(int(piece.id))
            # Empty dummy-prefix pieces are context only. Convert segment-local
            # byte offsets back to conversation offsets before applying the mask.
            loss_mask.append(
                piece.begin < piece.end
                and spans_intersect(start + piece.begin, start + piece.end, assistant_spans)
            )
        token_ids.append(tokenizer.eos_id())
        loss_mask.append(True)
        start = end
    return token_ids, loss_mask


def iter_chunks(
    token_ids: list[int],
    loss_mask: list[bool],
    bos_id: int,
    eos_id: int,
    context_length: int,
    overlap: int,
) -> list[tuple[list[int], list[bool]]]:
    """Split content tokens into independent chunks with non-repeating loss."""
    payload_size = context_length - 2
    chunks: list[tuple[list[int], list[bool]]] = []
    start = 0
    while start < len(token_ids):
        end = min(start + payload_size, len(token_ids))
        chunk_ids = [bos_id, *token_ids[start:end]]
        chunk_mask = [False, *loss_mask[start:end]]
        # Real turn-ending EOS already belongs to the payload and is supervised.
        # Only artificial chunk terminators are masked; never duplicate EOS.
        if chunk_ids[-1] != eos_id:
            chunk_ids.append(eos_id)
            chunk_mask.append(False)
        if start > 0:
            repeated_tokens = min(overlap, end - start)
            chunk_mask[1 : 1 + repeated_tokens] = [False] * repeated_tokens
        chunks.append((chunk_ids, chunk_mask))
        if end == len(token_ids):
            break
        start = end - overlap
    return chunks


def input_source_label(row: dict[str, Any]) -> str:
    source = row.get("source")
    subset = row.get("subset")
    if isinstance(source, str) and isinstance(subset, str):
        return f"{source}/{subset}"
    if isinstance(source, str):
        return source
    return "unknown"


def package(args: argparse.Namespace) -> None:
    input_paths = [path.resolve() for path in args.paths]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input JSONL file(s) not found:\n" + "\n".join(missing))

    model_path = tokenizer_model_path(args.tokenizer).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    bos_id, eos_id = tokenizer.bos_id(), tokenizer.eos_id()
    if bos_id < 0 or eos_id < 0:
        raise ValueError("Tokenizer must define both BOS and EOS IDs.")
    token_dtype = token_dtype_for_vocab(tokenizer.vocab_size())
    required_pieces = [*ROLE_MARKERS.values(), *THINKING_MARKERS]
    missing_pieces = [
        piece
        for piece in required_pieces
        if tokenizer.id_to_piece(tokenizer.piece_to_id(piece)) != piece
    ]
    if missing_pieces:
        raise ValueError("Tokenizer is missing required protocol piece(s): " + ", ".join(missing_pieces))

    output = args.output.resolve()
    offsets_path, loss_mask_path, metadata_path = sidecar_paths(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_outputs([output, offsets_path, loss_mask_path, metadata_path], args.overwrite)
    partial_output = output.with_suffix(output.suffix + ".partial")
    partial_offsets = offsets_path.with_suffix(offsets_path.suffix + ".partial")
    partial_mask = loss_mask_path.with_suffix(loss_mask_path.suffix + ".partial")
    for path in (partial_output, partial_offsets, partial_mask):
        path.unlink(missing_ok=True)

    system_prompt = None if args.no_system_prompt else args.system_prompt
    source_documents = 0
    chunk_count = 0
    token_count = 0
    supervised_token_count = 0
    source_counts: Counter[str] = Counter()
    try:
        with (
            partial_output.open("wb") as token_file,
            partial_offsets.open("wb") as offsets_file,
            partial_mask.open("wb") as mask_file,
        ):
            for input_path in input_paths:
                with input_path.open("r", encoding="utf-8") as input_file:
                    for line_number, line in enumerate(input_file, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(f"{input_path}:{line_number}: invalid JSONL: {error.msg}") from error
                        if not isinstance(row, dict):
                            raise ValueError(f"{input_path}:{line_number}: each JSONL line must be an object")
                        messages = validate_messages(row.get("messages"), input_path, line_number)
                        rendered, assistant_spans, assistant_turn_ends = render_document(messages, system_prompt)
                        content_ids, content_mask = tokenize_document(tokenizer, rendered, assistant_spans, assistant_turn_ends)
                        chunks = iter_chunks(
                            content_ids,
                            content_mask,
                            bos_id,
                            eos_id,
                            args.context_length,
                            args.overlap,
                        )
                        for chunk_ids, chunk_mask in chunks:
                            np.asarray([token_count], dtype="<u8").tofile(offsets_file)
                            write_tokens(token_file, chunk_ids, token_dtype)
                            write_mask(mask_file, chunk_mask)
                            token_count += len(chunk_ids)
                            supervised_token_count += sum(chunk_mask)
                            chunk_count += 1
                            if chunk_count % args.flush_every == 0:
                                token_file.flush()
                                offsets_file.flush()
                                mask_file.flush()
                                os.fsync(token_file.fileno())
                                os.fsync(offsets_file.fileno())
                                os.fsync(mask_file.fileno())
                        source_documents += 1
                        source_counts[input_source_label(row)] += 1
                        if source_documents % 1_000 == 0:
                            print(
                                f"Packed {source_documents:,} source documents into {chunk_count:,} chunks and "
                                f"{token_count:,} tokens.",
                                flush=True,
                            )
                        if args.max_documents is not None and source_documents >= args.max_documents:
                            break
                if args.max_documents is not None and source_documents >= args.max_documents:
                    break
            np.asarray([token_count], dtype="<u8").tofile(offsets_file)

        if source_documents == 0:
            raise RuntimeError("No non-empty JSONL records were found.")
        partial_output.replace(output)
        partial_offsets.replace(offsets_path)
        partial_mask.replace(loss_mask_path)
        metadata = {
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "format": f"little-endian signed {token_dtype.itemsize * 8}-bit token IDs",
            "token_file": output.name,
            "document_offsets_file": offsets_path.name,
            "loss_mask_file": loss_mask_path.name,
            "token_dtype": token_dtype.str,
            "offset_dtype": "<u8",
            "loss_mask_dtype": "|u1",
            "loss_mask_semantics": (
                "1 means this token is a supervised next-token target; trainers must shift the mask with labels "
                "(mask[start + 1 : start + sequence_length + 1])"
            ),
            "token_count": token_count,
            "supervised_token_count": supervised_token_count,
            "document_count": chunk_count,
            "source_document_count": source_documents,
            "bos_id": bos_id,
            "eos_id": eos_id,
            "special_token_ids": {piece: tokenizer.piece_to_id(piece) for piece in required_pieces},
            "tokenizer_model": str(model_path),
            "tokenizer_vocab_size": tokenizer.vocab_size(),
            "input_jsonl_files": [str(path) for path in input_paths],
            "source_document_counts": dict(sorted(source_counts.items())),
            "context_length": args.context_length,
            "chunk_payload_limit": args.context_length - 2,
            "chunk_overlap": args.overlap,
            "system_prompt": system_prompt,
            "template": {
                "role_markers": ROLE_MARKERS,
                "thinking_markers": THINKING_MARKERS,
                "role_markers_are_context_only": True,
                "assistant_content_and_reasoning_delimiters_are_supervised": True,
                "system_and_user_tokens_are_masked": True,
                "overlap_tokens_are_masked_in_nonfirst_chunks": True,
                "assistant_turn_eos_is_supervised": True,
                "artificial_chunk_eos_is_masked": True,
            },
            "output_durability": {"flush_every_chunks": args.flush_every, "fsync_after_flush": True},
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {output}")
        print(f"Wrote {offsets_path}")
        print(f"Wrote {loss_mask_path}")
        print(f"Wrote {metadata_path}")
    except Exception:
        partial_output.unlink(missing_ok=True)
        partial_offsets.unlink(missing_ok=True)
        partial_mask.unlink(missing_ok=True)
        raise


def main() -> None:
    package(parse_args())


if __name__ == "__main__":
    main()
