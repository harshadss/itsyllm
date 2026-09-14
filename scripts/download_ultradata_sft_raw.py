#!/usr/bin/env python3
"""Stream a bounded-shuffle UltraData SFT sample to normalized JSONL.

The output retains user/assistant messages and assistant reasoning separately.
It deliberately does not render role markers, thinking markers, a system
prompt, token IDs, or SFT labels; those are packaging responsibilities.

The UltraData repository is gated.  After accepting its access conditions on
Hugging Face, authenticate in the invoking environment (for example, export
``HF_TOKEN``) before running this script.

Examples:
    uv run python scripts/download_ultradata_sft_raw.py
    uv run python scripts/download_ultradata_sft_raw.py --domains IF Math --think-rows 25000
    uv run python scripts/download_ultradata_sft_raw.py --shuffle-buffer-size 50000 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ID = "openbmb/UltraData-SFT-2605"
DOMAINS = ("IF", "Knowledge", "Math")
THINK_TYPES = ("no_think", "think")
DEFAULT_NO_THINK_ROWS = 100_000
DEFAULT_THINK_ROWS = 50_000
DEFAULT_MAX_TURNS = 5
DEFAULT_MAX_WORDS = 3_000
DEFAULT_SEED = 1337
DEFAULT_SHUFFLE_BUFFER_SIZE = 100_000
DEFAULT_FLUSH_EVERY = 1_000
DEFAULT_ROOT = Path("/home/harshad/Workspace/Learning/DeepLearning/Datasets/openbmb/ultradata-sft-2605")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS), help="Domains to sample (default: all).")
    parser.add_argument(
        "--no-think-rows",
        type=int,
        default=DEFAULT_NO_THINK_ROWS,
        help=f"Target valid rows per domain/no_think bucket (default: {DEFAULT_NO_THINK_ROWS}).",
    )
    parser.add_argument(
        "--think-rows",
        type=int,
        default=DEFAULT_THINK_ROWS,
        help=f"Target valid rows per domain/think bucket (default: {DEFAULT_THINK_ROWS}).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Maximum user-assistant exchanges per conversation (default: {DEFAULT_MAX_TURNS}).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_WORDS,
        help=f"Drop a conversation with more than this many whitespace-split words (default: {DEFAULT_MAX_WORDS}).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Streaming-shuffle seed (default: {DEFAULT_SEED}).")
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER_SIZE,
        help=(
            "Maximum source rows retained for bounded streaming shuffle "
            f"(default: {DEFAULT_SHUFFLE_BUFFER_SIZE})."
        ),
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=DEFAULT_FLUSH_EVERY,
        help=(
            "Flush and fsync the JSONL file after this many output records "
            f"per bucket (default: {DEFAULT_FLUSH_EVERY})."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Dataset root when --output is omitted (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument("--output", type=Path, help="Combined JSONL output path. Defaults under --output-root.")
    parser.add_argument("--revision", default="main", help="Hugging Face revision to resolve (default: main).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing completed output.")
    args = parser.parse_args()
    positive_args = (args.no_think_rows, args.think_rows, args.max_turns, args.max_words, args.shuffle_buffer_size, args.flush_every)
    if any(value <= 0 for value in positive_args):
        parser.error("all row limits, --max-turns, --max-words, --shuffle-buffer-size, and --flush-every must be positive")
    return args


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output is None:
        domain_label = "-".join(args.domains).lower()
        output = args.output_root / "samples" / (
            f"ultradata-sft-{domain_label}-no-think-{args.no_think_rows}-think-{args.think_rows}-shuffled.jsonl"
        )
    else:
        output = args.output
    if output.suffix != ".jsonl":
        raise ValueError(f"Output must end in .jsonl: {output}")
    return output, output.with_suffix(".manifest.json")


def resolve_revision(revision: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(REPO_ID, revision=revision).sha


def ensure_target(output: Path, manifest: Path, overwrite: bool) -> bool:
    """Prepare a target. Return False when a completed target is reusable."""
    if output.exists() or manifest.exists():
        if not overwrite:
            if output.exists() and manifest.exists():
                print(f"Skipping completed output: {output}")
                return False
            raise FileExistsError(f"Refusing incomplete existing output for {output}. Remove it or use --overwrite.")
        output.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    return True


def nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def normalise_messages(
    value: object,
    max_turns: int,
) -> tuple[list[dict[str, str]] | None, str | None]:
    """Validate an alternating user/assistant exchange sequence.

    Assistant ``reasoning_content`` is preserved as a separate field.  The
    packer can later surround it with the project's thinking marker tokens.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None, "malformed_messages"
    if not value:
        return None, "empty_messages"
    if len(value) % 2 != 0:
        return None, "incomplete_final_turn"
    if len(value) // 2 > max_turns:
        return None, "turn_count_out_of_range"

    messages: list[dict[str, str]] = []
    for index, source_message in enumerate(value):
        if not isinstance(source_message, dict):
            return None, "malformed_message"
        expected_role = "user" if index % 2 == 0 else "assistant"
        if source_message.get("role") != expected_role:
            return None, "non_alternating_or_unsupported_roles"
        content = nonempty_string(source_message.get("content"))
        if content is None:
            return None, "empty_content"
        message: dict[str, str] = {"role": expected_role, "content": content}

        reasoning = source_message.get("reasoning_content")
        if expected_role == "assistant" and reasoning is not None:
            reasoning_text = nonempty_string(reasoning)
            if reasoning_text is None:
                return None, "empty_or_malformed_reasoning_content"
            message["reasoning_content"] = reasoning_text
        elif expected_role == "user" and reasoning not in (None, ""):
            return None, "unexpected_user_reasoning_content"
        messages.append(message)
    return messages, None


def conversation_word_count(messages: Sequence[dict[str, str]]) -> int:
    """Count all user, answer, and reasoning text using the requested simple rule."""
    parts: list[str] = []
    for message in messages:
        parts.append(message["content"])
        if "reasoning_content" in message:
            parts.append(message["reasoning_content"])
    return len(" ".join(parts).split())


def normalise_row(
    row: dict[str, Any],
    domain: str,
    think_type: str,
    max_turns: int,
    max_words: int,
) -> tuple[dict[str, Any] | None, str | None, int]:
    messages, reason = normalise_messages(row.get("messages"), max_turns)
    if messages is None:
        return None, reason, 0
    word_count = conversation_word_count(messages)
    if word_count > max_words:
        return None, "word_count_exceeded", word_count

    document: dict[str, Any] = {
        "source": "ultradata-sft-2605",
        "subset": domain,
        "think_type": think_type,
        "messages": messages,
    }
    uid = nonempty_string(row.get("uid"))
    if uid is not None:
        document["uid"] = uid
    upstream_source = nonempty_string(row.get("source"))
    if upstream_source is not None:
        document["upstream_source"] = upstream_source
    return document, None, word_count


def load_stream(domain: str, think_type: str, revision: str, seed: int, buffer_size: int) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    # UltraData organizes thinking modes as splits, not a think_type column.
    dataset = load_dataset(REPO_ID, domain, split=think_type, streaming=True, revision=revision)
    available = set(dataset.features)
    if "messages" not in available:
        raise ValueError(f"{domain}/{think_type} is missing its required messages column.")
    columns = ["messages", *[name for name in ("uid", "source") if name in available]]
    return dataset.select_columns(columns).shuffle(seed=seed, buffer_size=buffer_size)


def target_rows(args: argparse.Namespace, think_type: str) -> int:
    return args.no_think_rows if think_type == "no_think" else args.think_rows


def write_bucket(
    handle: Any,
    domain: str,
    think_type: str,
    args: argparse.Namespace,
    revision: str,
    bucket_seed: int,
) -> dict[str, Any]:
    """Stream, validate, and write one domain/thinking-mode bucket."""
    requested_rows = target_rows(args, think_type)
    stream = load_stream(domain, think_type, revision, bucket_seed, args.shuffle_buffer_size)
    source_rows_scanned = 0
    accepted_rows = 0
    accepted_words = 0
    assistant_messages = 0
    assistant_messages_with_reasoning = 0
    rejected: Counter[str] = Counter()

    print(f"Streaming {domain}/{think_type}: target {requested_rows:,} valid rows", flush=True)
    for row in stream:
        source_rows_scanned += 1
        document, reason, word_count = normalise_row(row, domain, think_type, args.max_turns, args.max_words)
        if document is None:
            rejected[reason or "invalid_conversation"] += 1
            continue

        handle.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
        accepted_rows += 1
        accepted_words += word_count
        for message in document["messages"]:
            if message["role"] == "assistant":
                assistant_messages += 1
                assistant_messages_with_reasoning += int("reasoning_content" in message)
        if accepted_rows % args.flush_every == 0:
            handle.flush()
            os.fsync(handle.fileno())
        if accepted_rows % 1_000 == 0:
            print(
                f"  {domain}/{think_type}: {accepted_rows:,}/{requested_rows:,} rows",
                file=sys.stderr,
                flush=True,
            )
        if accepted_rows >= requested_rows:
            break

    target_reached = accepted_rows == requested_rows
    if not target_reached:
        print(
            f"WARNING: Only {accepted_rows:,} valid rows were available for {domain}/{think_type}; "
            f"requested {requested_rows:,}. Continuing with the available rows.",
            file=sys.stderr,
            flush=True,
        )
    return {
        "domain": domain,
        "think_type": think_type,
        "requested_rows": requested_rows,
        "source_rows_scanned": source_rows_scanned,
        "accepted_rows": accepted_rows,
        "target_reached": target_reached,
        "accepted_word_count": accepted_words,
        "assistant_messages": assistant_messages,
        "assistant_messages_with_reasoning": assistant_messages_with_reasoning,
        "rejections": dict(sorted(rejected.items())),
        "shuffle_seed": bucket_seed,
    }


def main() -> None:
    args = parse_args()
    output, manifest = output_paths(args)
    if not ensure_target(output, manifest, args.overwrite):
        return
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)

    # Isolate package-managed temporary files so streaming leaves no durable
    # Hugging Face cache beside the requested JSONL output.
    with tempfile.TemporaryDirectory(prefix="ultradata-sft-hf-") as cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")
        os.environ["HF_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        revision = resolve_revision(args.revision)
        print(f"Using {REPO_ID} revision {revision}", flush=True)

        bucket_stats: list[dict[str, Any]] = []
        try:
            with partial.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
                bucket_index = 0
                for domain in args.domains:
                    for think_type in THINK_TYPES:
                        bucket_stats.append(
                            write_bucket(handle, domain, think_type, args, revision, args.seed + bucket_index)
                        )
                        bucket_index += 1
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    partial.replace(output)
    total_records = sum(int(stats["accepted_rows"]) for stats in bucket_stats)
    write_json(
        manifest,
        {
            "created_at": now(),
            "dataset": REPO_ID,
            "revision": revision,
            "format": "jsonl",
            "record_schema": {
                "source": "ultradata-sft-2605",
                "subset": "UltraData domain/config name",
                "think_type": "source split: think or no_think",
                "messages": "ordered user/assistant messages; assistant reasoning_content is optional",
                "uid": "optional upstream record ID",
                "upstream_source": "optional original source identifier",
            },
            "selection": (
                "streaming bounded-buffer shuffle; first valid rows in each domain/split bucket; "
                "not exact global-uniform sampling"
            ),
            "buckets": "think and no_think are Hugging Face splits, not a table column",
            "normalization": {
                "system_prompt_included": False,
                "role_markers_included": False,
                "thinking_markers_included": False,
                "assistant_reasoning": "preserved in optional reasoning_content for the packer",
                "maximum_user_assistant_exchanges": args.max_turns,
                "maximum_whitespace_split_words": args.max_words,
            },
            "shuffle": {"seed": args.seed, "buffer_size": args.shuffle_buffer_size},
            "output_durability": {"flush_every_records": args.flush_every, "fsync_after_flush": True},
            "total_output_records": total_records,
            "bucket_stats": bucket_stats,
            "path": str(output),
        },
    )
    print(f"Completed {total_records:,} JSONL records -> {output}", flush=True)
    print(f"Wrote manifest -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
