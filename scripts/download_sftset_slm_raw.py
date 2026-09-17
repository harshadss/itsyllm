#!/usr/bin/env python3
"""Stream and normalize ``AxiomicLabs/SFTset-SLM`` to SFT JSONL.

The source ``messages`` column is converted to the project's ordered
user/assistant message schema. Source system messages are omitted because the
SFT packager injects this project's system prompt. A ``gpt`` role is normalized
to ``assistant``. User messages without an immediately following assistant
message are dropped while the remaining complete turns are retained.

Rows are kept only when the source ``token_count`` is at most 4096 and the
normalized conversation contains at most eight complete user/assistant turns.
The output intentionally contains no rendered role markers or token IDs.

Examples:
    uv run python scripts/download_sftset_slm_raw.py
    uv run python scripts/download_sftset_slm_raw.py --max-rows 100000
    uv run python scripts/download_sftset_slm_raw.py --max-tokens 4096 --max-turns 8 --overwrite
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


REPO_ID = "AxiomicLabs/SFTset-SLM"
SPLIT = "train"
DEFAULT_MAX_TOKENS = 4_096
DEFAULT_MAX_TURNS = 8
DEFAULT_FLUSH_EVERY = 1_000
DEFAULT_ROOT = Path("/home/harshad/Workspace/Learning/DeepLearning/Datasets/AxiomicLabs/SFTset-SLM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Stop after this many valid output rows (default: consume the full train split).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum source token_count, inclusive (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Maximum complete user-assistant turns per conversation (default: {DEFAULT_MAX_TURNS}).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=DEFAULT_FLUSH_EVERY,
        help=f"Flush and fsync after this many output records (default: {DEFAULT_FLUSH_EVERY}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Dataset root when --output is omitted (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument("--output", type=Path, help="JSONL output path. Defaults under --output-root.")
    parser.add_argument("--revision", default="main", help="Hugging Face revision to resolve (default: main).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing completed output.")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if args.max_tokens <= 0 or args.max_turns <= 0 or args.flush_every <= 0:
        parser.error("--max-tokens, --max-turns, and --flush-every must be positive")
    return args


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output is None:
        row_label = "all" if args.max_rows is None else str(args.max_rows)
        output = args.output_root / "samples" / (
            f"sftset-slm-max-{args.max_tokens}-tokens-max-{args.max_turns}-turns-{row_label}-rows.jsonl"
        )
    else:
        output = args.output
    if output.suffix != ".jsonl":
        raise ValueError(f"Output must end in .jsonl: {output}")
    return output, output.with_suffix(".manifest.json")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def source_message_parts(value: object) -> tuple[str, str] | None:
    """Read either a message object or a two-item ``[role, content]`` list."""
    if isinstance(value, dict):
        role = nonempty_string(value.get("role"))
        content = nonempty_string(value.get("content"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        role = nonempty_string(value[0])
        content = nonempty_string(value[1])
    else:
        return None
    if role is None or content is None:
        return None
    return role, content


def normalise_messages(
    value: object,
    max_turns: int,
) -> tuple[list[dict[str, str]] | None, str | None, Counter[str]]:
    """Keep complete adjacent user/assistant pairs and report dropped messages."""
    changes: Counter[str] = Counter()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None, "malformed_messages", changes
    if not value:
        return None, "empty_messages", changes

    parsed: list[tuple[str, str]] = []
    for source_message in value:
        parts = source_message_parts(source_message)
        if parts is None:
            return None, "malformed_message", changes
        role, content = parts
        if role == "gpt":
            role = "assistant"
            changes["gpt_roles_normalized"] += 1
        if role == "system":
            changes["system_messages_dropped"] += 1
            continue
        if role not in {"user", "assistant"}:
            return None, "unsupported_role", changes
        parsed.append((role, content))

    messages: list[dict[str, str]] = []
    index = 0
    while index < len(parsed):
        role, content = parsed[index]
        if role == "user":
            if index + 1 < len(parsed) and parsed[index + 1][0] == "assistant":
                messages.extend(
                    (
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": parsed[index + 1][1]},
                    )
                )
                index += 2
                continue
            changes["unanswered_user_messages_dropped"] += 1
        else:
            changes["orphan_assistant_messages_dropped"] += 1
        index += 1

    if not messages:
        return None, "no_complete_turns", changes
    if len(messages) // 2 > max_turns:
        return None, "turn_count_exceeded", changes
    return messages, None, changes


def source_token_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def normalise_row(
    row: dict[str, Any],
    max_tokens: int,
    max_turns: int,
) -> tuple[dict[str, Any] | None, str | None, Counter[str]]:
    token_count = source_token_count(row.get("token_count"))
    if token_count is None:
        return None, "invalid_token_count", Counter()
    if token_count > max_tokens:
        return None, "token_count_exceeded", Counter()

    messages, reason, changes = normalise_messages(row.get("messages"), max_turns)
    if messages is None:
        return None, reason, changes

    document: dict[str, Any] = {
        "source": "sftset-slm",
        "messages": messages,
        "token_count": token_count,
    }
    upstream_source = nonempty_string(row.get("source"))
    if upstream_source is not None:
        document["upstream_source"] = upstream_source
    return document, None, changes


def load_stream(revision: str) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(REPO_ID, split=SPLIT, streaming=True, revision=revision)
    available = set(dataset.features)
    required = {"messages", "token_count"}
    missing = required - available
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{SPLIT} is missing required column(s): {names}")
    columns = ["messages", "token_count", *(["source"] if "source" in available else [])]
    return dataset.select_columns(columns)


def write_stream(handle: Any, args: argparse.Namespace, revision: str) -> dict[str, Any]:
    stream = load_stream(revision)
    source_rows_scanned = 0
    accepted_rows = 0
    accepted_tokens = 0
    accepted_turns = 0
    rejected: Counter[str] = Counter()
    normalizations: Counter[str] = Counter()
    upstream_sources: Counter[str] = Counter()

    target = "all valid rows" if args.max_rows is None else f"{args.max_rows:,} valid rows"
    print(f"Streaming {REPO_ID}/{SPLIT}: target {target}", flush=True)
    for row in stream:
        source_rows_scanned += 1
        document, reason, changes = normalise_row(row, args.max_tokens, args.max_turns)
        normalizations.update(changes)
        if document is None:
            rejected[reason or "invalid_conversation"] += 1
            continue

        handle.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
        accepted_rows += 1
        accepted_tokens += int(document["token_count"])
        accepted_turns += len(document["messages"]) // 2
        upstream_sources[document.get("upstream_source", "unknown")] += 1
        if accepted_rows % args.flush_every == 0:
            handle.flush()
            os.fsync(handle.fileno())
        if accepted_rows % 10_000 == 0:
            print(
                f"  {accepted_rows:,} accepted rows from {source_rows_scanned:,} scanned",
                file=sys.stderr,
                flush=True,
            )
        if args.max_rows is not None and accepted_rows >= args.max_rows:
            break

    target_reached = args.max_rows is None or accepted_rows == args.max_rows
    if not target_reached:
        print(
            f"WARNING: Only {accepted_rows:,} valid rows were available; requested {args.max_rows:,}.",
            file=sys.stderr,
            flush=True,
        )
    return {
        "requested_valid_rows": "all" if args.max_rows is None else args.max_rows,
        "source_rows_scanned": source_rows_scanned,
        "accepted_rows": accepted_rows,
        "target_reached": target_reached,
        "accepted_source_token_count": accepted_tokens,
        "accepted_complete_turns": accepted_turns,
        "output_rows_by_upstream_source": dict(sorted(upstream_sources.items())),
        "normalizations": dict(sorted(normalizations.items())),
        "rejections": dict(sorted(rejected.items())),
    }


def main() -> None:
    args = parse_args()
    output, manifest = output_paths(args)
    if not ensure_target(output, manifest, args.overwrite):
        return
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="sftset-slm-hf-") as cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")
        os.environ["HF_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        revision = resolve_revision(args.revision)
        print(f"Using {REPO_ID} revision {revision}", flush=True)

        try:
            with partial.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
                stats = write_stream(handle, args, revision)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    partial.replace(output)
    write_json(
        manifest,
        {
            "created_at": now(),
            "dataset": REPO_ID,
            "revision": revision,
            "split": SPLIT,
            "format": "jsonl",
            "record_schema": {
                "source": "sftset-slm",
                "messages": "ordered user/assistant message objects",
                "token_count": "source LFM tokenizer count before message normalization",
                "upstream_source": "optional original source dataset identifier",
            },
            "selection": "streaming source order; optional first-N valid-row cap",
            "normalization": {
                "system_prompt_included": False,
                "role_markers_included": False,
                "gpt_role": "renamed to assistant",
                "unanswered_user_message": "dropped without rejecting other complete turns",
                "maximum_source_token_count_inclusive": args.max_tokens,
                "maximum_complete_user_assistant_turns": args.max_turns,
            },
            "output_durability": {"flush_every_records": args.flush_every, "fsync_after_flush": True},
            "stats": stats,
            "path": str(output),
        },
    )
    print(f"Completed {stats['accepted_rows']:,} JSONL records -> {output}", flush=True)
    print(f"Wrote manifest -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
