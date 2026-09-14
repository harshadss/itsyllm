#!/usr/bin/env python3
"""Stream normalized conversations from ``ai4bharat/indic-align`` to JSONL.

The output intentionally contains source messages rather than a rendered chat
template.  Packaging is responsible for adding role-marker tokens, any system
prompt, tokenization, sequence boundaries, and SFT loss masks.

Examples:
    uv run python scripts/download_indic_align_raw.py
    uv run python scripts/download_indic_align_raw.py --rows 5000 --subsets Anudesh Wiki_Chat
    uv run python scripts/download_indic_align_raw.py --shuffle-buffer-size 50000 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ID = "ai4bharat/indic-align"
SPLIT = "train"
DEFAULT_ROWS = 20_000
DEFAULT_SEED = 1337
DEFAULT_SHUFFLE_BUFFER_SIZE = 100_000
DEFAULT_FLUSH_EVERY = 1_000
DEFAULT_ROOT = Path("/home/harshad/Workspace/Learning/DeepLearning/Datasets/ai4bharat/indic-align")
LANGUAGE_COLUMNS = ("eng_Latn", "hin_Deva", "hin_Latn")


@dataclass(frozen=True)
class SubsetSpec:
    """Dataset-specific schema and sampling rules."""

    config: str
    source_column: str | None
    uses_declared_turn_count: bool
    take_all_valid_rows: bool = False


SUBSETS = {
    "Anudesh": SubsetSpec("Anudesh", "interactions", uses_declared_turn_count=True),
    "Wiki_Chat": SubsetSpec("Wiki_Chat", None, uses_declared_turn_count=False),
    "Wiki_Conv": SubsetSpec("Wiki_Conv", None, uses_declared_turn_count=False),
    "Toxic_Matrix": SubsetSpec("Toxic_Matrix", None, uses_declared_turn_count=False),
    # This config has fewer than 20,000 rows.  Stream its complete train split
    # and retain every valid conversation rather than treating --rows as an
    # error condition.
    "OpenAssistant_T": SubsetSpec("OpenAssistant_T", None, uses_declared_turn_count=True, take_all_valid_rows=True),
}
DEFAULT_SUBSETS = tuple(SUBSETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=DEFAULT_SUBSETS,
        default=list(DEFAULT_SUBSETS),
        help="IndicAlign subsets to include (default: all supported subsets).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=(
            "Valid source rows to select per subset after the five-turn filter "
            f"(default: {DEFAULT_ROWS}). OpenAssistant_T always consumes all valid rows."
        ),
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
            f"per subset (default: {DEFAULT_FLUSH_EVERY})."
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
    if args.rows <= 0 or args.shuffle_buffer_size <= 0 or args.flush_every <= 0:
        parser.error("--rows, --shuffle-buffer-size, and --flush-every must be positive")
    return args


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output is None:
        subset_label = "-".join(args.subsets).lower().replace("_", "-")
        output = args.output_root / "samples" / f"indic-align-{subset_label}-shuffled-{args.rows}.jsonl"
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
    """Return a source string only when it contains non-whitespace content."""
    return value if isinstance(value, str) and value.strip() else None


def declared_turn_count_is_valid(row: dict[str, Any]) -> bool:
    """Apply the source ``num_turns`` filter for configs where it is reliable."""
    value = row.get("num_turns")
    if isinstance(value, bool):
        return False
    try:
        count = int(value)
    except (TypeError, ValueError):
        return False
    return 1 <= count <= 5


def normalise_interactions(value: object) -> tuple[list[dict[str, str]] | None, str | None]:
    """Convert a list of [user, assistant] pairs to canonical message objects.

    A final user-only turn (represented by an empty assistant response) is
    omitted.  An empty or malformed turn before the end makes the full source
    conversation unusable, because retaining it would corrupt turn order.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None, "malformed_interactions"
    if not value:
        return None, "empty_interactions"

    messages: list[dict[str, str]] = []
    last_index = len(value) - 1
    for index, turn in enumerate(value):
        if not isinstance(turn, Sequence) or isinstance(turn, (str, bytes)) or len(turn) != 2:
            return None, "malformed_turn"
        user = nonempty_string(turn[0])
        assistant = nonempty_string(turn[1])
        if user is None:
            return None, "empty_user"
        if assistant is None:
            if index == last_index:
                break
            return None, "empty_assistant"
        messages.extend(({"role": "user", "content": user}, {"role": "assistant", "content": assistant}))

    if not messages:
        return None, "no_complete_turns"
    return messages, None


def candidate_language_values(row: dict[str, Any], spec: SubsetSpec) -> Iterable[tuple[str | None, object]]:
    if spec.source_column is not None:
        yield None, row.get(spec.source_column)
        return
    for language in LANGUAGE_COLUMNS:
        yield language, row.get(language)


def source_has_at_most_five_turns(row: dict[str, Any], spec: SubsetSpec) -> bool:
    """Apply the documented five-turn criterion before output expansion."""
    if spec.uses_declared_turn_count:
        return declared_turn_count_is_valid(row)
    # Wiki_* and Toxic_Matrix have unreliable num_turns values.  One source
    # row may have a different number of turns in each language column, so a
    # row survives if at least one language has a valid outer-list length.
    return any(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and 1 <= len(value) <= 5
        for _, value in candidate_language_values(row, spec)
    )


def normalise_row(row: dict[str, Any], spec: SubsetSpec) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Return every valid language-specific conversation for one source row."""
    rejected: Counter[str] = Counter()
    outputs: list[dict[str, Any]] = []
    for language, interactions in candidate_language_values(row, spec):
        # For the configs with unreliable num_turns, the real outer-list size
        # is the turn count and must be checked independently per language.
        if not spec.uses_declared_turn_count:
            if not isinstance(interactions, Sequence) or isinstance(interactions, (str, bytes)):
                rejected["malformed_interactions"] += 1
                continue
            if not 1 <= len(interactions) <= 5:
                rejected["turn_count_out_of_range"] += 1
                continue
        messages, reason = normalise_interactions(interactions)
        if messages is None:
            rejected[reason or "invalid_conversation"] += 1
            continue
        document: dict[str, Any] = {
            "source": "indic-align",
            "subset": spec.config,
            "messages": messages,
        }
        if language is not None:
            document["language"] = language
        outputs.append(document)
    return outputs, rejected


def expected_columns(spec: SubsetSpec) -> tuple[str, ...]:
    if spec.source_column is not None:
        return ("num_turns", spec.source_column)
    return ("num_turns", *LANGUAGE_COLUMNS)


def load_stream(spec: SubsetSpec, revision: str, seed: int, buffer_size: int) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(REPO_ID, spec.config, split=SPLIT, streaming=True, revision=revision)
    available = set(dataset.features)
    missing = set(expected_columns(spec)) - available
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{spec.config}/{SPLIT} is missing expected column(s): {names}")
    return dataset.select_columns(list(expected_columns(spec))).shuffle(seed=seed, buffer_size=buffer_size)


def write_subset(
    handle: Any,
    spec: SubsetSpec,
    args: argparse.Namespace,
    revision: str,
    subset_seed: int,
) -> dict[str, Any]:
    """Stream, validate, and write one subset; return its audit statistics."""
    stream = load_stream(spec, revision, subset_seed, args.shuffle_buffer_size)
    source_rows_scanned = 0
    accepted_source_rows = 0
    outputs_written = 0
    output_languages: Counter[str] = Counter()
    rejected: Counter[str] = Counter()

    target_label = "all valid rows" if spec.take_all_valid_rows else f"{args.rows:,} valid source rows"
    print(f"Streaming {spec.config}/{SPLIT}: target {target_label}", flush=True)
    for row in stream:
        source_rows_scanned += 1
        if not source_has_at_most_five_turns(row, spec):
            rejected["turn_count_out_of_range_or_invalid"] += 1
            continue
        documents, document_rejections = normalise_row(row, spec)
        rejected.update(document_rejections)
        if not documents:
            rejected["no_valid_language_conversation"] += 1
            continue

        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
            outputs_written += 1
            output_languages[document.get("language", "und")] += 1
            if outputs_written % args.flush_every == 0:
                handle.flush()
                os.fsync(handle.fileno())
        accepted_source_rows += 1
        if accepted_source_rows % 1_000 == 0:
            print(
                f"  {spec.config}: {accepted_source_rows:,} accepted source rows, "
                f"{outputs_written:,} JSONL records",
                file=sys.stderr,
                flush=True,
            )
        if not spec.take_all_valid_rows and accepted_source_rows >= args.rows:
            break

    target_reached = spec.take_all_valid_rows or accepted_source_rows == args.rows
    if not target_reached:
        print(
            f"WARNING: Only {accepted_source_rows:,} valid source rows were available for "
            f"{spec.config}/{SPLIT}; requested {args.rows:,}. Continuing with the available rows.",
            file=sys.stderr,
            flush=True,
        )
    return {
        "config": spec.config,
        "requested_valid_source_rows": "all" if spec.take_all_valid_rows else args.rows,
        "source_rows_scanned": source_rows_scanned,
        "accepted_source_rows": accepted_source_rows,
        "target_reached": target_reached,
        "output_records": outputs_written,
        "output_records_by_language": dict(sorted(output_languages.items())),
        "rejections": dict(sorted(rejected.items())),
        "shuffle_seed": subset_seed,
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
    with tempfile.TemporaryDirectory(prefix="indic-align-hf-") as cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")
        os.environ["HF_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        revision = resolve_revision(args.revision)
        print(f"Using {REPO_ID} revision {revision}", flush=True)

        subset_stats: list[dict[str, Any]] = []
        try:
            with partial.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
                for index, subset_name in enumerate(args.subsets):
                    subset_stats.append(
                        write_subset(handle, SUBSETS[subset_name], args, revision, args.seed + index)
                    )
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    partial.replace(output)
    total_records = sum(int(stats["output_records"]) for stats in subset_stats)
    write_json(
        manifest,
        {
            "created_at": now(),
            "dataset": REPO_ID,
            "revision": revision,
            "split": SPLIT,
            "format": "jsonl",
            "record_schema": {
                "source": "indic-align",
                "subset": "source Hugging Face config name",
                "language": "optional source language-column name",
                "messages": "ordered user/assistant message objects",
            },
            "selection": (
                "streaming bounded-buffer shuffle; first valid source rows after the configured five-turn "
                "filter; not exact global-uniform sampling"
            ),
            "five_turn_filter": {
                "Anudesh": "source num_turns",
                "OpenAssistant_T": "source num_turns",
                "Wiki_Chat": "actual outer-list length per language",
                "Wiki_Conv": "actual outer-list length per language",
                "Toxic_Matrix": "actual outer-list length per language",
            },
            "normalization": {
                "system_prompt_included": False,
                "role_markers_included": False,
                "terminal_user_only_turn": "discarded",
                "nonterminal_empty_or_malformed_turn": "discard conversation",
            },
            "shuffle": {"seed": args.seed, "buffer_size": args.shuffle_buffer_size},
            "output_durability": {"flush_every_records": args.flush_every, "fsync_after_flush": True},
            "requested_rows_per_non_openassistant_subset": args.rows,
            "total_output_records": total_records,
            "subsets": subset_stats,
            "path": str(output),
        },
    )
    print(f"Completed {total_records:,} JSONL records -> {output}", flush=True)
    print(f"Wrote manifest -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
