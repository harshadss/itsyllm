#!/usr/bin/env python3
"""Estimate model size, training memory, and inference memory.

Examples:
    uv run python scripts/model_memory_report.py mqa
    uv run python scripts/model_memory_report.py gqa --batch-size 4
    uv run python scripts/model_memory_report.py small --vocab-size 32000 --context-length 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DecoderOnlyTransformer, model_config_from_name

BYTES_PER_GIB = 1024**3
BF16_BYTES = 2
FP32_BYTES = 4


def gibibytes(byte_count: int | float) -> float:
    return byte_count / BYTES_PER_GIB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_name", help="Model name: smoke, small, small_gqa, mqa, or gqa.")
    parser.add_argument("--vocab-size", type=int, default=16_384, help="Vocabulary size (default: 16384).")
    parser.add_argument("--batch-size", type=int, default=1, help="Training micro-batch or concurrent inference sequences (default: 1).")
    parser.add_argument("--context-length", type=int, help="Context length override. Defaults to the model preset.")
    args = parser.parse_args()
    if args.vocab_size <= 0 or args.batch_size <= 0:
        parser.error("--vocab-size and --batch-size must be positive")
    if args.context_length is not None and args.context_length <= 0:
        parser.error("--context-length must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = model_config_from_name(args.model_name, args.vocab_size)
    context_length = args.context_length or config.max_context_length
    parameter_count = DecoderOnlyTransformer.estimate_parameter_count(config)

    bf16_weights = parameter_count * BF16_BYTES
    fp32_weights = parameter_count * FP32_BYTES
    kv_cache = (
        args.batch_size
        * context_length
        * config.num_layers
        * config.num_key_value_heads
        * config.head_dim
        * 2  # Keys and values.
        * BF16_BYTES
    )

    # This trainer retains FP32 weights, FP32 gradients, and two FP32 AdamW moments.
    training_persistent = parameter_count * FP32_BYTES * 4
    # Autocast can retain BF16 weight copies during a forward/backward pass.
    autocast_weight_cache = bf16_weights
    # Checkpointing retains one BF16 hidden state per layer instead of all layer internals.
    checkpointed_hidden_states = args.batch_size * context_length * config.hidden_size * config.num_layers * BF16_BYTES
    logits = args.batch_size * context_length * config.vocab_size * BF16_BYTES
    sequence_memory = checkpointed_hidden_states + logits
    workspace = max(BYTES_PER_GIB // 2, sequence_memory // 2)
    training_modeled_peak = training_persistent + autocast_weight_cache + sequence_memory + workspace
    training_headroom = max(BYTES_PER_GIB, training_modeled_peak // 4)

    inference_workspace = max(BYTES_PER_GIB // 4, args.batch_size * config.hidden_size * context_length * BF16_BYTES)
    inference_resident = bf16_weights + kv_cache
    inference_peak = inference_resident + inference_workspace

    print(f"Model: {args.model_name}")
    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Context length: {context_length:,}")
    print(f"Batch size: {args.batch_size:,}")
    print(f"Parameters: {parameter_count:,} ({parameter_count / 1_000_000:.2f}M)")
    print()
    print("Inference estimate (BF16 weights and KV cache):")
    print(f"  Weights: {gibibytes(bf16_weights):.2f} GiB")
    print(f"  KV cache: {gibibytes(kv_cache):.2f} GiB")
    print(f"  Resident model + cache: {gibibytes(inference_resident):.2f} GiB")
    print(f"  Approximate peak with workspace: {gibibytes(inference_peak):.2f} GiB")
    print()
    print("Training estimate (BF16 autocast, FP32 parameters, FP32 AdamW, activation checkpointing):")
    print(f"  Parameters + gradients + AdamW moments: {gibibytes(training_persistent):.2f} GiB")
    print(f"  BF16 autocast weight cache: {gibibytes(autocast_weight_cache):.2f} GiB")
    print(f"  Checkpointed hidden states: {gibibytes(checkpointed_hidden_states):.2f} GiB")
    print(f"  Logits: {gibibytes(logits):.2f} GiB")
    print(f"  Modeled peak: {gibibytes(training_modeled_peak):.2f} GiB")
    print(f"  Plan for at least: {gibibytes(training_modeled_peak + training_headroom):.2f} GiB")
    print()
    print("These are ballpark figures. They assume compiled FlexAttention, so no dense attention matrix is stored.")
    print("CUDA allocator state, compilation, and kernels can raise the actual peak.")


if __name__ == "__main__":
    main()
