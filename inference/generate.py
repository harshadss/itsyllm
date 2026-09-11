#!/usr/bin/env python3
"""Generate a completion from a pretrained checkpoint.

Example:
    uv run python inference/generate.py \
      --checkpoint artifacts/checkpoints/extra_small_gqa_4096_v1/checkpoint.pt \
      --tokenizer artifacts/tokenizers/sangraha_ultrafineweb_l3_en_indic_unigram_v1 \
      --prompt "भारत की राजधानी" --max-new-tokens 128

The checkpoint supplies the model architecture. Prompt context is trimmed from the
left when necessary; the most recent tokens are retained.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any

import sentencepiece as spm
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DecoderOnlyTransformer, ModelConfig


def tokenizer_model_path(path: Path) -> Path:
    return path / "tokenizer.model" if path.is_dir() else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trainer checkpoint.pt file.")
    parser.add_argument("--tokenizer", required=True, type=Path, help="SentencePiece model or tokenizer artifact directory.")
    parser.add_argument("--prompt", required=True, help="Text to complete.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens (default: 128).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; 0 selects greedily (default: 0.8).")
    parser.add_argument("--top-k", type=int, default=128, help="Keep this many most likely tokens; 0 disables it (default: 50).")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling probability; 1 disables it (default: 0.95).")
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after EOS (default: enabled; use --no-early-stopping to ignore EOS).",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed (default: 1337).")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda when available, otherwise cpu).",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    return args


def causal_block_mask(sequence_length: int, device: torch.device) -> BlockMask:
    def causal_mask(batch: Tensor, head: Tensor, query_index: Tensor, key_index: Tensor) -> Tensor:
        del batch, head
        return query_index >= key_index

    return create_block_mask(
        causal_mask,
        B=1,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        BLOCK_SIZE=128,
    )


def trim_context(token_ids: list[int], max_context_length: int, bos_id: int) -> tuple[list[int], bool]:
    if len(token_ids) <= max_context_length:
        return token_ids, False
    # Keep a BOS token at position zero while retaining the newest prompt tokens.
    return [bos_id, *token_ids[-(max_context_length - 1):]], True


def sample_token(logits: Tensor, temperature: float, top_k: int, top_p: float, generator: torch.Generator) -> int:
    if temperature == 0:
        return int(torch.argmax(logits).item())

    scores = logits.float() / temperature
    if 0 < top_k < scores.numel():
        cutoff = torch.topk(scores, top_k).values[-1]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probabilities = torch.softmax(sorted_scores, dim=-1)
        remove_sorted = torch.cumsum(sorted_probabilities, dim=-1) > top_p
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        scores = scores.scatter(0, sorted_indices, sorted_scores.masked_fill(remove_sorted, float("-inf")))
    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())


def load_model(checkpoint_path: Path, tokenizer_vocab_size: int, device: torch.device) -> DecoderOnlyTransformer:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        config = ModelConfig(**checkpoint["model_config"])
        model_state = checkpoint["model"]
    except KeyError as error:
        raise ValueError("Checkpoint does not have the trainer's model_config and model fields.") from error
    if config.vocab_size != tokenizer_vocab_size:
        raise ValueError(f"Checkpoint vocabulary size ({config.vocab_size}) does not match tokenizer ({tokenizer_vocab_size}).")
    model = DecoderOnlyTransformer(config)
    model.load_state_dict(model_state)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    tokenizer_path = tokenizer_model_path(args.tokenizer).resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.bos_id() < 0 or tokenizer.eos_id() < 0:
        raise ValueError("Tokenizer must define BOS and EOS token IDs.")

    model = load_model(args.checkpoint.resolve(), tokenizer.vocab_size(), device)
    token_ids = tokenizer.encode(args.prompt, out_type=int, add_bos=True, add_eos=False)
    token_ids, prompt_trimmed = trim_context(token_ids, model.config.max_context_length, tokenizer.bos_id())
    if prompt_trimmed:
        print(f"Prompt exceeded {model.config.max_context_length:,} tokens; trimmed from the left.", file=sys.stderr)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    generated_ids: list[int] = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast:
        for _ in range(args.max_new_tokens):
            context_ids, context_trimmed = trim_context(token_ids, model.config.max_context_length, tokenizer.bos_id())
            if context_trimmed and not prompt_trimmed:
                print(f"Generation reached {model.config.max_context_length:,} tokens; trimming left context.", file=sys.stderr)
                prompt_trimmed = True
            input_ids = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
            position_ids = torch.arange(input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(0)
            logits = model(input_ids, position_ids, causal_block_mask(input_ids.shape[1], device))
            next_token = sample_token(logits[0, -1], args.temperature, args.top_k, args.top_p, generator)
            token_ids.append(next_token)
            if next_token == tokenizer.eos_id() and args.early_stopping:
                break
            generated_ids.append(next_token)

    completion = tokenizer.decode(generated_ids)
    print("Prompt:")
    print(args.prompt)
    print("\nCompletion:")
    print(completion)
    print(f"\nGenerated tokens: {len(generated_ids)}", file=sys.stderr)


if __name__ == "__main__":
    main()
