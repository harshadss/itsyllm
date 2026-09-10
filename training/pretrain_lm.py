#!/usr/bin/env python3
"""Single-GPU pretraining for the decoder-only transformer.

Example:
    uv run python training/pretrain_lm.py \
      --data artifacts/datasets/example.bin \
      --tokenizer artifacts/tokenizers/example \
      --output-dir artifacts/checkpoints/example \
      --model-config extra-small --max-train-tokens 1000000000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.utils.data import DataLoader, Dataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DecoderOnlyTransformer, model_config_from_name

IGNORE_INDEX = -100


def tokenizer_model_path(path: Path) -> Path:
    return path / "tokenizer.model" if path.is_dir() else path


def dataset_sidecars(data_path: Path) -> tuple[Path, Path]:
    stem = data_path.with_suffix("")
    return stem.with_name(stem.name + ".document_offsets.bin"), stem.with_suffix(".json")


@dataclass
class TrainConfig:
    data_path: Path
    tokenizer_path: Path
    output_dir: Path
    model_config: str
    max_train_tokens: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_tokens: int
    weight_decay: float
    num_workers: int
    seed: int
    log_every_tokens: int
    save_every_tokens: int
    activation_checkpointing: bool
    compile_model: bool
    resume: Path | None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, dest="data_path", help="Packager .bin output.")
    parser.add_argument("--tokenizer", required=True, type=Path, dest="tokenizer_path", help="SentencePiece model or artifact directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for the one rolling checkpoint.")
    parser.add_argument("--model-config", default="extra-small", choices=("smoke", "extra-small", "extra-small-gqa"))
    parser.add_argument("--max-train-tokens", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-tokens", type=int, default=100_000_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-every-tokens", type=int, default=1_000_000)
    parser.add_argument("--save-every-tokens", type=int, default=500_000_000)
    parser.add_argument("--no-activation-checkpointing", action="store_false", dest="activation_checkpointing")
    parser.add_argument("--no-compile", action="store_false", dest="compile_model")
    parser.add_argument("--resume", type=Path, help="Checkpoint file to resume.")
    parser.set_defaults(activation_checkpointing=True, compile_model=True)
    args = parser.parse_args()
    positive_names = (
        "max_train_tokens", "batch_size", "gradient_accumulation_steps", "learning_rate",
        "warmup_tokens", "log_every_tokens", "save_every_tokens",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.min_learning_rate < 0 or args.num_workers < 0:
        parser.error("minimum learning rate and workers cannot be negative")
    if args.min_learning_rate > args.learning_rate:
        parser.error("--min-learning-rate cannot exceed --learning-rate")
    return TrainConfig(**vars(args))


class RandomWindowDataset(Dataset[dict[str, Tensor]]):
    """Random context windows with document IDs and local RoPE positions."""

    def __init__(self, data_path: Path, context_length: int, bos_id: int, seed: int) -> None:
        self.data_path = data_path
        self.context_length = context_length
        self.bos_id = bos_id
        offsets_path, metadata_path = dataset_sidecars(data_path)
        if not offsets_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Dataset sidecars are missing. Run the data packager first.")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata["bos_id"] != bos_id:
            raise ValueError("Dataset BOS ID does not match the tokenizer.")
        self.token_count = int(self.metadata["token_count"])
        if self.token_count <= context_length:
            raise ValueError("Dataset must contain more tokens than the model context length.")
        self.offsets_path = offsets_path
        self.seed = seed
        self._tokens: np.memmap | None = None
        self._offsets: np.memmap | None = None
        self._rng: np.random.Generator | None = None

    def _open(self) -> None:
        if self._tokens is None:
            self._tokens = np.memmap(self.data_path, mode="r", dtype="<i2")
            self._offsets = np.memmap(self.offsets_path, mode="r", dtype="<u8")
            if self._tokens.size != self.token_count:
                raise ValueError("Token file size disagrees with dataset metadata.")
            if self._offsets.size != int(self.metadata["document_count"]) + 1:
                raise ValueError("Document offsets disagree with dataset metadata.")
            worker = get_worker_info()
            worker_id = worker.id if worker else 0
            self._rng = np.random.default_rng(self.seed + worker_id)

    def __len__(self) -> int:
        return 2**31

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        del index
        self._open()
        assert self._tokens is not None and self._offsets is not None and self._rng is not None
        start = int(self._rng.integers(0, self.token_count - self.context_length))
        token_positions = np.arange(start, start + self.context_length, dtype=np.int64)
        document_indices = np.searchsorted(self._offsets, token_positions, side="right") - 1
        document_starts = self._offsets[document_indices].astype(np.int64)
        input_ids = np.asarray(self._tokens[start:start + self.context_length], dtype=np.int64)
        labels = np.asarray(self._tokens[start + 1:start + self.context_length + 1], dtype=np.int64)
        labels[labels == self.bos_id] = IGNORE_INDEX
        return {
            "input_ids": torch.from_numpy(input_ids.copy()),
            "labels": torch.from_numpy(labels.copy()),
            "document_ids": torch.from_numpy(document_indices.astype(np.int64, copy=False)),
            "position_ids": torch.from_numpy(token_positions - document_starts),
        }


def make_document_block_mask(document_ids: Tensor) -> BlockMask:
    """Allow only causal attention within each source document."""
    batch_size, sequence_length = document_ids.shape

    def document_causal_mask(batch: Tensor, head: Tensor, query_index: Tensor, key_index: Tensor) -> Tensor:
        del head
        return (query_index >= key_index) & (document_ids[batch, query_index] == document_ids[batch, key_index])

    return create_block_mask(
        document_causal_mask,
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=document_ids.device,
        BLOCK_SIZE=128,
    )


def learning_rate_at(token_count: int, config: TrainConfig) -> float:
    if token_count < config.warmup_tokens:
        return config.learning_rate * token_count / config.warmup_tokens
    progress = min(1.0, (token_count - config.warmup_tokens) / max(1, config.max_train_tokens - config.warmup_tokens))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


def save_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    model_config: dict[str, int | float],
    trained_tokens: int,
    optimizer_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(train_config).items()},
        "model_config": model_config,
        "trained_tokens": trained_tokens,
        "optimizer_steps": optimizer_steps,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    print(f"Saved checkpoint at {trained_tokens:,} tokens: {path}")


def load_checkpoint(path: Path, model: DecoderOnlyTransformer, optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_data["model"])
    optimizer.load_state_dict(checkpoint_data["optimizer"])
    torch.set_rng_state(checkpoint_data["torch_rng_state"])
    torch.cuda.set_rng_state(checkpoint_data["cuda_rng_state"])
    return int(checkpoint_data["trained_tokens"]), int(checkpoint_data["optimizer_steps"])


def train(config: TrainConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This first trainer requires one CUDA GPU.")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda")

    model_path = tokenizer_model_path(config.tokenizer_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    if tokenizer.bos_id() < 0:
        raise ValueError("Tokenizer must define a BOS ID.")
    model_config = model_config_from_name(config.model_config, tokenizer.vocab_size())
    dataset = RandomWindowDataset(config.data_path.resolve(), model_config.max_context_length, tokenizer.bos_id(), config.seed)
    if dataset.metadata["tokenizer_vocab_size"] != tokenizer.vocab_size():
        raise ValueError("Dataset vocabulary size does not match the tokenizer.")

    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    data_loader = DataLoader(dataset, **loader_kwargs)
    data_iterator = iter(data_loader)

    model = DecoderOnlyTransformer(model_config).to(device)
    model.set_gradient_checkpointing(config.activation_checkpointing)
    parameter_count = model.parameter_count()
    print(f"Model: {config.model_config} with {parameter_count:,} parameters.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=config.weight_decay, fused=True)
    trained_tokens = 0
    optimizer_steps = 0
    if config.resume:
        trained_tokens, optimizer_steps = load_checkpoint(config.resume, model, optimizer)
        print(f"Resumed from {config.resume} at {trained_tokens:,} tokens.")
    training_model = torch.compile(model) if config.compile_model else model

    checkpoint_path = config.output_dir / "checkpoint.pt"
    next_log_tokens = trained_tokens + config.log_every_tokens
    next_save_tokens = trained_tokens + config.save_every_tokens
    loss_total = 0.0
    metric_tokens = 0
    started_at = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    try:
        while trained_tokens < config.max_train_tokens:
            for _ in range(config.gradient_accumulation_steps):
                batch = next(data_iterator)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                position_ids = batch["position_ids"].to(device, non_blocking=True)
                document_ids = batch["document_ids"].to(device, non_blocking=True)
                block_mask = make_document_block_mask(document_ids)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = training_model(input_ids, position_ids, block_mask)
                    loss_sum = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE_INDEX, reduction="sum")
                    valid_tokens = (labels != IGNORE_INDEX).sum()
                    loss = loss_sum / valid_tokens / config.gradient_accumulation_steps
                loss.backward()
                valid_count = int(valid_tokens.item())
                trained_tokens += valid_count
                metric_tokens += valid_count
                loss_total += float(loss_sum.detach().item())
                if trained_tokens >= config.max_train_tokens:
                    break

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            learning_rate = learning_rate_at(trained_tokens, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            if trained_tokens >= next_log_tokens:
                elapsed = time.monotonic() - started_at
                mean_loss = loss_total / metric_tokens
                throughput = metric_tokens / elapsed
                print(
                    f"tokens={trained_tokens:,} step={optimizer_steps:,} loss={mean_loss:.4f} "
                    f"ppl={math.exp(min(mean_loss, 20.0)):.2f} lr={learning_rate:.3e} tok/s={throughput:,.0f}"
                )
                loss_total = 0.0
                metric_tokens = 0
                started_at = time.monotonic()
                next_log_tokens = trained_tokens + config.log_every_tokens
            if trained_tokens >= next_save_tokens:
                save_checkpoint(checkpoint_path, model, optimizer, config, model_config.to_dict(), trained_tokens, optimizer_steps)
                next_save_tokens = trained_tokens + config.save_every_tokens
    except KeyboardInterrupt:
        print("Interrupted; saving the rolling checkpoint.")
        save_checkpoint(checkpoint_path, model, optimizer, config, model_config.to_dict(), trained_tokens, optimizer_steps)
        raise
    save_checkpoint(checkpoint_path, model, optimizer, config, model_config.to_dict(), trained_tokens, optimizer_steps)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
