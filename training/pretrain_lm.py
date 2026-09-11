#!/usr/bin/env python3
"""Single-GPU pretraining for the decoder-only transformer.

Example:
    uv run python training/pretrain_lm.py --config configs/training/example.toml
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, replace
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
from training.tracking import create_tracker

IGNORE_INDEX = -100


def tokenizer_model_path(path: Path) -> Path:
    return path / "tokenizer.model" if path.is_dir() else path


def dataset_sidecars(data_path: Path) -> tuple[Path, Path]:
    stem = data_path.with_suffix("")
    return stem.with_name(stem.name + ".document_offsets.bin"), stem.with_suffix(".json")


def acquire_output_lock(output_dir: Path) -> Any:
    """Prevent two trainers from writing the same rolling checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = (output_dir / ".training.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(
            f"Another trainer already owns {output_dir}. "
            "Use a different --output-dir or stop the existing run first."
        ) from error
    return lock


@dataclass
class TrainConfig:
    data_path: Path
    tokenizer_path: Path
    output_dir: Path
    model_config: str
    context_length: int | None
    repeat_blocks: bool
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
    optimizer_name: str
    optimizer_betas: tuple[float, float]
    optimizer_eps: float
    scheduler_name: str
    wandb_mode: str
    wandb_project: str | None
    wandb_entity: str | None
    wandb_run_name: str | None


class ConfigError(ValueError):
    """A TOML experiment configuration is missing or has an invalid value."""


_MISSING = object()


def config_table(config_data: dict[str, Any], name: str) -> dict[str, Any]:
    value = config_data.pop(name, _MISSING)
    if value is _MISSING:
        raise ConfigError(f"missing [{name}] table")
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def config_value(
    table: dict[str, Any],
    table_name: str,
    name: str,
    expected_type: type[Any] | tuple[type[Any], ...],
    default: Any = _MISSING,
) -> Any:
    value = table.pop(name, default)
    if value is _MISSING:
        raise ConfigError(f"missing [{table_name}].{name}")
    if value is None and default is None:
        return None
    expected_types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
    if not isinstance(value, expected_types) or (isinstance(value, bool) and any(item in (int, float) for item in expected_types)):
        expected_names = ", ".join(item.__name__ for item in expected_types)
        raise ConfigError(f"[{table_name}].{name} must be {expected_names}")
    return value


def assert_no_unknown_keys(config_data: dict[str, Any], tables: dict[str, dict[str, Any]]) -> None:
    unknown_tables = ", ".join(sorted(config_data))
    if unknown_tables:
        raise ConfigError(f"unknown top-level table(s): {unknown_tables}")
    for table_name, table in tables.items():
        if table:
            unknown_keys = ", ".join(sorted(table))
            raise ConfigError(f"unknown key(s) in [{table_name}]: {unknown_keys}")


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def validate_config(config: TrainConfig) -> TrainConfig:
    positive_names = (
        "max_train_tokens", "batch_size", "gradient_accumulation_steps", "learning_rate",
        "warmup_tokens", "log_every_tokens", "save_every_tokens",
    )
    for name in positive_names:
        if getattr(config, name) <= 0:
            raise ConfigError(f"{name.replace('_', ' ')} must be positive")
    if config.min_learning_rate < 0 or config.num_workers < 0:
        raise ConfigError("minimum learning rate and workers cannot be negative")
    if config.context_length is not None and config.context_length <= 0:
        raise ConfigError("context length must be positive")
    if config.min_learning_rate > config.learning_rate:
        raise ConfigError("minimum learning rate cannot exceed learning rate")
    if config.optimizer_name not in ("adamw", "adamw8bit"):
        raise ConfigError("optimizer.name must be 'adamw' or 'adamw8bit'")
    if config.scheduler_name != "cosine":
        raise ConfigError("only scheduler.name = 'cosine' is currently supported")
    if any(beta < 0 or beta >= 1 for beta in config.optimizer_betas):
        raise ConfigError("optimizer.betas values must be in [0, 1)")
    if config.optimizer_eps <= 0:
        raise ConfigError("optimizer.eps must be positive")
    if config.wandb_mode not in ("disabled", "online", "offline"):
        raise ConfigError("logging.wandb_mode must be disabled, online, or offline")
    if config.wandb_mode != "disabled" and not config.wandb_project:
        raise ConfigError("logging.wandb_project is required unless W&B is disabled")
    return config


def load_config(config_path: Path) -> TrainConfig:
    try:
        with config_path.open("rb") as config_file:
            config_data = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise ConfigError(f"configuration file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {config_path}: {error}") from error

    model = config_table(config_data, "model")
    data = config_table(config_data, "data")
    training = config_table(config_data, "training")
    optimizer = config_table(config_data, "optimizer")
    scheduler = config_table(config_data, "scheduler")
    checkpointing = config_table(config_data, "checkpointing")
    logging = config_table(config_data, "logging")

    betas = config_value(optimizer, "optimizer", "betas", list, [0.9, 0.95])
    if len(betas) != 2 or any(not isinstance(beta, (int, float)) or isinstance(beta, bool) for beta in betas):
        raise ConfigError("[optimizer].betas must contain exactly two numbers")
    config = TrainConfig(
        data_path=resolve_config_path(config_path, config_value(data, "data", "path", str)),
        tokenizer_path=resolve_config_path(config_path, config_value(data, "data", "tokenizer", str)),
        output_dir=resolve_config_path(config_path, config_value(checkpointing, "checkpointing", "output_dir", str)),
        model_config=config_value(model, "model", "preset", str, "extra-small"),
        context_length=config_value(model, "model", "context_length", int, None),
        repeat_blocks=config_value(model, "model", "repeat_blocks", bool, False),
        max_train_tokens=config_value(training, "training", "max_train_tokens", int),
        batch_size=config_value(training, "training", "batch_size", int, 1),
        gradient_accumulation_steps=config_value(training, "training", "gradient_accumulation_steps", int, 16),
        learning_rate=float(config_value(optimizer, "optimizer", "learning_rate", (int, float), 3e-4)),
        min_learning_rate=float(config_value(scheduler, "scheduler", "min_learning_rate", (int, float), 3e-5)),
        warmup_tokens=config_value(scheduler, "scheduler", "warmup_tokens", int, 100_000_000),
        weight_decay=float(config_value(optimizer, "optimizer", "weight_decay", (int, float), 0.1)),
        num_workers=config_value(data, "data", "num_workers", int, 2),
        seed=config_value(training, "training", "seed", int, 1337),
        log_every_tokens=config_value(logging, "logging", "log_every_tokens", int, 1_000_000),
        save_every_tokens=config_value(checkpointing, "checkpointing", "save_every_tokens", int, 500_000_000),
        activation_checkpointing=config_value(training, "training", "activation_checkpointing", bool, True),
        compile_model=config_value(training, "training", "compile_model", bool, True),
        resume=(
            resolve_config_path(config_path, config_value(checkpointing, "checkpointing", "resume", str))
            if "resume" in checkpointing
            else None
        ),
        optimizer_name=config_value(optimizer, "optimizer", "name", str, "adamw"),
        optimizer_betas=(float(betas[0]), float(betas[1])),
        optimizer_eps=float(config_value(optimizer, "optimizer", "eps", (int, float), 1e-8)),
        scheduler_name=config_value(scheduler, "scheduler", "name", str, "cosine"),
        wandb_mode=config_value(logging, "logging", "wandb_mode", str, "disabled"),
        wandb_project=config_value(logging, "logging", "wandb_project", str, None),
        wandb_entity=config_value(logging, "logging", "wandb_entity", str, None),
        wandb_run_name=config_value(logging, "logging", "wandb_run_name", str, None),
    )
    assert_no_unknown_keys(
        config_data,
        {
            "model": model,
            "data": data,
            "training": training,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "checkpointing": checkpointing,
            "logging": logging,
        },
    )
    return validate_config(config)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="TOML experiment configuration.")
    parser.add_argument("--resume", type=Path, help="Override checkpointing.resume for this invocation.")
    parser.add_argument("--output-dir", type=Path, help="Override checkpointing.output_dir for this invocation.")
    parser.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), help="Override logging.wandb_mode for this invocation.")
    args = parser.parse_args()
    try:
        config = load_config(args.config.resolve())
        if args.resume is not None:
            config.resume = args.resume.resolve()
        if args.output_dir is not None:
            config.output_dir = args.output_dir.resolve()
        if args.wandb_mode is not None:
            config.wandb_mode = args.wandb_mode
        return validate_config(config)
    except ConfigError as error:
        parser.error(str(error))


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


def create_optimizer(model: DecoderOnlyTransformer, config: TrainConfig) -> torch.optim.Optimizer:
    optimizer_kwargs = {
        "lr": config.learning_rate,
        "betas": config.optimizer_betas,
        "eps": config.optimizer_eps,
        "weight_decay": config.weight_decay,
    }
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), fused=True, **optimizer_kwargs)
    if config.optimizer_name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(model.parameters(), **optimizer_kwargs)
    raise AssertionError(f"unsupported optimizer: {config.optimizer_name}")


def train(config: TrainConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This first trainer requires one CUDA GPU.")
    output_lock = acquire_output_lock(config.output_dir.resolve())
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
    model_config = replace(
        model_config,
        max_context_length=config.context_length or model_config.max_context_length,
        repeat_blocks=config.repeat_blocks,
    )
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
    optimizer = create_optimizer(model, config)
    trained_tokens = 0
    optimizer_steps = 0
    if config.resume:
        trained_tokens, optimizer_steps = load_checkpoint(config.resume, model, optimizer)
        print(f"Resumed from {config.resume} at {trained_tokens:,} tokens.")
    training_model = torch.compile(model) if config.compile_model else model
    torch.cuda.reset_peak_memory_stats(device)

    tracking_train_config = asdict(config)
    # Paths are operational details, not useful experiment metadata, and may be sensitive.
    for field in (
        "data_path",
        "tokenizer_path",
        "output_dir",
        "resume",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
    ):
        tracking_train_config.pop(field)
    tracker = create_tracker(
        mode=config.wandb_mode,
        project=config.wandb_project,
        entity=config.wandb_entity,
        run_name=config.wandb_run_name,
        output_dir=config.output_dir.resolve(),
        stage="pretrain",
        run_config={
            "stage": "pretrain",
            "train": tracking_train_config,
            "model": model_config.to_dict(),
            "dataset": {
                "document_count": int(dataset.metadata["document_count"]),
                "token_count": dataset.token_count,
                "tokenizer_vocab_size": tokenizer.vocab_size(),
            },
            "parameter_count": parameter_count,
        },
    )

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

            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            learning_rate = learning_rate_at(trained_tokens, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if optimizer_steps == 1:
                gib = 1024**3
                print(
                    "GPU memory after first optimizer step: "
                    f"allocated={torch.cuda.memory_allocated(device) / gib:.2f} GiB "
                    f"reserved={torch.cuda.memory_reserved(device) / gib:.2f} GiB "
                    f"peak_allocated={torch.cuda.max_memory_allocated(device) / gib:.2f} GiB "
                    f"peak_reserved={torch.cuda.max_memory_reserved(device) / gib:.2f} GiB"
                )

            if trained_tokens >= next_log_tokens:
                elapsed = time.monotonic() - started_at
                mean_loss = loss_total / metric_tokens
                throughput = metric_tokens / elapsed
                print(
                    f"tokens={trained_tokens:,} step={optimizer_steps:,} loss={mean_loss:.4f} "
                    f"ppl={math.exp(min(mean_loss, 20.0)):.2f} lr={learning_rate:.3e} tok/s={throughput:,.0f}"
                )
                gib = 1024**3
                tracker.log(
                    {
                        "train/tokens": trained_tokens,
                        "train/optimizer_step": optimizer_steps,
                        "train/loss": mean_loss,
                        "train/perplexity": math.exp(min(mean_loss, 20.0)),
                        "train/learning_rate": learning_rate,
                        "train/tokens_per_second": throughput,
                        "train/gradient_norm": gradient_norm,
                        "system/gpu_allocated_gib": torch.cuda.memory_allocated(device) / gib,
                        "system/gpu_reserved_gib": torch.cuda.memory_reserved(device) / gib,
                        "system/gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
                        "system/gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
                    }
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
    finally:
        # Persistent workers otherwise survive until interpreter teardown after an error.
        shutdown_workers = getattr(data_iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
        try:
            tracker.finish()
        except Exception as error:
            print(f"W&B finalization failed: {error}", file=sys.stderr)
    save_checkpoint(checkpoint_path, model, optimizer, config, model_config.to_dict(), trained_tokens, optimizer_steps)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
