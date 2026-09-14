#!/usr/bin/env python3
"""Single-GPU supervised fine-tuning for packed chat data.

The packager writes each chat chunk as a FlexAttention document plus an
aligned per-token loss mask.  This trainer packs whole chunks into fixed-size
sequences, resets RoPE positions at every chunk, and only optimizes assistant
targets.

Example:
    uv run python training/sft_train.py --config configs/training/sft_example.toml

Run the CPU-only dataset correctness check with:
    uv run python training/sft_train.py --self-test
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
import sys
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DecoderOnlyTransformer, ModelConfig
from training.tracking import create_tracker


REQUIRED_PROTOCOL_PIECES = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|thinking|>",
    "<|end_thinking|>",
)


class ConfigError(ValueError):
    """A TOML experiment configuration is missing or has an invalid value."""


_MISSING = object()


def tokenizer_model_path(path: Path) -> Path:
    return path / "tokenizer.model" if path.is_dir() else path


def dataset_sidecars(data_path: Path) -> tuple[Path, Path, Path]:
    stem = data_path.with_suffix("")
    return (
        stem.with_name(stem.name + ".document_offsets.bin"),
        stem.with_name(stem.name + ".loss_mask.bin"),
        stem.with_suffix(".json"),
    )


def acquire_output_lock(output_dir: Path) -> Any:
    """Prevent two SFT jobs from updating the same rolling checkpoint."""
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
class SFTTrainConfig:
    data_path: Path
    tokenizer_path: Path
    num_workers: int
    init_checkpoint: Path
    output_dir: Path
    context_length: int | None
    epochs: int
    max_optimizer_steps: int | None
    batch_size: int
    gradient_accumulation_steps: int
    seed: int
    activation_checkpointing: bool
    compile_model: bool
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    optimizer_name: str
    optimizer_betas: tuple[float, float]
    optimizer_eps: float
    scheduler_name: str
    warmup_fraction: float
    gradient_clip_norm: float
    save_every_supervised_tokens: int
    resume: Path | None
    log_every_supervised_tokens: int
    wandb_mode: str
    wandb_project: str | None
    wandb_entity: str | None
    wandb_run_name: str | None


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
    if not isinstance(value, expected_types) or (
        isinstance(value, bool) and any(item in (int, float) for item in expected_types)
    ):
        expected_names = ", ".join(item.__name__ for item in expected_types)
        raise ConfigError(f"[{table_name}].{name} must be {expected_names}")
    return value


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def assert_no_unknown_keys(config_data: dict[str, Any], tables: dict[str, dict[str, Any]]) -> None:
    if config_data:
        raise ConfigError("unknown top-level table(s): " + ", ".join(sorted(config_data)))
    for table_name, table in tables.items():
        if table:
            raise ConfigError(f"unknown key(s) in [{table_name}]: " + ", ".join(sorted(table)))


def validate_config(config: SFTTrainConfig) -> SFTTrainConfig:
    for name in (
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "optimizer_eps",
        "gradient_clip_norm",
        "save_every_supervised_tokens",
        "log_every_supervised_tokens",
    ):
        if getattr(config, name) <= 0:
            raise ConfigError(f"{name.replace('_', ' ')} must be positive")
    if config.max_optimizer_steps is not None and config.max_optimizer_steps <= 0:
        raise ConfigError("max optimizer steps must be positive when set")
    if config.num_workers < 0 or config.min_learning_rate < 0:
        raise ConfigError("number of workers and minimum learning rate cannot be negative")
    if config.context_length is not None and config.context_length <= 0:
        raise ConfigError("model.context_length must be positive")
    if config.min_learning_rate > config.learning_rate:
        raise ConfigError("minimum learning rate cannot exceed learning rate")
    if config.optimizer_name != "adamw":
        raise ConfigError("only optimizer.name = 'adamw' is currently supported")
    if config.scheduler_name != "cosine":
        raise ConfigError("only scheduler.name = 'cosine' is currently supported")
    if not 0.0 <= config.warmup_fraction < 1.0:
        raise ConfigError("scheduler.warmup_fraction must be in [0, 1)")
    if any(beta < 0 or beta >= 1 for beta in config.optimizer_betas):
        raise ConfigError("optimizer.betas values must be in [0, 1)")
    if config.wandb_mode not in ("disabled", "online", "offline"):
        raise ConfigError("logging.wandb_mode must be disabled, online, or offline")
    if config.wandb_mode != "disabled" and not config.wandb_project:
        raise ConfigError("logging.wandb_project is required unless W&B is disabled")
    return config


def load_config(config_path: Path) -> SFTTrainConfig:
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

    config = SFTTrainConfig(
        data_path=resolve_config_path(config_path, config_value(data, "data", "path", str)),
        tokenizer_path=resolve_config_path(config_path, config_value(data, "data", "tokenizer", str)),
        num_workers=config_value(data, "data", "num_workers", int, 2),
        init_checkpoint=resolve_config_path(
            config_path, config_value(checkpointing, "checkpointing", "init_checkpoint", str)
        ),
        output_dir=resolve_config_path(config_path, config_value(checkpointing, "checkpointing", "output_dir", str)),
        context_length=config_value(model, "model", "context_length", int, None),
        epochs=config_value(training, "training", "epochs", int, 1),
        max_optimizer_steps=config_value(training, "training", "max_optimizer_steps", int, None),
        batch_size=config_value(training, "training", "batch_size", int, 1),
        gradient_accumulation_steps=config_value(training, "training", "gradient_accumulation_steps", int, 1),
        seed=config_value(training, "training", "seed", int, 1337),
        activation_checkpointing=config_value(training, "training", "activation_checkpointing", bool, True),
        compile_model=config_value(training, "training", "compile_model", bool, True),
        learning_rate=float(config_value(optimizer, "optimizer", "learning_rate", (int, float), 2e-5)),
        min_learning_rate=float(config_value(scheduler, "scheduler", "min_learning_rate", (int, float), 0.0)),
        weight_decay=float(config_value(optimizer, "optimizer", "weight_decay", (int, float), 0.0)),
        optimizer_name=config_value(optimizer, "optimizer", "name", str, "adamw"),
        optimizer_betas=(float(betas[0]), float(betas[1])),
        optimizer_eps=float(config_value(optimizer, "optimizer", "eps", (int, float), 1e-8)),
        scheduler_name=config_value(scheduler, "scheduler", "name", str, "cosine"),
        warmup_fraction=float(config_value(scheduler, "scheduler", "warmup_fraction", (int, float), 0.03)),
        gradient_clip_norm=float(config_value(training, "training", "gradient_clip_norm", (int, float), 1.0)),
        save_every_supervised_tokens=config_value(
            checkpointing, "checkpointing", "save_every_supervised_tokens", int, 1_000_000
        ),
        resume=(
            resolve_config_path(config_path, config_value(checkpointing, "checkpointing", "resume", str))
            if "resume" in checkpointing
            else None
        ),
        log_every_supervised_tokens=config_value(
            logging, "logging", "log_every_supervised_tokens", int, 100_000
        ),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="TOML SFT experiment configuration.")
    parser.add_argument("--resume", type=Path, help="Override checkpointing.resume for this invocation.")
    parser.add_argument("--output-dir", type=Path, help="Override checkpointing.output_dir for this invocation.")
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "online", "offline"), help="Override logging.wandb_mode."
    )
    parser.add_argument("--self-test", action="store_true", help="Run the CPU-only packed-data correctness test.")
    args = parser.parse_args()
    if args.self_test:
        if any((args.config, args.resume, args.output_dir, args.wandb_mode)):
            parser.error("--self-test cannot be combined with training options")
        return args
    if args.config is None:
        parser.error("--config is required unless --self-test is used")
    try:
        config = load_config(args.config.resolve())
        if args.resume is not None:
            config.resume = args.resume.resolve()
        if args.output_dir is not None:
            config.output_dir = args.output_dir.resolve()
        if args.wandb_mode is not None:
            config.wandb_mode = args.wandb_mode
        args.config = validate_config(config)
        return args
    except ConfigError as error:
        parser.error(str(error))


def integer_dtype(name: object) -> np.dtype[Any]:
    try:
        dtype = np.dtype(name)
    except TypeError as error:
        raise ValueError(f"Dataset token_dtype is invalid: {name!r}") from error
    if dtype.kind not in "iu" or dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError(f"Dataset token_dtype must be an integer dtype; got {dtype}")
    return dtype


class PackedSFTDataset(Dataset[dict[str, Tensor]]):
    """Greedily pack complete SFT chunks into fixed-length model sequences.

    The source token stream remains memory-mapped.  A dataset item contains at
    most ``context_length + 1`` raw tokens; the trainer performs the ordinary
    one-token shift, yielding a fixed ``context_length`` input sequence.
    """

    def __init__(self, data_path: Path, context_length: int, bos_id: int) -> None:
        self.data_path = data_path
        self.context_length = context_length
        self.bos_id = bos_id
        offsets_path, loss_mask_path, metadata_path = dataset_sidecars(data_path)
        if not offsets_path.is_file() or not loss_mask_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("SFT dataset sidecars are missing. Run scripts/package_sft_jsonl_dataset.py first.")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("token_count", "document_count", "token_dtype", "loss_mask_dtype", "bos_id"):
            if key not in self.metadata:
                raise ValueError(f"SFT metadata is missing {key!r}")
        if int(self.metadata["bos_id"]) != bos_id:
            raise ValueError("Dataset BOS ID does not match the tokenizer.")
        self.token_count = int(self.metadata["token_count"])
        self.document_count = int(self.metadata["document_count"])
        self.token_dtype = integer_dtype(self.metadata["token_dtype"])
        if self.token_count < 2 or self.document_count <= 0:
            raise ValueError("SFT dataset must contain at least one non-empty document.")
        if np.dtype(self.metadata["loss_mask_dtype"]) != np.dtype("u1"):
            raise ValueError("SFT loss_mask_dtype must be unsigned byte ('|u1').")
        self._tokens = np.memmap(data_path, mode="r", dtype=self.token_dtype)
        self._offsets = np.memmap(offsets_path, mode="r", dtype="<u8")
        self._loss_mask = np.memmap(loss_mask_path, mode="r", dtype="u1")
        if self._tokens.size != self.token_count or self._loss_mask.size != self.token_count:
            raise ValueError("SFT token or loss-mask file size disagrees with metadata.")
        if self._offsets.size != self.document_count + 1:
            raise ValueError("SFT document offsets disagree with metadata.")
        offsets = np.asarray(self._offsets, dtype=np.int64)
        if offsets[0] != 0 or offsets[-1] != self.token_count or np.any(np.diff(offsets) <= 0):
            raise ValueError("SFT document offsets must be strictly increasing and end at token_count.")
        if np.any(np.asarray(self._loss_mask) > 1):
            raise ValueError("SFT loss mask must contain only 0 and 1.")
        document_lengths = np.diff(offsets)
        if int(document_lengths.max()) - 1 > context_length:
            raise ValueError(
                "A packed SFT chunk does not fit the trainer context after shifting. "
                "Repackage with a smaller --context-length or increase model.context_length."
            )

        # A group of N complete chunks needs N_total_tokens - 1 model inputs:
        # the final EOS is the last target, not an input.  This retains complete
        # chat chunks and their full prompt context instead of cutting windows at
        # arbitrary stream positions.
        groups: list[tuple[int, int]] = []
        first_document = 0
        packed_tokens = 0
        for document_index, length in enumerate(document_lengths):
            length_int = int(length)
            if packed_tokens and packed_tokens + length_int - 1 > context_length:
                groups.append((first_document, document_index))
                first_document = document_index
                packed_tokens = 0
            packed_tokens += length_int
        groups.append((first_document, self.document_count))

        self.groups: list[tuple[int, int]] = []
        self.total_target_tokens = 0
        self.total_supervised_tokens = 0
        for first, last in groups:
            start, stop = int(offsets[first]), int(offsets[last])
            target_mask = np.asarray(self._loss_mask[start + 1 : stop], dtype=np.bool_)
            target_document_ids = np.searchsorted(offsets, np.arange(start + 1, stop), side="right") - 1
            input_document_ids = np.searchsorted(offsets, np.arange(start, stop - 1), side="right") - 1
            target_mask &= target_document_ids == input_document_ids
            supervised = int(target_mask.sum())
            if supervised == 0:
                continue
            self.groups.append((first, last))
            self.total_target_tokens += stop - start - 1
            self.total_supervised_tokens += supervised
        if not self.groups:
            raise ValueError("SFT dataset has no supervised assistant target tokens after boundary masking.")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        first_document, last_document = self.groups[index]
        start = int(self._offsets[first_document])
        stop = int(self._offsets[last_document])
        real_length = stop - start
        full_length = self.context_length + 1
        token_ids = np.full(full_length, self.bos_id, dtype=np.int64)
        token_ids[:real_length] = np.asarray(self._tokens[start:stop], dtype=np.int64)
        loss_mask = np.zeros(full_length, dtype=np.bool_)
        loss_mask[:real_length] = np.asarray(self._loss_mask[start:stop], dtype=np.bool_)
        token_mask = np.zeros(full_length, dtype=np.bool_)
        token_mask[:real_length] = True

        positions = np.arange(start, stop, dtype=np.int64)
        document_ids = np.searchsorted(self._offsets, positions, side="right") - 1
        document_starts = np.asarray(self._offsets[document_ids], dtype=np.int64)
        position_ids = np.zeros(full_length, dtype=np.int64)
        position_ids[:real_length] = positions - document_starts
        padded_document_ids = np.arange(self.document_count, self.document_count + full_length, dtype=np.int64)
        padded_document_ids[:real_length] = document_ids
        return {
            "token_ids": torch.from_numpy(token_ids),
            "loss_mask": torch.from_numpy(loss_mask),
            "token_mask": torch.from_numpy(token_mask),
            "document_ids": torch.from_numpy(padded_document_ids),
            "position_ids": torch.from_numpy(position_ids),
        }


def make_document_block_mask(document_ids: Tensor) -> BlockMask:
    """Allow only causal attention within one packed SFT chunk."""
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


def learning_rate_at(optimizer_step: int, total_updates: int, config: SFTTrainConfig) -> float:
    warmup_updates = int(math.ceil(total_updates * config.warmup_fraction))
    if warmup_updates and optimizer_step <= warmup_updates:
        return config.learning_rate * optimizer_step / warmup_updates
    progress = min(1.0, (optimizer_step - warmup_updates) / max(1, total_updates - warmup_updates))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


def create_optimizer(model: DecoderOnlyTransformer, config: SFTTrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.weight_decay,
        fused=True,
    )


def checkpoint_model_config(checkpoint: dict[str, Any]) -> ModelConfig:
    values = checkpoint.get("model_config")
    if not isinstance(values, dict):
        raise ValueError("Checkpoint is missing its saved model_config.")
    try:
        return ModelConfig(**values)
    except TypeError as error:
        raise ValueError("Checkpoint model_config does not match this model implementation.") from error


def load_initial_checkpoint(path: Path) -> tuple[DecoderOnlyTransformer, ModelConfig, Path | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Initial pretrained checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Initial checkpoint does not contain model weights.")
    model_config = checkpoint_model_config(checkpoint)
    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(checkpoint["model"])
    train_config = checkpoint.get("train_config")
    tokenizer_path = train_config.get("tokenizer_path") if isinstance(train_config, dict) else None
    return model, model_config, Path(tokenizer_path).resolve() if isinstance(tokenizer_path, str) else None


def validate_pretraining_tokenizer(
    configured_model_path: Path, pretraining_tokenizer_path: Path | None
) -> None:
    """Require the tokenizer recorded in the pretrained checkpoint when it remains available."""
    if pretraining_tokenizer_path is None:
        print("Warning: pretrained checkpoint has no recorded tokenizer path; cannot byte-verify it.", file=sys.stderr)
        return
    expected_model_path = tokenizer_model_path(pretraining_tokenizer_path)
    if not expected_model_path.is_file():
        print(
            f"Warning: pretrained checkpoint tokenizer is unavailable at {expected_model_path}; "
            "cannot byte-verify it.",
            file=sys.stderr,
        )
        return
    if expected_model_path.read_bytes() != configured_model_path.read_bytes():
        raise ValueError(
            "Configured tokenizer does not byte-match the tokenizer recorded in the pretrained checkpoint."
        )


@dataclass
class ResumeState:
    supervised_tokens: int
    total_tokens: int
    optimizer_steps: int
    epoch: int
    batches_completed_in_epoch: int


def save_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    config: SFTTrainConfig,
    model_config: ModelConfig,
    state: ResumeState,
    initial_checkpoint: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sft_train_config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "model_config": model_config.to_dict(),
        "initial_checkpoint": str(initial_checkpoint),
        "supervised_tokens": state.supervised_tokens,
        "total_tokens": state.total_tokens,
        "optimizer_steps": state.optimizer_steps,
        "epoch": state.epoch,
        "batches_completed_in_epoch": state.batches_completed_in_epoch,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    print(f"Saved SFT checkpoint at {state.supervised_tokens:,} supervised tokens: {path}")


def load_sft_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
) -> ResumeState:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "sft_train_config" not in checkpoint:
        raise ValueError("--resume must point to a checkpoint produced by training/sft_train.py.")
    if checkpoint_model_config(checkpoint) != model_config:
        raise ValueError("SFT resume checkpoint model_config does not match the initial checkpoint.")
    required = ("model", "optimizer", "supervised_tokens", "total_tokens", "optimizer_steps", "epoch", "batches_completed_in_epoch")
    if any(key not in checkpoint for key in required):
        raise ValueError("SFT resume checkpoint is incomplete.")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    return ResumeState(
        supervised_tokens=int(checkpoint["supervised_tokens"]),
        total_tokens=int(checkpoint["total_tokens"]),
        optimizer_steps=int(checkpoint["optimizer_steps"]),
        epoch=int(checkpoint["epoch"]),
        batches_completed_in_epoch=int(checkpoint["batches_completed_in_epoch"]),
    )


def batch_target_mask(batch: dict[str, Tensor]) -> Tensor:
    """Shift the packager mask with labels and remove cross-document targets."""
    return (
        batch["loss_mask"][:, 1:]
        & batch["token_mask"][:, 1:]
        & (batch["document_ids"][:, :-1] == batch["document_ids"][:, 1:])
    )


def validate_dataset_special_tokens(metadata: dict[str, Any], tokenizer: spm.SentencePieceProcessor) -> None:
    """Reject packed IDs made with a tokenizer that differs at protocol tokens."""
    if int(metadata.get("eos_id", -1)) != tokenizer.eos_id():
        raise ValueError("Dataset EOS ID does not match the tokenizer.")
    recorded = metadata.get("special_token_ids")
    if recorded is None:
        # The initial packager format only recorded BOS/EOS.  It is still safe
        # for ordinary chats, but new artifacts record the full protocol below.
        print("Warning: SFT metadata lacks special_token_ids; only BOS/EOS were verified.", file=sys.stderr)
        return
    if not isinstance(recorded, dict):
        raise ValueError("Dataset special_token_ids must be an object.")
    for piece in REQUIRED_PROTOCOL_PIECES:
        expected_id = tokenizer.piece_to_id(piece)
        if tokenizer.id_to_piece(expected_id) != piece:
            raise ValueError(f"Tokenizer is missing required protocol piece {piece!r}.")
        if recorded.get(piece) != expected_id:
            raise ValueError(f"Dataset protocol ID for {piece!r} does not match the tokenizer.")


def train(config: SFTTrainConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SFT training requires one CUDA GPU.")
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
    model, model_config, pretraining_tokenizer_path = load_initial_checkpoint(config.init_checkpoint.resolve())
    validate_pretraining_tokenizer(model_path, pretraining_tokenizer_path)
    if model_config.vocab_size != tokenizer.vocab_size():
        raise ValueError("Pretrained checkpoint vocabulary size does not match the tokenizer.")
    context_length = config.context_length or model_config.max_context_length
    if context_length > model_config.max_context_length:
        raise ValueError(
            f"Requested context length {context_length} exceeds checkpoint limit {model_config.max_context_length}."
        )
    dataset = PackedSFTDataset(config.data_path.resolve(), context_length, tokenizer.bos_id())
    if int(dataset.metadata.get("tokenizer_vocab_size", -1)) != tokenizer.vocab_size():
        raise ValueError("SFT dataset vocabulary size does not match the tokenizer.")
    validate_dataset_special_tokens(dataset.metadata, tokenizer)
    if int(dataset.metadata.get("context_length", context_length)) > model_config.max_context_length:
        raise ValueError("SFT dataset was chunked beyond the pretrained model context limit.")

    microbatches_per_epoch = math.ceil(len(dataset) / config.batch_size)
    updates_per_epoch = math.ceil(microbatches_per_epoch / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.epochs
    planned_updates = min(total_updates, config.max_optimizer_steps) if config.max_optimizer_steps else total_updates
    print(
        f"SFT dataset: {len(dataset):,} packed sequences, {dataset.total_supervised_tokens:,} supervised targets, "
        f"{dataset.total_target_tokens:,} total targets per epoch."
    )
    print(f"Schedule: {planned_updates:,} optimizer updates across up to {config.epochs} epoch(s).")

    model = model.to(device)
    model.set_gradient_checkpointing(config.activation_checkpointing)
    parameter_count = model.parameter_count()
    optimizer = create_optimizer(model, config)
    state = ResumeState(0, 0, 0, 0, 0)
    if config.resume:
        state = load_sft_checkpoint(config.resume.resolve(), model, optimizer, model_config)
        print(f"Resumed from {config.resume} at {state.supervised_tokens:,} supervised tokens.")
    if state.epoch > config.epochs or (state.epoch == config.epochs and state.batches_completed_in_epoch):
        raise ValueError("Resume checkpoint is beyond the configured number of epochs.")
    training_model = torch.compile(model) if config.compile_model else model
    torch.cuda.reset_peak_memory_stats(device)

    tracking_config = asdict(config)
    for field in ("data_path", "tokenizer_path", "init_checkpoint", "output_dir", "resume", "wandb_project", "wandb_entity", "wandb_run_name"):
        tracking_config.pop(field)
    tracker = create_tracker(
        mode=config.wandb_mode,
        project=config.wandb_project,
        entity=config.wandb_entity,
        run_name=config.wandb_run_name,
        output_dir=config.output_dir.resolve(),
        stage="sft",
        run_config={
            "stage": "sft",
            "train": tracking_config,
            "model": model_config.to_dict(),
            "dataset": {
                "document_count": dataset.document_count,
                "packed_sequence_count": len(dataset),
                "token_count": dataset.token_count,
                "supervised_tokens_per_epoch": dataset.total_supervised_tokens,
                "total_targets_per_epoch": dataset.total_target_tokens,
                "tokenizer_vocab_size": tokenizer.vocab_size(),
            },
            "parameter_count": parameter_count,
            "total_optimizer_updates": planned_updates,
        },
    )

    checkpoint_path = config.output_dir / "checkpoint.pt"
    next_log_tokens = state.supervised_tokens + config.log_every_supervised_tokens
    next_save_tokens = state.supervised_tokens + config.save_every_supervised_tokens
    loss_total = 0.0
    metric_supervised_tokens = 0
    metric_total_targets = 0
    started_at = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(state.epoch, config.epochs):
            generator = torch.Generator().manual_seed(config.seed + epoch)
            loader_kwargs: dict[str, Any] = {
                "batch_size": config.batch_size,
                "shuffle": True,
                "generator": generator,
                "num_workers": config.num_workers,
                "pin_memory": True,
                "persistent_workers": config.num_workers > 0,
            }
            if config.num_workers > 0:
                loader_kwargs["prefetch_factor"] = 2
            data_loader = DataLoader(dataset, **loader_kwargs)
            data_iterator = iter(data_loader)
            batches_to_skip = state.batches_completed_in_epoch if epoch == state.epoch else 0
            for _ in range(batches_to_skip):
                next(data_iterator)
            batches_completed = batches_to_skip
            while True:
                accumulation: list[dict[str, Tensor]] = []
                for _ in range(config.gradient_accumulation_steps):
                    try:
                        accumulation.append(next(data_iterator))
                        batches_completed += 1
                    except StopIteration:
                        break
                if not accumulation:
                    break
                supervised_in_update = sum(int(batch_target_mask(batch).sum().item()) for batch in accumulation)
                if supervised_in_update == 0:
                    # Dataset construction currently filters these, but retaining this
                    # guard makes a malformed future data reader safe to use.
                    print("Warning: skipped an accumulation update with zero supervised targets.", file=sys.stderr)
                    continue
                total_targets_in_update = sum(int(batch["token_mask"][:, 1:].sum().item()) for batch in accumulation)
                for batch in accumulation:
                    token_ids = batch["token_ids"].to(device, non_blocking=True)
                    input_ids = token_ids[:, :-1]
                    labels = token_ids[:, 1:]
                    loss_mask = batch_target_mask(batch).to(device, non_blocking=True)
                    position_ids = batch["position_ids"][:, :-1].to(device, non_blocking=True)
                    document_ids = batch["document_ids"][:, :-1].to(device, non_blocking=True)
                    block_mask = make_document_block_mask(document_ids)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = training_model(input_ids, position_ids, block_mask)
                        token_losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
                        loss_sum = (token_losses * loss_mask.flatten().to(token_losses.dtype)).sum()
                        loss = loss_sum / supervised_in_update
                    loss.backward()
                    loss_total += float(loss_sum.detach().item())

                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm).item())
                state.optimizer_steps += 1
                learning_rate = learning_rate_at(state.optimizer_steps, planned_updates, config)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state.supervised_tokens += supervised_in_update
                state.total_tokens += total_targets_in_update
                metric_supervised_tokens += supervised_in_update
                metric_total_targets += total_targets_in_update
                state.epoch = epoch
                state.batches_completed_in_epoch = batches_completed

                if state.optimizer_steps == 1:
                    gib = 1024**3
                    print(
                        "GPU memory after first optimizer step: "
                        f"allocated={torch.cuda.memory_allocated(device) / gib:.2f} GiB "
                        f"reserved={torch.cuda.memory_reserved(device) / gib:.2f} GiB "
                        f"peak_allocated={torch.cuda.max_memory_allocated(device) / gib:.2f} GiB "
                        f"peak_reserved={torch.cuda.max_memory_reserved(device) / gib:.2f} GiB"
                    )
                if state.supervised_tokens >= next_log_tokens:
                    elapsed = time.monotonic() - started_at
                    mean_loss = loss_total / metric_supervised_tokens
                    throughput = metric_supervised_tokens / elapsed
                    fraction = metric_supervised_tokens / max(1, metric_total_targets)
                    print(
                        f"supervised_tokens={state.supervised_tokens:,} total_tokens={state.total_tokens:,} "
                        f"step={state.optimizer_steps:,} loss={mean_loss:.4f} "
                        f"ppl={math.exp(min(mean_loss, 20.0)):.2f} lr={learning_rate:.3e} "
                        f"assistant_fraction={fraction:.2%} supervised_tok/s={throughput:,.0f}"
                    )
                    gib = 1024**3
                    tracker.log(
                        {
                            "train/tokens": state.supervised_tokens,
                            "train/supervised_tokens": state.supervised_tokens,
                            "train/total_tokens": state.total_tokens,
                            "train/optimizer_step": state.optimizer_steps,
                            "train/loss": mean_loss,
                            "train/perplexity": math.exp(min(mean_loss, 20.0)),
                            "train/learning_rate": learning_rate,
                            "train/supervised_tokens_per_second": throughput,
                            "train/assistant_token_fraction": fraction,
                            "train/gradient_norm": gradient_norm,
                            "system/gpu_allocated_gib": torch.cuda.memory_allocated(device) / gib,
                            "system/gpu_reserved_gib": torch.cuda.memory_reserved(device) / gib,
                            "system/gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
                            "system/gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
                        }
                    )
                    loss_total = 0.0
                    metric_supervised_tokens = 0
                    metric_total_targets = 0
                    started_at = time.monotonic()
                    next_log_tokens = state.supervised_tokens + config.log_every_supervised_tokens
                if state.supervised_tokens >= next_save_tokens:
                    save_checkpoint(checkpoint_path, model, optimizer, config, model_config, state, config.init_checkpoint)
                    next_save_tokens = state.supervised_tokens + config.save_every_supervised_tokens
                if state.optimizer_steps >= planned_updates:
                    break
            shutdown_workers = getattr(data_iterator, "_shutdown_workers", None)
            if shutdown_workers is not None:
                shutdown_workers()
            state.epoch = epoch + 1
            state.batches_completed_in_epoch = 0
            if state.optimizer_steps >= planned_updates:
                break
    except KeyboardInterrupt:
        print("Interrupted; saving the rolling SFT checkpoint.")
        raise
    finally:
        try:
            tracker.finish()
        except Exception as error:
            print(f"W&B finalization failed: {error}", file=sys.stderr)
        try:
            save_checkpoint(checkpoint_path, model, optimizer, config, model_config, state, config.init_checkpoint)
        finally:
            output_lock.close()


def run_correctness_test() -> None:
    """Exercise shifted masks, EOS supervision, packed boundaries, and RoPE resets."""
    with tempfile.TemporaryDirectory(prefix="itsyllm-sft-test-") as temporary_directory:
        data_path = Path(temporary_directory) / "sft.bin"
        offsets_path, mask_path, metadata_path = dataset_sidecars(data_path)
        # Two independently packed conversations.  Each has context-only BOS and
        # user marker followed by a supervised assistant token and terminal EOS.
        tokens = np.asarray([1, 10, 11, 2, 1, 20, 21, 2], dtype="<i2")
        mask = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype="u1")
        offsets = np.asarray([0, 4, 8], dtype="<u8")
        tokens.tofile(data_path)
        mask.tofile(mask_path)
        offsets.tofile(offsets_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "token_count": 8,
                    "document_count": 2,
                    "token_dtype": "<i2",
                    "loss_mask_dtype": "|u1",
                    "bos_id": 1,
                    "tokenizer_vocab_size": 32,
                    "context_length": 8,
                }
            ),
            encoding="utf-8",
        )
        dataset = PackedSFTDataset(data_path, context_length=8, bos_id=1)
        assert len(dataset) == 1
        item = dataset[0]
        shifted_mask = batch_target_mask({key: value.unsqueeze(0) for key, value in item.items()})[0]
        assert item["position_ids"][:8].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
        assert item["document_ids"][:8].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        # Targets 11/EOS in conversation one and 21/EOS in conversation two.
        assert shifted_mask[:7].tolist() == [False, True, True, False, False, True, True]
        assert not bool(shifted_mask[3])  # The next conversation's BOS is never predicted across a boundary.
        assert bool(shifted_mask[2]) and bool(shifted_mask[6])  # Both assistant-terminating EOS tokens train.
        assert int(shifted_mask.sum()) == 4
    print("SFT packed-data correctness test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_correctness_test()
    else:
        train(args.config)


if __name__ == "__main__":
    main()
