"""Named model sizes for the decoder-only transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_context_length: int
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-6
    repeat_blocks: bool = False

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


MODEL_PRESETS = {
    "smoke": {
        "hidden_size": 128,
        "num_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 352,
        "max_context_length": 128,
    },
    "extra-small": {
        "hidden_size": 768,
        "num_layers": 16,
        "num_attention_heads": 12,
        "num_key_value_heads": 1,
        "intermediate_size": 2048,
        "max_context_length": 8192,
    },
    "extra-small-gqa": {
        "hidden_size": 768,
        "num_layers": 16,
        "num_attention_heads": 12,
        "num_key_value_heads": 3,
        "intermediate_size": 2048,
        "max_context_length": 8192,
    },
}

MODEL_CONFIG_ALIASES = {
    "mqa": "extra-small",
    "gqa": "extra-small-gqa",
}


def model_config_from_name(name: str, vocab_size: int) -> ModelConfig:
    name = MODEL_CONFIG_ALIASES.get(name, name)
    try:
        values = MODEL_PRESETS[name]
    except KeyError as error:
        available = ", ".join(MODEL_PRESETS)
        raise ValueError(f"Unknown model config {name!r}. Available: {available}") from error
    return ModelConfig(vocab_size=vocab_size, **values)
