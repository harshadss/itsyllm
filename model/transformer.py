"""A small, readable decoder-only transformer."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        # Educational reference implementation:
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.float()
        # variance = hidden_states.square().mean(dim=-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # return (self.weight * hidden_states).to(input_dtype)
        return F.rms_norm(
            hidden_states,
            (self.weight.numel(),),
            self.weight.to(dtype=hidden_states.dtype),
            self.eps,
        )


def rotate_half(hidden_states: Tensor) -> Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inverse_frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=False)

    def forward(self, query: Tensor, key: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        frequencies = torch.einsum("bl,d->bld", position_ids.float(), self.inverse_frequencies)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cos = angles.cos().unsqueeze(1).to(query.dtype)
        sin = angles.sin().unsqueeze(1).to(query.dtype)
        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)
        return query, key


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        if self.num_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.wq = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, hidden_states: Tensor, position_ids: Tensor, block_mask: BlockMask) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.wq(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.wk(hidden_states).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.wv(hidden_states).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary(query, key, position_ids)
        attention_output = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=self.num_heads != self.num_key_value_heads,
        )
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.wo(attention_output)


class SwiGLUMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, hidden_states: Tensor, position_ids: Tensor, block_mask: BlockMask) -> Tensor:
        hidden_states = hidden_states + self.attention(self.input_norm(hidden_states), position_ids, block_mask)
        return hidden_states + self.mlp(self.post_attention_norm(hidden_states))


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    @staticmethod
    def estimate_parameter_count(config: ModelConfig) -> int:
        attention_parameters = (
            config.hidden_size * config.hidden_size
            + 2 * config.hidden_size * config.num_key_value_heads * config.head_dim
            + config.hidden_size * config.hidden_size
            + 2 * config.head_dim
        )
        mlp_parameters = 3 * config.hidden_size * config.intermediate_size
        layer_norm_parameters = 2 * config.hidden_size
        layer_parameters = attention_parameters + mlp_parameters + layer_norm_parameters
        embedding_parameters = config.vocab_size * config.hidden_size
        return embedding_parameters + config.num_layers * layer_parameters + config.hidden_size

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_ids: Tensor, position_ids: Tensor, block_mask: BlockMask) -> Tensor:
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                layer_forward: Callable[[Tensor], Tensor] = lambda states, layer=layer: layer(states, position_ids, block_mask)
                hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states, position_ids, block_mask)
        return self.lm_head(self.norm(hidden_states))
